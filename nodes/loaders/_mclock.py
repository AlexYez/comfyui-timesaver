"""Формат `.tsmodel` — чтение. Ни ComfyUI, ни торча на уровне модуля.

Пара к `model-converter`: тот запирает модель, эти ноды её открывают. Формат —
обычный safetensors, у которого:

* первые 8 байт (длина заголовка u64) заменены магией ``TSMODEL\\x01``. Любой
  ридер видит «длину» около 1.6e17 и падает на первом же шаге: ``safe_open`` →
  «header too large», torch → ``UnpicklingError``;
* JSON-заголовок на своём месте сжат zlib и добит нулями до прежней длины —
  область та же, а имён, форм и смещений в hex-редакторе не видно;
* **данные тензоров лежат байт-в-байт на своих местах.** Поэтому замок ставится
  на готовый файл любого размера за миллисекунды, а нода отдаёт тензоры тем же
  mmap-путём, что и сам ComfyUI;
* в хвосте короткий трейлер: версия, режим, исходная длина заголовка, исходный
  размер файла и CRC32 заголовка.

⚠️ **Секрета в формате нет, и это намеренно.** Ключа, пароля и криптографии
здесь не будет: замок закрывает файл от «просто взять и загрузить», а не от
человека с исходником ноды. Не называйте это DRM ни в коде, ни в справке.

⚠️ **Здесь только ЧТЕНИЕ.** Запирание и отпирание живут в конвертере: ноде они
не нужны, а модуль, который умеет переписывать файлы моделей на диске, в паке
держать незачем.
"""

from __future__ import annotations

import json
import mmap
import os
import struct
import threading
import warnings

MAGIC_HEAD = b"TSMODEL\x01"
MAGIC_TAIL = b"TSMODELT"
VERSION = 2
EXT = ".tsmodel"

_MODE_RAW, _MODE_ZLIB = 0, 1
_TRAILER_FMT = "<8sIBQQII"      # magic, version, mode, header_len, orig_size, clen, crc

#: Строки dtype из safetensors -> имена torch (как `comfy.utils._TYPES`).
_DTYPES = {
    "F64": "float64", "F32": "float32", "F16": "float16", "BF16": "bfloat16",
    "I64": "int64", "I32": "int32", "I16": "int16", "I8": "int8", "U8": "uint8",
    "BOOL": "bool", "F8_E4M3": "float8_e4m3fn", "F8_E5M2": "float8_e5m2",
    "C64": "complex64", "U64": "uint64", "U32": "uint32", "U16": "uint16",
}

LOG_PREFIX = "[TS ModelLock]"


class LockedModelError(RuntimeError):
    """Файл не `.tsmodel`, повреждён или собран другой версией конвертера."""


class Trailer:
    __slots__ = ("version", "mode", "header_len", "orig_size", "clen", "crc", "trailer_len")

    def __init__(self, version, mode, header_len, orig_size, clen, crc):
        self.version, self.mode = version, mode
        self.header_len, self.orig_size, self.clen, self.crc = header_len, orig_size, clen, crc
        self.trailer_len = 0

    @classmethod
    def read(cls, handle) -> "Trailer":
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        if size < 16:
            raise LockedModelError(f"{LOG_PREFIX} file is too short to be a .tsmodel")
        handle.seek(size - 16)
        length, magic = struct.unpack("<Q8s", handle.read(16))
        if magic != MAGIC_TAIL or length > size or length < 16 + struct.calcsize(_TRAILER_FMT):
            raise LockedModelError(f"{LOG_PREFIX} not a .tsmodel file: no trailer at the end")
        handle.seek(size - length)
        fields = struct.unpack(_TRAILER_FMT, handle.read(struct.calcsize(_TRAILER_FMT)))
        magic_again, version, mode, header_len, orig_size, clen, crc = fields
        if magic_again != MAGIC_TAIL:
            raise LockedModelError(f"{LOG_PREFIX} the .tsmodel trailer is damaged")
        if version != VERSION:
            raise LockedModelError(
                f"{LOG_PREFIX} .tsmodel version {version} is not supported by this node "
                f"(it reads version {VERSION}). Re-lock the model with a matching converter."
            )
        trailer = cls(version, mode, header_len, orig_size, clen, crc)
        trailer.trailer_len = length
        return trailer


def is_locked(path: str) -> bool:
    """Заперт ли файл — по магии в первых восьми байтах."""
    try:
        with open(path, "rb") as handle:
            return handle.read(8) == MAGIC_HEAD
    except OSError:
        return False


def _open_trailer(handle) -> Trailer:
    handle.seek(0)
    if handle.read(8) != MAGIC_HEAD:
        raise LockedModelError(f"{LOG_PREFIX} not a .tsmodel file: no magic at the start")
    return Trailer.read(handle)


def _read_header(handle, trailer: Trailer) -> bytes:
    import zlib

    handle.seek(8)
    packed = handle.read(trailer.header_len)
    try:
        # ⚠️ Битые байты внутри сжатого заголовка дают `zlib.error`, а не наш
        # класс. Без этой обёртки человек получил бы «invalid distance too far
        # back» и ни слова о том, что речь про .tsmodel.
        header = (zlib.decompress(packed[:trailer.clen]) if trailer.mode == _MODE_ZLIB
                  else packed[:trailer.clen])
    except zlib.error as error:
        raise LockedModelError(
            f"{LOG_PREFIX} the .tsmodel header is damaged ({error})") from None
    if (zlib.crc32(header) & 0xFFFFFFFF) != trailer.crc or len(header) != trailer.header_len:
        raise LockedModelError(f"{LOG_PREFIX} the .tsmodel header is damaged (CRC mismatch)")
    return header


class LockedFile:
    """Разобранный заголовок. Данные тензоров с диска здесь не читаются."""

    def __init__(self, path: str):
        self.path = path
        with open(path, "rb") as handle:
            self.trailer = _open_trailer(handle)
            self.header = json.loads(_read_header(handle, self.trailer).decode("utf-8"))
        self.data_base = 8 + self.trailer.header_len
        # Как `safe_open().metadata()`: None, если секции нет.
        self.metadata = self.header.get("__metadata__")

    def keys(self) -> list[str]:
        return [name for name in self.header if name != "__metadata__"]

    def tensors(self) -> list[tuple]:
        """``[(имя, dtype, форма, начало, конец)]`` в порядке смещений в файле."""
        out = []
        for name, info in self.header.items():
            if name == "__metadata__":
                continue
            start, end = info["data_offsets"]
            out.append((name, info["dtype"], info["shape"],
                        self.data_base + start, self.data_base + end))
        out.sort(key=lambda entry: entry[3])
        return out


def _torch_dtype(torch, name: str):
    try:
        return getattr(torch, _DTYPES[name])
    except (KeyError, AttributeError):
        raise LockedModelError(f"{LOG_PREFIX} unsupported dtype {name}") from None


def load_state_dict(path: str):
    """Запасной ридер: stdlib mmap + ``torch.frombuffer``, как делает safetensors.

    Тензоры — вид на отображённый файл, копий нет. Возвращает ``(state_dict,
    metadata)``.
    """
    import torch

    locked = LockedFile(path)
    handle = open(path, "rb")                       # noqa: SIM115 - живёт вместе с mmap
    mapped = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
    view = memoryview(mapped)
    state: dict = {}
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The given buffer is not writable")
        for name, dtype_name, shape, start, end in locked.tensors():
            dtype = _torch_dtype(torch, dtype_name)
            if start == end:
                state[name] = torch.empty(shape, dtype=dtype)
                continue
            state[name] = torch.frombuffer(view[start:end], dtype=dtype).view(shape)

    # ⚠️ Отображение обязано жить, пока живы тензоры — иначе это указатели в
    # никуда. Тот же приём у safetensors: ссылка вешается на хранилище.
    holder = (mapped, view, handle)
    for tensor in state.values():
        try:
            tensor.untyped_storage()._ts_mclock_mmap_refs = holder
        except Exception:                           # noqa: BLE001 - чужой тип хранилища
            pass
    return state, locked.metadata


def load_state_dict_comfy(path: str):
    """Чтение ТЕМ ЖЕ путём, что `comfy.utils.load_safetensors`.

    ``comfy_aimdo.ModelMMAP`` + ``TensorFileSlice`` — прямая подкачка тензоров с
    диска в видеопамять, ленивая загрузка. Работает потому, что тензоры в
    `.tsmodel` лежат на своих исходных смещениях.

    ⚠️ Любое расхождение с будущим ComfyUI уводит на запасной ридер, а не роняет
    граф: грузится медленнее, но грузится.
    """
    try:
        import ctypes

        import comfy.memory_management as memory_management
        import comfy.utils
        import torch

        if not getattr(memory_management, "aimdo_enabled", False) \
                or not hasattr(memory_management, "TensorFileSlice"):
            raise ImportError("aimdo is off")
        import comfy_aimdo.model_mmap

        locked = LockedFile(path)
        file_size = os.path.getsize(path)
        file_lock = threading.Lock()
        model_mmap = comfy_aimdo.model_mmap.ModelMMAP(path)
        file_handle = model_mmap.get_file_handle()
        view = memoryview((ctypes.c_uint8 * file_size).from_address(model_mmap.get()))
        types = getattr(comfy.utils, "_TYPES", None)

        state: dict = {}
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The given buffer is not writable")
            for name, dtype_name, shape, start, end in locked.tensors():
                dtype = (types[dtype_name] if types and dtype_name in types
                         else _torch_dtype(torch, dtype_name))
                if start == end:
                    state[name] = torch.empty(shape, dtype=dtype)
                    continue
                tensor = torch.frombuffer(view[start:end], dtype=dtype).view(shape)
                storage = tensor.untyped_storage()
                storage._comfy_tensor_file_slice = memory_management.TensorFileSlice(
                    file_handle, file_lock, start, end - start)
                storage._comfy_tensor_mmap_refs = (model_mmap, view)
                state[name] = tensor
        # `load_safetensors` отдаёт пустой словарь, а не None — повторяем.
        return state, (locked.metadata if locked.metadata is not None else {})
    except LockedModelError:
        raise
    except Exception as error:                      # noqa: BLE001 - расхождение версий
        warnings.warn(f"{LOG_PREFIX} fast path unavailable "
                      f"({error.__class__.__name__}: {error}), falling back to plain mmap")
        state, metadata = load_state_dict(path)
        try:
            import comfy.utils

            if getattr(comfy.utils, "DISABLE_MMAP", False):
                state = {name: tensor.to(copy=True) for name, tensor in state.items()}
        except Exception:                           # noqa: BLE001 - старый ComfyUI
            pass
        return state, metadata
