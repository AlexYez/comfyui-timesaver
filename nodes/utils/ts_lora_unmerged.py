"""TS LoRA Unmerged — a LoRA as a side branch, ``y = W x + B A x``, never merged.

ComfyUI applies a LoRA by **merging** it: ``W' = W + strength * B A`` is written
into the weight once, and the model then runs as usual. That is free at run
time and exact in float32 — but a diffusion model rarely sits in float32:

* **bf16 weights** keep 8 bits of mantissa. A LoRA update is small next to the
  weight it lands on, so round-to-nearest throws a large share of it away.
  Viggle measured about 70% of their turbo LoRA surviving the merge.
* **int8 weights** are requantized after the merge. ComfyUI does that
  stochastically, which keeps the update on average but adds noise several
  times its size.

A distilled few-step LoRA (Viggle's viggle-turbo for Qwen Image 2.1 is the case
this was written for) is exactly the kind that suffers: every one of its six
steps has to land. Viggle's own comparison against the diffusers reference:
LPIPS 0.093 merged vs 0.052 unmerged on bf16, 0.086 vs 0.038 on int8.

This node keeps the weights untouched and adds ``B (A x)`` to the output of
every targeted linear layer on the fly, in the activation dtype — what
diffusers/PEFT do without ``fuse_lora()``. The price is two thin matmuls per
layer, roughly 10-25% per step, and the LoRA's own tensors in VRAM for the
duration of sampling (they are dropped when the sampler finishes).

⚠️ **Qwen Image 2.1 fuses its MLP.** ``gate_layer`` and ``proj`` are one
``gate_up`` layer in ComfyUI, while LoRAs address the two halves separately.
Which half goes where is taken from ComfyUI's own key map
(``comfy.lora.model_lora_keys_unet``: ``(key, (0, offset, size))``), not
guessed here. The down projection ``out`` is worse: under int8 it runs inside a
fused kernel together with the SwiGLU (``comfy.ops.linear_input_act``), and its
module hooks never fire. Its branch is therefore added to the output of the
whole MLP, computed from the (LoRA'd) ``gate_up`` output — the same number, just
reached from one level up.

⚠️ **The hooks live exactly one forward call.** The diffusion model is shared by
every MODEL output cloned from the same checkpoint. Hooks are attached inside a
``DIFFUSION_MODEL`` wrapper and removed in ``finally``, so a plain branch of the
graph that uses the same model without this node never sees them.
"""

from __future__ import annotations

import json
import logging
import math
import threading

import torch
import torch.nn.functional as F
from comfy_api.v0_0_2 import IO

logger = logging.getLogger("comfyui_timesaver.ts_lora_unmerged")
LOG_PREFIX = "[TS LoRA Unmerged]"

WRAPPER_KEY = "ts_lora_unmerged"

#: Пары (up, down) для форматов, которые встречаются в LoRA для DiT: суффикс
#: матрицы B, суффикс матрицы A. Тот же набор, что понимает ядро.
PAIR_SUFFIXES = (
    (".lora_B.weight", ".lora_A.weight"),                     # diffusers / PEFT
    (".lora_up.weight", ".lora_down.weight"),                 # kohya
    ("_lora.up.weight", "_lora.down.weight"),
    (".lora.up.weight", ".lora.down.weight"),
    (".lora_B.default.weight", ".lora_A.default.weight"),     # PEFT без слияния имён
    (".lora_linear_layer.up.weight", ".lora_linear_layer.down.weight"),
    (".lora_B", ".lora_A"),                                   # mochi
)

#: Компоненты пайплайна, чьи метаданные PEFT относятся к диффузионной модели.
#: `text_encoder.lora_alpha` к ней не относится — и не должен перебить свой.
_MODEL_COMPONENTS = ("", "transformer", "unet")


class _GateUpSlot(threading.local):
    """Выход ``gate_up`` ТЕКУЩЕГО блока — его читает хук всего MLP.

    ⚠️ Один слот, а не словарь по модулям: словарь держал выходы ВСЕХ блоков до
    конца прохода — у Qwen 2.1 на 1024² это 32 x ~0.2 ГБ, около 6 ГБ видеопамяти
    лишь ради того, чтобы прочитать каждый один раз. MLP блока выполняется
    целиком (``gate_up`` → ``out`` → хук MLP) прежде, чем начнётся следующий,
    так что слота на поток хватает. Поток — ради multi-GPU, где клоны модели
    считают параллельно.

    Слот общий для всех экземпляров ноды: при двух LoRA подряд хуки на
    ``gate_up`` срабатывают в порядке установки, последняя запись несёт
    поправки обеих, и хуки MLP обеих LoRA читают её.

    ⚠️⚠️ Тензор не переживает MLP СВОЕГО блока: последний читатель очищает слот
    сразу. ComfyUI 0.38 записывает выделения видеопамяти по блокам (comfy_aimdo
    malloc graph, ``iterate("block")``), а блоки префикса на первом шаге считает
    при ПРИОСТАНОВЛЕННОЙ записи. Слот, державший выход прошлого блока до записи
    следующего, освобождал его внутри паузы — и запись падала с «aimdo memory
    compile error» (жалоба 30.09.2026, 0.37.4 работала). Теперь выделение и
    освобождение всегда в одном и том же состоянии записи. Поэтому же слот пишут
    только тогда, когда у этого MLP есть читатель: без него тензор лежал бы
    зря и дожил бы до следующего блока.
    """

    def __init__(self):
        self.module = None
        self.tensor = None
        self.seen = 0
        # gate_up -> сколько хуков-читателей на его MLP стоит в ЭТОМ потоке.
        self.readers: dict = {}

    def add_readers(self, module, count: int = 1) -> None:
        self.readers[module] = self.readers.get(module, 0) + count

    def remove_readers(self, module, count: int = 1) -> None:
        left = self.readers.get(module, 0) - count
        if left > 0:
            self.readers[module] = left
        else:
            self.readers.pop(module, None)

    def wanted(self, module) -> bool:
        return self.readers.get(module, 0) > 0

    def remember(self, module, tensor) -> None:
        self.module, self.tensor, self.seen = module, tensor, 0

    def recall(self, module):
        return self.tensor if self.module is module else None

    def consumed(self, module) -> None:
        """One reader is done; the last one releases the tensor right here."""
        if self.module is not module:
            return
        self.seen += 1
        if self.seen >= self.readers.get(module, 0):
            self.forget()

    def forget(self) -> None:
        self.module = self.tensor = None
        self.seen = 0


_GATE_UP = _GateUpSlot()


class LoraTarget:
    """One LoRA pair aimed at one linear layer (or at a slice of its output)."""

    __slots__ = ("name", "down", "up", "scale", "offset", "size")

    def __init__(self, name: str, down: torch.Tensor, up: torch.Tensor, scale: float = 1.0,
                 offset: int | None = None, size: int | None = None):
        self.name = name          # путь модуля внутри diffusion_model
        self.down = down          # A: [rank, in], как в файле
        self.up = up              # B: [out, rank], как в файле
        self.scale = scale        # alpha / rank * strength
        self.offset = offset      # None — весь выход; иначе кусок выхода
        self.size = size


# ─────────────────────────────── разбор файла ────────────────────────────────

def _adapter_config(metadata: dict | None) -> dict:
    """PEFT's ``lora_adapter_metadata`` with the component prefix stripped."""
    raw = (metadata or {}).get("lora_adapter_metadata")
    if not raw:
        return {}
    try:
        config = json.loads(raw)
    except (TypeError, ValueError):
        logger.warning("%s lora_adapter_metadata is not valid JSON — ignored", LOG_PREFIX)
        return {}
    if not isinstance(config, dict):
        return {}
    # Ключи вида "transformer.lora_alpha": префикс — имя компонента пайплайна.
    # Берём только диффузионную модель: иначе `text_encoder.lora_alpha` молча
    # перебивал бы её собственный, и LoRA ослабевала бы в разы.
    picked = {}
    for key, value in config.items():
        component, _, field = key.rpartition(".")
        if component in _MODEL_COMPONENTS and field in ("lora_alpha", "alpha_pattern", "use_rslora"):
            picked[field] = value
    return picked


def _pattern_value(pattern, module_name: str, default):
    if isinstance(pattern, dict):
        for suffix, value in pattern.items():
            if module_name == suffix or module_name.endswith("." + suffix):
                return value
    return default


def lora_scale(state_dict: dict, base: str, rank: int, config: dict) -> float:
    """``alpha / rank`` exactly as the file means it.

    An explicit ``<base>.alpha`` tensor wins (kohya files carry one). A PEFT
    file keeps alpha in its metadata instead; without either, the scale is 1 —
    ComfyUI's own convention.
    """
    alpha_key = base + ".alpha"
    if alpha_key in state_dict:
        return float(state_dict[alpha_key].item()) / rank
    if "lora_alpha" in config:
        alpha = float(_pattern_value(config.get("alpha_pattern"), base, config["lora_alpha"]))
        if config.get("use_rslora"):
            return alpha / math.sqrt(rank)
        return alpha / rank
    return 1.0


def split_pairs(state_dict: dict) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], list[str]]:
    """Group the file into ``base -> (down, up)``; return the leftover keys too."""
    pairs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    used: set[str] = set()
    for key in state_dict:
        for up_suffix, down_suffix in PAIR_SUFFIXES:
            if not key.endswith(up_suffix):
                continue
            base = key[: -len(up_suffix)]
            down_key = base + down_suffix
            if down_key in state_dict and base not in pairs:
                pairs[base] = (state_dict[down_key], state_dict[key])
                used.update((key, down_key))
                if base + ".alpha" in state_dict:
                    used.add(base + ".alpha")
            break
    leftover = [key for key in state_dict if key not in used]
    return pairs, leftover


def build_targets(state_dict: dict, metadata: dict | None, key_map: dict, diffusion_model: torch.nn.Module,
                  strength: float) -> tuple[list[LoraTarget], list[str]]:
    """Match every LoRA pair to a linear layer of ``diffusion_model``.

    Args:
        state_dict: the LoRA file.
        metadata: its safetensors metadata (PEFT keeps alpha there).
        key_map: ``comfy.lora.model_lora_keys_unet`` for this model — LoRA
            name to ``"diffusion_model.<path>.weight"`` or to
            ``(that, (0, offset, size))`` when the LoRA addresses a slice of a
            fused layer.
        diffusion_model: the module the paths are relative to.
        strength: multiplier on top of ``alpha / rank``.

    Returns:
        The targets and the names that matched nothing in the model.

    Raises:
        ValueError: when a pair matches a layer but its shapes do not fit it,
            or the file is a DoRA — applied as a plain LoRA it would give a
            wrong picture without any error.
    """
    dora = [key for key in state_dict if key.endswith(".dora_scale")]
    if dora:
        raise ValueError(
            f"{LOG_PREFIX} This is a DoRA ({len(dora)} dora_scale tensors), not a plain LoRA. "
            "Its update is renormalised per column of the weight, which a side branch cannot "
            "reproduce. Use a stock LoRA loader for it."
        )
    config = _adapter_config(metadata)
    pairs, leftover = split_pairs(state_dict)
    targets: list[LoraTarget] = []
    unmatched = [key for key in leftover if not key.endswith(".alpha")]
    for base, (down, up) in pairs.items():
        if down.ndim != 2 or up.ndim != 2:
            unmatched.append(base)      # conv/LoCon — это не линейный слой
            continue
        mapped = key_map.get(base)
        if mapped is None:
            unmatched.append(base)
            continue
        offset = size = None
        if isinstance(mapped, tuple):
            # Ядро кладёт сюда и другие формы — например, (key, None, swap)
            # для norm_out у Flux/SD3. Понимаем только срез выхода слитого слоя.
            slice_spec = mapped[1] if len(mapped) == 2 else None
            if not (isinstance(mapped[0], str) and isinstance(slice_spec, tuple)
                    and len(slice_spec) == 3 and slice_spec[0] == 0):
                unmatched.append(base)
                continue
            mapped, (_dim, offset, size) = mapped[0], slice_spec
        if not (isinstance(mapped, str) and mapped.startswith("diffusion_model.")
                and mapped.endswith(".weight")):
            unmatched.append(base)
            continue
        name = mapped[len("diffusion_model."):-len(".weight")]
        try:
            module = diffusion_model.get_submodule(name)
        except AttributeError:
            unmatched.append(base)
            continue
        weight = getattr(module, "weight", None)
        if weight is None or len(weight.shape) != 2:
            unmatched.append(base)
            continue
        out_features, in_features = int(weight.shape[0]), int(weight.shape[1])
        rank = int(down.shape[0])
        expected_out = size if size is not None else out_features
        if int(down.shape[1]) != in_features or int(up.shape[0]) != expected_out or int(up.shape[1]) != rank:
            raise ValueError(
                f"{LOG_PREFIX} '{base}' does not fit layer '{name}': LoRA {tuple(up.shape)} x "
                f"{tuple(down.shape)}, layer expects [{expected_out}, r] x [r, {in_features}]. "
                "The LoRA was trained for a different model."
            )
        scale = lora_scale(state_dict, base, rank, config) * float(strength)
        targets.append(LoraTarget(name, down, up, scale, offset, size))
    return targets, unmatched


# ─────────────────────────────── исполнение ──────────────────────────────────

#: Типы, в которых ветку можно хранить без потери. fp8 сюда НЕ входит: B·A
#: маленькие, и в fp8 треть их элементов обнуляется (замерено: ошибка ветки
#: 28-56% против 0.3% в bf16) — нода, придуманная ради точности, теряла бы
#: LoRA сильнее штатного слияния.
_BRANCH_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _is_fused_swiglu(module: torch.nn.Module) -> bool:
    """Qwen Image 2.1's MLP: ``out`` runs inside a fused kernel, not through its hooks."""
    return bool(getattr(module, "fused", False)) and hasattr(module, "gate_up") and hasattr(module, "out")


def branch_dtype(*candidates) -> torch.dtype:
    """The first candidate that can hold the branch without loss, else float32."""
    for dtype in candidates:
        if dtype in _BRANCH_DTYPES:
            return dtype
    return torch.float32


class LoraBranch:
    """The LoRA of one node, ready to be attached to a forward call."""

    def __init__(self, targets: list[LoraTarget], label: str):
        self.targets = targets
        self.label = label
        # (device, dtype) -> [(A, B·scale), ...]. Словарь, а не один слот: при
        # multi-GPU клоны модели считают параллельно на разных устройствах.
        self._on_device: dict[tuple, list[tuple[torch.Tensor, torch.Tensor]]] = {}
        self._lock = threading.Lock()
        self.applied = False
        self._warned_silent = False

    def nbytes(self, dtype: torch.dtype) -> int:
        size = torch.empty((), dtype=dtype).element_size()
        return sum((t.down.numel() + t.up.numel()) * size for t in self.targets)

    # Веса на устройстве живут одну выборку: OUTER_SAMPLE их отпускает.
    def weights_on(self, device: torch.device, dtype: torch.dtype) -> list[tuple[torch.Tensor, torch.Tensor]]:
        key = (torch.device(device), dtype)
        with self._lock:
            weights = self._on_device.get(key)
            if weights is None:
                # Та же карта в другом типе больше не нужна — не держать две копии.
                for stale in [k for k in self._on_device if k[0] == key[0]]:
                    del self._on_device[stale]
                weights = []
                for target in self.targets:
                    up = target.up.to(device=device, dtype=torch.float32)
                    if target.scale != 1.0:
                        # Масштаб — в float32 и до приведения: в bf16 он округлился бы дважды.
                        up = up * target.scale
                    weights.append((target.down.to(device=device, dtype=dtype), up.to(dtype)))
                self._on_device[key] = weights
            return weights

    def release(self) -> None:
        with self._lock:
            self._on_device.clear()

    def prefetch(self, guider) -> None:
        """Put the weights on the device BEFORE the model is loaded.

        Сделанное здесь видит менеджер памяти ComfyUI: он освобождает место под
        ветку заранее и потом грузит модель в то, что осталось. Лениво, посреди
        первого шага, эти ~0.7 ГБ появлялись бы мимо него.
        """
        try:
            import comfy.model_management as mm

            patcher = guider.model_patcher
            device = patcher.load_device
            dtype = branch_dtype(patcher.model.get_dtype_inference())
            if torch.device(device).type == "cpu":
                return
            mm.free_memory(self.nbytes(dtype), device)
            self.weights_on(device, dtype)
        except Exception as error:  # noqa: BLE001 - без предзагрузки ветка всё равно ляжет на первом шаге
            logger.debug("%s %s: prefetch skipped: %s", LOG_PREFIX, self.label, error)

    def attach(self, diffusion_model: torch.nn.Module, device: torch.device, dtype: torch.dtype,
               handles: list, fired: set, readers: list) -> set:
        """Register the hooks into ``handles``; return the layer names expected to fire.

        ``readers`` receives the ``gate_up`` of every MLP this LoRA reads from;
        the caller unregisters them when the forward call is over.
        """
        weights = self.weights_on(device, dtype)
        plain: dict[str, list[tuple[int | None, int | None, torch.Tensor, torch.Tensor]]] = {}
        mlp_down: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for target, (down, up) in zip(self.targets, weights):
            parent_name, _, leaf = target.name.rpartition(".")
            parent = diffusion_model.get_submodule(parent_name) if parent_name else diffusion_model
            if leaf == "out" and _is_fused_swiglu(parent):
                mlp_down[parent_name] = (down, up)
            else:
                plain.setdefault(target.name, []).append((target.offset, target.size, down, up))

        # Выход gate_up может записать КАЖДЫЙ наш хук на нём, а не только тот, чьей
        # LoRA он нужен: срабатывают они в порядке установки, и последняя запись
        # несёт поправки всех LoRA сразу. Пишут они лишь тогда, когда у MLP есть
        # читатель (_GateUpSlot.wanted) — хоть от этой LoRA, хоть от соседней.
        recorders = {parent_name + ".gate_up" for parent_name in mlp_down}
        for name in plain:
            parent_name, _, leaf = name.rpartition(".")
            if leaf == "gate_up" and _is_fused_swiglu(
                    diffusion_model.get_submodule(parent_name) if parent_name else diffusion_model):
                recorders.add(name)

        for name in sorted(set(plain) | recorders):
            module = diffusion_model.get_submodule(name)
            handles.append(module.register_forward_hook(
                _linear_hook(name, plain.get(name, []), record=name in recorders, fired=fired)))
        for parent_name, (down, up) in mlp_down.items():
            parent = diffusion_model.get_submodule(parent_name)
            handles.append(parent.register_forward_hook(
                _mlp_hook(parent_name + ".out", parent.gate_up, down, up, fired)))
            _GATE_UP.add_readers(parent.gate_up)
            readers.append(parent.gate_up)
        return set(plain) | {name + ".out" for name in mlp_down}

    def run(self, executor, *args, **kwargs):
        diffusion_model = executor.class_obj
        x = _reference_tensor(args[0] if args else kwargs.get("x"))
        # x уже приведён ядром к типу вычислений (get_dtype_inference); тип
        # весов модели (fp8, int8) здесь ни при чём.
        dtype = branch_dtype(x.dtype, getattr(diffusion_model, "dtype", None))
        handles: list = []
        readers: list = []
        fired: set[str] = set()
        expected: set[str] = set()
        try:
            # Регистрация — ВНУТРИ try: сбой посреди неё не должен оставить
            # хуки навсегда на общей для всех клонов модели.
            expected = self.attach(diffusion_model, x.device, dtype, handles, fired, readers)
            return executor(*args, **kwargs)
        finally:
            for handle in handles:
                handle.remove()
            for module in readers:
                _GATE_UP.remove_readers(module)
            _GATE_UP.forget()
            # Ни один хук не сработал — модель этот шаг не считала (EasyCache,
            # LazyCache отдают кэш, не вызывая её). Это не «слои мимо модулей».
            if fired:
                self.applied = True
                silent = expected - fired
                if silent and not self._warned_silent:
                    self._warned_silent = True
                    logger.warning(
                        "%s %s: %d of %d layers were never called through their module, so their "
                        "LoRA branch was not applied (first: %s). The model runs them another way.",
                        LOG_PREFIX, self.label, len(silent), len(expected), sorted(silent)[0])


def _reference_tensor(x):
    """The tensor whose device and dtype the branch follows.

    Audio-video models hand the diffusion model a LIST of latents: MiniMax H3
    in ComfyUI 0.38.2 passes ``[video, audio]``. Both streams share one device
    and the inference dtype, so the first one stands for all of them.
    """
    while isinstance(x, (list, tuple)):
        if not x:
            raise TypeError(f"{LOG_PREFIX} the diffusion model received an empty latent list")
        x = x[0]
    return x


def _branch(x: torch.Tensor, down: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """``B (A x)`` in the dtype of the layer's input, as PEFT computes it."""
    if down.dtype != x.dtype:
        down, up = down.to(x.dtype), up.to(x.dtype)
    return F.linear(F.linear(x, down), up)


def _linear_hook(name, branches, record, fired):
    def hook(module, inputs, output):
        fired.add(name)
        x = inputs[0]
        for offset, size, down, up in branches:
            delta = _branch(x, down, up).to(output.dtype)
            if offset is None:
                output = output + delta
            else:
                # Кусок слитого слоя: поправка только своей половине выхода.
                # Выход — свежий тензор этого же вызова, править его на месте можно.
                output[..., offset:offset + size] += delta
        if record and _GATE_UP.wanted(module):
            _GATE_UP.remember(module, output)
        return output
    return hook


def _mlp_hook(name, gate_up, down, up, fired):
    def hook(module, inputs, output):
        hidden = _GATE_UP.recall(gate_up)
        if hidden is None:
            return output
        fired.add(name)
        gate, value = hidden.chunk(2, dim=-1)
        # Тот же SwiGLU, что делает ядро перед out: silu(первая половина) * вторая.
        result = output + _branch(F.silu(gate) * value, down, up).to(output.dtype)
        # Последний читатель отпускает тензор ЗДЕСЬ, внутри MLP своего блока.
        _GATE_UP.consumed(gate_up)
        return result
    return hook


def install(patcher, branch: LoraBranch) -> None:
    """Attach ``branch`` to a cloned ModelPatcher through its wrappers."""
    import comfy.patcher_extension as extension

    def diffusion_wrapper(executor, *args, **kwargs):
        return branch.run(executor, *args, **kwargs)

    def outer_sample_wrapper(executor, *args, **kwargs):
        branch.applied = False
        branch.prefetch(getattr(executor, "class_obj", None))
        try:
            result = executor(*args, **kwargs)
        finally:
            branch.release()
        if not branch.applied:
            logger.warning(
                "%s %s: this model never called the diffusion-model wrapper, so the LoRA "
                "was NOT applied. Use a stock LoRA loader for this model.",
                LOG_PREFIX, branch.label)
        return result

    patcher.add_wrapper_with_key(extension.WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY, diffusion_wrapper)
    patcher.add_wrapper_with_key(extension.WrappersMP.OUTER_SAMPLE, WRAPPER_KEY, outer_sample_wrapper)


def _known_loras() -> list[str]:
    try:
        import folder_paths

        return list(folder_paths.get_filename_list("loras"))
    except Exception as error:              # noqa: BLE001 - вне ComfyUI и в тестах
        logger.debug("%s could not list loras: %s", LOG_PREFIX, error)
        return []


class TS_LoraUnmerged(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_LoraUnmerged",
            display_name="TS LoRA Unmerged",
            category="TS/Utils",
            description=(
                "Applies a LoRA as a side branch, y = Wx + BAx, instead of merging it into "
                "the weights. Merging loses a large part of a small LoRA on bf16 weights and "
                "adds noise on int8 — this keeps it whole, at the cost of roughly 10-25% "
                "per step. Made for few-step turbo LoRAs such as Viggle's for Qwen Image 2.1."
            ),
            inputs=[
                IO.Model.Input("model", tooltip="The diffusion model. It is not modified; a clone gets the LoRA."),
                IO.Combo.Input(
                    "lora_name",
                    options=_known_loras(),
                    tooltip="A LoRA of linear layers (diffusers/PEFT or kohya naming).",
                ),
                IO.Float.Input(
                    "strength",
                    # Границы и шаг — как у штатного LoraLoaderModelOnly: TS LoRA
                    # Loader разворачивает строку списка то в него, то в эту ноду,
                    # и сила, принятая одним, не должна отвергаться другой.
                    default=1.0, min=-100.0, max=100.0, step=0.01,
                    tooltip="Multiplier on top of the file's own alpha / rank. Turbo LoRAs want 1.0.",
                ),
            ],
            outputs=[
                IO.Model.Output(
                    display_name="model",
                    tooltip="The same model with the LoRA added at run time.",
                ),
            ],
            search_aliases=["lora", "unmerged lora", "runtime lora", "viggle", "turbo lora",
                            "qwen image 2.1 turbo", "peft"],
        )

    @classmethod
    def execute(cls, model, lora_name: str, strength: float = 1.0) -> IO.NodeOutput:
        patcher = model.clone()
        if float(strength) == 0.0:
            return IO.NodeOutput(patcher)

        import comfy.lora
        import comfy.utils
        import folder_paths

        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        state_dict, metadata = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
        key_map = comfy.lora.model_lora_keys_unet(patcher.model, {})
        diffusion_model = patcher.get_model_object("diffusion_model")
        targets, unmatched = build_targets(state_dict, metadata, key_map, diffusion_model, float(strength))
        if not targets:
            raise RuntimeError(
                f"{LOG_PREFIX} None of the {len(state_dict)} tensors in '{lora_name}' match a linear "
                "layer of this model. It is either for another model or not a plain LoRA "
                "(LoKr, LoHa and conv LoRAs are not supported here)."
            )
        diffs = [key for key in unmatched if key.endswith((".diff", ".diff_b"))]
        if diffs:
            # Полные разности весов (часто — к нормам и смещениям): ветка B·A их
            # не выразит, а молча выкинуть — значит отдать не ту LoRA.
            logger.warning("%s %d full-weight differences (.diff/.diff_b) are NOT applied — a side "
                           "branch can only add B·A. First: %s. A stock LoRA loader applies them.",
                           LOG_PREFIX, len(diffs), diffs[0])
        unmatched = [key for key in unmatched if key not in diffs]
        if unmatched:
            logger.warning("%s %d LoRA entries match nothing in this model and are skipped (first: %s)",
                           LOG_PREFIX, len(unmatched), unmatched[0])
        install(patcher, LoraBranch(targets, lora_name))
        logger.info("%s '%s': %d layers, strength %g", LOG_PREFIX, lora_name, len(targets), strength)
        return IO.NodeOutput(patcher)


NODE_CLASS_MAPPINGS = {"TS_LoraUnmerged": TS_LoraUnmerged}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_LoraUnmerged": "TS LoRA Unmerged"}
