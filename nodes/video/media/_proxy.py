"""Маленькая копия исходника для плеера в теле ноды.

Зачем она нужна. Браузер умеет ровно четыре видеокодека — H.264, VP8, VP9, AV1.
ProRes, DNxHD, MPEG-2, HEVC он не декодирует: звук из такого файла играет, а на
месте картинки чёрный прямоугольник. Сами кадры нода при этом читает прекрасно —
ломается только предпросмотр, потому что маршрут ``/view`` отдаёт файл как есть.

⚠️ Копия строится ТОЛЬКО по прямой просьбе человека. Это один полный проход
ffmpeg по всему ролику: на часовом 4K ProRes он идёт минутами, и запускать такое
молча, потому что кто-то выбрал файл, нельзя. Зато уже построенную копию
подхватывают сразу, без кнопки.

Готовая копия лежит рядом с пробами и лентами кадров в ``.cache/ts_video`` и
подметается тем же сроком (``_probe._sweep_cache_once``): ключ — путь вместе с
размером и временем правки, поэтому перезапись исходника тем же именем честно
даёт новую копию.

Такая же копия есть у TS Video Saver (``_encode.write_proxy``), но там на входе
кадры уже в памяти, а здесь — файл на диске, и гонять его через Python незачем:
один вызов ffmpeg дешевле и не держит ролик в памяти.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from ..._ffmpeg import require_ffmpeg
from ._common import LOG_PREFIX, cache_dir, file_identity, safe_log_path

logger = logging.getLogger("comfyui_timesaver.ts_video.proxy")

#: Длинная сторона копии. Предпросмотру больше не нужно, а каждый лишний пиксель
#: — это время полного прохода по ролику.
MAX_LONG_SIDE = 1280

#: Кодеки, которые браузер проигрывает сам. Всё остальное просит копию.
BROWSER_CODECS = frozenset({"h264", "vp8", "vp9", "av1"})

_SUFFIX = ".proxy.mp4"

#: Идёт ли сейчас сборка и насколько далеко зашла: ключ файла -> состояние.
#: Живёт в модуле, а не на классе ноды — V3 запрещает писать в атрибуты класса.
_jobs: dict[str, dict] = {}


def browser_playable(codec: str | None) -> bool:
    """Проиграет ли браузер этот кодек своими силами."""
    return str(codec or "").lower() in BROWSER_CODECS


def proxy_file(source: str | os.PathLike[str]) -> Path:
    """Куда ляжет копия для этого исходника."""
    return cache_dir() / f"{file_identity(source)}{_SUFFIX}"


def status(source: str | os.PathLike[str]) -> dict:
    """Что сейчас с копией: есть, строится, нет или не получилось."""
    key = file_identity(source)
    target = proxy_file(source)
    job = _jobs.get(key)
    if job is not None and job.get("state") == "building":
        return {"state": "building", "progress": float(job.get("progress", 0.0))}
    if target.is_file() and target.stat().st_size > 0:
        return {"state": "ready", "progress": 1.0}
    if job is not None and job.get("error"):
        return {"state": "failed", "progress": 0.0, "error": str(job["error"])}
    return {"state": "absent", "progress": 0.0}


def _command(ffmpeg: str, source: str, target: str) -> list[str]:
    """Один проход: уменьшить длинную сторону, сжать в H.264, звук в AAC.

    ⚠️ Масштаб выражением, а не числами: материал приходит и альбомный, и
    портретный, и «ширина 1280» на вертикальном ролике значит совсем не то же
    самое. Ограничивается именно ДЛИННАЯ сторона, а короткая считается сама
    (``-2`` — ближайшее чётное, кодировщику нужны чётные стороны).
    """
    long_side = str(MAX_LONG_SIDE)
    scale = (f"scale='if(gte(iw,ih),min({long_side},iw),-2)'"
             f":'if(gte(iw,ih),-2,min({long_side},ih))'")
    return [
        ffmpeg, "-y", "-nostdin",
        "-i", source,
        "-map", "0:v:0", "-map", "0:a:0?",
        "-vf", scale,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", "-ac", "2",
        # Индекс в начало: иначе браузер ради перемотки тянет весь файл.
        "-movflags", "+faststart",
        "-progress", "pipe:1", "-nostats", "-loglevel", "error",
        # ⚠️ Контейнер НАЗЫВАЕТСЯ ЯВНО: пишем во временный файл с суффиксом
        # `.tmp`, и по расширению ffmpeg мультиплексор не угадывает — отвечает
        # «Invalid argument» и не пишет ничего.
        "-f", "mp4",
        target,
    ]


async def _run(source: str, target: Path, duration: float, job: dict) -> None:
    """Собрать копию, обновляя прогресс по ходу дела."""
    ffmpeg = require_ffmpeg()
    target.parent.mkdir(parents=True, exist_ok=True)
    # ⚠️ Пишем во временный файл и переименовываем: параллельный читатель не
    # должен увидеть половину ролика и решить, что копия готова.
    tmp = target.with_name(f"{target.name}.{os.getpid()}.tmp")

    process = await asyncio.create_subprocess_exec(
        *_command(ffmpeg, source, str(tmp)),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        assert process.stdout is not None
        async for raw in process.stdout:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("out_time_us=") or duration <= 0:
                continue
            try:
                seconds = int(line.split("=", 1)[1]) / 1_000_000.0
            except ValueError:
                continue
            job["progress"] = max(0.0, min(0.999, seconds / duration))
        await process.wait()
        if process.returncode != 0:
            details = (await process.stderr.read()).decode("utf-8", "replace").strip()
            raise RuntimeError(details.splitlines()[-1] if details else
                               f"ffmpeg exited with {process.returncode}")
        os.replace(tmp, target)
        job["progress"] = 1.0
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
        try:
            tmp.unlink()
        except OSError:
            pass


async def build(source: str, duration: float) -> dict:
    """Запустить сборку копии, если её ещё нет и она не строится.

    Возвращает текущее состояние — тот же словарь, что и ``status``.
    """
    key = file_identity(source)
    current = status(source)
    if current["state"] in ("ready", "building"):
        return current

    job: dict = {"state": "building", "progress": 0.0, "error": None}
    _jobs[key] = job
    target = proxy_file(source)

    async def worker() -> None:
        try:
            await _run(source, target, float(duration), job)
            job["state"] = "ready"
            logger.info("%s preview copy ready for %s", LOG_PREFIX, safe_log_path(source))
        except Exception as error:          # noqa: BLE001 - причина уходит в интерфейс
            job["state"] = "failed"
            job["error"] = str(error)
            logger.warning("%s preview copy failed for %s: %s",
                           LOG_PREFIX, safe_log_path(source), error)

    job["task"] = asyncio.create_task(worker())
    return {"state": "building", "progress": 0.0}
