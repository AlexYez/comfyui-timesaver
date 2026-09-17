"""TS DLSS Upscaler — NVIDIA DLSS 5 Neural Rendering for images and video batches.

One node: pictures in, upscaled pictures out. On its first run it downloads the
runtime it needs into ``models/DLSS`` and lays it out the way the engine expects.

⚠️ Since 17.09.2026 the node runs the upstream **v9 "Neuroframe Engine"**: the
feature is evaluated in this process through a C ABI, not by a worker process
over a pipe. The picture is resized to the output size first and the network
re-renders it there — the factor chooses a size, the network adds the detail.

⚠️ This is a TEMPORAL renderer. A batch of consecutive video frames shares one
temporal history, so detail is carried across frames — that is where the quality
above a still upscaler comes from. A batch of unrelated pictures must be run
with ``temporal`` off, or each picture drags the previous one's detail behind it.

Windows and an NVIDIA RTX card only: the work is done by NVIDIA's signed D3D12
runtime, and there is no other implementation of it.
"""

from __future__ import annotations

import logging
import sys
import time

from comfy_api.v0_0_2 import IO

from ..._deps import TSDependencyManager
from . import _assets
from ._dither import quantise_float_to_rgba8
from ._session import (
    DLSSFrameSession,
    NR_STYLES,
    MIN_EDGE,
    UPSCALING_LABELS,
    factor_from_label,
    resolve_native_settings,
    resize_fit,
    resolve_output_size,
    resolve_upscaling_mode,
    verify_feature_18,
)
from ._tonemap import TRANSFER_CHOICES, TRANSFER_LABELS, needs_tone_map, tone_map_for

logger = logging.getLogger("comfyui_timesaver.ts_dlss_upscaler")
LOG_PREFIX = "[TS DLSS Upscaler]"

DEFAULT_FACTOR_LABEL = "2× (Performance)"
_CURVE_LABELS = [TRANSFER_LABELS[name] for name in TRANSFER_CHOICES]
_LABEL_TO_CURVE = {TRANSFER_LABELS[name]: name for name in TRANSFER_CHOICES}

#: ⚠️ Мёртвый выбор, оставленный НАРОЧНО. У движка v9 пресетов сети нет — он
#: выбирает её сам, — но `widgets_values` в сохранённом графе позиционный:
#: убрать виджет значит сдвинуть все следующие, и у человека в открывшемся
#: workflow стиль окажется равен "M". Стоит третьим, как стоял.
LEGACY_MODEL_PRESETS = ["Default", "J", "K", "L", "M"]

_assets.register_model_folder()


class TS_DLSSUpscaler(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_DLSSUpscaler",
            display_name="TS DLSS Upscaler",
            category="TS/Video",
            description=(
                "Upscale a picture or a video batch with NVIDIA DLSS 5 Neural Rendering. "
                "Downloads its runtime into models/DLSS on first use. Windows + RTX only."
            ),
            inputs=[
                IO.Image.Input(
                    "images",
                    tooltip=(
                        "One picture or a batch of frames [B,H,W,C]. Both sides must be at "
                        "least 64 pixels."
                    ),
                ),
                IO.Combo.Input(
                    "upscaling_factor",
                    options=UPSCALING_LABELS,
                    default=DEFAULT_FACTOR_LABEL,
                    tooltip=(
                        "Output size. 1× is DLAA: no resize, the network re-renders at the "
                        "same size. The output is capped at 7680×4320."
                    ),
                ),
                IO.Combo.Input(
                    "dlss_model_preset",
                    options=LEGACY_MODEL_PRESETS,
                    default="M",
                    tooltip=(
                        "Ignored since the v9 runtime — the engine picks the network itself. "
                        "Kept so older workflows keep their other settings lined up."
                    ),
                ),
                IO.Combo.Input(
                    "nr_style",
                    options=list(NR_STYLES),
                    default="Default",
                    tooltip="Neural rendering look: Default, Natural or Cinematic.",
                ),
                IO.Float.Input(
                    "nr_intensity",
                    default=1.0, min=0.0, max=1.0, step=0.01,
                    tooltip="How strongly neural rendering is applied.",
                ),
                IO.Float.Input(
                    "local_tone_strength",
                    default=1.0, min=0.0, max=2.0, step=0.01,
                    tooltip="Local contrast recovered by the network.",
                ),
                IO.Float.Input(
                    "local_structure_strength",
                    default=1.0, min=0.0, max=2.0, step=0.01,
                    tooltip="Local detail recovered by the network.",
                ),
                IO.Float.Input(
                    "skin_structure_strength",
                    default=2.0, min=-1.0, max=2.0, step=0.01,
                    tooltip="Detail on skin, kept separate so faces do not turn plastic.",
                ),
                IO.Boolean.Input(
                    "automatic_mask",
                    default=True,
                    tooltip=(
                        "Let the runtime decide where neural rendering is applied. Skin "
                        "structure and face protection need this on."
                    ),
                ),
                IO.Boolean.Input(
                    "temporal",
                    default=True,
                    tooltip=(
                        "Treat the batch as consecutive video frames: one temporal history, "
                        "restarted at scene cuts. Switch OFF for a batch of unrelated "
                        "pictures. Ignored for a single image."
                    ),
                ),
                IO.Combo.Input(
                    "source_curve",
                    options=_CURVE_LABELS,
                    default=TRANSFER_LABELS["sdr"],
                    tooltip=(
                        "What the incoming values mean. The network only understands "
                        "display-referred SDR, so log/HDR/linear pictures are converted before "
                        "it and converted back after — exactly, by the same curve."
                    ),
                ),
                IO.Boolean.Input(
                    "dither",
                    default=True,
                    tooltip=(
                        "Blue-noise dither on the way down to 8 bits, so gradients reach the "
                        "network as detail instead of steps."
                    ),
                ),
                # ⚠️ Умолчание снова ON — по прямому решению владельца пака
                # (17.09.2026). Выключенным его ставили ради сканера реестра,
                # который всё равно держит пак во Flagged независимо от этого;
                # цена была в том, что нода при первом запуске просто падала с
                # текстом, пока человек не найдёт этот выключатель. Уведомление
                # о происхождении файлов печатается ДО первого запроса в сеть и
                # никуда не делось.
                IO.Boolean.Input(
                    "download_if_missing",
                    default=True,
                    tooltip=(
                        "Fetch the runtime into models/DLSS when it is not there "
                        "(~486 MB, once) from the upstream project's release. It contains "
                        "NVIDIA's proprietary DLSSNR runtime and the MIT-licensed engine; "
                        "this pack hosts none of it. Off, the node downloads nothing and "
                        "you place the files yourself."
                    ),
                ),
                # ---------------------------------------------------- v9 controls
                # ⚠️ Всё новое добавлено В КОНЕЦ, после `download_if_missing`.
                # Порядок входов — это порядок `widgets_values`: вставка в
                # середину переписала бы значения в уже сохранённых графах.
                IO.Int.Input(
                    "nr_passes",
                    default=1, min=1, max=4,
                    tooltip=(
                        "How many times the network goes over the frame. Each pass costs "
                        "another full evaluation; 1 is what the reference application ships."
                    ),
                ),
                IO.Float.Input(
                    "nr_color_strength",
                    default=1.0, min=0.0, max=1.0, step=0.01,
                    tooltip=(
                        "How much of the network's colour reaches the result. 1 is the "
                        "network's own; lower keeps more of the source's."
                    ),
                ),
                IO.Float.Input(
                    "tone_preservation",
                    default=0.0, min=0.0, max=1.0, step=0.01,
                    tooltip="Hold the source's tonality against the network's own.",
                ),
                IO.Float.Input(
                    "face_skin_protection",
                    default=0.0, min=0.0, max=1.0, step=0.01,
                    tooltip="Extra restraint on faces. Needs the automatic mask.",
                ),
                IO.Float.Input(
                    "grain_preservation",
                    default=0.0, min=0.0, max=1.0, step=0.01,
                    tooltip=(
                        "Keep the source's grain instead of letting the network clean it "
                        "away. Raise it for film scans."
                    ),
                ),
                IO.Float.Input(
                    "shimmer_suppression",
                    default=0.70, min=0.0, max=1.0, step=0.01,
                    tooltip=(
                        "The engine's GPU temporal stabiliser: it estimates motion itself "
                        "and steadies detail between frames. Used only with 'temporal' on "
                        "and more than one frame; a still is always rendered with 0."
                    ),
                ),
            ],
            outputs=[
                IO.Image.Output(
                    display_name="images",
                    tooltip="The upscaled batch, same order, same channel count.",
                )
            ],
            search_aliases=["dlss", "dlss 5", "neural rendering", "upscale", "nvidia"],
        )

    # ------------------------------------------------------------------ guards
    @classmethod
    def _require_platform(cls) -> None:
        if sys.platform != "win32":
            raise RuntimeError(
                f"{LOG_PREFIX} DLSS 5 Neural Rendering runs through NVIDIA's signed D3D12 "
                "runtime, which exists for Windows only."
            )

    @classmethod
    def _require_numpy(cls):
        numpy = TSDependencyManager.import_optional("numpy")
        if numpy is None:
            raise RuntimeError(f"{LOG_PREFIX} numpy is required.")
        return numpy

    # ------------------------------------------------------------------- frames
    @classmethod
    def _to_rgba8(cls, np, frame, tone_map, dither: bool):
        """One IMAGE frame -> the RGBA8 bytes the network is fed."""
        rgb = np.asarray(frame[..., :3], dtype=np.float32)
        if tone_map is not None:
            rgb = tone_map.forward(rgb)
        alpha = np.asarray(frame[..., 3], dtype=np.float32) if frame.shape[-1] >= 4 else None
        return np.ascontiguousarray(quantise_float_to_rgba8(rgb, alpha, dither=dither))

    @classmethod
    def _undo_curve(cls, np, tone_map, into):
        """Bring the rendered frame back to the curve it arrived on."""
        if tone_map is None:
            return
        into[...] = tone_map.inverse(into)

    @classmethod
    def _resize_alpha(cls, np, alpha, width: int, height: int):
        """Carry the source alpha to the output size; the network ignores alpha."""
        if alpha.shape[0] == height and alpha.shape[1] == width:
            return alpha
        cv2 = TSDependencyManager.import_optional("cv2")
        if cv2 is None:
            # Without OpenCV a nearest-neighbour lift is still better than losing it.
            ys = (np.arange(height) * alpha.shape[0] // height).clip(0, alpha.shape[0] - 1)
            xs = (np.arange(width) * alpha.shape[1] // width).clip(0, alpha.shape[1] - 1)
            return alpha[ys][:, xs]
        return cv2.resize(alpha, (width, height), interpolation=cv2.INTER_LANCZOS4)

    # ------------------------------------------------------------------ execute
    @classmethod
    def execute(
        cls,
        images,
        upscaling_factor: str,
        dlss_model_preset: str,
        nr_style: str,
        nr_intensity: float,
        local_tone_strength: float,
        local_structure_strength: float,
        skin_structure_strength: float,
        automatic_mask: bool,
        temporal: bool,
        source_curve: str,
        dither: bool,
        download_if_missing: bool,
        nr_passes: int = 1,
        nr_color_strength: float = 1.0,
        tone_preservation: float = 0.0,
        face_skin_protection: float = 0.0,
        grain_preservation: float = 0.0,
        shimmer_suppression: float = 0.70,
    ) -> IO.NodeOutput:
        import torch  # noqa: PLC0415 - always present in ComfyUI, never at import time

        del dlss_model_preset  # the v9 engine picks the network itself
        cls._require_platform()
        np = cls._require_numpy()

        if images is None or images.ndim != 4:
            raise ValueError(f"{LOG_PREFIX} 'images' must be a batch shaped [B,H,W,C].")
        batch, height, width, channels = (int(value) for value in images.shape)
        if height < MIN_EDGE or width < MIN_EDGE:
            raise ValueError(
                f"{LOG_PREFIX} DLSS needs at least {MIN_EDGE}×{MIN_EDGE} pixels; "
                f"this batch is {width}×{height}."
            )

        factor, mode = resolve_upscaling_mode(factor_from_label(upscaling_factor))
        output_width, output_height = resolve_output_size(width, height, factor)
        # ⚠️ Стабилизатор — понятие временнОе. На одиночной картинке или с
        # выключенным temporal ему нечего сравнивать, и включённым он будет
        # держать историю от предыдущего, ничем не связанного кадра.
        temporal_active = bool(temporal) and batch > 1
        native = resolve_native_settings(
            nr_style=nr_style,
            nr_intensity=nr_intensity,
            nr_passes=nr_passes,
            local_tone_strength=local_tone_strength,
            local_structure_strength=local_structure_strength,
            skin_structure_strength=skin_structure_strength,
            automatic_mask=automatic_mask,
            nr_color_strength=nr_color_strength,
            tone_preservation=tone_preservation,
            face_skin_protection=face_skin_protection,
            grain_preservation=grain_preservation,
            shimmer_suppression=shimmer_suppression if temporal_active else 0.0,
        )

        curve = _LABEL_TO_CURVE.get(source_curve, source_curve)
        tone_map = tone_map_for(curve) if needs_tone_map(curve) else None

        root = _assets.ensure_runtime(
            download_if_missing=download_if_missing,
            progress=cls._download_progress(),
        )
        cls._mention_obsolete_files(root)

        # ⚠️ The source tensor is never written to: everything below reads from a
        # CPU copy and builds new arrays.
        source = images.detach().to("cpu")

        session = DLSSFrameSession(
            root,
            output_width=output_width,
            output_height=output_height,
            native_settings=native,
            cuda_ordinal=cls._cuda_ordinal(torch),
            cancelled=cls._cancelled,
        )
        logger.info(
            "%s %d frame(s) %d×%d -> %d×%d, %s, %d pass(es)%s, on %s.",
            LOG_PREFIX, batch, width, height, output_width, output_height,
            mode["name"], int(nr_passes),
            ", temporal" if temporal_active else "",
            session.bridge_status.get("gpu_name", "unknown"),
        )

        guides = None
        if temporal_active:
            from ._guides import TemporalGuideGenerator  # noqa: PLC0415 - needs OpenCV

            guides = TemporalGuideGenerator(output_width, output_height)

        progress = cls._progress_bar(batch)
        results = np.empty((batch, output_height, output_width, min(channels, 4)),
                           dtype=np.float32)
        # ⚠️ Секундомер на каждой стороне. Вопрос «почему медленно» иначе решается
        # догадками, а ответ у него не один: сеть, подготовка кадра и выдача
        # стоят по-разному, и на 4K они одного порядка.
        prepare_seconds = 0.0
        deliver_seconds = 0.0
        # ⚠️ Сессия пишет результат прямо в батч, а для этого ей нужен
        # непрерывный кусок памяти. Он такой и есть, пока каналов три; с альфой
        # срез `[..., :3]` идёт с пропусками, и тогда нужен свой буфер.
        scratch = (np.empty((output_height, output_width, 3), dtype=np.float32)
                   if channels >= 4 else None)
        started = time.perf_counter()
        try:
            for index in range(batch):
                cls._raise_if_interrupted()
                mark = time.perf_counter()
                frame = source[index].numpy()
                rgba8 = cls._to_rgba8(np, frame, tone_map, dither)
                rgba8 = resize_fit(rgba8, output_width, output_height)
                reset = True if guides is None else guides.process(rgba8).reset
                prepare_seconds += time.perf_counter() - mark

                target = results[index] if scratch is None else scratch
                session.render_into(index=index, rgba=rgba8, reset=reset,
                                    destination=target)

                mark = time.perf_counter()
                cls._undo_curve(np, tone_map, target)
                if scratch is not None:
                    results[index, ..., :3] = scratch
                    results[index, ..., 3] = cls._resize_alpha(
                        np, np.asarray(frame[..., 3], dtype=np.float32),
                        output_width, output_height,
                    )
                deliver_seconds += time.perf_counter() - mark
                if progress is not None:
                    progress.update_absolute(index + 1, batch)
            cls._report_evidence(session)
            cls._report_time(batch, time.perf_counter() - started, prepare_seconds,
                             session.evaluate_seconds, deliver_seconds, session.memory_path)
            session.close()
        except BaseException:
            session.abort()
            raise

        # ⚠️ Кламп только там, где он может понадобиться. Восьмибитный результат
        # движка, поделённый на 255, за границы не выходит по построению, а
        # `np.clip` на батче 4K — это ещё одна копия на 755 МБ и 24 мс на кадр.
        # Обратная кривая тон-мапа — другое дело: она вправе промахнуться.
        if tone_map is not None:
            np.clip(results, 0.0, 1.0, out=results)
        return IO.NodeOutput(torch.from_numpy(results).to(images.device))

    # ------------------------------------------------------------------ plumbing
    @classmethod
    def _cuda_ordinal(cls, torch) -> int:
        """The card ComfyUI is working on; the engine is brought up on the same one."""
        try:
            import comfy.model_management as mm  # noqa: PLC0415

            device = mm.get_torch_device()
            if getattr(device, "type", "") == "cuda":
                index = getattr(device, "index", None)
                if index is not None:
                    return int(index)
                return int(torch.cuda.current_device())
        except Exception as exc:  # noqa: BLE001 - outside ComfyUI, or a CPU device
            logger.debug("%s Could not read ComfyUI's device: %s", LOG_PREFIX, exc)
        return 0

    @classmethod
    def _mention_obsolete_files(cls, root) -> None:
        """Say once that the v5 runtime in models/DLSS is no longer read."""
        leftovers = _assets.obsolete_files(root)
        if leftovers:
            logger.info(
                "%s The old v5 runtime is still in %s (%s). Nothing reads it since the v9 "
                "engine; you can delete host/ and dlss/ to get ~180 MB back.",
                LOG_PREFIX, root, ", ".join(leftovers),
            )

    @classmethod
    def _cancelled(cls) -> bool:
        try:
            import comfy.model_management as mm  # noqa: PLC0415

            return bool(mm.processing_interrupted())
        except Exception:  # noqa: BLE001 - outside ComfyUI there is nothing to cancel
            return False

    @classmethod
    def _raise_if_interrupted(cls) -> None:
        try:
            import comfy.model_management as mm  # noqa: PLC0415

            mm.throw_exception_if_processing_interrupted()
        except ImportError:
            pass

    @classmethod
    def _progress_bar(cls, total: int):
        try:
            import comfy.utils  # noqa: PLC0415

            return comfy.utils.ProgressBar(total)
        except Exception:  # noqa: BLE001 - no server, no bar
            return None

    @classmethod
    def _download_progress(cls):
        """A progress bar for the one-off runtime download, in whole megabytes."""
        state: dict[str, object] = {"bar": None}

        def report(done: int, total: int) -> None:
            if not total:
                return
            bar = state["bar"]
            if bar is None:
                bar = cls._progress_bar(total // (1024 * 1024) or 1)
                state["bar"] = bar
            if bar is not None:
                bar.update_absolute(done // (1024 * 1024), total // (1024 * 1024) or 1)

        return report

    @classmethod
    def _report_time(cls, frames: int, total: float, prepare: float,
                     engine: float, deliver: float, memory_path: str) -> None:
        """Say where the time went, in the same line every run."""
        if frames <= 0:
            return
        logger.info(
            "%s Time per frame: %.2f s total — %.2f s preparing, %.2f s in the network, "
            "%.2f s delivering (%d frame(s) in %.1f s, frames travel through %s memory).",
            LOG_PREFIX, total / frames, prepare / frames, engine / frames,
            deliver / frames, frames, total, memory_path,
        )

    @classmethod
    def _report_evidence(cls, session) -> None:
        """Say what actually ran, instead of leaving it to be assumed.

        ⚠️ Reported, never enforced: the pictures are already rendered, and
        refusing them because a counter is missing would throw away the run.
        """
        try:
            status = session.structured_status()
            evidence = verify_feature_18(session.logs, status)
        except Exception as exc:  # noqa: BLE001 - diagnostics must not fail a good run
            logger.debug("%s Could not read the session evidence: %s", LOG_PREFIX, exc)
            return
        if not evidence["verified"]:
            logger.warning(
                "%s The run finished but no frame was recorded as evaluated — the result "
                "may not be DLSS-enhanced.", LOG_PREFIX,
            )
            return
        frames = evidence["successful_frames"]
        seconds = float(evidence["evaluate_seconds"]) or 0.0
        logger.info(
            "%s Neuroframe Engine %s on %s: %d frame(s) in %.1f s (%.2f s/frame), "
            "%d scene reset(s), motion %s, CUDA %s.",
            LOG_PREFIX, evidence["engine_version"], evidence["gpu_name"], frames,
            seconds, seconds / max(1, frames), evidence["scene_resets"],
            evidence["motion_backend"], evidence["cuda_status"],
        )


NODE_CLASS_MAPPINGS = {"TS_DLSSUpscaler": TS_DLSSUpscaler}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_DLSSUpscaler": "TS DLSS Upscaler"}
