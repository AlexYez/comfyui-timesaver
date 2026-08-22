"""TS Music Stems — разделение музыки на стемы.

Три движка за одним выбором ``model_name``:

* **BS-RoFormer SW** — шесть стемов, дефолт;
* **Mel-Band RoFormer** — вокал и инструментал, специалист на одну границу;
* **Demucs** — как было, ради сохранённых графов.

⚠️ **Контракт заморожен.** Пять первых выходов и все входы остались на своих
местах; guitar и piano ДОПИСАНЫ шестым и седьмым, потому что дописывание в
конец не сдвигает индексы в сохранённых связях. ``htdemucs*`` остались
валидными значениями ``model_name``, и на них нода считает ровно как раньше.

⚠️ Стемы, которых выбранная модель не даёт, отдаются ``ExecutionBlocker``, а не
тишиной: ветка графа просто не исполняется. Тишина выглядела бы как поломка
модели, и человек искал бы её часами.
"""

from __future__ import annotations

import logging

import comfy.model_management as mm
import comfy.utils
import folder_paths
import torch
from comfy_api.v0_0_2 import IO
from comfy_execution.graph_utils import ExecutionBlocker

from . import _catalog, _demucs, _roformer

logger = logging.getLogger("comfyui_timesaver.ts_music_stems")
LOG_PREFIX = "[TS Music Stems]"

TARGET_SR = _catalog.SAMPLE_RATE

# ⚠️ Порядок значений — контракт: он лежит в сохранённых графах.
MODEL_OPTIONS = [
    _catalog.BS_ROFORMER_SW.key,
    _catalog.MELBAND_VOCALS.key,
    *_demucs.MODEL_NAMES,
]


class TS_MusicStems(IO.ComfyNode):
    """Разделение на стемы: два роформера плюс исторический Demucs."""

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_MusicStems",
            display_name="TS Music Stems",
            category="TS/Audio",
            description=(
                "Split music into stems. BS-RoFormer SW gives six; Mel-Band RoFormer "
                "gives vocals and instrumental and is the better choice when that is "
                "all you need; Demucs is kept for older workflows."
            ),
            inputs=[
                IO.Audio.Input("audio", tooltip="Audio track to split into stems."),
                IO.Combo.Input(
                    "model_name",
                    options=MODEL_OPTIONS,
                    default=_catalog.BS_ROFORMER_SW.key,
                    tooltip=(
                        "bs_roformer_sw: six stems, best overall. "
                        "melband_roformer: vocals + instrumental only, and better at "
                        "that one split than any six-stem model. "
                        "htdemucs*: the older engine, kept so saved workflows keep "
                        "producing what they always produced."
                    ),
                ),
                IO.Combo.Input(
                    "device",
                    options=["cuda", "cpu", "auto"],
                    default="auto",
                    tooltip="Compute device. auto picks GPU when available, otherwise CPU.",
                ),
                IO.Int.Input(
                    "shifts", default=2, min=0, max=10,
                    tooltip="Demucs only: TTA passes. 2 = high quality, 4 = very slow. "
                            "Ignored by the RoFormer engines.",
                ),
                IO.Float.Input(
                    "overlap", default=0.5, min=0.0, max=0.9,
                    tooltip="Chunk overlap for smoother stitching. The RoFormer engines "
                            "cap this at 0.5, which is all their windowing can use.",
                ),
                IO.Int.Input(
                    "jobs", default=0, min=0, max=16,
                    tooltip="Demucs only: CPU workers for pre-processing. 0 = auto. "
                            "Ignored by the RoFormer engines.",
                ),
                # ⚠️ Новый вход ДОПИСАН последним: widgets_values позиционный, и
                # вставка в середину переставила бы значения в старых графах.
                #
                # ⚠️ И он ОБЯЗАТЕЛЬНО optional. Интерфейс подставил бы значение
                # сам, но граф в API-формате, сохранённый до этой правки, несёт
                # ровно шесть входов — и сервер отвечал на него 400 «Required
                # input is missing: precision». Замерено на живом сервере.
                IO.Combo.Input(
                    "precision",
                    options=list(_roformer.PRECISIONS),
                    default="fp16",
                    optional=True,
                    tooltip=(
                        "RoFormer engines only. fp16 runs about twice as fast on half the "
                        "VRAM; measured against fp32 on real music its error stays at "
                        "-61 dBFS or below, under the noise floor of the recording. "
                        "bfloat16 is not offered: these models build their mask through "
                        "view_as_complex, which does not accept it."
                    ),
                ),
            ],
            outputs=[
                IO.Audio.Output(display_name="vocal", tooltip="Isolated vocals stem."),
                IO.Audio.Output(display_name="bass", tooltip="Isolated bass stem."),
                IO.Audio.Output(display_name="drums", tooltip="Isolated drums stem."),
                IO.Audio.Output(
                    display_name="others",
                    tooltip="Everything that is not vocals, bass or drums. With "
                            "BS-RoFormer SW that includes guitar and piano, which are "
                            "also available separately below.",
                ),
                IO.Audio.Output(
                    display_name="instrumental",
                    tooltip="The full mix minus vocals.",
                ),
                IO.Audio.Output(
                    display_name="guitar",
                    tooltip="BS-RoFormer SW only. Blocked on the other engines.",
                ),
                IO.Audio.Output(
                    display_name="piano",
                    tooltip="BS-RoFormer SW only. Blocked on the other engines.",
                ),
            ],
        )

    # ── общее ────────────────────────────────────────────────────────────
    @staticmethod
    def _normalize_waveform_shape(waveform):
        if not torch.is_tensor(waveform):
            raise TypeError("[TS Music Stems] Audio waveform must be a torch.Tensor.")
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)
        if waveform.ndim != 3:
            raise ValueError(
                f"[TS Music Stems] Expected waveform shape [batch, channels, samples], "
                f"got {tuple(waveform.shape)}."
            )
        if waveform.shape[0] < 1 or waveform.shape[1] < 1 or waveform.shape[2] < 1:
            raise ValueError(f"[TS Music Stems] Waveform has invalid shape: {tuple(waveform.shape)}.")
        return waveform

    @staticmethod
    def _prepare_demucs_waveform(waveform):
        return _demucs.prepare_waveform(waveform)

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return mm.get_torch_device()
        target_device = torch.device(device)
        if target_device.type == "cuda" and not torch.cuda.is_available():
            # "cuda" stays in the list because saved workflows carry it as a
            # widget value, and dropping the option would break them. What
            # changes is the answer on a machine that has no CUDA — an Apple
            # Silicon Mac, most of all: it gets whatever accelerator it does
            # have instead of "Torch not compiled with CUDA enabled".
            target_device = mm.get_torch_device()
            logger.warning(
                "%s CUDA was selected but is unavailable; using %s instead.",
                LOG_PREFIX, target_device,
            )
        return target_device

    @staticmethod
    def _resample(waveform, sample_rate):
        if sample_rate == TARGET_SR:
            return waveform
        logger.info("%s Resampling %s -> %s Hz", LOG_PREFIX, sample_rate, TARGET_SR)
        # Lazy: torchaudio pulls in sox/ffmpeg adapters; load only when actually resampling.
        import torchaudio

        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=TARGET_SR).to(waveform.device)
        return resampler(waveform)

    @staticmethod
    def _audio(tensor) -> dict:
        return {"waveform": tensor, "sample_rate": TARGET_SR}

    # ── роформеры ────────────────────────────────────────────────────────
    @classmethod
    def _run_roformer(cls, model_key, waveform, target_device, overlap, precision):
        model_spec = _catalog.BY_KEY[model_key]
        weights = _catalog.locate_weights(model_spec)

        cached = _model_cache.get(model_key)
        if cached is None:
            logger.info("%s Loading %s", LOG_PREFIX, model_spec.display)
            cached = _roformer.build_model(model_spec, weights)
            _model_cache[model_key] = cached
        net = cached

        dtype, autocast = _roformer.resolve_dtype(precision, target_device)
        net = net.to(device=target_device)
        if dtype is torch.float16:
            net = net.to(dtype=torch.float16)
        elif next(net.parameters()).dtype is not torch.float32:
            net = net.to(dtype=torch.float32)

        # ⚠️ Роформер видит МОНО-СВЕДЕНИЕ ОДНОЙ дорожки пачки за раз: модель
        # обучена на стерео-миксе, а не на батче, и её внутренний STFT не
        # рассчитан на чужую размерность.
        mix = waveform[0]
        if mix.shape[0] == 1:
            mix = mix.repeat(2, 1)
        elif mix.shape[0] > 2:
            logger.info("%s Input has %d channels, using the first two.",
                        LOG_PREFIX, mix.shape[0])
            mix = mix[:2]

        chunks_total = len(_roformer.plan_chunks(
            mix.shape[-1], model_spec.chunk_samples, min(overlap, 0.5)))
        pbar = comfy.utils.ProgressBar(chunks_total)

        def on_chunk(done, total):
            # Настоящий прогресс, а не пульс: у нас свой цикл по чанкам.
            pbar.update_absolute(done, total=total)
            mm.throw_exception_if_processing_interrupted()

        residual = model_spec.stems.index("other") if "other" in model_spec.stems else None

        try:
            stems = _roformer.separate(
                net, mix,
                num_stems=model_spec.num_stems,
                chunk=model_spec.chunk_samples,
                overlap=min(overlap, 0.5),
                device=target_device,
                dtype=dtype,
                autocast=autocast,
                residual_index=residual,
                on_chunk=on_chunk,
            )
        finally:
            # Освобождаем VRAM, но модель оставляем в кэше на CPU.
            net.to(mm.unet_offload_device())
            if target_device.type == "cuda":
                torch.cuda.empty_cache()

        by_name = {name: stems[i].unsqueeze(0) for i, name in enumerate(model_spec.stems)}
        by_name["instrumental"] = (mix.cpu() - stems[model_spec.stems.index("vocals")]).unsqueeze(0)
        return by_name

    # ── Demucs ───────────────────────────────────────────────────────────
    @classmethod
    def _run_demucs(cls, model_name, waveform, target_device, shifts, overlap, jobs):
        total_progress_steps = 100
        pbar = comfy.utils.ProgressBar(total_progress_steps)
        pbar.update_absolute(1, total=total_progress_steps)

        model = _demucs.load(model_name, folder_paths.models_dir)
        model.to(target_device)
        model.eval()
        pbar.update_absolute(12, total=total_progress_steps)

        work_waveform, original_channels = _demucs.prepare_waveform(waveform)
        pbar.update_absolute(20, total=total_progress_steps)

        ref = work_waveform.mean(0)
        wav_mean = ref.mean()
        wav_std = ref.std() + 1e-8
        normalized = ((work_waveform - wav_mean) / wav_std).to(target_device)

        logger.info("%s Processing (shifts=%s, overlap=%s)", LOG_PREFIX, shifts, overlap)
        stop_event, thread = _demucs.start_ui_progress(pbar, total_progress_steps, 25, 92)
        try:
            sources = _demucs.apply(
                model, normalized, shifts=shifts, overlap=overlap,
                jobs=jobs if jobs > 0 else 0, device=target_device,
            )
        finally:
            stop_event.set()
            thread.join(timeout=1.0)

        pbar.update_absolute(95, total=total_progress_steps)
        sources = sources.cpu()
        if sources.ndim != 4 or sources.shape[1] < 4:
            raise RuntimeError(f"[TS Music Stems] Unexpected Demucs output shape: {tuple(sources.shape)}.")
        if original_channels == 1:
            sources = sources.mean(dim=2, keepdim=True)

        drums_t, bass_t, other_t, vocals_t = (sources[:, i, :, :] for i in range(4))
        instrumental_t = drums_t + bass_t + other_t

        # The stats follow the INPUT waveform's device, while `sources` was pulled
        # to CPU above. AUDIO is CPU by convention, but an upstream node may hand
        # us a CUDA waveform — mixing devices here would raise.
        mean = wav_mean.to(sources.device)
        std = wav_std.to(sources.device)

        def restore(tensor):
            return tensor * std + mean

        pbar.update_absolute(total_progress_steps, total=total_progress_steps)
        return {
            "vocals": restore(vocals_t),
            "bass": restore(bass_t),
            "drums": restore(drums_t),
            "other": restore(other_t),
            "instrumental": restore(instrumental_t),
        }

    # ── точка входа ──────────────────────────────────────────────────────
    @classmethod
    def execute(cls, audio, model_name, device, shifts, overlap, jobs,
                precision="fp16") -> IO.NodeOutput:
        # Умолчание совпадает со схемой: прямой вызов обязан считать так же,
        # как показывает интерфейс. Отдельно гасим None — необязательный вход,
        # отсутствующий в старом графе, может прийти именно им, а не пропуском.
        precision = precision or "fp16"
        target_device = cls._resolve_device(device)
        waveform = cls._normalize_waveform_shape(audio["waveform"])
        waveform = cls._resample(waveform.clone(), audio["sample_rate"])

        if _catalog.is_roformer(model_name):
            stems = cls._run_roformer(model_name, waveform, target_device, overlap, precision)
        else:
            stems = cls._run_demucs(model_name, waveform, target_device, shifts, overlap, jobs)

        blocked = ExecutionBlocker(None)

        def out(name):
            tensor = stems.get(name)
            return cls._audio(tensor) if tensor is not None else blocked

        # ⚠️ «others» — это всё, что не вокал, не бас и не барабаны. У шестистемной
        # модели туда входят гитара и пианино: иначе смысл выхода тихо сузился
        # бы, и старый граф, где others шёл в микс, потерял бы инструменты.
        others = stems.get("other")
        if others is not None:
            for extra in ("guitar", "piano"):
                if extra in stems:
                    others = others + stems[extra]

        vocals = stems.get("vocals")
        logger.info("%s Done (%s). Output shape: %s", LOG_PREFIX, model_name,
                    tuple(vocals.shape) if vocals is not None else "-")

        slots = (
            ("vocal", stems.get("vocals")),
            ("bass", stems.get("bass")),
            ("drums", stems.get("drums")),
            ("others", others),
            ("instrumental", stems.get("instrumental")),
            ("guitar", stems.get("guitar")),
            ("piano", stems.get("piano")),
        )
        cls._report_blocked(model_name, slots)

        return IO.NodeOutput(
            out("vocals"),
            out("bass"),
            out("drums"),
            cls._audio(others) if others is not None else blocked,
            out("instrumental"),
            out("guitar"),
            out("piano"),
        )

    @staticmethod
    def _report_blocked(model_name, slots) -> None:
        """Сказать вслух, каких стемов у выбранной модели нет.

        ⚠️ Без этой строки происходящее выглядит поломкой: человек подключил
        семь выходов, получил два файла, статус промпта — ``success``, и ничего
        не объясняет разницу. Заблокированная ветка исполняется молча, поэтому
        объяснить должны мы.
        """
        missing = [name for name, tensor in slots if tensor is None]
        if not missing:
            return
        available = [name for name, tensor in slots if tensor is not None]
        logger.info(
            "%s '%s' produces %s. It has no %s, so those outputs are blocked: "
            "anything wired to them is skipped rather than fed silence, and a "
            "node that mixes a blocked output with a live one is skipped whole. "
            "Switch to bs_roformer_sw if you need them.",
            LOG_PREFIX, model_name, ", ".join(available), ", ".join(missing),
        )


# ⚠️ Кэш моделей — мутация словаря уровня модуля. V3 клонирует класс ноды и
# запирает его, поэтому присваивание атрибута классу упало бы (CLAUDE.md §5).
_model_cache: dict = {}


NODE_CLASS_MAPPINGS = {
    "TS_MusicStems": TS_MusicStems,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TS_MusicStems": "TS Music Stems",
}
