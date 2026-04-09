import os

import comfy.model_management as mm
import comfy.utils
import folder_paths
import threading
import torch
import torchaudio

from ..ts_dependency_manager import TSDependencyManager

_demucs_pretrained = TSDependencyManager.import_optional("demucs.pretrained")
_demucs_apply = TSDependencyManager.import_optional("demucs.apply")
_demucs_get_model = getattr(_demucs_pretrained, "get_model", None) if _demucs_pretrained is not None else None
_demucs_apply_model = getattr(_demucs_apply, "apply_model", None) if _demucs_apply is not None else None


class TS_MusicStems:
    """
    TS_MusicStems v1.5 (Stable High-Fidelity)
    - Reverted `segment` parameter causing tensor mismatch in transformer models.
    - Optimized for quality via overlap/shifts logic.
    """

    def __init__(self):
        self.model_cache = {}

    @staticmethod
    def _normalize_waveform_shape(waveform):
        if not torch.is_tensor(waveform):
            raise TypeError("[TS Music Stems] Audio waveform must be a torch.Tensor.")
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)
        if waveform.ndim != 3:
            raise ValueError(
                f"[TS Music Stems] Expected waveform shape [batch, channels, samples], got {tuple(waveform.shape)}."
            )
        if waveform.shape[0] < 1 or waveform.shape[1] < 1 or waveform.shape[2] < 1:
            raise ValueError(f"[TS Music Stems] Waveform has invalid shape: {tuple(waveform.shape)}.")
        return waveform

    @staticmethod
    def _prepare_demucs_waveform(waveform):
        original_channels = int(waveform.shape[1])
        if original_channels == 1:
            print("[TS Music Stems] Mono input detected, duplicating channel to stereo for Demucs compatibility.")
            return waveform.repeat(1, 2, 1), original_channels
        if original_channels == 2:
            return waveform, original_channels

        print(
            f"[TS Music Stems] Input has {original_channels} channels, using the first two channels for Demucs."
        )
        return waveform[:, :2, :], original_channels

    @staticmethod
    def _start_ui_progress(total_steps, warmup_step, processing_start_step, processing_cap_step):
        pbar = comfy.utils.ProgressBar(total_steps)
        pbar.update_absolute(warmup_step, total=total_steps)

        stop_event = threading.Event()

        def pulse():
            current_value = processing_start_step
            while not stop_event.wait(0.5):
                if current_value >= processing_cap_step:
                    continue
                current_value += 1
                pbar.update_absolute(current_value, total=total_steps)

        worker = threading.Thread(target=pulse, name="ts-music-stems-progress", daemon=True)
        worker.start()
        return pbar, stop_event, worker

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "model_name": (["htdemucs", "htdemucs_ft", "hdemucs_mmi"], {"default": "htdemucs_ft"}),
                "device": (["cuda", "cpu", "auto"], {"default": "auto"}),
                "shifts": (
                    "INT",
                    {"default": 2, "min": 0, "max": 10, "tooltip": "TTA passes. 2 = high quality, 4 = very slow."},
                ),
                "overlap": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 0.9, "tooltip": "Chunk overlap for smoother stitching."},
                ),
                "jobs": (
                    "INT",
                    {"default": 0, "min": 0, "max": 16, "tooltip": "CPU workers for pre-processing. 0 = auto."},
                ),
            }
        }

    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO", "AUDIO", "AUDIO")
    RETURN_NAMES = ("vocal", "bass", "drums", "others", "instrumental")
    FUNCTION = "process_stems"
    CATEGORY = "TS/Audio"

    def process_stems(self, audio, model_name, device, shifts, overlap, jobs):
        total_progress_steps = 100
        pbar = comfy.utils.ProgressBar(total_progress_steps)
        pbar.update_absolute(1, total=total_progress_steps)

        if _demucs_get_model is None or _demucs_apply_model is None:
            raise RuntimeError(
                "[TS Music Stems] Missing dependency 'demucs'. Install it to enable stem separation."
            )

        if device == "auto":
            target_device = mm.get_torch_device()
        else:
            target_device = torch.device(device)

        print(f"[TS Music Stems] Initializing model: {model_name}")

        models_base_path = folder_paths.models_dir
        demucs_model_path = os.path.join(models_base_path, "demucs")
        os.makedirs(demucs_model_path, exist_ok=True)

        original_hub_dir = torch.hub.get_dir()
        torch.hub.set_dir(demucs_model_path)
        try:
            if model_name not in self.model_cache:
                model = _demucs_get_model(model_name)
                self.model_cache[model_name] = model
                submodels = getattr(model, "models", None)
                if submodels is not None and len(submodels) > 1:
                    print(
                        f"[TS Music Stems] Model '{model_name}' is a Demucs bag model; "
                        f"the first run downloads {len(submodels)} checkpoints once."
                    )
            else:
                model = self.model_cache[model_name]
        except Exception as exc:
            raise RuntimeError(f"[TS Music Stems] Model load failed: {exc}") from exc
        finally:
            torch.hub.set_dir(original_hub_dir)

        model.to(target_device)
        model.eval()
        pbar.update_absolute(12, total=total_progress_steps)

        waveform = self._normalize_waveform_shape(audio["waveform"])  # [batch, channels, samples]
        sample_rate = audio["sample_rate"]
        target_sr = 44100

        work_waveform = waveform.clone()
        if sample_rate != target_sr:
            print(f"[TS Music Stems] Resampling {sample_rate} -> {target_sr} Hz")
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr).to(waveform.device)
            work_waveform = resampler(work_waveform)

        work_waveform, original_channels = self._prepare_demucs_waveform(work_waveform)
        pbar.update_absolute(20, total=total_progress_steps)

        ref = work_waveform.mean(0)
        wav_mean = ref.mean()
        wav_std = ref.std() + 1e-8
        normalized_waveform = ((work_waveform - wav_mean) / wav_std).to(target_device)

        sys_jobs = jobs if jobs > 0 else 0
        print(f"[TS Music Stems] Processing (shifts={shifts}, overlap={overlap})")
        pbar, progress_stop_event, progress_thread = self._start_ui_progress(
            total_steps=total_progress_steps,
            warmup_step=25,
            processing_start_step=25,
            processing_cap_step=92,
        )
        with torch.no_grad():
            try:
                sources = _demucs_apply_model(
                    model,
                    normalized_waveform,
                    shifts=shifts,
                    split=True,
                    overlap=overlap,
                    progress=True,
                    num_workers=sys_jobs,
                    device=target_device,
                )
            finally:
                progress_stop_event.set()
                progress_thread.join(timeout=1.0)

        pbar.update_absolute(95, total=total_progress_steps)

        sources = sources.cpu()
        if sources.ndim != 4 or sources.shape[1] < 4:
            raise RuntimeError(f"[TS Music Stems] Unexpected Demucs output shape: {tuple(sources.shape)}.")
        if original_channels == 1:
            sources = sources.mean(dim=2, keepdim=True)

        drums_t = sources[:, 0, :, :]
        bass_t = sources[:, 1, :, :]
        other_t = sources[:, 2, :, :]
        vocals_t = sources[:, 3, :, :]
        instrumental_t = drums_t + bass_t + other_t

        def restore(tensor, mean, std):
            return tensor * std + mean

        vocals = restore(vocals_t, wav_mean, wav_std)
        bass = restore(bass_t, wav_mean, wav_std)
        drums = restore(drums_t, wav_mean, wav_std)
        other = restore(other_t, wav_mean, wav_std)
        instrumental = restore(instrumental_t, wav_mean, wav_std)

        out_vocal = {"waveform": vocals, "sample_rate": target_sr}
        out_bass = {"waveform": bass, "sample_rate": target_sr}
        out_drums = {"waveform": drums, "sample_rate": target_sr}
        out_others = {"waveform": other, "sample_rate": target_sr}
        out_inst = {"waveform": instrumental, "sample_rate": target_sr}

        pbar.update_absolute(total_progress_steps, total=total_progress_steps)
        print(f"[TS Music Stems] Done. Output shape: {vocals.shape}")
        return (out_vocal, out_bass, out_drums, out_others, out_inst)


NODE_CLASS_MAPPINGS = {
    "TS_MusicStems": TS_MusicStems,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TS_MusicStems": "TS Music Stems",
}

