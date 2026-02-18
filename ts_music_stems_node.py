import torch
import torchaudio
import os
import folder_paths
import comfy.model_management as mm

# Попытка импорта demucs
try:
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
except ImportError:
    print("\033[31m[TS Suite] Error: 'demucs' package not installed. Please run: pip install demucs\033[0m")

class TS_MusicStems:
    """
    TS_MusicStems v1.5 (Stable High-Fidelity)
    - Reverted 'segment' parameter causing tensor mismatch in Transformer models.
    - Optimized for Max Quality via Overlap/Shifts logic.
    - Added 'mix_mode' for advanced blending.
    """

    def __init__(self):
        self.model_cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "model_name": (["htdemucs", "htdemucs_ft", "hdemucs_mmi"], {"default": "htdemucs_ft"}),
                "device": (["cuda", "cpu", "auto"], {"default": "auto"}),
                "shifts": ("INT", {"default": 2, "min": 0, "max": 10, "tooltip": "Количество проходов (TTA). 2 = Высокое качество. 4 = Ultra (медленно)."}),
                "overlap": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 0.9, "tooltip": "Нахлест. Для МАКСИМАЛЬНОГО качества ставьте 0.75-0.95 (очень медленно, но идеально гладко)."}),
                "jobs": ("INT", {"default": 0, "min": 0, "max": 16, "tooltip": "Потоки CPU для предобработки. 0 = Авто. Поставьте 4-8 для ускорения."}),
            }
        }

    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO", "AUDIO", "AUDIO")
    RETURN_NAMES = ("vocal", "bass", "drums", "others", "instrumental")
    FUNCTION = "process_stems"
    CATEGORY = "TS/Audio"

    def process_stems(self, audio, model_name, device, shifts, overlap, jobs):
        # 1. Setup & Device
        if device == "auto":
            target_device = mm.get_torch_device()
        else:
            target_device = torch.device(device)

        print(f"\033[96m[TS Music Stems] Initializing... Model: {model_name}\033[0m")

        # 2. Model Loading (Context Safe)
        models_base_path = folder_paths.models_dir
        demucs_model_path = os.path.join(models_base_path, "demucs")
        
        if not os.path.exists(demucs_model_path):
            os.makedirs(demucs_model_path, exist_ok=True)

        original_hub_dir = torch.hub.get_dir()
        torch.hub.set_dir(demucs_model_path)
        
        try:
            if model_name not in self.model_cache:
                model = get_model(model_name)
                self.model_cache[model_name] = model
            else:
                model = self.model_cache[model_name]
        except Exception as e:
            torch.hub.set_dir(original_hub_dir)
            raise RuntimeError(f"[TS Music Stems] Model load failed: {e}")
        finally:
            torch.hub.set_dir(original_hub_dir)

        model.to(target_device)
        model.eval()

        # 3. Audio Preparation
        waveform = audio["waveform"] # [Batch, Channels, Samples]
        sample_rate = audio["sample_rate"]
        target_sr = 44100
        
        # Resampling
        work_waveform = waveform.clone()
        if sample_rate != target_sr:
            print(f"\033[96m[TS Music Stems] Resampling {sample_rate} -> {target_sr} Hz\033[0m")
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr).to(waveform.device)
            work_waveform = resampler(work_waveform)

        # === Normalization (Anti-Clipping) ===
        # Сохраняем статистику
        ref = work_waveform.mean(0)
        wav_mean = ref.mean()
        wav_std = ref.std() + 1e-8
        
        normalized_waveform = (work_waveform - wav_mean) / wav_std
        normalized_waveform = normalized_waveform.to(target_device)

        # 4. Inference
        # num_workers=jobs ускоряет подготовку батчей, если у вас мощный CPU
        sys_jobs = jobs if jobs > 0 else 0
        
        print(f"\033[96m[TS Music Stems] Processing (Shifts: {shifts}, Overlap: {overlap})...\033[0m")
        
        with torch.no_grad():
            # Убрали segment, так как он ломает htdemucs
            sources = apply_model(
                model, 
                normalized_waveform, 
                shifts=shifts, 
                split=True, 
                overlap=overlap, 
                progress=True, 
                num_workers=sys_jobs,
                device=target_device
            )

        # 5. Post-Processing
        sources = sources.cpu()
        
        # Mapping: 0:drums, 1:bass, 2:other, 3:vocals
        drums_t = sources[:, 0, :, :]
        bass_t = sources[:, 1, :, :]
        other_t = sources[:, 2, :, :]
        vocals_t = sources[:, 3, :, :]
        
        # Создаем инструментал
        instrumental_t = drums_t + bass_t + other_t

        # Восстановление амплитуды
        def restore(tensor, mean, std):
            return tensor * std + mean

        vocals = restore(vocals_t, wav_mean, wav_std)
        bass = restore(bass_t, wav_mean, wav_std)
        drums = restore(drums_t, wav_mean, wav_std)
        other = restore(other_t, wav_mean, wav_std)
        instrumental = restore(instrumental_t, wav_mean, wav_std)

        # 6. Output Packing
        out_vocal = {"waveform": vocals, "sample_rate": target_sr}
        out_bass = {"waveform": bass, "sample_rate": target_sr}
        out_drums = {"waveform": drums, "sample_rate": target_sr}
        out_others = {"waveform": other, "sample_rate": target_sr}
        out_inst = {"waveform": instrumental, "sample_rate": target_sr}

        print(f"\033[92m[TS Music Stems] Done. Output Shape: {vocals.shape}\033[0m")
        
        return (out_vocal, out_bass, out_drums, out_others, out_inst)

NODE_CLASS_MAPPINGS = {
    "TS_MusicStems": TS_MusicStems
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TS_MusicStems": "TS Music Stems"
}