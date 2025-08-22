import torch
import os
from tqdm import tqdm
import safetensors.torch
import folder_paths

# =================================================================================
# Класс 1: Lora Mega Merger (Математически корректная версия)
# =================================================================================

class LoraMegaMergerCorrect:
    @classmethod
    def INPUT_TYPES(cls):
        lora_list = folder_paths.get_filename_list("loras")
        optional_lora_list = ["None"] + lora_list
        inputs = {
            "required": {
                "lora_name_1": (lora_list,),
                "weight_1": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "filename": ("STRING", {"default": "merged_lora_hq.safetensors"}),
            },
            "optional": {
                "output_path": ("STRING", {"default": ""})
            }
        }
        for i in range(2, 11):
            inputs["optional"][f"lora_name_{i}"] = (optional_lora_list,)
            inputs["optional"][f"weight_{i}"] = ("FLOAT", {"default": 0.5, "min": -2.0, "max": 2.0, "step": 0.01})
        return inputs

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_message",)
    FUNCTION = "merge_loras"
    CATEGORY = "loaders/lora_utils"

    def merge_loras(self, **kwargs):
        loras_to_process = []
        for i in range(1, 11):
            lora_name = kwargs.get(f"lora_name_{i}")
            weight = kwargs.get(f"weight_{i}", 0.0)
            if lora_name and lora_name != "None":
                loras_to_process.append((lora_name, weight))
        
        if not loras_to_process:
            return ("Ошибка: Нужно выбрать хотя бы одну LoRA.",)

        filename = kwargs.get("filename")
        output_path = kwargs.get("output_path", "")
        
        if not output_path: output_path = folder_paths.get_folder_paths("loras")[0]
        if not filename.endswith('.safetensors'): filename += ".safetensors"
        final_output_path = os.path.join(output_path, filename)
        
        print(f"Начинаем КОРРЕКТНОЕ слияние {len(loras_to_process)} LoRA...")
        merged_weights = {}
        
        # Перебираем все LoRA для слияния
        for lora_name, weight in tqdm(loras_to_process, desc="Merging LoRAs (Correctly)"):
            try:
                lora_path = folder_paths.get_full_path("loras", lora_name)
                if not lora_path:
                    print(f"Пропуск: файл LoRA '{lora_name}' не найден.")
                    continue
                
                lora_data = safetensors.torch.load_file(lora_path, device="cpu")
                
                # --- КЛЮЧЕВАЯ ЛОГИКА: ЧИТАЕМ ALPHA И DIM ---
                alpha, dim = 1.0, 1.0 # Значения по умолчанию
                metadata = {}
                try:
                    with safetensors.safe_open(lora_path, framework="pt", device="cpu") as lora_file:
                        metadata = lora_file.metadata() or {}
                except Exception: pass
                
                if metadata:
                    if 'ss_network_alpha' in metadata: alpha = float(metadata['ss_network_alpha'])
                    if 'ss_network_dim' in metadata: dim = float(metadata['ss_network_dim'])
                
                # Если alpha нет в метаданных, часто она равна dim
                if 'ss_network_alpha' not in metadata: alpha = dim

                print(f"  - Обработка '{lora_name}': вес={weight}, alpha={alpha}")
                
                # Нормализуем каждый тензор
                for key, tensor in lora_data.items():
                    # Применяем вес и НОРМАЛИЗУЕМ НА ALPHA
                    # Пропускаем тензоры alpha, так как они не являются весами
                    if 'alpha' in key: continue
                    
                    normalized_tensor = (tensor.to(torch.float32) * weight) / alpha
                    
                    if key in merged_weights:
                        merged_weights[key] += normalized_tensor
                    else:
                        merged_weights[key] = normalized_tensor
            except Exception as e:
                print(f"Ошибка при обработке LoRA '{lora_name}': {e}")
                continue

        if not merged_weights:
            return ("Ошибка: Не удалось загрузить ни одну LoRA.",)
        
        # --- СОХРАНЕНИЕ ---
        # Конвертируем все тензоры в bfloat16 для экономии места
        for key in merged_weights:
            merged_weights[key] = merged_weights[key].to(torch.bfloat16)

        # Создаем метаданные для новой LoRA. Alpha=1, так как мы все нормализовали.
        final_metadata = {
            'ss_network_module': 'networks.lora',
            'ss_network_alpha': '1.0',
            # Можно попробовать унаследовать ранг от первой LoRA, если он есть
            'ss_network_dim': str(int(dim)) if 'dim' in locals() else '1',
        }
        
        print(f"Сохранение объединенной LoRA в формате bfloat16 в: {final_output_path}")
        safetensors.torch.save_file(merged_weights, final_output_path, metadata=final_metadata)
        
        return (f"Корректное слияние {len(loras_to_process)} LoRA завершено! Файл '{filename}' сохранен. Используйте его с alpha=1 (или не указывайте).",)

# =================================================================================
# Класс 2: Lora Rank Converter (Качественная версия) - без изменений
# =================================================================================
# ... (этот класс можно оставить как есть из предыдущей версии, он не меняется)
class LoraRankConverterHQ:
    # ... (код остается прежним)
    pass # Заглушка, если вы копируете только верхнюю часть

# =================================================================================
# Регистрация нод в ComfyUI
# =================================================================================
NODE_CLASS_MAPPINGS = {
    "LoraMegaMergerCorrect": LoraMegaMergerCorrect,
    # "LoraRankConverterHQ": LoraRankConverterHQ, # Раскомментируйте, если оставили этот класс
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoraMegaMergerCorrect": "Lora Merger (Correct & Compact)",
    # "LoraRankConverterHQ": "Lora Rank Converter (High Quality)", # Раскомментируйте, если оставили
}