import os
import json
import glob
import gc
import torch
import folder_paths # Используем нативные пути ComfyUI
from comfy.model_patcher import ModelPatcher
import comfy.model_patcher
from safetensors.torch import save_file, load_file
from safetensors import safe_open
from collections import OrderedDict
from tqdm import tqdm

# ==========================
# Simple Converter (In-Memory)
# ==========================
class TS_ModelConverterNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "convert_to_fp8"
    CATEGORY = "conversion"

    def convert_to_fp8(self, model):
        try:
            # Логика для разных типов объектов модели в ComfyUI
            if hasattr(model, 'diffusion_model'):
                model.diffusion_model = model.diffusion_model.to(torch.float8_e4m3fn)
            elif isinstance(model, ModelPatcher):
                model.model = model.model.to(torch.float8_e4m3fn)
            else:
                model = model.to(torch.float8_e4m3fn)
            
            # Чистим кэш после конвертации в памяти
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            return (model,)
        except Exception as e:
            print(f"FP8 Conversion Error: {str(e)}")
            return (model,) 

# ==========================
# Advanced Converter (On-Disk)
# ==========================
class TS_ModelConverterAdvancedNode:
    """
    Convert large AI models to FP8 (e4m3fn / e5m2).
    Использует нативные пути ComfyUI для поиска моделей.
    """

    @classmethod
    def INPUT_TYPES(s):
        # 1. Получаем список чекпоинтов через API ComfyUI (гарантированно работает)
        checkpoints = folder_paths.get_filename_list("checkpoints")
        
        # 2. Получаем список diffusion models (UNETs)
        unets = folder_paths.get_filename_list("diffusion_models")
        
        # 3. Собираем всё вместе, фильтруем только safetensors для безопасности
        # (хотя safe_open может читать и другие, но для конвертации лучше safetensors)
        file_list = []
        
        for f in checkpoints:
            if f.endswith(".safetensors"):
                file_list.append(f"checkpoints | {f}")
                
        for f in unets:
            if f.endswith(".safetensors"):
                file_list.append(f"diffusion_models | {f}")

        # 4. Добавляем сканирование папки Output (как в оригинале)
        output_dir = folder_paths.get_output_directory()
        output_diff_dir = os.path.join(output_dir, "diffusion_models")
        if os.path.exists(output_diff_dir):
            for f in os.listdir(output_diff_dir):
                if f.endswith(".safetensors"):
                    file_list.append(f"output | {f}")

        if not file_list:
            file_list = ["No .safetensors models found"]

        return {
            "required": {
                "model_name": (sorted(file_list), ),
                "fp8_mode": (["e4m3fn", "e5m2"], {"default": "e5m2"}),
                "shard_subdir": ("STRING", {"multiline": False, "default": "fp8_shards"}),
                "final_filename": ("STRING", {"multiline": False, "default": "converted_model_fp8.safetensors"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("log",)
    FUNCTION = "convert_model"
    CATEGORY = "Model Conversion"

    def should_convert_to_fp8(self, tensor_name: str) -> bool:
        # 1. Базовая проверка: работаем только с весами (.weight)
        if not tensor_name.endswith(".weight"):
            return False

        # 2. Исключаем FP32 параметры
        if "scale_weight" in tensor_name: return False
        if "patch_embedding" in tensor_name: return False

        # 3. Исключаем FP16 параметры (нормы и модуляции)
        if "norm" in tensor_name: return False
        if "modulation" in tensor_name: return False

        # 4. Логика для Блоков (основная часть модели)
        if "blocks." in tensor_name:
            if "cross_attn" in tensor_name or "ffn" in tensor_name or "self_attn" in tensor_name:
                return True
            return False

        # 5. Внешние слои
        if "head.head.weight" in tensor_name: return True
        if "text_embedding" in tensor_name or "time_embedding" in tensor_name or "time_projection" in tensor_name:
            return True

        return False

    def convert_model(self, model_name, fp8_mode, shard_subdir, final_filename):
        logs = []
        device = "cuda" if torch.cuda.is_available() else "cpu"
        target_dtype = torch.float8_e4m3fn if fp8_mode == "e4m3fn" else torch.float8_e5m2
        
        # Получаем пути
        output_dir = folder_paths.get_output_directory()
        
        # Парсим выбранное имя из списка (тип | имя)
        if " | " in model_name:
            type_key, filename = model_name.split(" | ", 1)
        else:
            # Fallback
            logs.append("❌ Invalid model selection")
            return ("\n".join(logs),)

        # Ищем полный путь к файлу
        model_path = None
        
        if type_key == "checkpoints":
            model_path = folder_paths.get_full_path("checkpoints", filename)
        elif type_key == "diffusion_models":
            model_path = folder_paths.get_full_path("diffusion_models", filename)
        elif type_key == "output":
            model_path = os.path.join(output_dir, "diffusion_models", filename)

        if not model_path or not os.path.exists(model_path):
             logs.append(f"❌ ERROR: File not found: {model_path}")
             return ("\n".join(logs),)

        logs.append(f"--- START FP8 CONVERSION ---")
        logs.append(f"File: {model_path}")
        logs.append(f"Target: {fp8_mode}")

        # --- CASE 1: Single file ---
        if os.path.isfile(model_path):
            shard_state = OrderedDict()
            out_path = os.path.join(output_dir, final_filename)

            try:
                with safe_open(model_path, framework="pt", device="cpu") as f_in:
                    tensor_names = f_in.keys()
                    for tensor_name in tqdm(tensor_names, desc="Converting"):
                        tensor = f_in.get_tensor(tensor_name)
                        
                        if self.should_convert_to_fp8(tensor_name):
                            # Конвертация через GPU для скорости (если есть), потом на CPU
                            tensor = tensor.to(device).to(target_dtype).to("cpu")
                            logs.append(f"  [FP8] {tensor_name}")
                        else:
                            # Оставляем как есть (обычно FP16/BF16/FP32)
                            tensor = tensor.to("cpu")
                            logs.append(f"  [KEEP] {tensor_name}")
                        
                        shard_state[tensor_name] = tensor

                save_file(shard_state, out_path)
                logs.append(f"✔ Saved to: {out_path}")
                
            except Exception as e:
                logs.append(f"❌ Conversion failed: {e}")
                
            # Чистка памяти
            del shard_state
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            return ("\n".join(logs),)

        return ("Folder conversion not fully supported in this simplified mode yet.",)

# ==========================
# Model Scanner
# ==========================
class ModelScanner:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "summary_only": ("BOOLEAN", {"default": False, "label_on": "Summary Only", "label_off": "Full Detail"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("model_info",)
    FUNCTION = "scan_model"
    CATEGORY = "utils/model_analysis"

    def scan_model(self, model, summary_only=False):
        real_model = None
        if isinstance(model, comfy.model_patcher.ModelPatcher):
            real_model = model.model
        else:
            real_model = model

        output_lines = []
        stats = {}
        total_params = 0

        output_lines.append("=== MODEL SCAN REPORT ===")
        output_lines.append(f"Type: {type(real_model).__name__}")
        output_lines.append("-" * 60)

        try:
            iterator = real_model.named_parameters()
            if hasattr(real_model, "diffusion_model"):
                 output_lines.append("Note: Scanning internal diffusion_model")
                 iterator = real_model.diffusion_model.named_parameters()

            for name, param in iterator:
                shape_str = str(tuple(param.shape))
                dtype_str = str(param.dtype).replace("torch.", "")
                device_str = str(param.device).split(":")[0]
                num_params = param.numel()

                total_params += num_params
                if dtype_str not in stats:
                    stats[dtype_str] = 0
                stats[dtype_str] += num_params

                if not summary_only:
                    output_lines.append(f"{name:<50} | {shape_str:<20} | {dtype_str:<10} | {device_str:<6}")

        except Exception as e:
            return (f"Error scanning model: {str(e)}",)

        output_lines.append("-" * 60)
        output_lines.append("=== SUMMARY STATISTICS ===")
        output_lines.append(f"Total Params: {total_params:,}")
        for dtype, count in stats.items():
            percent = (count / total_params) * 100 if total_params > 0 else 0
            output_lines.append(f" - {dtype}: {count:,} ({percent:.2f}%)")

        return ("\n".join(output_lines),)

# ==========================
# Registration
# ==========================
NODE_CLASS_MAPPINGS = {
    "TS_ModelConverter": TS_ModelConverterNode,
    "TS_ModelConverterAdvanced": TS_ModelConverterAdvancedNode,
    "ModelScanner": ModelScanner
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TS_ModelConverter": "TS Model Converter",
    "TS_ModelConverterAdvanced": "TS Model Converter Advanced",
    "ModelScanner": "🔍 Model Layer Scanner"
}