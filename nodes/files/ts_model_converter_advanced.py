"""TS Model Converter Advanced — disk-based safetensors converter with many flags.

node_id: TS_ModelConverterAdvanced
"""

import os
import json
import glob
import gc
import uuid
from collections import OrderedDict

import torch
from tqdm import tqdm

import folder_paths
from comfy.model_patcher import ModelPatcher
import comfy.model_patcher
import comfy.sd
from safetensors.torch import save_file, load_file
from safetensors import safe_open


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
                "conversion_preset": (["WAN", "Flux2"], {"default": "WAN"}),
                "shard_subdir": ("STRING", {"multiline": False, "default": "fp8_shards"}),
                "final_filename": ("STRING", {"multiline": False, "default": "converted_model_fp8.safetensors"}),
            },
            "optional": {
                "model": ("MODEL",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("log",)
    FUNCTION = "convert_model"
    CATEGORY = "Model Conversion"

    def should_convert_to_fp8(self, tensor_name: str, conversion_preset: str = "WAN") -> bool:
        tensor_name = self._normalize_tensor_name(tensor_name)
        if conversion_preset == "Flux2":
            return self._should_convert_to_fp8_flux2(tensor_name)

        # Default: WAN preset (legacy behavior)
        if "patch_embedding" in tensor_name:
            return False
        if "scale_weight" in tensor_name:
            return False
        return True

    def _normalize_tensor_name(self, tensor_name: str) -> str:
        name = tensor_name
        while name.startswith("model."):
            name = name[len("model."):]
        if name.startswith("diffusion_model."):
            name = name[len("diffusion_model."):]
        return name

    def _extract_block_index(self, tensor_name: str, prefix: str):
        if not tensor_name.startswith(prefix):
            return None
        rest = tensor_name[len(prefix):]
        digits = []
        for ch in rest:
            if ch.isdigit():
                digits.append(ch)
            else:
                break
        if not digits:
            return None
        try:
            return int("".join(digits))
        except Exception:
            return None

    def _should_convert_to_fp8_flux2(self, tensor_name: str) -> bool:
        if "patch_embedding" in tensor_name:
            return False
        if "scale_weight" in tensor_name:
            return False
        if not tensor_name.endswith(".weight"):
            return False

        if tensor_name.startswith("double_blocks."):
            block_idx = self._extract_block_index(tensor_name, "double_blocks.")
            if block_idx is None:
                return False

            if ".img_attn.qkv.weight" in tensor_name:
                return True
            if ".img_attn.proj.weight" in tensor_name:
                return True
            if ".txt_attn.qkv.weight" in tensor_name:
                return True
            if ".txt_attn.proj.weight" in tensor_name:
                return True
            if ".img_mlp.0.weight" in tensor_name or ".img_mlp.2.weight" in tensor_name:
                return True
            if ".txt_mlp.0.weight" in tensor_name or ".txt_mlp.2.weight" in tensor_name:
                return True
            return False

        if tensor_name.startswith("single_blocks."):
            if self._extract_block_index(tensor_name, "single_blocks.") is None:
                return False
            if ".linear1.weight" in tensor_name or ".linear2.weight" in tensor_name:
                return True
            return False

        return False

    def _convert_tensor_to_fp8(self, tensor, tensor_name, target_dtype, device, logs, conversion_preset="WAN"):
        if not tensor.is_floating_point():
            return tensor.to("cpu"), False
        if not self.should_convert_to_fp8(tensor_name, conversion_preset):
            return tensor.to("cpu"), False

        if device == "cuda":
            try:
                tensor = tensor.to(device, non_blocking=True)
                tensor = tensor.to(target_dtype)
                return tensor.to("cpu"), True
            except Exception as e:
                logs.append(f"  [WARN] {tensor_name} FP8 GPU convert failed: {e}")

        try:
            tensor = tensor.to("cpu")
            tensor = tensor.to(target_dtype)
            return tensor, True
        except Exception as e:
            logs.append(f"  [WARN] {tensor_name} FP8 CPU convert failed: {e}")
            return tensor.to("cpu"), False

    def _convert_loaded_model(self, model, fp8_mode, conversion_preset, shard_subdir, final_filename):
        logs = []
        device = "cuda" if torch.cuda.is_available() else "cpu"
        target_dtype = torch.float8_e4m3fn if fp8_mode == "e4m3fn" else torch.float8_e5m2

        output_dir = folder_paths.get_output_directory()
        out_path = os.path.join(output_dir, final_filename)

        logs.append("--- START FP8 CONVERSION (DIRECT) ---")
        logs.append(f"Target: {fp8_mode}")
        logs.append(f"Preset: {conversion_preset}")

        temp_filename = f"ts_tmp_export_{uuid.uuid4().hex}.safetensors"
        temp_path = os.path.join(output_dir, temp_filename)

        shard_state = OrderedDict()
        try:
            comfy.sd.save_checkpoint(temp_path, model, clip=None, vae=None, clip_vision=None, metadata={}, extra_keys={})
            logs.append(f"Temp saved: {temp_path}")

            with safe_open(temp_path, framework="pt", device="cpu") as f_in:
                tensor_names = f_in.keys()
                for tensor_name in tqdm(tensor_names, desc="Converting"):
                    tensor = f_in.get_tensor(tensor_name)

                    tensor, converted = self._convert_tensor_to_fp8(
                        tensor, tensor_name, target_dtype, device, logs, conversion_preset=conversion_preset
                    )
                    if converted:
                        logs.append(f"  [FP8] {tensor_name}")
                    else:
                        logs.append(f"  [KEEP] {tensor_name}")

                    shard_state[tensor_name] = tensor

            save_file(shard_state, out_path)
            logs.append(f"OK Saved to: {out_path}")
        except Exception as e:
            logs.append(f"Conversion failed: {e}")
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

        del shard_state
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return ("\n".join(logs),)

    def convert_model(self, model_name, fp8_mode, conversion_preset, shard_subdir, final_filename, model=None):
        if model is not None:
            return self._convert_loaded_model(model, fp8_mode, conversion_preset, shard_subdir, final_filename)

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
            logs.append("вќЊ Invalid model selection")
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
             logs.append(f"вќЊ ERROR: File not found: {model_path}")
             return ("\n".join(logs),)

        logs.append(f"--- START FP8 CONVERSION ---")
        logs.append(f"File: {model_path}")
        logs.append(f"Target: {fp8_mode}")
        logs.append(f"Preset: {conversion_preset}")

        # --- CASE 1: Single file ---
        if os.path.isfile(model_path):
            shard_state = OrderedDict()
            out_path = os.path.join(output_dir, final_filename)

            try:
                with safe_open(model_path, framework="pt", device="cpu") as f_in:
                    tensor_names = f_in.keys()
                    for tensor_name in tqdm(tensor_names, desc="Converting"):
                        tensor = f_in.get_tensor(tensor_name)

                        tensor, converted = self._convert_tensor_to_fp8(
                            tensor, tensor_name, target_dtype, device, logs, conversion_preset=conversion_preset
                        )
                        if converted:
                            logs.append(f"  [FP8] {tensor_name}")
                        else:
                            logs.append(f"  [KEEP] {tensor_name}")

                        shard_state[tensor_name] = tensor

                save_file(shard_state, out_path)
                logs.append(f"вњ” Saved to: {out_path}")
                
            except Exception as e:
                logs.append(f"вќЊ Conversion failed: {e}")
                
            # Чистка памяти
            del shard_state
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            return ("\n".join(logs),)

        return ("Folder conversion not fully supported in this simplified mode yet.",)

# ==========================
# Advanced Converter Direct (Model Input)
# ==========================


NODE_CLASS_MAPPINGS = {"TS_ModelConverterAdvanced": TS_ModelConverterAdvancedNode}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_ModelConverterAdvanced": "TS Model Converter Advanced"}
