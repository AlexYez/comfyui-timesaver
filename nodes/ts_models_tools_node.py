import os
import json
import glob
import gc
import torch
import uuid
import folder_paths # РСЃРїРѕР»СЊР·СѓРµРј РЅР°С‚РёРІРЅС‹Рµ РїСѓС‚Рё ComfyUI
from comfy.model_patcher import ModelPatcher
import comfy.model_patcher
import comfy.sd
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
            # Р›РѕРіРёРєР° РґР»СЏ СЂР°Р·РЅС‹С… С‚РёРїРѕРІ РѕР±СЉРµРєС‚РѕРІ РјРѕРґРµР»Рё РІ ComfyUI
            if hasattr(model, 'diffusion_model'):
                model.diffusion_model = model.diffusion_model.to(torch.float8_e4m3fn)
            elif isinstance(model, ModelPatcher):
                model.model = model.model.to(torch.float8_e4m3fn)
            else:
                model = model.to(torch.float8_e4m3fn)
            
            # Р§РёСЃС‚РёРј РєСЌС€ РїРѕСЃР»Рµ РєРѕРЅРІРµСЂС‚Р°С†РёРё РІ РїР°РјСЏС‚Рё
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
    РСЃРїРѕР»СЊР·СѓРµС‚ РЅР°С‚РёРІРЅС‹Рµ РїСѓС‚Рё ComfyUI РґР»СЏ РїРѕРёСЃРєР° РјРѕРґРµР»РµР№.
    """

    @classmethod
    def INPUT_TYPES(s):
        # 1. РџРѕР»СѓС‡Р°РµРј СЃРїРёСЃРѕРє С‡РµРєРїРѕРёРЅС‚РѕРІ С‡РµСЂРµР· API ComfyUI (РіР°СЂР°РЅС‚РёСЂРѕРІР°РЅРЅРѕ СЂР°Р±РѕС‚Р°РµС‚)
        checkpoints = folder_paths.get_filename_list("checkpoints")
        
        # 2. РџРѕР»СѓС‡Р°РµРј СЃРїРёСЃРѕРє diffusion models (UNETs)
        unets = folder_paths.get_filename_list("diffusion_models")
        
        # 3. РЎРѕР±РёСЂР°РµРј РІСЃС‘ РІРјРµСЃС‚Рµ, С„РёР»СЊС‚СЂСѓРµРј С‚РѕР»СЊРєРѕ safetensors РґР»СЏ Р±РµР·РѕРїР°СЃРЅРѕСЃС‚Рё
        # (С…РѕС‚СЏ safe_open РјРѕР¶РµС‚ С‡РёС‚Р°С‚СЊ Рё РґСЂСѓРіРёРµ, РЅРѕ РґР»СЏ РєРѕРЅРІРµСЂС‚Р°С†РёРё Р»СѓС‡С€Рµ safetensors)
        file_list = []
        
        for f in checkpoints:
            if f.endswith(".safetensors"):
                file_list.append(f"checkpoints | {f}")
                
        for f in unets:
            if f.endswith(".safetensors"):
                file_list.append(f"diffusion_models | {f}")

        # 4. Р”РѕР±Р°РІР»СЏРµРј СЃРєР°РЅРёСЂРѕРІР°РЅРёРµ РїР°РїРєРё Output (РєР°Рє РІ РѕСЂРёРіРёРЅР°Р»Рµ)
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
        
        # РџРѕР»СѓС‡Р°РµРј РїСѓС‚Рё
        output_dir = folder_paths.get_output_directory()
        
        # РџР°СЂСЃРёРј РІС‹Р±СЂР°РЅРЅРѕРµ РёРјСЏ РёР· СЃРїРёСЃРєР° (С‚РёРї | РёРјСЏ)
        if " | " in model_name:
            type_key, filename = model_name.split(" | ", 1)
        else:
            # Fallback
            logs.append("вќЊ Invalid model selection")
            return ("\n".join(logs),)

        # РС‰РµРј РїРѕР»РЅС‹Р№ РїСѓС‚СЊ Рє С„Р°Р№Р»Сѓ
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
                
            # Р§РёСЃС‚РєР° РїР°РјСЏС‚Рё
            del shard_state
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            return ("\n".join(logs),)

        return ("Folder conversion not fully supported in this simplified mode yet.",)

# ==========================
# Advanced Converter Direct (Model Input)
# ==========================
class TS_ModelConverterAdvancedDirectNode(TS_ModelConverterAdvancedNode):
    """
    Convert loaded MODEL to FP8 (e4m3fn / e5m2) and save to disk.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "fp8_mode": (["e4m3fn", "e5m2"], {"default": "e5m2"}),
                "conversion_preset": (["WAN", "Flux2"], {"default": "WAN"}),
                "shard_subdir": ("STRING", {"multiline": False, "default": "fp8_shards"}),
                "final_filename": ("STRING", {"multiline": False, "default": "converted_model_fp8.safetensors"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("log",)
    FUNCTION = "convert_model"
    CATEGORY = "TS/Model Conversion"

    def convert_model(self, model, fp8_mode, conversion_preset, shard_subdir, final_filename):
        return self._convert_loaded_model(model, fp8_mode, conversion_preset, shard_subdir, final_filename)

# ==========================
# Model Scanner
class TS_ModelScanner:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        diffusion_models = folder_paths.get_filename_list("diffusion_models")
        model_files = [f for f in diffusion_models if f.endswith(".safetensors")]
        if not model_files:
            model_files = ["No diffusion models found"]

        return {
            "required": {
                "model_name": (sorted(model_files),),
            },
            "optional": {
                "model": ("MODEL",),
                "summary_only": ("BOOLEAN", {"default": False, "label_on": "Summary Only", "label_off": "Full Detail"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("model_info",)
    FUNCTION = "scan_model"
    CATEGORY = "utils/model_analysis"

    def _scan_loaded_model(self, model, summary_only=False):
        real_model = None
        if isinstance(model, comfy.model_patcher.ModelPatcher):
            real_model = model.model
        else:
            real_model = model

        output_lines = []
        stats = {}
        total_params = 0

        output_lines.append("=== MODEL SCAN REPORT ===")
        output_lines.append("Source: loaded MODEL")
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

    def _scan_safetensors_file(self, model_path, summary_only=False):
        output_lines = []
        stats = {}
        total_params = 0

        output_lines.append("=== MODEL SCAN REPORT ===")
        output_lines.append("Source: diffusion_models (disk)")
        output_lines.append(f"File: {model_path}")
        output_lines.append("Note: Scanning safetensors file")
        output_lines.append("-" * 60)

        try:
            with safe_open(model_path, framework="pt", device="cpu") as f_in:
                for name in f_in.keys():
                    tensor = f_in.get_tensor(name)
                    shape_str = str(tuple(tensor.shape))
                    dtype_str = str(tensor.dtype).replace("torch.", "")
                    device_str = "cpu"
                    num_params = tensor.numel()

                    total_params += num_params
                    if dtype_str not in stats:
                        stats[dtype_str] = 0
                    stats[dtype_str] += num_params

                    if not summary_only:
                        output_lines.append(f"{name:<50} | {shape_str:<20} | {dtype_str:<10} | {device_str:<6}")

                    del tensor

        except Exception as e:
            return (f"Error scanning safetensors file: {str(e)}",)

        output_lines.append("-" * 60)
        output_lines.append("=== SUMMARY STATISTICS ===")
        output_lines.append(f"Total Params: {total_params:,}")
        for dtype, count in stats.items():
            percent = (count / total_params) * 100 if total_params > 0 else 0
            output_lines.append(f" - {dtype}: {count:,} ({percent:.2f}%)")

        return ("\n".join(output_lines),)

    def scan_model(self, model_name, model=None, summary_only=False):
        if model is not None:
            return self._scan_loaded_model(model, summary_only)

        if not model_name or model_name == "No diffusion models found":
            return ("Error: No diffusion_models available for scanning.",)

        model_path = folder_paths.get_full_path("diffusion_models", model_name)
        if not model_path or not os.path.exists(model_path):
            return (f"Error: File not found: {model_path}",)

        if not model_path.endswith(".safetensors"):
            return (f"Error: Unsupported format for scanning: {model_path}",)

        return self._scan_safetensors_file(model_path, summary_only)
# ==========================
# Registration
# ==========================
NODE_CLASS_MAPPINGS = {
    "TS_ModelConverter": TS_ModelConverterNode,
    "TS_ModelConverterAdvanced": TS_ModelConverterAdvancedNode,
    "TS_ModelConverterAdvancedDirect": TS_ModelConverterAdvancedDirectNode,
    "TS_ModelScanner": TS_ModelScanner
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TS_ModelConverter": "TS Model Converter",
    "TS_ModelConverterAdvanced": "TS Model Converter Advanced",
    "TS_ModelConverterAdvancedDirect": "TS Model Converter Advanced Direct",
    "TS_ModelScanner": "TS Model Layer Scanner"
}

import logging as _ts_logging
import re as _ts_re
import traceback as _ts_traceback
import comfy.utils as _ts_comfy_utils


class TS_CPULoraMergerNode:
    _SUPPORTED_MODEL_EXTENSIONS = (".safetensors", ".ckpt", ".pt", ".pth")
    _LORA_NONE = "None"
    _LOG_PREFIX = "[TS CPULoraMerger]"
    _logger = _ts_logging.getLogger("TS_CPULoraMerger")

    @classmethod
    def _build_model_choices(cls):
        model_choices = []
        for model_type in ("checkpoints", "diffusion_models"):
            for filename in folder_paths.get_filename_list(model_type):
                if filename.lower().endswith(cls._SUPPORTED_MODEL_EXTENSIONS):
                    model_choices.append(f"{model_type} | {filename}")

        if not model_choices:
            model_choices = ["No compatible models found"]
        return sorted(model_choices)

    @classmethod
    def _build_lora_choices(cls):
        loras = sorted(folder_paths.get_filename_list("loras"))
        return [cls._LORA_NONE] + loras

    @classmethod
    def INPUT_TYPES(cls):
        model_choices = cls._build_model_choices()
        lora_choices = cls._build_lora_choices()
        return {
            "required": {
                "base_model": (model_choices,),
                "lora_1_name": (lora_choices, {"default": cls._LORA_NONE}),
                "lora_1_strength": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "lora_2_name": (lora_choices, {"default": cls._LORA_NONE}),
                "lora_2_strength": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "lora_3_name": (lora_choices, {"default": cls._LORA_NONE}),
                "lora_3_strength": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "lora_4_name": (lora_choices, {"default": cls._LORA_NONE}),
                "lora_4_strength": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "output_model_name": ("STRING", {"multiline": False, "default": "ts_merged_model.safetensors"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("log", "saved_model_path")
    FUNCTION = "merge_to_file"
    CATEGORY = "TS/Model Tools"
    OUTPUT_NODE = True

    def _log(self, logs, message):
        msg = f"{self._LOG_PREFIX} {message}"
        logs.append(msg)
        self._logger.info(msg)

    def _resolve_model_path(self, selected_model):
        if " | " not in selected_model:
            raise ValueError(f"Invalid model selector: {selected_model}")

        model_type, model_name = selected_model.split(" | ", 1)
        model_path = folder_paths.get_full_path(model_type, model_name)
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_name}")
        return model_type, model_path

    def _sanitize_output_filename(self, output_model_name):
        filename = (output_model_name or "").strip()
        if not filename:
            filename = "ts_merged_model"
        filename = filename.replace("\\", "_").replace("/", "_")
        filename = _ts_re.sub(r"[^A-Za-z0-9._-]+", "_", filename)
        if not filename.lower().endswith(".safetensors"):
            filename = f"{filename}.safetensors"
        return filename

    def _unique_path(self, target_path):
        if not os.path.exists(target_path):
            return target_path

        base, ext = os.path.splitext(target_path)
        index = 1
        while True:
            candidate = f"{base}_{index:03d}{ext}"
            if not os.path.exists(candidate):
                return candidate
            index += 1

    def _prepare_output_path(self, model_type, output_model_name):
        if model_type == "checkpoints":
            target_dirs = folder_paths.get_folder_paths("checkpoints")
        else:
            target_dirs = folder_paths.get_folder_paths("diffusion_models")

        if target_dirs:
            target_dir = target_dirs[0]
        else:
            target_dir = os.path.join(folder_paths.get_output_directory(), "diffusion_models")

        os.makedirs(target_dir, exist_ok=True)
        filename = self._sanitize_output_filename(output_model_name)
        return self._unique_path(os.path.join(target_dir, filename))

    def _collect_lora_requests(self, lora_names, lora_strengths):
        requests = []
        for index, (name, strength) in enumerate(zip(lora_names, lora_strengths), start=1):
            if not name or name == self._LORA_NONE:
                continue
            strength = float(strength)
            if strength == 0.0:
                continue
            requests.append({
                "slot": index,
                "name": name,
                "strength": strength,
            })
        return requests

    def _load_base_assets(self, model_type, model_path):
        if model_type == "checkpoints":
            text_encoder_options = {
                "load_device": torch.device("cpu"),
                "offload_device": torch.device("cpu"),
                "initial_device": torch.device("cpu"),
            }
            return comfy.sd.load_checkpoint_guess_config(
                model_path,
                output_vae=True,
                output_clip=True,
                output_clipvision=False,
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
                te_model_options=text_encoder_options,
            )

        model = comfy.sd.load_diffusion_model(model_path)
        return model, None, None, None

    def _apply_loras(self, model, clip, lora_requests, logs):
        for request in lora_requests:
            lora_path = folder_paths.get_full_path("loras", request["name"])
            if not lora_path or not os.path.exists(lora_path):
                raise FileNotFoundError(f"LoRA file not found: {request['name']}")

            self._log(
                logs,
                f"Loading LoRA #{request['slot']}: {request['name']} (strength={request['strength']:.4f})",
            )
            lora_state = _ts_comfy_utils.load_torch_file(lora_path, safe_load=True)
            clip_strength = request["strength"] if clip is not None else 0.0
            model, clip = comfy.sd.load_lora_for_models(
                model,
                clip,
                lora_state,
                request["strength"],
                clip_strength,
            )
            del lora_state
            gc.collect()
        return model, clip

    def _build_cpu_state_dict(self, model, clip=None, vae=None, clip_vision=None):
        cpu_device = torch.device("cpu")
        unet_sd = model.model.diffusion_model.state_dict()
        unet_sd_cpu = OrderedDict()

        for key, tensor in unet_sd.items():
            out_tensor = tensor
            if key.endswith(".weight") or key.endswith(".bias"):
                patched_key = f"diffusion_model.{key}"
                out_tensor = model.patch_weight_to_device(
                    patched_key,
                    device_to=cpu_device,
                    return_weight=True,
                )
            if not isinstance(out_tensor, torch.Tensor):
                raise TypeError(f"Unexpected non-tensor value in UNet state dict for key: {key}")
            if out_tensor.device.type != "cpu":
                out_tensor = out_tensor.to("cpu")
            if not out_tensor.is_contiguous():
                out_tensor = out_tensor.contiguous()
            unet_sd_cpu[key] = out_tensor

        clip_sd = None
        if clip is not None:
            clip_sd = OrderedDict()
            clip_model_sd = clip.cond_stage_model.state_dict()
            for key, tensor in clip_model_sd.items():
                out_tensor = tensor
                if key.endswith(".weight") or key.endswith(".bias"):
                    out_tensor = clip.patcher.patch_weight_to_device(
                        key,
                        device_to=cpu_device,
                        return_weight=True,
                    )
                if not isinstance(out_tensor, torch.Tensor):
                    raise TypeError(f"Unexpected non-tensor value in CLIP state dict for key: {key}")
                if out_tensor.device.type != "cpu":
                    out_tensor = out_tensor.to("cpu")
                if not out_tensor.is_contiguous():
                    out_tensor = out_tensor.contiguous()
                clip_sd[key] = out_tensor

            # Preserve tokenizer state like native CLIP save path.
            clip_sd.update(clip.tokenizer.state_dict())

        vae_sd = vae.get_sd() if vae is not None else None
        clip_vision_sd = clip_vision.get_sd() if clip_vision is not None else None

        raw_state_dict = model.model.state_dict_for_saving(
            unet_sd_cpu,
            clip_state_dict=clip_sd,
            vae_state_dict=vae_sd,
            clip_vision_state_dict=clip_vision_sd,
        )

        for key, tensor in list(raw_state_dict.items()):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Unexpected non-tensor value in state dict for key: {key}")
            if tensor.device.type != "cpu":
                tensor = tensor.to("cpu")
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            raw_state_dict[key] = tensor

        return raw_state_dict

    def merge_to_file(
        self,
        base_model,
        lora_1_name,
        lora_1_strength,
        lora_2_name,
        lora_2_strength,
        lora_3_name,
        lora_3_strength,
        lora_4_name,
        lora_4_strength,
        output_model_name,
    ):
        logs = []
        model = None
        clip = None
        vae = None
        clip_vision = None
        merged_state_dict = None

        try:
            if base_model == "No compatible models found":
                return ("Error: no compatible models were found in checkpoints/diffusion_models.", "")

            model_type, model_path = self._resolve_model_path(base_model)
            output_path = self._prepare_output_path(model_type, output_model_name)

            lora_requests = self._collect_lora_requests(
                [lora_1_name, lora_2_name, lora_3_name, lora_4_name],
                [lora_1_strength, lora_2_strength, lora_3_strength, lora_4_strength],
            )

            self._log(logs, f"Base model: {model_path}")
            self._log(logs, f"Output file: {output_path}")
            self._log(logs, "Loading base model on CPU-oriented path...")

            model, clip, vae, clip_vision = self._load_base_assets(model_type, model_path)

            if lora_requests:
                self._log(logs, f"Applying {len(lora_requests)} LoRA file(s) with native ComfyUI mechanism...")
                model, clip = self._apply_loras(model, clip, lora_requests, logs)
            else:
                self._log(logs, "No LoRA selected. Saving base model as-is.")

            self._log(logs, "Baking patches on CPU RAM and preparing state dict...")
            merged_state_dict = self._build_cpu_state_dict(model, clip=clip, vae=vae, clip_vision=clip_vision)

            metadata = {
                "ts.node": "TS_CPULoraMerger",
                "ts.merge_device": "cpu",
                "ts.base_model": base_model,
                "ts.lora_stack": json.dumps(lora_requests, ensure_ascii=True),
            }
            self._log(logs, f"Saving {len(merged_state_dict)} tensors to safetensors...")
            _ts_comfy_utils.save_torch_file(merged_state_dict, output_path, metadata=metadata)
            self._log(logs, "Merge completed successfully.")
            return ("\n".join(logs), output_path)

        except Exception as e:
            error_text = str(e).strip()
            if not error_text:
                error_text = e.__class__.__name__
            self._log(logs, f"ERROR: {error_text}")
            traceback_text = _ts_traceback.format_exc()
            logs.append(traceback_text.rstrip())
            self._logger.error("%s %s\n%s", self._LOG_PREFIX, error_text, traceback_text)
            return ("\n".join(logs), "")
        finally:
            if merged_state_dict is not None:
                del merged_state_dict
            del model
            del clip
            del vae
            del clip_vision
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


NODE_CLASS_MAPPINGS["TS_CPULoraMerger"] = TS_CPULoraMergerNode
NODE_DISPLAY_NAME_MAPPINGS["TS_CPULoraMerger"] = "TS CPU LoRA Merger"

