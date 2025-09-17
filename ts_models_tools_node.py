import os
import json
import glob
import gc
import torch
from comfy.model_patcher import ModelPatcher
from safetensors.torch import save_file, load_file
from safetensors import safe_open
from collections import OrderedDict
from tqdm import tqdm



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
            if hasattr(model, 'diffusion_model'):
                model.diffusion_model = model.diffusion_model.to(torch.float8_e4m3fn)
                return (model,)
            elif isinstance(model, ModelPatcher):
                model.model = model.model.to(torch.float8_e4m3fn)
                return (model,)
            else:
                model = model.to(torch.float8_e4m3fn)
                return (model,)
        except Exception as e:
            print(f"float8_e4m3fn: {str(e)}")
            return (model,) 

COMFYUI_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SEARCH_DIRS = [
    os.path.join(COMFYUI_ROOT, "models", "checkpoints"),
    os.path.join(COMFYUI_ROOT, "models", "diffusion_models"),
    os.path.join(COMFYUI_ROOT, "output", "diffusion_models"),
]
OUTPUT_DIR = os.path.join(COMFYUI_ROOT, "output")

# ==========================
# ComfyUI Custom Node
# ==========================
class TS_ModelConverterAdvancedNode:
    """
    Convert large AI models to FP8 (e4m3fn / e5m2).
    - Автоматически находит модели в нескольких папках
    - Поддержка одиночного .safetensors файла и шардов HuggingFace
    - Сохраняет выходы в ComfyUI/output
    - Подробное логирование и прогресс
    """

    @classmethod
    def INPUT_TYPES(s):
        models = []
        for search_dir in SEARCH_DIRS:
            if os.path.exists(search_dir):
                for item in os.listdir(search_dir):
                    item_path = os.path.join(search_dir, item)
                    if item.endswith(".safetensors") or os.path.isdir(item_path):
                        models.append(f"{os.path.basename(search_dir)}/{item}")
        if not models:
            models = ["<no models found>"]

        return {
            "required": {
                "model_name": (models, {"default": models[0]}),
                "fp8_mode": (["e4m3fn", "e5m2"], {"default": "e5m2"}),
                "shard_subdir": ("STRING", {"multiline": False, "default": "fp8_shards"}),
                "final_filename": ("STRING", {"multiline": False, "default": "converted_model_fp8.safetensors"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("log",)
    FUNCTION = "convert_model"
    CATEGORY = "Model Conversion"

    # --------------------------
    # Filtering tensors
    # --------------------------
    def should_convert_to_fp8(self, tensor_name: str) -> bool:
        if not tensor_name.endswith(".weight"):
            return False
        if not "blocks." in tensor_name:
            return False
        if "cross_attn" in tensor_name or "ffn" in tensor_name or "self_attn" in tensor_name:
            if ".norm_k.weight" in tensor_name or ".norm_q.weight" in tensor_name or ".norm.weight" in tensor_name:
                return False
            return True
        return False

    # --------------------------
    # Main conversion logic
    # --------------------------
    def convert_model(self, model_name, fp8_mode, shard_subdir, final_filename):
        logs = []
        device = "cuda" if torch.cuda.is_available() else "cpu"
        target_dtype = torch.float8_e4m3fn if fp8_mode == "e4m3fn" else torch.float8_e5m2

        logs.append(f"--- START FP8 CONVERSION ---")
        logs.append(f"Selected: {model_name}")
        logs.append(f"Target dtype: {fp8_mode} ({target_dtype})")
        logs.append(f"Device: {device}")

        # Определяем реальный путь к модели
        for base in SEARCH_DIRS:
            candidate = os.path.join(base, model_name.split("/", 1)[-1])
            if os.path.exists(candidate):
                model_path = candidate
                break
        else:
            logs.append("❌ ERROR: model not found in search dirs")
            return ("\n".join(logs),)

        # --- CASE 1: Single-file safetensors ---
        if os.path.isfile(model_path) and model_path.endswith(".safetensors"):
            logs.append("Detected: single safetensors file")
            shard_state = OrderedDict()
            out_path = os.path.join(OUTPUT_DIR, final_filename)

            with safe_open(model_path, framework="pt", device="cpu") as f_in:
                tensor_names = f_in.keys()
                for tensor_name in tqdm(tensor_names, desc="Converting tensors"):
                    tensor = f_in.get_tensor(tensor_name)
                    if self.should_convert_to_fp8(tensor_name):
                        logs.append(f"  → {tensor_name} converted to {fp8_mode}")
                        tensor = tensor.to(device).to(target_dtype).to("cpu")
                    else:
                        logs.append(f"  → {tensor_name} kept as {tensor.dtype}")
                        tensor = tensor.to("cpu")
                    shard_state[tensor_name] = tensor

            save_file(shard_state, out_path)
            logs.append(f"✔ Final model saved: {out_path}")
            logs.append("--- DONE ---")
            return ("\n".join(logs),)

        # --- CASE 2: Folder with shards ---
        else:
            model_dir = model_path
            index_json_path = os.path.join(model_dir, "model.safetensors.index.json")
            if not os.path.exists(index_json_path):
                logs.append(f"❌ ERROR: index.json not found in {model_dir}")
                return ("\n".join(logs),)

            # Пути для выходов
            shard_dir = os.path.join(OUTPUT_DIR, shard_subdir)
            final_out = os.path.join(OUTPUT_DIR, final_filename)

            os.makedirs(shard_dir, exist_ok=True)
            logs.append(f"Shard dir: {shard_dir}")
            logs.append(f"Final: {final_out}")

            # Загружаем index.json
            with open(index_json_path, "r") as f:
                index_data = json.load(f)

            weight_map = index_data.get("weight_map", {})
            if not weight_map:
                logs.append("❌ ERROR: weight_map missing in index.json")
                return ("\n".join(logs),)

            # Группируем тензоры по шарам
            tensors_by_shard = {}
            for tensor_name, shard_filename in weight_map.items():
                tensors_by_shard.setdefault(shard_filename, []).append(tensor_name)

            logs.append(f"Found {len(tensors_by_shard)} shards")

            # Конвертация шардов
            for shard_idx, (orig_filename, tensor_names) in enumerate(
                tqdm(tensors_by_shard.items(), desc="Converting shards")
            ):
                in_shard_path = os.path.join(model_dir, orig_filename)
                out_shard_path = os.path.join(shard_dir, f"fp8_{orig_filename}")

                logs.append(f"\n--- Shard {shard_idx+1}/{len(tensors_by_shard)} ---")
                logs.append(f"Input: {in_shard_path}")
                logs.append(f"Output: {out_shard_path}")

                if os.path.exists(out_shard_path):
                    logs.append("✔ Already exists, skip")
                    continue

                shard_state = OrderedDict()
                with safe_open(in_shard_path, framework="pt", device="cpu") as f_in:
                    for tensor_name in tqdm(tensor_names, desc=f"Tensors in {orig_filename}", leave=False):
                        tensor = f_in.get_tensor(tensor_name)
                        if self.should_convert_to_fp8(tensor_name):
                            logs.append(f"  → {tensor_name} converted to {fp8_mode}")
                            tensor = tensor.to(device).to(target_dtype).to("cpu")
                        else:
                            logs.append(f"  → {tensor_name} kept as {tensor.dtype}")
                            tensor = tensor.to("cpu")
                        shard_state[tensor_name] = tensor

                save_file(shard_state, out_shard_path)
                logs.append(f"✔ Saved shard: {out_shard_path}")

                del shard_state
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Мердж в финальный safetensors
            logs.append("\n--- Merging shards ---")
            shard_files = sorted(glob.glob(os.path.join(shard_dir, "*.safetensors")))
            if not shard_files:
                logs.append("❌ ERROR: no shards for merging")
                return ("\n".join(logs),)

            merged = OrderedDict()
            for shard_path in tqdm(shard_files, desc="Merging"):
                logs.append(f"Adding {os.path.basename(shard_path)}")
                merged.update(load_file(shard_path, device="cpu"))

            save_file(merged, final_out)
            logs.append(f"✔ Final model saved: {final_out}")
            logs.append("--- DONE ---")

            return ("\n".join(logs),)


# ==========================
# Node Registration
# ==========================
NODE_CLASS_MAPPINGS = {
    "TS_ModelConverter": TS_ModelConverterNode,
    "TS_ModelConverterAdvanced": TS_ModelConverterAdvancedNode
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TS_ModelConverter": "TS Model Converter",
    "TS_ModelConverterAdvanced": "TS Model Converter Advanced"
}
