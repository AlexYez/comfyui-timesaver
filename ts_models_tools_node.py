import os
import json
import glob
import gc
import torch
from comfy.model_patcher import ModelPatcher
import comfy.model_patcher
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
    # def should_convert_to_fp8(self, tensor_name: str) -> bool:
    #         if not tensor_name.endswith(".weight"):
    #             return False
    #         if not "blocks." in tensor_name:
    #             return False
    #         if "cross_attn" in tensor_name or "ffn" in tensor_name or "self_attn" in tensor_name:
    #             if ".norm_k.weight" in tensor_name or ".norm_q.weight" in tensor_name or ".norm.weight" in tensor_name:
    #                 return False
    #             return True
    #         return False



    def should_convert_to_fp8(self, tensor_name: str) -> bool:
        # 1. Базовая проверка: работаем только с весами (.weight)
        if not tensor_name.endswith(".weight"):
            return False

        # 2. Исключаем FP32 параметры (согласно скану)
        if "scale_weight" in tensor_name: # Они везде float32
            return False
        if "patch_embedding" in tensor_name: # float32
            return False

        # 3. Исключаем FP16 параметры (согласно скану)
        # "norm" покроет: norm3, norm_q, norm_k
        if "norm" in tensor_name: 
            return False
        if "modulation" in tensor_name: # float16
            return False

        # 4. Логика для Блоков (основная часть модели)
        if "blocks." in tensor_name:
            # Так как мы выше уже исключили 'norm' и 'modulation',
            # всё оставшееся внутри блоков (attn q/k/v/o и ffn) должно быть FP8.
            # Дополнительная проверка на всякий случай:
            if "cross_attn" in tensor_name or "ffn" in tensor_name or "self_attn" in tensor_name:
                return True
            # Если вдруг внутри блока есть что-то еще, что мы не учли, лучше вернуть False
            return False

        # 5. Внешние слои (опционально, для полного соответствия эталону)
        # В вашем скане эти слои тоже FP8. 
        # Если вы хотите сэкономить максимум памяти, оставьте True.
        # Если хотите безопасный режим (только блоки), закомментируйте эти строки.
        
        if "head.head.weight" in tensor_name:
            return True
            
        if "text_embedding" in tensor_name or "time_embedding" in tensor_name or "time_projection" in tensor_name:
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
        # В ComfyUI модель обычно обернута в ModelPatcher
        # Нам нужно добраться до реальной torch.nn.Module
        
        real_model = None
        
        if isinstance(model, comfy.model_patcher.ModelPatcher):
            real_model = model.model
        else:
            real_model = model

        # Буфер для сбора информации
        output_lines = []
        stats = {}
        total_params = 0

        output_lines.append("=== MODEL SCAN REPORT ===")
        output_lines.append(f"Type: {type(real_model).__name__}")
        output_lines.append("-" * 60)

        if not summary_only:
            output_lines.append(f"{'Layer Name':<50} | {'Shape':<20} | {'Dtype':<10} | {'Device':<6}")
            output_lines.append("-" * 90)

        # Перебираем все параметры модели
        # Используем named_parameters, чтобы получить веса
        try:
            # Некоторые модели ComfyUI хранят веса в diffusion_model
            iterator = real_model.named_parameters()
            if hasattr(real_model, "diffusion_model"):
                 output_lines.append("Note: Scanning internal diffusion_model")
                 iterator = real_model.diffusion_model.named_parameters()

            for name, param in iterator:
                # Получаем характеристики
                shape_str = str(tuple(param.shape))
                dtype_str = str(param.dtype).replace("torch.", "")
                device_str = str(param.device).split(":")[0] # cpu или cuda
                num_params = param.numel()

                # Сбор статистики
                total_params += num_params
                if dtype_str not in stats:
                    stats[dtype_str] = 0
                stats[dtype_str] += num_params

                # Форматирование строки для детального отчета
                if not summary_only:
                    output_lines.append(f"{name:<50} | {shape_str:<20} | {dtype_str:<10} | {device_str:<6}")

        except Exception as e:
            return (f"Error scanning model: {str(e)}",)

        output_lines.append("-" * 60)
        output_lines.append("=== SUMMARY STATISTICS ===")
        output_lines.append(f"Total Parameters: {total_params:,}")
        output_lines.append("Distribution by Data Type:")
        
        for dtype, count in stats.items():
            percent = (count / total_params) * 100
            output_lines.append(f" - {dtype}: {count:,} params ({percent:.2f}%)")

        return ("\n".join(output_lines),)
    

class TS_WanContextWindowScript:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "context_length": ("INT", {"default": 81, "min": 17, "max": 241, "step": 4}),
                "overlap": ("INT", {"default": 16, "min": 4, "max": 60, "step": 4}),
                # True для High Noise (начало), False для Low Noise (конец)
                "anchor_first_frame": ("BOOLEAN", {"default": True, "label": "Anchor First Frame (ON for High Noise, OFF for Low)"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch_model"
    CATEGORY = "Wan 2.2"

    def patch_model(self, model, context_length, overlap, anchor_first_frame):
        m = model.clone()

        def context_scheduler_wrapper(model_function, params):
            input_x = params["input"]
            timestep = params["timestep"]
            c = params["c"]
            
            # --- ОПРЕДЕЛЕНИЕ РАЗМЕРНОСТИ (FIX для ошибки "too many values") ---
            # Wan может подавать [Batch, Channels, Frames, Height, Width] (5D)
            # Или [Frames, Channels, Height, Width] (4D)
            
            if input_x.ndim == 5:
                # 5D случай: (Batch, C, Time, H, W)
                B_batch, C, T, H, W = input_x.shape
                total_frames = T
                time_dim = 2 # Ось времени - это индекс 2
            elif input_x.ndim == 4:
                # 4D случай: (Time, C, H, W)
                T, C, H, W = input_x.shape
                total_frames = T
                time_dim = 0 # Ось времени - это индекс 0
            else:
                raise ValueError(f"WanContextWindow: Unsupported input shape {input_x.shape}")

            # Если видео короче окна, пропускаем без изменений
            if total_frames <= context_length:
                return model_function(input_x, timestep, **c)

            output = torch.zeros_like(input_x)
            
            # Маска для усреднения (учитываем размерности)
            if time_dim == 2:
                count_mask = torch.zeros((1, 1, total_frames, 1, 1), device=input_x.device)
            else:
                count_mask = torch.zeros((total_frames, 1, 1, 1), device=input_x.device)
            
            stride = context_length - overlap
            
            # Генерация окон
            windows = []
            start_idx = 0
            while start_idx < total_frames:
                end_idx = min(start_idx + context_length, total_frames)
                if (end_idx - start_idx) < context_length and total_frames > context_length:
                    start_idx = max(0, total_frames - context_length)
                    end_idx = total_frames
                windows.append((start_idx, end_idx))
                if end_idx == total_frames:
                    break
                start_idx += stride

            # Хелпер для нарезки тензоров по оси времени
            def slice_tensor(tensor, start, end, dim):
                if dim == 0:
                    return tensor[start:end]
                elif dim == 2:
                    return tensor[:, :, start:end, :, :]
                return tensor

            # Хелпер для вставки (add) результата
            def add_slice(dest, source, start, end, dim):
                if dim == 0:
                    dest[start:end] += source
                elif dim == 2:
                    dest[:, :, start:end, :, :] += source

            for i, (start, end) in enumerate(windows):
                # Нарезаем входной латент
                chunk_input = slice_tensor(input_x, start, end, time_dim).clone()
                
                # --- ЛОГИКА ANCHOR FRAME ---
                if anchor_first_frame and start > 0:
                    if time_dim == 0:
                        # Берем 0-й кадр всего видео
                        chunk_input[0] = input_x[0]
                    elif time_dim == 2:
                        # Берем 0-й кадр по оси времени (индекс 2)
                        chunk_input[:, :, 0, :, :] = input_x[:, :, 0, :, :]

                # Обработка timestep (если это тензор длины T)
                chunk_ts = timestep
                if isinstance(timestep, torch.Tensor) and timestep.shape[0] == total_frames:
                    chunk_ts = timestep[start:end]

                # Обработка conditioning
                chunk_c = {}
                for k, v in c.items():
                    if isinstance(v, torch.Tensor) and v.shape[0] == total_frames:
                        chunk_c[k] = v[start:end]
                    else:
                        chunk_c[k] = v
                
                # Запуск модели на чанке
                chunk_out = model_function(chunk_input, chunk_ts, **chunk_c)
                
                # --- БЛЕНДИНГ (Linear Fade) ---
                window_len = end - start
                
                # Создаем веса правильной формы
                if time_dim == 0:
                    weights = torch.ones((window_len, 1, 1, 1), device=input_x.device)
                    # Fade In
                    if start > 0:
                        fade_len = min(overlap, window_len)
                        weights[:fade_len] = torch.linspace(0, 1, fade_len, device=input_x.device).view(-1, 1, 1, 1)
                    # Fade Out
                    if end < total_frames:
                        fade_len = min(overlap, window_len)
                        weights[-fade_len:] = torch.linspace(1, 0, fade_len, device=input_x.device).view(-1, 1, 1, 1)
                        
                    count_mask[start:end] += weights
                    
                elif time_dim == 2:
                    weights = torch.ones((1, 1, window_len, 1, 1), device=input_x.device)
                    # Fade In
                    if start > 0:
                        fade_len = min(overlap, window_len)
                        w_vals = torch.linspace(0, 1, fade_len, device=input_x.device).view(1, 1, -1, 1, 1)
                        weights[:, :, :fade_len, :, :] = w_vals
                    # Fade Out
                    if end < total_frames:
                        fade_len = min(overlap, window_len)
                        w_vals = torch.linspace(1, 0, fade_len, device=input_x.device).view(1, 1, -1, 1, 1)
                        weights[:, :, -fade_len:, :, :] = w_vals
                    
                    count_mask[:, :, start:end, :, :] += weights

                # Складываем результат
                add_slice(output, chunk_out * weights, start, end, time_dim)
            
            return output / (count_mask + 1e-6)

        m.set_model_unet_function_wrapper(context_scheduler_wrapper)
        return (m,)


# ==========================
# Node Registration
# ==========================
NODE_CLASS_MAPPINGS = {
    "TS_ModelConverter": TS_ModelConverterNode,
    "TS_ModelConverterAdvanced": TS_ModelConverterAdvancedNode,
    "ModelScanner": ModelScanner,
    "TS_WanContextWindowScript": TS_WanContextWindowScript
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TS_ModelConverter": "TS Model Converter",
    "TS_ModelConverterAdvanced": "TS Model Converter Advanced",
    "ModelScanner": "🔍 Model Layer Scanner",
    "TS_WanContextWindowScript": "TS Wan Context Window"
}
