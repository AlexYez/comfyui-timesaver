"""TS CPU LoRA Merger — merge a LoRA into a base model on CPU and save the result.

node_id: TS_CPULoraMerger
"""

import gc
import json
import logging as _ts_logging
import os
import re as _ts_re
import traceback as _ts_traceback
from collections import OrderedDict

import torch

import comfy.sd
import comfy.utils as _ts_comfy_utils
import folder_paths
from comfy_api.latest import IO


_SUPPORTED_MODEL_EXTENSIONS = (".safetensors", ".ckpt", ".pt", ".pth")
_LORA_NONE = "None"
_LOG_PREFIX = "[TS CPULoraMerger]"
_logger = _ts_logging.getLogger("TS_CPULoraMerger")


def _build_model_choices():
    model_choices = []
    for model_type in ("checkpoints", "diffusion_models"):
        for filename in folder_paths.get_filename_list(model_type):
            if filename.lower().endswith(_SUPPORTED_MODEL_EXTENSIONS):
                model_choices.append(f"{model_type} | {filename}")

    if not model_choices:
        model_choices = ["No compatible models found"]
    return sorted(model_choices)


def _build_lora_choices():
    loras = sorted(folder_paths.get_filename_list("loras"))
    return [_LORA_NONE] + loras


class TS_CPULoraMergerNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        model_choices = _build_model_choices()
        lora_choices = _build_lora_choices()
        return IO.Schema(
            node_id="TS_CPULoraMerger",
            display_name="TS CPU LoRA Merger",
            category="TS/Files",
            is_output_node=True,
            inputs=[
                IO.Combo.Input("base_model", options=model_choices),
                IO.Combo.Input("lora_1_name", options=lora_choices, default=_LORA_NONE),
                IO.Float.Input("lora_1_strength", default=1.0, min=-100.0, max=100.0, step=0.01),
                IO.Combo.Input("lora_2_name", options=lora_choices, default=_LORA_NONE),
                IO.Float.Input("lora_2_strength", default=1.0, min=-100.0, max=100.0, step=0.01),
                IO.Combo.Input("lora_3_name", options=lora_choices, default=_LORA_NONE),
                IO.Float.Input("lora_3_strength", default=1.0, min=-100.0, max=100.0, step=0.01),
                IO.Combo.Input("lora_4_name", options=lora_choices, default=_LORA_NONE),
                IO.Float.Input("lora_4_strength", default=1.0, min=-100.0, max=100.0, step=0.01),
                IO.String.Input("output_model_name", default="ts_merged_model.safetensors", multiline=False),
            ],
            outputs=[
                IO.String.Output(display_name="log"),
                IO.String.Output(display_name="saved_model_path"),
            ],
        )

    @staticmethod
    def _log(logs, message):
        msg = f"{_LOG_PREFIX} {message}"
        logs.append(msg)
        _logger.info(msg)

    @staticmethod
    def _resolve_model_path(selected_model):
        if " | " not in selected_model:
            raise ValueError(f"Invalid model selector: {selected_model}")

        model_type, model_name = selected_model.split(" | ", 1)
        model_path = folder_paths.get_full_path(model_type, model_name)
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_name}")
        return model_type, model_path

    @staticmethod
    def _sanitize_output_filename(output_model_name):
        filename = (output_model_name or "").strip()
        if not filename:
            filename = "ts_merged_model"
        filename = filename.replace("\\", "_").replace("/", "_")
        filename = _ts_re.sub(r"[^A-Za-z0-9._-]+", "_", filename)
        if not filename.lower().endswith(".safetensors"):
            filename = f"{filename}.safetensors"
        return filename

    @staticmethod
    def _unique_path(target_path):
        if not os.path.exists(target_path):
            return target_path

        base, ext = os.path.splitext(target_path)
        index = 1
        while True:
            candidate = f"{base}_{index:03d}{ext}"
            if not os.path.exists(candidate):
                return candidate
            index += 1

    @classmethod
    def _prepare_output_path(cls, model_type, output_model_name):
        if model_type == "checkpoints":
            target_dirs = folder_paths.get_folder_paths("checkpoints")
        else:
            target_dirs = folder_paths.get_folder_paths("diffusion_models")

        if target_dirs:
            target_dir = target_dirs[0]
        else:
            target_dir = os.path.join(folder_paths.get_output_directory(), "diffusion_models")

        os.makedirs(target_dir, exist_ok=True)
        filename = cls._sanitize_output_filename(output_model_name)
        return cls._unique_path(os.path.join(target_dir, filename))

    @staticmethod
    def _collect_lora_requests(lora_names, lora_strengths):
        requests = []
        for index, (name, strength) in enumerate(zip(lora_names, lora_strengths), start=1):
            if not name or name == _LORA_NONE:
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

    @staticmethod
    def _load_base_assets(model_type, model_path):
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

    @classmethod
    def _apply_loras(cls, model, clip, lora_requests, logs):
        for request in lora_requests:
            lora_path = folder_paths.get_full_path("loras", request["name"])
            if not lora_path or not os.path.exists(lora_path):
                raise FileNotFoundError(f"LoRA file not found: {request['name']}")

            cls._log(
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

    @staticmethod
    def _build_cpu_state_dict(model, clip=None, vae=None, clip_vision=None):
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

    @classmethod
    def execute(
        cls,
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
    ) -> IO.NodeOutput:
        logs = []
        model = None
        clip = None
        vae = None
        clip_vision = None
        merged_state_dict = None

        try:
            if base_model == "No compatible models found":
                return IO.NodeOutput("Error: no compatible models were found in checkpoints/diffusion_models.", "")

            model_type, model_path = cls._resolve_model_path(base_model)
            output_path = cls._prepare_output_path(model_type, output_model_name)

            lora_requests = cls._collect_lora_requests(
                [lora_1_name, lora_2_name, lora_3_name, lora_4_name],
                [lora_1_strength, lora_2_strength, lora_3_strength, lora_4_strength],
            )

            cls._log(logs, f"Base model: {model_path}")
            cls._log(logs, f"Output file: {output_path}")
            cls._log(logs, "Loading base model on CPU-oriented path...")

            model, clip, vae, clip_vision = cls._load_base_assets(model_type, model_path)

            if lora_requests:
                cls._log(logs, f"Applying {len(lora_requests)} LoRA file(s) with native ComfyUI mechanism...")
                model, clip = cls._apply_loras(model, clip, lora_requests, logs)
            else:
                cls._log(logs, "No LoRA selected. Saving base model as-is.")

            cls._log(logs, "Baking patches on CPU RAM and preparing state dict...")
            merged_state_dict = cls._build_cpu_state_dict(model, clip=clip, vae=vae, clip_vision=clip_vision)

            metadata = {
                "ts.node": "TS_CPULoraMerger",
                "ts.merge_device": "cpu",
                "ts.base_model": base_model,
                "ts.lora_stack": json.dumps(lora_requests, ensure_ascii=True),
            }
            cls._log(logs, f"Saving {len(merged_state_dict)} tensors to safetensors...")
            _ts_comfy_utils.save_torch_file(merged_state_dict, output_path, metadata=metadata)
            cls._log(logs, "Merge completed successfully.")
            return IO.NodeOutput("\n".join(logs), output_path)

        except Exception as e:
            error_text = str(e).strip()
            if not error_text:
                error_text = e.__class__.__name__
            cls._log(logs, f"ERROR: {error_text}")
            traceback_text = _ts_traceback.format_exc()
            logs.append(traceback_text.rstrip())
            _logger.error("%s %s\n%s", _LOG_PREFIX, error_text, traceback_text)
            return IO.NodeOutput("\n".join(logs), "")
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


NODE_CLASS_MAPPINGS = {"TS_CPULoraMerger": TS_CPULoraMergerNode}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_CPULoraMerger": "TS CPU LoRA Merger"}
