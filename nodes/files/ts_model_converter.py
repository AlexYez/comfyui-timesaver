"""TS Model Converter — convert in-memory model to a target precision/format.

node_id: TS_ModelConverter
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


NODE_CLASS_MAPPINGS = {"TS_ModelConverter": TS_ModelConverterNode}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_ModelConverter": "TS Model Converter"}
