import sys
import traceback
import torch

# ==============================================================================
# TS Logger (Utility Class)
# ==============================================================================
class TS_Logger:
    """
    Утилита для логирования.
    """
    @staticmethod
    def log(node_name, message, color="cyan"):
        colors = {
            "cyan": "\033[96m",
            "green": "\033[92m",
            "yellow": "\033[93m",
            "red": "\033[91m",
            "reset": "\033[0m"
        }
        c = colors.get(color, colors["cyan"])
        print(f"{c}[TS Nodes: {node_name}]{colors['reset']} {message}")

    @staticmethod
    def error(node_name, message):
        TS_Logger.log(node_name, message, "red")

# ==============================================================================
# Node 1: TS Float Slider
# ==============================================================================
class TS_FloatSlider:
    def __init__(self): pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1,
                    "display": "slider", "round": 0.01
                }),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("float_value",)
    FUNCTION = "get_value"
    CATEGORY = "TS Tools/Sliders"
    DESCRIPTION = "Float slider (0.0 - 1.0)"

    def get_value(self, value):
        TS_Logger.log("FloatSlider", f"Value: {value:.2f}")
        return (float(value),)

# ==============================================================================
# Node 2: TS Int Slider
# ==============================================================================
class TS_Int_Slider:
    def __init__(self): pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("INT", {
                    "default": 512, "min": 320, "max": 2048, "step": 8,
                    "display": "slider" 
                }),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("int_value",)
    FUNCTION = "get_value"
    CATEGORY = "TS Tools/Sliders"
    DESCRIPTION = "Int slider (320 - 2048)"

    def get_value(self, value):
        TS_Logger.log("IntSlider", f"Value: {value}")
        return (int(value),)

# ==============================================================================
# Node 3: TS Smart Switch (Any Type)
# ==============================================================================
class TS_Smart_Switch:
    def __init__(self): pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "switch": ("BOOLEAN", {"default": True, "label_on": "Input 1", "label_off": "Input 2"}),
            },
            "optional": {
                "input_1": ("*",), 
                "input_2": ("*",),
            }
        }

    RETURN_TYPES = ("*",) 
    RETURN_NAMES = ("output",)
    FUNCTION = "smart_switch"
    CATEGORY = "TS Tools/Logic"
    DESCRIPTION = "Smart switch for ANY data. Auto-failover if one input is missing."

    def smart_switch(self, switch, input_1=None, input_2=None):
        selected_source = "None"
        result = None
        status_msg = ""

        try:
            has_input_1 = input_1 is not None
            has_input_2 = input_2 is not None

            if has_input_1 and has_input_2:
                if switch:
                    result = input_1
                    selected_source = "Input 1"
                    status_msg = "(Switch: ON)"
                else:
                    result = input_2
                    selected_source = "Input 2"
                    status_msg = "(Switch: OFF)"
            elif has_input_1:
                result = input_1
                selected_source = "Input 1"
                status_msg = "(Auto-Failover)"
            elif has_input_2:
                result = input_2
                selected_source = "Input 2"
                status_msg = "(Auto-Failover)"
            else:
                TS_Logger.error("SmartSwitch", "Warning: Both inputs are None.")
                return (None,)

            # Log info
            info = "Unknown"
            if hasattr(result, 'shape'): info = f"Tensor {result.shape}"
            elif isinstance(result, (int, float, str)): info = str(result)
            
            TS_Logger.log("SmartSwitch", f"Selected: {selected_source} {status_msg} | {info}")
            return (result,)

        except Exception as e:
            TS_Logger.error("SmartSwitch", f"Error: {str(e)}")
            return (None,)

# ==============================================================================
# Node 4: TS Smart Image Switch (Strict Image Type)
# ==============================================================================
class TS_Smart_Image_Switch:
    """
    Умный переключатель СПЕЦИАЛЬНО для изображений.
    Валидация: Если вход не является torch.Tensor (Image), он считается пустым.
    """
    def __init__(self): pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Обновленные лейблы
                "switch": ("BOOLEAN", {"default": True, "label_on": "Input 1", "label_off": "Input 2"}),
            },
            "optional": {
                # Обновленные ключи входов
                "input_1": ("*",), 
                "input_2": ("*",),
            }
        }

    RETURN_TYPES = ("IMAGE",) 
    RETURN_NAMES = ("image",)
    FUNCTION = "smart_image_switch"
    CATEGORY = "TS Tools/Logic"
    DESCRIPTION = "Switch for IMAGES only. Non-image inputs are treated as empty/missing."

    def _is_valid_image(self, data):
        """Проверка: является ли данными валидным тензором изображения."""
        if data is None:
            return False
        if isinstance(data, torch.Tensor):
            # Простая эвристика: ComfyUI изображения обычно [B, H, W, C]
            return True 
        return False

    def smart_image_switch(self, switch, input_1=None, input_2=None):
        selected_source = "None"
        result = None
        status_msg = ""

        try:
            # 1. Строгая валидация типов (проверяем input_1 и input_2)
            valid_1 = self._is_valid_image(input_1)
            valid_2 = self._is_valid_image(input_2)

            # Если подали мусор (например, String), логируем предупреждение
            if input_1 is not None and not valid_1:
                TS_Logger.log("ImageSwitch", f"Input 1 ignored: Received {type(input_1).__name__} instead of Tensor", "yellow")
            if input_2 is not None and not valid_2:
                TS_Logger.log("ImageSwitch", f"Input 2 ignored: Received {type(input_2).__name__} instead of Tensor", "yellow")

            # 2. Логика переключения (используем только valid_ переменные)
            if valid_1 and valid_2:
                if switch:
                    result = input_1
                    selected_source = "Input 1"
                    status_msg = "(Switch: ON)"
                else:
                    result = input_2
                    selected_source = "Input 2"
                    status_msg = "(Switch: OFF)"
            
            elif valid_1:
                result = input_1
                selected_source = "Input 1"
                status_msg = "(Auto-Failover: Input 2 invalid/empty)"
            
            elif valid_2:
                result = input_2
                selected_source = "Input 2"
                status_msg = "(Auto-Failover: Input 1 invalid/empty)"
            
            else:
                TS_Logger.error("ImageSwitch", "CRITICAL: No valid images found on inputs!")
                # Возвращаем черный квадрат 64x64 для безопасности
                empty_img = torch.zeros((1, 64, 64, 3), dtype=torch.float32, device="cpu")
                return (empty_img,)

            TS_Logger.log("ImageSwitch", f"Selected: \033[92m{selected_source}\033[0m {status_msg} | Shape: {result.shape}")
            return (result,)

        except Exception as e:
            TS_Logger.error("ImageSwitch", f"CRITICAL ERROR: {str(e)}")
            traceback.print_exc()
            return (torch.zeros((1, 64, 64, 3)),)

# ==============================================================================
# Node Registration
# ==============================================================================
NODE_CLASS_MAPPINGS = {
    "TS_FloatSlider": TS_FloatSlider,
    "TS_Int_Slider": TS_Int_Slider,
    "TS_Smart_Switch": TS_Smart_Switch,
    "TS_Smart_Image_Switch": TS_Smart_Image_Switch
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TS_FloatSlider": "TS Float Slider",
    "TS_Int_Slider": "TS Int Slider",
    "TS_Smart_Switch": "TS Smart Switch",
    "TS_Smart_Image_Switch": "TS Smart Image Switch"
}