"""TS Color Grade — basic color grading (hue/temperature/saturation/contrast/gain/lift/gamma/brightness).

node_id: TS_Color_Grade
"""

import math

import torch


class TS_Color_Grade:
    CATEGORY = "TS/Color"

    @staticmethod
    def _ts_slider(default, min_val, max_val, step, round_val=None):
        spec = {"default": default, "min": min_val, "max": max_val, "step": step}
        return ("FLOAT", spec)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "hue": cls._ts_slider(0.0, -180.0, 180.0, 0.1, 0.1),
                "temperature": cls._ts_slider(0.0, -1.0, 1.0, 0.01, 0.01),
                "saturation": cls._ts_slider(1.0, 0.0, 3.0, 0.01, 0.01),
                "contrast": cls._ts_slider(1.0, 0.0, 3.0, 0.01, 0.01),
                "gain": cls._ts_slider(1.0, 0.0, 3.0, 0.01, 0.01),
                "lift": cls._ts_slider(0.0, -1.0, 1.0, 0.01, 0.01),
                "gamma": cls._ts_slider(1.0, 0.1, 3.0, 0.01, 0.01),
                "brightness": cls._ts_slider(0.0, -1.0, 1.0, 0.01, 0.01),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"

    @staticmethod
    def adjust_hue(image, hue):
        if abs(hue) < 1e-6:
            return image
        hue_radians = hue / 180.0 * math.pi
        U = math.cos(hue_radians)
        W = math.sin(hue_radians)
        mat = image.new_tensor([
            [0.299 + 0.701 * U + 0.168 * W, 0.587 - 0.587 * U + 0.330 * W, 0.114 - 0.114 * U - 0.497 * W],
            [0.299 - 0.299 * U - 0.328 * W, 0.587 + 0.413 * U + 0.035 * W, 0.114 - 0.114 * U + 0.292 * W],
            [0.299 - 0.300 * U + 1.250 * W, 0.587 - 0.588 * U - 1.050 * W, 0.114 + 0.886 * U - 0.203 * W],
        ])
        return torch.clamp(image @ mat.T, 0, 1)

    @staticmethod
    def _ts_validate_image(image):
        if not isinstance(image, torch.Tensor):
            raise ValueError("TS Color Grade expects IMAGE tensor.")
        if image.ndim != 4 or image.shape[-1] != 3:
            raise ValueError("TS Color Grade expects IMAGE tensor [B,H,W,3].")
        return image.clone().float().clamp(0, 1)

    @staticmethod
    def _ts_apply_temperature(image, temperature):
        if abs(temperature) < 1e-6:
            return image
        shift = image.new_tensor([temperature * 0.05, 0.0, -temperature * 0.05])
        return image + shift

    @staticmethod
    def _ts_apply_saturation(image, saturation):
        if saturation == 1.0:
            return image
        gray = image.mean(dim=-1, keepdim=True)
        return torch.lerp(gray, image, saturation)

    @staticmethod
    def _ts_apply_contrast(image, contrast):
        if contrast == 1.0:
            return image
        return (image - 0.5) * contrast + 0.5

    @staticmethod
    def _ts_apply_gain_lift(image, gain, lift):
        if gain == 1.0 and abs(lift) < 1e-6:
            return image
        return image * gain + lift

    @staticmethod
    def _ts_apply_gamma(image, gamma):
        if gamma == 1.0:
            return image
        return torch.pow(torch.clamp(image, 0.0, 1.0), gamma)

    @staticmethod
    def _ts_apply_brightness(image, brightness):
        if abs(brightness) < 1e-6:
            return image
        return image + brightness

    def process(self, image, hue, temperature, saturation, contrast, gain, lift, gamma, brightness):
        img = self._ts_validate_image(image)

        img = self.adjust_hue(img, hue)
        img = self._ts_apply_temperature(img, temperature)
        img = self._ts_apply_saturation(img, saturation)
        img = self._ts_apply_contrast(img, contrast)
        img = self._ts_apply_gain_lift(img, gain, lift)
        img = self._ts_apply_gamma(img, gamma)
        img = self._ts_apply_brightness(img, brightness)

        return (torch.clamp(img, 0.0, 1.0),)


NODE_CLASS_MAPPINGS = {"TS_Color_Grade": TS_Color_Grade}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_Color_Grade": "TS Color Grade"}
