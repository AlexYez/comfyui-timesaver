import math
import torch


class TS_ResolutionSelector:
    ASPECT_PRESETS = [
        ("1:1", 1.0, 1.0),
        ("4:3", 4.0, 3.0),
        ("3:2", 3.0, 2.0),
        ("16:9", 16.0, 9.0),
        ("21:9", 21.0, 9.0),
        ("3:4", 3.0, 4.0),
        ("2:3", 2.0, 3.0),
        ("9:16", 9.0, 16.0),
        ("9:21", 9.0, 21.0),
    ]
    ASPECT_OPTIONS = [name for (name, _w, _h) in ASPECT_PRESETS]

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("img",)
    FUNCTION = "select_resolution"
    CATEGORY = "TS/Resolution"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "aspect_ratio": (cls.ASPECT_OPTIONS, {"default": "1:1"}),
                "resolution": ("FLOAT", {"default": 1.5, "min": 0.5, "max": 4.0, "step": 0.1, "display": "slider"}),
                "custom_ratio": ("STRING", {"default": "0:0"}),
            }
        }

    @classmethod
    def IS_CHANGED(cls, aspect_ratio, resolution, custom_ratio):
        res_value = 0.0 if resolution is None else float(resolution)
        ratio_value = custom_ratio if custom_ratio is not None else "0:0"
        return f"{aspect_ratio}-{ratio_value}-{res_value:.3f}"

    def _parse_ratio(self, ratio_text):
        if not ratio_text:
            return 1.0, 1.0
        parts = ratio_text.split(":")
        if len(parts) != 2:
            return 1.0, 1.0
        try:
            w = float(parts[0])
            h = float(parts[1])
        except ValueError:
            return 1.0, 1.0
        if w <= 0 or h <= 0:
            return 1.0, 1.0
        return w, h

    def _snap_to_divisible(self, value, divisor):
        if divisor <= 1:
            return max(1, int(round(value)))
        return max(divisor, int(round(value / divisor) * divisor))

    def _choose_best_dims(self, ideal_w, ideal_h, aspect, divisor, target_pixels):
        if divisor <= 1:
            w = max(1, int(round(ideal_w)))
            h = max(1, int(round(ideal_h)))
            return w, h

        w1 = self._snap_to_divisible(ideal_w, divisor)
        h1 = self._snap_to_divisible(w1 / aspect, divisor)
        w1 = max(divisor, w1)
        h1 = max(divisor, h1)

        h2 = self._snap_to_divisible(ideal_h, divisor)
        w2 = self._snap_to_divisible(h2 * aspect, divisor)
        w2 = max(divisor, w2)
        h2 = max(divisor, h2)

        def score(pair):
            w, h = pair
            pixel_error = abs((w * h) - target_pixels)
            ratio_error = abs((w / h) - aspect) if h != 0 else float("inf")
            return (pixel_error, ratio_error)

        candidate_a = (w1, h1)
        candidate_b = (w2, h2)
        return min([candidate_a, candidate_b], key=score)

    def select_resolution(self, aspect_ratio, resolution, custom_ratio):
        ratio_w, ratio_h = self._parse_ratio(aspect_ratio)
        custom_value = custom_ratio if custom_ratio is not None else "0:0"
        if custom_value.strip() and custom_value.strip() != "0:0":
            ratio_w, ratio_h = self._parse_ratio(custom_value)
        aspect = ratio_w / ratio_h if ratio_h != 0 else 1.0

        divisor = 32

        res_value = 0.0 if resolution is None else float(resolution)
        if res_value <= 0.0:
            res_value = 1.0

        total_pixels = res_value * 1_000_000.0
        ideal_h = math.sqrt(total_pixels / aspect) if aspect > 0 else 1.0
        ideal_w = ideal_h * aspect

        width, height = self._choose_best_dims(ideal_w, ideal_h, aspect, divisor, total_pixels)
        width = max(1, int(width))
        height = max(1, int(height))

        img = torch.zeros((1, height, width, 3), dtype=torch.float32)

        print(f"[TS Resolution Selector] aspect_ratio={aspect_ratio} custom_ratio={custom_value} resolution={res_value:.3f} divide_by={divisor}")
        print(f"[TS Resolution Selector] output={width}x{height} img_shape={tuple(img.shape)}")

        return (img,)


NODE_CLASS_MAPPINGS = {
    "TS_ResolutionSelector": TS_ResolutionSelector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TS_ResolutionSelector": "TS Resolution Selector",
}
