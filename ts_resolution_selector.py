import math
import torch
import torch.nn.functional as F

import folder_paths


class TS_ResolutionSelector:
    _LOG_PREFIX = "[TS Resolution Selector]"

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
            },
            "optional": {
                "image": ("IMAGE",),
            },
        }

    @classmethod
    def IS_CHANGED(cls, aspect_ratio, resolution, custom_ratio, image=None):
        if image is not None:
            return float("nan")
        res_value = 0.0 if resolution is None else float(resolution)
        ratio_value = custom_ratio if custom_ratio is not None else "0:0"
        return f"{aspect_ratio}-{ratio_value}-{res_value:.3f}"

    def _log(self, message):
        print(f"{self._LOG_PREFIX} {message}")

    def _log_tensor_shape(self, label, tensor):
        if not isinstance(tensor, torch.Tensor):
            return
        shape = tuple(tensor.shape)
        self._log(f"{label} shape={shape} dtype={tensor.dtype} device={tensor.device}")

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

    def _crop_alpha_to_bbox(self, image, pad_px=10):
        if image is None or not isinstance(image, torch.Tensor):
            return None
        if image.ndim != 4:
            return None

        batch, src_h, src_w, channels = image.shape
        if channels < 4:
            return image

        rgb = image[..., :3]
        alpha = image[..., 3:4].clamp(0.0, 1.0)
        output = []

        for i in range(batch):
            alpha_i = alpha[i, ..., 0]
            if not torch.any(alpha_i > 0):
                # Fully transparent image: composite on white without crop.
                comp = rgb[i] * alpha_i.unsqueeze(-1) + (1.0 - alpha_i.unsqueeze(-1))
                output.append(comp.unsqueeze(0))
                continue

            coords = torch.nonzero(alpha_i > 0, as_tuple=False)
            y_min = int(coords[:, 0].min().item())
            y_max = int(coords[:, 0].max().item())
            x_min = int(coords[:, 1].min().item())
            x_max = int(coords[:, 1].max().item())

            y_min = max(0, y_min - pad_px)
            x_min = max(0, x_min - pad_px)
            y_max = min(src_h - 1, y_max + pad_px)
            x_max = min(src_w - 1, x_max + pad_px)

            crop_rgb = rgb[i, y_min : y_max + 1, x_min : x_max + 1, :]
            crop_alpha = alpha_i[y_min : y_max + 1, x_min : x_max + 1].unsqueeze(-1)
            comp = crop_rgb * crop_alpha + (1.0 - crop_alpha)
            output.append(comp.unsqueeze(0))

        return torch.cat(output, dim=0)

    def _fit_image_to_canvas(self, image, target_w, target_h):
        if image is None or not isinstance(image, torch.Tensor):
            return None
        if image.ndim != 4:
            return None

        if image.shape[3] >= 4:
            image = self._crop_alpha_to_bbox(image, pad_px=10)
            if image is None:
                return None

        batch, src_h, src_w, channels = image.shape
        if src_h <= 0 or src_w <= 0 or channels <= 0:
            return None

        scale = min(target_w / float(src_w), target_h / float(src_h))
        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))
        new_w = min(target_w, new_w)
        new_h = min(target_h, new_h)

        image_nchw = image.permute(0, 3, 1, 2)
        resized_nchw = F.interpolate(image_nchw, size=(new_h, new_w), mode="bicubic", align_corners=False)
        resized = resized_nchw.permute(0, 2, 3, 1)

        canvas = torch.ones((batch, target_h, target_w, channels), dtype=image.dtype, device=image.device)
        top = max(0, (target_h - new_h) // 2)
        left = max(0, (target_w - new_w) // 2)
        canvas[:, top : top + new_h, left : left + new_w, :] = resized
        return canvas.clamp(0.0, 1.0)

    def select_resolution(self, aspect_ratio, resolution, custom_ratio, image=None):
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

        if image is not None:
            image = image.float()
            self._log_tensor_shape("input", image)
            fitted = self._fit_image_to_canvas(image, width, height)
            if fitted is not None:
                img = fitted
            else:
                img = torch.zeros((1, height, width, 3), dtype=torch.float32)
        else:
            img = torch.zeros((1, height, width, 3), dtype=torch.float32)

        self._log(
            f"aspect_ratio={aspect_ratio} custom_ratio={custom_value} resolution={res_value:.3f} divide_by={divisor}"
        )
        self._log(f"output={width}x{height}")
        self._log_tensor_shape("output", img)

        return (img,)


NODE_CLASS_MAPPINGS = {
    "TS_ResolutionSelector": TS_ResolutionSelector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TS_ResolutionSelector": "TS Resolution Selector",
}
