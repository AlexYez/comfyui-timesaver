import math
import torch
import torch.nn.functional as F

import logging

from comfy_api.latest import IO


logger = logging.getLogger("comfyui_timesaver.ts_resolution_selector")
LOG_PREFIX = "[TS Resolution Selector]"


_ASPECT_PRESETS = [
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
_ASPECT_OPTIONS = [name for (name, _w, _h) in _ASPECT_PRESETS]


class TS_ResolutionSelector(IO.ComfyNode):
    ASPECT_PRESETS = _ASPECT_PRESETS
    ASPECT_OPTIONS = _ASPECT_OPTIONS

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_ResolutionSelector",
            display_name="TS Resolution Selector",
            category="TS/Image",
            inputs=[
                IO.Combo.Input("aspect_ratio", options=_ASPECT_OPTIONS, default="1:1"),
                IO.Float.Input("resolution", default=1.5, min=0.5, max=4.0, step=0.1, display_mode=IO.NumberDisplay.slider),
                IO.String.Input("custom_ratio", default="0:0"),
                IO.Boolean.Input("original_aspect", default=False),
                IO.Image.Input("image", optional=True),
            ],
            outputs=[IO.Image.Output(display_name="img")],
        )

    @classmethod
    def fingerprint_inputs(cls, aspect_ratio, resolution, custom_ratio, original_aspect=False, image=None):
        if cls._is_valid_image(image):
            return float("nan")
        res_value = 0.0 if resolution is None else float(resolution)
        ratio_value = custom_ratio if custom_ratio is not None else "0:0"
        return f"{aspect_ratio}-{ratio_value}-{res_value:.3f}-{bool(original_aspect)}"

    @staticmethod
    def _is_valid_image(image):
        if image is None or not isinstance(image, torch.Tensor):
            return False
        if image.ndim != 4:
            return False
        batch, src_h, src_w, channels = image.shape
        if batch <= 0 or src_h <= 0 or src_w <= 0 or channels <= 0:
            return False
        return True

    @staticmethod
    def _log(message):
        logger.info("%s %s", LOG_PREFIX, message)

    @classmethod
    def _log_tensor_shape(cls, label, tensor):
        if not isinstance(tensor, torch.Tensor):
            return
        shape = tuple(tensor.shape)
        cls._log(f"{label} shape={shape} dtype={tensor.dtype} device={tensor.device}")

    @staticmethod
    def _parse_ratio(ratio_text):
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

    @staticmethod
    def _snap_to_divisible(value, divisor):
        if divisor <= 1:
            return max(1, int(round(value)))
        return max(divisor, int(round(value / divisor) * divisor))

    @classmethod
    def _choose_best_dims(cls, ideal_w, ideal_h, aspect, divisor, target_pixels):
        if divisor <= 1:
            w = max(1, int(round(ideal_w)))
            h = max(1, int(round(ideal_h)))
            return w, h

        w1 = cls._snap_to_divisible(ideal_w, divisor)
        h1 = cls._snap_to_divisible(w1 / aspect, divisor)
        w1 = max(divisor, w1)
        h1 = max(divisor, h1)

        h2 = cls._snap_to_divisible(ideal_h, divisor)
        w2 = cls._snap_to_divisible(h2 * aspect, divisor)
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

    @classmethod
    def _crop_alpha_to_bbox(cls, image, pad_px=10):
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

    @classmethod
    def _fit_image_to_canvas(cls, image, target_w, target_h):
        if image is None or not isinstance(image, torch.Tensor):
            return None
        if image.ndim != 4:
            return None

        if image.shape[3] >= 4:
            image = cls._crop_alpha_to_bbox(image, pad_px=10)
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

    @staticmethod
    def _get_image_aspect(image):
        if image is None or not isinstance(image, torch.Tensor):
            return None
        if image.ndim != 4:
            return None
        _batch, src_h, src_w, _channels = image.shape
        if src_h <= 0 or src_w <= 0:
            return None
        return float(src_w) / float(src_h)

    @classmethod
    def execute(cls, aspect_ratio, resolution, custom_ratio, original_aspect=False, image=None) -> IO.NodeOutput:
        if not cls._is_valid_image(image):
            if image is not None:
                cls._log("input image is missing or invalid; treating input as disconnected")
            image = None

        ratio_w, ratio_h = cls._parse_ratio(aspect_ratio)
        custom_value = custom_ratio if custom_ratio is not None else "0:0"
        if custom_value.strip() and custom_value.strip() != "0:0":
            ratio_w, ratio_h = cls._parse_ratio(custom_value)
        aspect = ratio_w / ratio_h if ratio_h != 0 else 1.0
        aspect_source = "preset/custom"

        if bool(original_aspect) and image is not None:
            image_aspect = cls._get_image_aspect(image)
            if image_aspect is not None and image_aspect > 0:
                aspect = image_aspect
                aspect_source = "image"

        divisor = 32

        res_value = 0.0 if resolution is None else float(resolution)
        if res_value <= 0.0:
            res_value = 1.0

        total_pixels = res_value * 1_000_000.0
        ideal_h = math.sqrt(total_pixels / aspect) if aspect > 0 else 1.0
        ideal_w = ideal_h * aspect

        width, height = cls._choose_best_dims(ideal_w, ideal_h, aspect, divisor, total_pixels)
        width = max(1, int(width))
        height = max(1, int(height))

        if image is not None:
            image = image.float()
            cls._log_tensor_shape("input", image)
            fitted = cls._fit_image_to_canvas(image, width, height)
            if fitted is not None:
                img = fitted
            else:
                img = torch.zeros((1, height, width, 3), dtype=torch.float32)
        else:
            img = torch.zeros((1, height, width, 3), dtype=torch.float32)

        cls._log(
            f"aspect_ratio={aspect_ratio} custom_ratio={custom_value} original_aspect={bool(original_aspect)} "
            f"aspect_source={aspect_source} resolution={res_value:.3f} divide_by={divisor}"
        )
        cls._log(f"output={width}x{height}")
        cls._log_tensor_shape("output", img)

        return IO.NodeOutput(img)


NODE_CLASS_MAPPINGS = {
    "TS_ResolutionSelector": TS_ResolutionSelector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TS_ResolutionSelector": "TS Resolution Selector",
}
