"""TS Despill — multi-algorithm chroma spill suppression for green/blue/red screens.

node_id: TS_Despill
"""

import logging

import torch
import torch.nn.functional as F

from comfy_api.latest import IO, UI

from ._keying_helpers import CHANNEL_TO_INDEX, gaussian_blur_4d


class TS_Despill(IO.ComfyNode):
    _LOGGER = logging.getLogger("comfyui_timesaver.ts_despill")
    _LOG_PREFIX = "[TS Despill]"

    _CHANNEL_TO_INDEX = CHANNEL_TO_INDEX

    _ALGORITHMS = (
        "classic",
        "balanced",
        "adaptive",
        "hue_preserve",
    )

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_Despill",
            display_name="TS Despill",
            category="TS/Image",
            description=(
                "Professional despill node for red/green/blue screens. "
                "Includes classic, balanced, adaptive edge, and hue-preserving algorithms."
            ),
            inputs=[
                IO.Image.Input(
                    "image",
                    tooltip="Input IMAGE tensor [B,H,W,3] or [B,H,W,4].",
                ),
                IO.Boolean.Input(
                    "enable",
                    default=True,
                    tooltip="If disabled, node passes input through.",
                ),
                IO.Combo.Input(
                    "screen_color",
                    options=["green", "blue", "red"],
                    default="green",
                    tooltip="Spill color to suppress.",
                ),
                IO.Combo.Input(
                    "algorithm",
                    options=list(cls._ALGORITHMS),
                    default="adaptive",
                    tooltip=(
                        "classic: key-channel limiter; balanced: luma-aware compensation; "
                        "adaptive: edge-focused despill; hue_preserve: hue-protective compensation."
                    ),
                ),
                IO.Float.Input(
                    "strength",
                    default=1.3,
                    min=0.0,
                    max=2.5,
                    step=0.01,
                    display_mode=IO.NumberDisplay.slider,
                    tooltip="Overall despill intensity.",
                ),
                IO.Float.Input(
                    "spill_threshold",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.001,
                    display_mode=IO.NumberDisplay.slider,
                    tooltip="Minimum key-channel excess before despill starts.",
                ),
                IO.Float.Input(
                    "spill_softness",
                    default=0.001,
                    min=0.001,
                    max=1.0,
                    step=0.001,
                    display_mode=IO.NumberDisplay.slider,
                    tooltip="Soft rolloff for spill detection around threshold.",
                ),
                IO.Float.Input(
                    "compensation",
                    default=1.0,
                    min=0.0,
                    max=2.0,
                    step=0.01,
                    advanced=True,
                    tooltip="How much removed spill is redistributed to non-key channels.",
                ),
                IO.Boolean.Input(
                    "preserve_luma",
                    default=True,
                    advanced=True,
                    tooltip="Preserve perceived brightness when compensating removed spill.",
                ),
                IO.Boolean.Input(
                    "use_input_alpha_for_edges",
                    default=False,
                    advanced=True,
                    tooltip="Use source alpha as an edge guide for adaptive despill.",
                ),
                IO.Float.Input(
                    "edge_boost",
                    default=1.0,
                    min=0.0,
                    max=2.0,
                    step=0.01,
                    advanced=True,
                    tooltip="Extra despill intensity near semi-transparent edges (adaptive algorithm).",
                ),
                IO.Float.Input(
                    "edge_blur",
                    default=0.0,
                    min=0.0,
                    max=8.0,
                    step=0.1,
                    advanced=True,
                    tooltip="Blur radius for edge guide mask stability.",
                ),
                IO.Float.Input(
                    "skin_protection",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    advanced=True,
                    tooltip="Protect probable skin tones from over-despill.",
                ),
                IO.Float.Input(
                    "saturation_restore",
                    default=0.35,
                    min=0.0,
                    max=2.0,
                    step=0.01,
                    advanced=True,
                    tooltip="Restore lost saturation in hue_preserve algorithm.",
                ),
                IO.Mask.Input(
                    "spill_mask",
                    optional=True,
                    tooltip="Optional mask where 1.0 means despill is allowed, 0.0 means protected.",
                ),
                IO.Boolean.Input(
                    "invert_spill_mask",
                    default=False,
                    advanced=True,
                    tooltip="Invert optional spill_mask before use.",
                ),
            ],
            outputs=[
                IO.Image.Output(display_name="image"),
                IO.Mask.Output(display_name="spill_mask"),
                IO.Image.Output(display_name="removed_spill"),
            ],
            search_aliases=[
                "despill",
                "green spill",
                "blue spill",
                "chroma spill",
                "edge despill",
            ],
        )

    @classmethod
    def validate_inputs(
        cls,
        screen_color: str,
        algorithm: str,
        spill_threshold: float,
        spill_softness: float,
        **kwargs,
    ) -> bool | str:
        if screen_color not in cls._CHANNEL_TO_INDEX:
            return "screen_color must be one of: red, green, blue."
        if algorithm not in cls._ALGORITHMS:
            return f"algorithm must be one of: {', '.join(cls._ALGORITHMS)}."
        if spill_threshold < 0.0 or spill_threshold > 1.0:
            return "spill_threshold must be in [0..1]."
        if spill_softness <= 0.0:
            return "spill_softness must be > 0."
        return True

    @classmethod
    def _prepare_image(cls, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, bool]:
        if not isinstance(image, torch.Tensor):
            raise ValueError("TS Despill expects IMAGE tensor input.")
        if image.ndim != 4 or image.shape[-1] not in (3, 4):
            raise ValueError("TS Despill expects IMAGE tensor with shape [B,H,W,3] or [B,H,W,4].")

        image = image.detach().float().clamp(0.0, 1.0)
        has_alpha = image.shape[-1] == 4
        rgb = image[..., :3]
        if has_alpha:
            alpha = image[..., 3]
        else:
            alpha = torch.ones(
                (image.shape[0], image.shape[1], image.shape[2]),
                dtype=image.dtype,
                device=image.device,
            )
        return rgb, alpha, has_alpha

    @classmethod
    def _prepare_optional_mask(
        cls,
        spill_mask: torch.Tensor | None,
        batch: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        invert: bool,
    ) -> torch.Tensor:
        if spill_mask is None:
            mask = torch.ones((batch, height, width), dtype=dtype, device=device)
        else:
            if not isinstance(spill_mask, torch.Tensor):
                raise ValueError("spill_mask must be a MASK tensor.")

            if spill_mask.ndim == 4 and spill_mask.shape[-1] == 1:
                spill_mask = spill_mask[..., 0]
            elif spill_mask.ndim == 2:
                spill_mask = spill_mask.unsqueeze(0)
            elif spill_mask.ndim != 3:
                raise ValueError("spill_mask must have shape [B,H,W], [H,W], or [B,H,W,1].")

            spill_mask = spill_mask.detach().float().to(device=device)

            if spill_mask.shape[0] == 1 and batch > 1:
                spill_mask = spill_mask.repeat(batch, 1, 1)
            elif spill_mask.shape[0] != batch:
                raise ValueError("spill_mask batch must match image batch or be 1.")

            if spill_mask.shape[1] != height or spill_mask.shape[2] != width:
                spill_mask = F.interpolate(
                    spill_mask.unsqueeze(1),
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)

            mask = torch.clamp(spill_mask, 0.0, 1.0).to(dtype=dtype)

        if invert:
            mask = 1.0 - mask
        return torch.clamp(mask, 0.0, 1.0)

    @classmethod
    def _smoothstep(cls, value: torch.Tensor, edge0: float, edge1: float) -> torch.Tensor:
        den = max(float(edge1 - edge0), 1e-6)
        t = torch.clamp((value - edge0) / den, 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    @classmethod
    def _compute_skin_mask(cls, rgb: torch.Tensor) -> torch.Tensor:
        r = rgb[..., 0]
        g = rgb[..., 1]
        b = rgb[..., 2]

        # YCbCr skin region approximation, robust enough for spill protection.
        cb = (-0.168736 * r) - (0.331264 * g) + (0.5 * b) + 0.5
        cr = (0.5 * r) - (0.418688 * g) - (0.081312 * b) + 0.5

        skin = (
            (cb >= 0.33)
            & (cb <= 0.63)
            & (cr >= 0.37)
            & (cr <= 0.73)
            & (r >= 0.2)
        )
        return skin.float()

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        enable: bool = True,
        screen_color: str = "green",
        algorithm: str = "adaptive",
        strength: float = 1.3,
        spill_threshold: float = 0.0,
        spill_softness: float = 0.001,
        compensation: float = 1.0,
        preserve_luma: bool = True,
        use_input_alpha_for_edges: bool = False,
        edge_boost: float = 1.0,
        edge_blur: float = 0.0,
        skin_protection: float = 0.5,
        saturation_restore: float = 0.35,
        spill_mask: torch.Tensor | None = None,
        invert_spill_mask: bool = False,
    ) -> IO.NodeOutput:
        rgb, input_alpha, has_alpha = cls._prepare_image(image)

        batch, height, width, _ = rgb.shape
        dtype = rgb.dtype
        device = rgb.device

        if not enable:
            passthrough = image if has_alpha else rgb
            zero_mask = torch.zeros((batch, height, width), dtype=dtype, device=device)
            zero_spill = torch.zeros_like(rgb)
            return IO.NodeOutput(
                passthrough,
                zero_mask,
                zero_spill,
                ui=UI.PreviewImage(passthrough, cls=cls),
            )

        key_idx = cls._CHANNEL_TO_INDEX[screen_color]
        other_idx = [idx for idx in (0, 1, 2) if idx != key_idx]
        o1, o2 = other_idx[0], other_idx[1]

        user_mask = cls._prepare_optional_mask(
            spill_mask=spill_mask,
            batch=batch,
            height=height,
            width=width,
            dtype=dtype,
            device=device,
            invert=bool(invert_spill_mask),
        )

        skin_guard = 1.0
        if skin_protection > 0.0:
            skin_mask = cls._compute_skin_mask(rgb)
            skin_guard = 1.0 - float(skin_protection) * skin_mask

        key_channel = rgb[..., key_idx]
        other_1 = rgb[..., o1]
        other_2 = rgb[..., o2]
        other_max = torch.maximum(other_1, other_2)

        excess = torch.clamp(key_channel - other_max, min=0.0)
        activation = cls._smoothstep(
            excess,
            edge0=float(spill_threshold),
            edge1=float(spill_threshold + spill_softness),
        )

        weight = activation * user_mask * skin_guard

        if algorithm == "adaptive" and use_input_alpha_for_edges:
            edge_weight = 1.0 - input_alpha
            if edge_blur > 0.0:
                edge_weight = gaussian_blur_4d(edge_weight.unsqueeze(1), float(edge_blur)).squeeze(1)
            weight = weight * (1.0 + float(edge_boost) * torch.clamp(edge_weight, 0.0, 1.0))

        weight = torch.clamp(weight, 0.0, 2.0)

        reduction = torch.clamp(excess * float(strength) * weight, min=0.0)
        reduction = torch.minimum(reduction, key_channel)

        out = rgb.clone()
        out[..., key_idx] = torch.clamp(key_channel - reduction, 0.0, 1.0)

        luma_coeff = rgb.new_tensor([0.2126, 0.7152, 0.0722])

        if algorithm in ("balanced", "adaptive"):
            comp = reduction * float(compensation)
            if preserve_luma:
                key_luma = luma_coeff[key_idx]
                other_luma_sum = torch.clamp(luma_coeff[o1] + luma_coeff[o2], min=1e-6)
                add_total = comp * key_luma
                add_o1 = add_total * (luma_coeff[o1] / other_luma_sum)
                add_o2 = add_total * (luma_coeff[o2] / other_luma_sum)
            else:
                add_o1 = comp * 0.5
                add_o2 = comp * 0.5

            out[..., o1] = torch.clamp(out[..., o1] + add_o1, 0.0, 1.0)
            out[..., o2] = torch.clamp(out[..., o2] + add_o2, 0.0, 1.0)

        elif algorithm == "hue_preserve":
            comp = reduction * float(compensation)
            other_sum = torch.clamp(other_1 + other_2, min=1e-6)
            ratio_o1 = other_1 / other_sum
            ratio_o2 = 1.0 - ratio_o1

            out[..., o1] = torch.clamp(out[..., o1] + comp * ratio_o1, 0.0, 1.0)
            out[..., o2] = torch.clamp(out[..., o2] + comp * ratio_o2, 0.0, 1.0)

            if saturation_restore > 0.0:
                chroma_before = rgb.amax(dim=-1) - rgb.amin(dim=-1)
                chroma_after = out.amax(dim=-1) - out.amin(dim=-1)
                chroma_loss = torch.clamp(chroma_before - chroma_after, min=0.0)
                sat_add = chroma_loss * float(saturation_restore) * torch.clamp(weight, 0.0, 1.0)

                dominant_o1 = (other_1 >= other_2).to(dtype=dtype)
                dominant_o2 = 1.0 - dominant_o1
                out[..., o1] = torch.clamp(out[..., o1] + sat_add * dominant_o1, 0.0, 1.0)
                out[..., o2] = torch.clamp(out[..., o2] + sat_add * dominant_o2, 0.0, 1.0)

        # classic: only key-channel reduction, no compensation.

        out = torch.clamp(out, 0.0, 1.0)

        if has_alpha:
            out_image = torch.cat((out, input_alpha.unsqueeze(-1)), dim=-1)
        else:
            out_image = out

        removed_spill = torch.zeros_like(out)
        removed_spill[..., key_idx] = torch.clamp(reduction, 0.0, 1.0)

        spill_mask_out = torch.clamp(weight * (excess > 0.0).to(dtype=dtype), 0.0, 1.0)

        cls._LOGGER.debug(
            "%s algorithm=%s screen=%s spill_mean=%.4f",
            cls._LOG_PREFIX,
            algorithm,
            screen_color,
            float(spill_mask_out.mean().item()),
        )

        return IO.NodeOutput(
            out_image,
            spill_mask_out,
            removed_spill,
            ui=UI.PreviewImage(out_image, cls=cls),
        )


NODE_CLASS_MAPPINGS = {"TS_Despill": TS_Despill}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_Despill": "TS Despill"}
