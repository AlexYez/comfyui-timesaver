"""TS Keyer — color-difference chroma keyer for green/blue/red screens.

node_id: TS_Keyer
"""

import logging

import torch

from comfy_api.latest import IO, UI

from ._keying_helpers import CHANNEL_TO_INDEX, INDEX_TO_CHANNEL, gaussian_blur_4d


class TS_Keyer(IO.ComfyNode):
    _LOGGER = logging.getLogger("comfyui_timesaver.ts_keyer")
    _LOG_PREFIX = "[TS Keyer]"

    _CHANNEL_TO_INDEX = CHANNEL_TO_INDEX
    _INDEX_TO_CHANNEL = INDEX_TO_CHANNEL

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_Keyer",
            display_name="TS Keyer",
            category="TS/Image",
            description=(
                "Advanced Color Difference keyer for green/blue screen. "
                "Preserves semi-transparency (hair/smoke), provides despill, and returns RGBA foreground + alpha mask."
            ),
            inputs=[
                IO.Image.Input(
                    "image",
                    tooltip="Input IMAGE tensor [B,H,W,3] in range [0..1].",
                ),
                IO.Boolean.Input(
                    "enable",
                    default=True,
                    tooltip="If disabled, node passes input through and returns full alpha mask.",
                ),
                IO.Color.Input(
                    "key_color",
                    default="#00ff00",
                    tooltip="Target screen color. COLOR widget supports precise eyedropper picking.",
                ),
                IO.Combo.Input(
                    "key_channel",
                    options=["auto", "green", "blue", "red"],
                    default="auto",
                    tooltip=(
                        "Primary key channel. 'auto' selects dominant channel from key_color "
                        "(green/blue/red screen)."
                    ),
                ),
                IO.Float.Input(
                    "screen_balance",
                    default=1.0,
                    min=0.0,
                    max=2.0,
                    step=0.01,
                    tooltip="Weight for non-key channels in color-difference matte extraction.",
                ),
                IO.Float.Input(
                    "key_strength",
                    default=1.0,
                    min=0.0,
                    max=4.0,
                    step=0.01,
                    tooltip="Linear key strength multiplier before levels.",
                ),
                IO.Float.Input(
                    "black_point",
                    default=0.05,
                    min=0.0,
                    max=0.99,
                    step=0.001,
                    display_mode=IO.NumberDisplay.slider,
                    tooltip="Alpha black point. Higher value makes background more transparent.",
                ),
                IO.Float.Input(
                    "white_point",
                    default=0.85,
                    min=0.01,
                    max=1.0,
                    step=0.001,
                    display_mode=IO.NumberDisplay.slider,
                    tooltip="Alpha white point. Lower value makes foreground denser.",
                ),
                IO.Float.Input(
                    "matte_gamma",
                    default=1.0,
                    min=0.1,
                    max=3.0,
                    step=0.01,
                    tooltip="Gamma shaping for semi-transparent matte details.",
                ),
                IO.Float.Input(
                    "matte_preblur",
                    default=0.0,
                    min=0.0,
                    max=8.0,
                    step=0.1,
                    advanced=True,
                    tooltip="Gaussian preblur (sigma) used only for matte extraction stability.",
                ),
                IO.Float.Input(
                    "edge_softness",
                    default=0.0,
                    min=0.0,
                    max=12.0,
                    step=0.1,
                    advanced=True,
                    tooltip="Gaussian blur (sigma) applied to final alpha edges.",
                ),
                IO.Float.Input(
                    "despill_strength",
                    default=1.0,
                    min=0.0,
                    max=2.5,
                    step=0.01,
                    tooltip="Suppression of key-channel spill on foreground edges.",
                ),
                IO.Boolean.Input(
                    "despill_edge_only",
                    default=True,
                    advanced=True,
                    tooltip="Apply despill primarily where alpha is transparent/semtransparent.",
                ),
                IO.Boolean.Input(
                    "despill_compensate",
                    default=True,
                    advanced=True,
                    tooltip="Compensate removed key color into other channels to keep luma stable.",
                ),
                IO.Boolean.Input(
                    "invert_alpha",
                    default=False,
                    advanced=True,
                    tooltip="Invert output alpha mask.",
                ),
            ],
            outputs=[
                IO.Image.Output(display_name="foreground"),
                IO.Mask.Output(display_name="alpha"),
                IO.Image.Output(display_name="despilled_rgb"),
            ],
            search_aliases=[
                "chroma key",
                "green screen",
                "blue screen",
                "keying",
                "despill",
            ],
        )

    @classmethod
    def validate_inputs(
        cls,
        key_color: str,
        key_channel: str,
        black_point: float,
        white_point: float,
        matte_gamma: float,
        **kwargs,
    ) -> bool | str:
        try:
            cls._parse_key_color(key_color)
        except ValueError as exc:
            return str(exc)

        if key_channel not in ("auto", "green", "blue", "red"):
            return "key_channel must be one of: auto, green, blue, red."

        if white_point <= black_point:
            return "white_point must be greater than black_point."

        if matte_gamma <= 0.0:
            return "matte_gamma must be > 0."

        return True

    @classmethod
    def _parse_key_color(cls, key_color: str) -> tuple[float, float, float]:
        if not isinstance(key_color, str):
            raise ValueError("key_color must be a string in #RRGGBB or #RGB format.")

        color = key_color.strip().lower()
        if color.startswith("#"):
            color = color[1:]

        if len(color) == 3:
            color = "".join(ch * 2 for ch in color)

        if len(color) != 6:
            raise ValueError("Invalid key_color format. Expected #RRGGBB or #RGB.")

        try:
            r = int(color[0:2], 16) / 255.0
            g = int(color[2:4], 16) / 255.0
            b = int(color[4:6], 16) / 255.0
        except Exception as exc:
            raise ValueError("Invalid key_color format. Expected hexadecimal color.") from exc

        return (r, g, b)

    @classmethod
    def _resolve_key_channel(cls, key_channel: str, key_rgb: torch.Tensor) -> int:
        if key_channel == "auto":
            return int(torch.argmax(key_rgb).item())
        return cls._CHANNEL_TO_INDEX[key_channel]

    @classmethod
    def _validate_and_prepare_image(cls, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(image, torch.Tensor):
            raise ValueError("TS Keyer expects IMAGE tensor input.")
        if image.ndim != 4 or image.shape[-1] not in (3, 4):
            raise ValueError("TS Keyer expects IMAGE tensor with shape [B,H,W,3] or [B,H,W,4].")

        image = image.detach().float().clamp(0.0, 1.0)
        rgb = image[..., :3]
        if image.shape[-1] == 4:
            input_alpha = image[..., 3]
        else:
            input_alpha = torch.ones(
                (image.shape[0], image.shape[1], image.shape[2]),
                dtype=image.dtype,
                device=image.device,
            )
        return rgb, input_alpha

    @classmethod
    def _compute_alpha(
        cls,
        image: torch.Tensor,
        key_rgb: torch.Tensor,
        key_channel: str,
        screen_balance: float,
        key_strength: float,
        black_point: float,
        white_point: float,
        matte_gamma: float,
        matte_preblur: float,
        edge_softness: float,
        invert_alpha: bool,
    ) -> tuple[torch.Tensor, int]:
        key_idx = cls._resolve_key_channel(key_channel, key_rgb)
        other_idx = [idx for idx in (0, 1, 2) if idx != key_idx]

        matte_source = image
        if matte_preblur > 0.0:
            matte_source = gaussian_blur_4d(
                image.permute(0, 3, 1, 2),
                matte_preblur,
            ).permute(0, 2, 3, 1)

        key_channel_data = matte_source[..., key_idx]
        other_max = torch.maximum(
            matte_source[..., other_idx[0]],
            matte_source[..., other_idx[1]],
        )

        key_primary = key_rgb[key_idx]
        key_other_max = torch.maximum(key_rgb[other_idx[0]], key_rgb[other_idx[1]])
        key_delta_ref = torch.clamp(key_primary - screen_balance * key_other_max, min=0.02)

        delta = key_channel_data - screen_balance * other_max
        delta_norm = delta / key_delta_ref

        alpha_raw = 1.0 - key_strength * delta_norm
        levels_denominator = max(float(white_point - black_point), 1e-6)
        alpha = (alpha_raw - black_point) / levels_denominator
        alpha = torch.clamp(alpha, 0.0, 1.0)

        if matte_gamma != 1.0:
            alpha = torch.pow(alpha, matte_gamma)

        if edge_softness > 0.0:
            alpha = gaussian_blur_4d(alpha.unsqueeze(1), edge_softness).squeeze(1)

        alpha = torch.clamp(alpha, 0.0, 1.0)
        if invert_alpha:
            alpha = 1.0 - alpha

        return alpha, key_idx

    @classmethod
    def _apply_despill(
        cls,
        image: torch.Tensor,
        alpha: torch.Tensor,
        key_idx: int,
        despill_strength: float,
        despill_edge_only: bool,
        despill_compensate: bool,
    ) -> torch.Tensor:
        if despill_strength <= 0.0:
            return image

        out = image.clone()
        other_idx = [idx for idx in (0, 1, 2) if idx != key_idx]

        key_channel = out[..., key_idx]
        other_max = torch.maximum(out[..., other_idx[0]], out[..., other_idx[1]])
        excess = torch.clamp(key_channel - other_max, min=0.0)

        if despill_edge_only:
            spill_weight = 1.0 - alpha
        else:
            spill_weight = torch.ones_like(alpha)

        reduction = excess * despill_strength * spill_weight
        out[..., key_idx] = torch.clamp(key_channel - reduction, 0.0, 1.0)

        if despill_compensate:
            compensation = 0.5 * reduction
            out[..., other_idx[0]] = torch.clamp(out[..., other_idx[0]] + compensation, 0.0, 1.0)
            out[..., other_idx[1]] = torch.clamp(out[..., other_idx[1]] + compensation, 0.0, 1.0)

        return out

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        enable: bool = True,
        key_color: str = "#00ff00",
        key_channel: str = "auto",
        screen_balance: float = 1.0,
        key_strength: float = 1.0,
        black_point: float = 0.05,
        white_point: float = 0.85,
        matte_gamma: float = 1.0,
        matte_preblur: float = 0.0,
        edge_softness: float = 0.0,
        despill_strength: float = 1.0,
        despill_edge_only: bool = True,
        despill_compensate: bool = True,
        invert_alpha: bool = False,
    ) -> IO.NodeOutput:
        image_rgb, input_alpha = cls._validate_and_prepare_image(image)

        if not enable:
            alpha_opacity = torch.ones(
                (image_rgb.shape[0], image_rgb.shape[1], image_rgb.shape[2]),
                dtype=image_rgb.dtype,
                device=image_rgb.device,
            )
            foreground = torch.cat((image_rgb, alpha_opacity.unsqueeze(-1)), dim=-1)
            alpha_mask = 1.0 - alpha_opacity
            return IO.NodeOutput(
                foreground,
                alpha_mask,
                image_rgb,
                ui=UI.PreviewImage(foreground, cls=cls),
            )

        key_rgb_tuple = cls._parse_key_color(key_color)
        key_rgb = torch.tensor(key_rgb_tuple, dtype=image_rgb.dtype, device=image_rgb.device)

        alpha, key_idx = cls._compute_alpha(
            image=image_rgb,
            key_rgb=key_rgb,
            key_channel=key_channel,
            screen_balance=float(screen_balance),
            key_strength=float(key_strength),
            black_point=float(black_point),
            white_point=float(white_point),
            matte_gamma=float(matte_gamma),
            matte_preblur=float(matte_preblur),
            edge_softness=float(edge_softness),
            invert_alpha=bool(invert_alpha),
        )

        alpha_opacity = torch.clamp(alpha * input_alpha, 0.0, 1.0)

        despilled = cls._apply_despill(
            image=image_rgb,
            alpha=alpha_opacity,
            key_idx=key_idx,
            despill_strength=float(despill_strength),
            despill_edge_only=bool(despill_edge_only),
            despill_compensate=bool(despill_compensate),
        )

        foreground_rgb = torch.clamp(despilled * alpha_opacity.unsqueeze(-1), 0.0, 1.0)
        foreground = torch.cat((foreground_rgb, alpha_opacity.unsqueeze(-1)), dim=-1)
        alpha_mask = 1.0 - alpha_opacity

        cls._LOGGER.debug(
            "%s channel=%s key_color=%s alpha_min=%.4f alpha_max=%.4f",
            cls._LOG_PREFIX,
            cls._INDEX_TO_CHANNEL[key_idx],
            key_color,
            float(alpha_opacity.amin().item()),
            float(alpha_opacity.amax().item()),
        )

        return IO.NodeOutput(
            foreground,
            alpha_mask,
            despilled,
            ui=UI.PreviewImage(foreground, cls=cls),
        )


NODE_CLASS_MAPPINGS = {"TS_Keyer": TS_Keyer}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_Keyer": "TS Keyer"}
