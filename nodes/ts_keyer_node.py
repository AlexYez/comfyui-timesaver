import logging

import torch
import torch.nn.functional as F

from comfy_api.latest import IO, UI


class TS_Keyer(IO.ComfyNode):
    _LOGGER = logging.getLogger("comfyui_timesaver.ts_keyer")
    _LOG_PREFIX = "[TS Keyer]"

    _CHANNEL_TO_INDEX = {
        "red": 0,
        "green": 1,
        "blue": 2,
    }
    _INDEX_TO_CHANNEL = ("red", "green", "blue")

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_Keyer",
            display_name="TS Keyer",
            category="TS/image",
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
    def _gaussian_blur_4d(cls, tensor_bchw: torch.Tensor, sigma: float) -> torch.Tensor:
        if sigma <= 0.0:
            return tensor_bchw

        sigma = float(sigma)
        radius = max(1, int(round(sigma * 2.5)))
        x = torch.arange(-radius, radius + 1, device=tensor_bchw.device, dtype=tensor_bchw.dtype)
        kernel_1d = torch.exp(-(x * x) / (2.0 * sigma * sigma))
        kernel_1d = kernel_1d / torch.clamp(kernel_1d.sum(), min=1e-12)

        channels = tensor_bchw.shape[1]
        kernel_x = kernel_1d.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
        kernel_y = kernel_1d.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)

        # Replicate padding prevents dark halos on borders after blur.
        out = F.pad(tensor_bchw, (radius, radius, 0, 0), mode="replicate")
        out = F.conv2d(out, kernel_x, groups=channels)
        out = F.pad(out, (0, 0, radius, radius), mode="replicate")
        out = F.conv2d(out, kernel_y, groups=channels)
        return out

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
            matte_source = cls._gaussian_blur_4d(
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
            alpha = cls._gaussian_blur_4d(alpha.unsqueeze(1), edge_softness).squeeze(1)

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

        # Preserve existing alpha from upstream RGBA images.
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


class TS_Despill(IO.ComfyNode):
    _LOGGER = logging.getLogger("comfyui_timesaver.ts_despill")
    _LOG_PREFIX = "[TS Despill]"

    _CHANNEL_TO_INDEX = {
        "red": 0,
        "green": 1,
        "blue": 2,
    }

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
            category="TS/image",
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
    def _gaussian_blur_4d(cls, tensor_bchw: torch.Tensor, sigma: float) -> torch.Tensor:
        if sigma <= 0.0:
            return tensor_bchw

        sigma = float(sigma)
        radius = max(1, int(round(sigma * 2.5)))
        x = torch.arange(-radius, radius + 1, device=tensor_bchw.device, dtype=tensor_bchw.dtype)
        kernel_1d = torch.exp(-(x * x) / (2.0 * sigma * sigma))
        kernel_1d = kernel_1d / torch.clamp(kernel_1d.sum(), min=1e-12)

        channels = tensor_bchw.shape[1]
        kernel_x = kernel_1d.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
        kernel_y = kernel_1d.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)

        out = F.pad(tensor_bchw, (radius, radius, 0, 0), mode="replicate")
        out = F.conv2d(out, kernel_x, groups=channels)
        out = F.pad(out, (0, 0, radius, radius), mode="replicate")
        out = F.conv2d(out, kernel_y, groups=channels)
        return out

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
                edge_weight = cls._gaussian_blur_4d(edge_weight.unsqueeze(1), float(edge_blur)).squeeze(1)
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


NODE_CLASS_MAPPINGS = {
    "TS_Keyer": TS_Keyer,
    "TS_Despill": TS_Despill,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TS_Keyer": "TS Keyer",
    "TS_Despill": "TS Despill",
}
