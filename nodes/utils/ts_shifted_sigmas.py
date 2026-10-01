"""TS Shifted Sigmas — a few-step schedule shifted for the size of the picture.

Few-step distilled LoRAs for Qwen Image 2.1 (Viggle's viggle-turbo is the one
this was written against) are trained on a handful of **raw** timesteps — the
student's nodes, e.g. ``1.0, 0.9375, 0.875, 0.75, 0.5, 0.25``. The diffusers
pipeline does not sample at those nodes directly: its
``FlowMatchEulerDiscreteScheduler`` bends them with the dynamic exponential
shift, and how far it bends depends on how many image tokens the picture has:

```
tokens = (height_px / 16) * (width_px / 16)
mu     = base_shift + (max_shift - base_shift) * (tokens - 256) / (8192 - 256)
sigma  = e^mu / (e^mu + (1 / t - 1))
```

followed by a final ``0``. None of ComfyUI's schedulers produce this: they
either space the steps themselves or apply a fixed shift that ignores the
resolution. The turbo LoRA was distilled against exactly this schedule, so this
node reproduces it number for number.

⚠️ **Where the picture size comes from.** The sampler, not the latent, decides
the final size: an *empty* latent carrying ``downscale_ratio_spacial`` (the stock
Empty Latent Image says 8) is rescaled to the model's own 16x grid before
sampling — ``comfy.sample.fix_empty_latent_channels``. A latent with content
(VAE Encode, a previous pass) is sampled at the size it has. The token count
here follows the same rule, so it matches what the model will actually see.
"""

from __future__ import annotations

import logging
import math

import torch
from comfy_api.v0_0_2 import IO

logger = logging.getLogger("comfyui_timesaver.ts_shifted_sigmas")
LOG_PREFIX = "[TS Shifted Sigmas]"

#: Узлы ученика viggle-turbo v0.2.1 (6 шагов) — умолчание их же ноды.
DEFAULT_NODES = "1.0, 0.9375, 0.875, 0.75, 0.5, 0.25"
#: Qwen Image 2.1 считает латент на сетке 1/16 картинки, один токен — одна клетка.
MODEL_GRID = 16
BASE_SEQ_LEN = 256
MAX_SEQ_LEN = 8192


def _parse_number(part: str) -> float:
    """``0.25`` or a fraction ``1 / 6`` — the way the model cards write their lists."""
    numerator, slash, denominator = part.partition("/")
    if not slash:
        return float(part)
    # Только одна черта: «1/2/3» — опечатка, а не число.
    if "/" in denominator:
        raise ValueError(part)
    bottom = float(denominator.strip())
    if bottom == 0.0:
        raise ValueError(part)
    return float(numerator.strip()) / bottom


def parse_nodes(text: str) -> list[float]:
    """Parse the comma separated raw timesteps.

    Accepts the list exactly as model cards print it: decimals or fractions
    (``1 / 6``), optionally wrapped in square brackets.

    Raises:
        ValueError: on a non-number, a value outside ``(0, 1]`` or a list that
            does not strictly decrease — each of those would give the sampler a
            schedule it cannot follow (``t = 0`` divides by zero).
    """
    cleaned = str(text or "").strip().removeprefix("[").removesuffix("]")
    parts = [part.strip() for part in cleaned.replace(";", ",").split(",")]
    parts = [part for part in parts if part]
    if not parts:
        raise ValueError(f"{LOG_PREFIX} nodes is empty: give at least one timestep, e.g. {DEFAULT_NODES}")
    values = []
    for part in parts:
        try:
            value = _parse_number(part)
        except ValueError:
            raise ValueError(f"{LOG_PREFIX} '{part}' in nodes is not a number") from None
        if not (0.0 < value <= 1.0) or not math.isfinite(value):
            raise ValueError(f"{LOG_PREFIX} every node must lie in (0, 1], got {part}")
        values.append(value)
    for previous, current in zip(values, values[1:]):
        if current >= previous:
            raise ValueError(
                f"{LOG_PREFIX} nodes must go strictly down from noise to image, "
                f"got {previous:g} then {current:g}"
            )
    return values


def image_tokens(latent: dict) -> int:
    """Image tokens the model will see for this latent, following the sampler.

    An empty latent with ``downscale_ratio_spacial`` is resized by the sampler to
    the model's own grid; any other latent is sampled at its own size.
    """
    samples = latent["samples"]
    if getattr(samples, "is_nested", False):
        # Вложенный латент (видео со звуком и т.п.) ядро не пересчитывает — и мы тоже.
        samples = samples.unbind()[0]
        return int(samples.shape[-2]) * int(samples.shape[-1])
    height, width = int(samples.shape[-2]), int(samples.shape[-1])
    ratio = latent.get("downscale_ratio_spacial")
    if ratio is not None and ratio != MODEL_GRID and not bool(torch.count_nonzero(samples)):
        # Та же арифметика, что в fix_empty_latent_channels: round(сторона * ratio / 16).
        scale = float(ratio) / MODEL_GRID
        height, width = round(height * scale), round(width * scale)
    return max(1, height) * max(1, width)


def shifted_sigmas(nodes: list[float], tokens: int, base_shift: float, max_shift: float) -> torch.Tensor:
    """The diffusers dynamic-shift schedule for ``nodes`` plus the final zero, float32."""
    mu = base_shift + (max_shift - base_shift) * (tokens - BASE_SEQ_LEN) / (MAX_SEQ_LEN - BASE_SEQ_LEN)
    shift = math.exp(mu)
    # float64 до самого конца: на 0.9375 разница в float32 уже в седьмом знаке.
    t = torch.tensor(nodes, dtype=torch.float64)
    sigmas = shift / (shift + (1.0 / t - 1.0))
    return torch.cat([sigmas, sigmas.new_zeros(1)]).float()


class TS_ShiftedSigmas(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_ShiftedSigmas",
            display_name="TS Shifted Sigmas",
            category="TS/Utils",
            description=(
                "Sigmas for few-step turbo LoRAs of Qwen Image 2.1: the raw timesteps "
                "the LoRA was distilled on, bent by the same resolution-dependent shift "
                "the diffusers pipeline uses, plus the final zero. Connect to "
                "SamplerCustom or SamplerCustomAdvanced with cfg 1 and the euler sampler."
            ),
            inputs=[
                IO.Latent.Input(
                    "latent",
                    tooltip=(
                        "The latent that goes into the sampler. Only its size is read: "
                        "a bigger picture gets a stronger shift."
                    ),
                ),
                IO.String.Input(
                    "nodes",
                    default=DEFAULT_NODES,
                    tooltip=(
                        "Raw timesteps from noise (1.0) down, comma separated; fractions "
                        "like 1/6 and the model card's square brackets are fine. The default "
                        "is viggle-turbo 6 steps (v0.2.1 and v0.3). Add or drop steps at the "
                        "noisy end only and keep 0.875, 0.75, 0.5, 0.25."
                    ),
                ),
                IO.Float.Input(
                    "base_shift",
                    default=0.5, min=0.0, max=5.0, step=0.01,
                    tooltip="Shift at 256 image tokens (256x256 px). Qwen Image 2.1: 0.5.",
                ),
                IO.Float.Input(
                    "max_shift",
                    default=0.9, min=0.0, max=5.0, step=0.01,
                    tooltip="Shift at 8192 image tokens (about 1448x1448 px). Qwen Image 2.1: 0.9.",
                ),
            ],
            outputs=[
                IO.Sigmas.Output(
                    display_name="sigmas",
                    tooltip="One sigma per node, then 0. Steps = number of nodes.",
                ),
            ],
            search_aliases=["turbo sigmas", "viggle", "qwen image 2.1 turbo", "few step",
                            "shift", "flow match sigmas", "dynamic shift"],
        )

    # ⚠️ Без **kwargs: с ними ядро считает, что нода проверяет ВСЕ входы сама, и
    # пропускает свои min/max у base_shift и max_shift (execution.py,
    # validate_has_kwargs) — max_shift 1e3 из API дал бы OverflowError в exp.
    @classmethod
    def validate_inputs(cls, nodes: str | None = None) -> bool | str:
        if nodes is None:
            return True  # приходит проводом — проверим в execute
        try:
            parse_nodes(nodes)
        except ValueError as error:
            return str(error)
        return True

    @classmethod
    def execute(cls, latent, nodes: str = DEFAULT_NODES, base_shift: float = 0.5,
                max_shift: float = 0.9) -> IO.NodeOutput:
        values = parse_nodes(nodes)
        tokens = image_tokens(latent)
        sigmas = shifted_sigmas(values, tokens, float(base_shift), float(max_shift))
        logger.debug("%s %d tokens, sigmas %s", LOG_PREFIX, tokens,
                     ", ".join(f"{s:.4f}" for s in sigmas.tolist()))
        return IO.NodeOutput(sigmas)


NODE_CLASS_MAPPINGS = {"TS_ShiftedSigmas": TS_ShiftedSigmas}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_ShiftedSigmas": "TS Shifted Sigmas"}
