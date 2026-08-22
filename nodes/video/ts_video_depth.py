"""TS Video Depth — карта глубины для последовательности кадров.

Video Depth Anything скользящим окном: соседние кадры видят друг друга, поэтому
карта не дрожит на статике. Для одиночного снимка есть отдельная нода
**TS Image Depth** — там работает Depth Anything V2, и она и быстрее, и резче.

⚠️ Весь счёт живёт в `nodes/_depth_core.py`, общем для обеих нод. Здесь только
схема: правку алгоритма делать там, иначе две ноды разъедутся.
"""

import logging

from comfy_api.v0_0_2 import IO

from .._depth_core import VIDEO_MODELS, offered, run_depth

logger = logging.getLogger("comfyui_timesaver.ts_video_depth")
LOG_PREFIX = "[TS Video Depth]"


class TS_VideoDepth(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        upscale_methods_list = ["Lanczos4", "Cubic", "Linear"]
        return IO.Schema(
            node_id="TS_VideoDepthNode",
            display_name="TS Video Depth",
            category="TS/Video",
            essentials_category="Video",
            description=(
                "Depth map for a sequence of frames, using Video Depth Anything over a "
                "sliding window so the result stays steady from frame to frame.\n"
                "For a still — or a batch of unrelated pictures — use TS Image Depth "
                "instead: it runs Depth Anything V2, which is both far quicker and sharper "
                "on a single frame."
            ),
            inputs=[
                IO.Image.Input(
                    "images",
                    tooltip=(
                        "Video frames as an IMAGE batch (N, H, W, 3) in 0..1. "
                        "Any aspect ratio; 16:9 is the model's sweet spot."
                    ),
                ),
                IO.Combo.Input(
                    "model_filename",
                    options=offered(VIDEO_MODELS),
                    default="video_depth_anything_vitl_fp16.safetensors",
                    tooltip=(
                        "Video Depth Anything checkpoint. Downloaded on first use to "
                        "ComfyUI/models/videodepthanything.\n"
                        "• vitl fp16 (~0.75 GB, default) — best quality. Recommended "
                        "for production.\n"
                        "• vits fp16 (~55 MB) — fast, less stable on fine detail.\n"
                        "Only safetensors are offered: half the download and an order of "
                        "magnitude quicker to read (0.01 s against 0.66 s, measured). A "
                        "graph that still names an old .pth keeps working — the file is "
                        "simply no longer suggested."
                    ),
                ),
                IO.Int.Input(
                    "input_size",
                    default=518,
                    min=64,
                    max=4096,
                    step=2,
                    tooltip=(
                        "Internal resolution the transformer sees (DINOv2 patch size 14, "
                        "snapped automatically). Higher = more depth detail, more VRAM / time.\n"
                        "For 16:9 source (after max_res cap):\n"
                        "• 518 (default, native) — model trained at this size; ~480 K depth pixels.\n"
                        "• 644 — ~740 K depth pixels (+54% detail, +54% VRAM/time). Safe on ≥24 GB.\n"
                        "• 700 — ~872 K depth pixels (+82% detail, +82% VRAM). OOM risk on 16 GB.\n"
                        "• ≥770 — out-of-distribution for DINOv2 (quality can REGRESS).\n"
                        "If OOM, the node auto-retries on 518 → 392 → 280 → 168."
                    ),
                ),
                IO.Int.Input(
                    "max_res",
                    default=1280,
                    min=-1,
                    max=8192,
                    step=1,
                    tooltip=(
                        "Cap on the longer side of input frames before model preprocessing.\n"
                        "• Does NOT change depth detail — the model always resamples to input_size.\n"
                        "• -1 — no cap. Keeps full-resolution RGB as the guide for "
                        "edge_aware_upscale, giving the sharpest silhouettes on the output.\n"
                        "• 1280 (default) — downscales 4K to HD first; saves preprocess RAM and "
                        "speeds up resize, at the cost of a slightly softer edge-aware upscale.\n"
                        "Recommendation: -1 for 4K when edge_aware_upscale=True, 1280 otherwise."
                    ),
                ),
                IO.Combo.Input(
                    "precision",
                    options=['fp16', 'fp32'],
                    default='fp16',
                    tooltip=(
                        "Inference dtype.\n"
                        "• fp16 (default) — 2× faster, ~50% less VRAM. Required for vitl @ 4K "
                        "on a 16 GB card.\n"
                        "• fp32 — marginally cleaner gradients on smooth surfaces; doubles "
                        "VRAM, almost always triggers OOM on 4K. Use only on small inputs or "
                        "≥24 GB cards."
                    ),
                ),
                IO.Combo.Input(
                    "colormap",
                    options=['gray', 'inferno', 'viridis', 'plasma', 'magma', 'cividis'],
                    default='gray',
                    tooltip=(
                        "Output color mapping.\n"
                        "• gray (default) — raw normalized depth in all 3 channels. Use this "
                        "if the depth map feeds downstream nodes (ControlNet, 3D, etc).\n"
                        "• inferno / viridis / plasma / magma / cividis — perceptually uniform "
                        "matplotlib colormaps for visualization only. Bilinear LUT interpolation "
                        "removes 8-bit banding."
                    ),
                ),
                IO.Float.Input(
                    "dithering_strength",
                    default=0.005,
                    min=0.0,
                    max=0.016,
                    step=0.0001,
                    round=0.0001,
                    tooltip=(
                        "Sub-LSB noise added to the normalized depth before colormap, to break "
                        "up 8-bit banding when the result is saved as PNG/JPEG.\n"
                        "• 0 — no dither.\n"
                        "• 0.005 (default) — light, OK with bayer pattern + bilinear LUT.\n"
                        "• 0.016 (max) — aggressive, guaranteed banding-free on gray output.\n"
                        "If you still see bands, raise toward 0.016 and prefer dither_pattern=bayer."
                    ),
                ),
                IO.Boolean.Input(
                    "apply_median_blur",
                    default=True,
                    tooltip=(
                        "Legacy denoise toggle (kept for workflow compatibility). Used only "
                        "when denoise_method=auto: True → 3×3 median, False → none.\n"
                        "When denoise_method is set explicitly (bilateral / median / none), "
                        "this toggle is ignored."
                    ),
                ),
                IO.Combo.Input(
                    "upscale_algorithm",
                    options=upscale_methods_list,
                    default="Lanczos4",
                    tooltip=(
                        "Resampling kernel for upscaling the depth map back to the original "
                        "frame size. Used only when edge_aware_upscale=False.\n"
                        "• Lanczos4 (default) — bicubic with antialias, the sharpest of the two.\n"
                        "• Cubic — the SAME kernel as Lanczos4. PyTorch has no Lanczos, and both "
                        "labels map to bicubic+antialias, so the two options are "
                        "pixel-for-pixel identical (verified). Kept because saved workflows "
                        "carry the value.\n"
                        "• Linear — bilinear, softer and cheaper. The only genuinely different "
                        "choice here."
                    ),
                ),
                # Optional quality controls. Defaults are tuned for visual
                # quality; the legacy widgets above (precision/apply_median_blur/
                # dithering_strength/upscale_algorithm) keep their original
                # defaults so existing workflows are bit-for-bit unaffected.
                IO.Combo.Input(
                    "normalization_mode",
                    options=["minmax", "percentile"],
                    default="percentile",
                    optional=True,
                    tooltip=(
                        "How to map raw depth onto [0..1]. "
                        "image is always normalized minmax.\n"
                        "• minmax — uses global min/max across the whole video. Simple, but "
                        "one outlier frame (object very close or far) can squash the contrast "
                        "of every other frame.\n"
                        "• percentile (default, quality) — robust 1%..99% range. Better contrast "
                        "and temporal stability on long clips. Slightly more memory (samples a "
                        "subset of pixels for quantile).\n"
                    ),
                ),
                IO.Combo.Input(
                    "denoise_method",
                    options=["auto", "none", "median", "bilateral"],
                    default="bilateral",
                    optional=True,
                    tooltip=(
                        "Spatial denoise applied at low-res depth (before upscale).\n"
                        "• auto — follow legacy apply_median_blur toggle.\n"
                        "• none — no denoise, maximum detail; may show grain on fine textures.\n"
                        "• median — 3×3 median, removes impulse noise, slightly blurs thin geometry.\n"
                        "• bilateral (default, quality) — edge-preserving 5×5; smooths surface "
                        "noise while keeping object silhouettes sharp."
                    ),
                ),
                IO.Combo.Input(
                    "dither_pattern",
                    options=["white", "bayer"],
                    default="bayer",
                    optional=True,
                    tooltip=(
                        "Dither distribution used by dithering_strength.\n"
                        "• white — TPDF (triangular) random noise, full antibanding standard, "
                        "but adds visible grain on flat surfaces.\n"
                        "• bayer (default, quality) — deterministic 8×8 ordered pattern. "
                        "Banding-free, no temporal flicker, no grain. Best paired with bilinear "
                        "colormap LUT (already enabled)."
                    ),
                ),
                IO.Boolean.Input(
                    "edge_aware_upscale",
                    default=True,
                    optional=True,
                    tooltip=(
                        "Final upscale strategy.\n"
                        "• False — plain resampling via upscale_algorithm (Lanczos4 etc). Fastest.\n"
                        "• True (default, quality) — Fast Guided Filter using the input RGB as "
                        "edge guide. Silhouettes snap to real object boundaries; thin geometry "
                        "is preserved. Costs ~5-10% extra postprocess time. Combine with "
                        "max_res=-1 to keep the guide at full 4K for the sharpest result."
                    ),
                ),
                # ⚠️ Все входы ниже ДОПИСАНЫ в конец и optional. widgets_values
                # позиционный, а граф в API-формате, сохранённый до них, не
                # несёт этих ключей вовсе — required здесь дал бы 400.
                IO.Float.Input(
                    "flicker_suppression",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    optional=True,
                    tooltip=(
                        "Blends in a temporal MEDIAN of the depth to kill "
                        "single-frame pops.\n"
                        "• 0 (default) — off, output unchanged.\n"
                        "• 0.3-0.5 — takes the twitch out of static shots.\n"
                        "• 1 — pure median; safe on a locked-off camera, can lag fast motion.\n"
                        "A median is used rather than an average on purpose: it drops "
                        "outliers without smearing real movement."
                    ),
                ),
                IO.Int.Input(
                    "flicker_radius",
                    default=1,
                    min=1,
                    max=8,
                    optional=True,
                    tooltip=(
                        "Half-width of the temporal median window, in frames "
                        "(1 = look at 3 frames, 2 = 5, and so on). Bigger is steadier and "
                        "slower to react. Ignored when flicker_suppression is 0."
                    ),
                ),
                IO.Int.Input(
                    "window_length",
                    default=32,
                    min=8,
                    max=32,
                    optional=True,
                    tooltip=(
                        "Frames the model sees at once.\n"
                        "32 is both the default and the ceiling: the temporal module carries "
                        "an absolute positional embedding with exactly 32 slots, so a longer "
                        "window cannot be built from these weights at all. Lower it only to "
                        "fit VRAM — shorter windows mean less context and less consistency."
                    ),
                ),
                IO.Int.Input(
                    "window_overlap",
                    default=10,
                    min=2,
                    max=24,
                    optional=True,
                    tooltip=(
                        "How many frames consecutive windows share. More "
                        "overlap means smoother joins and proportionally more compute: at "
                        "the default the model already runs ~1.45 frames for every frame "
                        "of output. Must stay below window_length."
                    ),
                ),
            ],
            outputs=[
                IO.Image.Output(
                    display_name="image",
                    tooltip=(
                        "Depth map as IMAGE (N, H, W, 3) float in 0..1 at the original RGB "
                        "resolution. Same as a regular IMAGE — feed directly into ControlNet, "
                        "save nodes, etc."
                    ),
                ),
            ],
            search_aliases=["depth", "video depth", "depth anything", "ts universal depth"],
        )

    @classmethod
    def validate_inputs(cls, model_filename) -> bool | str:
        """⚠️ Список виджета показывает только safetensors, но в сохранённом
        графе может лежать старый `.pth`. Ядро пропускает СВОЮ проверку combo
        для входов, названных здесь, — значит такой граф откроется и посчитает.
        """
        if model_filename in VIDEO_MODELS:
            return True
        return f"{LOG_PREFIX} unknown checkpoint: {model_filename}"

    @classmethod
    def execute(
        cls,
        images,
        model_filename,
        input_size,
        max_res,
        precision,
        colormap,
        dithering_strength,
        apply_median_blur,
        upscale_algorithm,
        normalization_mode="percentile",
        denoise_method="bilateral",
        dither_pattern="bayer",
        edge_aware_upscale=True,
        flicker_suppression=0.0,
        flicker_radius=1,
        window_length=32,
        window_overlap=10,
    ) -> IO.NodeOutput:
        # Необязательные входы, которых нет в старом графе, приходят None —
        # не пропуском. Гасим здесь, чтобы дальше об этом никто не думал.
        output = run_depth(
            images=images,
            model_filename=model_filename,
            single_image=False,
            precision=precision,
            colormap=colormap,
            max_res=max_res,
            input_size=input_size,
            normalization_mode=normalization_mode or "percentile",
            denoise_method=denoise_method or "bilateral",
            apply_median_blur=apply_median_blur,
            dithering_strength=dithering_strength,
            dither_pattern=dither_pattern or "bayer",
            edge_aware_upscale=True if edge_aware_upscale is None else bool(edge_aware_upscale),
            upscale_algorithm=upscale_algorithm,
            window_length=32 if window_length is None else int(window_length),
            window_overlap=10 if window_overlap is None else int(window_overlap),
            flicker_suppression=0.0 if flicker_suppression is None else float(flicker_suppression),
            flicker_radius=1 if flicker_radius is None else int(flicker_radius),
            log_prefix=LOG_PREFIX,
        )
        return IO.NodeOutput(output)


NODE_CLASS_MAPPINGS = {"TS_VideoDepthNode": TS_VideoDepth}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_VideoDepthNode": "TS Video Depth"}
