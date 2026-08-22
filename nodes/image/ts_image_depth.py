"""TS Image Depth — карта глубины для одиночного кадра.

Depth Anything V2, и ничего сверх неё. Порядок действий — как в эталонной
реализации: подрезать стороны до кратности 14, прогнать модель, нормировать
каждую картинку своими min/max, вернуть к исходному размеру билинейно.

⚠️ Обвеса тут сознательно НЕТ. Шумодав, дизеринг и направляемый апскейл
придумывались под видео; на снимке направляемый апскейл вдобавок давал
двоение по контурам (жалоба владельца пака, воспроизведено). Кому нужна
косметика — она осталась в TS Video Depth.

⚠️ Счёт живёт в `nodes/_depth_core.py`, общем с TS Video Depth. Здесь только
схема — правку алгоритма делать там, иначе две ноды разъедутся.
"""

import logging

from comfy_api.v0_0_2 import IO

from .._depth_core import IMAGE_MODELS, offered, run_depth

logger = logging.getLogger("comfyui_timesaver.ts_image_depth")
LOG_PREFIX = "[TS Image Depth]"


class TS_ImageDepth(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_ImageDepth",
            display_name="TS Image Depth",
            category="TS/Image/Depth",
            essentials_category="Image",
            description=(
                "Depth map for a still, or for a batch of pictures that have nothing to do "
                "with each other. Runs Depth Anything V2 on every picture on its own, the "
                "same way the reference implementation does.\n"
                "For a video, where the map has to stay steady from frame to frame, use "
                "TS Video Depth instead."
            ),
            inputs=[
                IO.Image.Input(
                    "images",
                    tooltip=(
                        "Picture, or a batch of pictures, as IMAGE (N, H, W, 3) in 0..1. "
                        "Every picture is processed and normalized on its own — a batch here "
                        "means unrelated images, not a video."
                    ),
                ),
                IO.Combo.Input(
                    "model_filename",
                    options=offered(IMAGE_MODELS),
                    default="depth_anything_v2_vitl_fp16.safetensors",
                    tooltip=(
                        "Depth Anything V2 checkpoint, downloaded on first use to "
                        "ComfyUI/models/depthanything.\n"
                        "fp16 safetensors is half the download and loads far quicker; "
                        "measured against pure fp32 it differs by 0.02% of the depth range, "
                        "which is nothing."
                    ),
                ),
                IO.Int.Input(
                    "max_res",
                    default=-1,
                    min=-1,
                    max=8192,
                    step=14,
                    tooltip=(
                        "Longest side the picture is processed at, snapped down to a multiple "
                        "of 14 (the DINOv2 patch grid). The only quality control here.\n"
                        "• -1 (default) — native resolution, exactly what the reference "
                        "implementation does. Nothing is resampled, so nothing is lost.\n"
                        "• 1540 — noticeably quicker on large photos and still sharp; the map "
                        "comes back up to full size bilinearly.\n"
                        "• lower — when VRAM is tight. Depth Anything V2 stops gaining much "
                        "above roughly 2000 px, so raising this without limit buys little.\n"
                        "On out-of-memory the node retries at half the size, and again, "
                        "logging each step."
                    ),
                ),
                IO.Combo.Input(
                    "precision",
                    options=["fp16", "fp32"],
                    default="fp16",
                    tooltip=(
                        "Inference dtype.\n"
                        "• fp16 (default) — 2x faster, ~50% less VRAM.\n"
                        "• fp32 — measured difference against fp16 is 0.02% of the depth "
                        "range. Worth it only if you are chasing the last bit on a smooth "
                        "surface and have the VRAM to spare."
                    ),
                ),
                IO.Combo.Input(
                    "colormap",
                    options=["gray", "inferno", "viridis", "plasma", "magma", "cividis"],
                    default="gray",
                    tooltip=(
                        "Output color mapping.\n"
                        "• gray (default) — the raw normalized depth in all three channels, "
                        "which is what the reference returns. Use this when the map feeds "
                        "another node (ControlNet, 3D, and so on).\n"
                        "• inferno / viridis / plasma / magma / cividis — perceptually "
                        "uniform matplotlib colormaps, for looking at. Bilinear LUT "
                        "interpolation removes 8-bit banding."
                    ),
                ),
            ],
            outputs=[
                IO.Image.Output(
                    display_name="image",
                    tooltip=(
                        "Depth map as IMAGE (N, H, W, 3) float in 0..1 at the original "
                        "resolution. An ordinary IMAGE — feed it straight into ControlNet, "
                        "a save node, and so on."
                    ),
                ),
            ],
            search_aliases=["depth", "image depth", "depth anything", "depth anything v2"],
        )

    @classmethod
    def validate_inputs(cls, model_filename) -> bool | str:
        """⚠️ Список виджета показывает только safetensors, но в сохранённом
        графе может лежать старый `.pth`. Ядро пропускает СВОЮ проверку combo
        для входов, названных здесь, — значит такой граф откроется и посчитает.
        """
        if model_filename in IMAGE_MODELS:
            return True
        return f"{LOG_PREFIX} unknown checkpoint: {model_filename}"

    @classmethod
    def execute(
        cls,
        images,
        model_filename,
        max_res,
        precision,
        colormap,
    ) -> IO.NodeOutput:
        output = run_depth(
            images=images,
            model_filename=model_filename,
            single_image=True,
            precision=precision,
            colormap=colormap,
            max_res=-1 if max_res is None else int(max_res),
            # ⚠️ Всё, чего у этой ноды нет во входах, — выключено, а не спрятано.
            #
            # `per_frame`: эталон нормирует КАЖДУЮ картинку своими min/max.
            # Общая пара на пачку нужна видео, чтобы карта не дышала между
            # кадрами; на пачке несвязанных снимков она же вредна — один кадр с
            # близким объектом сплющит контраст остальных.
            #
            # `edge_aware_upscale=False`: направляемый фильтр давал двоение по
            # контурам. `Linear`: эталон возвращает карту к исходному размеру
            # билинейно, и при max_res=-1 этого шага попросту нет.
            normalization_mode="per_frame",
            denoise_method="none",
            apply_median_blur=False,
            dithering_strength=0.0,
            dither_pattern="bayer",
            edge_aware_upscale=False,
            upscale_algorithm="Linear",
            log_prefix=LOG_PREFIX,
        )
        return IO.NodeOutput(output)


NODE_CLASS_MAPPINGS = {"TS_ImageDepth": TS_ImageDepth}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_ImageDepth": "TS Image Depth"}
