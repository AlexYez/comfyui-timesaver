"""LTX keyframe conditioning for the first and, optionally, the last frame.

⚠️ There are TWO native ways to pin a frame in LTX, and they are not
interchangeable — the official LTX 2.5 templates use one each:

* **In place** (`video_ltx2_5_i2v.json` → ``LTXVImgToVideoInplace``): the encoded
  image is written INTO the first latent frame and the noise mask holds it
  there. The latent keeps its length, and nothing has to be cleaned up
  afterwards. This is the whole of first-frame-to-video.
* **As keyframe tokens** (`video_ltx2_5_flf2v.json` → two ``LTXVAddGuide`` plus
  ``LTXVCropGuides``): the guide is APPENDED to the latent as extra frames with
  their own pixel coordinates, which is how a frame can be pinned anywhere in
  time — including at the end.

⚠️ Приложенные кадры-подсказки обязан снять `LTXVCropGuides` МЕЖДУ сэмплером и
декодером. Иначе декодер превращает их в лишние кадры В КОНЦЕ ролика — и это
ровно та жалоба, с которой нода сюда попала (17.09.2026): «генерирую по одному
первому кадру, а в конце несколько кадров с артефактами». Артефактными они
выглядят потому, что при strength < 1 сэмплер их ещё и зашумляет.

Поэтому первый кадр БЕЗ последнего пинится на месте: ничего не добавляется в
латент, и убирать нечего.
"""

import logging

import torch
from comfy_api.v0_0_2 import IO
from comfy_extras.nodes_lt import LTXVAddGuide

logger = logging.getLogger("comfyui_timesaver.ts_ltx_first_last_frame")
LOG_PREFIX = "[TS LTX First/Last Frame]"


class TS_LTX_FirstLastFrame(IO.ComfyNode):
    """
    Apply native LTX guide conditioning for the first and optional last frame.

    First frame alone is pinned in place (``LTXVImgToVideoInplace``); a last
    frame — alone or with a first — goes through ``LTXVAddGuide``, exactly as
    ComfyUI's own LTX 2.5 templates do it.
    """

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_LTX_FirstLastFrame",
            display_name="TS LTX First/Last Frame",
            category="TS/Video",
            description=(
                "Pin the first and, optionally, the last frame of an LTX video in one "
                "node. A first frame alone is written into the latent itself, so nothing "
                "has to be cropped afterwards; a last frame needs LTX keyframe guides, "
                "and those must be removed by an LTXV Crop Guides node placed between "
                "the sampler and the decoder."
            ),
            inputs=[
                IO.Conditioning.Input(
                    "positive",
                    tooltip="Positive conditioning that receives the LTX guide for the supplied frames.",
                ),
                IO.Conditioning.Input(
                    "negative",
                    tooltip="Negative conditioning that receives the LTX guide for the supplied frames.",
                ),
                IO.Vae.Input(
                    "vae",
                    tooltip="VAE used to encode the guide frames into the video latent.",
                ),
                IO.Latent.Input(
                    "latent",
                    tooltip="Video latent to guide. Passed through and returned with the frame guides applied.",
                ),
                IO.Float.Input(
                    "first_strength",
                    default=0.7,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Guide strength for the first frame. 0 disables the first-frame guide.",
                ),
                IO.Float.Input(
                    "last_strength",
                    default=0.7,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Guide strength for the last frame. 0 disables the last-frame guide.",
                ),
                IO.Image.Input(
                    "first_image",
                    optional=True,
                    tooltip=(
                        "Image pinned to the first frame. On its own it is written into "
                        "the latent in place — the video keeps its length and needs no "
                        "cropping afterwards."
                    ),
                ),
                IO.Image.Input(
                    "last_image",
                    optional=True,
                    tooltip=(
                        "Image pinned to the last frame. It can only be pinned as an LTX "
                        "keyframe guide, which is appended to the latent — put an LTXV "
                        "Crop Guides node between the sampler and the decoder, or those "
                        "guides come out as extra frames at the end of the video."
                    ),
                ),
            ],
            outputs=[
                IO.Conditioning.Output(
                    display_name="positive",
                    tooltip="Positive conditioning with the frame guides applied.",
                ),
                IO.Conditioning.Output(
                    display_name="negative",
                    tooltip="Negative conditioning with the frame guides applied.",
                ),
                IO.Latent.Output(
                    display_name="latent",
                    tooltip="Video latent with the first/last frame guides applied.",
                ),
            ],
        )

    @staticmethod
    def _is_valid_image(value) -> bool:
        return value is not None and isinstance(value, torch.Tensor)

    @staticmethod
    def _require_video_vae(vae) -> None:
        """Refuse the audio VAE before the LTX code trips over it.

        ⚠️ Реальный случай 17.09.2026: в загрузчике с заголовком «Load VAE —
        Video» был выбран `ltx-2.5-audio-vae`. У аудио-VAE нет геометрии
        латента, и ядро падало на `scale_factors[0]` с `'NoneType' object is
        not subscriptable` — по такому сообщению причину не найти.
        """
        if getattr(vae, "downscale_index_formula", None) is not None:
            return
        raise ValueError(
            f"{LOG_PREFIX} The 'vae' input is not an LTX video VAE: it has no latent "
            "geometry (downscale_index_formula). This is what an LTX audio VAE looks "
            "like — check that the VAE Loader feeding this node has the VIDEO vae "
            "selected, not the audio one."
        )

    @staticmethod
    def _log(message: str) -> None:
        logger.info("%s %s", LOG_PREFIX, message)

    @staticmethod
    def _clone_latent(latent: dict) -> dict:
        cloned = {"samples": latent["samples"].clone()}
        if "noise_mask" in latent and latent["noise_mask"] is not None:
            cloned["noise_mask"] = latent["noise_mask"].clone()
        for key, value in latent.items():
            if key in cloned:
                continue
            cloned[key] = value
        return cloned

    @staticmethod
    def _unpack_node_output(node_output):
        if hasattr(node_output, "result"):
            return node_output.result
        return node_output

    @classmethod
    def _pin_in_place(cls, vae, latent: dict, image, strength: float) -> dict:
        """Write the frame into the latent itself, the way LTX 2.5 does i2v.

        ⚠️ Ничего не добавляется в латент — значит, и снимать потом нечего.
        Это и есть лечение «лишних кадров в конце»: у пути с подсказками-токенами
        они появляются всегда, просто их обязан срезать `LTXVCropGuides`.
        """
        from comfy_extras.nodes_lt import LTXVImgToVideoInplace  # noqa: PLC0415

        return cls._unpack_node_output(
            LTXVImgToVideoInplace.execute(
                vae=vae, image=image, latent=latent, strength=strength,
            )
        )[0]

    @classmethod
    def execute(
        cls,
        positive,
        negative,
        vae,
        latent: dict,
        first_strength: float,
        last_strength: float,
        first_image: torch.Tensor = None,
        last_image: torch.Tensor = None,
    ) -> IO.NodeOutput:
        # No blanket try/except here: a failed LTXVAddGuide (VAE OOM, size
        # mismatch, upstream API change) must surface as a node error. The old
        # fallback returned the inputs untouched, silently turning a guided
        # I2V workflow into text-to-video.
        positive_out = positive
        negative_out = negative
        latent_out = cls._clone_latent(latent)

        first_on = cls._is_valid_image(first_image) and first_strength > 0.0
        last_on = cls._is_valid_image(last_image) and last_strength > 0.0

        if cls._is_valid_image(first_image) and not first_on:
            cls._log("first_strength is 0. Skipping first-frame guide.")
        if cls._is_valid_image(last_image) and not last_on:
            cls._log("last_strength is 0. Skipping last-frame guide.")

        if not first_on and not last_on:
            cls._log("Text to video (no frames)")
            return IO.NodeOutput(positive_out, negative_out, latent_out)

        cls._require_video_vae(vae)

        # ⚠️ Последний кадр можно закрепить ТОЛЬКО подсказками-токенами: в
        # латенте для него места нет — он и есть его конец. Первый же кадр без
        # последнего пинится на месте, и тогда латент не растёт.
        if not last_on:
            cls._log("First frame only — pinned in place, nothing to crop afterwards.")
            latent_out = cls._pin_in_place(vae, latent_out, first_image, first_strength)
            return IO.NodeOutput(positive_out, negative_out, latent_out)

        guides = 0
        if first_on:
            positive_out, negative_out, latent_out = cls._unpack_node_output(
                LTXVAddGuide.execute(
                    positive=positive_out,
                    negative=negative_out,
                    vae=vae,
                    latent=latent_out,
                    image=first_image,
                    frame_idx=0,
                    strength=first_strength,
                )
            )
            guides += 1

        positive_out, negative_out, latent_out = cls._unpack_node_output(
            LTXVAddGuide.execute(
                positive=positive_out,
                negative=negative_out,
                vae=vae,
                latent=latent_out,
                image=last_image,
                frame_idx=-1,
                strength=last_strength,
            )
        )
        guides += 1

        cls._log("First frame to last frame" if first_on else "Last frame only")
        # ⚠️ Сказать об этом обязательно: без `LTXVCropGuides` человек увидит
        # подсказки в хвосте ролика и подумает, что сломалась генерация.
        logger.warning(
            "%s %d guide frame(s) were appended to the latent. Put an LTXV Crop Guides "
            "node between the sampler and the decoder (conditioning from here, latent "
            "from the sampler), or the tail of the video will be those guides.",
            LOG_PREFIX, guides,
        )

        return IO.NodeOutput(positive_out, negative_out, latent_out)


NODE_CLASS_MAPPINGS = {
    "TS_LTX_FirstLastFrame": TS_LTX_FirstLastFrame,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TS_LTX_FirstLastFrame": "TS LTX First/Last Frame",
}
