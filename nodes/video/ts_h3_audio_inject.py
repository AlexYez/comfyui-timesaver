"""TS H3 Audio Inject — pin a MiniMax H3 clip's soundtrack to a given audio.

MiniMax H3 denoises video and stereo audio as ONE latent (a nested pair:
video [B,24,T,H/16,W/16] + audio [B,32,2,T*40]). Because both streams sit in the
same packed sequence under full self-attention, holding the audio rows at a known
soundtrack forces the video rows to agree with it — which is what makes a face
lip-sync to the supplied speech.

This node only prepares the latent:

* the soundtrack is encoded with the audio VAE into the audio stream, and
* a nested denoise mask marks the video stream free (1) and the audio stream
  preserved (0).

A stock KSampler then does the injection through ComfyUI's own masked-sampling
path: `MiniMaxH3.scale_latent_inpaint` splits the nested latent, converts the
audio between the two flow schedules (sigma_shift video 12.0 / audio 3.0) and
re-injects it at `AUDIO_COND_TIMESTEP` (1.0), i.e. clean at every step. No model
patching and no sampler wrapper — the core already carries all of it.

⚠️ The audio latent is stored UNSCALED. `MiniMaxH3.process_latent_in` multiplies
the audio stream by `audio_scale` (shift / audio_shift) on its way into
sampling, and `scale_latent_inpaint` undoes exactly that factor. Pre-scaling here
would double it and the model would see a soundtrack 4x too loud in latent space.
"""

from __future__ import annotations

import logging

import comfy.model_management
import comfy.nested_tensor
import torch
from comfy_api.v0_0_2 import IO

logger = logging.getLogger("comfyui_timesaver.ts_h3_audio_inject")
LOG_PREFIX = "[TS H3 Audio Inject]"

# The H3 audio VAE runs at 32 kHz with an 800-sample hop, i.e. 40 latent frames
# per second (comfy/ldm/minimax/audio_vae.py).
_AUDIO_LATENTS_PER_SECOND = 40.0
_VIDEO_CHANNELS = 24
_AUDIO_CHANNELS = 32
_AUDIO_STREAM_CHANNELS = 2
# Video token -> pixel frame mapping of the model, same tuple the core uses to
# turn a latent's temporal size back into a frame count.
_FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
_VIDEO_FPS = 24.0


def _is_h3_av_latent(samples) -> bool:
    """True for the nested (video, audio) pair MiniMax H3 samples."""
    if not getattr(samples, "is_nested", False):
        return False
    streams = samples.unbind()
    if len(streams) != 2:
        return False
    video, audio = streams
    return (video.ndim == 5 and video.shape[1] == _VIDEO_CHANNELS
            and audio.ndim == 4 and audio.shape[1] == _AUDIO_CHANNELS)


def _prepare_waveform(audio, vae_sample_rate: int) -> torch.Tensor:
    """One stereo item at the VAE's rate, shaped [1, 2, L]."""
    waveform = audio.get("waveform")
    if waveform is None:
        raise ValueError(f"{LOG_PREFIX} the audio input carries no waveform.")
    sample_rate = int(audio.get("sample_rate") or 0)
    if sample_rate <= 0:
        raise ValueError(f"{LOG_PREFIX} the audio input has no usable sample rate.")

    waveform = waveform[:1]  # the VAE encodes one item; one track drives the batch
    if sample_rate != vae_sample_rate:
        # Imported lazily: torchaudio is only needed when a resample is due.
        torchaudio = __import__("torchaudio")
        waveform = torchaudio.functional.resample(waveform, sample_rate, vae_sample_rate)

    # ⚠️ The stream is stereo. A mono voice track would otherwise encode to
    # [B, 32, 1, T] and only fail later, on a shape mismatch that reads like a
    # wrong-VAE error.
    channels = int(waveform.shape[1])
    if channels == 1:
        waveform = waveform.repeat(1, _AUDIO_STREAM_CHANNELS, 1)
    elif channels > _AUDIO_STREAM_CHANNELS:
        waveform = waveform[:, :_AUDIO_STREAM_CHANNELS]
    return waveform


def _fit_waveform(waveform: torch.Tensor, target_samples: int) -> tuple[torch.Tensor, str]:
    """Pad with real silence or trim, so the track spans the clip exactly.

    ⚠️ The padding MUST happen here, on the waveform, not on the encoded latent.
    The VAE normalises its latents (`z = (z - mean) / std`), so a zero latent is
    not silence at all — it decodes to a steady hum at the tail of the clip.
    Zero SAMPLES are silence, and the encoder turns them into whatever latent
    silence actually is.
    """
    have = int(waveform.shape[-1])
    if have == target_samples:
        return waveform, "exact"
    if have > target_samples:
        return waveform[..., :target_samples].clone(), "trimmed"
    padding = torch.zeros(
        waveform.shape[:-1] + (target_samples - have,),
        dtype=waveform.dtype,
        device=waveform.device,
    )
    return torch.cat((waveform, padding), dim=-1), "padded"


def _fit_latent_length(audio_latent: torch.Tensor, target_length: int) -> torch.Tensor:
    """Last-resort guard if the encoder rounds the latent length differently.

    The waveform is cut to an exact multiple of the hop, so this should never
    fire. When it does, the tail is repeated rather than zeroed — zeros would be
    the hum this node exists to avoid.
    """
    have = int(audio_latent.shape[-1])
    if have == target_length:
        return audio_latent
    logger.warning(
        "%s the encoder returned %d latent frames for a %d-frame slot; adjusting.",
        LOG_PREFIX, have, target_length,
    )
    if have > target_length:
        return audio_latent[..., :target_length].clone()
    tail = audio_latent[..., -1:].repeat(1, 1, 1, target_length - have)
    return torch.cat((audio_latent, tail), dim=-1)


class TS_H3AudioInject(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_H3AudioInject",
            display_name="TS H3 Audio Inject",
            category="TS/Video",
            essentials_category="Video",
            description=(
                "Pin a MiniMax H3 clip's soundtrack to a given audio so the video lip-syncs to it. "
                "Feeds a stock KSampler; the decoded clip carries exactly this audio."
            ),
            inputs=[
                IO.Latent.Input(
                    "latent",
                    tooltip="MiniMax H3 audio-video latent, from Empty MiniMax H3 Latent AV or MiniMax H3 Image to Video.",
                ),
                IO.Audio.Input(
                    "audio",
                    tooltip="Soundtrack to lock the clip to. Shorter audio is padded with silence, longer audio is trimmed.",
                ),
                IO.Vae.Input(
                    "audio_vae",
                    tooltip="MiniMax H3 audio VAE (the same one the decode side uses).",
                ),
            ],
            outputs=[
                IO.Latent.Output(
                    display_name="latent",
                    tooltip="The same latent with the soundtrack written in and its audio stream marked preserved.",
                )
            ],
            search_aliases=["h3 lipsync", "lip sync", "audio injection", "minimax h3 audio"],
        )

    @classmethod
    def execute(cls, latent, audio, audio_vae) -> IO.NodeOutput:
        samples = latent.get("samples")
        if samples is None or not _is_h3_av_latent(samples):
            raise ValueError(
                f"{LOG_PREFIX} expects a MiniMax H3 audio-video latent "
                "(a nested video+audio pair). Connect Empty MiniMax H3 Latent AV "
                "or MiniMax H3 Image to Video."
            )

        video, audio_stream = samples.unbind()
        target_length = int(audio_stream.shape[-1])
        batch = int(audio_stream.shape[0])

        vae_sample_rate = int(getattr(audio_vae, "audio_sample_rate", 32000))
        samples_per_latent = max(1, int(round(vae_sample_rate / _AUDIO_LATENTS_PER_SECOND)))
        waveform = _prepare_waveform(audio, vae_sample_rate)
        waveform, fit = _fit_waveform(waveform, target_length * samples_per_latent)

        # [1, C, L] -> [1, L, C]: the VAE wrapper takes channels last.
        encoded = audio_vae.encode(waveform.movedim(1, -1))
        encoded = encoded.to(device=audio_stream.device, dtype=audio_stream.dtype)
        encoded = _fit_latent_length(encoded, target_length)

        if encoded.shape[0] != batch:
            # One soundtrack drives every item of the batch.
            encoded = encoded[:1].expand(batch, *encoded.shape[1:]).contiguous()
        if encoded.shape[1:] != audio_stream.shape[1:]:
            raise ValueError(
                f"{LOG_PREFIX} the encoded soundtrack is {tuple(encoded.shape)}, "
                f"but the latent's audio stream is {tuple(audio_stream.shape)}. "
                "Check that audio_vae is the MiniMax H3 audio VAE."
            )

        # 1 = the sampler is free to denoise, 0 = keep what the latent holds.
        # The mask mirrors the latent's own nesting; the sampler unbinds it the
        # same way (comfy/samplers.py, `denoise_mask.is_nested`).
        mask = comfy.nested_tensor.NestedTensor((
            torch.ones_like(video),
            torch.zeros_like(audio_stream),
        ))

        out = latent.copy()
        out["samples"] = comfy.nested_tensor.NestedTensor((video, encoded))
        out["noise_mask"] = mask
        # A batch index would no longer address this rebuilt pair.
        out.pop("batch_index", None)

        # The two streams are sampled on one timeline but on different grids:
        # video at 24 fps, audio at 40 latent frames per second, and the audio
        # count is round(frames / 24 * 40). They line up exactly only when the
        # frame count divides by 3; otherwise the audio track ends up to 1/120 s
        # (a fifth of a frame) longer or shorter. Reported so an edit that has to
        # match the original soundtrack can be checked rather than assumed.
        frame_count = sum(_FRAME_PER_TOKEN[k % len(_FRAME_PER_TOKEN)] for k in range(int(video.shape[2])))
        audio_seconds = target_length / _AUDIO_LATENTS_PER_SECOND
        video_seconds = frame_count / _VIDEO_FPS
        logger.info(
            "%s locked %.3f s of audio (%d latent frames, %s) onto %d frames = %.3f s; "
            "audio-video offset %+.1f ms.",
            LOG_PREFIX, audio_seconds, target_length, fit, frame_count, video_seconds,
            (audio_seconds - video_seconds) * 1000.0,
        )
        return IO.NodeOutput(out)


NODE_CLASS_MAPPINGS = {"TS_H3AudioInject": TS_H3AudioInject}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_H3AudioInject": "TS H3 Audio Inject"}
