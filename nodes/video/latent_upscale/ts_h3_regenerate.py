"""Open-H3-Regenerate-2K: a separate refinement branch for native H3 models."""
from __future__ import annotations

import logging
import math

import folder_paths
import torch
from comfy_api.v0_0_2 import IO

from . import _h3_core as core
from . import _regenerate as regen
from . import _regenerate_ic as ic

LOG = logging.getLogger(__name__)
FIRST_FRAME = "first frame (preserve character)"
FIRST_LAST = "first and last frames"
FULL_REFERENCE = "full reference (Ref2VA)"
CONNECTED_FRAMES = "connected frames (auto)"
INDEPENDENT = "independent (crossfade)"
LOCK_OVERLAP = "continue (lock overlap)"
BLEND_OVERLAP = "continue (blend overlap)"


def _files(category, needle):
    needles = (needle,) if isinstance(needle, str) else needle
    files = folder_paths.get_filename_list(category)
    ordered = list(dict.fromkeys(p for n in needles for p in files if n in p.lower()))
    return ordered or [f"(missing {needles[0]})"]


class TS_H3RegenerateModel(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="TS_H3RegenerateModel", display_name="TS H3 Regenerate Model", category="TS/Video",
            description="Loads one self-contained Standard Regenerate bundle, including the merged LoRA, Sharp branches and reference table. Connect a clean pruned H3 model before generation LoRAs.",
            inputs=[
                IO.Model.Input("model", tooltip="Clean pruned H3 FL2VA or Ref2VA directly from its loader. Generation LoRAs stay on a separate branch."),
                IO.Combo.Input("lora_name", display_name="Regenerate",
                               options=_files("loras", "open_h3_regenerate_2k_lora_complete"),
                               tooltip="One complete Standard Regenerate file from models/loras: merged LoRA, all Sharp branches and the reference table. BF16 LoRA factors; no separate Sharp or reference checkpoint is required."),
            ],
            outputs=[IO.Model.Output("model"), IO.String.Output("report")],
        )

    @classmethod
    def execute(cls, model, lora_name):
        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        if not regen.is_bundle(path):
            raise ValueError("Select open_h3_regenerate_2k_lora_complete.safetensors: the complete file must include Sharp branches and the reference table.")
        return cls._load_recipe(model, lora_name, None, None)

    @classmethod
    def _load_recipe(cls, model, lora_name, sharp_name, reference_checkpoint, lora_precision="bf16",
                recipe=ic.STANDARD, hyperflow_name=None, lms_name=None):
        """Internal reference loader for parity tests and experimental IC work."""
        if model.patches or model.object_patches or model.model_options.get("transformer_options", {}).get("patches_replace"):
            raise ValueError("Connect the clean H3 loader to TS H3 Regenerate Model, before all LoRAs and block patches.")
        dm = model.get_model_object("diffusion_model")
        if recipe == ic.IC_UP3:
            if not hyperflow_name or not lms_name:
                raise ValueError("IC requires hyperflow_name and lms_name from the official IC release.")
            reference = regen.read_reference_table(folder_paths.get_full_path_or_raise("diffusion_models", reference_checkpoint))
            result = ic.install(model, folder_paths.get_full_path_or_raise("loras", lora_name),
                                folder_paths.get_full_path_or_raise("loras", hyperflow_name),
                                folder_paths.get_full_path_or_raise("loras", lms_name), reference,
                                torch.float32 if lora_precision == "fp32" else torch.bfloat16)
            report = f"IC up3: non-pruned Ref2VA; HyperFlow two-time embedding + LMS Sharp + 312 unmerged IC branches; {lora_precision} factors. Quantized base/merged HyperFlow and Sharp differ numerically from the BF16 release."
            LOG.info(report)
            return IO.NodeOutput(result, report)
        if recipe != ic.STANDARD:
            raise ValueError(f"Unknown Regenerate recipe: {recipe}")
        if "open_h3_regenerate_2k_ic_" in lora_name.lower():
            raise ValueError("The IC adapter requires recipe=IC up3 (HyperFlow), not Standard.")
        if not getattr(dm, "use_adaln_curves", False):
            raise ValueError("TS Regenerate currently supports pruned H3 models with AdaLN curves.")
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        complete = regen.is_bundle(lora_path)
        reference = regen.read_reference_table(lora_path if complete else folder_paths.get_full_path_or_raise("diffusion_models", reference_checkpoint))
        sharp_path = lora_path if complete else folder_paths.get_full_path_or_raise("loras", sharp_name)
        patches = regen.read_patches(lora_path, model.model.state_dict(),
                                     torch.float32 if lora_precision == "fp32" else torch.bfloat16)
        result = model.clone()
        loaded = result.add_patches(patches, strength_patch=1.0)
        if len(loaded) != len(patches):
            raise ValueError(f"Regenerate applied only {len(loaded)} of {len(patches)} patches.")
        is_ref2va = regen.install_sharp(result, sharp_path, reference)
        result.model_options["ts_h3_regenerate"] = True
        result.model_options["ts_h3_regenerate_ref2va"] = is_ref2va
        report = f"Regenerate: 312 merged LoRA patches + 50 Sharp branches; {lora_precision}; verified reference curve."
        if complete:
            report += " Self-contained TS bundle with embedded Sharp branches and reference table."
        LOG.info(report)
        return IO.NodeOutput(result, report)


class TS_H3Regenerate(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="TS_H3Regenerate", display_name="TS H3 Regenerate", category="TS/Video",
            description="NVIDIA H3 latent refinement with the complete Standard Regenerate model, optional first/last or Full Reference guidance, and source soundtrack preservation.",
            inputs=[
                IO.Model.Input("model", tooltip="Output of TS H3 Regenerate Model."),
                IO.Conditioning.Input("conditioning", tooltip="Original positive conditioning, including encoded media for Full Reference."),
                IO.Latent.Input("latent", tooltip="Denoised H3 video or audio+video latent."),
                IO.Combo.Input("upscale_model", options=core._scan_models()),
                IO.Int.Input("width", default=1920, min=64, max=4096, step=32),
                IO.Int.Input("height", default=1088, min=64, max=4096, step=32),
                IO.Float.Input("denoise", default=1.0, min=0.0, max=1.0, step=0.01,
                               tooltip="1 = release schedule; 0 = upscale only. Lower values reduce redraw with a shift-6 schedule."),
                IO.Combo.Input("steps", options=[3, 2], default=3, tooltip="Standard Regenerate supports 2 or 3 refinement steps."),
                IO.Noise.Input("noise", tooltip="Connect Random Noise. One noise tensor is generated for the full clip and shared by overlapping chunks."),
                IO.Combo.Input("audio_mode", options=["keep source audio (lipsync)", "silence (reference)"], default="keep source audio (lipsync)",
                               tooltip="Source audio or encoded silence is held clean during refinement. The original soundtrack latent is returned unchanged."),
                IO.Combo.Input("precision", options=["bf16", "fp16", "fp32"], default="bf16", tooltip="Latent upscaler precision. NVIDIA reference uses bf16."),
                IO.Int.Input("chunk_length", default=0, min=0, max=100000, step=17,
                             tooltip="0 = whole clip, release recipe. Nonzero = lower-memory chunks, in multiples of 17 frames. The stitching mode controls how chunks share context."),
                IO.Int.Input("temporal_overlap", default=17, min=0, max=100000, step=17),
                IO.Vae.Input("audio_vae", optional=True, tooltip="H3 audio VAE; required for silence (reference) mode."),
                IO.Combo.Input("reference_mode", options=["off", FIRST_FRAME, FIRST_LAST, FULL_REFERENCE, CONNECTED_FRAMES], default="off", optional=True,
                               tooltip="Connected frames (auto) uses whichever of first_frame and last_frame arrive. Keyframes use the normal H3 video VAE. Full Reference uses encoded media from the original Ref2VA conditioning with a clean Ref2VA Regenerate model."),
                IO.Image.Input("first_frame", optional=True,
                               tooltip="Start image prepared directly from the source at final output size, before strong draft downscaling. Only the first image is used; it is fitted to width/height without cropping."),
                IO.Vae.Input("video_vae", optional=True,
                             tooltip="H3 VAE with an encoder: original, compatible INT8, or the combined original-encoder/Fast-V2 VAE. Decoder-only weights cannot encode references."),
                IO.Combo.Input("chunk_mode", options=[INDEPENDENT, LOCK_OVERLAP, BLEND_OVERLAP],
                               default=INDEPENDENT, optional=True,
                               tooltip="Lock overlap fixes the whole overlap. Blend overlap fixes its early part and gradually transitions to the redrawn tail. Both continuation modes require nonzero overlap for multiple chunks."),
                IO.Image.Input("last_frame", optional=True,
                               tooltip="End image for connected frames (auto) or first and last frames. The first image in the batch is center-cropped to the final canvas if necessary, then encoded as a guide at the clip's last frame."),
            ],
            outputs=[IO.Latent.Output("latent"), IO.String.Output("report")],
        )

    @classmethod
    def validate_inputs(cls, width=1920, height=1088, chunk_length=0, temporal_overlap=17, **kwargs):
        # Linked values are unknown (None) during ComfyUI prompt validation.
        # execute calls this again with the actual upstream results.
        if any(value is not None and value % 32 for value in (width, height)):
            return "Width and height must be multiples of 32; use 1088 for 1080p and crop after decoding."
        if any(value is not None and (value < 0 or value % 17)
               for value in (chunk_length, temporal_overlap)):
            return "Chunks/overlap must be non-negative multiples of 17."
        if chunk_length is not None and temporal_overlap is not None and chunk_length > 0 and temporal_overlap >= chunk_length:
            return "Chunks/overlap must be multiples of 17, with overlap smaller than the chunk."
        return True

    @classmethod
    def execute(cls, model, conditioning, latent, upscale_model, noise, width=1920, height=1088,
                denoise=1.0, steps=3, audio_mode="keep source audio (lipsync)",
                precision="bf16", chunk_length=0, temporal_overlap=17, audio_vae=None,
                reference_mode="off", first_frame=None, video_vae=None,
                chunk_mode=INDEPENDENT, last_frame=None, ic_clip=None, draft_images=None, ic_caption=""):
        """Standard UI; IC kwargs remain only for direct experimental Python calls."""
        import comfy.model_management as mm
        import comfy.nested_tensor
        import comfy.samplers

        if any(value is None for value in (width, height, chunk_length, temporal_overlap)):
            raise ValueError("Width, height, chunk length and overlap must resolve to numbers; check the connected upstream nodes.")
        valid = cls.validate_inputs(width, height, chunk_length, temporal_overlap)
        if valid is not True:
            raise ValueError(valid)
        ic_metadata = model.model_options.get("ts_h3_regenerate_ic")
        is_ic = ic_metadata is not None
        sigma_grid = ic.sigmas(ic_metadata, steps, denoise) if is_ic else regen.sigmas(steps, denoise)
        if is_ic and denoise > 0 and reference_mode != "off":
            raise ValueError("IC uses the clean draft as its reference. Set reference_mode=off; first/last/Full Reference controls belong to Standard.")
        if reference_mode not in ("off", FIRST_FRAME, FIRST_LAST, FULL_REFERENCE, CONNECTED_FRAMES):
            raise ValueError(f"Unknown Regenerate reference mode: {reference_mode}")
        automatic = reference_mode == CONNECTED_FRAMES
        use_first = denoise > 0 and (
            reference_mode in (FIRST_FRAME, FIRST_LAST) or (automatic and first_frame is not None)
        )
        use_last = denoise > 0 and (reference_mode == FIRST_LAST or (automatic and last_frame is not None))
        use_reference = use_first or use_last
        use_full_reference = reference_mode == FULL_REFERENCE and denoise > 0
        if use_first and (first_frame is None or video_vae is None):
            raise ValueError("First-frame mode requires both first_frame and the normal H3 video_vae inputs.")
        if use_last and last_frame is None:
            raise ValueError("First and last frames mode requires the last_frame input.")
        if use_reference and video_vae is None:
            raise ValueError("Keyframe guidance requires the normal H3 video_vae input.")
        if use_reference and not conditioning:
            raise ValueError("Keyframe guidance requires H3 text conditioning.")
        video, source_audio = regen.video_audio(latent["samples"])
        if width < video.shape[-1] * 16 or height < video.shape[-2] * 16:
            raise ValueError("Regenerate targets must be at least as large as the input in both dimensions.")
        if denoise > 0 and not model.model_options.get("ts_h3_regenerate"):
            raise ValueError("Connect TS H3 Regenerate Model to load the required adapter and Sharp branches.")
        if use_full_reference:
            if not model.model_options.get("ts_h3_regenerate_ref2va"):
                raise ValueError("Full Reference requires the clean pruned Ref2VA checkpoint through TS H3 Regenerate Model.")
            cond, reference_kinds = regen.full_reference_conditioning(conditioning)
        else:
            cond = regen.text_conditioning(conditioning)
        frame_count = core.frames_for_tokens(video.shape[2])
        if is_ic and denoise > 0:
            if draft_images is None or tuple(draft_images.shape[1:3]) != (video.shape[-2] * 16, video.shape[-1] * 16):
                raise ValueError("IC draft_images must match the source latent's pixel dimensions.")
            if draft_images.shape[0] != frame_count:
                raise ValueError(f"IC draft_images must contain exactly {frame_count} frames.")
        if (is_ic or audio_mode == "silence (reference)") and denoise > 0:
            if audio_vae is None:
                raise ValueError("IC and silence reference mode require the H3 audio VAE.")
            rate = int(getattr(audio_vae, "audio_sample_rate", 32000))
            audio = audio_vae.encode(torch.zeros(1, round(frame_count / 24 * rate), 2))
        elif audio_mode == "keep source audio (lipsync)" or denoise == 0:
            audio = source_audio
        else:
            raise ValueError(f"Unknown Regenerate audio mode: {audio_mode}")
        if audio is None and denoise > 0:
            raise ValueError("Source audio mode requires an H3 audio+video latent; choose silence mode for video-only input.")
        if audio is not None:
            audio = audio.to(device="cpu", dtype=torch.float32)
        expected_audio = round(frame_count * 5 / 3)
        if audio is None and denoise == 0:
            audio = torch.zeros(1, 32, 2, expected_audio)
        if denoise > 0 and abs(audio.shape[-1] - expected_audio) > 1:
            raise ValueError(f"Audio/video durations differ: expected about {expected_audio} audio tokens, got {audio.shape[-1]}.")
        bounds = (core.compute_segments(video.shape[2], chunk_length, temporal_overlap)[0]
                  if chunk_length else [(0, 0, video.shape[2], frame_count)])
        if chunk_mode not in (INDEPENDENT, LOCK_OVERLAP, BLEND_OVERLAP):
            raise ValueError(f"Unknown Regenerate chunk mode: {chunk_mode}")
        continue_chunks = chunk_mode != INDEPENDENT and denoise > 0
        soft_overlap = chunk_mode == BLEND_OVERLAP
        if continue_chunks and len(bounds) > 1 and temporal_overlap == 0:
            raise ValueError("Continue chunk mode requires a nonzero temporal overlap (at least 17 frames).")
        balanced_tail = False
        if continue_chunks and chunk_length:
            balanced = regen.balance_short_tail(bounds, chunk_length, temporal_overlap)
            balanced_tail = balanced != bounds
            bounds = balanced
        LOG.info("H3 Regenerate frame windows: %s; %s", [(f0, f1) for _, f0, _, f1 in bounds], chunk_mode)
        device = "cuda" if mm.get_torch_device().type == "cuda" else "cpu"
        # Upscale every chunk before loading the diffusion model; no repeated
        # 20-GB model swaps between chunks. Cache only CPU video latents.
        mm.unload_model_and_clones(model, unload_additional_models=False)
        mm.soft_empty_cache()
        ic_contexts = None
        if is_ic and denoise > 0:
            try:
                ic_contexts = [ic.encode_context(ic_clip, draft_images[f0:f1], ic_caption, f1 - f0)
                               for _, f0, _, f1 in bounds]
            finally:
                if getattr(ic_clip, "patcher", None) is not None:
                    mm.unload_model_and_clones(ic_clip.patcher, unload_additional_models=False)
                mm.soft_empty_cache()
        reference = end_reference = None
        if use_reference:
            try:
                if use_first:
                    reference = regen.encode_keyframe(first_frame, video_vae, width, height)
                if use_last:
                    end_reference = regen.encode_keyframe(last_frame, video_vae, width, height, last=True)
            finally:
                # Release the encoder before the latent upscaler/diffusion
                # stages, including when encoding fails.
                if getattr(video_vae, "patcher", None) is not None:
                    mm.unload_model_and_clones(video_vae.patcher, unload_additional_models=False)
                mm.soft_empty_cache()
        upscaled = []
        ic_upscaled = ic.upscale_clip(video, upscale_model, width, height, precision) if is_ic else None
        for k0, f0, k1, f1 in bounds:
            if ic_upscaled is not None:
                upscaled.append(ic_upscaled[:, :, k0:k1].contiguous())
                continue
            part = video[:, :, k0:k1].contiguous()
            targets = []
            if width * height > 2560 * 1440 and part.shape[-1] * part.shape[-2] * 256 < 2560 * 1440:
                scale = math.sqrt(2560 * 1440 / (width * height))
                mid_w = max(part.shape[-1] * 16, int(width * scale / 32) * 32)
                mid_h = max(part.shape[-2] * 16, int(height * scale / 32) * 32)
                if (width, height) == (3840, 2176):
                    mid_w, mid_h = 2560, 1440
                elif (width, height) == (2176, 3840):
                    mid_w, mid_h = 1440, 2560
                targets.append((mid_w, mid_h))
            targets.append((width, height))
            for w, h in targets:
                part, _, _ = core.upscale_latent(part, {"model_name": upscale_model, "width": w, "height": h,
                                                       "device": device, "precision": precision})
            upscaled.append(part)
        del ic_upscaled
        sampler = comfy.samplers.sampler_object("euler")
        full_noise = None
        acc_v = acc_a = None
        try:
            for i, (k0, f0, k1, f1) in enumerate(bounds):
                v = upscaled[i]
                upscaled[i] = None
                a0, a1 = core.audio_range(f0, f1)
                a = audio[..., a0:a1].contiguous()
                if denoise > 0:
                    video_mask = torch.ones_like(v)
                    overlap = locked = 0
                    if continue_chunks and acc_v is not None:
                        overlap = min(acc_v.shape[2] - k0, v.shape[2])
                        if overlap <= 0:
                            raise ValueError("Previous refined chunk does not cover the continuation overlap.")
                        locked = max(1, overlap // 2) if soft_overlap else overlap
                        v = v.clone()
                        v[:, :, :locked] = acc_v[:, :, k0:k0 + locked]
                        video_mask[:, :, :locked] = 0
                    piece = {"samples": comfy.nested_tensor.NestedTensor((v, a)),
                             "noise_mask": comfy.nested_tensor.NestedTensor((video_mask, torch.zeros_like(a)))}
                    if full_noise is None:
                        full_noise = core.clip_noise(noise, v, a, video.shape[2], audio.shape[-1])
                    piece_cond = cond
                    if is_ic:
                        piece_cond = ic.conditioning(ic_contexts[i], video[:, :, k0:k1].contiguous())
                    final_guide = end_reference if f1 == frame_count else None
                    if reference is not None or final_guide is not None:
                        piece_cond = regen.keyframe_conditioning(cond, reference, f0, final_guide, frame_count - 1)
                    sampled = core.sample_piece(piece, piece_cond, model, noise, sampler, sigma_grid, None, 1.0,
                                                noise_tensor=core.slice_noise(full_noise, k0, k1, a0, a0 + a.shape[-1]))
                    v = sampled.tensors[0]
                    if soft_overlap and overlap > locked:
                        v = regen.blend_overlap(acc_v[:, :, k0:k0 + overlap], v, locked)
                        # This overlap is already stitched. Append it exactly once.
                        acc_v = acc_v[:, :, :k0]
                    elif locked:
                        # Keep the boundary bit-exact even if a sampler/backend
                        # introduces numerical residue on zero-mask rows.
                        v = torch.cat((acc_v[:, :, k0:k0 + locked], v[:, :, locked:]), dim=2)
                acc_v, acc_a = core.temporal_append(acc_v, acc_a, v, a, i, k0, f0)
        finally:
            mm.unload_model_and_clones(model, unload_additional_models=False)
            mm.soft_empty_cache()
        # Preserve the original soundtrack latent exactly, even in silence mode.
        # Silence is only the model's clean conditioning during refinement.
        output_audio = source_audio if source_audio is not None else (acc_a if denoise > 0 else None)
        result = {k: v for k, v in latent.items() if k not in ("samples", "noise_mask")}
        result["samples"] = (comfy.nested_tensor.NestedTensor((acc_v, output_audio))
                             if output_audio is not None else acc_v)
        report = f"{width}×{height}, {len(bounds)} chunk(s), denoise={denoise:g}, sigmas={sigma_grid.tolist()}, {audio_mode}."
        if is_ic and denoise > 0:
            report += " IC up3: clean draft latent + Qwen draft-video context; sampling audio is silence, original soundtrack retained."
            if denoise not in (0, 1):
                report += " Lower-denoise scaled schedule is experimental."
        elif is_ic:
            report += " IC refinement and draft context inactive at denoise=0."
        if reference is not None:
            report += f" Experimental first-frame guidance: normal H3 VAE, {width}×{height}; original global frame 0."
        if end_reference is not None:
            report += f" Last-frame guidance: global frame {frame_count - 1} (zero-based), final window only."
        elif use_full_reference:
            report += f" Full Reference blocks: {', '.join(reference_kinds)}."
        elif reference_mode == FULL_REFERENCE:
            report += " Full Reference inactive at denoise=0."
        elif reference_mode != "off" and denoise == 0:
            report += " Keyframe guidance inactive at denoise=0."
        if chunk_length:
            report += " Experimental chunked mode; differs from whole-clip reference."
            report += f" Stitching: {chunk_mode}."
            if balanced_tail:
                report += " Short final tail redistributed within the selected chunk length."
        return IO.NodeOutput(result, report)


NODE_CLASS_MAPPINGS = {"TS_H3RegenerateModel": TS_H3RegenerateModel, "TS_H3Regenerate": TS_H3Regenerate}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_H3RegenerateModel": "TS H3 Regenerate Model", "TS_H3Regenerate": "TS H3 Regenerate"}
