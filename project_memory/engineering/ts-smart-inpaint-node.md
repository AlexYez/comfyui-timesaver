---
name: ts-smart-inpaint-node
description: "Architecture & provenance of the TS Smart Inpaint node (headless ComfyUI-Angelo Xtra-Fine port)"
metadata:
  type: project
---

Architecture & provenance of the TS Smart Inpaint node (headless ComfyUI-Angelo Xtra-Fine port)

Исходники темы: [ts_smart_inpaint.py](../../nodes/image/ts_smart_inpaint.py).

**Граница записи:** ниже сохранены исходные наблюдения и причины решений. Замеры, версии окружения и слова «сейчас» относятся к дате соответствующего наблюдения; это не повторная проверка на новом окружении. Порядок чтения и ограничения: [инженерная память](README.md#как-читать-записи).

`TSSmartInpaint` (display "TS Smart Inpaint", category `TS/Image`) — single self-contained file
`nodes/image/ts_smart_inpaint.py` (auto-discovered by the loader; no manual registration). Given the
FULL image + a painted MASK (white = inpaint) + model/vae/positive/negative, it does the whole
crop→refine→composite in-node, so the workflow only feeds source + mask.

**Pipeline:** `bbox(mask)` + `context_pad` band → crop pixels → upscale to `megapixels` (capped at
`max_linear`) → VAE-encode → sample → VAE-decode → downscale → feather pixel-composite. Output = that
composite (node has ONE IMAGE output; no LATENT).

**Perf A0+A1 (commit `37ae6d4`, 2026-06-22):** the FULL frame never goes through the VAE in Replace /
Refine-with-upscale — only the masked crop does (the whole point for 6–8K inputs). Upstream's two
full-frame encodes (up-front `current` + post-composite) + feathered latent-blend were dead weight in
this port: they fed a `new_latent` the single-shot IMAGE-only `execute` discards (`out = new_pixels`).
They exist upstream only for the cross-click latent "canvas" (`state["history"]`) + the `LATENT` output,
neither of which this port has. **A0** drops the post-composite re-encode + blend (`return None,
new_pixels`). **A1** derives latent geom from `vae.spacial_compression_encode()` instead of a full encode
(comfy center-crops pixels to a multiple of it → latent side == `img_side // ratio` for `crop_input`
VAEs; real-encode fallback when `crop_input=False`). The lone consumer of the full latent — the
no-upscale Refine shortcut (`scale<=1.0`, Refine only) — now encodes it lazily, exactly when reached.
Output is byte-identical for crop_input VAEs (SD/SDXL/FLUX/Qwen/Wan). The inpaint algorithm itself is
untouched. Geometry inside the core now reads `mask.shape` (mask is already at latent res), not
`current.shape`.

**`replace` checkbox** (label_on "Replace" / label_off "Refine"):
- **Replace = Smart Inpaint** — `reference_latents` = the crop (`append=False`), masked latent zeroed,
  regenerates the region as a Kontext edit. `denoise` is IGNORED and locked to 1.0.
- **Refine** — standard Xtra-Fine partial-denoise refine at the `denoise` value, no reference.

**Local extension (NOT upstream):** optional `reference` IMAGE input (Replace only) is downscaled to the
`megapixels` budget, VAE-encoded, and CHAINED as a 2nd `reference_latents` entry (`append=True`) — "fill
the hole with THIS". Unconnected → identical to upstream.

**Gotchas baked into the code:** the resolution cap is unconditional — the old scale≤1.0 whole-frame
latent shortcut is REMOVED (every mode goes through the crop+composite path). Sampling uses a HARD
(binary) mask for **4D AND 5D** now (commit `d7d5b37`): the painted region is fully regenerated, the
context band preserved intact, and the feather lives ONLY in the pixel composite — a SOFT sampling mask
pulled the feather-band latent toward gray and showed as a dark line. The composite blends in LINEAR
light (`_srgb_to_linear`/`_linear_to_srgb`) — sRGB-space alpha blending darkened the cross-fade. Optional
custom-sampler trio (guider+sampler+sigmas, from @KursatAs) dispatches via `_do_sample`.

**Current local options (final state, commit `4ddbbdb`, 2026-06-22):** the node has ONE extra boolean
input beyond upstream — `color_correct` (default ON, LAST input). The earlier `hard_edge` (`4c1288e`) and
`limit_resolution` (`25dde3e`) toggles were CONSOLIDATED away per user feedback — do NOT re-add them:
- **`hard_edge` — REMOVED entirely** (code + input). User judged it gave no value in practice. The
  `_dilate_mask_2d` helper and `sample_mask_override` plumbing are gone; 5D still hardens its sample mask
  via the original `if latent_up.ndim == 5` guard.
- **Resolution cap — now UNCONDITIONAL** (no toggle). `_fine_upscale_factor` always treats `megapixels`
  as a true cap (small crops upscale capped by `max_linear`; oversized crops downscale to the budget),
  so the node never hangs/OOMs on a big mask in a 6-8K image. The dead whole-frame Refine shortcut and
  the Smart-Inpaint "never downscale" clamp were deleted. Refined patch is scaled back up to native bbox
  for the composite (output res unchanged, only processed detail bounded; raise `megapixels` for more).
- **`color_correct` (NEW, default ON):** fixes Flux's slight reddening on the generated patch. Estimates
  the per-channel shift from the PRESERVED ring around the mask (pixels inside the crop but outside the
  painted region, where the decoded patch and the original are the SAME content → their difference is
  purely Flux's shift), then applies the inverse (robust gain+offset, MAD-trimmed, clamped) to the patch
  before compositing — via module helper `_color_correct_patch`. Because it's measured on the unchanged
  surroundings it never drifts with the generated content and does NOT tint new content toward the
  surround (red object on a blue surround stays red). When ON, the crop's context band is grown to
  `_CC_ANALYSIS_MARGIN_PX` (32) so the ring always exists (this slightly changes generation framing — a
  documented coupling); a too-small ring → no-op. `color_correct=OFF` reproduces the prior shipped
  default (downscale-on, soft feather) byte-for-byte. Worklog + design rationale: [ts-smart-inpaint-handoff](../history/ts-smart-inpaint-handoff.md).

**`feather`/`context_pad` → PERCENT of mask (commit `b920d41`, 2026-06-22):** the old absolute-pixel
`feather` (Int 15) and `context_pad` (Int 0) were RENAMED + retyped to `feather_pct` (Float %, default
3.0) and `context_pct` (Float %, default 8.0) — percentages of the mask's short side (tight bbox, image
px), resolved via module helper `_pct_to_px` with clamps (`_FEATHER_FLOOR_PX=2`, `_FEATHER_CEIL_PX=256`,
`_CONTEXT_CEIL_PX=1024`). Rationale: a fixed px feather is ~20% of an 80px mask (over-soft → the
thin-stroke ghosting) but ~1% of a 1500px mask; since every crop is budget-normalised, "% of mask" ==
"% of processed crop". RENAMED (not reinterpreted in place) so a stale saved px value reads as an
obviously-wrong percent. All in `execute` (computes the tight bbox short side, scales feather_px/
context_px); the Angelo core still just receives the feathered `mask` + an int `context_pad_pixel`. The
CC ring keeps its separate 32px floor (`max(context_px, _CC_ANALYSIS_MARGIN_PX)`).

**UI declutter (commit `92a0a77`):** `max_linear` and `color_correct` were REMOVED from the node UI and
hoisted to module constants `_MAX_LINEAR=3.0` / `_COLOR_CORRECT=True` (always on) for code-level tuning.
Current UI inputs: replace, denoise, megapixels, context_pct, feather_pct, resize_method, seed, steps,
cfg, sampler_name, scheduler, + optional reference. **Dark-feather fix (commit `d7d5b37`)** — see Gotchas
above: binary sampling mask (4D too) + linear-light composite. Both fixes are unconditional (no toggle).

**Provenance:** byte-for-byte port of ComfyUI-Angelo's (`shootthesound/ComfyUI-Angelo`)
`_refine_with_fine_upscaling`; upstream is MIT (notice reproduced in the file header). Commit `9a5b610`
first landed it SPLIT as `_angelo_xtrafine.py` (helpers) + a thin `ts_smart_inpaint.py` wrapper; it was
then consolidated into ONE self-contained file and the helper file deleted (that's why git showed the
`_angelo_xtrafine.py → _smart_inpaint_core.py` rename + delete churn — discard it, single-file is final).
