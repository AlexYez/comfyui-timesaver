---
name: reference_h3_regenerate_character
description: "H3 Regenerate first-frame guidance at final resolution and continuation with locked overlap; installed workflow and verification"
metadata:
  type: reference
---

H3 Regenerate first-frame guidance at final resolution and continuation with locked overlap; installed workflow and verification

**Граница записи:** ниже сохранены исходные наблюдения и причины решений. Замеры, версии окружения и слова «сейчас» относятся к дате соответствующего наблюдения; это не повторная проверка на новом окружении. Порядок чтения и ограничения: [инженерная память](README.md#как-читать-записи).

# H3 Regenerate: character reference and temporal seams, 31.10.2026

User approved an experimental extra first-frame guide for TS_H3Regenerate.
Important correction: prepare it in a separate workflow branch directly from the
source image at final width/height (2x draft dimensions), rather than enlarging the
already reduced draft image or handing the raw source to the refiner implicitly.

New optional inputs appended without changing old widget order:
`reference_mode` (off by default), `first_frame` IMAGE, `video_vae` VAE,
`chunk_mode` (independent crossfade by default).
`_regenerate.encode_first_frame` takes only the first RGB image, resizes only when
needed (Lanczos, no crop), and encodes once with the normal H3 video VAE.
Expected shape [1,24,1,H/16,W/16]; INT8 normal video VAE tested. Fast VAE is decode-only.
The encoder is unloaded before latent upscaling/sampling. Denoise zero skips guidance.

The new guide replaces explicit draft keyframes/refs. Every chunk receives it at
global frame zero, using local index `-frame_start` so later chunks do not treat
the original pose as their new first pose. Native packed layout/RoPE accept signed
positions. This remains an experimental extrapolation beyond the release T2VA recipe;
native numerical execution does NOT prove identity or pose quality.

User also reported morphing at temporal seams. Added optional `continue (lock overlap)`:
reuse the completed video overlap in the next piece with zero video noise mask,
leave the new part denoisable, keep original audio pinned. Restore exact old overlap
after sampling to eliminate numerical residue; crossfade then combines identical
overlap values. No previous-frame keyframes/extra copies of overlapping frames.
Multiple chunks require overlap >0 (17-frame grid). Whole clip has no chunk boundaries.
Continuation may propagate visual drift, especially on long clips: do not promise
absence of seams/exposure drift without evaluating the user's full video.

Installed separate workflow:
`D:/AiApps/ComfyUI/comfyui/ComfyUI/user/default/workflows/Lipsync H3 - Regenerate - Character.json`.
Built from user's current installed Regenerate workflow; retained their steps=2,
denoise=1, seed, first-stage settings. New ImageScale node227 takes LoadImage157,
target205/207 (2x draft), feeds202.first_frame; normal VAE121 feeds202.video_vae.
Both new modes enabled. Original user workflow file was not overwritten.
Backups: tests/.cache/install-backups/h3-character-20261031_002452 and _002553.

Verification using ComfyUI Python:
- 97 relevant tests passed; compileall latent_upscale passed.
- GPU real downloaded merged adapter + normal INT8 video VAE: 5-frame whole and
  22-frame independent chunks, plus final 39-frame continuation with 17-frame overlap
  at 128x128, full release 3-step sigmas. Source audio exact, outputs finite.
  Native sampler fixed-overlap maximum residue 4.768e-7, eliminated in final stitching.
  Synthetic input video/text means no face-quality validation.
- Browser old/new workflow serialization and API links in classic and Vue nodes passed.
  Screenshot inspected: reference preparation node and two dimension nodes don't overlap.
- Server validation recursively checks the Character graph, rejects only deliberately
  invalid test-only saver fps (to prevent enqueueing). No actual workflow value changed.
- Installed files, served JS, and loaded 88 schemas match verify_live_sync after restart.

Evidence: tests/.cache/h3-character-gpu.json, h3-character-continuation-gpu.json,
h3-character-installed-ui.json, h3-character-validation-after.json.
Test-only process shutdown emitted ComfyUI ModelPatcher destructor ON_DETACH=None
warnings after successful inference; this is interpreter teardown, not sampler failure.

Published first-frame experiment: https://ai-muninn.com/en/blog/open-h3-regenerate-2k-rtx5090
Concerns lettering preservation, not a validated face-identity guarantee.

## Short final continuation tail: frame-221 camera jerk (31.10.2026)

User reported a camera-like jerk near the end of a 9-second clip, then confirmed
frame221. Compare the penultimate output 004952 (Regenerate) and antepenultimate
003507 (source), NOT latest005253 (DLSS). Files in output/video/h3-regenerate-character.
Both source/refined metadata confirm identical upstream seeds/settings. PyAV/OpenCV
background motion measured at common736x416: f221 source median(-.32,+.11) px,
Regenerate(-1.16,+2.19) px; another large transient f219, smaller at f151/152.
Evidence: tests/.cache/h3-jerk/motion.json, tail-comparison.png, h3-jerk-history.json.

For T67 /226frames, chunk85 overlap17 generated windows0:85,68:153,136:221,204:226.
The final22frames had17already locked, leaving only5new frames after221. Hard locking
overlap stopped morphing between versions, but this short continuation exposed a
motion discontinuity. Context/padding and generative drift may also contribute;
do not claim visual elimination before rerendering the real clip.

Added `_regenerate.balance_short_tail(bounds,chunk_length,temporal_overlap)`.
Only active continuation (denoise>0) uses it, and only when final new span<17frames.
It redistributes the same window count on the17frame/5token phase grid, with larger
hops first and the fractional tail in the last window. For226/85/17 the result is
0:68,51:119,102:170,153:226 (lengths68,68,68,73). No isolated5frame redraw after221.
Preserves cap85, exact overlap17, number4, total sampled token/frame work, full
video duration/audio. For very short clips there may be no better grid split:
then original bounds remain. Independent mode and denoise0 retain original plans.
No shared core changes or schema/default/widget changes. Workflows not overwritten.

111 relevant tests passed, including plans through T1067 (~151seconds) with varied
caps/overlaps. Actual GPU native H3+merged adapter+normalINT8VAE test T67 at128x128,
all226frames/four balanced windows succeeded, source audio exact, finite latents,
locked-mask residue4.768e-7 handled by existing exact restoration. Test input latent
and text synthetic, so this is not face/camera quality validation at1472x832.
Record: tests/.cache/h3-short-tail-gpu.json. Runtime backup:
tests/.cache/install-backups/h3-short-tail-20261031_010441.

## Release 12.13.3 (explicit user request: patch and push)

Published all accumulated runtime fixes and three H3 Regenerate nodes in commit
2a571c51120131d7ffac94134efbd9d3a76eeca8, pushed origin/master. Version/badges and
CHANGELOG updated to12.13.3. Personal workflow_exports/ and AUDIT_REPORT remain
local/untracked. Third-party fast_vae LICENSE/NOTICE included.

Release verification:358 relevant tests passed under ComfyUI Python,4 browser
tests deselected in final backend run.208 tracked Python files compile; changed
Python syntax parses for3.10;6 modified JS modules pass Node syntax checks.
Repeated browser suite could not connect because the user ComfyUI server was
off; original browser/GPU evidence remains valid for the already-tested code.
Do not misreport the unavailable repeat as a frontend pass.

Runtime git clone fast-forwarded to the same commit and is clean. Previously
copied patches were verified against published source, backed up to
tests/.cache/install-backups/release-12.13.3-20261031_011521, temporarily stashed
for git pull, then the redundant stash was dropped after exact verification.
No user workflow files changed. Dev tree only has the above local untracked files.

GitHub Publish to Comfy registry succeeded:
https://github.com/AlexYez/comfyui-timesaver/actions/runs/41136802327
Dependency Graph action was still running at the final check; no claim that all CI completed.
