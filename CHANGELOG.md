# Changelog

Only changes a **user** can notice: what appeared, what changed behaviour, and —
above all — what stopped working and what to do about it.

Node ids, input names, their order and their defaults are frozen on purpose: a
workflow saved a year ago must open today. When that promise is broken, it is
broken here, in writing, with a way back.

---

## 12.0.1 — 15 Aug 2026

### The size you gave a node is now the size it keeps

Every TS node that draws its own interface — Video Saver, Video Loader, LoRA
Loader, Super Prompt, Image Studio, Resolution Selector, Prompt Builder — used
to open at a different size than you left it. The workflow file had the right
numbers all along (measured: 520×640 saved, 520×480 restored); the layout
recomputes a node's height from its widgets *after* the graph is applied and
overwrote them. It could go either way — one node shrank to its minimum, another
grew by twenty pixels every time it was opened.

The height is now re-asserted once the layout has settled, on loading a workflow
and after a run finishes. TS Group Bypasser is deliberately excluded: its height
follows the number of groups in the graph, which is its own rule.

### TS Files Downloader — it now knows every loader your ComfyUI has

"Get models from workflow" missed models in `Load Latent Upscale Model`, and it
was not alone: the scanner matched loaders against a table written by hand, and
on the maintainer's machine **49 installed node types own a model widget that
table never heard of** — two of them from ComfyUI itself, reading a whole
category (`latent_upscale_models`) that was missing from it. A model in such a
node was reported as "no models found in this workflow".

The map is now derived from the running server. Every loader's dropdown is
filled from a `models/` folder, so the options themselves say where they came
from, and the answer is per WIDGET: a loader carrying both a text encoder and a
checkpoint sends each to its own folder. The download target is the directory
the files are actually read from, not the registry key — `models/clip` rather
than a folder named after `clip_gguf`. The old table stays as the fallback for a
category that has no files on this machine yet. Pressing `R` forgets the answer,
because that is the gesture of someone who just installed something.

### TS LoRA Loader — every row has a switch now

Turning a LoRA off without deleting it was already possible: you clicked the
row's name. Nothing said so — the row just faded — so the obvious move was to
delete the entry and add it back afterwards, losing its strength and its place
in the order. There is a visible switch on each row now, in the same shape the
group toggles use. The name still works, for whoever had learned it.

### TS LoRA Loader — `R` finally reaches it

A LoRA dropped into `models/loras` did not show up in the node's search until
the page was reloaded. ComfyUI refreshes stock dropdowns by walking a node's
widgets; this node draws its own picker, so the refresh went straight past it.
It now listens on the same hook ComfyUI offers for exactly this, and reads the
fresh list out of the refresh itself instead of asking the server again.

---

## 12.0.0 — 12 Aug 2026

### TS Super Prompt — the LTX preset now targets LTX 2.5

`Video Prompt Enhance LTX` learned the one thing that actually changed in 2.5:
the model can **cut inside a single generation** and hold the same person, place
and voice across the cut. The preset now explains how to write those cuts — name
the transition, re-establish the framing, repeat the character description
word-for-word, say what the sound does — while keeping a single continuous take
as the default and four cuts as the ceiling.

Also: camera moves must say where the subject ends up once the move is over
(that is what lets the model finish the motion), and length now scales with the
action instead of a fixed sentence count. Shot lists and scene headers stay
forbidden — LTX renders `INT. KITCHEN - DAY` as on-screen text rather than
reading it as a cut.

**A sign is no longer read aloud.** Quoted words that the idea calls a sign, a
label or a title are written into the shot as something the camera sees, and
nobody says them — the preset used to invent a person to speak a shop window,
and sometimes replaced the words while doing it. A title now appears *over* the
picture rather than hanging in the world on an invented board.

**Quoted text is no longer translated.** Anything you put in `" "`, `« »`, `' '`
or `( )` — a spoken line, a sign, a title — is copied character for character in
its own alphabet, and Russian stays in Cyrillic. It used to come back in English
on the 2B model: the rule sat in the middle of a long instruction, which is
exactly the part a small model drops. It is now the first line of the preset and
the last, worded as a mechanical instruction rather than an explanation, and it
covers text in the frame as well as speech. Sampling was tightened to match
(`temperature` 0.5 → 0.35): copying is not a task that benefits from invention.

The preset also follows the format LTX ships in its own ComfyUI pack — the
prompt opens with `Style: …`, verbs are present-progressive, the sound is woven
through the action instead of collected at the end, and camera movement appears
only when you asked for it.

The preset name is unchanged, so saved workflows keep working.

### New node — TS Smart Batch

Batches images the way core's **Batch Images** does, with two differences that
matter in practice:

- **Inputs grow.** Fill the last slot and the next one appears, up to 32.
- **Every slot is optional.** Core's two inputs are required, so muting or
  bypassing either side breaks the whole graph before it even runs.

Behaviour: several connected → one batch in slot order, gaps skipped; one
connected → it passes straight through on its own; none → a plain error saying
so, instead of a blank frame pretending to be a result.

Sizes and channels are reconciled exactly as core does — a missing alpha channel
is padded, a differently sized image is resized to the first one that actually
arrived — so it is a drop-in replacement. Built for first-frame / last-frame
pairs you want to switch on and off without rewiring.

### TS Video Saver understands date tokens

`filename_prefix` now expands `%date:yyyy-MM-dd%`, `%date:hhmmss%` and the rest
of the family (`yyyy yy MM M dd d hh h mm m ss s`) anywhere in the path, so
`videos/my-run/my-run-%date:yyyy-MM-dd%_%date:hhmmss%` works as written.

Worth knowing why it did not before: ComfyUI's own tooltips promise these tokens
on every saving node, but the **backend never expands them** — the frontend
rewrites the value before queueing, and only for its own nodes. Sent through the
API, even core's `SaveImage` fails with `OSError: Invalid argument`, because a
colon is not legal in a Windows filename. Doing it server-side means it now works
from the UI, from the API and from a script alike. ComfyUI's own `%year%`,
`%width%` and friends keep working as before.

### ⚠️ Breaking: downloads go only into registered model folders

`TS Files Downloader` used to accept any relative target that stayed inside the
ComfyUI folder — `input/downloads`, `user/…`, and also `custom_nodes/…`. That
last one is why it changed: a `file_list` line arrives inside someone else's
workflow, and combined with an archive carrying an `__init__.py` it meant
running a stranger's code on the next start.

A relative target must now name a registered model folder (optionally prefixed
with `models/`). Writing outside `models/` is still possible on purpose — give an
absolute path and set `TS_DOWNLOADER_ALLOW_EXTERNAL=1` on the machine, which is a
decision its owner makes outside any workflow.

### ⚠️ Breaking: 16 nodes were retired (11 Aug 2026)

A graph that used one of these opens with the node shown **in red** as missing.
Nothing else in the graph is affected, and the rest of the pack is unchanged.

| Node id | Was called | Was in |
| --- | --- | --- |
| `TS_QwenCanvas` | TS Qwen Canvas | TS/Image/Size |
| `TS_QwenSafeResize` | TS Qwen Safe Resize | TS/Image/Size |
| `TS_WAN_SafeResize` | TS WAN Safe Resize | TS/Image/Size |
| `TS_Color_Grade` | TS Color Grade | TS/Image/Color |
| `TS_FilmGrain` | TS Film Grain | TS/Image/Color |
| `TS_Keyer` | TS Keyer | TS/Image/Cutout |
| `TS_Despill` | TS Despill | TS/Image/Cutout |
| `TS Cube to Equirectangular` | TS Cube to Equirectangular | TS/Image/360 |
| `TS Equirectangular to Cube` | TS Equirectangular to Cube | TS/Image/360 |
| `TS_Video_Upscale_With_Model` | TS Video Upscale With Model | TS/Video |
| `TS_FilePathLoader` | TS File Path Loader | TS/Files |
| `TS_ModelScanner` | TS Model Scanner | TS/Files |
| `TS_ModelConverter` | TS Model Converter | TS/Files |
| `TS_ModelConverterAdvanced` | TS Model Converter Advanced | TS/Files |
| `TS_ModelConverterAdvancedDirect` | TS Model Converter Advanced Direct | TS/Files |
| `TS_CPULoraMerger` | TS CPU LoRA Merger | TS/Files |

**Getting one back.** Everything they need — code, help pages, README sections,
the Russian locale, tests, example workflows and a snapshot of their contracts —
is kept in `archive/removed-nodes-2026-08-11.zip`, together with instructions.
Restore from that archive rather than from memory: the contract snapshot inside
is what guarantees a restored node keeps the same inputs in the same order, so
old graphs still load.

### Renamed (display name only — node ids unchanged, graphs unaffected)

- `TS_Qwen3_VL_V3` is now shown as **TS Qwen 3**.
- `TS Files Downloader` dropped the "(Ultimate)" suffix.

### Security

- **The Hugging Face token no longer reaches mirrors.** It used to be computed
  once before the mirror loop, so the first failure of `huggingface.co` sent a
  private token — often with write access — to `hf-mirror.com`. Now every
  endpoint is asked whether it is the official origin. One rule for the whole
  pack, in `nodes/_hf_download.py`; the file downloader, the shared helper and
  the Qwen engine all use it.
- **Downloads go only into registered model folders.** A `file_list` line
  arrives inside someone else's workflow, and the node's own resolver still had
  a fallback that accepted any name under the ComfyUI root — `custom_nodes`
  included. Combined with an archive carrying an `__init__.py`, that meant
  running someone else's code on the next start. Absolute paths are unchanged:
  still allowed, still only with `TS_DOWNLOADER_ALLOW_EXTERNAL=1`.
- **Archives are checked member by member.** Executable names (`.py`, `.dll`,
  `.bat`, …) are refused inside a zip just as they are outside it, and there are
  now ceilings on unpacked size, member count and compression ratio. The same
  limits apply to studio content packs.
- **The download request re-derives its headers for the final URL** it actually
  fetches, instead of carrying the ones computed for the address before
  redirects.

### Fixed — work that used to be lost

- **TS Lama Cleanup:** opening a saved workflow wiped every retouch. State is
  now restored on load instead of being overwritten by the source poller.
- **TS SAM Media Loader:** after a reload the editor came up empty and the first
  click overwrote the saved points. Points, source and checkpoint are restored.
- **TS Prompt Builder:** loading a workflow replaced the block selection saved in
  the node with whatever this machine had configured. The node's own selection
  wins again; new files on disk are appended, switched off.
- **TS Ideogram Designer:** uploading a reference no longer overwrites a
  same-named file in `input/` — which another graph may well be using.
- **Studio queue:** reordering jobs cleared the queue and resubmitted them; one
  failed resubmit lost the rest silently. The remainder is now always sent and
  failures are reported.
- **Sliders:** a legacy 10× step stored in properties silently rewrote saved
  values on load.

### Fixed — correctness

- **Audio taken from a video** was handed on in the frame's raw format: 16-bit
  tracks arrived at ±32767 instead of ±1, and packed stereo arrived as distorted
  mono. Both are normalised now.
- **TS Image Tile Merger** drew black seams when feather was on but tiles did not
  overlap. Feather is now bounded by the overlap it has to live in.
- **TS Langevin Inpaint** crashed on a latent batch larger than one.
- **TS Smart Switch** rejected ComfyUI's own `VIDEO` object.
- **TS Qwen 3** honours a local model path, as its tooltip has always promised,
  and no longer lets two repositories with the same last name share one cache
  folder. An existing cache is moved, not re-downloaded.

### Fixed — responsiveness and memory

- **Cancel now stops long nodes.** Frame interpolation, BiRefNet and ViTMatte
  drew a progress bar but never asked whether the run had been cancelled, so
  pressing Cancel did nothing until they finished.
- **Studio pack and pass routes no longer freeze ComfyUI.** They ran blocking
  network calls directly on the event loop; with the host unreachable the whole
  interface — previews, progress, `/interrupt` — stopped for up to two minutes.
- **Whisper keeps one model in memory instead of every model ever loaded.**
- **The studio releases the microphone** when the prompt panel is torn down.
- Assorted leaks closed: preview frames, history frames, the stage itself.

### Documentation

- The built-in help for TS Video Loader described environment variables
  (`TS_VIDEO_EXTRA_ROOTS`, `TS_VIDEO_ALLOW_ANY_PATH`) that do not exist. The real
  names are `TS_MEDIA_EXTRA_ROOTS` and `TS_MEDIA_ALLOW_ANY_PATH`.
- `requirements.txt` was missing `comfyui-frontend-package`, which
  `pyproject.toml` declared.

### For maintainers

- The contract snapshot now records **input order and outputs**. It used to sort
  widgets by name, which meant the single most dangerous change in the pack —
  reordering inputs, because `widgets_values` is positional — was invisible to
  the guard.
- A broken internal import is reported as `ERROR`, not `SKIPPED`. `SKIPPED` is
  for a third-party package the user did not install; using it for our own typo
  meant a node could vanish and CI would call it normal.
- The CI smoke test asks `comfy_entrypoint()` for the node list and compares it
  against the snapshot by name. It used to read `NODE_CLASS_MAPPINGS`, which the
  pack deletes on any modern ComfyUI — so it checked nothing at all.
- One failing `define_schema` no longer takes the rest of the pack with it.
- `tools/preflight.py` also checks that the built-in help matches the README.
