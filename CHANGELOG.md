# Changelog

Only changes a **user** can notice: what appeared, what changed behaviour, and —
above all — what stopped working and what to do about it.

Node ids, input names, their order and their defaults are frozen on purpose: a
workflow saved a year ago must open today. When that promise is broken, it is
broken here, in writing, with a way back.

---

## 12.1.0 — 22 Aug 2026

### New node: TS Angle Select

Point a camera at the subject and get the prompt that asks a model for exactly
that view. The node shows a small 3D preview — subject, orbit, camera — with
three controls under it: rotation, height and zoom.

The wording is not the node's. With the bundled **Qwen Multi-Angle** preset the
output is the trigger phrase the Multiple-Angles LoRA was trained on:
`<sks> back view elevated shot close-up`, and nothing more. It reads like a
fragment because it is one — the LoRA learned these exact words, and prettier
English breaks the conditioning.

Presets are plain JSON, one file per model in `nodes/text/angle_presets`, so a
new model is a new file rather than a code change. A preset missing a phrase is
skipped with a line in the log instead of quietly producing a prompt with a hole
where the angle should be.

Three.js ships with the pack for the preview and loads **only when the node
appears**. It is deliberately kept out of the web folder: ComfyUI imports every
script in there on page load, and 675 KB would otherwise be paid by everyone who
never places this node.

### TS Super Prompt asks before it downloads, and checks transformers first

Two halves of one complaint. The node used to go and fetch its language model
without a word — several gigabytes — and, on an older transformers, it did that
first and only then failed with *"Transformers does not recognize this
architecture `qwen3_5`"*. So the download was both invisible and, sometimes,
entirely wasted.

**It asks now.** When the model is not on the machine, pressing the enhance
button opens a dialog naming the model, the exact size and the folder it will
land in. *Not now*, Escape, or a click outside all mean no, and nothing is
downloaded. Agreeing is remembered for the session, so it asks once, not on
every press. The size is the real repository size read from the Hugging Face
API — no bytes are fetched to find it out.

**It checks the library first.** Before any download, the node asks the hub what
architecture the model is and compares it with what the installed transformers
knows. If the library cannot load it, the run stops immediately with the model
type, the installed version, and the command that fixes it — instead of
spending a few minutes and gigabytes to arrive at the same conclusion. The check
asks the library what it supports rather than comparing version strings, so a
build from git or a partial upgrade is judged on what it can actually do.

**Requirement raised: `transformers>=5.2.0`** (was 4.57.0). The default model is
a Qwen3.5, and 5.2.0 is the first release that knows the `qwen3_5` architecture
— verified against the repository tags, it is absent in 5.1.0. The floor now
says what the pack actually needs.

### TS Video Loader stopped refusing clips the machine could easily hold

The memory ceiling was a flat 8 GB written into the code. On a 64 GB machine
that turned down a **13-second 4K clip** — 323 frames of 4096x2160, about
31.9 GB as float32 — with a message telling you to trim the timeline. The
ceiling now comes from the machine: 60% of its RAM, never below the old 8 GB, and
`TS_VIDEO_MAX_BYTES` still overrides everything. That clip now loads in 51 s.

Free memory is only ever a warning, never a refusal: it moves with whatever
models ComfyUI happens to be holding, so refusing on it would make the same
graph pass one minute and fail the next. And if an allocation really does fail,
the message reads like the guard's instead of a bare MemoryError.

**New input `when_too_large`** (appended, optional, so saved graphs are
untouched). Left at `stop` it behaves as before. Set to `use disk`, the frames
go into a memory-mapped file in the ComfyUI temp folder: what comes out is an
ordinary IMAGE tensor, and the allocation cannot fail outright however long the
clip is. It is not free — decoding writes every frame once, so memory still
climbs while it runs, and the same 31.9 GB clip took 92 s against 51 s in RAM —
but those pages are backed by a real file the system can drop, and downstream
only the frames a node touches are read back. Leftover files are swept on the
next decode.

### Depth is now two nodes: TS Video Depth and TS Image Depth

A still and a clip want different models, and one node with a `mode` switch made
that hard to see: half the widgets were dead at any moment. They are two nodes
now.

**TS Video Depth** keeps its node id, its first thirteen inputs and their order,
so saved graphs open unchanged. What it lost is the `mode` switch and the two
single-image inputs, which existed only in unreleased builds.

**TS Image Depth** is new, on the `TS/Image/Depth` shelf. It runs **Depth
Anything V2 Large**, the model actually trained on stills, and shows only what
applies to one: no window, no flicker filter, no `input_size`.

It runs the reference pipeline and nothing else: trim to a multiple of 14, run
the model, normalize **each picture on its own** min/max, resize back
bilinearly. Denoise, dithering and the guided upscale are gone from it — they
were built for video, and on a still the guided filter put a halo on contours.
Measured against the reference implementation, the map now differs by **0.36 %**,
under one 8-bit level, at identical detail; the old single-image path differed
by **4.10 %** and looked soft.

Two things caused that. `input_size` belongs to the video model — it lifts the
short side to 518 px, so a 1600 px photo went into the model at 784×518. And
`percentile` normalization buys temporal stability by clipping 1 %/99 %, which a
single picture has no neighbours to need: the clip only burned the nearest and
farthest pixels to flat white and black. On the image node the resolution is
`max_res`, native by default, and normalization is per picture.

### Only safetensors are offered in the depth model lists

Both depth nodes now suggest fp16 safetensors only — half the download and an
order of magnitude quicker to read (0.01 s against 0.66 s, measured). The small
video model was converted and published alongside the large one, so nothing was
lost from the choice.

A saved workflow that still names an old `.pth` keeps working: the file resolves
and downloads exactly as before, it is simply no longer suggested in the list.

The old single-image path handed the video model one picture as 32 duplicated
frames and kept the first result. That is 32 runs of the model for one answer, and the answer was
worse: measured on a portrait, the duplicated window flattens the depth range
until the face and hair are one white shape. The dedicated engine keeps the
structure and takes **0.18 s instead of 6.1 s**, on 2.6 GB instead of 9.2 GB.

Short clips gained from the same correction. Anything that fits in one window is
now run as a single window of exactly its own length rather than being padded out
with copies of the last frame — an input the model never saw while training. A
one-frame clip in *video* mode dropped from about 6 s to 0.22 s.

Two new controls are exposed rather than guessed: `flicker_suppression` blends in
a temporal median of the depth — a median, not an average, so single-frame pops
disappear while real movement stays — with `flicker_radius` for its reach; and
`window_length` / `window_overlap` open up the sliding window itself.

Weights moved to fp16 safetensors: half the download, and they load in
hundredths of a second instead of two thirds of one. Measured against pure fp32,
the depth differs by 0.02 % of its range. The `.pth` files stay in the list.

Moving between the two nodes used to cost about four seconds each time, because
the previous model was thrown away. Both engines now stay cached and ComfyUI's
own model manager decides what to evict; a switch costs 0.2 s.

Under both nodes sits one shared engine (`nodes/_depth_core.py`), so the two
cannot drift apart: normalization, denoise, dithering and upscale are the same
code in both.

### TS Music Stems moves to RoFormer, and the stems finally add back up

`model_name` now picks the engine. **BS-RoFormer SW** is the default and
returns six stems — vocals, bass, drums, guitar, piano and everything else.
**Mel-Band RoFormer** returns only vocals and instrumental, which is exactly
why it is better at that split: it spends its whole capacity on one boundary.
Demucs is still there, unchanged, so a workflow saved last year still sounds
like it did last year.

Mask separation does not sum back to the mix, and a null test finds the gap
immediately. So one stem is no longer taken from the model: it is the mix minus
all the others, which makes the set exact by construction. Measured on real
music, `vocal + instrumental` nulls against the source at 161 dB and the six
stems at 144 dB — the floating-point floor, not a modelling error.

Three things that were wrong before are now right. The last chunk is
back-shifted instead of being padded with silence, because a transformer that
attends across the whole segment has never seen a synthetic tail. The
overlap-add window is a raised cosine across the chunk rather than a short
linear fade, so the frames with the least context are the ones suppressed. And
the progress bar counts real chunks instead of pulsing on a timer, which also
means the run can be cancelled.

`precision` chooses fp16 or fp32 for the RoFormer engines; fp16 is roughly
twice as fast on half the VRAM and its error was measured at -61 dBFS or below on
real music. bfloat16 is deliberately absent — `view_as_complex`, which these
models use to build their mask, does not accept it, so offering it would only
guarantee a crash on the first chunk.

Saved graphs are safe: every existing input and output kept its position, and
`guitar` and `piano` were appended after them. An output the chosen model
cannot produce returns an ExecutionBlocker, so that branch of the graph is
skipped rather than fed silence that would look like a broken model.

### TS Prompt Builder now works in packs, and the packs know what goes together

A pack is a folder of wildcards plus a semantic map. The map is what makes the
difference: it says what each wildcard is, where it belongs in the phrase, what
it excludes and what it pairs with — so a run stops producing a winter street in
a swimsuit, or a close-up with a full-body pose, or two incompatible scenes at
once. Assembly follows the eight steps written in the packs themselves.

Any packs combine, in any combination. Their roles interleave into one sentence
instead of one pack's output being glued onto another's, and wildcards are
namespaced by pack so two `face.txt` never collide. Where two packs both offer a
face, a light or a pose, one survives — drawn with a weight, not by rank, so a mix
of five reads as a blend instead of as the highest-priority pack talking over
everyone.

And the scene now holds together. Place lives in the words of a line, not in the
links between files, which is why the semantic map alone could not stop a pool, a
rainstorm and a kitchen from sharing one sentence — 22% of assemblies did exactly
that. The place is chosen first and every other line is read against it; the ones
that disagree about where or when we are get dropped. The same measurement
afterwards reads 3%, and those are metaphors rather than mistakes.

Drop a folder into `nodes/prompts/` and press Reload; no restart. The node groups
wildcards by role, dims the ones that will collapse to a single pick, lets you pin
one so it wins a collision, and previews the result live with the same code the
run uses. A second output, `info`, says what the map threw out and why.

Your saved graphs keep working: the node id and its inputs are unchanged, the old
flat block list is still read, and `seed = 0` still means a new prompt every run.
One deliberate change — a wildcard that appears in a pack later is added switched
**off**, so an author adding a file no longer shifts everyone's prompts.

### Both LTX HDR technologies, chosen by one dropdown

There are two of them and they are not variations of one thing. The native 2.5
path **preserves** the range an EXR already carried and works in ACEScct. The HDR
IC-LoRA **grows** range out of ordinary SDR and works in ARRI LogC3. Feed one
path's output through the other's inverse curve and the colour shifts.

`hdr_mode` on the settings node picks which, and everything downstream follows:
the decode applies the right inverse, the stats node reports against the right
ceiling (linear 222.86 for ACEScct, 55.08 for LogC3), and in IC-LoRA mode the
guide is an ordinary SDR frame with no EXR read at all. Our LogC3 inverse matches
the official `LTXVHDRDecodePostprocess` to within 1e-6, measured over 501 points —
and unlike theirs, writing the EXR needs neither OpenCV nor
`OPENCV_IO_ENABLE_OPENEXR`.

The IC-LoRA is validated by Lightricks on LTX 2.3; support for 2.5 is officially
in development.

### TS Video Saver — the quality dropdown speaks Russian now

Every sub-setting of the format dropdown — quality, the H.265 10-bit switch, the
ProRes profile — stayed English in a Russian interface, and had since the node
was written. The translation was there all along; it simply never matched. The
frontend builds an i18n key out of the widget name, and a sub-widget is called
`format.quality`, but a dot means "go one level deeper" to i18next — so the key
had to be spelled `format_quality`. Measured on a live server by feeding four
candidate spellings at once: only the underscore one reached the screen.

### Native HDR for LTX 2.5 — seven nodes and one checkbox

An EXR goes in with the sun still in it, and an EXR comes out with the sun still
in it. The path is the official one: ACEScct working space, guides prepared
separately for each stage, float32 only for the final decode, scene-linear
Rec.709 master. No HDR LoRA is involved and none is needed — native HDR is part
of the ordinary LTX 2.5 inference path.

**Off by default, and free while off.** With the switch down no EXR is read, no
float32 VAE is loaded, and the graph behaves exactly as before. Measured on the
live server: with HDR off, a run whose EXR loader pointed at a file that does not
exist still finished successfully, because the lazy branch is genuinely never
walked. The EXR saver produced nothing at all — not a black frame, not a stub
file.

Two things the ordinary nodes cannot do, which is why these exist: `Load Image`
flattens everything above 1.0 without saying so, and `LTXVPreprocess` pushes the
frame through H.264 and 8-bit bytes. The HDR branch bypasses both.

`TS LTX HDR Stats` is the one to reach for when something feels wrong: lost HDR
looks completely normal until someone tries to pull the sky back in the edit.

### TS Video Saver — EXR sequences, and ProRes 4444 confirmed

A fourth format: **EXR sequence**, one scene-linear file per frame in its own
folder, 32-bit float or 16-bit half. It reads from a new `hdr_image` socket,
because the ordinary `images` input is clamped to 0..1 long before the saver sees
it. No compression setting — this encoder does not offer one. Sequences carry no
audio; the small H.264 preview is written in the same pass over the frames, so a
streamed source is only read once.

ProRes **4444** and **4444 XQ** were already there and are now verified end to
end, alpha variant included.

Everything else about the node is unchanged: existing formats keep their order,
and the new socket is last in the schema, so old workflows open exactly as they
were saved.

### Windows: the console stops screaming about closed sockets

`ConnectionResetError: [WinError 10054]` from `_ProactorBasePipeTransport._call_connection_lost`
was 402 of 1865 lines in a live session's log — 22% of it — arriving in bursts
every time a websocket closed. It is a CPython bug (python/cpython#83191, open
since 2020), not a ComfyUI one, and updating Python does not help.

It was not only noise: the exception escapes before `self._sock.close()`, so the
socket stayed open until the garbage collector got to it. The pack now wraps
that one method and finishes the interrupted cleanup, for six socket-teardown
codes and nothing else — an unknown code is re-raised, because your protocol's
`connection_lost` runs through the same method. Measured over eight page reloads
and two jobs: 9 tracebacks before, 0 after. Kill switch:
`TS_DISABLE_PROACTOR_GUARD=1`.

### New on the canvas — "Tidy up"

Right-click the canvas or a node → **Tidy up** → **Tidy layout**: the selection
(or the whole graph, when nothing is selected) is arranged into columns that
follow the wiring, each node shrunk to the size its content asks for, everything
on the grid. No node to add — it is a canvas command, and both entries are in
the command palette too, so keys can be bound to them.

The link reroutes — ComfyUI's small round dots on a wire — are spread evenly
along the straight line between the sockets they connect, so a wire that ran as
a dogleg becomes a straight run; a dot shared by several links settles between
them. A wire whose straight line would cut through a node keeps its bend: a
detour that exists for a reason is not undone. **Align link dots only** does
just that part, leaving the nodes where they are.

**Tidy layout + route the wires** goes further: after the layout, every wire
that would cross somebody else's node gets dots that take it into the corridor
between columns, along one free lane, and back — and every dot that earns
nothing is dropped again. Wires that already have dots are left to their owner,
wires running against the flow are left alone, and running it twice changes
nothing.

**Pack as tiles** is the other way round: nodes packed as tightly as they go,
every node in a column the same width, a column holding several consecutive
layers rather than one, and the column height chosen so the whole schema lands
near 16:9 with no column left half empty. Nodes of a kind stay together; the
wires are not touched at all. On a real 32-node workflow: 17 columns and
5540×1308 become 5 columns and 1760×1128, with 76% of the area actually used
instead of 17%.

Columns come from the graph, not from where things happened to sit: a node's
column is its distance from the start of the flow, and the order inside a column
is chosen to keep links from crossing, starting from the order you already had.
Groups are laid out from the inside out — the nodes of a group are arranged
within it, the frame is fitted to them, and the group joins the outer layout as
one block. Pinned nodes are never moved. The top-left corner of the schema stays
where it was.

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
