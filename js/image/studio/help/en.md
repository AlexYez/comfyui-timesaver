# TS Image Studio — built-in help

A fullscreen studio for generating and editing images. Everything under the hood is native ComfyUI workflows: the queue, cancellation and model cache are shared with your normal generations.

## Modes (left rail)

- **Generate** — text to image. Describe the picture, pick a model, format and resolution, press **Run** (Ctrl+Enter).
- **Edit** — instruction-based editing (Qwen Image Edit): up to three references, the first is the source. Describe what to change.
- **Inpaint** — two engines on one canvas. **Cleanup**: paint over an object and release — it vanishes in a second (LaMa, no prompt). **Repaint**: mask + description + Run — a diffusion model redraws the region.
- **Upscale** — enlarge the gallery-selected result. **SeedVR2** restores and grows up to 4x; **Diffusion 2-pass** adds detail with a second generative pass.

## The prompt field

- Microphone — dictate text (HQ is slower but more accurate). Text lands at the cursor.
- Image — attach a picture: the ✨ button combines it with your text through Qwen VL.
- Palette — the style library: picked styles append to the prompt as chips, removable by their cross.
- ✨ — prompt enhancement; the preset selects next to it.

## References and LoRA

- Reference slots accept dropped files, gallery cards and Artius Browser cards; click to pick a file; the cross clears.
- **+ Add LoRA** — search across installed LoRAs; strength from −2 to +2; reorder rows by their drag handle.

## Gallery and library (right)

- **Session** — every result of this node; clicking selects the image, and the node's IMAGE output emits the same one. Cards drag into slots and onto the canvas.
- **Library** — recent server images (or Artius Browser when installed). The panel collapses by its edge grip or the Tab key.

## Missing models

When a family lacks its file, the deck shows a list: **Download** searches Hugging Face and fetches with live progress (speed, ETA, SHA256 verification) — one model at a time, with a total bar above. An interrupted download resumes where it stopped.

## Your own workflow backends

Build a workflow in ComfyUI, add the markers from the `TS/Studio` category (parameter inputs, `TS Studio Output`, `TS Studio Manifest` with the JSON descriptor), export in API format and drop it into `user/default/ts-studio/workflows/`. A matching id overrides the built-in backend.

## Shortcuts

- Ctrl+Enter — Run; Esc — close the studio
- Tab — collapse/expand the right panel
- In Inpaint: [ and ] — brush size; Ctrl+Z / Ctrl+Y — undo and redo
