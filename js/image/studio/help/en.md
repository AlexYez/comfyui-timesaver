# TS Image Studio — built-in help

A fullscreen studio for generating and editing images. Everything under the hood is native ComfyUI workflows: the queue, cancellation and model cache are shared with your normal generations.

## Modes (left rail)

- **Generate** — text-to-image and instruction editing in one mode. Describe the picture, pick a model, format and resolution, press **Run** (Ctrl+Enter). When the model can use references (Flux 2 Klein, Qwen Image Edit), slots appear under the prompt: leave them empty for a plain generation, fill one and the run switches to editing, taking its frame from the first reference.

- **Inpaint** — two engines on one canvas. **Cleanup**: paint over an object and release — it vanishes in a second (LaMa, no prompt). **Repaint**: mask + description + Run — a diffusion model redraws the region.
- **Upscale** — enlarge the result selected in the gallery, or whatever you drop onto the stage. The factor is a choice of **1x / 2x / 4x** (1x being a second pass at the original size), and the model comes from the same list as everywhere else: each one enlarges by its own recipe.

## The prompt field

- Microphone — dictate text (HQ is slower but more accurate). Text lands at the cursor.
- Image — attach a picture: the ✨ button combines it with your text through Qwen VL.
- Palette — the style library: picked styles append to the prompt as chips, removable by their cross.
- ✨ — prompt enhancement; the preset selects next to it.

## Choosing a repaint strength

Measured on identical tasks, so the numbers mean something:

- **0.1-0.25** — refinement: the object stays itself and gains texture and fine detail. This is what "improve the quality" wants.
- **0.3-0.5** — visible rework: the object's shape and pattern change.
- **0.6-1.0** — replacement: the mask fills with whatever the prompt describes.

**Flux 2 Klein** keeps a deliberately narrow range (0.1-0.5) — it is the gentle-retouch tool, and full replacement lives on its **Replace** switch. It is also the fastest: an object swap takes about 40 seconds.

From the runs: to swap an object reach for Klein with Replace, or Krea 2; to add detail without substituting the subject, Qwen Image at 0.35 or Klein at 0.2.

## Which model to upscale with

Every family enlarges through the same shape — lanczos stretch, tile split, one pass per tile, seams merged back. What differs is the model and what holds it steady:

- **Z-Image Turbo** — the only one with a tile ControlNet: structure holds best, so its default denoise is 0.45. The most faithful at 4x.
- **Flux 2 Klein** — with the Samsung detail LoRA, 3 steps.
- **Krea 2 Turbo** — 8 steps, its own recipe.
- **Qwen Image** — 4 steps with the Lightning LoRA.
- **Ideogram 4** — through its dual-model guider; the default preset is **Turbo** (8 steps) rather than the 12 it generates with, because an enlargement converges in fewer steps than a picture born from noise.
- **SeedVR2** — restores rather than invents: it clears compression artefacts and noise.

Measured on a 512x512 source at 2x: Z-Image 42s, Flux 37s, Krea 53s, Qwen 95s, Ideogram on Turbo 34s (it took 124s at 12 steps). 4x on Z-Image took 83s for 2048x2048.

Families without a tile ControlNet default to 0.35 denoise: at 0.45 they began rewriting the ornament instead of sharpening it.

## Frame format

Each format button is the rectangle of that proportion, with the ratio written inside it. If the one you want is not there, type it into the **w:h** field (`21:9`, `5:4`, `2.35:1`) — it joins the row and applies at once.

## Processing resolution (Flux 2 Klein)

Klein does not redraw the whole frame: it cuts out the masked region, generates there, and feathers the result back in. The **Processing resolution** slider sets how many megapixels that generation runs at. A small selection is scaled up toward the budget — more detail, more time; an oversized one is scaled down to it so memory stays bounded.

Rough guide: 0.6 MP for a quick pass, 1.5 MP as the working default, 3-4 MP when texture matters over a large area.

The other models work differently — they redraw the whole frame (LanPaint), so resolution there follows the image itself and no slider appears.

## Placing a specific object (Flux 2 Klein)

Klein's inpaint deck has a **Reference object** slot. Drop a picture of the thing you want, mask where it should go, and that exact object appears there instead of a generic one from the description. The prompt still directs the staging: where it stands, how it turns, what light falls on it.

The reference is only read during a full redraw, so filling the slot switches **Replace** on by itself — the strength slider greys out because it plays no part in that mode.

Paint the area at the size of the object: what fits in the mask is what you get. A thin stroke yields a fragment rather than the thing itself — a bottle needs a tall patch reaching from the table to where its neck will be.

## Ideogram: the designer, hosted

Ideogram 4 has a node of its own — **TS Image Ideogram Designer** — and the studio does not replace it: the **Layout** button opens that very editor, with all its panels, presets and its prompt-generating button. The model reads a structured description rather than free text, and it is the node that builds one, so the image's metadata carries exactly the prompt the render used.

Skip the designer and the short path applies: plain text from the prompt field goes through the node's Auto mode and becomes a description via the same helper its editor uses.

The deck is deliberately short for this model: format, resolution and the whole layout live in the editor itself rather than being mirrored beside it. What stays in the deck is what the node does not decide — prompt, seed, LoRA, and the sampler settings under Advanced.

## References and LoRA

- Reference slots accept dropped files, gallery cards and Artius Browser cards; click to pick a file; the cross clears.
- **+ Add LoRA** — search across installed LoRAs; strength from −2 to +2; reorder rows by their drag handle.

## The left panel: session, library, queue

- **Session** — every result of this node; clicking selects the image, and the node's IMAGE output emits the same one. Cards drag into slots and onto the canvas.
- **Library** — recent server images (or Artius Browser when installed). Its fullscreen viewer opens above the studio and closes with its own Escape.
- **Queue** — ComfyUI's own queue, so studio jobs and graph runs appear side by side. Drag to reorder, × drops a job, **Clear** removes every pending one, **Stop** interrupts what is running.

The panel collapses by its edge grip or the Tab key.

## Recreating a session from an image

Every result carries a snapshot of its settings in a PNG text chunk of its own — beside ComfyUI's standard metadata, competing with neither it nor image browsers. The **Recreate** button under the picture (or dropping such a PNG onto the canvas) restores the mode, model, prompt, seed, format and LoRA chain, and puts the source back where it was used: the inpaint canvas, the upscale stage or the reference slots. Dropping someone else's image simply makes it the current mode's source.

The prompt is additionally written into the standard metadata (through TS Image Prompt Injector), so Artius Browser and other viewers show it exactly as they do for ordinary generations.

Cards in the Artius browser carry two studio buttons: **S** sends the picture into whichever mode is open (the inpaint canvas, the upscale stage, or a free reference slot), and **R** restores the session it was made in. Both are in the right-click menu as well.

The same is available from the Artius browser: right-click an image and choose **Restore studio session**. The studio opens — or the one already on screen is reused — with that render's settings back in place. It works both when the studio is closed and when the browser is open inside it, in the Library tab.

Artius also marks studio work on the thumbnails: the badge names the mode and the model ("Generate · Krea 2 Turbo"), and the tooltip adds seed, steps, CFG and repaint strength. Images made elsewhere look as they always did. If older work shows no badge yet, press Rebuild Cache in the browser.

## Packs and the subscription

Some models arrive not with the pack itself but in **content packs** — the Packs screen on the left rail. Everything that exists is shown there: a cover, what it does, what is new this month. The catalogue is open to everyone; no key is needed to look.

Models from a pack you do not have appear in the model list too — greyed out, marked "in a pack". Clicking one opens Packs on that card.

The month's code comes with a subscription (Boosty, Patreon, VK) and is entered once: Packs → Get access, or Settings → Subscription. With no internet, paste the token from the same post instead of the code. A key runs for a month from the moment it is activated, not from the moment it was issued.

**What is installed keeps working.** The key is needed only to receive a pack or an update to one; once installed, a pack keeps working after the subscription lapses. Settings shows the date the key runs to, and a Forget button that wipes the key from this machine without touching anything already installed.

With a `user/default/ts-studio/dev.json` file present, Settings grows a **Testing** section: read the catalogue from a local build instead of the published one, and show the pass as absent or expired without losing the real one. An ordinary install has no such section.

## Missing models

When a family lacks its file, the deck shows a list: **Download** searches Hugging Face and fetches with live progress (speed, ETA, SHA256 verification) — one model at a time, with a total bar above. An interrupted download resumes where it stopped.

## Your own workflow backends

Build a workflow in ComfyUI, add the markers from the `TS/Studio` category (parameter inputs, `TS Studio Output`, `TS Studio Manifest` with the JSON descriptor), export in API format and drop it into `user/default/ts-studio/workflows/`. A matching id overrides the built-in backend.

## Shortcuts

- Ctrl+Enter — Run; Esc — close the studio
- Tab — collapse/expand the asset panel
- In Inpaint: [ and ] — brush size; Ctrl+Z / Ctrl+Y — undo and redo
