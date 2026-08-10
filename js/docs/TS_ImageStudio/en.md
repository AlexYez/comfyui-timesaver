# TS Image Studio

A fullscreen studio: generate, edit, inpaint and upscale with local models
through native ComfyUI workflows under the hood. Open the interface from the
node; the IMAGE output emits the result selected in the studio gallery.

**Modes:** Generate (t2i), Edit (Qwen Image Edit, up to 3 references),
Inpaint (instant LaMa Cleanup + diffusion Repaint on one canvas), Upscale
(SeedVR2 restoration or a diffusion second pass).

**Prompt field:** voice dictation, attach-an-image with AI combining,
enhance presets, a 157-style library with chips.

**Backends are plain ComfyUI workflows** with TS/Studio marker nodes — drop
your own into `user/default/ts-studio/workflows/` to add or override.

Press F1 inside the studio for the full built-in help.
