# TS Ideogram Designer

Visual JSON-prompt designer for Ideogram 4. Open a full-screen editor, drag and resize **text** and **object** blocks on an aspect-correct artboard (optionally over a reference image), and design with **two-level presets** — 10 layout templates (*what* you're making) and 10 styles (palette + fonts + look) — in a **RU/EN** interface. The node emits a valid Ideogram 4 **structured-JSON caption** as a `STRING` plus **`width` and `height`** (INT), sized from the aspect ratio and a **0.5–2 MP** slider, always rounded to multiples of 32 — wire them straight into an empty-latent / canvas node. Editor rectangles become normalized `[y_min, x_min, y_max, x_max]` bounding boxes (integers 0–1000, top-left origin) and the whole caption is assembled to the **exact Ideogram 4 schema** — verified section-by-section key order (incl. the photo-vs-non-photo `medium`/`art_style` ordering). The **in-node preview is a true WYSIWYG miniature of the editor** — real fonts, weights, colours, outlines and solid plates, with auto-fitted, word-wrapped text — so what you see after *Save* is what Ideogram is asked to draw, and the final prompt is shown with **JSON syntax highlighting**. Style each text block with a single **Text style** dropdown (fonts are *described*, not named — Ideogram has no typeface selector), a **Thin / Regular / Bold** weight and a case; **text size comes from how big you draw the block**, not an abstract picker. Add an **outline** and/or a **solid plate** for legibility — each with its own colour, rendered live on the canvas and in the preview. Colour is steered with separate palettes for the **whole image, the background and the lighting** plus per-element colours, all folded into the caption and previewed live on the artboard. **Save, export and import** individual layouts and styles — or a **full design** (the entire artboard) — as JSON (imports are copied into the node's `user_presets/` folder). The inspector is organised into clear steps — *what you're making* → *how it should look* → *what's in the scene* — and **every control has a friendly, fully-localized hover tooltip**. Edit text inline by double-clicking a block, clone with **Alt-drag** or **Ctrl+C / Ctrl+V**, and the text stays the same size in edit and preview. First-class **Russian / Cyrillic** support (UPPERCASE + bold defaults) plus a *visual-only* mode that emits a clean placeholder block so you can overlay the text by hand for print-critical work. Fluid in-node preview that works in both the LiteGraph (Nodes 1.0) and Vue (Nodes 2.0) front-ends.

**Use when:** designing YouTube thumbnails, posters and covers where you need precise control over where text and elements land — and which style Ideogram renders.


<a id="files"></a>
### 📁 Files & Models (8 nodes)

Tools for managing model files, downloads, EDLs, and inspecting weights.

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
