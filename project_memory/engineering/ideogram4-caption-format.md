---
name: ideogram4-caption-format
description: "Verified Ideogram 4 structured-JSON caption format + Cyrillic/font gotchas, used by the TS_IdeogramDesigner node"
metadata:
  type: reference
---

Verified Ideogram 4 structured-JSON caption format + Cyrillic/font gotchas, used by the TS_IdeogramDesigner node

**Граница записи:** ниже сохранены исходные наблюдения и причины решений. Замеры, версии окружения и слова «сейчас» относятся к дате соответствующего наблюдения; это не повторная проверка на новом окружении. Порядок чтения и ограничения: [инженерная память](README.md#как-читать-записи).

Open-weight **Ideogram 4** (`ideogram-oss/ideogram4`, weights `ideogram-ai/ideogram-4-fp8`/`nf4`) is driven by a **structured-JSON caption**, not natural language. The commercial API accepts the same `json_prompt` object. Facts below were adversarially verified (≈930k tokens of research) against the primary sources `ideogram-oss/ideogram4/docs/prompting.md` and the `developer.ideogram.ai` OpenAPI (`V4JsonPrompt`).

**Schema** (top-level keys in this exact order): `high_level_description` (optional but strongly recommended — **omit when empty**, don't emit `""`), optional `style_description`, required `compositional_deconstruction` = `{background: str, elements: []}` (background before elements). **`style_description` key order is strict AND differs by type** (re-verified 2026-06-28 against `docs/prompting.md` "Key order is strict"): photo → `aesthetics, lighting, photo, medium, color_palette`; non-photo → `aesthetics, lighting, medium, art_style, color_palette`. I.e. `medium` sits AFTER `photo` but BEFORE `art_style` — this asymmetry was the **#1 order bug we shipped & fixed** (we had emitted `art_style, medium`). `aesthetics`/`lighting`/`medium` + exactly one of `photo`/`art_style` are required when style is present; `color_palette` LAST & optional. **Element key order differs by type**: obj → `type, bbox, desc, color_palette`; text → `type, bbox, text, desc, color_palette`. `type` ∈ `obj`|`text` only (no `shape`); `bbox` + `color_palette` optional within an element (omit when absent).

**bbox** = `[y_min, x_min, y_max, x_max]` — **y FIRST (row-major)**, 4 **integers 0–1000** (NOT 0–1 floats), origin top-left, square grid independent of aspect. The #1 bug risk: swap happens once in `frac_to_bbox` / `fracToBbox`, fixture `[55,642,1000,937]`.

**Text element**: `text` = literal byte-for-byte (multi-line via `\n`); `desc` = styling prose; multiple fonts ⇒ multiple text elements. **color_palette**: UPPERCASE `#RRGGBB` only (`#fff`/lowercase invalid); image-level ≤16, element-level ≤5. Serialize `json.dumps(separators=(",",":"), ensure_ascii=False)`.

**Two verified gotchas** that shape the UI:
1. **Named fonts do NOT work** — official: *"not currently possible to specify a typeface by name"*. Only generic descriptors steer type; a font name is at most an inert `"<name>-like"` hint.
2. **Cyrillic is unreliable** (English-only 0.97 OCR; non-Latin *"may have difficulty… if at all"*). Guard: short (1–3 words), UPPERCASE, bold/grotesque, append `"Cyrillic script, Russian text"`, offer visual-only placeholder for print.

Aspect ratio is a **separate** param (`16x9`, `x`-separator), not a caption field. Node output = `STRING json_prompt` + `STRING aspect_ratio` (NOT CONDITIONING). Implemented in [ts-ideogram-designer-node](ts-ideogram-designer-node.md).

**Measured 2026-08-02 (TS Image Studio integration).** Two failure modes, both
producing the model's own grey card "Image blocked by safety filter" rather
than an error:
1. **Free text.** A plain sentence ("a vintage travel poster of a lighthouse")
   is blocked; the identical graph fed a full JSON caption from a previous good
   run rendered a clean poster. A *minimal* `{"high_level_description": "..."}`
   is ALSO blocked — the caption needs real substance (style_description etc.),
   which is what the SuperPrompt preset "Ideogram Prompt Enhance" produces.
2. **Guider wiring.** `CFGOverride` belongs on the MAIN model, and the
   unconditional UNet goes straight into `DualModelGuider.model_negative`. We
   had them swapped; that alone blocked every render. Owner's reference graph:
   `Image Gen Ideogram.json`. Scheduler: `Ideogram4Scheduler(steps 12, mu 0.5,
   std 1.75)`, VAE `flux2/flux2-vae-v2.safetensors`.

The studio does not format captions itself — it hosts this node (see
[studio-hosts-node-editors](../history/studio-hosts-node-editors.md)).
