# TS Style Prompt Selector

Visual style picker: a library of **157 styles** with thumbnail previews, grouped into 15 categories that run from cave painting and Byzantine mosaic through the twentieth-century avant-garde and film noir to pixel art and vaporwave. Pick one — get the matching `STRING`.

Each entry is a **pure style modifier** — medium, technique, palette and texture, with no subject of its own — and ends with a comma and a space. That is deliberate: the output is meant to be **prepended** to your own prompt (a `Concatenate` node with this node in `string_a` and your prompt in `string_b`), so the two halves read as one sentence:

```text
style   ukiyo-e woodblock print style, bold black keyblock outlines, visible paper grain,
yours   portrait of an old fisherman
result  ukiyo-e woodblock print style, bold black keyblock outlines, visible paper grain, portrait of an old fisherman
```

Names, categories and descriptions are bilingual and follow the ComfyUI interface language (English / Russian).

**Use when:** stylising a generation without rewriting the same "in the style of …" phrase, or browsing for a look you cannot name yet.

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
