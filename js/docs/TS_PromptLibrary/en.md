# TS Prompt Library

A library of ready-made prompts. Pick a **section**, a **model** and a **prompt**: the node shows the prompt with a **Copy** button and sends the same text on as a `STRING`.

**The first section is context editing** — 51 tasks for two models: restoration and colorization, removing and adding objects, background removal, face, head and hair swaps, clothing and pose, relighting, camera angle, framing and outpainting, background, time of day and weather, color, material, style, lettering and hands. Each prompt comes with a short explanation, the number of input images it expects and a note on its limits.

- **Qwen Image 2.1** — an author's catalog written to the official PE-I2I instructions; inputs are named `<image1>`, `<image2>`.
- **FLUX.2 Klein 9B** — the same 51 tasks, rewritten to Black Forest Labs' FLUX.2 guidance: inputs named in words (`image 1`, `image 2`), no prohibitions in the text — FLUX has no negative prompt and whatever the text names tends to appear — and an explicit list of what stays unchanged. Klein has no alpha channel, so its isolation prompt puts the subject on white for TS Remove Background to cut out.

Task codes match across models, so switching from Qwen to Klein stays on the same task.

**Background removal with real transparency (O13).** The model finds the main subject itself — nothing to describe — and removes the whole background; for a chosen object among several there is O12, where you name it. Qwen Image 2.1 works in RGBA natively: its VAE decodes four channels and the stock **Save Image** writes a PNG with alpha, so the cut-out arrives transparent with no separate matting step — as long as nothing between the VAE Decode and Save Image drops the fourth channel. An input that already has transparency is split by **Load Image** into IMAGE and MASK; join it back with **Join Image with Alpha** before the text encoder, or the model sees it as opaque. Klein has no alpha, so its O13 puts the subject on white.

**Fields in braces are filled in on the node.** `{TARGET}`, `{POSITION}`, `{LIGHTING}` and the rest appear as input rows with an explanation and an example (one click puts the example in). The prompt highlights each field — filled or still empty — and the output is the finished text, not a template with braces. An empty field is left in braces and named under the prompt.

**Search — on the node and full screen.** The search field sits right on the node, above the prompt. It looks through the code, title, group, description and the prompt text itself in both languages; every word must match, in any order, and «ё» counts as «е». While you type, the matches take the card's place: ↑ / ↓ move through them, **Enter** or a click picks one and brings the card back, **Esc** clears the search. With a search active the ‹ › buttons step only through what it found.

**Full screen.** The **Open interface** button turns the node into a catalog: every prompt of the chosen model in a list on the left — grouped, under the same search field (the query comes along), and with thumbnails once presets have previews — and the chosen prompt on the right with a large preview. ↑ / ↓ step through the prompts — through the matches while a search is active, from the search field too — and Esc closes. Thumbnails load only there, so the compact node never fetches them.

**The library is data, not code.** It lives in `nodes/text/prompt_library/` as `section/model/collection.json`: a new model or a whole new section — video prompts, say — is a new folder. Each preset may carry a picture in the collection's `previews/` folder, shown above the prompt. The format is described in the folder's own README. A preset that uses an undeclared field, or a preview that points outside its collection, is skipped with a line in the log.

**Use when:** editing images with Qwen Image 2.1 or FLUX.2 Klein and you want a tested wording for the task instead of writing it from scratch — or to hand a collaborator a prompt they can copy.


<a id="ideogram"></a>

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
