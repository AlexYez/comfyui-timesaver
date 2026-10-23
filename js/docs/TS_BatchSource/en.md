# TS Batch Source

Turns a folder, a text file or a plain count into a **job list**. Everything wired below this node runs once per item — that is ComfyUI's own batch engine: a list input makes the whole branch execute N times, and each run is independent of the others.

That independence is the point. A captioning model gets a fresh conversation per picture instead of one context that keeps growing for a hundred images.

Modes: `Files in folder` — the paths of a folder's files of any type (natural order, so `img2` comes before `img10`), `Lines in text file`, `Count only` and `Text files in folder` — a folder with one prompt per file (`.txt` / `.md`): one file is one job, its whole text, line breaks inside kept, the same natural order, empty files skipped. Outputs: `item` (path / line / number / prompt text), `index`, `total`, `seed`.

**`Files in folder` — any files, not just pictures.** The node hands out paths, so the file type is none of its business: pictures go to TS Batch Load Image, videos to the `path` socket of TS Video Loader, sound to the `path` socket of TS Audio Loader. The `file_types` list narrows the choice: `All` — every file; `Images`, `Videos` or `Audio` — exactly the formats the matching loader opens. `include_subfolders` walks nested folders too, keeping natural order per path level (`a/2` before `a/10` before `b/1`). Hidden files and folders (a leading dot) and the system's `Thumbs.db` / `desktop.ini` are always skipped.

⚠️ Until 23.10.2026 the mode was called `Images in folder`. Graphs saved with it open and run as before — images only, so `.txt` captions lying beside them do not end up in an image loader. The old name is no longer in the list.

**`Prompts from images in folder` — prompts instead of pictures.** Each image in the folder gives the positive prompt it was made with, and that prompt becomes the job — to regenerate a set with another model, say, or to build a prompt file. It reads ComfyUI metadata (the graph: from the sampler's `positive` to the text encoder, through switches, `PreviewAny`, `StringConcatenate`, TS Super Prompt and TS Style Prompt Selector), A1111 / Forge / Civitai (`parameters` in PNG, UserComment in JPEG and WebP), ComfyUI's WebP and JPEG, and NovelAI. The negative is never taken. An image without a prompt — a photo, an upscale, a frame whose prompt an LLM wrote during the run — is skipped with a line in the log: the LLM's text is not stored in the file, and the node will not hand out the instruction given to the model in its place. Such a prompt is kept when TS Image Prompt Injector sits before the save.

**Why there is a seed output.** A seed *widget* on the model node holds one number for all hundred calls, so a hundred iterations of the same task come back identical. This output gives every item its own derived seed — reproducible from the base value, different from its neighbours.

**Resuming.** `start_at` skips the finished part and keeps the original numbering, so TS Batch Write appends to the existing file instead of starting it over. `limit` caps the run.

⚠️ **It emits paths, not pictures** — deliberately. A hundred 4K frames passed along as images would sit in the output cache (roughly 10 GB) before the first caption is written. TS Batch Load Image reads them one at a time, so the peak stays at a single frame.

**⚠️ `one_per_run` — read this before a long batch.** With the list, ComfyUI finishes **every copy of one node before it starts the next** (measured on a live server: the loader logged items 1, 2, 3 and only then the writer logged 1/3, 2/3, 3/3). Two consequences: results reach disk only after the model has done every item, and a node that stamps per-item metadata — TS Image Prompt Injector — is overwritten by the last item before the save nodes run, so every picture ends up with the same prompt.

Switch `one_per_run` on and set ComfyUI's **Batch count** to the number of jobs. Each queued run is then a full pass through the graph: the picture is generated, stamped and saved before the next job starts. Slightly slower, and the only correct choice when you want to watch results arrive or need honest per-image metadata.

**Typical chain:** TS Batch Source → TS Batch Load Image → TS Qwen 3 VL → TS Batch Write.

**Use when:** captioning a folder into a dataset, or generating N unique prompts one fresh iteration at a time.


<a id="conditioning"></a>

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
