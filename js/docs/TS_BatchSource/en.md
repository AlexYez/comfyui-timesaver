# TS Batch Source

Turns a folder, a text file or a plain count into a **job list**. Everything wired below this node runs once per item — that is ComfyUI's own batch engine: a list input makes the whole branch execute N times, and each run is independent of the others.

That independence is the point. A captioning model gets a fresh conversation per picture instead of one context that keeps growing for a hundred images.

Modes: `Images in folder` (natural order, so `img2` comes before `img10`), `Lines in text file`, `Count only`. Outputs: `item` (path / line / number), `index`, `total`, `seed`.

**Why there is a seed output.** A seed *widget* on the model node holds one number for all hundred calls, so a hundred iterations of the same task come back identical. This output gives every item its own derived seed — reproducible from the base value, different from its neighbours.

**Resuming.** `start_at` skips the finished part and keeps the original numbering, so TS Batch Write appends to the existing file instead of starting it over. `limit` caps the run.

⚠️ **It emits paths, not pictures** — deliberately. A hundred 4K frames passed along as images would sit in the output cache (roughly 10 GB) before the first caption is written. TS Batch Load Image reads them one at a time, so the peak stays at a single frame.

**⚠️ `one_per_run` — read this before a long batch.** With the list, ComfyUI finishes **every copy of one node before it starts the next** (measured on a live server: the loader logged items 1, 2, 3 and only then the writer logged 1/3, 2/3, 3/3). Two consequences: results reach disk only after the model has done every item, and a node that stamps per-item metadata — TS Image Prompt Injector — is overwritten by the last item before the save nodes run, so every picture ends up with the same prompt.

Switch `one_per_run` on and set ComfyUI's **Batch count** to the number of jobs. Each queued run is then a full pass through the graph: the picture is generated, stamped and saved before the next job starts. Slightly slower, and the only correct choice when you want to watch results arrive or need honest per-image metadata.

**Typical chain:** TS Batch Source → TS Batch Load Image → TS Qwen 3 VL → TS Batch Write.

**Use when:** captioning a folder into a dataset, or generating N unique prompts one fresh iteration at a time.


<a id="conditioning"></a>
### 🎨 Conditioning (1 node)

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
