# TS Files Downloader

Multi-file downloader that takes a list of `URL <space> target_path` lines and downloads them sequentially. Auto-replaces HuggingFace mirrors with reachability check across the full mirror list, supports `models/<subdir>` aliases, resumes interrupted downloads, validates archives against zip-slip on auto-unzip, and shows progress (including SHA256 verification). Handy for one-shot pulling all assets a workflow needs.

**Download now, without running the graph.** The second button on the node pulls the whole list straight away — the same engine, the same tokens, mirrors and unzip settings as a normal run. It shows `3/10 · 42% · model.safetensors` while it works, and pressing it again cancels: the partial file stays as `.part` and the next attempt resumes from there.

That button is what makes `enable` useful as a mode. Turn `enable` off and the node stops doing anything when the workflow runs — no checks, no downloads — while you still fetch the models by hand, once, when you actually need them. The button ignores `enable` on purpose: it is the one way left to download.

**The list reads as two things, not one.** Each line is `<url> → <folder>`. The arrow is there to be read: a long address wraps in the field, and a folder pressed against its tail looks like part of the link. A plain space still works, so lists written earlier — and lists arriving with someone else's workflow — keep running.

**Every model says where it stands.** A dot in front of each line: green — the file is on disk, red — it is not, amber — a `.part` is waiting to be resumed, grey — nothing is known yet (no folder given, or the check has not run). The check reads the disk only, never the network, and runs when the node is drawn, when the list changes, and when a download ends. While a model is being fetched its own line carries a progress bar, so a list of ten answers "has *this* one arrived?" without counting.

**Settings live behind a button.** Mirrors, tokens, proxy, chunk size, integrity mode and `enable` are all in one panel inside the node, opened by **Settings** and closed by **Done**. The node itself stays what it is for: the list, and the two buttons under it. Nothing about the inputs changed — the same eleven, in the same order, with the same defaults; a workflow saved earlier opens with its values in place.

**Get models from workflow.** The button on the node fills that list for you: it walks the open graph — **including inside subgraphs**, where template loaders normally live — and collects every model it needs. It reads the `{name, url, directory}` metadata ComfyUI stamps onto each loader, cross-checks it against the workflow's Markdown note, and falls back to the loader's own filename when neither carries a link. You get a report first; **Append** adds only what is missing and never rewrites lines you wrote by hand, **Replace list** starts over.

Models you already have are listed too, on purpose: the list travels with the workflow, so whoever you send it to still needs those lines.

**Which loaders it understands is asked of your ComfyUI, not written down here.** Every loader's dropdown is filled from a `models/` folder, so the options themselves say which folder they came from — the node reads that from the running server and maps each widget of each installed node to its folder. That is why a model in `Load Latent Upscale Model`, or in a node from a pack installed yesterday, is found the same as a checkpoint. A written-down table could not do it: on the maintainer's machine 49 installed node types own a model widget such a table never heard of, two of them from ComfyUI itself. One node with two model widgets from different folders keeps them apart — a text encoder and a checkpoint on the same loader go to their own places.

The folder it proposes is the one your models of that category are **already in**. ComfyUI reads two directories per category — `clip` and `text_encoders`, `unet` and `diffusion_models` — and both are real; if your encoders live in `clip`, that is where the download is aimed, not at the empty folder next to it. A line you wrote in the list yourself is never rewritten.

**Cancelling the run stops everything.** ComfyUI's cancel button ends the file in flight *and* every file still queued after it. A partial file is kept as `.part`, so the next run resumes from where it stopped instead of starting over. Progress shows twice: one bar for the whole list, and a small one on the line of the model in flight.

**The rest of the workflow waits.** This node brings in the models the graph has nothing to load without, so it holds the run until the last file has landed rather than handing the graph back while the bytes are still arriving.

**Use when:** distributing a workflow that needs N specific models — open it, press the button, and the node is filled in.

> **Network behaviour (for security review):** the node issues standard HTTPS `HEAD`/`GET` requests **only** to the URLs you type into `file_list`, identifying itself with an honest `comfyui-timesaver/<version>` User-Agent. It does **not** execute, import, or run anything it downloads — files are written to disk only. There are no hardcoded callback/telemetry endpoints. Optional `hf_token` / `modelscope_token` are sent as an `Authorization` header **only** to their matching host (HuggingFace / ModelScope respectively) and are never logged or forwarded elsewhere. Auto-unzip is validated against zip-slip path traversal before extraction.

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
