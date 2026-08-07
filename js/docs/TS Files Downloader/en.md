# TS Files Downloader (Ultimate)

Multi-file downloader that takes a list of `URL <space> target_path` lines and downloads them sequentially. Auto-replaces HuggingFace mirrors with reachability check across the full mirror list, supports `models/<subdir>` aliases, resumes interrupted downloads, validates archives against zip-slip on auto-unzip, and shows progress (including SHA256 verification). Handy for one-shot pulling all assets a workflow needs.

**Get models from workflow.** The button on the node fills that list for you: it walks the open graph — **including inside subgraphs**, where template loaders normally live — and collects every model it needs. It reads the `{name, url, directory}` metadata ComfyUI stamps onto each loader, cross-checks it against the workflow's Markdown note, and falls back to the loader's own filename when neither carries a link. You get a report first; **Append** adds only what is missing and never rewrites lines you wrote by hand, **Replace list** starts over.

Models you already have are listed too, on purpose: the list travels with the workflow, so whoever you send it to still needs those lines.

The folder it proposes is the one your models of that category are **already in**. ComfyUI reads two directories per category — `clip` and `text_encoders`, `unet` and `diffusion_models` — and both are real; if your encoders live in `clip`, that is where the download is aimed, not at the empty folder next to it. A line you wrote in the list yourself is never rewritten.

**Cancelling the run stops everything.** ComfyUI's cancel button ends the file in flight *and* every file still queued after it. A partial file is kept as `.part`, so the next run resumes from where it stopped instead of starting over. Progress is one bar for the whole list, from the first model to the last.

**The rest of the workflow waits.** This node brings in the models the graph has nothing to load without, so it holds the run until the last file has landed rather than handing the graph back while the bytes are still arriving.

**Use when:** distributing a workflow that needs N specific models — open it, press the button, and the node is filled in.

> **Network behaviour (for security review):** the node issues standard HTTPS `HEAD`/`GET` requests **only** to the URLs you type into `file_list`, identifying itself with an honest `comfyui-timesaver/<version>` User-Agent. It does **not** execute, import, or run anything it downloads — files are written to disk only. There are no hardcoded callback/telemetry endpoints. Optional `hf_token` / `modelscope_token` are sent as an `Authorization` header **only** to their matching host (HuggingFace / ModelScope respectively) and are never logged or forwarded elsewhere. Auto-unzip is validated against zip-slip path traversal before extraction.

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
