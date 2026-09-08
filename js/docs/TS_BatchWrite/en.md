# TS Batch Write

Writes each batch result the moment it is ready, instead of holding everything until the run ends. A batch that dies at item 90 leaves 89 results on disk rather than nothing.

**Three layouts, and the middle one closes a loop.** `One file, blocks` separates results with a blank line (read back by TS Batch Prompt Loader). `One file, one line per item` collapses each result to a single line — read back by TS Batch Source in `Lines in text file` mode, which is how a file of captions becomes a file of generation jobs without any conversion. `One .txt per item` names the file after its picture, the layout caption datasets expect.

**Watching it happen.** ComfyUI collects the UI previews of every iteration and sends them in a single event after the last one, so a hundred pictures otherwise appear all at once, at the end. Connect `image` here and the current result is pushed through the **progress bar** instead — you see item 47 while it is item 47.

`index 0` starts the file fresh, so a new run never silently continues the previous one's file; any other index appends, which is what makes a resumed run add to what is already there.

**Use when:** any long batch whose results you want on disk — and in front of you — before it finishes.


<a id="utils"></a>
### 🛠️ Utils (8 nodes)

Tiny helpers that make the graph less cluttered.

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
