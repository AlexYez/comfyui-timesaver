# TS Super Prompt

Prompt enhancement node with a built-in **voice button** — speak your idea, Whisper transcribes it (with cinematography-aware grammar fixes), then a small Qwen3 model expands it into a rich prompt. Optional image input for image-conditioned prompting. Two modes: fast turbo or high-quality. Internals split (v9.5) into `nodes/llm/super_prompt/` (`_helpers`, `_voice`, `_qwen` over the shared Qwen engine) so the prompt-enhancement path stays in sync with TS Qwen 3 VL V3.

**Two reference images.** Drop an image straight onto the node — from the Artius browser, from the desktop, or from another node's preview. One image is a reference; drop a second and the two become the **first and last frame** of the shot, which the model is told explicitly. The second picker appears once the first is taken. The thumbnails carry a **1** and a **2** so you can see which frame is which; drag one onto the other to swap them. Remove the first of a pair and the second takes its place.

**Frames can come from the graph too.** The optional `images` input takes a plain ComfyUI `IMAGE`. A single image is a plain reference; in a batch the **first image is the first frame** and the **last is the last frame**. One input rather than a socket per frame: the order inside the batch is what says which frame is which, so three or four frames need no new wiring (up to four are read). A wired input **wins** over images attached in the node, and wins as a whole — the batch already states the order, and mixing it with attachments could only produce a sequence nobody asked for. Each frame is shrunk to about 1 MP by area on the way in — no crop, and no upscale when it is already smaller.

**Presets for MiniMax H3.** `Video Prompt Enhance H3` follows H3's own schema for speech: a stable speaker ID `(S1)`, the voice described outside the dialogue block, and inside it only the language tag and the line, verbatim — `<d>[Russian] Привет, красавчик!</d>`. For Russian it always asks for *a native Russian speaker with natural, neutral standard Russian pronunciation and authentic native Russian prosody* instead of *a Russian accent*: the first gets a Russian voice, the second an English one with Russian colouring.

**The Enhance button sees the input too.** The value on a wire does not exist until something computes it, so the button computes it — but only the branch that feeds this input. The nodes that branch depends on are pulled out of the graph into a prompt of their own and run; nothing else in the workflow is in that prompt, so no sampler and no save fires along with it. Two loaders joined into a batch, a resize, a crop — all of it works, and none of it needs a run of the whole workflow first. Nothing is remembered between presses on purpose: swap the file behind a loader and the graph reads exactly the same, so a remembered result would quietly enhance the picture you replaced. ComfyUI does the caching one level down, by what the nodes actually read. If the branch produces no image, the node says so instead of quietly enhancing the text alone.

**On-screen text is not translated.** Anything in quotes is what should appear in the picture — a sign, a title, a lyric. It is copied through unchanged, in its original language, while everything else is translated to English. An obliging translation used to turn a Russian shop sign into an English one nobody asked for.

**The `Video Prompt Enhance` and `Image Prompt Enhance` presets** are written for a small model (Qwen 2B/4B): short numbered steps, an explicit output format, one example. The video preset is tuned for LTX-2.3 and MiniMax H3 — camera move first, then the motion in order, then light and mood.

**Use when:** quick prompt brainstorming, voice-driven workflows, or bridging a sketchy idea into a production-ready prompt.


<a id="text"></a>
### 📝 Text & Prompts (4 nodes)

Build, randomise and manage prompts at scale.

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
