# TS Super Prompt RT

The same node, on a different engine: **Gemma 4 through Google's LiteRT-LM**,
the on-device runtime behind AI Edge. Measured on an RTX 3080 Ti Laptop against
the transformers path, same machine, same prompt:

| | TS Super Prompt RT (Gemma 4 E4B) | TS Super Prompt (Qwen3.5-4B, bf16) |
|---|---|---|
| speed | **43 tok/s** | 20 tok/s |
| VRAM | **~1.7 GB** | 8.5 GB |
| unload | 0.9 s | 3.4 s |

Twice the speed at a fifth of the memory — and the same model that writes the
prompt also **hears the recording**, so this node needs no Whisper at all. One
switch, `high_quality`, chooses between **E2B** (2.4 GB, quicker) and **E4B**
(3.4 GB, better) — for both jobs at once, because it is one model doing both.

**The model leaves the card when the work is done, and that is not an
optimisation.** LiteRT runs on WebGPU rather than CUDA, and ComfyUI's memory
manager cannot see a single byte of it: `torch.cuda.mem_get_info` reported the
same free memory whether Gemma was resident or not, while `nvidia-smi` moved by
3.5 GB. So a resident model is memory ComfyUI believes it still has, and the
sampler that follows plans accordingly. Unloading costs about a second and a
reload about five, which is why `keep_loaded` is off by default and its tooltip
says plainly what turning it on costs you.

**The context is 8192 tokens.** 4096 turned out to be only the runtime's default
when nobody names a size, not a limit of the files: at 8192 a code word planted
at the start of a 5976-token prompt came back exact, at the same speed, for
+0.5 GB on E4B (+0.37 GB on E2B). Every preset in the pack fits, checked with a
test, and a prompt that would not fit is refused with a readable message
*before* three gigabytes are read from disk rather than being silently
truncated.

**Every press can give a new wording.** Speculative decoding (MTP) made sampling
greedy: the same idea came back word for word at any seed and any temperature,
so "Enhance" twice returned the same text and the presets' temperatures did
nothing. It is off now, at a measured cost of about 10 % speed. Each preset's
answer ceiling and repetition penalty now reach the model too — without them one
answer in twelve could loop ("0, 0,0, …") for 46 seconds.

**The transcription prompt is written for Russian speech about software** —
Russian in Cyrillic, technical terms and product names in Latin script the way
the industry writes them (`ComfyUI`, `workflow`, `LoRA`, `Stable Diffusion`), and
direct speech in quotation marks. It also protects names it does not know:
measured on a real recording, «Artius Diffusion» used to come back as «Artus»
because the model snapped an unfamiliar name onto a familiar one.

**Speech is transcribed in 30-second segments** and stitched, with the overlap
removed. Thirty seconds is measured, not chosen: shorter cuts returned the same
amount of text but stuttered at the seams. A segment that comes back
suspiciously thin for its length is retried once — the model occasionally
decides one tidy sentence is a whole transcript.

**One switch, two jobs.** The toolbar puts what you *do* on the left — record,
attach, Enhance, pick a preset — and the two settings that apply to everything on
the right: **HQ** (E2B or E4B, for the prompt *and* the transcription, since it is
one model doing both) and a **memory chip** that keeps the model on the card
between runs. Turning the chip off releases the card immediately rather than at
the end of the next run.

**Only one thing talks to the engine at a time.** LiteRT holds a single engine
per process, and unloading it while it is writing takes the whole ComfyUI
process down with an access violation — measured, not theorised, when a second
model was loaded from another tab mid-generation. So every request queues, and a
waiting one says so in the progress panel instead of looking frozen.

**A forgotten Stop button no longer costs you a quarter of an hour.** The
microphone stops itself after three minutes, counting down out loud for the last
fifteen seconds, and says afterwards why it stopped — including when the
recording turned out to be silence, which is what a forgotten microphone usually
records. A recording that reaches the server another way is cut at five minutes
with a line in the log.

Three minutes is not a model limit. Google's documentation puts **one audio clip
at 30 seconds**, at 25 tokens per second, and the node already respects that by
transcribing in 30-second segments — measured to lose nothing: the same minute of
speech gave 141 words in two segments against 140 in a single oversized pass.
This runtime does not enforce the 30 s itself (85 s went through here, and only
at 90 s did it stop with `4688 >= 4096` under the old window), which is exactly
why the boundary is kept deliberately rather than by accident.

Models are pulled from
[`hfmaster/Gemma-4-RT`](https://huggingface.co/hfmaster/Gemma-4-RT) into
`models/LLM/litert` on first use — public, no token needed. These are the
**abliterated** builds of Gemma 4 E2B and E4B: the same weights and the same
speed, with the refusal behaviour trained out, which matters for a node whose
whole job is writing prompts. The runtime itself is not in
`requirements.txt` and installs separately:

```
python -m pip install litert-lm-api==0.16.1
```

`litert-lm-api` is the runtime itself; the plain `litert-lm` package installs the
same runtime plus a command-line tool, and keeps working if you already have it.

> **Windows, macOS (Apple Silicon) and Linux.** Wheels exist for all three
> (Linux: x86_64 and aarch64) — an earlier version of this page said Linux had
> none, which was wrong. Linux is not measured here; if its GPU backend does
> not come up, the node falls back to the CPU on its own.

**A prompt from another node.** The optional `prompt` input takes a string from the graph — TS Prompt Library, a text file, another LLM. Connected and not empty, it replaces the text field for the run and **goes out as it is**; tick **Enhance the incoming prompt on run** and Gemma enhances it with the chosen preset. Empty, the field is used — and enhanced on the run as before. With a recording on `audio` as well, the transcript is added after the wired prompt. While the input is connected, a panel above the field says so, holds that switch and shows the last run's result with a Copy button — shown there rather than written into the field, because a changed field would make ComfyUI re-run everything downstream on the next queue.

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
