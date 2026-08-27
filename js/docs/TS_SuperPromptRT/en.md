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

**The context is 4096 tokens** — the artefact's limit, not Gemma's. Every preset
in the pack fits, checked with a test, and a prompt that would not fit is
refused with a readable message *before* three gigabytes are read from disk
rather than being silently truncated.

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
at 90 s did it stop with `4688 >= 4096`), which is exactly why the boundary is
kept deliberately rather than by accident.

Models are pulled from
[`hfmaster/Gemma-4-RT`](https://huggingface.co/hfmaster/Gemma-4-RT) into
`models/LLM/litert` on first use — public, no token needed. These are the
**abliterated** builds of Gemma 4 E2B and E4B: the same weights and the same
speed, with the refusal behaviour trained out, which matters for a node whose
whole job is writing prompts. The runtime itself is not in
`requirements.txt` and installs separately:

```
python -m pip install litert-lm==0.16.1
```

> **Windows and macOS only.** LiteRT-LM publishes no Linux wheels. On Linux the
> node loads and explains itself instead of failing obscurely — use TS Super
> Prompt, which runs on transformers everywhere.

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
