# TS RT Prompt Enhancer

The same Gemma 4 on LiteRT-LM as TS Super Prompt RT, as an **ordinary graph
node** — what TS Qwen 3 is to TS Super Prompt. No buttons, no panel: text,
pictures and sound in, text out, run by the queue like any other node. Build it
into a workflow wherever a prompt needs enhancing, a picture or a sound
describing, speech writing out or a phrase translating.

It reads **the same preset file** as TS Qwen 3 and both Super Prompts
(`nodes/qwen_3_vl_presets.json`): pick a preset, or choose *Your instruction*
and connect your own system prompt to `custom_system_prompt`. Each preset brings
its own sampling settings and answer ceiling; `max_new_tokens` at **0** keeps
the preset's, any other number replaces it. The `seed` works — same seed, same
answer; change it for a different wording.

**Pictures:** one, two or a whole clip's frames on `images`. At most **four**
reach the model; a longer batch is sampled evenly from the first frame to the
last rather than cut after four. There is deliberately no VIDEO input: ComfyUI
would decode the whole clip into frames — a minute of 1080p is tens of
gigabytes — for the four the model can take. Feed a clip's frames to `images`
and its soundtrack to `audio`.

**Sound** on `audio`, in one of two modes (`audio_mode`):

- **listen** (default) — the model hears the recording itself: music, sounds, a
  short phrase. Good with the audio presets (ACE-Step, Minimax, Stable Audio
  SFX) to write a prompt *from* a sound. Only the **first 30 seconds** — the
  longest clip Gemma is trained on. Feeding a long recording as several clips
  does not work: measured, two different tracks in one message came back as the
  same song described twice.
- **transcribe** — speech of any length (up to five minutes) is written out in
  30-second pieces by TS Super Prompt RT's transcriber and added to the prompt,
  which the preset then works on. Tuned for Russian speech with English terms.

With pictures or a recording connected the text may stay empty.

The model is **unloaded after every run** by default, for the same reason as in
TS Super Prompt RT: ComfyUI cannot see the memory it takes. `keep_loaded` keeps
it for faster repeats, at that price. `enable` off passes the prompt through
without loading anything. When something goes wrong — nothing to work from, a
prompt too long for the window — the run stops with a readable message instead
of sending the word "ERROR" downstream as a prompt.

Models (E2B 2.4 GB, E4B 3.4 GB) download into `models/LLM/litert` on first use;
the runtime installs as described for TS Super Prompt RT.

**Use when:** a workflow needs an LLM step without a hand on a button — batch
captioning, prompt enhancement inside a pipeline, a music or sound prompt from a
reference track, a voice note turned into a prompt, translation before a text
encoder.

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
