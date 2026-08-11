# TS Qwen 3

Multimodal Qwen 3 VL (image + video + text) running locally. Built-in model picker (Qwen 2B / 4B / 8B variants and uncensored mods), system-prompt presets ("Image Edit Command Translation", "Prompt Enhancement", …), 4-bit/8-bit quantisation via `bitsandbytes`, FlashAttention-2 support, on-the-fly download from HuggingFace. Since v9.5 the heavy pipeline lives in a shared `nodes/llm/_qwen_engine.py` reused by Super Prompt — bug fixes and perf improvements land in both nodes at once.

**Use when:** describing images for prompts, translating user intents into edit commands, building VLM-driven pipelines.

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
