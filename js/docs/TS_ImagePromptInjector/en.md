# TS Image Prompt Injector

Injects a custom string into the workflow's positive prompt at runtime — useful when you generate prompts dynamically (LLM nodes) and want them to land in the actual `CLIPTextEncode` connected to the sampler. Operates on the workflow graph, leaves the image unchanged.

**Use when:** chaining an LLM that writes prompts and you want the next sampler to use the result without manually rewiring text encoders.


<a id="video"></a>
### 🎬 Video (8 nodes)

Reading and writing video files, frame interpolation, model-based upscale, depth, animation preview.

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
