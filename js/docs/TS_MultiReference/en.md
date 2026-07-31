# TS Multi Reference

Add up to three reference images as `reference_latents` into the conditioning stream. Built for Qwen-Image-Edit and similar multi-reference pipelines. Per-slot output (`image_1` / `image_2` / `image_3`) with `ExecutionBlocker` for unconnected slots, automatic resize to a megapixel budget aligned to a divisor (default 32). Handles RGBA + MASK inputs (composites onto white).

**Use when:** running Qwen-Edit / Flux-with-references style pipelines that accept multiple reference images.


## 🔰 Tips for Beginners

### Just starting out?

1. **Search by category** in the right-click menu: every node lives under `TS/<Category>`.
2. **Trust defaults**: every input has a sensible default. Change one parameter at a time to learn what it does.
3. **Use [TS Resolution Selector](#image)** as your latent-image source — it always returns a sampler-friendly size.
4. **Drop a [TS Animation Preview](#video) at the end** of any video graph to QA without re-running.
5. **Need a quick voice prompt?** [TS Super Prompt](#llm) — click the mic, describe your idea, get a polished prompt.

### My VRAM is tight, what should I use?

| Need | Try |
|---|---|
| Upscale a 4K image | TS Image Tile Splitter → upscaler → TS Image Tile Merger |
| Process only the face/object | TS Crop To Mask → upscaler/restorer → TS Restore From Crop |
| FP8 a model | TS Model Converter Advanced |

### Where do model files live?

| Node | Default folder |
|---|---|
| TS Lama Cleanup | `models/lama/` |
| TS Whisper | `models/whisper/` |
| TS Silero TTS | `models/silerotts/` |
| TS Silero Stress | `models/silero-stress/` |
| TS Qwen 3 VL | `models/LLM/` |
| TS Super Prompt | `models/LLM/` |
| TS Music Stems | demucs default cache |

You can override these with `extra_model_paths.yaml` — Timesaver respects ComfyUI's path resolution.


## 🛟 Troubleshooting

<details>
<summary><b>"Module not found" on startup</b></summary>

Check the startup log — Timesaver prints a load report. Missing optional dependencies appear under **Optional missing imports** with the file that needs them. Install with:

```bash
python -m pip install <missing_module>
```

Use the same Python ComfyUI runs from. On Windows portable: `python_embeded\python.exe -m pip install <module>`.
</details>

<details>
<summary><b>A node doesn't appear in the menu</b></summary>

Look at the startup log for **Module load issues**. The most common cause is a missing optional dependency — e.g. `py360convert` is required for the cube/equirect nodes. Install it and restart.
</details>

<details>
<summary><b>Workflow fails after updating</b></summary>

Timesaver freezes node ids and inputs across versions on purpose. If something breaks after `git pull`:
1. Check `doc/migration.md` for breaking changes.
2. Make sure `pip install -r requirements.txt` was run.
3. Restart ComfyUI fully — not just refresh the browser tab.
</details>

<details>
<summary><b>OOM (out of memory) errors</b></summary>

- Reduce `process_resolution` (BiRefNet) or `compute_max_side` (Color Match).
- For upscaling, use `TS Image Tile Splitter` + tiled processing.
- For LLM, drop precision to int8 or int4 (`TS Qwen 3 VL V3` → `precision=int8`).
- Use `unload_after_generation=True` to free model VRAM after each run.
</details>


## 🗂️ Repo Layout

```text
comfyui-timesaver/
├─ nodes/                  # 64 modules: 61 nodes + 3 sampler/scheduler injectors
├─ js/                     # frontend extensions for DOM-widget nodes
├─ doc/screenshots/        # node screenshots (this README uses them)
├─ requirements.txt        # runtime dependencies
└─ pyproject.toml          # version + ComfyRegistry metadata
```


## 📜 License & Credits

Licensed under the terms in [LICENSE.txt](LICENSE.txt).

**Built on top of:**
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) — the graph engine and V3 API.
- [BiRefNet](https://github.com/zhengpeng7/BiRefNet) — background removal.
- [LaMa](https://github.com/advimman/lama) — image inpainting.
- [Whisper](https://github.com/openai/whisper) — speech recognition.
- [Demucs](https://github.com/facebookresearch/demucs) — music source separation.
- [Silero](https://github.com/snakers4/silero-models) — Russian TTS / stress.
- [Qwen](https://github.com/QwenLM/Qwen3-VL) — vision-language model.
- [Spandrel](https://github.com/chaiNNer-org/spandrel) — model loading for upscalers.
- [py360convert](https://github.com/sunset1995/py360convert) — 360° conversions.
- [RIFE](https://github.com/megvii-research/ECCV2022-RIFE) / [FILM](https://github.com/google-research/frame-interpolation) — frame interpolation.

**Maintainer:** [@AlexYez](https://github.com/AlexYez)

**Issues / feature requests:** https://github.com/AlexYez/comfyui-timesaver/issues


<div align="center">

**Found this useful?** ⭐ Star the repo to help others find it.

</div>

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
