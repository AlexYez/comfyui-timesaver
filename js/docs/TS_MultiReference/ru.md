# TS Multi Reference

Добавляет до трёх референсных изображений в conditioning как `reference_latents`. Сделана для Qwen-Image-Edit и подобных multi-reference пайплайнов. Per-slot выходы (`image_1` / `image_2` / `image_3`) с `ExecutionBlocker` для неподключённых слотов, автоматический resize к мегапиксельному бюджету с выравниванием по делителю (по умолчанию 32). Корректно обрабатывает RGBA + MASK-входы (композит на белый фон).

**Когда использовать:** Qwen-Edit / Flux-with-references пайплайны, принимающие несколько референсов.

---

Полный справочник нод: [README](https://github.com/AlexYez/comfyui-timesaver/blob/master/README.ru.md)
