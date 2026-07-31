# TS Qwen 3 VL V3

Мультимодальный Qwen 3 VL (image + video + text) локально. Встроенный выбор модели (Qwen 2B / 4B / 8B и uncensored-варианты), пресеты системных промптов ("Image Edit Command Translation", "Prompt Enhancement", …), 4-bit/8-bit квантование через `bitsandbytes`, поддержка FlashAttention-2, скачивание с HuggingFace на лету. С v9.5 тяжёлый пайплайн вынесен в общий `nodes/llm/_qwen_engine.py`, который переиспользует Super Prompt — исправления и оптимизации применяются к обеим нодам одновременно.

**Когда использовать:** описание изображений для промптов, перевод намерений пользователя в команды редактирования, VLM-driven пайплайны.

---

Полный справочник нод: [README](https://github.com/AlexYez/comfyui-timesaver/blob/master/README.ru.md)
