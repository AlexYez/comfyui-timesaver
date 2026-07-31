# TS Video Upscale With Model

Покадровый апскейл любой моделью, поддерживаемой spandrel (RealESRGAN, 4x-Ultrasharp и т.п.). Три стратегии устройства: `auto`, `load_unload_each_frame` (мало VRAM, медленно), `keep_loaded` (быстро, больше VRAM), `cpu_only`.

**Когда использовать:** апскейл видео без OOM или batch-апскейл с контролируемым расходом VRAM.

---

Полный справочник нод: [README](https://github.com/AlexYez/comfyui-timesaver/blob/master/README.ru.md)
