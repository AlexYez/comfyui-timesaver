# TS Model Scanner

Инспекция любого `.safetensors` (из `models/diffusion_models/`) или загруженного `MODEL`. Печатает подробный отчёт: имя, shape, dtype, device каждого параметра + сводная статистика по dtype.

**Когда использовать:** дебаг загрузки модели, проверка точности (fp16 vs fp8 vs bf16), знакомство с незнакомым чекпоинтом.

---

Полный справочник нод: [README](https://github.com/AlexYez/comfyui-timesaver/blob/master/README.ru.md)
