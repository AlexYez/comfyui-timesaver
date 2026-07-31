# TS SAM Media Loader

Загружает изображение или видео и позволяет накликать позитивные/негативные точки прямо на превью первого кадра. На выходе: `IMAGE`, `AUDIO` (для видео) и `positive_coords`/`negative_coords` — STRING JSON ровно в том формате, который ждут нативные ComfyUI ноды **SAM3 Detect** / **SAM3 Video Track**. С опциональным входом `model` (SAM3) дополнительно отдаёт рендер `initial_mask`, готовый идти в `SAM3 Video Track`.

**Когда использовать:** строите SAM3-сегментацию/трекинг и хотите кликабельный UI для seed-точек вместо ручного ввода JSON.

---

Полный справочник нод: [README](https://github.com/AlexYez/comfyui-timesaver/blob/master/README.ru.md)
