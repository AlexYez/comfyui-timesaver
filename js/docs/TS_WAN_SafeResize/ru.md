# TS WAN Safe Resize

Аналог Qwen Safe Resize, но для WAN-Video. Определяет ближайшую пропорцию (16:9, 9:16, 1:1) и выбирает один из трёх пресетов качества: Fast (240p), Standard (480p / 832p), High (720p / 1280p). Строка `interconnection_in/out` позволяет нескольким WAN-нодам делиться одним уровнем качества.

**Когда использовать:** подготовка кадров для WAN i2v / t2v моделей.

---

Полный справочник нод: [README](https://github.com/AlexYez/comfyui-timesaver/blob/master/README.ru.md)
