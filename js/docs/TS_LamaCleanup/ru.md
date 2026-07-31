# TS Lama Cleanup

Встроенный инпейнтинг через LaMa — рисуйте маску прямо на канвасе ноды (кисть + undo/redo + reset), затем запускайте, чтобы заполнить. Хранит промежуточные правки по сессиям, не нужно ходить в Photoshop. С v9.3 архитектура — чистый PyTorch (без зависимости от upstream `lama-cleaner`), веса загружаются из `.safetensors` в `models/lama/`, а не из pickled `.ckpt`.

**Когда использовать:** убрать туристов с фото, стереть водяные знаки, починить артефакты, прототипировать чистку перед тяжёлым inpainter'ом.

---

Полный справочник нод: [README](https://github.com/AlexYez/comfyui-timesaver/blob/master/README.ru.md)
