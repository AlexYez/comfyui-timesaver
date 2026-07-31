# TS Video Depth

Покадровое определение глубины через Depth-Anything, оптимизированное для видео (временна́я согласованность). В v9.4 — полная переделка GPU-пайплайна: SDPA-attention, TPDF-дизеринг на выходе, sub-chunk обработка длинных клипов, численно-эквивалентный DPT tail. Результат тот же, скорость на RTX-картах резко выше.

**Когда использовать:** depth-aware ControlNet, parallax-эффекты, 3D-репроекция.

---

Полный справочник нод: [README](https://github.com/AlexYez/comfyui-timesaver/blob/master/README.ru.md)
