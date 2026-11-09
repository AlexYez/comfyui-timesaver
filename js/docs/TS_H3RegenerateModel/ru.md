# TS H3 Regenerate Model

Подключите чистую **pruned H3 FL2VA** для стандартного уточнения или чистую **pruned H3
Ref2VA** для режима Full Reference. Подайте модель прямо из загрузчика, до LoRA и патчей
блоков. Нода клонирует модель и применяет 312 проекций объединённой LoRA и 50 веток
Sharp. Ветка первой генерации сохраняет собственные настройки.

Файлы в `models/loras` (или в объявленной внешней папке):
- `open_h3_regenerate_2k_lora.safetensors`;
- `sharp_curve_branches.safetensors`.

`reference_checkpoint` — установленный `minimax_h3_ref2va_pruned_int8_convrot.safetensors`:
из него читается только таблица 32 КБ, а её контрольная сумма сверяется с эталоном.
Это чтение не загружает модель на GPU. Для Full Reference подайте модель отдельно на
`model`: её таблица должна совпасть с эталонной. Поддерживаются pruned H3 с 50 блоками
и таблицей AdaLN 1025×8.

`lora_precision=bf16` занимает около 6,2 ГБ ОЗУ для факторов; `fp32` — около 12,45 ГБ.
При загрузке исправляется порядок половин SwiGLU и разделяются Q/K/V внутри общей проекции.
Дополнительные Turbo / Pika / DMAD на эту ветку не накладывайте.

[Подключение TS H3 Regenerate](https://github.com/AlexYez/comfyui-timesaver/blob/master/js/docs/TS_H3Regenerate/ru.md)
