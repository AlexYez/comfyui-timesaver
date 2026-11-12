# TS H3 Regenerate Model

Загружает обычный Standard Regenerate из одного самодостаточного файла.
В ноде один вход `model` и одно поле **Regenerate** (`lora_name`).

1. Подключите чистую **pruned H3 FL2VA** напрямую из загрузчика. Для Full Reference используйте чистую **pruned H3 Ref2VA**. LoRA первичной генерации подключаются к отдельной ветке.
2. В поле **Regenerate** выберите `open_h3_regenerate_2k_lora_complete.safetensors` из `models/loras`.
3. Выход `model` подключите к **TS H3 Regenerate**. Denoise, шаги и размер задаются там.

При обновлении старого графа заново выберите complete-файл. Прежний раздельный
формат в пользовательском загрузчике больше не используется.

Единый файл уже содержит объединённую LoRA, все 50 специальных веток Sharp и
проверенную эталонную таблицу AdaLN. Дополнительные файлы Sharp и отдельный
Ref2VA-чекпойнт для таблицы не требуются. Факторы LoRA загружаются в bf16;
таблица и вычисление поправки Sharp сохраняют FP32, как в прежнем режиме по умолчанию.

Веса и сила адаптеров сохранены. Внутри остаётся авторская **LightX2V FL2VA Turbo
v1.2, 4-step, 768p** (`minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16.safetensors`).
Дополнительные Turbo / Pika / DMAD на эту ветку не накладывайте.

Нода загружает именно полный комплект TS. Обычный LoRA Loader не применяет
специальные ветки Sharp. Размер FP32-файла — около 13,69 ГБ.

## Сборка единого файла

В пак входит отдельная утилита сборки; сами веса с нодами не поставляются.
Используйте Python вашего ComfyUI. Из каталога ComfyUI:

```sh
python custom_nodes/comfyui-timesaver/nodes/video/latent_upscale/_regenerate_bundle.py --lora models/loras/open_h3_regenerate_2k_lora.safetensors --sharp models/loras/sharp_curve_branches.safetensors --reference models/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors --output models/loras/open_h3_regenerate_2k_lora_complete.safetensors --manifest models/loras/regenerate-complete-manifest.json
```

Подставьте свои пути. Первые два файла доступны в
[исходном релизе Standard](https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K/tree/main/lora).
Третий — уже установленный pruned Ref2VA: из него копируется только таблица AdaLN
32 КБ, сама модель не загружается. Утилита переносит тензоры без изменений,
проверяет хеши и сохраняет исходники. Требуется место для готового файла;
упаковка не уменьшает суммарный размер весов.
