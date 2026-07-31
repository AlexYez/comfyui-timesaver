# TS Multi Reference

Добавляет до трёх референсных изображений в conditioning как `reference_latents`. Сделана для Qwen-Image-Edit и подобных multi-reference пайплайнов. Per-slot выходы (`image_1` / `image_2` / `image_3`) с `ExecutionBlocker` для неподключённых слотов, автоматический resize к мегапиксельному бюджету с выравниванием по делителю (по умолчанию 32). Корректно обрабатывает RGBA + MASK-входы (композит на белый фон).

**Когда использовать:** Qwen-Edit / Flux-with-references пайплайны, принимающие несколько референсов.


## 🔰 Подсказки для новичков

### Только начинаете?

1. **Ищите по категориям** в правом клике: каждая нода живёт под `TS/<Категория>`.
2. **Доверяйте дефолтам**: у каждого входа есть разумное значение по умолчанию. Меняйте по одному параметру, чтобы понять, что он делает.
3. **Используйте [TS Resolution Selector](#image)** как источник latent-изображения — он всегда возвращает sampler-friendly размер.
4. **Бросьте [TS Animation Preview](#video) в конец** любого видео-графа, чтобы делать QA без перезапуска.
5. **Нужен быстрый голосовой промпт?** [TS Super Prompt](#llm) — кликнули по микрофону, описали идею, получили готовый промпт.

### VRAM не хватает, что использовать?

| Задача | Решение |
|---|---|
| Апскейл 4K-картинки | TS Image Tile Splitter → апскейлер → TS Image Tile Merger |
| Обработать только лицо/объект | TS Crop To Mask → апскейл/restore → TS Restore From Crop |
| FP8 модель | TS Model Converter Advanced |

### Где модели лежат?

| Нода | Папка по умолчанию |
|---|---|
| TS Lama Cleanup | `models/lama/` |
| TS Whisper | `models/whisper/` |
| TS Silero TTS | `models/silerotts/` |
| TS Silero Stress | `models/silero-stress/` |
| TS Qwen 3 VL | `models/LLM/` |
| TS Super Prompt | `models/LLM/` |
| TS Music Stems | demucs default cache |

Можно переопределить через `extra_model_paths.yaml` — Timesaver уважает path-resolution ComfyUI.


## 🛟 Если что-то сломалось

<details>
<summary><b>"Module not found" при старте</b></summary>

Смотрите startup-лог — Timesaver печатает load report. Отсутствующие опциональные зависимости появляются под **Optional missing imports** с указанием, какому файлу они нужны. Установите:

```bash
python -m pip install <missing_module>
```

Используйте тот же Python, что запускает ComfyUI. На Windows portable: `python_embeded\python.exe -m pip install <module>`.
</details>

<details>
<summary><b>Нода не появилась в меню</b></summary>

В startup-логе ищите **Module load issues**. Самая частая причина — отсутствие опциональной зависимости (например `py360convert` нужен для cube/equirect нод). Установите её и перезапустите.
</details>

<details>
<summary><b>Workflow ломается после обновления</b></summary>

Timesaver специально замораживает id нод и входы между версиями. Если что-то сломалось после `git pull`:
1. Проверьте `doc/migration.md` на breaking changes.
2. Убедитесь, что `pip install -r requirements.txt` запускался.
3. Полностью перезапустите ComfyUI — не просто обновите вкладку браузера.
</details>

<details>
<summary><b>OOM (out of memory)</b></summary>

- Уменьшите `process_resolution` (BiRefNet) или `compute_max_side` (Color Match).
- Для апскейла используйте `TS Image Tile Splitter` + тайловую обработку.
- Для LLM понизьте точность до int8 или int4 (`TS Qwen 3 VL V3` → `precision=int8`).
- `unload_after_generation=True` освобождает VRAM модели после каждого запуска.
</details>


## 🗂️ Структура репозитория

```text
comfyui-timesaver/
├─ nodes/                  # 64 модуля: 61 нода + 3 инжектора samplers/schedulers
├─ js/                     # frontend extensions для DOM-widget нод
├─ doc/screenshots/        # скриншоты нод (этот README их использует)
├─ requirements.txt        # runtime-зависимости
└─ pyproject.toml          # версия + ComfyRegistry-метаданные
```


## 📜 Лицензия и благодарности

Лицензия — см. [LICENSE.txt](LICENSE.txt).

**Построено на:**
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) — graph engine и V3 API.
- [BiRefNet](https://github.com/zhengpeng7/BiRefNet) — удаление фона.
- [LaMa](https://github.com/advimman/lama) — image inpainting.
- [Whisper](https://github.com/openai/whisper) — распознавание речи.
- [Demucs](https://github.com/facebookresearch/demucs) — разделение музыки на источники.
- [Silero](https://github.com/snakers4/silero-models) — русский TTS / ударения.
- [Qwen](https://github.com/QwenLM/Qwen3-VL) — vision-language модель.
- [Spandrel](https://github.com/chaiNNer-org/spandrel) — загрузка апскейлеров.
- [py360convert](https://github.com/sunset1995/py360convert) — 360° конвертация.
- [RIFE](https://github.com/megvii-research/ECCV2022-RIFE) / [FILM](https://github.com/google-research/frame-interpolation) — интерполяция кадров.

**Мейнтейнер:** [@AlexYez](https://github.com/AlexYez)

**Issues / feature requests:** https://github.com/AlexYez/comfyui-timesaver/issues


<div align="center">

**Понравилось?** ⭐ Поставьте звезду, чтобы помочь другим найти проект.

</div>

---

Полный справочник нод: [README](https://github.com/AlexYez/comfyui-timesaver/blob/master/README.ru.md)
