---
name: reference_audit_v994
description: "История: Read-only аудит всего пака 2026-07-31 → фиксы в v9.94; что исправлено, что сознательно НЕ трогали, и почему находка про redirect-утечку токена оказалась ложной"
metadata:
  type: reference
  status: historical
---

История: Read-only аудит всего пака 2026-07-31 → фиксы в v9.94; что исправлено, что сознательно НЕ трогали, и почему находка про redirect-утечку токена оказалась ложной

**Историческая запись, не описание текущей реализации и не действующее разрешение.** Даты и результаты ниже относятся к указанным сессиям. Пути `tests/`, `tools/`, `dist/` и `archive/` с артефактами — локальная лаборатория; они не гарантированы в чистом клоне.

# Whole-pack read-only аудит → v9.94 (2026-07-31)

Аудит по скиллу `universal-read-only-code-auditor` (6 доменных агентов: HTTP-роуты,
downloader+scan, Ideogram/LLM, фронтенд, корректность нод, зависимости/CI/тесты/доки),
затем реализация всех подтверждённых находок. Продолжает [reference_audit_v978](reference_audit_v978.md) и
[reference_pack_review_2026_07](reference_pack_review_2026_07.md). Коммит `3202657`, версия 9.93 → 9.94.

## Топ-находка (P1): скан workflow превратил недоверенный JSON в цель загрузки

Кнопка «Взять модели из workflow» (v9.90+) пишет `properties.models[].directory`
и значения виджетов лоудеров прямо в `file_list`, а `_resolve_target_directory`
принимал всё, что выглядит абсолютным. Крафтовый workflow с
`directory: "C:/…/Startup"` или `%APPDATA%/…` → файл с чужого сервера пишется в
автозапуск. Раньше абсолютный путь мог появиться только если его напечатал
пользователь — фича автоматизировала подстановку, гард не менялся.

Фикс двухслойный:
- **JS** `safeSegments()` + `LOCATION_SEGMENT` в `js/files/ts-files-downloader.js`:
  сегмент с `:`, ведущим `~`, `%VAR%` или `$VAR` — это не имя папки, а место;
  такой target отбрасывается целиком (запись падает в noLink, не в «куда-нибудь»).
- **Python** `_is_within()` + проверка в трёх относительных ветках. Абсолютный
  target ОСТАВЛЕН рабочим (документированная фича), но логируется WARNING, если
  он ведёт вне ComfyUI.

## Остальное исправленное

- `/ts_super_prompt/enhance` описывал ЛЮБУЮ картинку на диске (fallback на сырой
  абсолютный путь без проверки корня) → `_is_inside_allowed_root` (input/output/temp).
  Три старых теста кодировали дырку и обновлены; их docstring утверждал, что
  строгая проверка «сломает фронтенд» — проверено: фронтенд всегда шлёт
  annotated-путь из `/upload/image`, не абсолютный.
- `/ts_sam_media_loader/preview_mask`: убран fallback «принять любой путь как
  чекпойнт» (OOM одним запросом; pickle при `--disable-safe-unpickle`).
- `hf_token` больше не уезжает на зеркало, названное в workflow (`hf_domain` —
  сериализуемый виджет). Allowlist статический = `_is_hf_url`.
- LaMa: `_acquire_model()` — локальная ссылка под локом; `unload_model` во время
  инференса больше не даёт TypeError и не освобождает VRAM под forward-pass.
- Whisper: `LOAD_LOCK` + double-check (две вкладки грузили две копии turbo).
- `mm.unload_all_models()` из HTTP-потока: `comfy_prompt_is_running()` +
  `allow_unload_others`; кнопка Generate больше не выгружает модели идущего рендера,
  а честно говорит «дождитесь окончания» при нехватке VRAM.
- `max_new_tokens`: `setdefault` вместо перезаписи — пресеты Ideogram просят 1024,
  движок резал до 512 → обрезанный JSON молча уходил дальше как caption.
- Seed Ideogram теперь ДОХОДИТ до сэмплера (`seed` в теле `/enhance` +
  `active_seed`): повторный Generate давал байт-в-байт тот же результат.
- `hideWidget`: `node.removeInput()` вместо `splice` (линки правее скрытого входа
  съезжали на соседний слот).
- Animation Preview: `loadedGraphNode` переприменяет payload (нода теряла клип
  после перезагрузки страницы).
- Потолки параллелизма: `_INFERENCE_GATE` (LaMa), `_TRANSCRIBE_GATE` (Whisper),
  `_DECODE_GATE(2)` (ffmpeg) — общий default executor больше не исчерпать.
- Мелочи: `.part` под per-path локом (resume сохранён — имя не менялось);
  атомарная запись preview-WAV и Ideogram-пресетов; `_slug` + hash (коллизия
  имён файлов пресетов); «Заменить список» больше не стирает ручные строки и
  комментарии; `--ts-radius-md` (несуществующий токен); прогресс чужой операции
  не течёт в простаивающий SuperPrompt; RTX Upscaler alpha `.to(device)`;
  log-injection через `_log_safe`; `requests` объявлен, `qwen-vl-utils` убран.

## Ложная находка — НЕ чинить

«Токен уезжает по кросс-хост редиректу» — **опровергнуто**: requests 2.32.3
`Session.rebuild_auth` → `should_strip_auth` снимает `Authorization` при смене
хоста (CVE-2018-18074, с 2.20). Проверено чтением исходника установленной версии.

## Сознательно НЕ трогали

- `refine_sam_mask_tensor` берёт первый слой при B>1 — задокументировано в
  docstring как намеренное (совпадает с превью); менять форму выхода = ломать
  workflow.
- `str(exc)` в ответах роутов (раскрывает пути ФС) — ценность для отладки выше
  риска на localhost-first сервисе.
- «Фантомные socket-строки» у старых сейвов (повторный `hideWidget` в
  `loadedGraphNode`) — TENTATIVE, спекулятивно, §18 запрещает.
- `tests/`, `tools/`, `project_memory/` вне git — политика пользователя, не баг.
  Остаточный риск: регрессионный корпус живёт в одной копии.

## Verification (под ComfyUI Python)

- `python -m compileall .` — ok
- `python -m pytest tests` — **738 offline passed** / **835 passed** с запущенным
  ComfyUI (live+browser дают +97; прежние числа 780/801 в ARCHITECTURE устарели)
- `tools/build_node_contracts.py --check` — **61 нода, drift нет** (контракты нод
  не тронуты → старые workflow целы)
- live+browser отдельно: 97 passed (`test_all_nodes_roundtrip` гоняет все 61 ноду
  edit→save→reload→queue)
- P1-фикс проверен функционально в браузере на реально отдаваемом JS: 13/13
  (drive letter, UNC, `%VAR%`, `$VAR`, `~`, `..` — отбрасываются; `vae/qwen`,
  `loras/flux\v2` — сохраняются)

⚠️ Бэкенд-фиксы требуют РЕСТАРТА ComfyUI: Python-модули импортируются на старте,
`git pull` в рантайм их не подхватывает. JS подхватывается перезагрузкой страницы.

Новые гарды: `tests/test_audit_v994_regressions.py` (27), `tests/test_style_library_data.py`
(5), `tests/test_smart_inpaint.py` наконец собирается pytest (был скрипт без единой
`def test_*`, при этом на импорте правил глобальные `comfy` args).
