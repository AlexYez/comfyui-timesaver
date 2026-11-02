---
name: reference_release_12_13_4
description: "История: Patch 12.13.4 pushed, Registry action succeeded, runtime fast-forward updated"
metadata:
  type: reference
  status: historical
---

История: Patch 12.13.4 pushed, Registry action succeeded, runtime fast-forward updated

**Историческая запись, не описание текущей реализации и не действующее разрешение.** Даты и результаты ниже относятся к указанным сессиям. Пути `tests/`, `tools/`, `dist/` и `archive/` с артефактами — локальная лаборатория; они не гарантированы в чистом клоне.

01.11.2026 владелец возобновил остановленный релиз словом «путч и пуш».
Опубликована ранее подготовленная 12.13.4 (не 12.13.5, так как .4 до этого
не коммитилась/не пушилась). Коммит 212dcb879ba3f0cc548ee30b6172c727aa9a5955:
release: 12.13.4 — пресеты LLM, EOS Qwen и Check Links.
Push origin/master успешен. Publish Custom Node action 41246403318 завершился
success, включая сам шаг публикации, а не только проверку версии.
https://github.com/AlexYez/comfyui-timesaver/actions/runs/41246403318

В коммите 21 runtime/documentation файл: Check Links frontend/backend,
обновление 14 из 15 исходных пресетов, алиасы экспериментальных имён только
для старых графов, RT finish_answer/JSON/форматирование перевода, EOS Qwen,
два недостающих перевода H3 Fast VAE, корректные README counts (88 нод).
Erotism целиком сохранён равным прежнему HEAD. LFM3 никогда не публиковалась
и отсутствует в релизе. Её веса остаются в корзине по просьбе владельца.

Проверки уже прошли: 659 целевых backend тестов и 12 Check Links browser тестов
до удаления VL, затем 42 целевых после её удаления. Повторять весь набор не
стали. Перед commit ещё раз прошли compileall nodes, Ruff изменённых Python,
ES module syntax обоих downloader JS, git diff --check. Полный pytest не запускали.
Manifest в tests/.cache/release-12.13.4-final/manifest.json: проверены 21 файла,
версия origin до push 12.13.3, имена/порядок 15 пресетов, исходный Erotism.

Установленный пак D: обновлён git pull --ff-only до того же коммита, clean.
20 предварительно синхронизированных файлов сохранены в runtime-before-pull.zip
в том же cache, временно stashed, после pull stash снят после проверки.
Единственное различие dev/runtime перед pull — 2 изменения порядка imports
Ruff в _downloader_links.py; проверены до обновления. Все 21 файла сверены
по normalized SHA256 после pull. DEV clean по tracked файлам; оставлены
посторонние untracked AUDIT_REPORT_2026-10-30.md и workflow_exports/.
ComfyUI не перезапускался. tools/comfy_server.py — локальный ignored инструмент,
его исправление --enable-manager остаётся локальным и не включается в Registry.
