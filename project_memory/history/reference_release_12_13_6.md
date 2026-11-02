---
name: reference_release_12_13_6
description: "История: Патч 12.13.6 — DLSS v21, подсказки движков, совместимость NVIDIA VFX; опубликован и установлен"
metadata:
  type: reference
  status: historical
---

История: Патч 12.13.6 — DLSS v21, подсказки движков, совместимость NVIDIA VFX; опубликован и установлен

**Историческая запись, не описание текущей реализации и не действующее разрешение.** Даты и результаты ниже относятся к указанным сессиям. Пути `tests/`, `tools/`, `dist/` и `archive/` с артефактами — локальная лаборатория; они не гарантированы в чистом клоне.

02.11.2026 по прямой команде «патч и пуш» опубликована 12.13.6.
Коммит 8dc3ea1ef976d1ec1ef79c728c9f4e7817872094 в origin/master.
Включены 13 файлов: внешний DLSS v21 backend и optional-входы в конце схемы,
подсказки EN/RU по доступности параметров, README/help, снимок контрактов,
совместимость optional nvidia-vfx с API 0.1/0.2, changelog и badges версии.
Локальные тесты, отчёт HTML, установочные файлы/модели, AUDIT_REPORT и
workflow_exports в коммит не входят. Новые тесты остаются ignored по правилам проекта.

155 связанных тестов passed; snapshot OK (88 нод), Ruff, compileall DLSS,
git diff --check passed. Полный suite не запускался. Ранее реальные GPU-прогоны
статичного кадра и 3-кадрового RGBA батча проверены, см. reference_dlss_upscaler.

Registry action https://github.com/AlexYez/comfyui-timesaver/actions/runs/41460113603
завершился success, сам шаг Publish Custom Node тоже success.

Runtime D: обновлён git pull --ff-only до того же SHA, clean. Предварительно
синхронизированные файлы сохранены в tests/.cache/release-12.13.6-final/runtime-before-pull.zip,
временно stashed, после проверки 13 файлов и чистоты клона stash снят.
Manifest с normalized SHA256: та же папка /manifest.json. Dev clean по tracked
файлам; сохранены посторонние untracked AUDIT_REPORT_2026-10-30.md и workflow_exports/.
ComfyUI не перезапускался: новые Python-модули и EN tooltips подхватятся после
ручного рестарта и обновления страницы.
