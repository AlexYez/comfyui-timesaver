---
name: history_index
description: "Завершённые работы, аудиты, снятые подсистемы и старые состояния. Эти записи не дают разрешения возобновлять старую работу или менять текущую реализацию."
metadata:
  type: reference
---

Завершённые работы, аудиты, снятые подсистемы и старые состояния. Эти записи не дают разрешения возобновлять старую работу или менять текущую реализацию.

# История проекта

[Общий индекс](../MEMORY.md)

- [new-session-start](new-session-start.md) — История: Готовый стартовый промпт для новой сессии из dev-папки C: — продолжение работы над нодой TS Smart Inpaint
- [project_image_studio](project_image_studio.md) — История: TS Image Studio — утверждённый план, операционные разрешения владельца (перезапуск ComfyUI, ветка, Artius), статус реализации
- [project_studio_hub](project_studio_hub.md) — История: TS Image Studio — точка входа во всю память и документацию подсистемы: что где лежит, с чем связано, что читать перед какой правкой
- [reference_audit_2026_08_12](reference_audit_2026_08_12.md) — История: Аудит всего пака 12.08.2026 и внесённые исправления — что чинили, что отклонили и почему
- [reference_audit_2026_09_29](reference_audit_2026_09_29.md) — История: Аудит всего пака 29.09.2026 — что нашли (5 параллельных проверяющих + живой сервер), что исправили, что проверено живьём, что ждёт видеокарты
- [reference_audit_v978](reference_audit_v978.md) — История: Whole-codebase quality audit → v9.78 fixes; torch-flag scoping decision; fingerprint-mean fix intentionally SKIPPED (caching semantics = workflow breakage)
- [reference_audit_v994](reference_audit_v994.md) — История: Read-only аудит всего пака 2026-07-31 → фиксы в v9.94; что исправлено, что сознательно НЕ трогали, и почему находка про redirect-утечку токена оказалась ложной
- [reference_lfm3](reference_lfm3.md) — История: LFM3 VL and Omni retired at owner request; VL weights recycled on 2026-11-01
- [reference_lfm3_vl_peer_comparison](reference_lfm3_vl_peer_comparison.md) — История: Fair BF16 LFM3 VL and Qwen 2B comparison with existing Gemma E2B LiteRT, 36 text responses
- [reference_pack_review_2026_07](reference_pack_review_2026_07.md) — История: Whole-pack reviews 2026-07-03 (Ready) и 2026-07-15 (Needs Attention → всё исправлено в v9.77); что где живёт и что не перепроверять
- [reference_release_12_13_4](reference_release_12_13_4.md) — История: Patch 12.13.4 pushed, Registry action succeeded, runtime fast-forward updated
- [reference_release_12_13_6](reference_release_12_13_6.md) — История: Патч 12.13.6 — DLSS v21, подсказки движков, совместимость NVIDIA VFX; опубликован и установлен
- [reference_removed_nodes_2026_08](reference_removed_nodes_2026_08.md) — История: 11.08.2026 с пака сняты 16 нод (v10.1.0 → v11.0.0) — что именно, где архив, что пришлось поправить кроме кода
- [reference_studio_backend_delivery](reference_studio_backend_delivery.md) — История: Откуда студия берёт графы и как установленный набор молча перекрывает встроенный — ловушка, стоившая часа измерений
- [reference_studio_iterations_and_batch](reference_studio_iterations_and_batch.md) — История: Итерации студии (черновик → Save) и пачка (фазы по всем заданиям) — три слоя, пять граблей, где что лежит
- [reference_studio_persistence-part-1](reference_studio_persistence-part-1.md) — Исторические наблюдения, часть 1; не действующая реализация.
- [reference_studio_persistence-part-2](reference_studio_persistence-part-2.md) — Исторические наблюдения, часть 2; не действующая реализация.
- [reference_studio_persistence](reference_studio_persistence.md) — Историческая тема разделена на небольшие части; исходные наблюдения и причины сохранены.
- [reference_studio_stage_ux](reference_studio_stage_ux.md) — История: Рабочая область студии — заглушка по центру, превью во всю площадь, анимация ожидания до первого кадра, меню по правому клику; в расширении работу показывает рамка, а н…
- [reference_studio_subscription](reference_studio_subscription.md) — История: Подписка TS Image Studio — пропуск, наборы, два репозитория, где что лежит и что можно сломать
- [studio-hosts-node-editors](studio-hosts-node-editors.md) — История: TS Image Studio hosts other nodes' editors instead of reimplementing them — the pattern, and why
- [studio-model-behaviour](studio-model-behaviour.md) — История: Measured behaviour of each image family in TS Image Studio — which model suits which repaint job, and what each denoise range actually does
- [ts-smart-inpaint-handoff](ts-smart-inpaint-handoff.md) — История: Worklog/handoff по ноде TS Smart Inpaint — что сделано (A0/A1 perf) и что дальше (feather/blur, large-crop). Контекст для продолжения работы в новой сессии.
