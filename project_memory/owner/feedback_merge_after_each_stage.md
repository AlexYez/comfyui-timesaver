---
name: feedback_merge_after_each_stage
description: "После каждого мигрированного батча/этапа сразу делать merge в master, чтобы ComfyUI (запущен из main repo) видел обновления"
metadata:
  type: feedback
---

После каждого мигрированного батча/этапа сразу делать merge в master, чтобы ComfyUI (запущен из main repo) видел обновления

## Применимость

Это исторический порядок миграции из worktree, а не разрешение автоматически сливать или публиковать текущую работу. С 2026-08-08 коммит и пуш выполняются только по команде владельца: [действующее правило](feedback_commit_then_sync.md). Текущий рабочий процесс описан в [разработке](../../doc/operations/development.md). Причина старой записи остаётся полезной: проверка должна выполняться на той копии, чьи файлы действительно загружены сервером.

## Историческая запись

После каждого завершённого этапа миграции (или другого нетривиального батча в worktree) **обязательно делать merge в master** в main repo.

**Why:** ComfyUI запускается из main repo `D:/AiApps/ComfyUI/comfyui/ComfyUI/custom_nodes/comfyui-timesaver/`, а я работаю в git worktree. Worktree файлы main repo НЕ видит. Без merge в master браузерные smoke-тесты бесполезны (ComfyUI обслуживает старую версию).

**How to apply:**
1. В worktree: `git add` + `git commit` (один или несколько осмысленных commits на этап).
2. `git merge master --no-edit` в worktree (втянуть свежие изменения master, на случай параллельной активности).
3. `git -C "<main-repo-path>" merge --ff-only claude/<branch>` — fast-forward в master.
4. Сказать пользователю: «Сделал merge, рестарт ComfyUI» (Python-изменения нужен restart; чисто JS — достаточно reload страницы).
5. После рестарта пользователя — браузерный smoke-test в Chrome MCP.
