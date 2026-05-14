---
name: Merge to master after each stage
description: После каждого мигрированного батча/этапа сразу делать merge в master, чтобы ComfyUI (запущен из main repo) видел обновления
type: feedback
originSessionId: 166438cb-713b-4ab0-b416-6d7d7ebdb2ff
---
После каждого завершённого этапа миграции (или другого нетривиального батча в worktree) **обязательно делать merge в master** в main repo.

**Why:** ComfyUI запускается из main repo `D:/AiApps/ComfyUI/comfyui/ComfyUI/custom_nodes/comfyui-timesaver/`, а я работаю в git worktree. Worktree файлы main repo НЕ видит. Без merge в master браузерные smoke-тесты бесполезны (ComfyUI обслуживает старую версию).

**How to apply:**
1. В worktree: `git add` + `git commit` (один или несколько осмысленных commits на этап).
2. `git merge master --no-edit` в worktree (втянуть свежие изменения master, на случай параллельной активности).
3. `git -C "<main-repo-path>" merge --ff-only claude/<branch>` — fast-forward в master.
4. Сказать пользователю: «Сделал merge, рестарт ComfyUI» (Python-изменения нужен restart; чисто JS — достаточно reload страницы).
5. После рестарта пользователя — браузерный smoke-test в Chrome MCP.
