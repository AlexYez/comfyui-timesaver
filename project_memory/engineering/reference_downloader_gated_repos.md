---
name: reference_downloader_gated_repos
description: "TS Files Downloader — закрытый (gated) репозиторий HF распознаётся по X-Error-Code и показывается окном со ссылками; замеренные ответы HF"
metadata:
  type: reference
---

TS Files Downloader — закрытый (gated) репозиторий HF распознаётся по X-Error-Code и показывается окном со ссылками; замеренные ответы HF

**Граница записи:** ниже сохранены исходные наблюдения и причины решений. Замеры, версии окружения и слова «сейчас» относятся к дате соответствующего наблюдения; это не повторная проверка на новом окружении. Порядок чтения и ограничения: [инженерная память](README.md#как-читать-записи).

Жалоба владельца 22.10.2026: автор закрыл репозиторий на Hugging Face
(нужно принять условия), а загрузчик отвечал «не удалось 1» и `401` в консоли.

**Ответы Hugging Face (замерено curl 22.10.2026):**
- закрытый репозиторий без токена — `401`, `X-Error-Code: GatedRepo`,
  `X-Error-Message: Access to model … is restricted…`;
- с токеном, но условия не приняты (или ждут одобрения автора) — `403 GatedRepo`
  (по документации HF; живьём без токена не проверить);
- несуществующий ИЛИ приватный без токена — `401` БЕЗ кода и
  «Invalid username or password» — HF не говорит, есть ли репозиторий;
- файла нет в открытом репозитории — `404 EntryNotFound`;
- `hf-mirror.com` на закрытый репозиторий отвечает `308` на huggingface.co,
  поэтому заголовки приходят от самого HF.

**Как устроено:** `_probe_remote_file(..., failure=dict)` кладёт последний
HTTP-ответ; `_hf_access_problem` классифицирует (gated / unavailable /
missing_file, 5xx и чужие хосты — None); `_note_access_problem` пишет понятную
строку в лог и копит в `problems`; `_RunProgress.finished(problems=...)` везёт
список в итоговом событии `ts_downloader.run_progress`. Фронтенд
`showAccessProblems` — одно окно на страницу (событие графа слышат ВСЕ ноды
загрузчика), ссылки только на `https://huggingface.co/…` (HF_PAGE), страница
файла (`/blob/`) — только для gated.

**Как проверять без сети:** `tests/test_downloader_access.py`; окно вживую —
`api.dispatchCustomEvent("ts_downloader.run_progress", {operation_id:"graph",
stage:"finished", problems:[…]})` на странице с нодой.

Связано: [reference_downloader_progress](reference_downloader_progress.md), [reference_downloader_fixes](reference_downloader_fixes.md).
