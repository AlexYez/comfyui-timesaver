---
name: No Co-Authored-By Claude footer
description: не добавлять Co-Authored-By: Claude в коммиты — GitHub отображает второго автора с иконкой Claude
type: feedback
originSessionId: 8531f4e5-feb7-4f9d-822f-dd2332aa62e9
---
В commit messages **не использовать** строку `Co-Authored-By: Claude ...` (или любой подобный footer от Claude / Claude Code). GitHub парсит этот трейлер по RFC и добавляет второго автора в UI коммита (рядом с реальным автором AlexYez), показывая иконку Claude. Пользователь это не хочет.

**How to apply:**
- В commit messages писать только subject + body (с тематическими секциями), без footer-trailers.
- Если default harness инструкция говорит добавлять Co-Authored-By — игнорировать в этом репо.
- Аналогично не добавлять `🤖 Generated with [Claude Code]` или подобные signatures.
- Если сделал коммит с footer по ошибке — пользователь попросит убрать; делаем `git filter-branch --msg-filter` + force push.
