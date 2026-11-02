---
name: feedback_no_coauthor_footer
description: "не добавлять Co-Authored-By: Claude в коммиты — GitHub отображает второго автора с иконкой Claude"
metadata:
  type: feedback
---

не добавлять Co-Authored-By: Claude в коммиты — GitHub отображает второго автора с иконкой Claude

В commit messages **не использовать** строку `Co-Authored-By: Claude ...` (или любой подобный footer от Claude / Claude Code). GitHub парсит этот трейлер по RFC и добавляет второго автора в UI коммита (рядом с реальным автором AlexYez), показывая иконку Claude. Пользователь это не хочет.

**How to apply:**
- В commit messages писать только subject + body (с тематическими секциями), без footer-trailers.
- Если default harness инструкция говорит добавлять Co-Authored-By — игнорировать в этом репо.
- Аналогично не добавлять `🤖 Generated with [Claude Code]` или подобные signatures.
- Если сделал коммит с footer по ошибке — пользователь попросит убрать; делаем `git filter-branch --msg-filter` + force push.

## Что уже пришлось сделать (2026-06-19)

Пользователь попросил убрать Claude из контрибьюторов GitHub. История master
переписана (`git filter-branch --msg-filter`, вырезаны все строки с
`noreply@anthropic.com` — 27 коммитов), сделан force-push (`e7b8921` →
`1638568`); контент файлов не менялся. Бэкап исходной истории — локальная ветка
`backup-before-claude-strip` (плюс `refs/original/`).

GitHub считает контрибьюторов по авторам коммитов **и** по трейлерам
`Co-Authored-By`, поэтому одна такая строка вернёт Claude в граф. Трейлеры
`Co-Authored-By: AlexYez` оставлять можно.
