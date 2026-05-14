---
name: Не трогать publish_action.yml
description: .github/workflows/publish_action.yml с Comfy-Org/publish-node-action@main — это design choice, не security defect; не предлагать pin к @v1
type: feedback
originSessionId: 735fb877-19df-48cd-bb11-714ae6800a94
---
Не предлагать менять `.github/workflows/publish_action.yml`, в частности — не пинить `Comfy-Org/publish-node-action@main` к тегу/SHA. Этот action и его триггер (`push: paths: pyproject.toml`) — устоявшаяся конфигурация публикации в Comfy Registry, она работает как задумано Comfy-Org.

**Why:** общие правила supply-chain (pin к commit SHA для GitHub Actions) здесь не применимы — Comfy-Org намеренно публикует action на ветке `main`, чтобы пользователи получали актуальную интеграцию с Registry API. Пользователь знает специфику этого action-а лучше, чем я мог вывести из общих best-practices.

**How to apply:** в любых аудитах / планах модернизации не помечать `publish_action.yml` как ⚠ из-за `@main`. Не предлагать "pin to @v1 for security". Если кажется, что есть проблема — спросить пользователя, а не предлагать правку.
