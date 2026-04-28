# docs/AGENTS.md — Documentation Rules

Эта папка содержит пользовательскую и developer-документацию.

Codex всегда отвечает пользователю на русском языке. Документация может быть на русском или английском в зависимости от стиля проекта, но если пользователь пишет по-русски — новые пояснения и отчёты писать по-русски.

Следуй root `AGENTS.md` сначала.

---

## 1. Documentation Mission

Documentation must be accurate, practical, concise, specific to this project, compatible with current ComfyUI behavior, clear about limitations, and clear about verification steps.

Do not write marketing fluff when technical docs are needed.

---

## 2. Documentation as Project Memory

Use documentation to preserve engineering lessons:

- `docs/ai-lessons.md`
- `docs/troubleshooting.md`
- `docs/migration.md`
- `docs/developer-notes.md`
- nearest relevant `AGENTS.md` for permanent rules

If a mistake can happen twice, add a guardrail: doc, test, contract snapshot or AGENTS rule.

---

## 3. When To Update Docs

Update docs when changing:

- Node behavior, inputs, outputs, defaults.
- Installation or optional dependencies.
- Model locations and configs.
- Frontend behavior.
- Migration paths.
- Known limitations and troubleshooting.
- Test/verification workflow.
- E2E/Playwright instructions.
- One-node-one-file structure.

Do not change docs for tiny internal refactors unless public behavior changes.

---

## 4. Required Docs for New Nodes

For every new public node, document:

- Display name.
- Node ID.
- Main Python file path.
- Optional JS file path.
- Category.
- Purpose.
- Inputs, outputs and defaults.
- Expected tensor format.
- Batch behavior.
- Optional dependencies.
- Known limitations.
- Minimal workflow example.
- Troubleshooting notes.
- Verification commands.

Keep docs consistent with actual schema.

---

## 5. Migration Notes

When changing compatibility-sensitive behavior, add migration notes:

- What changed.
- Why it changed.
- Whether old workflows still load.
- Whether `io.NodeReplace` or alias mappings are provided.
- What user must do manually.
- Version/date.
- Verification performed.
- Contract tests updated.

Never hide breaking changes.

---

## 6. Verification Docs

Good verification docs include:

```bash
python -m compileall .
python -m pytest tests
python -m ruff check .
npm run lint
npm run test
npm run build
npm run test:e2e
```

Also document manual ComfyUI checks:

- where to place files
- how to start ComfyUI
- which node category to check
- which workflow to open
- expected output shape/UI behavior
- how to check browser console
- how to run Playwright tests

---

## 7. Troubleshooting Style

Use:

```md
### Problem

### Cause

### Fix

### Verification

### Notes
```

Avoid vague fixes like "reinstall everything" unless necessary.

---

## 8. Style and Definition of Done

Use clear headings, short paragraphs, exact paths, copy-pasteable commands, platform-aware notes, and plain language.

Avoid unverified claims, outdated version claims, broken links, emojis, ANSI/color references, secrets and private paths.

Documentation task is complete only when:

- Ответ пользователю на русском.
- Docs match current code.
- Migration risks are explicit when relevant.
- Commands are copy-pasteable.
- Optional dependencies are marked.
- Verification steps are included.
- Browser/E2E steps are included for frontend changes.
- One-node-one-file expectation is documented when relevant.
