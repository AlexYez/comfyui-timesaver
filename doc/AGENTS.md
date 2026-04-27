# docs/AGENTS.md — Documentation Rules

This directory contains user and developer documentation.

Follow the root `AGENTS.md` first. This file adds documentation-specific rules.

---

## 1. Documentation Mission

Documentation must help users and developers use the nodes safely.

Good documentation is:

- accurate
- practical
- short enough to read
- specific to this project
- compatible with current ComfyUI behavior
- clear about limitations
- clear about migration risks

Do not write marketing fluff when technical docs are needed.

---

## 2. Documentation as Project Memory

Use documentation to compound engineering knowledge.

When the agent or user discovers a recurring issue, record it in one of these places:

- `docs/ai-lessons.md`
- `docs/troubleshooting.md`
- `docs/migration.md`
- `docs/developer-notes.md`
- nearest relevant `AGENTS.md` if it is a permanent rule

Do not overload `AGENTS.md` with long stories. Keep `AGENTS.md` as rules and move examples/details into docs.

---

## 3. When To Update Docs

Update docs when changing:

- Node behavior.
- Node inputs.
- Node outputs.
- Default values.
- Installation steps.
- Optional dependencies.
- Model locations.
- Config files.
- Frontend behavior.
- Migration paths.
- Known limitations.
- Troubleshooting steps.
- Test or verification workflow.

Do not change docs for tiny internal refactors unless public behavior changes.

---

## 4. Required Docs for New Nodes

For every new public node, document:

- Display name.
- Node ID when useful for debugging.
- Category.
- Purpose.
- Inputs.
- Outputs.
- Defaults.
- Expected tensor format.
- Batch behavior.
- Optional dependencies.
- Known limitations.
- Minimal example workflow description.
- Troubleshooting notes.

Keep docs consistent with the actual schema.

---

## 5. Migration Notes

When changing compatibility-sensitive behavior, add migration notes.

Include:

- What changed.
- Why it changed.
- Whether old workflows still load.
- Whether `io.NodeReplace` or alias mappings are provided.
- What the user must do manually, if anything.
- Version/date of the change.
- Verification performed.

Never hide breaking changes.

---

## 6. Verification Notes

Docs should include clear verification steps for developers.

Good verification docs include:

```bash
python -m compileall .
python -m pytest tests
npm run lint
npm run test
npm run build
```

Also document manual ComfyUI checks when needed:

- where to place files
- how to start ComfyUI
- which node category to check
- which workflow to open
- what output shape or UI behavior is expected

---

## 7. Troubleshooting Style

Troubleshooting entries should follow this structure:

```md
### Problem

Short description of the error.

### Cause

Why it happens.

### Fix

Concrete steps.

### Verification

How to confirm the fix worked.

### Notes

Optional details, limitations, or version-specific behavior.
```

Avoid vague fixes like "reinstall everything" unless it is truly necessary.

---

## 8. Installation Docs

Installation docs must distinguish:

- Required dependencies.
- Optional dependencies.
- GPU-specific dependencies.
- Windows-specific notes.
- Portable ComfyUI notes.
- Model download/location notes.

Never tell users that node code auto-installs dependencies unless it truly does.

Prefer explicit commands and paths.

---

## 9. Style

Use:

- Clear headings.
- Short paragraphs.
- Bullet lists where helpful.
- Code blocks for commands.
- Exact file names and paths.
- Plain language.

Avoid:

- Overly long paragraphs.
- Unverified claims.
- Outdated version claims.
- Broken links.
- Emojis in technical instructions.
- ANSI/color references.

---

## 10. Version Awareness

When docs depend on a ComfyUI version, mention it clearly.

Example:

```md
Tested with ComfyUI 0.xx.x and comfy_api vX.
```

If uncertain, say so.

Do not claim compatibility with all future versions.

---

## 11. Documentation Self-Review Checklist

Before finalizing docs, check:

- Does this match the current code?
- Are commands copy-pasteable?
- Are paths Windows-safe?
- Are optional dependencies marked optional?
- Are migration risks explicit?
- Are verification steps present?
- Is the text concise?
- Does it avoid outdated claims?
- Does it avoid secrets and private paths?

---

## 12. Documentation Definition of Done

A documentation change is complete only when:

- It matches the current code.
- It does not contradict AGENTS rules.
- It states migration risks when relevant.
- Commands are copy-pasteable.
- Paths are platform-aware.
- Optional dependencies are clearly marked.
- Verification steps are included when relevant.
- The text is concise and useful.
