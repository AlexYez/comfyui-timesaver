---
name: studio-hosts-node-editors
description: "История: TS Image Studio hosts other nodes' editors instead of reimplementing them — the pattern, and why"
metadata:
  type: reference
  status: historical
---

История: TS Image Studio hosts other nodes' editors instead of reimplementing them — the pattern, and why

TS Image Studio снята из текущего пака. Код до удаления доступен в [истории Git](https://github.com/AlexYez/comfyui-timesaver/tree/c1668b3cfa2161e36bf9b9fa91288b949b4b0b1f/nodes/image/studio); удаление зафиксировано [коммитом a2251b1](https://github.com/AlexYez/comfyui-timesaver/commit/a2251b1). Записанное решение владельца датировано 2026-09-27, коммит — 2026-09-28: это разные события.

**Историческая запись, не описание текущей реализации и не действующее разрешение.** Даты и результаты ниже относятся к указанным сессиям. Пути `tests/`, `tools/`, `dist/` и `archive/` с артефактами — локальная лаборатория; они не гарантированы в чистом клоне.

**Owner's rule (2026-08-02):** when a node already has an interface, the studio
must open THAT interface, not grow a parallel one. The node keeps being
updated; the app has to inherit those updates for free and use its features
100%.

**The pattern** (first applied to Ideogram):

- **Graph side** — the node itself sits in the backend workflow
  (`TS_IdeogramDesigner` in `js/image/studio/workflows/ideogram/*.json`). It
  produces the prompt and the frame; the studio only feeds it parameters.
- **UI side** — `js/_studio/_editors.js` is a provider registry. A provider
  imports the node's PUBLIC entry points only (`openIdeogramEditor`,
  `loadPresets`) — never its LiteGraph wiring (`_ideogram_node.js`). The editor
  mounts its own fullscreen overlay; the provider lifts it above the studio
  with `markOverlayAbove` so it paints on top and owns Escape.
- **Manifest side** — a `designer` block names the provider, the mode/caption
  inputs and the fallback preset; a `prompt_source` block points the metadata
  injector at the node's own output, so the saved PNG carries exactly the
  prompt the model was given.
- **Guard** — `tests/test_studio_designer_contract.py` fails if the node
  renames an input, drops an export, or the manifest stops matching the graph.
  Node updates then break a test, not a user's render.

Adding another hosted editor = one provider + one manifest block. Do NOT copy
an editor's logic into the studio.
