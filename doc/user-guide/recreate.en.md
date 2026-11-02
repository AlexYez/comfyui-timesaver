# Recreate node — back to defaults, wires kept

Frontend command; no separate node. Source: [ts-recreate-node.js](../../js/utils/ts-recreate-node.js). Measurements below are historical examples from the former README, not guarantees for every graph.

Right-click any node → **Recreate node**. The node comes back as if freshly placed from the menu: every value at its default, and **the title, color, size and properties too**. What stays is its place on the canvas, its number (nothing in the graph that points at it moves), its mode (bypass or mute is graph state, not a node setting) and **every wire, including the dots on them**. It mends a node saved by an older version of the pack or of ComfyUI, and it simply resets one you have fiddled with. With several nodes selected and the click on one of them, the whole selection is recreated. The command is in ComfyUI's palette too, and **Ctrl+Z** brings the old node back.

A wire looks for the socket with the same name on the new node, and — if the input was renamed in a newer version — for the socket in the same place with the same type. Inputs that appear as you connect them (Qwen Image 2.1's images: `image_1`, `image_2` …) are reconnected in order. **If even one wire finds no socket, the node is left exactly as it was**, with its values and all its wires, and the message names what did not match: no wire is ever lost. Subgraphs and nodes this install does not have get no menu entry.

---

[Навигация / index](README.md).
