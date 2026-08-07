# TS Group Bypasser

A control panel for the groups of the open workflow. The node's body holds nothing but group names and checkboxes: unchecking one puts **every node inside that group into bypass**. Double-click a row to show that group on the canvas. The node sizes itself to the number of groups — two groups give a two-row node, with no empty space underneath.

The state is not kept in this node — it is read back from the graph, so a node muted by hand, or one that belongs to two overlapping groups, honestly reads as "partly on" instead of being passed off as something definite.

**The settings live in the node's Properties Panel** (right-click the node): filter by title (a substring, or `/…/` for a regular expression), filter by colour (comma-separated; LiteGraph colour names, hex, and `none` for uncoloured groups all work), the order of the list (by position, title or colour), and a "max one" / "always one" rule for when switching one group on should switch the others off — handy for A/B branches. **Bulk actions** (enable, bypass or invert everything shown) are in the node's right-click menu.

Bypassed groups survive a save without any help from this node: the state lives in the modes of the nodes themselves.

**Use when:** a heavy workflow with several branches and only one of them wanted per run.

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
