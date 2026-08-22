# TS Prompt Builder

Builds a prompt out of **wildcard packs**. A pack is just a folder in
`nodes/prompts/` holding `.txt` wildcards plus a **semantic map**, and the map is the
whole point: without one, picking a random line from twenty lists gives you a winter
street in a swimsuit, a close-up with a full-body pose, and two incompatible scenes at
once.

The map says what each wildcard *is* — its role, where it belongs in the phrase, what it
excludes, what it goes well with — and the node assembles by that instead of by luck.
Eight steps, taken from the packs' own `algorithm` section: profile or your own toggles,
then the people-in-frame policy, mutual exclusions, incompatible pairs, optional
companions, an affinity pass, one line per surviving wildcard, and finally the phrase in
role order.

**Any packs combine, in any combination.** Turn several on and the roles interleave into
one sentence — all the identity first, then clothing, then the act, then place and camera
— rather than one pack's output glued onto another's. Wildcards are namespaced by pack, so
two `face.txt` never collide, and where two packs both offer a face, a light or a pose,
exactly one survives: a draw weighted by each pack's `mix.priority`, so a mix of five is a
genuine blend and not the loudest pack talking over the rest.

**The scene holds together.** Place lives in the text of the lines, not in the links
between files, so semantics alone could not stop a prompt from putting a pool, a rainstorm
and a kitchen in one sentence — measured at 22% of assemblies. The node now picks the
place first and then reads every other line against it, dropping the ones that argue about
where or when we are. Same measurement afterwards: 3%, and what is left are metaphors
rather than mistakes.

Drop a folder in by hand and press **Reload** — no ComfyUI restart. The node shows the
wildcards grouped by role, dims the ones that will collapse to a single pick at run time,
lets you pin one so it survives a collision, and previews the assembled prompt live using
the very same code the run will use. A second output reports what the semantic map threw
out and why.

`seed = 0` gives a new prompt every run; anything above 0 is reproducible.

**Use when:** running batches with controlled variation — every wildcard is a category,
every line a flavour, and the semantic map keeps the combination coherent.

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
