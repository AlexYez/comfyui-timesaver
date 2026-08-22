# TS Angle Select

Point a camera at the subject and get the prompt that asks a model for exactly
that view. The node shows a small 3D preview — the subject, the orbit around it
and the camera on that orbit — and under it three controls: **rotation**,
**height** and **zoom**. Move a control and the preview shows where the camera
went.

The preview is only that: a preview. Setting three values by dragging one canvas
turned out to be fiddly, so each value has its own slider, and nothing in the
widget changes size with its value — the node never shifts under the cursor
while a slider is being dragged.

**The wording belongs to the model, not to the node.** With the bundled
**Qwen Multi-Angle** preset the output is the trigger phrase the
Multiple-Angles LoRA was trained on — `<sks> back view elevated shot close-up`
and nothing else. It reads like a fragment because it is one: the LoRA learned
these exact words next to the upstream node that emits them, and prettier
English breaks the conditioning. The `<sks>` token has to be there.

**Presets are plain JSON**, one file per model in `nodes/text/angle_presets`.
A preset says what template to fill and which phrase belongs to each camera
position, so supporting a new model is a new file rather than a code change. A
preset missing a phrase is skipped with a line in the log — half a vocabulary
would quietly produce a prompt with a hole in it.

**Eight rotations, four heights, three framings.** Those are the buckets the
model was trained on; there is nothing in between, because a phrase for it does
not exist.

Three.js ships with the pack and loads **only when this node appears** — it is
deliberately kept out of the web folder, because ComfyUI imports every script in
there on page load and nobody should pay for a 3D library they never use.

**Use when:** re-shooting the same subject from another angle with Qwen-Image-Edit
and the Multiple-Angles LoRA, or building a turnaround by stepping through the
eight rotations.

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
