# TS NAG

**A negative prompt where there cannot be one** — on a distilled model sampled at `cfg = 1`.

At `cfg = 1` ComfyUI does not compute the negative branch at all: the wire is there, the meaning is not. NAG (*Normalized Attention Guidance*, [paper](https://github.com/ChenDarYen/Normalized-Attention-Guidance)) comes at it from the other side — it applies the negative **inside every cross-attention block**. With the same query, attention is computed twice, against the positive context and against a negative one, and then:

```text
guidance = x_pos · scale − x_neg · (scale − 1)      extrapolate
r        = ‖guidance‖₁ / ‖x_pos‖₁                   per token
if r > tau:  guidance ← guidance · (‖x_pos‖₁ · tau / ‖guidance‖₁)
out      = guidance · alpha + x_pos · (1 − alpha)
```

The clamp by `tau` is what the "normalized" is for: an extrapolation with a scale of 11 would otherwise tear the activations apart. `nag_scale` is the push, `nag_tau` the ceiling on the deviation, `nag_alpha` how much of it reaches the result.

**The cost is one extra cross-attention per block, not a second pass of the model.** That is the whole point: the query comes from the picture and meets a short text context, which is cheap.

⚠️ **It does not work with every model, and that is not a whim.** It needs a real, separate cross-attention module that can be called a second time:

| model | |
|---|---|
| **Wan** | `blocks[i].cross_attn`, T2V and I2V alike — supported |
| **LTX** | `transformer_blocks[i].attn2` — supported |
| **Krea 2**, **MiniMax H3** | text and picture are joined into one sequence before the stack — **refused, with the reason** |

On a single-stream model the negative variant would have to be carried through the whole stack, which is a second full forward — exactly what CFG does. The node says so and points you at `cfg` instead of pretending it saved you something.

The family is recognised **by the structure of the blocks**, not by a file name: `model_type` = `auto`. Which rows of the batch are positive is asked of the core (`cond_or_uncond`) rather than guessed from the shape — otherwise a batch of two pictures at `cfg = 1` would be indistinguishable from a positive/negative pair.

> The idea and the defaults come from the `WanVideoNAG` node in [kijai/ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes). For Wan the maths is reproduced one to one, including where the two branches meet, so settings shared for it transfer as they are. For LTX the original `forward` is called twice and the combination happens after the output projection: duplicating LTX's internals (RoPE, guide masks, per-head gating) would mean breaking on every update to them.

**When to use it:** a distilled model at `cfg = 1` and something you want gone — "no text", "not cartoonish", "no extra fingers".

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
