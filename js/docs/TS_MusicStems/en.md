# TS Music Stems

Splits music into stems, with the engine chosen by `model_name`.

**BS-RoFormer SW** is the default and gives six: `vocal`, `bass`, `drums`,
`guitar`, `piano`, and `others` for everything left over. **Mel-Band RoFormer**
gives only vocals and instrumental — and is the better choice when that is all
you want, because a specialist spends its whole capacity on the one boundary
that matters instead of splitting it six ways. Both are transformers and both
are a clear step up from what came before. **Demucs** (`htdemucs`,
`htdemucs_ft`, `hdemucs_mmi`) is still selectable so that workflows saved
before this change keep producing exactly what they always produced.

**The stems add back up to the mix.** Mask separation does not do that on its
own — the error is easy to hear in a null test. So one stem is not taken from
the model at all: it is the mix minus everything else, which makes the set
exact by construction. Measured on real music: `vocal + instrumental` nulls
against the source at 161 dB, and the six stems sum to it at 144 dB, which is
the floating-point floor rather than a modelling error.

**Outputs a model cannot produce are blocked, not silenced.** Ask Mel-Band for
drums and that branch of the graph simply does not run. A silent stem would
look like a broken model and cost you an afternoon.

`precision` picks fp16 or fp32 for the RoFormer engines. fp16 runs about twice as
fast on half the VRAM, and its error against fp32 was measured at -61 dBFS or
below on real music — under the noise floor of the recording. bfloat16 is not
offered: these models build their mask through `view_as_complex`, which does not
accept it. `shifts` and `jobs` apply to Demucs only.

The weights download once on first use, into `models/roformer/`. If you already
have the Mel-Band checkpoint from another pack it is found where it lies rather
than fetched again.

**Use when:** isolating vocals for remixing, extracting karaoke instrumentals,
or feeding cleaner stems into another model.

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
