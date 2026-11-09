# TS H3 Regenerate Model

Connect a clean pruned **H3 FL2VA** for standard refinement or a clean pruned **H3 Ref2VA**
for Full Reference. Connect the loader directly, before LoRAs or block patches. The node clones
the model and installs 312 merged LoRA projection patches plus 50 Sharp branches, leaving
the draft branch unchanged.

Select `open_h3_regenerate_2k_lora.safetensors` and `sharp_curve_branches.safetensors` from
`models/loras` or a registered external model folder. For `reference_checkpoint` select the
installed `minimax_h3_ref2va_pruned_int8_convrot.safetensors`. This reads and verifies only its
32 KB AdaLN table. For Full Reference connect the actual Ref2VA model separately to `model`:
its table must match the reference. Supported pruned H3 models have 50 blocks and a 1025×8
AdaLN table.

LoRA factor storage uses about 6.2 GB CPU RAM in bf16, or 12.45 GB in fp32.
The loader maps fused Q/K/V slices and swaps the Diffusers SwiGLU halves for native ComfyUI.
Do not stack Pika, DMAD or another Turbo on this refinement branch.

[Connect TS H3 Regenerate](https://github.com/AlexYez/comfyui-timesaver/blob/master/js/docs/TS_H3Regenerate/en.md)
