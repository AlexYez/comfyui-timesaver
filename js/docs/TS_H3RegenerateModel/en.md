# TS H3 Regenerate Model

Connect a clean pruned H3 FL2VA loader, before all LoRAs. Clones model settings and installs
312 merged LoRA projection patches plus 50 Sharp branches, leaving the draft branch unchanged.

Select `open_h3_regenerate_2k_lora.safetensors` and `sharp_curve_branches.safetensors` from
`models/loras` or a registered external model folder. The reference checkpoint is an installed
pruned Ref2VA file: only its 32 KB AdaLN table is read and verified, without loading Ref2VA onto the GPU.
Current support requires a pruned H3 model with a 1025×8 AdaLN table.

LoRA factor storage uses about 6.2 GB CPU RAM in bf16, or 12.45 GB in fp32.
The loader maps fused Q/K/V slices and swaps the Diffusers SwiGLU halves for native ComfyUI.
Do not stack Pika, DMAD or another Turbo on this refinement branch.
