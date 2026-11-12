# TS H3 Regenerate Model

Loads Standard Regenerate from one self-contained checkpoint.
The node has one `model` input and one **Regenerate** selector (`lora_name`).

1. Connect a clean pruned **H3 FL2VA** directly from its loader, or a clean pruned **H3 Ref2VA** for Full Reference. Keep generation LoRAs on a separate branch.
2. Select `open_h3_regenerate_2k_lora_complete.safetensors` from `models/loras` in **Regenerate**.
3. Connect the output `model` to **TS H3 Regenerate**, where denoise, steps and dimensions are configured.

When updating an older graph, select the complete file again. The public loader
no longer accepts the previous separate-file configuration.

The complete file already includes the merged LoRA, all 50 Sharp curve branches
and the verified AdaLN reference table. No separate Sharp file or reference-table
checkpoint is required. LoRA factors load as bf16; the table and Sharp correction
math remain FP32, matching the previous default path.

Adapter weights and strengths are unchanged. The bundle retains the released
**LightX2V FL2VA Turbo v1.2, 4-step, 768p**
(`minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16.safetensors`).
Do not stack another Turbo, Pika or DMAD on this refinement branch.

This node requires the complete TS bundle. A generic LoRA Loader does not apply
the Sharp curve branches. The FP32 checkpoint is approximately 13.69 GB.

## Build the complete file

The pack includes a standalone builder; weights are not bundled with the node pack.
Use the Python interpreter that runs ComfyUI. From the ComfyUI directory:

```sh
python custom_nodes/comfyui-timesaver/nodes/video/latent_upscale/_regenerate_bundle.py --lora models/loras/open_h3_regenerate_2k_lora.safetensors --sharp models/loras/sharp_curve_branches.safetensors --reference models/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors --output models/loras/open_h3_regenerate_2k_lora_complete.safetensors --manifest models/loras/regenerate-complete-manifest.json
```

Adjust paths to your folders. The first two inputs come from the
[original Standard release](https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K/tree/main/lora).
The reference input is your existing pruned Ref2VA checkpoint: only its 32 KB AdaLN
table is copied, without loading the model. The builder copies every tensor unchanged,
verifies hashes and keeps the original files. It needs space for the complete output;
packaging does not reduce the combined size.
