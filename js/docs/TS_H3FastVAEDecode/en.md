# TS H3 Fast VAE Decode

Supports original FP32 and native INT8 convrot checkpoints, detected from per-layer `comfy_quant` metadata. Only 96 transformer block linears are quantized; norms, convolution and input/output projections remain FP32. The INT8 file is 1.64 GB versus 6.47 GB for FP32. Clip activation memory still grows with resolution and duration.

Decodes H3 video through NVIDIA StreamRender-H3's noncausal 24-layer decoder.
Select `fast_h3_vae_noncausal_24l.safetensors` from `models/vae` or a registered external VAE folder.
Accepts video or AV latents; audio is ignored.

Select `auto` to try the fast path first. If it does not fit, the node uses the standard H3
**video VAE** connected to `fallback_vae`, or tries `low VRAM` when no VAE is connected.
The `fast` and `low VRAM` options select the first decoder explicitly.
Low VRAM keeps Q/K/V for up to two native blocks at a time. It uses the original attention
windows and global context and preserves frame count. The report shows the path that ran.
Memory is estimated before loading the weights. CUDA out-of-memory errors can trigger a fallback.

Output dimensions and frame count come from the latent. Crop 1088/2176 and trim any padding
after decoding, along with the soundtrack. The NVIDIA decoder can show brightness or color
changes at 17-frame block boundaries. Compare it with the standard H3 video VAE on the same
latent. This runs the PyTorch SDPA path, without SoL-Engine optimizations.
