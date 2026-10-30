# TS H3 Fast VAE Decode

Supports original FP32 and native INT8 convrot checkpoints, detected from per-layer `comfy_quant` metadata. Only 96 transformer block linears are quantized; norms, convolution and input/output projections remain FP32. The INT8 file is 1.64 GB versus 6.47 GB for FP32. Clip activation memory still grows with resolution and duration.

Decodes H3 video through NVIDIA StreamRender-H3's noncausal 24-layer decoder.
Select `fast_h3_vae_noncausal_24l.safetensors` from `models/vae` or a registered external VAE folder.
Accepts video or AV latents; audio is ignored.

Connect the standard H3 **video VAE** to `fallback_vae` for clips that do not fit the fast decoder.
Memory is estimated before weights are loaded, with a second fallback on an actual out-of-memory
error. Other errors remain visible. Fast decoder GPU weights are released after decoding.

Output dimensions and frame count come from the latent. Crop 1088/2176 and trim any padding
after decoding, along with the soundtrack. Fast and standard decoding can differ slightly.
This runs the PyTorch SDPA path, without SoL-Engine optimizations.
