# TS Remove Background

State-of-the-art background removal via BiRefNet. Outputs the cut-out image, an alpha mask, and a "mask preview" image. Options: model picker (HR-matting / general / portrait / DIS), `process_resolution` (with `use_custom_resolution` override), `precision` (auto/fp16/fp32), `mask_blur`, `mask_offset`, `invert_output`, `temporal_smooth` for video (`none`/`median3`/`ema` with `ema_alpha`), background mode (Alpha / colour via the COLOR widget). v9.4 cleanup removed the unstable `refine_foreground` option.

**Use when:** isolating subjects, building product shots, or feeding clean alpha masks into compositing nodes.

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
