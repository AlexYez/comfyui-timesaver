# TS WAN Safe Resize

Same idea as Qwen Safe Resize but for WAN-Video. Detects the closest aspect (16:9, 9:16, 1:1) and picks one of three quality presets: Fast (240p), Standard (480p / 832p), High (720p / 1280p). The `interconnection_in/out` string lets several WAN nodes share the same quality tier.

**Use when:** preparing video frames for WAN i2v / t2v models.

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
