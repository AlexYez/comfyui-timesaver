# TS LTX Final Latent Selector

Picks the first- or second-stage latent **before** the decode, instead of decoding both
and throwing one away. The inputs are lazy, so switching the upscaler off stops costing
sampler time, not just decoder time — and one decode downstream means one place where
the HDR conversion happens.

**Use when:** your graph has a two-stage switch. It is worth wiring even without HDR.

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
