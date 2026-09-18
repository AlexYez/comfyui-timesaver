# TS Load Diffusion Model

The `UNETLoader` counterpart, with the same weight casting options (`fp8_e4m3fn`, `fp8_e5m2` and the fast variant). The loaded model remembers how to load itself again — without that, a deep clone or a second GPU would end up holding no weights.

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
