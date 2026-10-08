# TS VAE Fix

Removes the faint **2-pixel grid** — a checkerboard and fine stripes at 1–5/255 — that some VAEs leave on every decoded image. It is easy to miss at 100% zoom and shows worst in flat, dark areas, but any sharpening or upscaling afterwards amplifies it, and a restorer such as SeedVR2 takes it for detail worth rebuilding.

**Put it right after `VAE Decode`**, before any resize, sharpening or upscaler. After a resize the grid is already scrambled and there is nothing left to remove.

The grid is not tied to one model: it comes from the stride of the decoder's last upsample stage. Measured on real decodes:

| VAE | grid, /255 | what the node does |
|---|---|---|
| Qwen Image | 1.6–1.7 | removes it |
| Qwen Image 2.1 | 1.8 | removes it |
| Wan 2.1 (also used by Krea 2), Flux, Flux 2, texture-fix VAE for Qwen 2.1 | 0.05–0.4 | passes the image through untouched |
| anything after an upscaler or a resize | ≤ 0.05 | passes the image through untouched |

That is the point of the detector: the node **measures** the grid on every run, so it is safe to leave in any workflow. Below 0.5/255 the image comes out bit for bit as it went in.

How it works, briefly: a notch filter cuts exactly the 2-pixel period (stripes in both directions and the checkerboard) and passes smooth gradients without banding. The correction is capped at `limit` before it is subtracted, so real edges and fine texture survive. The grid sits on the same pixel positions across the whole frame, so averaging those positions keeps it while real detail cancels out — that is how the node tells a grid from a busy picture.

- `mode` — `auto` measures the limit from the image itself; `manual` uses `limit`.
- `limit` — manual only: the largest correction per pixel. Too low and the grid survives in contrasty areas; too high and the finest texture (pores, fabric) softens slightly.
- `skip_when_clean` — off forces the filter to run even when no grid was found.
- `preview_gain`, `preview_view` — only for the `removed_grid` preview. In `full frame` a 2-pixel grid looks like gray noise in a thumbnail; `4x zoom` / `8x zoom` show a magnified centre crop. A healthy preview is a uniform fine pattern; recognisable faces or edges there mean the limit is too high.

A batch (video frames) gets **one** decision and one limit for all frames — deciding per frame would make a clip flicker. The node prints what it measured, for example `grid 1.78/255 (V-stripe) - removed, limit 0.015 auto | edges protected: 6.4%`.

**Use when:** you generate with Qwen Image / Qwen Image 2.1 and then sharpen, upscale or restore the result. The approach follows the notch filter shared by u/Haiku-575 on r/StableDiffusion and [ComfyUI-DeGrid](https://github.com/lunaaispace-eng/ComfyUI-DeGrid); reimplemented here from the maths.

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
