# Apache-2.0; source revision 9b686c5, see LICENSE and NOTICE in this directory.
"""Memory-lean forward for NVIDIA's fast H3 VAE decoder (fast_vae_decoder.py, unchanged).

The reference plain-PyTorch forward builds full-clip temporaries every layer (zero-padded q/k/v windows, padded block
copies for the global tokens, a fp32 [T, H, W, 3072] projection). At 2K x 5 s that peaks around 70 GB, which on a
72 GB card spills into system RAM. This forward computes the same thing one temporal window at a time:

* q / k / v for the whole clip are kept once (bf16); each temporal window (5 latents, shifted by 2 on odd layers) is
  partitioned into its 16x16 spatial windows and attended on its own, with the global tokens of the native blocks it
  overlaps — exactly the reference windows, masks and window order;
* a native block's global tokens attend to that block's real tokens + its own global tokens (the reference pads to
  whole blocks and masks the padding out, which is the same softmax);
* feed-forward, output projection and the final pixel projection run in token / frame chunks.

Low VRAM keeps at most two native blocks of projected Q/K/V on the GPU and
updates the FP32 residual as soon as each window finishes. Every frame is
projected once, with the original attention windows and global tokens.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from .fast_vae_decoder import OUT_CH, PATCH, PATCH_T, NonCausalViTDecoder, _apply_qk_norm, _apply_rotary_pos_emb_impl

ROWS = 65536          # tokens per chunk for the per-token linears


def _rows(fn, x, out_dim=None):
    """fn applied to x (..., D) in token chunks."""
    flat = x.reshape(-1, x.shape[-1])
    first = fn(flat[:ROWS])
    out = first.new_empty(flat.shape[0], first.shape[-1])
    out[:first.shape[0]] = first
    for start in range(ROWS, flat.shape[0], ROWS):
        out[start:start + ROWS] = fn(flat[start:start + ROWS])
    return out.view(*x.shape[:-1], first.shape[-1])


class LeanNonCausalViTDecoder(NonCausalViTDecoder):
    def _ff(self, blk, x):
        def ff(t):
            gate, up = blk.ff.w1(blk.norm2(t.float()).to(t.dtype)).chunk(2, dim=-1)
            return blk.ff.w2(F.silu(gate) * up)
        return ff(x) if x.shape[:-1].numel() <= ROWS else _rows(ff, x)

    def _project_frames(self, blk, x, cos, sin):
        B, T, H, W, _ = x.shape
        nh_ = self.heads
        qs = ks = vs = None
        for t in range(T):
            h = blk.norm1(x[:, t].float()).to(x.dtype)
            qt, kt, vt = blk.attn.to_qkv(h).view(B, H, W, nh_, 3 * self.dh).chunk(3, -1)
            if blk.attn.norm_q is not None:
                qt, kt = _apply_qk_norm(blk.attn.norm_q, qt), _apply_qk_norm(blk.attn.norm_k, kt)
            if qs is None:
                qs = qt.new_empty((B, T, H, W, nh_, self.dh))
                ks, vs = torch.empty_like(qs), torch.empty_like(qs)
            rope = (cos[:, t].expand(B, -1, -1, -1), sin[:, t].expand(B, -1, -1, -1))
            qs[:, t] = _apply_rotary_pos_emb_impl(qt.reshape(B, H * W, nh_, self.dh), rope).view(B, H, W, nh_, self.dh)
            ks[:, t] = _apply_rotary_pos_emb_impl(kt.reshape(B, H * W, nh_, self.dh), rope).view(B, H, W, nh_, self.dh)
            vs[:, t] = vt
            del h, qt, kt, vt
        return qs, ks, vs

    def _attend_window(self, qkv, kg, vg, b0, b1, shift, offset):
        B, T, H, W, nh_, dh = qkv[0].shape
        S = kg.shape[2]
        qw, vmask, info = self._partition(qkv[0], shift, offset)
        kw, _, _ = self._partition(qkv[1], shift, offset)
        vw, _, _ = self._partition(qkv[2], shift, offset)
        nwin = info[8] * info[9]
        kgw = kg[:, [b0, b1]].reshape(B, 1, 2 * S, nh_, dh).expand(B, nwin, 2 * S, nh_, dh)
        vgw = vg[:, [b0, b1]].reshape(B, 1, 2 * S, nh_, dh).expand(B, nwin, 2 * S, nh_, dh)
        gmask = torch.tensor([True] * S + [b1 != b0] * S, device=kg.device).expand(nwin, 2 * S)
        keys = torch.cat([kw, kgw.reshape(B * nwin, 2 * S, nh_, dh)], 1).transpose(1, 2)
        del kw
        values = torch.cat([vw, vgw.reshape(B * nwin, 2 * S, nh_, dh)], 1).transpose(1, 2)
        del vw
        mask = torch.cat([vmask, gmask], 1).repeat(B, 1)[:, None, None, :]
        attended = F.scaled_dot_product_attention(qw.transpose(1, 2), keys, values, attn_mask=mask).transpose(1, 2)
        return self._unpartition(attended, info, shift, offset).reshape(B, T, H, W, nh_ * dh)

    @staticmethod
    def _attend_global(qg, kg, vg, k, v):
        B, S, heads, width = qg.shape
        keys = torch.cat([k.reshape(B, -1, heads, width), kg], 1).transpose(1, 2)
        values = torch.cat([v.reshape(B, -1, heads, width), vg], 1).transpose(1, 2)
        return F.scaled_dot_product_attention(qg.transpose(1, 2), keys, values).transpose(1, 2).reshape(B, 1, S, -1)

    def _layer(self, li, blk, x, g, low_vram=False):
        B, T, H, W, nb, S = self._geom
        tb, nh_ = self.block, self.heads
        s, st = self._shifts(li)
        qg, kg, vg = blk.attn.to_qkv(blk.norm1(g.float()).to(g.dtype)).view(B, nb, S, nh_, 3 * self.dh).chunk(3, -1)
        if blk.attn.norm_q is not None:
            qg, kg = _apply_qk_norm(blk.attn.norm_q, qg), _apply_qk_norm(blk.attn.norm_k, kg)
        cos, sin = self._rope(T, H, W, s, st, x.device)
        cos, sin = cos.view(1, T, H * W, 1, -1), sin.view(1, T, H * W, 1, -1)
        all_qkv = None if low_vram else self._project_frames(blk, x, cos, sin)
        blocks = {}
        og = []
        o = None
        stream_residual = low_vram and B == 1 and x.dtype == torch.float32

        def project_output(t):
            return blk.attn.to_out(t) * blk.scale1

        for j in range((T + st + tb - 1) // tb):
            lo, hi = max(0, j * tb - st), min(T, (j + 1) * tb - st)
            b0, b1 = lo // tb, (hi - 1) // tb
            if low_vram:
                blocks.pop(b0 - 1, None)
                for block in range(b0, b1 + 1):
                    if block not in blocks:
                        start, end = block * tb, min(T, (block + 1) * tb)
                        blocks[block] = self._project_frames(blk, x[:, start:end], cos[:, start:end], sin[:, start:end])
                        og.append(self._attend_global(qg[:, block], kg[:, block], vg[:, block], *blocks[block][1:]))
                if b0 == b1:
                    qkv = tuple(p[:, lo - b0 * tb:hi - b0 * tb] for p in blocks[b0])
                else:
                    qkv = tuple(torch.cat((a[:, lo - b0 * tb:], b[:, :hi - b1 * tb]), dim=1)
                                for a, b in zip(blocks[b0], blocks[b1], strict=True))
            else:
                qkv = tuple(p[:, lo:hi] for p in all_qkv)
            ow = self._attend_window(qkv, kg, vg, b0, b1, s, lo + st - j * tb)
            if stream_residual:
                # Both contributing blocks have already been projected from the
                # original x. Any later window reuses those cached projections.
                self._residual(x[:, lo:hi], ow, project_output)
            else:
                if o is None:
                    o = ow.new_empty((B, T, H, W, nh_ * self.dh))
                o[:, lo:hi] = ow
            del qkv, ow
        if all_qkv is not None:
            for b in range(nb):
                lo, hi = b * tb, min(T, (b + 1) * tb)
                og.append(self._attend_global(qg[:, b], kg[:, b], vg[:, b], all_qkv[1][:, lo:hi], all_qkv[2][:, lo:hi]))
        del all_qkv, blocks, cos, sin
        og = torch.cat(og, 1)

        if o is not None:
            x = self._residual(x, o, project_output)
            del o
        g = g + blk.attn.to_out(og) * blk.scale1
        x = self._residual(x, x, lambda t: self._ff(blk, t) * blk.scale2)
        g = g + self._ff(blk, g) * blk.scale2
        return x, g

    @staticmethod
    def _residual(x, src, fn):
        """x + fn(src) computed in token chunks and added in place. The reference's out-of-place `x + delta` promotes
        x to the delta's dtype (fp32: the layer scales are fp32), so x is promoted once before adding in place."""
        flat_x, flat_src = x.view(-1, x.shape[-1]), src.reshape(-1, src.shape[-1])
        for start in range(0, flat_x.shape[0], ROWS):
            delta = fn(flat_src[start:start + ROWS])
            if delta.dtype != flat_x.dtype and torch.promote_types(flat_x.dtype, delta.dtype) != flat_x.dtype:
                if src is x:
                    raise RuntimeError("residual dtype changed mid-update")
                x = x.to(torch.promote_types(x.dtype, delta.dtype))
                flat_x = x.view(-1, x.shape[-1])
            flat_x[start:start + ROWS] += delta
        return x

    def forward(self, z, sink=None, *, low_vram=False):
        """Same output as the reference forward. With `sink`, frames are handed over as they are made —
        sink(first_frame_index, pixels [B, 3, n, 16H, 16W] fp32) — and nothing clip-sized is kept on the device."""
        B, T, _, H, W = z.shape
        tb = self.block
        nb = (T + tb - 1) // tb
        compute_dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else z.dtype
        raw = z.float() * self.latents_std.view(1, 1, -1, 1, 1) + self.latents_mean.view(1, 1, -1, 1, 1)
        raw = self.post_quant_conv(raw.transpose(1, 2)).permute(0, 2, 3, 4, 1)
        with torch.autocast("cuda", enabled=False):
            x = _rows(lambda t: self.x_embedder(t.float()).to(compute_dtype), raw)
        D = x.shape[-1]
        S = self.register_tokens.shape[1] + 1
        g = torch.cat([self.register_tokens.expand(B, -1, -1), x.new_zeros(B, 1, D)], 1).to(compute_dtype)
        g = g.unsqueeze(1).expand(B, nb, S, D).contiguous()
        self._geom = (B, T, H, W, nb, S)
        for li, blk in enumerate(self.blocks):
            x, g = self._layer(li, blk, x, g, low_vram)
        # pixels frame by frame; the first latent of every native block keeps its last frame only
        keep = [PATCH_T - (self.frames_to_trim if t % tb == 0 else 0) for t in range(T)]
        out = None
        if sink is None:
            out = torch.empty(B, OUT_CH, sum(keep), H * PATCH, W * PATCH, device=z.device, dtype=torch.float32)
            sink = lambda f, p: out.__setitem__((slice(None), slice(None), slice(f, f + p.shape[2])), p)  # noqa: E731
        f = 0
        for t in range(T):
            xt = self.norm_out(x[:, t])
            with torch.autocast("cuda", enabled=False):
                pt = self.proj_out(xt.float())                                   # B, H, W, 3*4*16*16
            pt = pt.view(B, H, W, OUT_CH, PATCH_T, PATCH, PATCH).permute(0, 3, 4, 1, 5, 2, 6)
            pt = pt.reshape(B, OUT_CH, PATCH_T, H * PATCH, W * PATCH)[:, :, PATCH_T - keep[t]:]
            sink(f, pt)
            f += keep[t]
        return out
