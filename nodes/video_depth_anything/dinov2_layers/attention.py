# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging

import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn


logger = logging.getLogger("dinov2")


try:
    from xformers.ops import memory_efficient_attention, unbind, fmha

    XFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("xFormers not available")
    XFORMERS_AVAILABLE = False


# PyTorch 2.x ships `F.scaled_dot_product_attention`, which selects between
# Flash Attention 2, memory-efficient, and math backends automatically. It
# matches the math of the legacy implementation but is ~2-3× faster on CUDA
# and avoids materialising the full BxHxNxN attention matrix.
SDPA_AVAILABLE = hasattr(F, "scaled_dot_product_attention")


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop_p = float(attn_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        head_dim = C // self.num_heads
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if SDPA_AVAILABLE:
            dropout_p = self.attn_drop_p if self.training else 0.0
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=False)
            x = x.transpose(1, 2).reshape(B, N, C)
        else:
            attn = (q * self.scale) @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        if XFORMERS_AVAILABLE:
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
            q, k, v = unbind(qkv, 2)
            x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
            x = x.reshape([B, N, C])
            x = self.proj(x)
            x = self.proj_drop(x)
            return x

        # xFormers fallback: nested-tensor attn_bias only meaningful with xFormers,
        # which is the only path that should produce one. Plain Tensors go through
        # SDPA (or the math fallback) via the parent class.
        assert attn_bias is None, "xFormers is required for nested tensors usage"
        return super().forward(x)

