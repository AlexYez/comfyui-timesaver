"""Pure-PyTorch FFCResNetGenerator for big-lama (advimman/lama).

Mirrors the TorchScript graph of big-lama.pt so a state_dict (e.g. from
big-lama.safetensors) loads via ``load_state_dict(strict=True)`` with the
leading ``generator.`` prefix stripped.

Submodule layout (matches state_dict keys ``generator.model.<idx>.*``):

    [0]  ReflectionPad2d(3)
    [1]  FFC_BN_ACT(in=4, out=64,  k=7, s=1, p=0, ratio_gin=0,    ratio_gout=0)
    [2]  FFC_BN_ACT(64,  128,      k=3, s=2, p=1, ratio_gin=0,    ratio_gout=0)
    [3]  FFC_BN_ACT(128, 256,      k=3, s=2, p=1, ratio_gin=0,    ratio_gout=0)
    [4]  FFC_BN_ACT(256, 512,      k=3, s=2, p=1, ratio_gin=0,    ratio_gout=0.75)  # transition
    [5..22]  FFCResnetBlock(512, ratio=0.75) x 18
    [23] ConcatTupleLayer
    [24] ConvTranspose2d(512, 256, k=3, s=2, p=1, op=1)
    [25] BatchNorm2d(256)
    [26] ReLU(inplace=True)
    [27] ConvTranspose2d(256, 128, k=3, s=2, p=1, op=1)
    [28] BatchNorm2d(128)
    [29] ReLU(inplace=True)
    [30] ConvTranspose2d(128, 64,  k=3, s=2, p=1, op=1)
    [31] BatchNorm2d(64)
    [32] ReLU(inplace=True)
    [33] ReflectionPad2d(3)
    [34] Conv2d(64, 3, k=7, p=0)
    [35] Sigmoid

Wrapped by LamaInpaintingModel which composites:
    output = mask * generator(cat(img * (1 - mask), mask)) + (1 - mask) * img
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class FourierUnit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv_layer = nn.Conv2d(
            in_channels=in_channels * 2,
            out_channels=out_channels * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        ffted = torch.fft.rfftn(x, dim=(-2, -1), norm="ortho")
        ffted = torch.stack([ffted.real, ffted.imag], dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view(batch, -1, ffted.size(3), ffted.size(4))

        ffted = self.conv_layer(ffted)
        ffted = self.relu(self.bn(ffted))

        ffted = ffted.view(batch, -1, 2, ffted.size(2), ffted.size(3))
        ffted = ffted.permute(0, 1, 3, 4, 2).contiguous()
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])
        return torch.fft.irfftn(ffted, s=x.shape[-2:], dim=(-2, -1), norm="ortho")


class SpectralTransform(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.downsample = nn.Identity() if stride == 1 else nn.AvgPool2d(kernel_size=2, stride=2)
        mid = out_channels // 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )
        self.fu = FourierUnit(mid, mid)
        self.conv2 = nn.Conv2d(mid, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        x = self.conv1(x)
        out = self.fu(x)
        return self.conv2(x + out)


class FFC(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        ratio_gin: float,
        ratio_gout: float,
        stride: int = 1,
        padding: int = 0,
        padding_mode: str = "reflect",
    ) -> None:
        super().__init__()
        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg

        def _local_conv(ic: int, oc: int) -> nn.Module:
            if ic <= 0 or oc <= 0:
                return nn.Identity()
            return nn.Conv2d(
                ic,
                oc,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                padding_mode=padding_mode,
                bias=False,
            )

        self.convl2l = _local_conv(in_cl, out_cl)
        self.convl2g = _local_conv(in_cl, out_cg)
        self.convg2l = _local_conv(in_cg, out_cl)
        if in_cg > 0 and out_cg > 0:
            self.convg2g = SpectralTransform(in_cg, out_cg, stride=stride)
        else:
            self.convg2g = nn.Identity()

    def forward(self, x: Any):
        if isinstance(x, tuple):
            x_l, x_g = x
        else:
            x_l, x_g = x, 0

        out_xl: Any = 0
        out_xg: Any = 0

        if self.ratio_gout != 1:
            l_path = self.convl2l(x_l) if isinstance(self.convl2l, nn.Conv2d) else 0
            if isinstance(self.convg2l, nn.Conv2d) and torch.is_tensor(x_g):
                g_path = self.convg2l(x_g)
            else:
                g_path = 0
            out_xl = l_path + g_path if torch.is_tensor(l_path) or torch.is_tensor(g_path) else 0

        if self.ratio_gout != 0:
            l_path = self.convl2g(x_l) if isinstance(self.convl2g, nn.Conv2d) else 0
            if isinstance(self.convg2g, SpectralTransform) and torch.is_tensor(x_g):
                g_path = self.convg2g(x_g)
            else:
                g_path = 0
            out_xg = l_path + g_path if torch.is_tensor(l_path) or torch.is_tensor(g_path) else 0

        return out_xl, out_xg


class FFC_BN_ACT(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        ratio_gin: float,
        ratio_gout: float,
        stride: int = 1,
        padding: int = 0,
        padding_mode: str = "reflect",
    ) -> None:
        super().__init__()
        self.ffc = FFC(
            in_channels,
            out_channels,
            kernel_size,
            ratio_gin=ratio_gin,
            ratio_gout=ratio_gout,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
        )
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg

        self.bn_l = nn.BatchNorm2d(out_cl) if out_cl > 0 else nn.Identity()
        self.bn_g = nn.BatchNorm2d(out_cg) if out_cg > 0 else nn.Identity()
        self.act_l = nn.ReLU(inplace=True) if out_cl > 0 else nn.Identity()
        self.act_g = nn.ReLU(inplace=True) if out_cg > 0 else nn.Identity()

    def forward(self, x: Any):
        x_l, x_g = self.ffc(x)
        if torch.is_tensor(x_l):
            x_l = self.act_l(self.bn_l(x_l))
        if torch.is_tensor(x_g):
            x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g


class FFCResnetBlock(nn.Module):
    def __init__(self, dim: int, ratio_gin: float = 0.75, ratio_gout: float = 0.75) -> None:
        super().__init__()
        self.conv1 = FFC_BN_ACT(
            dim,
            dim,
            kernel_size=3,
            ratio_gin=ratio_gin,
            ratio_gout=ratio_gout,
            stride=1,
            padding=1,
            padding_mode="reflect",
        )
        self.conv2 = FFC_BN_ACT(
            dim,
            dim,
            kernel_size=3,
            ratio_gin=ratio_gin,
            ratio_gout=ratio_gout,
            stride=1,
            padding=1,
            padding_mode="reflect",
        )

    def forward(self, x):
        id_l, id_g = x
        x_l, x_g = self.conv1(x)
        x_l, x_g = self.conv2((x_l, x_g))
        return id_l + x_l, id_g + x_g


class ConcatTupleLayer(nn.Module):
    def forward(self, x):
        x_l, x_g = x
        if not torch.is_tensor(x_l):
            return x_g
        if not torch.is_tensor(x_g):
            return x_l
        return torch.cat([x_l, x_g], dim=1)


class FFCResNetGenerator(nn.Module):
    def __init__(
        self,
        input_nc: int = 4,
        output_nc: int = 3,
        ngf: int = 64,
        n_downsampling: int = 3,
        n_blocks: int = 18,
        ratio: float = 0.75,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            FFC_BN_ACT(
                input_nc,
                ngf,
                kernel_size=7,
                ratio_gin=0,
                ratio_gout=0,
                stride=1,
                padding=0,
                padding_mode="reflect",
            ),
        ]

        for i in range(n_downsampling):
            mult = 2 ** i
            in_c = ngf * mult
            out_c = ngf * mult * 2
            ratio_gout_here = 0.0 if i < n_downsampling - 1 else ratio
            layers.append(
                FFC_BN_ACT(
                    in_c,
                    out_c,
                    kernel_size=3,
                    ratio_gin=0,
                    ratio_gout=ratio_gout_here,
                    stride=2,
                    padding=1,
                    padding_mode="reflect",
                )
            )

        bottleneck = ngf * (2 ** n_downsampling)
        for _ in range(n_blocks):
            layers.append(FFCResnetBlock(bottleneck, ratio_gin=ratio, ratio_gout=ratio))

        layers.append(ConcatTupleLayer())

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            in_c = ngf * mult
            out_c = ngf * mult // 2
            layers.append(
                nn.ConvTranspose2d(
                    in_c,
                    out_c,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                )
            )
            layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.ReflectionPad2d(3))
        layers.append(nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class LamaInpaintingModel(nn.Module):
    """LaMa wrapper: forward(img, mask) -> Tensor (matches big-lama.pt)."""

    def __init__(self) -> None:
        super().__init__()
        self.generator = FFCResNetGenerator()

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        masked_img = img * (1 - mask)
        inp = torch.cat([masked_img, mask], dim=1)
        gen_out = self.generator(inp)
        return mask * gen_out + (1 - mask) * img


def build_lama_inpainter(state_dict: dict[str, torch.Tensor]) -> LamaInpaintingModel:
    """Instantiate LamaInpaintingModel and load state_dict (strict).

    Accepts the dump produced by ``torch.jit.load(big-lama.pt).state_dict()``
    or ``safetensors.torch.load_file(big-lama.safetensors)`` — both share the
    ``generator.<...>`` key prefix.
    """
    model = LamaInpaintingModel()
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:  # strict=True raises on mismatch; this is defence in depth
        raise RuntimeError(
            f"LaMa state_dict mismatch: missing={missing[:5]}... unexpected={unexpected[:5]}..."
        )
    model.eval()
    return model
