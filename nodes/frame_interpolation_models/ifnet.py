"""RIFE IFNet model adapted from the ComfyUI frame interpolation implementation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.ops

ops = comfy.ops.disable_weight_init


def _warp(img: torch.Tensor, flow: torch.Tensor, warp_grids: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    """Warp an image using dense optical flow."""
    batch, _, height, width = img.shape
    base_grid, flow_div = warp_grids[(height, width)]
    flow_norm = torch.cat([flow[:, 0:1] / flow_div[0], flow[:, 1:2] / flow_div[1]], dim=1).float()
    grid = (base_grid.expand(batch, -1, -1, -1) + flow_norm).permute(0, 2, 3, 1)
    return F.grid_sample(
        img.float(),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    ).to(img.dtype)


class Head(nn.Module):
    """Lightweight feature extractor used by RIFE."""

    def __init__(
        self,
        out_ch: int = 4,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        operations=ops,
    ) -> None:
        super().__init__()
        self.cnn0 = operations.Conv2d(3, 16, 3, 2, 1, device=device, dtype=dtype)
        self.cnn1 = operations.Conv2d(16, 16, 3, 1, 1, device=device, dtype=dtype)
        self.cnn2 = operations.Conv2d(16, 16, 3, 1, 1, device=device, dtype=dtype)
        self.cnn3 = operations.ConvTranspose2d(16, out_ch, 4, 2, 1, device=device, dtype=dtype)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.cnn0(x))
        x = self.relu(self.cnn1(x))
        x = self.relu(self.cnn2(x))
        return self.cnn3(x)


class ResConv(nn.Module):
    """Residual convolution block used inside IFBlock."""

    def __init__(
        self,
        channels: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        operations=ops,
    ) -> None:
        super().__init__()
        self.conv = operations.Conv2d(channels, channels, 3, 1, 1, device=device, dtype=dtype)
        self.beta = nn.Parameter(torch.ones((1, channels, 1, 1), device=device, dtype=dtype))
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(torch.addcmul(x, self.conv(x), self.beta))


class IFBlock(nn.Module):
    """Single refinement stage inside IFNet."""

    def __init__(
        self,
        in_planes: int,
        channels: int = 64,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        operations=ops,
    ) -> None:
        super().__init__()
        self.conv0 = nn.Sequential(
            nn.Sequential(
                operations.Conv2d(in_planes, channels // 2, 3, 2, 1, device=device, dtype=dtype),
                nn.LeakyReLU(0.2, True),
            ),
            nn.Sequential(
                operations.Conv2d(channels // 2, channels, 3, 2, 1, device=device, dtype=dtype),
                nn.LeakyReLU(0.2, True),
            ),
        )
        self.convblock = nn.Sequential(
            *(ResConv(channels, device=device, dtype=dtype, operations=operations) for _ in range(8))
        )
        self.lastconv = nn.Sequential(
            operations.ConvTranspose2d(channels, 4 * 13, 4, 2, 1, device=device, dtype=dtype),
            nn.PixelShuffle(2),
        )

    def forward(
        self,
        x: torch.Tensor,
        flow: torch.Tensor | None = None,
        scale: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = F.interpolate(x, scale_factor=1.0 / scale, mode="bilinear")
        if flow is not None:
            flow = F.interpolate(flow, scale_factor=1.0 / scale, mode="bilinear").div_(scale)
            x = torch.cat((x, flow), dim=1)
        feat = self.convblock(self.conv0(x))
        tmp = F.interpolate(self.lastconv(feat), scale_factor=scale, mode="bilinear")
        return tmp[:, :4] * scale, tmp[:, 4:5], tmp[:, 5:]


class IFNet(nn.Module):
    """RIFE IFNet with feature caching support across adjacent frame pairs."""

    def __init__(
        self,
        head_ch: int = 4,
        channels: tuple[int, int, int, int, int] = (192, 128, 96, 64, 32),
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        operations=ops,
    ) -> None:
        super().__init__()
        self.encode = Head(out_ch=head_ch, device=device, dtype=dtype, operations=operations)
        block_in = [7 + 2 * head_ch] + [8 + 4 + 8 + 2 * head_ch] * 4
        self.blocks = nn.ModuleList(
            [IFBlock(block_in[i], channels[i], device=device, dtype=dtype, operations=operations) for i in range(5)]
        )
        self.scale_list = [16, 8, 4, 2, 1]
        self.pad_align = 64
        self._warp_grids: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}

    def get_dtype(self) -> torch.dtype:
        """Return the active dtype of the model."""
        return self.encode.cnn0.weight.dtype

    def _build_warp_grids(self, height: int, width: int, device: torch.device) -> None:
        if (height, width) in self._warp_grids:
            return
        self._warp_grids = {}
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=device, dtype=torch.float32),
            torch.linspace(-1.0, 1.0, width, device=device, dtype=torch.float32),
            indexing="ij",
        )
        self._warp_grids[(height, width)] = (
            torch.stack((grid_x, grid_y), dim=0).unsqueeze(0),
            torch.tensor([(width - 1.0) / 2.0, (height - 1.0) / 2.0], dtype=torch.float32, device=device),
        )

    def warp(self, img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """Warp an image tensor by flow."""
        return _warp(img, flow, self._warp_grids)

    def extract_features(self, img: torch.Tensor) -> torch.Tensor:
        """Extract reusable encoder features for a single frame."""
        return self.encode(img)

    def forward(
        self,
        img0: torch.Tensor,
        img1: torch.Tensor,
        timestep: float | torch.Tensor = 0.5,
        cache: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.full(
                (img0.shape[0], 1, img0.shape[2], img0.shape[3]),
                timestep,
                device=img0.device,
                dtype=img0.dtype,
            )

        self._build_warp_grids(img0.shape[2], img0.shape[3], img0.device)

        batch = img0.shape[0]
        feat0 = cache["img0"].expand(batch, -1, -1, -1) if cache and "img0" in cache else self.encode(img0)
        feat1 = cache["img1"].expand(batch, -1, -1, -1) if cache and "img1" in cache else self.encode(img1)

        flow = None
        mask = None
        feat = None
        warped_img0 = img0
        warped_img1 = img1

        for index, block in enumerate(self.blocks):
            if flow is None:
                flow, mask, feat = block(
                    torch.cat((img0, img1, feat0, feat1, timestep), dim=1),
                    None,
                    scale=self.scale_list[index],
                )
            else:
                flow_delta, mask, feat = block(
                    torch.cat(
                        (
                            warped_img0,
                            warped_img1,
                            self.warp(feat0, flow[:, :2]),
                            self.warp(feat1, flow[:, 2:4]),
                            timestep,
                            mask,
                            feat,
                        ),
                        dim=1,
                    ),
                    flow,
                    scale=self.scale_list[index],
                )
                flow = flow.add_(flow_delta)
            warped_img0 = self.warp(img0, flow[:, :2])
            warped_img1 = self.warp(img1, flow[:, 2:4])

        return torch.lerp(warped_img1, warped_img0, torch.sigmoid(mask))


def detect_rife_config(state_dict: dict[str, torch.Tensor]) -> tuple[int, tuple[int, int, int, int, int]]:
    """Detect RIFE configuration from checkpoint weights."""
    head_ch = state_dict["encode.cnn3.weight"].shape[1]
    channels: list[int] = []
    for index in range(5):
        key = f"blocks.{index}.conv0.1.0.weight"
        if key in state_dict:
            channels.append(state_dict[key].shape[0])
    if len(channels) != 5:
        raise ValueError(f"Unsupported RIFE model: expected 5 blocks, found {len(channels)}")
    return head_ch, tuple(channels)
