"""FILM model adapted from the ComfyUI frame interpolation implementation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.ops

ops = comfy.ops.disable_weight_init


class FilmConv2d(nn.Module):
    """Conv2d with optional LeakyReLU and FILM-style padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        size: int,
        activation: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        operations=ops,
    ) -> None:
        super().__init__()
        self.even_pad = not size % 2
        self.conv = operations.Conv2d(
            in_channels,
            out_channels,
            kernel_size=size,
            padding=size // 2 if size % 2 else 0,
            device=device,
            dtype=dtype,
        )
        self.activation = nn.LeakyReLU(0.2) if activation else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.even_pad:
            x = F.pad(x, (0, 1, 0, 1))
        x = self.conv(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


def _warp_core(image: torch.Tensor, flow: torch.Tensor, grid_x: torch.Tensor, grid_y: torch.Tensor) -> torch.Tensor:
    dtype = image.dtype
    height, width = flow.shape[2], flow.shape[3]
    dx = flow[:, 0].float() / (width * 0.5)
    dy = flow[:, 1].float() / (height * 0.5)
    grid = torch.stack([grid_x[None, None, :] + dx, grid_y[None, :, None] + dy], dim=3)
    return F.grid_sample(
        image.float(),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    ).to(dtype)


def build_image_pyramid(image: torch.Tensor, pyramid_levels: int) -> list[torch.Tensor]:
    pyramid = [image]
    for _ in range(1, pyramid_levels):
        image = F.avg_pool2d(image, 2, 2)
        pyramid.append(image)
    return pyramid


def flow_pyramid_synthesis(residual_pyramid: list[torch.Tensor]) -> list[torch.Tensor]:
    flow = residual_pyramid[-1]
    flow_pyramid = [flow]
    for residual_flow in residual_pyramid[:-1][::-1]:
        flow = F.interpolate(flow, size=residual_flow.shape[2:4], mode="bilinear", scale_factor=None).mul_(2).add_(
            residual_flow
        )
        flow_pyramid.append(flow)
    flow_pyramid.reverse()
    return flow_pyramid


def multiply_pyramid(pyramid: list[torch.Tensor], scalar: torch.Tensor) -> list[torch.Tensor]:
    return [image * scalar[:, None, None, None] for image in pyramid]


def pyramid_warp(
    feature_pyramid: list[torch.Tensor],
    flow_pyramid: list[torch.Tensor],
    warp_fn,
) -> list[torch.Tensor]:
    return [warp_fn(features, flow) for features, flow in zip(feature_pyramid, flow_pyramid)]


def concatenate_pyramids(pyramid1: list[torch.Tensor], pyramid2: list[torch.Tensor]) -> list[torch.Tensor]:
    return [torch.cat([feat1, feat2], dim=1) for feat1, feat2 in zip(pyramid1, pyramid2)]


class SubTreeExtractor(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        channels: int = 64,
        n_layers: int = 4,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        operations=ops,
    ) -> None:
        super().__init__()
        convs = []
        for index in range(n_layers):
            out_ch = channels << index
            convs.append(
                nn.Sequential(
                    FilmConv2d(in_channels, out_ch, 3, device=device, dtype=dtype, operations=operations),
                    FilmConv2d(out_ch, out_ch, 3, device=device, dtype=dtype, operations=operations),
                )
            )
            in_channels = out_ch
        self.convs = nn.ModuleList(convs)

    def forward(self, image: torch.Tensor, levels: int) -> list[torch.Tensor]:
        head = image
        pyramid: list[torch.Tensor] = []
        for index, layer in enumerate(self.convs):
            head = layer(head)
            pyramid.append(head)
            if index < levels - 1:
                head = F.avg_pool2d(head, 2, 2)
        return pyramid


class FeatureExtractor(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        channels: int = 64,
        sub_levels: int = 4,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        operations=ops,
    ) -> None:
        super().__init__()
        self.extract_sublevels = SubTreeExtractor(
            in_channels,
            channels,
            sub_levels,
            device=device,
            dtype=dtype,
            operations=operations,
        )
        self.sub_levels = sub_levels

    def forward(self, image_pyramid: list[torch.Tensor]) -> list[torch.Tensor]:
        sub_pyramids = [
            self.extract_sublevels(image_pyramid[index], min(len(image_pyramid) - index, self.sub_levels))
            for index in range(len(image_pyramid))
        ]
        feature_pyramid = []
        for index in range(len(image_pyramid)):
            features = sub_pyramids[index][0]
            for sub_index in range(1, self.sub_levels):
                if sub_index <= index:
                    features = torch.cat([features, sub_pyramids[index - sub_index][sub_index]], dim=1)
            feature_pyramid.append(features)
            if index >= self.sub_levels - 1:
                sub_pyramids[index - self.sub_levels + 1] = None
        return feature_pyramid


class FlowEstimator(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_convs: int,
        num_filters: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        operations=ops,
    ) -> None:
        super().__init__()
        self._convs = nn.ModuleList()
        for _ in range(num_convs):
            self._convs.append(
                FilmConv2d(in_channels, num_filters, 3, device=device, dtype=dtype, operations=operations)
            )
            in_channels = num_filters
        self._convs.append(
            FilmConv2d(in_channels, num_filters // 2, 1, device=device, dtype=dtype, operations=operations)
        )
        self._convs.append(
            FilmConv2d(num_filters // 2, 2, 1, activation=False, device=device, dtype=dtype, operations=operations)
        )

    def forward(self, features_a: torch.Tensor, features_b: torch.Tensor) -> torch.Tensor:
        net = torch.cat([features_a, features_b], dim=1)
        for conv in self._convs:
            net = conv(net)
        return net


class PyramidFlowEstimator(nn.Module):
    def __init__(
        self,
        filters: int = 64,
        flow_convs: tuple[int, ...] = (3, 3, 3, 3),
        flow_filters: tuple[int, ...] = (32, 64, 128, 256),
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        operations=ops,
    ) -> None:
        super().__init__()
        in_channels = filters << 1
        predictors = []
        for index in range(len(flow_convs)):
            predictors.append(
                FlowEstimator(
                    in_channels,
                    flow_convs[index],
                    flow_filters[index],
                    device=device,
                    dtype=dtype,
                    operations=operations,
                )
            )
            in_channels += filters << (index + 2)
        self._predictor = predictors[-1]
        self._predictors = nn.ModuleList(predictors[:-1][::-1])

    def forward(self, feature_pyramid_a: list[torch.Tensor], feature_pyramid_b: list[torch.Tensor], warp_fn) -> list[torch.Tensor]:
        levels = len(feature_pyramid_a)
        flow = self._predictor(feature_pyramid_a[-1], feature_pyramid_b[-1])
        residuals = [flow]
        steps = [(index, self._predictor) for index in range(levels - 2, len(self._predictors) - 1, -1)]
        steps += [(len(self._predictors) - 1 - offset, predictor) for offset, predictor in enumerate(self._predictors)]
        for index, predictor in steps:
            flow = F.interpolate(flow, size=feature_pyramid_a[index].shape[2:4], mode="bilinear").mul_(2)
            residual_flow = predictor(feature_pyramid_a[index], warp_fn(feature_pyramid_b[index], flow))
            residuals.append(residual_flow)
            flow = flow.add_(residual_flow)
        residuals.reverse()
        return residuals


def _get_fusion_channels(level: int, filters: int) -> int:
    return (sum(filters << index for index in range(level)) + 3 + 2) * 2


class Fusion(nn.Module):
    def __init__(
        self,
        n_layers: int = 4,
        specialized_layers: int = 3,
        filters: int = 64,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        operations=ops,
    ) -> None:
        super().__init__()
        self.output_conv = operations.Conv2d(filters, 3, kernel_size=1, device=device, dtype=dtype)
        self.convs = nn.ModuleList()
        in_channels = _get_fusion_channels(n_layers, filters)
        increase = 0
        for index in range(n_layers)[::-1]:
            num_filters = (filters << index) if index < specialized_layers else (filters << specialized_layers)
            self.convs.append(
                nn.ModuleList(
                    [
                        FilmConv2d(in_channels, num_filters, 2, activation=False, device=device, dtype=dtype, operations=operations),
                        FilmConv2d(
                            in_channels + (increase or num_filters),
                            num_filters,
                            3,
                            device=device,
                            dtype=dtype,
                            operations=operations,
                        ),
                        FilmConv2d(num_filters, num_filters, 3, device=device, dtype=dtype, operations=operations),
                    ]
                )
            )
            in_channels = num_filters
            increase = _get_fusion_channels(index, filters) - num_filters // 2

    def forward(self, pyramid: list[torch.Tensor]) -> torch.Tensor:
        net = pyramid[-1]
        for reverse_index, layers in enumerate(self.convs):
            index = len(self.convs) - 1 - reverse_index
            net = layers[0](F.interpolate(net, size=pyramid[index].shape[2:4], mode="nearest"))
            net = layers[2](layers[1](torch.cat([pyramid[index], net], dim=1)))
        return self.output_conv(net)


class FILMNet(nn.Module):
    def __init__(
        self,
        pyramid_levels: int = 7,
        fusion_pyramid_levels: int = 5,
        specialized_levels: int = 3,
        sub_levels: int = 4,
        filters: int = 64,
        flow_convs: tuple[int, ...] = (3, 3, 3, 3),
        flow_filters: tuple[int, ...] = (32, 64, 128, 256),
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        operations=ops,
    ) -> None:
        super().__init__()
        self.pyramid_levels = pyramid_levels
        self.fusion_pyramid_levels = fusion_pyramid_levels
        self.extract = FeatureExtractor(3, filters, sub_levels, device=device, dtype=dtype, operations=operations)
        self.predict_flow = PyramidFlowEstimator(
            filters,
            flow_convs,
            flow_filters,
            device=device,
            dtype=dtype,
            operations=operations,
        )
        self.fuse = Fusion(sub_levels, specialized_levels, filters, device=device, dtype=dtype, operations=operations)
        self._warp_grids: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}

    def get_dtype(self) -> torch.dtype:
        return self.extract.extract_sublevels.convs[0][0].conv.weight.dtype

    def _build_warp_grids(self, height: int, width: int, device: torch.device) -> None:
        if (height, width) in self._warp_grids:
            return
        self._warp_grids = {}
        for _ in range(self.pyramid_levels):
            self._warp_grids[(height, width)] = (
                torch.linspace(-(1 - 1 / width), 1 - 1 / width, width, dtype=torch.float32, device=device),
                torch.linspace(-(1 - 1 / height), 1 - 1 / height, height, dtype=torch.float32, device=device),
            )
            height //= 2
            width //= 2

    def warp(self, image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        grid_x, grid_y = self._warp_grids[(flow.shape[2], flow.shape[3])]
        return _warp_core(image, flow, grid_x, grid_y)

    def extract_features(self, img: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        image_pyramid = build_image_pyramid(img, self.pyramid_levels)
        feature_pyramid = self.extract(image_pyramid)
        return image_pyramid, feature_pyramid

    def forward(
        self,
        img0: torch.Tensor,
        img1: torch.Tensor,
        timestep: float | torch.Tensor = 0.5,
        cache: dict[str, tuple[list[torch.Tensor], list[torch.Tensor]]] | None = None,
    ) -> torch.Tensor:
        t_value = timestep.mean(dim=(1, 2, 3)).item() if isinstance(timestep, torch.Tensor) else timestep
        return self.forward_multi_timestep(img0, img1, [t_value], cache=cache)

    def forward_multi_timestep(
        self,
        img0: torch.Tensor,
        img1: torch.Tensor,
        timesteps: list[float],
        cache: dict[str, tuple[list[torch.Tensor], list[torch.Tensor]]] | None = None,
    ) -> torch.Tensor:
        self._build_warp_grids(img0.shape[2], img0.shape[3], img0.device)

        image_pyr0, feat_pyr0 = cache["img0"] if cache and "img0" in cache else self.extract_features(img0)
        image_pyr1, feat_pyr1 = cache["img1"] if cache and "img1" in cache else self.extract_features(img1)

        fwd_flow = flow_pyramid_synthesis(self.predict_flow(feat_pyr0, feat_pyr1, self.warp))[: self.fusion_pyramid_levels]
        bwd_flow = flow_pyramid_synthesis(self.predict_flow(feat_pyr1, feat_pyr0, self.warp))[: self.fusion_pyramid_levels]

        fusion_levels = self.fusion_pyramid_levels
        pyr_to_warp = [
            concatenate_pyramids(image_pyr0[:fusion_levels], feat_pyr0[:fusion_levels]),
            concatenate_pyramids(image_pyr1[:fusion_levels], feat_pyr1[:fusion_levels]),
        ]
        del image_pyr0, image_pyr1, feat_pyr0, feat_pyr1

        results = []
        dt_tensors = torch.tensor(timesteps, device=img0.device, dtype=img0.dtype)
        for index in range(len(timesteps)):
            batch_dt = dt_tensors[index : index + 1]
            bwd_scaled = multiply_pyramid(bwd_flow, batch_dt)
            fwd_scaled = multiply_pyramid(fwd_flow, 1 - batch_dt)
            fwd_warped = pyramid_warp(pyr_to_warp[0], bwd_scaled, self.warp)
            bwd_warped = pyramid_warp(pyr_to_warp[1], fwd_scaled, self.warp)
            aligned = [
                torch.cat([fw, bw, bf, ff], dim=1)
                for fw, bw, bf, ff in zip(fwd_warped, bwd_warped, bwd_scaled, fwd_scaled)
            ]
            del fwd_warped, bwd_warped, bwd_scaled, fwd_scaled
            results.append(self.fuse(aligned))
            del aligned
        return torch.cat(results, dim=0)
