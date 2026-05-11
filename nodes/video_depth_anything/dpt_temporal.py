# Copyright (2025) Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import torch.nn.functional as F
import torch.nn as nn
from .dpt import DPTHead
from .motion_module.motion_module import TemporalModule
from easydict import EasyDict


class DPTHeadTemporal(DPTHead):
    def __init__(self,
        in_channels,
        features=256,
        use_bn=False,
        out_channels=[256, 512, 1024, 1024],
        use_clstoken=False,
        num_frames=32,
        pe='ape'
    ):
        super().__init__(in_channels, features, use_bn, out_channels, use_clstoken)

        assert num_frames > 0
        motion_module_kwargs = EasyDict(num_attention_heads                = 8,
                                        num_transformer_block              = 1,
                                        num_attention_blocks               = 2,
                                        temporal_max_len                   = num_frames,
                                        zero_initialize                    = True,
                                        pos_embedding_type                 = pe)

        self.motion_modules = nn.ModuleList([
            TemporalModule(in_channels=out_channels[2],
                           **motion_module_kwargs),
            TemporalModule(in_channels=out_channels[3],
                           **motion_module_kwargs),
            TemporalModule(in_channels=features,
                           **motion_module_kwargs),
            TemporalModule(in_channels=features,
                           **motion_module_kwargs)
        ])

    def forward(self, out_features, patch_h, patch_w, frame_length, device="cuda"):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]

            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w)).contiguous()

            B, T = x.shape[0] // frame_length, frame_length
            x = self.projects[i](x)
            x = self.resize_layers[i](x)

            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out

        B, T = layer_1.shape[0] // frame_length, frame_length

        layer_3 = self.motion_modules[0](layer_3.unflatten(0, (B, T)).permute(0, 2, 1, 3, 4), None, None).permute(0, 2, 1, 3, 4).flatten(0, 1)
        layer_4 = self.motion_modules[1](layer_4.unflatten(0, (B, T)).permute(0, 2, 1, 3, 4), None, None).permute(0, 2, 1, 3, 4).flatten(0, 1)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_4 = self.motion_modules[2](path_4.unflatten(0, (B, T)).permute(0, 2, 1, 3, 4), None, None).permute(0, 2, 1, 3, 4).flatten(0, 1)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_3 = self.motion_modules[3](path_3.unflatten(0, (B, T)).permute(0, 2, 1, 3, 4), None, None).permute(0, 2, 1, 3, 4).flatten(0, 1)

        # ------------------------------------------------------------------
        # Memory-budget tail: refinenet2 → refinenet1 → output_conv1 →
        # F.interpolate → output_conv2.
        # All of the above are batch-independent in eval mode (Conv2d / BN
        # running-stats / FeatureFusionBlock with no temporal mixing) so we
        # can slice over the T dimension without changing the result.
        # On 4K source @ INFER_LEN=32 the **peak** sits at refinenet1's
        # `F.interpolate(scale_factor=2)`: path_2 (32, 256, 148, 264) →
        # (32, 256, 296, 528) ≈ 2.5 GiB, plus cudnn workspace ≈ 5 GiB total.
        # Sub-chunking by 4 frames drops the peak by ~8×.
        # ------------------------------------------------------------------
        T_out = path_3.shape[0]
        target_h, target_w = int(patch_h * 14), int(patch_w * 14)
        pixels = target_h * target_w
        # Thresholds intentionally aggressive: refinenet1 alone allocates
        # `(T, 256, patch_h*8, patch_w*8) × 2 bytes` which is already >1 GB
        # at 448×798 (358k px) — so sub-chunking has to kick in well below
        # the 518-class boundary, otherwise OOM recurs on the very first
        # retry attempt.
        if pixels >= 500 * 800:        # ≥ ~400 k
            tail_sub_chunk = min(4, T_out)
        elif pixels >= 350 * 600:      # ≥ ~210 k
            tail_sub_chunk = min(8, T_out)
        elif pixels >= 250 * 400:      # ≥ ~100 k
            tail_sub_chunk = min(16, T_out)
        else:
            tail_sub_chunk = T_out

        if tail_sub_chunk >= T_out:
            # Legacy single-batch path (low-res inputs, fits comfortably).
            path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
            path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
            del path_2, path_3
            out = self.scratch.output_conv1(path_1)
            del path_1
            ori_type = out.dtype
            out = F.interpolate(out, (target_h, target_w), mode="bilinear", align_corners=True)
            with torch.autocast(device_type=device, enabled=False):
                out = self.scratch.output_conv2(out.float())
            return out.to(ori_type)

        result = None
        l1_target_size = layer_1_rn.shape[2:]
        cuda_active = path_3.is_cuda
        for i in range(0, T_out, tail_sub_chunk):
            p3_chunk = path_3[i:i + tail_sub_chunk]
            l2_chunk = layer_2_rn[i:i + tail_sub_chunk]
            l1_chunk = layer_1_rn[i:i + tail_sub_chunk]
            p2_chunk = self.scratch.refinenet2(p3_chunk, l2_chunk, size=l1_target_size)
            p1_chunk = self.scratch.refinenet1(p2_chunk, l1_chunk)
            del p2_chunk, p3_chunk, l2_chunk, l1_chunk
            out_chunk = self.scratch.output_conv1(p1_chunk)
            del p1_chunk
            ori_type = out_chunk.dtype
            out_chunk = F.interpolate(out_chunk, (target_h, target_w), mode="bilinear", align_corners=True)
            with torch.autocast(device_type=device, enabled=False):
                out_chunk = self.scratch.output_conv2(out_chunk.float())
            out_chunk = out_chunk.to(ori_type)
            if result is None:
                result = torch.empty(
                    (T_out, out_chunk.shape[1], out_chunk.shape[2], out_chunk.shape[3]),
                    dtype=out_chunk.dtype,
                    device=out_chunk.device,
                )
            result[i:i + tail_sub_chunk].copy_(out_chunk)
            del out_chunk
            if cuda_active:
                torch.cuda.empty_cache()
        return result
