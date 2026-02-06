#!/usr/bin/env python3
"""
Self-contained BasicVSR++ implementation for video super-resolution.
No mmcv/mmedit dependencies - uses torchvision for deformable convolution.

Paper: BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation and Alignment
Original: https://github.com/ckkelvinchan/BasicVSR_PlusPlus
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

import torchvision.ops


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def flow_warp(x, flow, interpolation="bilinear", padding_mode="zeros", align_corners=True):
    """Warp an image or feature map with optical flow.

    Args:
        x: Tensor (n, c, h, w).
        flow: Tensor (n, h, w, 2) – width/height relative offsets.
    """
    if x.size()[-2:] != flow.size()[1:3]:
        raise ValueError(
            f"Spatial sizes of input ({x.size()[-2:]}) and "
            f"flow ({flow.size()[1:3]}) differ."
        )
    _, _, h, w = x.size()
    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, h, device=x.device, dtype=x.dtype),
        torch.arange(0, w, device=x.device, dtype=x.dtype),
        indexing="ij",
    )
    grid = torch.stack((grid_x, grid_y), 2)  # (h, w, 2)
    grid_flow = grid + flow
    grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
    grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
    return F.grid_sample(
        x, grid_flow, mode=interpolation,
        padding_mode=padding_mode, align_corners=align_corners,
    )


def default_init_weights(module, scale=1):
    """Initialize network weights (kaiming)."""
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
            m.weight.data *= scale
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
            m.weight.data *= scale
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def make_layer(block, num_blocks, **kwargs):
    """Stack *num_blocks* copies of *block*."""
    return nn.Sequential(*(block(**kwargs) for _ in range(num_blocks)))


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResidualBlockNoBN(nn.Module):
    """Residual block without batch-norm."""

    def __init__(self, mid_channels=64, res_scale=1.0):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        if res_scale == 1.0:
            self._init_weights()

    def _init_weights(self):
        for m in [self.conv1, self.conv2]:
            default_init_weights(m, 0.1)

    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x))) * self.res_scale


class PixelShufflePack(nn.Module):
    """Conv → PixelShuffle upsample."""

    def __init__(self, in_channels, out_channels, scale_factor, upsample_kernel):
        super().__init__()
        self.upsample_conv = nn.Conv2d(
            in_channels, out_channels * scale_factor * scale_factor,
            upsample_kernel, padding=(upsample_kernel - 1) // 2,
        )
        self.scale_factor = scale_factor
        default_init_weights(self, 1)

    def forward(self, x):
        return F.pixel_shuffle(self.upsample_conv(x), self.scale_factor)


class ResidualBlocksWithInputConv(nn.Module):
    """Input conv + N residual blocks."""

    def __init__(self, in_channels, out_channels=64, num_blocks=30):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            make_layer(ResidualBlockNoBN, num_blocks, mid_channels=out_channels),
        )

    def forward(self, feat):
        return self.main(feat)


# ---------------------------------------------------------------------------
# SPyNet  (optical flow)
# ---------------------------------------------------------------------------

class _ConvModule(nn.Module):
    """Drop-in replacement for mmcv ConvModule (Conv2d + optional ReLU).

    State-dict compatible: weight lives at ``self.conv.weight``.
    """

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, *, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=True)
        self.activate = nn.ReLU(inplace=True) if act else None

    def forward(self, x):
        x = self.conv(x)
        if self.activate is not None:
            x = self.activate(x)
        return x


class SPyNetBasicModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.basic_module = nn.Sequential(
            _ConvModule(8, 32, 7, 1, 3, act=True),
            _ConvModule(32, 64, 7, 1, 3, act=True),
            _ConvModule(64, 32, 7, 1, 3, act=True),
            _ConvModule(32, 16, 7, 1, 3, act=True),
            _ConvModule(16, 2, 7, 1, 3, act=False),
        )

    def forward(self, x):
        return self.basic_module(x)


class SPyNet(nn.Module):
    """SPyNet optical flow network (6-level pyramid)."""

    def __init__(self):
        super().__init__()
        self.basic_module = nn.ModuleList(
            [SPyNetBasicModule() for _ in range(6)]
        )
        self.register_buffer(
            "mean", torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            "std", torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def compute_flow(self, ref, supp):
        n, _, h, w = ref.size()
        ref = [(ref - self.mean) / self.std]
        supp = [(supp - self.mean) / self.std]
        for _ in range(5):
            ref.append(F.avg_pool2d(ref[-1], 2, 2, count_include_pad=False))
            supp.append(F.avg_pool2d(supp[-1], 2, 2, count_include_pad=False))
        ref = ref[::-1]
        supp = supp[::-1]

        flow = ref[0].new_zeros(n, 2, h // 32, w // 32)
        for level in range(len(ref)):
            flow_up = flow if level == 0 else (
                F.interpolate(flow, scale_factor=2, mode="bilinear", align_corners=True) * 2.0
            )
            flow = flow_up + self.basic_module[level](torch.cat([
                ref[level],
                flow_warp(supp[level], flow_up.permute(0, 2, 3, 1), padding_mode="border"),
                flow_up,
            ], 1))
        return flow

    def forward(self, ref, supp):
        h, w = ref.shape[2:4]
        w_up = w if (w % 32) == 0 else 32 * (w // 32 + 1)
        h_up = h if (h % 32) == 0 else 32 * (h // 32 + 1)
        ref = F.interpolate(ref, size=(h_up, w_up), mode="bilinear", align_corners=False)
        supp = F.interpolate(supp, size=(h_up, w_up), mode="bilinear", align_corners=False)
        flow = F.interpolate(
            self.compute_flow(ref, supp), size=(h, w),
            mode="bilinear", align_corners=False,
        )
        flow[:, 0, :, :] *= float(w) / float(w_up)
        flow[:, 1, :, :] *= float(h) / float(h_up)
        return flow


# ---------------------------------------------------------------------------
# Deformable alignment  (uses torchvision DCNv2)
# ---------------------------------------------------------------------------

class SecondOrderDeformableAlignment(nn.Module):
    """Second-order deformable alignment via ``torchvision.ops.deform_conv2d``.

    offset_groups is encoded implicitly in the offset tensor shape and is
    inferred by the torchvision C++ kernel.
    """

    def __init__(
        self, in_channels, out_channels, kernel_size,
        stride=1, padding=0, dilation=1, groups=1,
        deform_groups=1, bias=True, max_residue_magnitude=10,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deform_groups = deform_groups
        self.max_residue_magnitude = max_residue_magnitude

        # Deformable conv weight / bias  (matches mmcv state-dict keys)
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            fan_in = (in_channels // groups) * self.kernel_size[0] * self.kernel_size[1]
            nn.init.uniform_(self.bias, -(fan_in ** -0.5), fan_in ** -0.5)

        # Offset + mask prediction
        self.conv_offset = nn.Sequential(
            nn.Conv2d(3 * out_channels + 4, out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(out_channels, 27 * deform_groups, 3, 1, 1),
        )
        nn.init.constant_(self.conv_offset[-1].weight, 0)
        nn.init.constant_(self.conv_offset[-1].bias, 0)

    def forward(self, x, extra_feat, flow_1, flow_2):
        out = self.conv_offset(torch.cat([extra_feat, flow_1, flow_2], dim=1))
        o1, o2, mask = torch.chunk(out, 3, dim=1)

        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        offset_1, offset_2 = torch.chunk(offset, 2, dim=1)
        offset_1 = offset_1 + flow_1.flip(1).repeat(1, offset_1.size(1) // 2, 1, 1)
        offset_2 = offset_2 + flow_2.flip(1).repeat(1, offset_2.size(1) // 2, 1, 1)
        offset = torch.cat([offset_1, offset_2], dim=1)

        mask = torch.sigmoid(mask)

        return torchvision.ops.deform_conv2d(
            x, offset, self.weight, self.bias,
            stride=self.stride, padding=self.padding,
            dilation=self.dilation, mask=mask,
        )


# ---------------------------------------------------------------------------
# BasicVSR++
# ---------------------------------------------------------------------------

class BasicVSRPlusPlus(nn.Module):
    """BasicVSR++ – x4 video super-resolution.

    Args:
        mid_channels: Intermediate feature channels (default 64).
        num_blocks: Residual blocks per propagation branch (default 7).
        max_residue_magnitude: Offset residue clamp (Eq. 6, default 10).
        is_low_res_input: True → 4x upsample; False → same-size output.
        cpu_cache_length: Offload features to CPU when T > this value.
    """

    def __init__(
        self, mid_channels=64, num_blocks=7, max_residue_magnitude=10,
        is_low_res_input=True, cpu_cache_length=100,
    ):
        super().__init__()
        self.mid_channels = mid_channels
        self.is_low_res_input = is_low_res_input
        self.cpu_cache_length = cpu_cache_length

        # Optical flow
        self.spynet = SPyNet()

        # Feature extraction
        if is_low_res_input:
            self.feat_extract = ResidualBlocksWithInputConv(3, mid_channels, 5)
        else:
            self.feat_extract = nn.Sequential(
                nn.Conv2d(3, mid_channels, 3, 2, 1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(mid_channels, mid_channels, 3, 2, 1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                ResidualBlocksWithInputConv(mid_channels, mid_channels, 5),
            )

        # Propagation branches
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        modules = ["backward_1", "forward_1", "backward_2", "forward_2"]
        for i, module in enumerate(modules):
            self.deform_align[module] = SecondOrderDeformableAlignment(
                2 * mid_channels, mid_channels, 3,
                padding=1, deform_groups=16,
                max_residue_magnitude=max_residue_magnitude,
            )
            self.backbone[module] = ResidualBlocksWithInputConv(
                (2 + i) * mid_channels, mid_channels, num_blocks,
            )

        # Upsampling
        self.reconstruction = ResidualBlocksWithInputConv(
            5 * mid_channels, mid_channels, 5)
        self.upsample1 = PixelShufflePack(mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample2 = PixelShufflePack(mid_channels, 64, 2, upsample_kernel=3)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.is_mirror_extended = False
        self.cpu_cache = False

    # ------------------------------------------------------------------

    def check_if_mirror_extended(self, lqs):
        self.is_mirror_extended = False
        if lqs.size(1) % 2 == 0:
            lqs_1, lqs_2 = torch.chunk(lqs, 2, dim=1)
            if torch.norm(lqs_1 - lqs_2.flip(1)) == 0:
                self.is_mirror_extended = True

    def compute_flow(self, lqs):
        n, t, c, h, w = lqs.size()
        lqs_1 = lqs[:, :-1].reshape(-1, c, h, w)
        lqs_2 = lqs[:, 1:].reshape(-1, c, h, w)

        flows_backward = self.spynet(lqs_1, lqs_2).view(n, t - 1, 2, h, w)

        if self.is_mirror_extended:
            flows_forward = None
        else:
            flows_forward = self.spynet(lqs_2, lqs_1).view(n, t - 1, 2, h, w)

        if self.cpu_cache:
            flows_backward = flows_backward.cpu()
            if flows_forward is not None:
                flows_forward = flows_forward.cpu()

        return flows_forward, flows_backward

    def propagate(self, feats, flows, module_name):
        n, t, _, h, w = flows.size()

        frame_idx = range(0, t + 1)
        flow_idx = range(-1, t)
        mapping_idx = list(range(0, len(feats["spatial"])))
        mapping_idx += mapping_idx[::-1]

        if "backward" in module_name:
            frame_idx = frame_idx[::-1]
            flow_idx = frame_idx

        feat_prop = flows.new_zeros(n, self.mid_channels, h, w)
        for i, idx in enumerate(frame_idx):
            feat_current = feats["spatial"][mapping_idx[idx]]
            if self.cpu_cache:
                feat_current = feat_current.cuda()
                feat_prop = feat_prop.cuda()

            if i > 0:
                flow_n1 = flows[:, flow_idx[i], :, :, :]
                if self.cpu_cache:
                    flow_n1 = flow_n1.cuda()

                cond_n1 = flow_warp(feat_prop, flow_n1.permute(0, 2, 3, 1))

                feat_n2 = torch.zeros_like(feat_prop)
                flow_n2 = torch.zeros_like(flow_n1)
                cond_n2 = torch.zeros_like(cond_n1)

                if i > 1:
                    feat_n2 = feats[module_name][-2]
                    if self.cpu_cache:
                        feat_n2 = feat_n2.cuda()

                    flow_n2 = flows[:, flow_idx[i - 1], :, :, :]
                    if self.cpu_cache:
                        flow_n2 = flow_n2.cuda()

                    flow_n2 = flow_n1 + flow_warp(
                        flow_n2, flow_n1.permute(0, 2, 3, 1))
                    cond_n2 = flow_warp(feat_n2, flow_n2.permute(0, 2, 3, 1))

                cond = torch.cat([cond_n1, feat_current, cond_n2], dim=1)
                feat_prop = torch.cat([feat_prop, feat_n2], dim=1)
                feat_prop = self.deform_align[module_name](
                    feat_prop, cond, flow_n1, flow_n2)

            feat = (
                [feat_current]
                + [feats[k][idx] for k in feats if k not in ("spatial", module_name)]
                + [feat_prop]
            )
            if self.cpu_cache:
                feat = [f.cuda() for f in feat]

            feat = torch.cat(feat, dim=1)
            feat_prop = feat_prop + self.backbone[module_name](feat)
            feats[module_name].append(feat_prop)

            if self.cpu_cache:
                feats[module_name][-1] = feats[module_name][-1].cpu()
                torch.cuda.empty_cache()

        if "backward" in module_name:
            feats[module_name] = feats[module_name][::-1]

        return feats

    def upsample(self, lqs, feats):
        outputs = []
        num_outputs = len(feats["spatial"])
        mapping_idx = list(range(0, num_outputs))
        mapping_idx += mapping_idx[::-1]

        for i in range(lqs.size(1)):
            hr = [feats[k].pop(0) for k in feats if k != "spatial"]
            hr.insert(0, feats["spatial"][mapping_idx[i]])
            hr = torch.cat(hr, dim=1)
            if self.cpu_cache:
                hr = hr.cuda()

            hr = self.reconstruction(hr)
            hr = self.lrelu(self.upsample1(hr))
            hr = self.lrelu(self.upsample2(hr))
            hr = self.lrelu(self.conv_hr(hr))
            hr = self.conv_last(hr)
            if self.is_low_res_input:
                hr += self.img_upsample(lqs[:, i, :, :, :])
            else:
                hr += lqs[:, i, :, :, :]

            if self.cpu_cache:
                hr = hr.cpu()
                torch.cuda.empty_cache()

            outputs.append(hr)

        return torch.stack(outputs, dim=1)

    def forward(self, lqs):
        """Forward pass.

        Args:
            lqs: (n, t, c, h, w) low-quality sequence.

        Returns:
            (n, t, c, 4h, 4w) high-resolution sequence.
        """
        n, t, c, h, w = lqs.size()

        self.cpu_cache = (t > self.cpu_cache_length) and lqs.is_cuda

        if self.is_low_res_input:
            lqs_downsample = lqs.clone()
        else:
            lqs_downsample = F.interpolate(
                lqs.view(-1, c, h, w), scale_factor=0.25, mode="bicubic",
            ).view(n, t, c, h // 4, w // 4)

        self.check_if_mirror_extended(lqs)

        # Spatial features
        feats = {}
        if self.cpu_cache:
            feats["spatial"] = []
            for i in range(t):
                feat = self.feat_extract(lqs[:, i, :, :, :]).cpu()
                feats["spatial"].append(feat)
                torch.cuda.empty_cache()
        else:
            feats_ = self.feat_extract(lqs.view(-1, c, h, w))
            h_, w_ = feats_.shape[2:]
            feats_ = feats_.view(n, t, -1, h_, w_)
            feats["spatial"] = [feats_[:, i, :, :, :] for i in range(t)]

        # Optical flow
        assert lqs_downsample.size(3) >= 64 and lqs_downsample.size(4) >= 64, (
            f"Input must be at least 64x64, got "
            f"{lqs_downsample.size(3)}x{lqs_downsample.size(4)}."
        )
        flows_forward, flows_backward = self.compute_flow(lqs_downsample)

        # Propagation
        for iter_ in [1, 2]:
            for direction in ["backward", "forward"]:
                module = f"{direction}_{iter_}"
                feats[module] = []

                if direction == "backward":
                    flows = flows_backward
                elif flows_forward is not None:
                    flows = flows_forward
                else:
                    flows = flows_backward.flip(1)

                feats = self.propagate(feats, flows, module)
                if self.cpu_cache:
                    del flows
                    torch.cuda.empty_cache()

        return self.upsample(lqs, feats)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_basicvsr_checkpoint(model, checkpoint_path):
    """Load an mmediting-format BasicVSR++ checkpoint.

    Handles the ``generator.`` prefix and ``state_dict`` wrapper that
    mmediting checkpoints use.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    state_dict = ckpt.get("state_dict", ckpt)
    # Strip 'generator.' prefix added by mmediting
    state_dict = {
        k.replace("generator.", "", 1): v
        for k, v in state_dict.items()
    }

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    return model
