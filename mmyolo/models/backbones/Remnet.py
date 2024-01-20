from functools import partial
from typing import Union, List, Tuple

from mmdet.utils import ConfigType, OptMultiConfig
from .csp_darknet import CSPLayerWithTwoConv

from mmyolo.models.utils import make_divisible, make_round
from timm.layers import DropPath, LayerNorm2d
from .base_backbone import BaseBackbone
from mmyolo.registry import MODELS
from torch import nn
import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule as v8ConvModule
from einops import rearrange


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class ConvModule(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride, groups=1, dilation=1, bias=False, norm_layer='bn_2d',
                 act_layer='none', drop_path_rate=0.):
        super().__init__()
        assert stride in [1, 2], 'stride must 1 or 2'
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size, stride, padding=autopad(kernel_size, None, dilation),
                              dilation=dilation, groups=groups, bias=bias)
        self.norm = get_norm(norm_layer)(dim_out)
        self.act = get_act(act_layer)()
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


def get_act(act_name):
    dict_act = {
        'none': nn.Identity,
        'relu': nn.ReLU,
        'relu6': nn.ReLU6,
        'silu': nn.SiLU,
        'hs': nn.Hardswish
    }
    return dict_act[act_name]


def get_norm(norm_layer='in_1d'):
    eps = 1e-6
    norm_dict = {
        'none': nn.Identity,
        'in_1d': partial(nn.InstanceNorm1d, eps=eps),
        'in_2d': partial(nn.InstanceNorm2d, eps=eps),
        'in_3d': partial(nn.InstanceNorm3d, eps=eps),
        'bn_1d': partial(nn.BatchNorm1d, eps=eps),
        'bn_2d': partial(nn.BatchNorm2d, eps=eps),
        # 'bn_2d': partial(nn.SyncBatchNorm, eps=eps),
        'bn_3d': partial(nn.BatchNorm3d, eps=eps),
        'gn': partial(nn.GroupNorm, eps=eps),
        'ln_1d': partial(nn.LayerNorm, eps=eps),
        'ln_2d': partial(LayerNorm2d, eps=eps),
    }
    return norm_dict[norm_layer]


class Conv2d_BN(nn.Sequential):
    def __init__(self, inp, oup, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', nn.Conv2d(
            inp, oup, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(oup))
        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        m = nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation,
                      groups=self.c.groups,
                      device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class DW_1x1(torch.nn.Module):
    def __init__(self, ed) -> None:
        super().__init__()
        self.conv = Conv2d_BN(ed, ed, 3, 1, 1, groups=ed)
        self.conv1 = nn.Conv2d(ed, ed, 1, 1, 0, groups=1)
        self.dim = ed
        self.bn = nn.BatchNorm2d(ed)

    def forward(self, x):
        return self.bn((self.conv(x) + self.conv1(x)) + x)


class RepDW(torch.nn.Module):
    def __init__(self, ed) -> None:
        super().__init__()
        self.conv = Conv2d_BN(ed, ed, 3, 1, 1, groups=ed)
        self.conv1 = nn.Conv2d(ed, ed, 1, 1, 0, groups=ed)
        self.dim = ed
        self.bn = nn.BatchNorm2d(ed)

    def forward(self, x):
        return self.bn((self.conv(x) + self.conv1(x)) + x)

    @torch.no_grad()
    def fuse(self):
        conv = self.conv.fuse()
        conv1 = self.conv1

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = F.pad(conv1_w, [1, 1, 1, 1])

        identity = F.pad(torch.ones(conv1_w.shape[0], conv1_w.shape[1], 1, 1, device=conv1_w.device),
                         [1, 1, 1, 1])

        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        bn = self.bn
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = conv.weight * w[:, None, None, None]
        b = bn.bias + (conv.bias - bn.running_mean) * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        conv.weight.data.copy_(w)
        conv.bias.data.copy_(b)
        return conv


class CED(nn.Module):
    """
    split channel for down sample
    """

    def __init__(self, dim_in, dim_out, exp_ratio, kernel_size, act_layer='relu', use_ca=1, dropout=0.):
        super().__init__()
        stride = 1
        expansion_factor = 4
        self.hidden_dims = int(dim_in * exp_ratio)
        self.split_dims = int(self.hidden_dims * expansion_factor)

        self.conv_1x1 = ConvModule(dim_in, self.hidden_dims, kernel_size=1, stride=stride, bias=False,
                                   act_layer=act_layer)
        # conv
        self.dw_conv = ConvModule(self.hidden_dims, self.hidden_dims, kernel_size, stride=1, groups=self.hidden_dims,
                                  bias=False, act_layer=act_layer, drop_path_rate=dropout)

        self.channel_attn = nn.Identity()
        self.pw_linear = ConvModule(self.split_dims, dim_out, kernel_size=1, stride=stride, bias=False,
                                    norm_layer='bn_2d',
                                    act_layer='none')
        # residual
        self.conv_down_sample = None
        if dim_in != dim_out:
            self.conv_down_sample = nn.Sequential(
                ConvModule(dim_in, dim_in, kernel_size=kernel_size, stride=2, groups=dim_in, bias=False,
                           norm_layer='bn_2d', act_layer=act_layer),
                ConvModule(dim_in, dim_out, 1, 1, 1, bias=True, norm_layer='bn_2d', act_layer='none')
            )
        elif dim_out == dim_in:
            self.conv_down_sample = ConvModule(dim_in, dim_in, kernel_size=kernel_size, stride=2, groups=dim_in,
                                               bias=False, act_layer='none')
        self.drop_path = DropPath(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.conv_1x1(x)
        x = self.dw_conv(x)
        x = rearrange(x, 'b d (h n1) (w n2) -> b (d n1 n2) h w', n1=2, n2=2).contiguous()
        # x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], dim=1)
        x = self.channel_attn(x)
        x = self.pw_linear(x)
        if self.conv_down_sample is not None:
            x = self.drop_path(x) + self.conv_down_sample(shortcut)
        return x


class IMRDB(nn.Module):
    def __init__(self, dim_in, dim_out, exp_ratio, kernel_size, act_layer='relu', use_ca=1, dropout=0.):
        super().__init__()
        stride = 1
        self.hidden_dims = int(dim_in * exp_ratio)
        self.has_skip = dim_in == dim_out

        if dim_in == self.hidden_dims:
            self.conv_1x1 = nn.Identity()
            self.dw_conv = RepDW(self.hidden_dims)
            self.act = get_act(act_layer)()
            self.pw_linear = ConvModule(self.hidden_dims, dim_out, 1, 1, bias=False, act_layer='none',
                                        norm_layer='bn_2d')
        else:
            self.conv_1x1 = ConvModule(dim_in, self.hidden_dims, 1, 1, groups=1, bias=False,
                                       norm_layer='bn_2d', act_layer=act_layer)
            self.dw_conv = RepDW(self.hidden_dims)
            self.act = get_act(act_layer)()
            self.pw_linear = ConvModule(self.hidden_dims, dim_out, 1, 1, groups=1, bias=False, norm_layer='bn_2d',
                                        act_layer='none')

        self.drop_path = DropPath(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.conv_1x1(x)
        residual = x
        x = self.dw_conv(x)
        x = self.act(x) + residual
        x = self.pw_linear(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x


class RemStage(nn.Module):
    def __init__(self, num_blocks, dim_out, exp_ratio, kernel_size, act_layer, use_se, block=None):
        super().__init__()
        if block is None:
            block = IMRDB
        self.block = nn.ModuleList(
            block(
                dim_out, dim_out,
                exp_ratio=exp_ratio,
                kernel_size=kernel_size,
                act_layer=act_layer,
                use_ca=use_se,
                dropout=0.
            )
            for _ in range(num_blocks)
        )

    def forward(self, x):
        for blk in self.block:
            x = blk(x)
        return x


@MODELS.register_module()
class RemNet(BaseBackbone):
    arch_settings = {
        'P5': [[2, 24, 128, 6, 3, 'silu', 0],
               [8, 128, 256, 4, 3, 'silu', 0],
               [2, 256, 512, 2, 3, 'silu', 0],
               [2, 512, 1024, 1, 3, 'silu', 0]],
    }

    def __init__(self,
                 output_stem=24,
                 arch: str = 'P5',
                 last_stage_out_channels: int = 1024,
                 plugins: Union[dict, List[dict]] = None,
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 input_channels: int = 3,
                 out_indices: Tuple[int] = (2, 3, 4),
                 frozen_stages: int = -1,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 norm_eval: bool = False,
                 init_cfg: OptMultiConfig = None):
        self.output_stem = output_stem
        self.arch_settings[arch][-1][2] = last_stage_out_channels
        super().__init__(
            self.arch_settings[arch],
            deepen_factor,
            widen_factor,
            input_channels=input_channels,
            out_indices=out_indices,
            plugins=plugins,
            frozen_stages=frozen_stages,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            norm_eval=norm_eval,
            init_cfg=init_cfg)

    def build_stem_layer(self) -> nn.Module:
        """build stem"""
        return ConvModule(self.input_channels,
                          self.output_stem,
                          kernel_size=3,
                          stride=2,
                          bias=True,
                          norm_layer='bn_2d',
                          act_layer='silu')

    def build_stage_layer(self, stage_idx: int, setting: list) -> list:
        """Build a stage layer.

        Args:
            stage_idx (int): The index of a stage layer.
            setting (list): The architecture setting of a stage layer.
        """
        deepen_factor = 1 if self.deepen_factor == 0.33 else self.deepen_factor
        num_blocks, dim_in, dim_out, exp_ratio, kernel_size, act_layer, use_se = setting
        dim_in = make_divisible(dim_in, self.widen_factor)
        dim_out = make_divisible(dim_out, self.widen_factor)
        num_blocks = make_round(num_blocks, deepen_factor)
        stage = []
        # downsample
        conv_layer = CED(
            self.output_stem if stage_idx == 0 else dim_in,
            dim_out=dim_out,
            exp_ratio=exp_ratio,
            kernel_size=kernel_size,
            act_layer=act_layer,
            use_ca=use_se,
            dropout=0.
        )
        stage.append(conv_layer)
        csp_layer = RemStage(
            num_blocks=num_blocks,
            dim_out=dim_out,
            exp_ratio=exp_ratio,
            kernel_size=kernel_size,
            act_layer=act_layer,
            use_se=use_se
        )
        stage.append(csp_layer)
        return stage

    def init_weights(self):
        """Initialize the parameters."""
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, torch.nn.Conv2d):
                    # In order to be consistent with the source code,
                    # reset the Conv2d initialization parameters
                    m.reset_parameters()
        else:
            super().init_weights()
