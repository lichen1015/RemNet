from functools import partial
from typing import Union, List, Tuple, Optional

from mmdet.utils import ConfigType, OptMultiConfig
from .csp_darknet import CSPLayerWithTwoConv

from mmyolo.models.utils import make_divisible, make_round
from timm.layers import DropPath, LayerNorm2d
from .base_backbone import BaseBackbone
from mmyolo.registry import MODELS
from torch import nn, Tensor
import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule as v8ConvModule, build_norm_layer
from einops import rearrange
import numpy as np

from mmcv.cnn import ConvModule as ModelConv


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


class RepDWBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 padding_mode='zeros',
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='ReLU', inplace=True),
                 use_se: bool = False,
                 use_alpha: bool = False,
                 use_bn_first=True,
                 deploy: bool = False):
        super().__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = MODELS.build(act_cfg)

        if use_se:
            raise NotImplementedError('se block not supported yet')
        else:
            self.se = nn.Identity()

        if use_alpha:
            alpha = torch.ones([
                1,
            ], dtype=torch.float32, requires_grad=True)
            self.alpha = nn.Parameter(alpha, requires_grad=True)
        else:
            self.alpha = None

        if deploy:
            self.rbr_reparam = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=True,
                padding_mode=padding_mode)

        else:
            if use_bn_first and (out_channels == in_channels) and stride == 1:
                self.rbr_identity = build_norm_layer(
                    norm_cfg, num_features=in_channels)[1]
            else:
                self.rbr_identity = None

            self.rbr_dense = ModelConv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
                norm_cfg=norm_cfg,
                act_cfg=None)
            self.rbr_1x1 = ModelConv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=padding_11,
                groups=groups,
                bias=False,
                norm_cfg=norm_cfg,
                act_cfg=None)

    def forward(self, inputs: Tensor) -> Tensor:
        """Forward process.
        Args:
            inputs (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        if self.alpha:
            return self.nonlinearity(
                self.se(
                    self.rbr_dense(inputs) +
                    self.alpha * self.rbr_1x1(inputs) + id_out))
        else:
            return self.nonlinearity(
                self.se(
                    self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))

    def get_equivalent_kernel_bias(self):
        """Derives the equivalent kernel and bias in a differentiable way.

        Returns:
            tuple: Equivalent kernel and bias
        """
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        if self.alpha:
            return kernel3x3 + self.alpha * self._pad_1x1_to_3x3_tensor(
                kernel1x1) + kernelid, bias3x3 + self.alpha * bias1x1 + biasid
        else:
            return kernel3x3 + self._pad_1x1_to_3x3_tensor(
                kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        """Pad 1x1 tensor to 3x3.
        Args:
            kernel1x1 (Tensor): The input 1x1 kernel need to be padded.

        Returns:
            Tensor: 3x3 kernel after padded.
        """
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: nn.Module) -> Tuple[np.ndarray, Tensor]:
        """Derives the equivalent kernel and bias of a specific branch layer.

        Args:
            branch (nn.Module): The layer that needs to be equivalently
                transformed, which can be nn.Sequential or nn.Batchnorm2d

        Returns:
            tuple: Equivalent kernel and bias
        """
        if branch is None:
            return 0, 0
        if isinstance(branch, ConvModule):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, (nn.SyncBatchNorm, nn.BatchNorm2d))
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3),
                                        dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(
                    branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        """Switch to deploy mode."""
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
            in_channels=self.rbr_dense.conv.in_channels,
            out_channels=self.rbr_dense.conv.out_channels,
            kernel_size=self.rbr_dense.conv.kernel_size,
            stride=self.rbr_dense.conv.stride,
            padding=self.rbr_dense.conv.padding,
            dilation=self.rbr_dense.conv.dilation,
            groups=self.rbr_dense.conv.groups,
            bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True


class CED(nn.Module):
    """
    Stride Shuffle for down sample
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
                                    norm_layer='bn_2d', act_layer='none')
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
            self.dw_conv = RepDWBlock(self.hidden_dims, self.hidden_dims, kernel_size, groups=self.hidden_dims)
            self.act = get_act(act_layer)()
            self.pw_linear = ConvModule(self.hidden_dims, dim_out, 1, 1, bias=False, act_layer='none',
                                        norm_layer='bn_2d')
        else:
            self.conv_1x1 = ConvModule(dim_in, self.hidden_dims, 1, 1, groups=1, bias=False,
                                       norm_layer='bn_2d', act_layer=act_layer)
            self.dw_conv = RepDWBlock(self.hidden_dims, self.hidden_dims, kernel_size, groups=self.hidden_dims)
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
