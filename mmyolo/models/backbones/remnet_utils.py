from functools import partial

from timm.layers import DropPath, LayerNorm2d
from torch import nn
import torch
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