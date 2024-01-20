from typing import Union, List, Tuple

from mmdet.utils import ConfigType, OptMultiConfig
from timm.layers import SqueezeExcite as SE

from mmyolo.models.utils import make_divisible
from mmyolo.registry import MODELS
from .base_backbone import BaseBackbone
from .remnet_utils import *


class InvertedResidualV3(nn.Module):
    def __init__(self, kernel_size, in_size, expand_size, out_size, act, se, stride):
        super(InvertedResidualV3, self).__init__()
        self.stride = stride

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.act1 = act(inplace=True)

        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride,
                               padding=kernel_size // 2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.act2 = act(inplace=True)
        self.se = SE(expand_size) if se else nn.Identity()

        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)
        self.act3 = act(inplace=True)

        self.skip = None
        if stride == 1 and in_size != out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_size)
            )

        if stride == 2 and in_size != out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels=in_size, out_channels=in_size, kernel_size=3, groups=in_size, stride=2, padding=1,
                          bias=False),
                nn.BatchNorm2d(in_size),
                nn.Conv2d(in_size, out_size, kernel_size=1, bias=True),
                nn.BatchNorm2d(out_size)
            )

        if stride == 2 and in_size == out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels=in_size, out_channels=out_size, kernel_size=3, groups=in_size, stride=2,
                          padding=1, bias=False),
                nn.BatchNorm2d(out_size)
            )

    def forward(self, x):
        skip = x

        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.se(out)
        out = self.bn3(self.conv3(out))

        if self.skip is not None:
            skip = self.skip(skip)
        return self.act3(out + skip)


@MODELS.register_module()
class MobileNetV3(BaseBackbone):
    act = nn.Hardswish
    arch_settings = {
        # kernel_size, in_size, expand_size, out_size, act, se, stride
        'P5': [
            [[3, 16, 16, 16, nn.ReLU, False, 1]],  # 1
            [[3, 16, 64, 24, nn.ReLU, False, 2], [3, 24, 72, 24, nn.ReLU, False, 1]],  # 2
            [[5, 24, 72, 40, nn.ReLU, True, 2], [5, 40, 120, 40, nn.ReLU, True, 1], [5, 40, 120, 40, nn.ReLU, True, 1]],
            # 3
            [[3, 40, 240, 80, act, False, 2],
             [3, 80, 200, 80, act, False, 1],
             [3, 80, 184, 80, act, False, 1],
             [3, 80, 184, 80, act, False, 1],
             [3, 80, 480, 112, act, True, 1],
             [3, 112, 672, 112, act, True, 1],
             ],  # 4
            [[5, 112, 672, 160, act, True, 2], [5, 160, 672, 160, act, True, 1], [5, 160, 960, 160, act, True, 1]],  # 5
        ]
    }
    """
        kernel_size, in_size, expand_size, out_size, act, se, stride
            Block(3, 16, 16, 16, nn.ReLU, False, 1),
            Block(3, 16, 64, 24, nn.ReLU, False, 2),
            Block(3, 24, 72, 24, nn.ReLU, False, 1),
            Block(5, 24, 72, 40, nn.ReLU, True, 2),
            Block(5, 40, 120, 40, nn.ReLU, True, 1),
            Block(5, 40, 120, 40, nn.ReLU, True, 1),
            Block(3, 40, 240, 80, act, False, 2),
            Block(3, 80, 200, 80, act, False, 1),
            Block(3, 80, 184, 80, act, False, 1),
            Block(3, 80, 184, 80, act, False, 1),
            Block(3, 80, 480, 112, act, True, 1),
            Block(3, 112, 672, 112, act, True, 1),
            Block(5, 112, 672, 160, act, True, 2),
            Block(5, 160, 672, 160, act, True, 1),
            Block(5, 160, 960, 160, act, True, 1),
    """

    def __init__(self,
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
        # self.arch_settings[arch][-1][-1][3] = last_stage_out_channels
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
        oup = make_divisible(self.arch_setting[0][0][3], self.widen_factor)
        return ConvModule(self.input_channels,
                          oup,
                          kernel_size=3,
                          stride=2,
                          bias=True,
                          norm_layer='bn_2d',
                          act_layer='relu')

    def build_stage_layer(self, stage_idx: int, setting: list) -> list:
        """Build a stage layer.

        Args:
            stage_idx (int): The index of a stage layer.
            setting (list): The architecture setting of a stage layer.
        """
        stage = []
        csp_layer = self.make_layer(setting)
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

    def make_layer(self, setting):
        layers = []
        for kernel_size, in_size, expand_size, out_size, act, se, stride in setting:
            layers.append(
                InvertedResidualV3(
                    kernel_size, in_size, expand_size, out_size, act, se, stride
                )
            )

        return nn.Sequential(*layers)
