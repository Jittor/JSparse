import math
from typing import List, Tuple, Union

import numpy as np
import jittor as jt
from jittor import nn
from jittor import init
from jittor.misc import _pair, _triple

from JSparse import SparseTensor
from JSparse.nn import functional as F

__all__ = ['Conv3d']

class Conv3d(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, ...]] = 3,
                 stride: Union[int, Tuple[int, ...]] = 1,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = False,
                 transposed: bool = False) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation, dilation)
        self.groups = groups
        assert in_channels % groups == 0, 'in_channels must be divisible by groups'
        assert out_channels % groups == 0, 'out_channels must be divisible by groups'
        self.transposed = transposed

        self.kernel_volume = int(np.prod(self.kernel_size))
        fan = (self.out_channels if self.transposed else self.in_channels) * self.kernel_volume
        std = 1 / math.sqrt(fan)
            
        if self.kernel_volume > 1:
            self.weight = init.uniform([self.kernel_volume, in_channels, out_channels], 'float32', -std, std)
        else:
            self.weight = init.uniform([in_channels, out_channels], 'float32', -std, std)
        if bias:
            self.bias = init.uniform([out_channels], "float32", -std, std)
        else:
            self.bias = None
        # self.reset_parameters()

    def execute(self, input: SparseTensor) -> SparseTensor:
        return F.conv3d(input,
                        weight=self.weight,
                        kernel_size=self.kernel_size,
                        bias=self.bias,
                        stride=self.stride,
                        dilation=self.dilation,
                        groups=self.groups,
                        transposed=self.transposed)
            