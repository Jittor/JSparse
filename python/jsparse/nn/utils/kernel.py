from typing import Tuple, Union

import numpy as np
import jittor as jt

from python.jsparse.utils import make_ntuple, trunc

__all__ = ['get_kernel_offsets']

def get_kernel_offsets(kernel_size: Union[int, Tuple[int, ...]],
                       stride: Union[int, Tuple[int, ...]] = 1,
                       dilation: Union[int, Tuple[int, ...]] = 1) -> jt.Var:
    kernel_size = make_ntuple(kernel_size, ndim=3)
    stride = make_ntuple(stride, ndim=3)
    dilation = make_ntuple(dilation, ndim=3)

    offsets = [(np.arange(-kernel_size[k] // 2 + 1, kernel_size[k] // 2 + 1) * stride[k]
                * dilation[k]) for k in range(3)]

    if np.prod(kernel_size) % 2 == 1:
        offsets = [[x, y, z] for z in offsets[2] for y in offsets[1]
                   for x in offsets[0]]
    else:
        offsets = [[x, y, z] for x in offsets[0] for y in offsets[1]
                   for z in offsets[2]]

    offsets = jt.int32(offsets)
    return offsets