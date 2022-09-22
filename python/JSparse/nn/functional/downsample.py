from typing import Tuple, Union
import numpy as np

import jittor as jt
from jittor.misc import _pair, _triple

from JSparse.nn.utils import get_kernel_offsets
from JSparse.utils import make_ntuple, trunc, unique1d
from JSparse.nn import functional as F

__all__ = ['spdownsample']


def spdownsample(
        indices: jt.Var,
        stride: Union[int, Tuple[int, ...]] = 2,
        kernel_size: Union[int, Tuple[int, ...]] = 2,
        tensor_stride: Union[int, Tuple[int, ...]] = 1) -> jt.Var:
    kernel_size = _triple(kernel_size)
    stride = _triple(stride)
    tensor_stride = _triple(tensor_stride)

    sample_stride = [stride[k] * tensor_stride[k] for k in range(3)]
    sample_stride = jt.array(sample_stride, dtype='int32').unsqueeze(dim=0)

    if all(stride[k] in [1, kernel_size[k]] for k in range(3)):
        indices = indices.clone()
        indices[:, 1:] = trunc(jt.divide(indices[:, 1:], sample_stride.float())) * sample_stride
    else:
        offsets = get_kernel_offsets(kernel_size, tensor_stride)
        kernel_volume = offsets.size(0)

        indices_min = indices[:, 1:].min(dim=0, keepdims=True)

        b = indices[:, :1].repeat(1, kernel_volume)
        x = indices[:, 1:].unsqueeze(dim=1).repeat(1, kernel_volume, 1) + offsets
        indices = jt.concat([b.view(-1, 1), x.view(-1, 3)], dim=1)

        # TODO: We need to also filter `indices` based on `indices_max`.
        mask = (indices[:, 1:] % sample_stride == 0)
        mask &= (indices[:, 1:] >= indices_min)
        mask = jt.all(mask, dim=1)
        indices = indices[mask]

    indices = jt.unique(indices, dim=0)
    return indices