from typing import Tuple, Union

import jittor as jt
from jittor.misc import _pair, _triple

from JSparse.nn.utils import get_kernel_offsets
from JSparse.utils import make_ntuple, trunc

__all__ = ['spdownsample']

def spdownsample(
        indices: jt.Var,
        stride: Union[int, Tuple[int, ...]] = 2,
        kernel_size: Union[int, Tuple[int, ...]] = 2,
        tensor_stride: Union[int, Tuple[int, ...]] = 1) -> jt.Var:
    # stride = make_ntuple(stride, ndim=3)
    # kernel_size = make_ntuple(kernel_size, ndim=3)
    # tensor_stride = make_ntuple(tensor_stride, ndim=3)
    kernel_size = _triple(kernel_size)
    stride = _triple(stride)
    tensor_stride = _triple(tensor_stride)

    sample_stride = [stride[k] * tensor_stride[k] for k in range(3)]
    sample_stride = jt.Var(sample_stride,
                           dtype='int32').unsqueeze(dim=0)

    if all(stride[k] in [1, kernel_size[k]] for k in range(3)):
        indices = indices.clone()
        indices[:, 1:] = trunc(jt.divide(indices[:, 1:], sample_stride)) * sample_stride
    else:
        offsets = get_kernel_offsets(kernel_size,
                                     tensor_stride)
        kernel_volume = offsets.size(0)

        indices_min = indices[:, :3].min(dim=0, keepdims=True)

        b = indices[:, :1].repeat(1, kernel_volume)
        x = indices[:, 1:].unsqueeze(dim=1).repeat(1, kernel_volume, 1) + offsets
        indices = jt.cat([b.view(-1, 1), x.view(-1, 3)], dim=1)

        # TODO: We need to also filter `indices` based on `indices_max`.
        mask = (indices[:, 1:] % sample_stride == 0)
        mask &= (indices[:, 1:] >= indices_min)
        mask = jt.all(mask, dim=1)
        indices = indices[mask]

    # we may have to unique the indices when we define the SparesTensor
    # indices = indices[:, [3, 0, 1, 2]]
    # indices = jt.unique(indices, dim=0)
    # indices = indices[:, [1, 2, 3, 0]]
    return indices