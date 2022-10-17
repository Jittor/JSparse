import jittor as jt
from typing import Union, Optional, Tuple
from jittor.misc import _pair, _triple
from jittor import nn 
from jsparse import SparseTensor
from jsparse.nn import functional as F
from jsparse.nn.utils import get_kernel_offsets

__all__ = ['max_pool']

def apply_pool(
    input: jt.Var,
    nbmaps: jt.Var,
    nbsizes: jt.Var,
    sizes: Tuple[int, int],
    transposed: bool = False,
    method: str = 'max'
) -> jt.Var:
    if not transposed:
        output = jt.zeros((sizes[1], input.size(-1)), dtype=input.dtype)
    else:
        output = jt.zeros((sizes[0], input.size(-1)), dtype=input.dtype)

    kernel_volume = nbsizes.size(0)
    in_channels = input.size(1)
    out_size = output.size(0)
    cur_offset = 0
    for i in range(kernel_volume):
        n_active_feats = int(nbsizes[i])
        t = 1 if transposed else 0
        
        in_buffer_activated = input.reindex([n_active_feats, in_channels], ['@e0(i0)', 'i1'], 
                                             extras=[nbmaps[cur_offset:cur_offset + n_active_feats, t]])

        output = jt.maximum(output, in_buffer_activated.reindex_reduce(method, [out_size, in_channels], ['@e0(i0)', 'i1'], 
                                                                       extras=[nbmaps[cur_offset:cur_offset + n_active_feats, 1-t]]))

        #output = output.scatter_(0, nbmaps[cur_offset:cur_offset + n_active_feats, 1 - t], 
        #                         in_buffer_activated, reduce=method)
        
        cur_offset += n_active_feats
    return output

def max_pool(
    input: SparseTensor,
    kernel_size: Union[int, Tuple[int, ...]] = 1,
    stride: Union[int, Tuple[int, ...]] = 1,
    dilation: Union[int, Tuple[int, ...]] = 1,
    transposed: bool = False
) -> SparseTensor:
    kernel_size = _triple(kernel_size)
    stride = _triple(stride)
    dilation = _triple(dilation)

    if (kernel_size == _triple(1) and stride == _triple(1) and dilation == _triple(1)):
        return input
    elif not transposed:
        output_stride = tuple(input.stride[k] * stride[k] for k in range(3))

        if output_stride in input.cmaps:
            output_indices = input.cmaps[output_stride]
        elif all(stride[k] == 1 for k in range(3)):
            output_indices = input.indices
        else:
            output_indices = F.spdownsample(
                input.indices, stride, kernel_size, input.stride,
            )

        if (input.stride, kernel_size, stride, dilation) not in input.kmaps:
            offsets = get_kernel_offsets(
                kernel_size,
                stride=input.stride,
                dilation=dilation
            )
            references = F.sphash(input.indices) # (N,)
            queries = F.sphash(output_indices, offsets) # (|K|, N)
            results = F.spquery(queries, references) # (|K|, N)

            nbsizes = jt.sum(results != -1, dim=1)
            nbmaps = jt.nonzero(results != -1)

            indices = nbmaps[:, 0] * results.size(1) + nbmaps[:, 1]
            nbmaps[:, 0] = results.view(-1)[indices]

            input.kmaps[(input.stride, kernel_size, stride, dilation)] = [
                nbmaps, nbsizes, (input.indices.shape[0], output_indices.shape[0])
            ]

        output_values = apply_pool(
            input.values,
            *input.kmaps[(input.stride, kernel_size, stride, dilation)],
            transposed,
        )
    else:
        output_stride = tuple(input.stride[k] // stride[k] for k in range(3))
        output_indices = input.cmaps[output_stride]
        output_values = apply_pool(
            input.values,
            *input.kmaps[(output_stride, kernel_size, stride, dilation)],
            transposed,
        )

    output = SparseTensor(
        indices=output_indices,
        values=output_values,
        stride=output_stride,
        size=input.size
    )
    output.cmaps = input.cmaps
    output.cmaps.setdefault(output_stride, output_indices)
    output.kmaps = input.kmaps
    return output


