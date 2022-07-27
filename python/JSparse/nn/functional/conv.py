from typing import List, Optional, Tuple, Union

import jittor as jt
from jittor import Function
from jittor.misc import _pair, _triple

from JSparse import SparseTensor
from JSparse.nn import functional as F
from JSparse.nn.utils import get_kernel_offsets
from JSparse import make_ntuple

__all__ = ['conv3d', 'Convolution']

class Convolution(Function):
    def execute(
        self, 
        input: jt.Var,
        weight: jt.Var,
        nbmaps: jt.Var,
        nbsizes: jt.Var,
        sizes: Tuple[int, int],
        transposed: bool = False,
    ) -> jt.Var:
        if not transposed:
            output = jt.zeros((sizes[1], weight.size(-1)))
        else:
            output = jt.zeros((sizes[0], weight.size(-1)))

        assert input.size(1) == weight.size(1)
        in_size = input.size(0)
        in_channels = input.size(1)
        out_size = output.size(0)
        out_channels = output.size(1)

        kernel_volume = weight.size(0)
        flag = False
        mid = kernel_volume // 2
        in_buffer_size = jt.Var(1)
        if kernel_volume % 2 and out_size == in_size:
            flag = True
            in_buffer_size = max(in_buffer_size, jt.max(nbsizes[:mid]))
            in_buffer_size = max(in_buffer_size, jt.max(nbsizes[mid + 1:]))
            output = jt.matmul(input, weight[mid, :, :])
        else:
            in_buffer_size = jt.max(nbsizes)
        
        # in_buffer_activated : in_buffer_size * in_channels
        # weight              : in_channels    * out_channels
        # out_buffer_activated: in_buffer_size * out_channels
        # out_buffer_activated = in_buffer_activated * weight
        in_buffer_activated = jt.zeros(in_buffer_size.tolist() + [in_channels])
        cur_offset = jt.Var(0)
        for i in range(kernel_volume):
            n_active_feats = nbsizes[i]
            if (flag and (i == mid)):
                cur_offset += n_active_feats
                continue
            if n_active_feats == 0:
                continue

            t = 1 if transposed else 0
            ################
            ## gather
            ################
            # print(n_active_feats, in_channels, cur_offset, t)
            gather(n_active_feats, in_channels, cur_offset, t, input, in_buffer_activated, nbmaps)
            ################
            ## matmul
            ################
            out_buffer_activated = jt.matmul(in_buffer_activated, weight[i, :, :])
            ################
            ## scatter
            ################
            scatter(n_active_feats, out_channels, cur_offset, t, out_buffer_activated, output, nbmaps)
            cur_offset += n_active_feats
        self.save_vars = input, weight, nbmaps, nbsizes, transposed
        return output
    
    def grad(
        self, 
        grad_output: jt.Var
    ) -> Tuple[Optional[jt.Var], ...]:
        input, weight, nbmaps, nbsizes, transposed = self.save_vars

        grad_input = jt.zeros_like(input)
        grad_weight = jt.zeros_like(weight)

        # n_in_feats = input.size(0)
        # n_out_feats = grad_output.size(0)
        n_in_channels = input.size(1)
        n_out_channels = weight.size(-1)

        kernel_volume = weight.size(0)
        flag = False
        in_buffer_size = jt.max(nbsizes)
        # out_grad_buffer_activated     n_active_feats x n_out_channels
        # in_grad_buffer_activated      n_active_feats x n_in_channels
        # in_buffer_activated           n_active_feats x n_in_channels
        out_grad_buffer_activated = jt.zeros(in_buffer_size.tolist() + [n_out_channels])
        in_grad_buffer_activated = jt.zeros(in_buffer_size.tolist() + [n_in_channels])
        in_buffer_activated = jt.zeros(in_buffer_size.tolist() + [n_in_channels])

        cur_offset = jt.Var(0)
        mid = kernel_volume // 2
        for i in range(kernel_volume):
            # kernel_grad_buffer = grad_weight[i, :, :]
            n_active_feats = nbsizes[i]
            # if flag and (i == mid):
            #     cur_offset += n_active_feats
            #     continue

            if n_active_feats == 0:
                continue

            t = 1 if transposed else 0
            ################
            ## gather
            ################
            gather(n_active_feats, n_out_channels, cur_offset, 1 - t, grad_output, out_grad_buffer_activated, nbmaps)
            gather(n_active_feats, n_in_channels , cur_offset, t    , input      , in_buffer_activated      , nbmaps)
            ################
            ## matmul
            ################
            # grad for input
            #    in_grad_buffer_activated     =     out_grad_buffer_activated     @            weight^T
            # n_active_feats x n_in_channels     n_active_feats x n_out_channels     n_out_channels x n_in_channels
            in_grad_buffer_activated = jt.nn.matmul_transpose(out_grad_buffer_activated, weight[i, :, :])
            # grad for weight
            #       kernel_grad_buffer        =     in_buffer_activated^T         @    out_grad_buffer_activated
            # n_in_channels x n_out_channels     n_in_channels x n_active_feats      n_active_feats x n_out_channels 
            grad_weight[i, :, :] = jt.nn.matmul(in_buffer_activated.t(), out_grad_buffer_activated)
            ################
            ## scatter
            ################
            scatter(n_active_feats, n_in_channels, cur_offset, 1 - t, in_grad_buffer_activated, grad_input, nbmaps)
            cur_offset += n_active_feats  
        return grad_input, grad_weight, None, None, None, None

def gather(
    n_active_feats,
    channels,
    cur_offset,
    transpose,
    in_feat,
    out_feat,
    kmap,
):
    shape = n_active_feats.tolist() + cur_offset.tolist() + [channels, transpose, 0]
    gather_args = jt.zeros(shape, dtype='int32')
    return jt.code((0, ), out_feat.dtype, [in_feat, out_feat, kmap, gather_args],
        cuda_header="""
            @alias(in_feat, in0)
            @alias(out_feat, in1)
            @alias(kmap, in2)
            @alias(args, in3)
        """,
        cuda_src="""
            __global__ void gather_kernel(@ARGS_DEF) {
                @PRECALC
                const int n_k = args_shape0;
                const int st = args_shape1;
                const int c = args_shape2;
                const int transpose = args_shape3;

                int idx = blockDim.x * blockIdx.x + threadIdx.x;
                int i = idx / c;
                int j = idx % c;
                if (i >= n_k) return;
                int in_pos = @kmap(st + i, transpose);
                // if (in_pos < 0) return;
                @out_feat(i, j) = @in_feat(in_pos, j);
            }

            gather_kernel<<< (out_feat_shape0 * out_feat_shape1 + 255) / 256, 256 >>>(@ARGS);
        """,
        cpu_header="""
            @alias(in_feat, in0)
            @alias(out_feat, in1)
            @alias(kmap, in2)
            @alias(args, in3)
        """,
        cpu_src="""
            const int n_k = args_shape0;
            const int st = args_shape1;
            const int c = args_shape2;
            const int transpose = args_shape3;

            for (int i = 0; i < n_k; ++ i ) {
                int in_pos = @kmap(st + i, transpose);
                // if (in_pos < 0) {
                //     continue;
                // }
                #pragma omp parallel for 
                for (int j = 0; j < c; ++ j ) {
                    @out_feat(i, j) = @in_feat(in_pos, j);
                    // @out(i, j) = @in_feat(in_pos, j);
                }
            }
        """
    ).sync()
    

def scatter(
    n_active_feats,
    channels,
    cur_offset,
    transpose,
    in_feat,
    out_feat,
    kmap,
):  
    shape = n_active_feats.tolist() + cur_offset.tolist() + [channels, transpose, 0]
    scatter_args = jt.zeros(shape, dtype='int32')
    return jt.code((0, ), out_feat.dtype, [in_feat, out_feat, kmap, scatter_args], 
        cuda_header="""
            @alias(in_feat, in0)
            @alias(out_feat, in1)
            @alias(kmap, in2)
            @alias(args, in3)
        """,
        cuda_src="""
            __global__ void scatter_kernel(@ARGS_DEF) {
                @PRECALC
                const int n_k = args_shape0;
                const int st = args_shape1;
                const int c = args_shape2;
                const int transpose = args_shape3;

                int idx = blockDim.x * blockIdx.x + threadIdx.x;
                int i = idx / c;
                int j = idx % c;
                if (i >= n_k) return;
                int out_pos = @kmap(st + i, 1 - transpose);
                // if (out_pos < 0) return;
                @out_feat(out_pos, j) += @in_feat(i, j);
            }

            scatter_kernel<<< (out_feat_shape0 * out_feat_shape1 + 255) / 256, 256 >>>(@ARGS);
        """,
        cpu_header="""
            @alias(in_feat, in0)
            @alias(out_feat, in1)
            @alias(kmap, in2)
            @alias(args, in3)
        """,
        cpu_src="""
            const int n_k = args_shape0;
            const int st = args_shape1;
            const int c = args_shape2;
            const int transpose = args_shape3;

            for (int i = 0; i < n_k; ++ i ) {
                int out_pos = @kmap(st + i, 1 - transpose);
                // if (out_pos < 0) {
                //     continue;
                // }
                #pragma omp parallel for
                for (int j = 0; j < c; ++ j ) {
                    @out_feat(out_pos, j) += @in_feat(i, j);
                    // @out(out_pos, j) += @in_feat(i, j);
                }
            }
        """
    ).sync()

def conv3d(
    input: SparseTensor,
    weight: jt.Var,
    kernel_size: Union[int, Tuple[int, ...]],
    bias: Optional[jt.Var] = None,
    stride: Union[int, Tuple[int, ...]] = 1,
    dilation: Union[int, Tuple[int, ...]] = 1,
    group: int = 1,
    transposed: bool = False,
) -> SparseTensor:
    # kernel_size = make_ntuple(kernel_size, ndim=3)
    # stride = make_ntuple(stride, ndim=3)
    # dilation = make_ntuple(dilation, ndim=3)
    kernel_size = _triple(kernel_size)
    stride = _triple(stride)
    dilation = _triple(dilation)

    if (kernel_size == _triple(1) and stride == _triple(1) and dilation == _triple(1)):
        output_stride = input.stride
        output_indices = input.indices
        output_values = input.values.matmul(weight)
    elif not transposed:
        output_stride = tuple(input.stride[k] * stride[k] for k in range(3))

        if output_stride in input.cmaps:
            output_indices = input.cmaps[output_stride]
        elif all(stride[k] == 1 for k in range(3)):
            output_indices = input.indices
        else:
            output_indices = F.spdownsample(
                input.indices,
                stride,
                kernel_size,
                input.stride,
            )
        
        if (input.stride, kernel_size, stride, dilation) not in input.kmaps:
            offsets = get_kernel_offsets(
                kernel_size,
                stride=input.stride,
                dilation=dilation,
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

        output_values = Convolution.apply(
            input.values,
            weight,
            *input.kmaps[(input.stride, kernel_size, stride, dilation)],
            transposed,
        )
    else:
        output_stride = tuple(input.stride[k] // stride[k] for k in range(3))
        output_indices = input.cmaps[output_stride]
        output_values = Convolution.apply(
            input.values,
            weight,
            *input.kmaps[(output_stride, kernel_size, stride, dilation)],
            transposed,
        )
    
    if bias is not None:
        output_values += bias
    
    # size have to be set
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
