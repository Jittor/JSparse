import jittor as jt
from jittor import Function

from JSparse import SparseTensor, PointTensor
from JSparse.nn import functional as F
from JSparse.nn.utils import get_kernel_offsets


__all__ = ['calc_ti_weights', 'spdevoxelize', 'voxel_to_point']

def calc_ti_weights(
    indices: jt.Var,
    idx_query: jt.Var,
    scale: float = 1
) -> jt.Var:
    with jt.no_grad():
        p = indices
        if scale != 1:
            pf = jt.floor(indices / scale) * scale
        else:
            pf = jt.floor(indices)
        pc = pf + scale

        x = p[:, 1].view(-1, 1)
        y = p[:, 2].view(-1, 1)
        z = p[:, 3].view(-1, 1)

        xf = pf[:, 1].view(-1, 1).float()
        yf = pf[:, 2].view(-1, 1).float()
        zf = pf[:, 3].view(-1, 1).float()

        xc = pc[:, 1].view(-1, 1).float()
        yc = pc[:, 2].view(-1, 1).float()
        zc = pc[:, 3].view(-1, 1).float()

        w0 = (xc - x) * (yc - y) * (zc - z)
        w1 = (xc - x) * (yc - y) * (z - zf)
        w2 = (xc - x) * (y - yf) * (zc - z)
        w3 = (xc - x) * (y - yf) * (z - zf)
        w4 = (x - xf) * (yc - y) * (zc - z)
        w5 = (x - xf) * (yc - y) * (z - zf)
        w6 = (x - xf) * (y - yf) * (zc - z)
        w7 = (x - xf) * (y - yf) * (z - zf)

        w = jt.concat([w0, w1, w2, w3, w4, w5, w6, w7], dim=1).t()
        if scale != 1:
            w /= scale ** 3
        w[idx_query == -1] = 0
        w /= jt.sum(w, dim=0) + 1e-8
    return w
        

class Devoxelize(Function):
    def execute(
        self,
        values: jt.Var,
        idx_query: jt.Var,
        weights: jt.Var
    ) -> jt.Var:
        output = jt.zeros((idx_query.shape[0], values.shape[1]), dtype=values.dtype)
        jt.code((0, ), values.dtype, [values, idx_query, weights, output], 
            cuda_src="""
                __global__ void devoxelize_forward_kernel(@ARGS_DEF) {
                    @PRECALC
                    @alias(values, in0)
                    @alias(idx_query, in1)
                    @alias(weights, in2)
                    @alias(output, in3)

                    int index = blockIdx.x * blockDim.x + threadIdx.x;
                    int i = index / values_shape1;
                    int j = index % values_shape1;

                    if (i < idx_query_shape0) {
                        values_type cur_values = 0;
                        #pragma unroll
                        for (int k = 0; k < 8; ++ k ) {
                            int idx = @idx_query(i, k);
                            cur_values = (idx >= 0) ? @values(idx, j) : (values_type)(0.0f);
                            @output(i, j) += @weights(i, k) * cur_values;
                        }
                    }
                }
                @alias(output, in3)
                devoxelize_forward_kernel<<<output_shape0, output_shape1>>>(@ARGS);
            """,
            cpu_header="""
                #include <iostream>
                #include <omp.h>
                using namespace std;
            """,
            cpu_src="""
                @alias(values, in0)
                @alias(idx_query, in1)
                @alias(weights, in2)
                @alias(output, in3)

                #pragma omp parallel for
                for (int i = 0; i < idx_query_shape0; ++ i ) {
                    #pragma omp parallel for
                    for (int j = 0; j < values_shape1; ++ j ) {
                        #pragma unroll 8
                        for (int k = 0; k < 8; ++ k ) {
                            int idx = @idx_query(i, k);
                            values_type cur_values = (idx >= 0) ? @values(idx, j) : (values_type)(0.0f);
                            #pragma omp atomic
                            @output(i ,j) += @weights(i, k) * cur_values;
                        }
                    }
                }
            """
        ).sync()
        self.save_vars = (idx_query, weights, values.shape[0])
        return output
    
    def grad(self, grad_output: jt.Var):
        idx_query, weights, input_size = self.save_vars
        grad_values = jt.zeros((input_size, grad_output.shape[1]), dtype=weights.dtype)
        jt.code((0, ), weights.dtype, [idx_query, weights, grad_output, grad_values], 
            cuda_header="""           
                #include <stdio.h>    
                #include <stdlib.h>   
                #include <cuda_runtime.h>  
            """,
            cuda_src="""
                __global__ void devoxelize_backward_kernel(@ARGS_DEF) {
                    @PRECALC
                    @alias(idx_query, in0)
                    @alias(weights, in1)
                    @alias(grad_output, in2)
                    @alias(grad_values, in3)

                    int index = blockIdx.x * blockDim.x + threadIdx.x;
                    int c = grad_output_shape1;
                    int i = index / c;
                    int j = index % c;
                  
                    if (i < grad_output_shape0) {
                        weights_type cur_grad_output = @grad_output(i, j);

                        #pragma unroll
                        for (int k = 0; k < 8; ++ k ) {
                            int idx = @idx_query(i, k);
                            if (idx >= 0) {
                                atomicAdd(&@grad_values(idx, j), @weights(i, k) * cur_grad_output);
                            }
                        }
                    }
                }
                @alias(grad_output, in2)
                devoxelize_backward_kernel<<<grad_output_shape0, grad_output_shape1>>>(@ARGS);
            """,
            cpu_header="""
                #include <omp.h>
            """,
            cpu_src="""
                @alias(idx_query, in0)
                @alias(weights, in1)
                @alias(grad_output, in2)
                @alias(grad_values, in3)

                #pragma omp parallel for
                for (int i = 0; i < grad_output_shape0; ++ i ) {
                    #pragma omp parallel for
                    for (int j = 0; j < grad_output_shape1; ++ j ) {
                        weight_type cur_grad_output = 0;
                        #pragma unroll 8
                        for (int k = 0; k < 8; ++ k ) {
                            int idx = @idx_query(i, k);
                            cur_grad_output = (idx >= 0) ? @grad_output(i, j) : 0;
                            #pragma omp atomic
                            @grad_values(idx, j) += (float)(@weights(i, k)) * (float)(cur_grad_output);
                        }
                    }
                }
            """
        ).sync()
        return grad_values, None, None

def spdevoxelize(
    values: jt.Var,
    idx_query: jt.Var,
    weights: jt.Var
) -> jt.Var:
    return Devoxelize.apply(values, idx_query, weights)

def voxel_to_point(x: SparseTensor, z: PointTensor, nearest: bool=False) -> PointTensor:
    
    if z.idx_query is None or z.weights is None or z.idx_query.get(
        x.stride) is None or z.weights.get(x.stride) is None:
        offsets = get_kernel_offsets(kernel_size=2, stride=x.stride, dilation=1)
        cube_hash = F.sphash(
            jt.concat([
                z.indices[:, 0].int().view(-1, 1),
                jt.floor(z.indices[:, 1:] / x.stride[0]).int() * x.stride[0]
            ], 1), offsets)
        indices_hash = F.sphash(x.indices)
        idx_query = F.spquery(cube_hash, indices_hash).int()
        weights = F.calc_ti_weights(z.indices, idx_query,
                            scale=x.stride[0]).t()
        idx_query = idx_query.t()
        if nearest:
            weights[:, 1:] = 0.
            idx_query[:, 1:] = -1
        new_values = F.spdevoxelize(x.values, idx_query, weights)
        new_tensor = PointTensor(new_values,
                                 z.indices,
                                 idx_query=z.idx_query,
                                 weights=z.weights)
        new_tensor.additional_values = z.additional_values
        new_tensor.idx_query[x.stride] = idx_query
        new_tensor.weights[x.stride] = weights
        z.idx_query[x.stride] = idx_query
        z.weights[x.stride] = weights
    else:
        new_values = F.spdevoxelize(x.values, z.idx_query.get(x.stride), z.weights.get(x.stride))
        new_tensor = PointTensor(new_values,
                                 z.indices,
                                 idx_query=z.idx_query,
                                 weights=z.weights)
        new_tensor.additional_values = z.additional_values

    return new_tensor
