import jittor as jt
from jittor import Function

from JSparse import SparseTensor

__all__ = ['calc_ti_weights', 'spdevoxelize']

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
        # c = values_shape1
        # N = idx_query_shape0
        output = jt.code((idx_query.shape[0], values.shape[1]), jt.float32, [values, idx_query, weights], 
            cuda_src="""
                __global__ void devoxelize_forward_kernel(@ARGS_DEF) {
                    @PRECALC
                    @alias(values, in0)
                    @alias(idx_query, in1)
                    @alias(weights, in2)
                    int index = blockIdx.x * blockDim.x + threadIdx.x;
                    int i = index / values_shape1;
                    int j = index % values_shape1;

                    if (i < idx_query_shape0) {
                        float cur_values = 0;
                        for (int k = 0; k < 8; ++ k ) {
                            int idx = @idx_query(i, k);
                            cur_values = (idx >= 0) ? @values(idx, j) : 0;
                            @out(i, j) += @weights(i, k) * cur_values;
                        }
                    }
                }
                devoxelize_forward_kernel<<<out_shape0, out_shape1>>>(@ARGS);
            """,
            cpu_src="""
                @alias(values, in0)
                @alias(idx_query, in1)
                @alias(weights, in2)

                #pragma omp parallel for
                for (int i = 0; i < idx_query_shape0; ++ i ) {
                    for (int j = 0; j < values_shape1; ++ j ) {
                        float cur_values = 0;
                        for (int k = 0; k < 8; ++ k ) {
                            int idx = @idx_query(i, k);
                            cur_values = (idx >= 0) ? @values(idx, j) : 0;
                            #pragma omp atomic
                            @out(i ,j) += @weights(i, k) * cur_values;
                        }
                    }
                }
            """
        )
        self.save_vars = (idx_query, weights, values.shape[0])
        return output
    
    def grad(self, grad_output: jt.Var):
        idx_query, weights, input_size = self.save_vars

        grad_values = jt.code((input_size, grad_output.shape[0]), jt.float, [idx_query, weights, grad_output], 
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

                    int index = blockIdx.x * blockDim.x + threadIdx.x;
                    int c = grad_output_shape1;
                    int i = index / c;
                    int j = index % c;
                  
                    if (i < grad_output_shape0) {
                        float cur_grad_output = @grad_output(i, j);

                        #pragma unroll
                        for (int k = 0; k < 8; ++ k ) {
                            int idx = @idx_query(i, k);
                            if (idx >= 0) {
                                atomicAdd(&@out(idx, j), @weights(i, k) * cur_grad_output);
                            }
                        }
                    }
                }
                @alias(grad_output, in2)
                devoxelize_backward_kernel<<<grad_output_shape0, grad_output_shape1>>>(@ARGS);
            """,
            cpu_src="""
                @alias(idx_query, in0)
                @alias(weights, in1)
                @alias(grad_output, in2)

                for (int i = 0; i < grad_output_shape0; ++ i ) {
                    #pragma omp parallel for
                    for (int j = 0; j < grad_output_shape1; ++ j ) {
                        float cur_grad_output = 0;
                        for (int k = 0; k < 8; ++ k ) {
                            int idx = @idx_query(i, k);
                            cur_grad_output = (idx >= 0) ? @grad_output(i, j) : 0;
                            #pragma omp atomic
                            @out(idx, j) += @weights(i, k) * cur_grad_output;
                        }
                    }
                }
            """
        )
        return grad_values, None, None

def spdevoxelize(
    values: jt.Var,
    idx_query: jt.Var,
    weights: jt.Var
) -> jt.Var:
    return Devoxelize.apply(values, idx_query, weights)

