import jittor as jt
from jittor import Function

from JSparse import SparseTensor

__all__ = ['spvoxelize']

class Voxelize(Function):
    def execute(
        self,
        values: jt.Var,
        idx_query: jt.Var,
        counts: jt.Var
    ) -> jt.Var:
        # N = values_shape0
        # c = values_shape1
        # N1 = counts_shape0
        # out: N1 x c
        output = jt.code((counts.shape[0], values.shape[1]), "float32", [values, idx_query, counts],
            cuda_header="""           
                #include <stdio.h>    
                #include <stdlib.h>   
                #include <cuda_runtime.h>  
            """,
            cuda_src="""
                __global__ void voxelize_forward_kernel(@ARGS_DEF) {
                    @PRECALC
                    @alias(values, in0)
                    @alias(idx_query, in1)
                    @alias(counts, in2)
                    
                    int index = blockDim.x * blockIdx.x + threadIdx.x;
                    int c = values_shape1;
                    int i = index / c;
                    int j = index % c;

                    if (i < values_shape0) {
                        int pos = @idx_query(i);
                        if (pos < 0 || pos >= counts_shape0 || @counts(pos) == 0) return;
                        atomicAdd(&@out(pos, j), @values(i, j) / (float)(@counts(pos)));
                    }
                }
                @alias(values, in0)
                voxelize_forward_kernel<<< values_shape0, values_shape1 >>>(@ARGS);
            """,
            cpu_src="""
                @alias(values, in0)
                @alias(idx_query, in1)
                @alias(counts, in2)
                for (int i = 0; i < values_shape0; ++ i ) {
                    int pos = @idx_query(i);
                    if (@counts(pos) == 0)
                        continue;
                    #pragma omp parallel for
                    for (int j = 0; j < values_shape1; ++ j ) {
                        #pragma omp atomic
                        @out(pos, j) += @values(i, j) / (float)@counts(pos);
                    }
                }
            """
        )
        self.save_vars = idx_query, counts, values.shape[0]
        return output
    
    def grad(self, grad_output: jt.Var):
        idx_query, counts, input_size = self.save_vars

        grad_values = jt.code((input_size, grad_output.shape[1]), jt.float32, [idx_query, counts, grad_output],
            cuda_header="""
                #include <stdio.h>    
                #include <stdlib.h>   
                #include <cuda_runtime.h>  
            """,
            cuda_src="""
                __global__ void voxelize_backward_kernel(@ARGS_DEF) {
                    @PRECALC
                    @alias(idx_query, in0)
                    @alias(counts, in1)
                    @alias(grad_output, in2)
                    int index = blockDim.x * blockIdx.x + threadIdx.x;
                    int i = index / grad_output_shape1;
                    int j = index % grad_output_shape1;
                    if (i < out_shape0) {
                        int pos = @idx_query(i);
                        if (pos < 0 || pos >= counts_shape0 || @counts(pos) == 0) return;
                        atomicAdd(&@out(pos, j), @grad_output(pos, j) / @counts(pos));
                    }
                }

                voxelize_backward_kernel<<<out_shape0, out_shape1>>>(@ARGS);
            """,
            cpu_src="""
                @alias(idx_query, in0)
                @alias(counts, in1)
                @alias(grad_output, in2)

                for (int i = 0; i < out_shape0; ++ i ) {
                    int pos = @idx_query(i);
                    if (@counts(pos) == 0) continue;
                    #pragma omp parallel for 
                    for (int j = 0; j < grad_output_shape1; ++ j ) {
                        @out(i, j) = @grad_output(pos, j) / (float)@counts(pos);
                    }
                }
            """
        )
        return grad_values, None, None

def spvoxelize(
    values: jt.Var,
    idx_query: jt.Var,
    counts: jt.Var
) -> jt.Var:
    return Voxelize.apply(values, idx_query, counts)