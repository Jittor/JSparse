import jittor as jt
from jittor import Function

from JSparse import SparseTensor, PointTensor
from JSparse.nn import functional as F

__all__ = ['spvoxelize', 'point_to_voxel']

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
        output = jt.zeros((counts.shape[0], values.shape[1]), dtype='float32')
        jt.code((0, ), 'float32', [values, idx_query, counts, output],
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
                    @alias(output, in3)
                    
                    int index = blockDim.x * blockIdx.x + threadIdx.x;
                    int c = values_shape1;
                    int i = index / c;
                    int j = index % c;

                    if (i < values_shape0) {
                        int pos = @idx_query(i);
                        if (pos < 0 || pos >= counts_shape0 || @counts(pos) == 0) return;
                        atomicAdd(&@output(pos, j), @values(i, j) / (float)(@counts(pos)));
                    }
                }
                @alias(values, in0)
                voxelize_forward_kernel<<< values_shape0, values_shape1 >>>(@ARGS);
            """,
            cpu_src="""
                @alias(values, in0)
                @alias(idx_query, in1)
                @alias(counts, in2)
                @alias(output, in3)

                #pragma omp parallel for
                for (int i = 0; i < values_shape0; ++ i ) {
                    int pos = @idx_query(i);
                    if (@counts(pos) == 0)
                        continue;
                    #pragma omp parallel for
                    for (int j = 0; j < values_shape1; ++ j ) {
                        #pragma omp atomic
                        @output(pos, j) += @values(i, j) / (float)@counts(pos);
                    }
                }
            """
        ).sync()
        self.save_vars = idx_query, counts, values.shape[0]
        return output
    
    def grad(self, grad_output: jt.Var):
        idx_query, counts, input_size = self.save_vars
        grad_values = jt.zeros((input_size, grad_output.shape[1]), dtype='float32')
        jt.code((0, ), 'float32', [idx_query, counts, grad_output, grad_values],
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
                    @alias(grad_values, in3)

                    int index = blockDim.x * blockIdx.x + threadIdx.x;
                    int i = index / grad_output_shape1;
                    int j = index % grad_output_shape1;
                    if (i < grad_values_shape0) {
                        int pos = @idx_query(i);
                        if (pos < 0 || pos >= counts_shape0 || @counts(pos) == 0) return;
                        @grad_values(pos, j) = @grad_output(pos, j) /(float)@counts(pos);
                    }
                }

                voxelize_backward_kernel<<< grad_values_shape0, grad_values_shape1 >>>(@ARGS);
            """,
            cpu_src="""
                @alias(idx_query, in0)
                @alias(counts, in1)
                @alias(grad_output, in2)
                @alias(grad_values, in3)

                #pragma omp parallel for 
                for (int i = 0; i < grad_values_shape0; ++ i ) {
                    int pos = @idx_query(i);
                    if (@counts(pos) == 0) continue;
                    #pragma omp parallel for 
                    for (int j = 0; j < grad_output_shape1; ++ j ) {
                        @grad_values(i, j) = @grad_output(pos, j) / (float)@counts(pos);
                    }
                }
            """
        ).sync()
        return grad_values, None, None

def spvoxelize(
    values: jt.Var,
    idx_query: jt.Var,
    counts: jt.Var
) -> jt.Var:
    return Voxelize.apply(values, idx_query, counts)

def point_to_voxel(x: SparseTensor, z: PointTensor) -> SparseTensor:
    if z.additional_values is None or z.additional_values.get(
        'idx_query') is None or z.additional_values['idx_query'].get(
            x.stride) is None:
        point_hash = F.sphash(
        jt.concat([
            z.indices[:, 0].int().view(-1, 1),
            jt.floor(z.indices[:, 1:] / x.stride[0]).int() * x.stride[0]
        ], 1))
        sparse_hash = F.sphash(x.indices)
        idx_query = F.spquery(point_hash, sparse_hash).int()
        counts = F.spcount(idx_query, x.indices.shape[0])
        z.additional_values['idx_query'][x.stride] = idx_query
        z.additional_values['counts'][x.stride] = counts
    else:
        idx_query = z.additional_values['idx_query'][x.stride]
        counts = z.additional_values['counts'][x.stride]
    
    voxelized_values = F.spvoxelize(z.values, idx_query, counts)
    new_tensor = SparseTensor(voxelized_values, x.indices, x.stride, x.size, False)
    new_tensor.cmaps = x.cmaps
    new_tensor.kmaps = x.kmaps

    return new_tensor

