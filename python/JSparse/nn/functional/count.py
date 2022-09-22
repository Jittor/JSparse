import jittor as jt

def spcount(idx_query: jt.Var, num: int) -> jt.Var:
    return jt.code((num,), jt.int32, [idx_query],
        cuda_src="""
            __global__ void count_kernel(@ARGS_DEF) {
                @PRECALC
                @alias(idx_query, in0)
                int i = blockDim.x * blockIdx.x + threadIdx.x;
                int cur_idx = @idx_query(i);
                if (i < idx_query_shape0 && cur_idx >= 0) {
                    atomicAdd(out_p + cur_idx, 1);
                }
            }
            @alias(idx_query, in0)
            count_kernel<<<(idx_query_shape0 + 511) / 512, 512>>>(@ARGS);
        """,
        cpu_src="""
            @alias(idx_query, in0)
            #pragma omp parallel for
            for (int i = 0; i < idx_query_shape0; ++ i ) {
                int cur_idx = @idx_query(i);
                if (cur_idx < 0) {
                    continue;
                }
                #pragma omp atomic
                @out(cur_idx) ++;
            }
        """
    )
