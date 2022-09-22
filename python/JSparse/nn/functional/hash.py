from typing import Optional

import jittor as jt

__all__ = ['sphash']

def sphash(indices: jt.Var,
           offsets: Optional[jt.Var] = None) -> jt.Var:
    assert indices.ndim == 2 and indices.shape[1] == 4, indices.shape

    if offsets is None:
        return jt.code((indices.shape[0],), jt.int64, [indices], 
            cuda_header="""
                #include <stdio.h>
                #include <stdlib.h>
                #include <cmath>
                #include <vector>
            """,
            cuda_src="""
                __global__ static void hash_kernel(@ARGS_DEF) {
                    @PRECALC
                    @alias(indices, in0)
                    
                    int i = blockIdx.x * blockDim.x + threadIdx.x;
                    if (i < indices_shape0) {
                        indices_p += i * 4;
                        uint64_t hash = 14695981039346656037UL;
                        
                        //for (int j = 0; j < 4; ++ j ) {
                        //    //hash ^= (unsigned int)@indices(i, j);
                        //    //hash ^= (unsigned int)indices_p[j];
                        //    hash ^= (uint64_t)indices_p[j];
                        //    hash *= 1099511628211UL;
                        //}

                        for (int j = 1; j < 4; ++ j ) {
                            //hash ^= (unsigned int)@indices(i, j);
                            //hash ^= (unsigned int)indices_p[j];
                            hash ^= (uint64_t)indices_p[j];
                            hash *= 1099511628211UL;
                        }
                        hash ^= (uint64_t)indices_p[0];
                        hash *= 1099511628211UL;

                        hash = (hash >> 60) ^ (hash & 0xFFFFFFFFFFFFFFF);
                        @out(i) = hash;
                    }
                }
                @alias(indices, in0)
                hash_kernel<<< (indices_shape0 + 511) / 512, 512 >>>(@ARGS);
            """,
            cpu_header="""
                #include <vector>
                @alias(indices, in0)
            """,
            cpu_src="""
                #pragma omp parallel for
                for (int i = 0; i < indices_shape0; ++ i ) {
                    uint64_t hash = 14695981039346656037UL;
                    for (int j = 0; j < 4; ++ j ) {
                        //hash ^= (unsigned int)@indices(i, j);
                        hash ^= (uint64_t)@indices(i, j);
                        hash *= 1099511628211UL;
                    }
                    hash = (hash >> 60) ^ (hash & 0xFFFFFFFFFFFFFFF);
                    @out(i) = hash;
                }
            """)
    else:
        assert offsets.dtype == jt.int32, offsets.dtype
        assert offsets.ndim == 2 and offsets.shape[1] == 3, offsets.shape

        return jt.code((offsets.shape[0], indices.shape[0]), jt.int64, [indices, offsets],
            cuda_header="""
                #include <stdio.h>
                #include <stdlib.h>
                #include <iostream>
                #include <cmath>
                #include <vector>
            """,
            cuda_src="""
                __global__ void kernel_hash_kernel(@ARGS_DEF){
                    @PRECALC
                    @alias(indices, in0)
                    @alias(offsets, in1)
                    //extern __shared__ int offsets_shared[];

                    int idx = blockDim.x * blockIdx.x + threadIdx.x;
                    int k = idx % offsets_shape0;
                    int i = idx / offsets_shape0;

                    if (i < indices_shape0) {
                        int cur_indices[4];

                        //for (int j = 1; j <= 3; ++ j ) {
                        //    cur_indices[j] = @indices(i, j) + @offsets(k, j - 1);
                        //}
                        //cur_indices[0] = @indices(i, 0);

                        for (int j = 0; j < 3; ++ j ) {
                            cur_indices[j] = @indices(i, j + 1) + @offsets(k, j);
                        }
                        cur_indices[3] = @indices(i, 0);

                        uint64_t hash = 14695981039346656037UL;
                        for (int j = 0; j < 4; ++ j ) {
                            //hash ^= (unsigned int)cur_indices[j];
                            hash ^= (uint64_t)cur_indices[j];
                            hash *= 1099511628211UL;
                        }
                        hash = (hash >> 60) ^ (hash & 0xFFFFFFFFFFFFFFF);
                        @out(k, i) = hash;
                    }
                }
                @alias(indices, in0)
                @alias(offsets, in1)
                int thread_nums = 512;
                kernel_hash_kernel <<< (indices_shape0 * offsets_shape0 + thread_nums - 1) / thread_nums, thread_nums >>> (@ARGS);
            """,
            cpu_header="""
                #include <vector>
                @alias(indices, in0)
                @alias(offsets, in1)
            """,
            cpu_src="""
                auto K = offsets_shape0;
                auto N = indices_shape0;

                for (int k = 0; k < offsets_shape0; ++ k ) {
                    #pragma omp parallel for
                    for (int i = 0; i < indices_shape0; ++ i ) {
                        int cur_indices[4];
                        for (int j = 1; j <= 3; ++ j ) {
                            cur_indices[j] = @indices(i, j) + @offsets(k, j - 1);
                        }
                        cur_indices[0] = @indices(i, 0);
                        uint64_t hash = 14695981039346656037UL;
                        for (int j = 0; j < 4; ++ j ) {
                            //hash ^= (unsigned int)cur_indices[j];
                            hash ^= (uint64_t)cur_indices[j];
                            hash *= 1099511628211UL;
                        }
                        hash = (hash >> 60) ^ (hash & 0xFFFFFFFFFFFFFFF);
                        @out(k, i) = hash;
                    }
                }
            """
        )