from typing import Optional

import jittor as jt

__all__ = ['sphash']

def sphash(indices: jt.Var,
           offsets: Optional[jt.Var] = None) -> jt.Var:
    assert indices.dtype == jt.int, indices.dtype
    assert indices.ndim == 2 and indices.shape[1] == 4, indices.shape

    if offsets is None:
        return jt.code((indices.shape[0],), jt.int64, [indices], 
            cuda_header="""
                #include <stdio.h>
                #include <stdlib.h>
                #include <cmath>
                #include <vector>
                @alias(indices, in0)
            """,
            cuda_src="""
                __global__ static void hash_kernel(@ARGS_DEF) {
                    @PRECALC
                    
                    int i = blockIdx.x * blockDim.x + threadIdx.x;
                    if (i < indices_shape0) {
                        uint64_t hash = 14695981039346656037UL;
                        for (int j = 0; j < 4; ++ j ) {
                            hash ^= (unsigned int)@indices(i, j);
                            hash *= 1099511628211UL;
                        }
                        hash = (hash >> 60) ^ (hash & 0xFFFFFFFFFFFFFFF);
                        @out(i) = hash;
                    }
                }
                hash_kernel<<<ceil((double)indices_shape0 / 512), 512>>>(@ARGS);
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
                        hash ^= (unsigned int)@indices(i, j);
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
                
                @alias(indices, in0)
                @alias(offsets, in1)
            """,
            # cuda_src="""
            #     __global__ void kernel_hash_kernel(@ARGS_DEF){
            #         @PRECALC
            #         extern __shared__ int offsets_shared[];

            #         int ix = blockDim.x * blockIdx.x + threadIdx.x;
            #         int iy = blockIdx.y;

            #         // if (!threadIdx.x) {
            #         //     for (int j = 0; j < 3; ++ j ) {
            #         //         offsets_shared[iy * 3 + j] = @offsets(iy, j);
            #         //     }
            #         // }
            #         // __syncthreads();

            #         if (!threadIdx.x) {
            #             for (int j = 0; j < 3; ++ j ) {
            #                 offsets_shared[iy * 3 + j] = @offsets(iy, j);
            #             }
            #         }
            #         __syncthreads();

            #         if (ix < indices_shape0 && iy < offsets_shape0) {
            #             int cur_indices[4];
            #             for (int j = 1; j <= 3; ++ j ) {
            #                 // cur_indices[j] = @indices(ix, j) + @offsets(iy, j - 1);
            #                 cur_indices[j] = @indices(ix, j) + offsets_shared[iy * 3 + j - 1];
            #             }
            #             cur_indices[0] = @indices(ix, 0);
            #             uint64_t hash = 14695981039346656037UL;
            #             for (int j = 0; j < 4; ++ j ) {
            #                 hash ^= (unsigned int)cur_indices[j];
            #                 hash *= 1099511628211UL;
            #             }
            #             hash = (hash >> 60) ^ (hash & 0xFFFFFFFFFFFFFFF);
            #             @out0(iy, ix) = hash;
            #         }
            #     }
            #     dim3 block(512);
            #     dim3 grid((indices_shape0 + block.x - 1) / block.x, offsets_shape0);
            #     kernel_hash_kernel<<< grid, block, offsets_shape0 * 3 * sizeof(offsets_type) >>>(@ARGS);
            # """,
            # cuda_src="""
            #     __global__ void kernel_hash_kernel(@ARGS_DEF){
            #         @PRECALC
            #         extern __shared__ int offsets_shared[];

            #         int ix = blockDim.x * blockIdx.x + threadIdx.x;
            #         int iy = blockDim.y * blockIdx.y + threadIdx.y;

            #         // if (!threadIdx.x) {
            #         //     for (int j = 0; j < 3; ++ j ) {
            #         //         offsets_shared[iy * 3 + j] = @offsets(iy, j);
            #         //     }
            #         // }
            #         // __syncthreads();

            #         if (iy < indices_shape0 && ix < offsets_shape0) {
            #             int cur_indices[4];
            #             // for (int j = 1; j <= 3; ++ j ) {
            #             //     cur_indices[j] = @indices(iy, j) + @offsets(ix, j - 1);
            #                 // cur_indices[j] = @indices(iy, j) + offsets_shared[ix * 3 + j - 1];
            #             // }
            #             cur_indices[0] = @indices(iy, 0);
            #             cur_indices[1] = @indices(iy, 1) + @offsets(ix, 0);
            #             cur_indices[2] = @indices(iy, 2) + @offsets(ix, 1);
            #             cur_indices[3] = @indices(iy, 3) + @offsets(ix, 2);
            #             uint64_t hash = 14695981039346656037UL;
            #             for (int j = 0; j < 4; ++ j ) {
            #                 hash ^= (unsigned int)cur_indices[j];
            #                 hash *= 1099511628211UL;
            #             }
            #             hash = (hash >> 60) ^ (hash & 0xFFFFFFFFFFFFFFF);
            #             @out0(ix, iy) = hash;
            #         }
            #     }
            #     dim3 block(16, 64);
            #     dim3 grid((offsets_shape0 + block.x - 1) / block.x), (indices_shape0 + block.y - 1) / block.y);
            #     kernel_hash_kernel<<< grid, block, offsets_shape0 * 3 * sizeof(offsets_type) >>>(@ARGS);
            # """,
            cuda_src="""
                __global__ void kernel_hash_kernel(@ARGS_DEF){
                    @PRECALC
                    extern __shared__ int offsets_shared[];

                    int idx = blockDim.x * blockIdx.x + threadIdx.x;
                    int k = idx % offsets_shape0;
                    int i = idx / offsets_shape0;

                    if (i < indices_shape0) {
                        int cur_indices[4];
                        for (int j = 1; j <= 3; ++ j ) {
                            cur_indices[j] = @indices(i, j) + @offsets(k, j - 1);
                        }
                        cur_indices[0] = @indices(i, 0);

                        uint64_t hash = 14695981039346656037UL;
                        for (int j = 0; j < 4; ++ j ) {
                            hash ^= (unsigned int)cur_indices[j];
                            hash *= 1099511628211UL;
                        }
                        hash = (hash >> 60) ^ (hash & 0xFFFFFFFFFFFFFFF);
                        @out(k, i) = hash;
                    }
                }
                int thread_nums = 512;
                kernel_hash_kernel<<<ceil((double)(indices_shape0 * offsets_shape0) / thread_nums), thread_nums, offsets_shape0 * 3 * sizeof(int)>>>(@ARGS);
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
                            hash ^= (unsigned int)cur_indices[j];
                            hash *= 1099511628211UL;
                        }
                        hash = (hash >> 60) ^ (hash & 0xFFFFFFFFFFFFFFF);
                        @out(k, i) = hash;
                    }
                }
            """
        )
