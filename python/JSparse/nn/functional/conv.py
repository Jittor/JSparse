import numpy as np
from typing import List, Optional, Tuple, Union

import jittor as jt
from jittor import Function, reduce
from jittor.misc import _pair, _triple

from JSparse import SparseTensor
from JSparse.nn import functional as F
from JSparse.nn.utils import get_kernel_offsets
from JSparse import make_ntuple

__all__ = ['conv3d', 'Convolution']

def convolution(
    input: jt.Var,
    weight: jt.Var,
    nbmaps: jt.Var,
    nbsizes: jt.Var,
    sizes: Tuple[int, int],
    transposed: bool = False,
) -> jt.Var:
    if not transposed:
        output = jt.zeros((sizes[1], weight.size(-1)), dtype=input.dtype)
    else:
        output = jt.zeros((sizes[0], weight.size(-1)), dtype=input.dtype)

    assert input.size(1) == weight.size(1)
    in_size = input.size(0)
    in_channels = input.size(1)
    out_size = output.size(0)
    # out_channels = output.size(1)

    kernel_volume = weight.size(0)
    flag = False
    mid = kernel_volume // 2
    in_buffer_size = jt.Var(1)
    if kernel_volume % 2 == 1 and out_size == in_size:
        flag = True
        in_buffer_size = max(in_buffer_size, jt.max(nbsizes[:mid]))
        in_buffer_size = max(in_buffer_size, jt.max(nbsizes[mid + 1:]))
        output += jt.matmul(input, weight[mid, :, :])
    else:
        in_buffer_size = jt.max(nbsizes)
    
    # in_buffer_activated : in_buffer_size * in_channels
    # weight              : in_channels    * out_channels
    # out_buffer_activated: in_buffer_size * out_channels
    cur_offset = 0
    for i in range(kernel_volume):
        n_active_feats = int(nbsizes[i])
        if (flag and (i == mid)):
            cur_offset += n_active_feats
            continue
        if n_active_feats == 0:
            continue

        t = 1 if transposed else 0
        # print(n_active_feats, in_channels, cur_offset, t)
        ################
        ## gather
        ################
        in_buffer_activated = input.reindex([n_active_feats, in_channels], ['@e0(i0)', 'i1'], extras=[nbmaps[cur_offset:cur_offset + n_active_feats, t]])
        ################
        ## matmul
        ################
        out_buffer_activated = jt.matmul(in_buffer_activated, weight[i, :, :])
        ################
        ## scatter
        ################
        output = output.scatter_(0, nbmaps[cur_offset:cur_offset + n_active_feats, 1 - t], out_buffer_activated, reduce='add')
        cur_offset += n_active_feats
    return output

class Convolution(Function):

    def __init__(self):
        self.cuda_header = '''
            #undef out
            #include <cuda.h>
            #include <cublas_v2.h>
            #include <cuda_runtime.h>
            #include <thrust/extrema.h>
            #include <thrust/device_ptr.h>
            #include <executor.h>
            #include <algorithm>
            #include <nvToolsExt.h> 

            template <typename scalar_t>
            __global__ void gather_kernel(const int n_k, const int c,
                                          const scalar_t *in_feat, scalar_t *out_feat,
                                          const int *nbmaps, const int t) {
                int idx = blockDim.x * blockIdx.x + threadIdx.x;
                int i = idx / c, j = idx % c;
                if (i >= n_k) return;
                int in_pos = nbmaps[2 * i + t];
                out_feat[i * c + j] = in_feat[in_pos * c + j];
            }

            template <typename scalar_t>
            __global__ void scatter_kernel(const int n_k, const int c,
                                           const scalar_t *in_feat, scalar_t *out_feat,
                                           const int *nbmaps, const int t) {
                int idx = blockDim.x * blockIdx.x + threadIdx.x;
                int i = idx / c, j = idx % c;
                if (i >= n_k) return;
                int out_pos = nbmaps[2 * i + 1 - t];
                out_feat[out_pos * c + j] += in_feat[i * c + j];
            }

            template <typename scalar_t>
            __global__ void gather_kernel_quick(const int n_k, const int c,
                                                const scalar_t *in_feat, scalar_t *out_feat,
                                                const int *nbmaps, const int t) {
                int i = blockIdx.x, j = threadIdx.x;
                if (i >= n_k) return;
                int in_pos = nbmaps[2 * i + t];
                out_feat[i * c + j] = in_feat[in_pos * c + j];
            }

            template <typename scalar_t>
            __global__ void scatter_kernel_quick(const int n_k, const int c,
                                                 const scalar_t *in_feat, scalar_t *out_feat,
                                                 const int *nbmaps, const int t) {
                int i = blockIdx.x, j = threadIdx.x;
                if (i >= n_k) return;
                int out_pos = nbmaps[2 * i + 1 - t];
                out_feat[out_pos * c + j] += in_feat[i * c + j];
            }

            template <typename scalar_t>
            void gather(const int n_k, const int c,
                        const scalar_t *in_feat, scalar_t *out_feat,
                        const int *nbmaps, const int t) {
                static int threadNum = 0;
                if (threadNum == 0) {
                    cudaDeviceProp prop;
                    cudaGetDeviceProperties(&prop, 0);
                    threadNum = prop.maxThreadsPerBlock;
                }
                if (c <= threadNum)
                    gather_kernel_quick <<< n_k, c >>> (n_k, c, in_feat, out_feat, nbmaps, t);
                else
                    gather_kernel <<< (n_k * c + threadNum - 1) / threadNum, threadNum >>> (n_k, c, in_feat, out_feat, nbmaps, t);
            }

            template <typename scalar_t>
            void scatter(const int n_k, const int c,
                        const scalar_t *in_feat, scalar_t *out_feat,
                        const int *nbmaps, const int t) {
                static int threadNum = 0;
                if (threadNum == 0) {
                    cudaDeviceProp prop;
                    cudaGetDeviceProperties(&prop, 0);
                    threadNum = prop.maxThreadsPerBlock;
                }
                if (c <= threadNum)
                    scatter_kernel_quick <<< n_k, c >>> (n_k, c, in_feat, out_feat, nbmaps, t);
                else
                    scatter_kernel <<< (n_k * c + threadNum - 1) / threadNum, threadNum >>> (n_k, c, in_feat, out_feat, nbmaps, t);
            }

            template <typename scalar_t>
            cudaDataType getDtype(const scalar_t *ptr) {
                return std::is_same<scalar_t, jittor::float16>::value ? CUDA_R_16F : CUDA_R_32F;
            }

            template <typename scalar_t0, typename scalar_t1, typename scalar_t2, typename scalar_t3, typename scalar_t4>
            cublasStatus_t cublasgemm(cublasHandle_t handle,
                                      cublasOperation_t transa, cublasOperation_t transb,
                                      int m, int n, int k,
                                      const scalar_t0 *alpha,
                                      const scalar_t1 *A, int lda,
                                      const scalar_t2 *B, int ldb,
                                      const scalar_t3 *beta,
                                      scalar_t4 *C, int ldc) {
                cublasGemmEx(
                    handle, transa, transb,
                    m, n, k,
                    alpha,
                    A, getDtype(A), lda,
                    B, getDtype(B), ldb,
                    beta,
                    C, getDtype(C), ldc,
                    getDtype(C),
                    CUBLAS_GEMM_DEFAULT
                );
            }
        '''

    def execute(
        self,
        input: jt.Var,
        weight: jt.Var,
        nbmaps: jt.Var,
        nbsizes: jt.Var,
        sizes: Tuple[int, int],
        transposed: bool = False,
    ) -> jt.Var:
        nbmaps = nbmaps.int()
        nbsizes = nbsizes.int()
        if not transposed:
            output_size = (sizes[1], weight.size(-1)) 
        else:
            output_size = (sizes[0], weight.size(-1))

        assert input.size(1) == weight.size(1)

        self.save_vars = input, weight, nbmaps.numpy(), nbsizes.numpy(), transposed
        t = 1 if transposed else 0

        return jt.code(output_size, input.dtype, [input, weight, nbmaps, nbsizes], 
            cuda_header=self.cuda_header,
            cuda_src="""
                @alias(input, in0)
                @alias(weight, in1)
                @alias(nbmaps, in2)
                @alias(nbsizes, in3)
                @alias(output, out0)
            """ + f"""
                int t = {t};
            """ + """   
                int in_size = input_shape0;
                int n_in_channels = input_shape1;
                int out_size = output_shape0;
                int n_out_channels = output_shape1;
                int kernel_volume = weight_shape0;

                int in_buffer_size = 1;
                bool flag = false;
                int mid = kernel_volume / 2;

                cudaMemset(output_p, 0, output->size);
                const input_type alpha = 1.0f;
                const input_type beta  = 0.0f;

                int* nbsizes_h;
                cudaMallocHost((void **)&nbsizes_h, nbsizes->size);
                cudaMemcpy(nbsizes_h, nbsizes_p, nbsizes->size, cudaMemcpyDeviceToHost);

                cublasHandle_t handle;
                cublasCreate(&handle);

                if (kernel_volume & 1 && out_size == input_shape0) {
                    flag = true;

                    in_buffer_size = std::max(in_buffer_size, 
                        *std::max_element(nbsizes_h, nbsizes_h + mid));

                    in_buffer_size = std::max(in_buffer_size, 
                        *std::max_element(nbsizes_h + mid + 1, nbsizes_h + kernel_volume));

                    cublasgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                                n_out_channels, in_size, n_in_channels,
                                &alpha,
                                weight_p + mid * weight_shape1 * weight_shape2, n_out_channels,
                                input_p, n_in_channels,
                                &beta,
                                output_p, n_out_channels);
                } else {
                    in_buffer_size = std::max(in_buffer_size, 
                        *std::max_element(nbsizes_h, nbsizes_h + kernel_volume));
                }

                input_type *in_buffer_activated, *out_buffer_activated;
                size_t in_buffer_allocation, out_buffer_allocation;
                in_buffer_activated = (input_type *)exe.allocator->alloc(in_buffer_size * n_in_channels * sizeof(input_type), in_buffer_allocation);
                out_buffer_activated = (input_type *)exe.allocator->alloc(in_buffer_size * n_out_channels * sizeof(input_type), out_buffer_allocation);

                cudaMemset(in_buffer_activated, 0, in_buffer_size * n_in_channels * sizeof(input_type));
                cudaMemset(out_buffer_activated, 0, in_buffer_size * n_out_channels * sizeof(input_type));

                int cur_offset = 0;

                for (int i = 0; i < kernel_volume; ++ i ) {
                    int n_active_feats = nbsizes_h[i];
                    if (n_active_feats == 0) continue;

                    if ((i == mid) && flag) {
                        cur_offset += 2 * n_active_feats;
                        continue;
                    }

                    // gather
                    gather(n_active_feats, n_in_channels, input_p, in_buffer_activated, nbmaps_p + cur_offset, t);
                    
                    // matmul 
                    // gemm: (o, c) x (c, i) = (o, i)
                    
                    cublasgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                                n_out_channels, n_active_feats, n_in_channels,
                                &alpha,
                                weight_p + i * weight_shape1 * weight_shape2, n_out_channels,
                                in_buffer_activated, n_in_channels,
                                &beta,
                                out_buffer_activated, n_out_channels);

                    // scatter
                    scatter(n_active_feats, n_out_channels, out_buffer_activated, output_p, nbmaps_p + cur_offset, t);

                    cur_offset += 2 * n_active_feats;
                }
                cudaFreeHost(nbsizes_h);

                exe.allocator->free(in_buffer_activated, in_buffer_size * n_in_channels * sizeof(input_type), in_buffer_allocation);
                exe.allocator->free(out_buffer_activated, in_buffer_size * n_out_channels * sizeof(input_type), out_buffer_allocation);

                cublasDestroy(handle);
            """,
        )
    
    def grad(
        self, 
        grad_output: jt.Var
    ):
        input, weight, nbmaps, nbsizes, transposed = self.save_vars
        nbmaps = jt.array(nbmaps)
        nbsizes = jt.array(nbsizes)

        grad_input, grad_weight = jt.code([input.shape, weight.shape], [input.dtype, weight.dtype], [input, weight, nbmaps, nbsizes, grad_output], 
            cuda_header=self.cuda_header, 
            cuda_src="""
                @alias(input, in0)
                @alias(weight, in1)
                @alias(nbmaps, in2)
                @alias(nbsizes, in3)
                @alias(grad_output, in4)
                @alias(grad_input, out0)
                @alias(grad_weight, out1)
            """ + f"""
                int t = {1 if transposed else 0};
            """ + """   

                int in_size = input_shape0;
                int n_in_channels = input_shape1;
                int out_size = grad_output_shape0;
                int n_out_channels = weight_shape2;

                int kernel_volume = weight_shape0;
                bool flag = false;
                int in_buffer_size = 1;

                int* nbsizes_h;
                cudaMallocHost((void **)&nbsizes_h, nbsizes->size);
                cudaMemcpy(nbsizes_h, nbsizes_p, nbsizes->size, cudaMemcpyDeviceToHost);

                in_buffer_size = std::max(in_buffer_size, *std::max_element(nbsizes_h, nbsizes_h + kernel_volume));

                input_type *in_buffer_activated, *in_grad_buffer_activated, *out_grad_buffer_activated;
                size_t in_buffer_allocation, in_grad_buffer_allocation, out_grad_buffer_allocation;
                in_buffer_activated = (input_type *)exe.allocator->alloc(in_buffer_size * n_in_channels * sizeof(input_type), in_buffer_allocation);
                in_grad_buffer_activated = (input_type *)exe.allocator->alloc(in_buffer_size * n_in_channels * sizeof(input_type), in_grad_buffer_allocation);
                out_grad_buffer_activated = (input_type *)exe.allocator->alloc(in_buffer_size * n_out_channels * sizeof(input_type), out_grad_buffer_allocation);

                cudaMemset(in_buffer_activated, 0, in_buffer_size * n_in_channels * sizeof(input_type));
                cudaMemset(out_grad_buffer_activated, 0, in_buffer_size * n_out_channels * sizeof(input_type));
                cudaMemset(grad_input_p, 0, grad_input->size);
                cudaMemset(grad_weight_p, 0, grad_weight->size);

                int cur_offset = 0;
                int mid = kernel_volume / 2;
                const input_type alpha = 1.0f;
                const input_type beta  = 0.0f;

                cublasHandle_t handle;
                cublasCreate(&handle);

                for (int i = 0; i < kernel_volume; ++ i ) {
                    int n_active_feats = nbsizes_h[i];
                    if (flag && (i == mid)) {
                        cur_offset += 2 * n_active_feats;
                        continue;
                    }

                    if (n_active_feats == 0) continue;

                    // gather
                    gather(n_active_feats, n_out_channels, grad_output_p, out_grad_buffer_activated, nbmaps_p + cur_offset, 1 - t);
                    gather(n_active_feats, n_in_channels, input_p, in_buffer_activated, nbmaps_p + cur_offset, t);
                    
                    // gemm
                    
                    // grad for input
                    //    in_grad_buffer_activated     =     out_grad_buffer_activated     @            weight^T
                    // n_active_feats x n_in_channels     n_active_feats x n_out_channels     n_out_channels x n_in_channels
                    cublasgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 
                                n_in_channels, n_active_feats, n_out_channels,
                                &alpha,
                                weight_p + i * weight_shape1 * weight_shape2, n_out_channels,
                                out_grad_buffer_activated, n_out_channels,
                                &beta,
                                in_grad_buffer_activated, n_in_channels);
                    
                    // grad for weight
                    //       kernel_grad_buffer        =     in_buffer_activated^T         @    out_grad_buffer_activated
                    // n_in_channels x n_out_channels     n_in_channels x n_active_feats      n_active_feats x n_out_channels
                    cublasgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, 
                                n_out_channels, n_in_channels, n_active_feats,
                                &alpha,
                                out_grad_buffer_activated, n_out_channels,
                                in_buffer_activated, n_in_channels,
                                &beta,
                                grad_weight_p + i * grad_weight_shape1 * grad_weight_shape2, n_out_channels);

                    // scatter
                    //cudaDeviceSynchronize();
                    scatter(n_active_feats, n_in_channels, in_grad_buffer_activated, grad_input_p, nbmaps_p + cur_offset, 1 - t);

                    cur_offset += 2 * n_active_feats;
                }
                cudaFreeHost(nbsizes_h);

                exe.allocator->free(in_buffer_activated, in_buffer_size * n_in_channels * sizeof(input_type), in_buffer_allocation);
                exe.allocator->free(in_grad_buffer_activated, in_buffer_size * n_in_channels * sizeof(input_type), in_grad_buffer_allocation);
                exe.allocator->free(out_grad_buffer_activated, in_buffer_size * n_out_channels * sizeof(input_type), out_grad_buffer_allocation);

                cublasDestroy(handle);
            """
        )
        return grad_input, grad_weight, None, None, None, None

def conv3d(
    input: SparseTensor,
    weight: jt.Var,
    kernel_size: Union[int, Tuple[int, ...]],
    bias: Optional[jt.Var] = None,
    stride: Union[int, Tuple[int, ...]] = 1,
    dilation: Union[int, Tuple[int, ...]] = 1,
    groups: int = 1,
    transposed: bool = False,
    algorithm: str = "cuda"
) -> SparseTensor:
    kernel_size = _triple(kernel_size)
    stride = _triple(stride)
    dilation = _triple(dilation)

    if algorithm == "cuda":
        algo = Convolution.apply
    elif algorithm == "jittor":
        algo = convolution

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

        output_values = algo(
            input.values, weight,
            *input.kmaps[(input.stride, kernel_size, stride, dilation)],
            transposed,
        )
    else:
        output_stride = tuple(input.stride[k] // stride[k] for k in range(3))
        output_indices = input.cmaps[output_stride]
        output_values = algo(
            input.values, weight,
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
