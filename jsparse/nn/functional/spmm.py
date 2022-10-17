from unittest import result
from typing import Union, Tuple

import jittor as jt
from jittor import Function

from jsparse import SparseTensor

__all__ = ['spmm']


def spmm(
    rows: jt.Var,
    cols: jt.Var,
    vals: jt.Var,
    size: Union[Tuple[int, int], jt.NanoVector],
    mat: jt.Var,
    is_sorted: bool = False,
    cuda_spmm_alg: int = 1,
) -> jt.Var:

    assert len(rows) == len(cols), "Invalid length"
    assert len(rows) == len(vals), "Invalid length"
    assert vals.dtype == mat.dtype, "dtype mismatch"

    if jt.flags.use_cuda > 0:
        rows = rows.int32()
        cols = cols.int32()
        output_size = (mat.shape[1], size[0])
        result = jt.code(output_size, vals.dtype, [rows, cols, vals, mat],
            cuda_header="""
                #undef out
                #include <assert.h>
                #include <executor.h>

                #include <cuda.h>
                #include <cuda_runtime.h>
                #include <cusparse.h>

                #include <thrust/sort.h>
                #include <thrust/tuple.h>
                #include <thrust/iterator/zip_iterator.h>

                template <typename scalar_t>
                cudaDataType getDtype(const scalar_t *ptr) {
                    assert((std::is_same<scalar_t, jittor::float32>::value || std::is_same<scalar_t, jittor::float64>::value));
                    //if (std::is_same<scalar_t, jittor::float32>::value) 
                    //    return CUDA_R_32F;
                    //else if (std::is_same<scalar_t, jittor::float64>::value)
                    //    return CUDA_R_64F;
                    return std::is_same<scalar_t, jittor::float32>::value ? CUDA_R_32F : CUDA_R_64F;
                }
            """,
            cuda_src="""
                @alias(rows, in0)
                @alias(cols, in1)
                @alias(vals, in2)
                @alias(mat2, in3)
                @alias(result, out0)
            """ + f"""
                const int64_t dim_i = {size[0]};
                const int64_t dim_j = {size[1]};
                const int64_t spmm_algorithm_id = {cuda_spmm_alg};
                const bool is_sorted = {'true' if is_sorted else 'false'};
                const int64_t nnz = {rows.numel()};
            """ + """
                const bool is_int32 = true;

                cusparseHandle_t handle = 0;
                cusparseCreate(&handle);

                cusparseSpMMAlg_t mm_alg;
                switch (spmm_algorithm_id) {
                    case 1:
                        mm_alg = CUSPARSE_COOMM_ALG1;
                        break;
                    case 2:
                        mm_alg = CUSPARSE_COOMM_ALG2;
                        break;
                    case 3:
                        mm_alg = CUSPARSE_COOMM_ALG3;
                        break;
                    case 4:
                        mm_alg = CUSPARSE_SPMM_COO_ALG4;
                        break;
                    default:
                        mm_alg = CUSPARSE_MM_ALG_DEFAULT;
                }

                cudaDeviceSynchronize();
                //std::cout << "step " << 1 << std::endl;

                int64_t dim_k = mat2_shape1;

                cudaMemset(result_p, 0.0, result->size);

                const float alpha = 1.0f;
                const float beta  = 0.0f;

                cudaDataType cuda_data_type = getDtype<mat2_type>(mat2_p);

                int* sorted_rows_ptr, *sorted_cols_ptr;
                float *sorted_vals_ptr;
                size_t sorted_rows_allocation, sorted_cols_allocation, sorted_vals_allocation;

                if (!is_sorted) {
                    sorted_rows_ptr = (int *)exe.allocator->alloc(2 * nnz * sizeof(int), sorted_rows_allocation);
                    sorted_cols_ptr = sorted_rows_ptr + nnz;
                    sorted_vals_ptr = (float *)exe.allocator->alloc(nnz * sizeof(float), sorted_vals_allocation);

                    cudaMemcpy(sorted_rows_ptr, rows_p, nnz * sizeof(int), cudaMemcpyDeviceToDevice);
                    cudaMemcpy(sorted_cols_ptr, cols_p, nnz * sizeof(int), cudaMemcpyDeviceToDevice);
                    cudaMemcpy(sorted_vals_ptr, vals_p, nnz * sizeof(float), cudaMemcpyDeviceToDevice);

                    thrust::sort_by_key(thrust::device,
                                        sorted_rows_ptr,
                                        sorted_rows_ptr + nnz,
                                        thrust::make_zip_iterator(
                                            thrust::make_tuple(
                                                sorted_cols_ptr,
                                                sorted_vals_ptr
                                            )));

                    cudaDeviceSynchronize();
                } else {
                    sorted_rows_ptr = rows_p;
                    sorted_cols_ptr = cols_p;
                    sorted_vals_ptr = vals_p;
                }

                cudaDeviceSynchronize();
                //std::cout << "step " << 2 << std::endl;

                size_t workspace_buffer_size = 0;
                void *workspace_buffer = nullptr;
                
                cusparseSpMatDescr_t sparse_descr;
                cusparseCreateCoo(
                    &sparse_descr,
                    dim_i, dim_j, nnz,
                    (void*) sorted_rows_ptr,
                    (void*) sorted_cols_ptr,
                    (void*) sorted_vals_ptr,
                    CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO, cuda_data_type);

                cudaDeviceSynchronize();
                //std::cout << "step " << 3 << std::endl;
                
                cusparseDnMatDescr_t dense_descr;
                cusparseCreateDnMat(&dense_descr, 
                                    dim_k, dim_j, dim_k,
                                    (void*) mat2_p,
                                    cuda_data_type, CUSPARSE_ORDER_COL);

                cudaDeviceSynchronize();
                //std::cout << "step " << 4 << std::endl;

                cusparseDnMatDescr_t result_descr;
                cusparseCreateDnMat(&result_descr, 
                                    dim_i, dim_k, dim_i,
                                    (void*) result_p,
                                    cuda_data_type, CUSPARSE_ORDER_COL);

                cudaDeviceSynchronize();
                //std::cout << "step " << 5 << std::endl;
                
                size_t required_workspace_buffer_size = 0;
                cusparseSpMM_bufferSize(
                    handle, 
                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                    CUSPARSE_OPERATION_TRANSPOSE,
                    (void*) &alpha,
                    sparse_descr, dense_descr,
                    (void*) &beta,
                    result_descr,
                    cuda_data_type, mm_alg,
                    &required_workspace_buffer_size);
                
                cudaDeviceSynchronize();
                //std::cout << "step " << 6 << std::endl;
                
                if (required_workspace_buffer_size > workspace_buffer_size) {
                    if (workspace_buffer != nullptr) {
                        cudaFree(workspace_buffer);
                    }
                    workspace_buffer_size = required_workspace_buffer_size;
                    cudaMallocManaged(&workspace_buffer, workspace_buffer_size);
                }

                cudaDeviceSynchronize();
                //std::cout << "step " << 7 << std::endl;

                cusparseSpMM(handle,
                             CUSPARSE_OPERATION_NON_TRANSPOSE,
                             CUSPARSE_OPERATION_TRANSPOSE, 
                             (void*) &alpha, 
                             sparse_descr, dense_descr,
                             (void*) &beta,
                             result_descr,
                             cuda_data_type, mm_alg,
                             workspace_buffer);
                
                cudaDeviceSynchronize();
                //std::cout << "step " << 8 << std::endl;

                cusparseDestroySpMat(sparse_descr);
                cusparseDestroyDnMat(dense_descr);
                cusparseDestroyDnMat(result_descr);

                cudaDeviceSynchronize();
                //std::cout << "step " << 9 << std::endl;

                if (!is_sorted) {
                    exe.allocator->free(sorted_rows_ptr, 2 * nnz * sizeof(int), sorted_rows_allocation);
                    exe.allocator->free(sorted_vals_ptr, nnz * sizeof(float), sorted_vals_allocation);
                }

                if (workspace_buffer != nullptr) {
                    cudaFree(workspace_buffer);
                }

                cudaDeviceSynchronize();
            """)
        result = result.t()
    else:
        raise NotImplementedError()
    return result

class SPMM(Function):
    def execute(
        self,
        rows: jt.Var,
        cols: jt.Var,
        vals: jt.Var,
        size: Union[Tuple[int, int], jt.NanoVector],
        mat: jt.Var,
        cuda_spmm_alg: int = 1,
    ):
        size = tuple(size)
        self.save_vars = rows, cols, vals, size, mat, cuda_spmm_alg
        result = spmm(
            rows,
            cols,
            vals,
            size,
            mat,
            is_sorted=False,
            cuda_spmm_alg=cuda_spmm_alg,
        )
        return result

    def grad(
        self, 
        grad: jt.Var
    ):
        rows, cols, vals, size, mat, cuda_spmm_alg = self.save_vars
        new_size = (size[1], size[0])
        vals_grad = grad.matmul(mat.t())
        mat_grad = spmm(
            cols,
            rows,
            vals,
            new_size,
            grad,
            is_sorted=False,
            cuda_spmm_alg=cuda_spmm_alg
        )

        return None, None, vals_grad, None, mat_grad, None