from unittest import result
from typing import Union, Tuple

import jittor as jt
from jittor import Function

__all__ = ['spmm', 'gtsv']

Header = """
    #undef out
    #include <assert.h>
    #include <executor.h>
    #include <iostream>

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
"""

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
        with jt.flag_scope(compile_options = {"FLAGS: -lcusparse ": 1}):
            result = jt.code(output_size, vals.dtype, [rows, cols, vals, mat],
                cuda_header=Header,
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

                    cusparseDnMatDescr_t dense_descr;
                    cusparseCreateDnMat(&dense_descr, 
                                        dim_k, dim_j, dim_k,
                                        (void*) mat2_p,
                                        cuda_data_type, CUSPARSE_ORDER_COL);

                    cusparseDnMatDescr_t result_descr;
                    cusparseCreateDnMat(&result_descr, 
                                        dim_i, dim_k, dim_i,
                                        (void*) result_p,
                                        cuda_data_type, CUSPARSE_ORDER_COL);

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

                    if (required_workspace_buffer_size > workspace_buffer_size) {
                        if (workspace_buffer != nullptr) {
                            cudaFree(workspace_buffer);
                        }
                        workspace_buffer_size = required_workspace_buffer_size;
                        cudaMallocManaged(&workspace_buffer, workspace_buffer_size);
                    }

                    cusparseSpMM(handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_TRANSPOSE, 
                                 (void*) &alpha, 
                                 sparse_descr, dense_descr,
                                 (void*) &beta,
                                 result_descr,
                                 cuda_data_type, mm_alg,
                                 workspace_buffer);

                    cusparseDestroySpMat(sparse_descr);
                    cusparseDestroyDnMat(dense_descr);
                    cusparseDestroyDnMat(result_descr);

                    if (!is_sorted) {
                        exe.allocator->free(sorted_rows_ptr, 2 * nnz * sizeof(int), sorted_rows_allocation);
                        exe.allocator->free(sorted_vals_ptr, nnz * sizeof(float), sorted_vals_allocation);
                    }

                    if (workspace_buffer != nullptr) {
                        cudaFree(workspace_buffer);
                    }
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

def gtsv(
    dl: jt.Var,
    d: jt.Var,
    du: jt.Var,
    B: jt.Var,
    pivoting: bool = True
):
    '''
    General Tridiagonal Solve:

        This function can solve A * X = B, return matrix X.
        The coefficient matrix A of each of these tri-diagonal linear system is defined with three vectors corresponding to its lower (dl), main (d), and upper (du) matrix diagonals; the right-hand sides are stored in the dense matrix B. Notice that solution X overwrites right-hand-side matrix B on exit.

        Assuming A is of size m and base-1, dl, d and du are defined by the following formula:
        dl(i) := A(i, i-1) for i=1,2,...,m
        The first element of dl is out-of-bound (dl(1) := A(1,0)), so dl(1) = 0.
        d(i) = A(i,i) for i=1,2,...,m
        du(i) = A(i,i+1) for i=1,2,...,m
        The last element of du is out-of-bound (du(m) := A(m,m+1)), so du(m) = 0.

    The routine does perform pivoting, which usually results in more accurate and more stable results than no-pivoting.

    Hint: You can use float64 to get more accurate results to avoid Nan or 0 (or you can choose CPU Version).
    '''
 
    assert dl.dtype == d.dtype and d.dtype == du.dtype and du.dtype == B.dtype
    assert dl.ndim == 1 and d.ndim == 1 and du.ndim == 1 and B.ndim == 2
    assert d.shape[0] <= B.shape[0]
 
    with jt.flag_scope(compile_options = {"FLAGS: -lcusparse ": 1}):
        prefix = "cusparse" + ("S" if dl.dtype == jt.float32 else "D") + "gtsv2" + ("" if pivoting else "_nopivot")
        result = jt.code(B.shape, B.dtype, [dl, d, du, B], cuda_header = Header,
            cuda_src = """
            @alias(dl, in0)
            @alias(d, in1)
            @alias(du, in2)
            @alias(B, in3)
            @alias(result, out)

            int m = d_shape0, ldb = B_shape0, n = B_shape1;

            cusparseHandle_t handle = 0;
            cusparseCreate(&handle);

            size_t bufferSize;
            auto stat = """ + prefix + """_bufferSizeExt(
                handle, m, n, dl_p, d_p, du_p, B_p, ldb, &bufferSize);
            assert(stat == CUSPARSE_STATUS_SUCCESS);

            unsigned char *buffer;
            cudaMalloc(&buffer, bufferSize);
            cudaMemcpy(result_p, B_p, m * n * sizeof(B_p), cudaMemcpyDeviceToDevice);

            stat = """ + prefix + """(handle, m, n, dl_p, d_p, du_p, result_p, ldb, (void *)buffer);
            if(stat != CUSPARSE_STATUS_SUCCESS)
                std::cout << "cusparse solve error: " << (int)stat << std::endl;
            cudaFree(buffer);
            """,
            cpu_header = """
            #include <cmath>
            #include <iostream>
            #include <vector>
            """,
            cpu_src = """
            @alias(dl, in0)
            @alias(d, in1)
            @alias(du, in2)
            @alias(B, in3)
            @alias(result, out)

            int m = d_shape0, ldb = B_shape0, n = B_shape1;
            for(int k = 0; k < n; ++k) {
                std::vector<double> c_star(m, 0), d_star(m, 0);

                c_star[0] = @du(0) / @d(0);
                d_star[0] = @B(0, k) / @d(0);
                for(int i = 1; i < m; ++i) {
                    double temp = 1.0 / (@d(i) - @dl(i) * c_star[i - 1]);
                    c_star[i] = @du(i) * temp;
                    d_star[i] = (@B(i, k) - @dl(i) * d_star[i - 1]) * temp;
                }

                for (int i = m; ~i; --i)
                    @result(i, k) = d_star[i] - (i < m - 1 ? c_star[i] * @result(i + 1, k) : 0);
            }
            """
       )

    return result

if __name__ == '__main__':
    jt.flags.use_cuda = False
    #n = 20000
    #n = 1280 * 720
    n = 1920 * 1080
    d = jt.Var([1.0] * n)
    dl = jt.Var([0.2] * n)
    dl[0] = 0
    du = jt.Var([0.3] * n)
    du[n - 1] = 0
    B = jt.Var([[1.0] * n]).transpose()

    d = d.float64()
    dl = dl.float64()
    du = du.float64()
    B = B.float64()
    import time
    jt.sync_all()
    t = time.time()
    for i in range(1000):
        T = gtsv(dl, d, du, B)
        jt.sync_all()
    print(time.time() - t)