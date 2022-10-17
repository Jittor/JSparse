import jittor as jt

__all__ = ['unique']


def unique(
    input: jt.Var, 
    return_inverse: bool=False, 
    return_counts: bool=False, 
    dim: int=None):

    temp_shape = None
    if dim == None:
        temp_shape = list(input.shape)
        input_flatten = input.flatten()
        dim = 0
    else:
        input_flatten = input

    input_flatten = input_flatten.transpose(dim, 0)
    orig_shape = input_flatten.shape
    input_flatten = input_flatten.view(orig_shape[0], -1)

    indice = jt.code((input_flatten.shape[0], ), 'int32', [input_flatten],
        cpu_header='''
        #include <algorithm>
        ''',
        cpu_src='''
        @alias(input_flatten, in0)
        @alias(indice, out)

        int dimlen = input_flatten_shape0, dimsize = input_flatten_shape1;
        for(int i = 0; i < dimlen; ++i) @indice(i) = i;
        std::sort(&@indice(0), &@indice(dimlen), [&](int a, int b){
            for(int i = 0; i < dimsize; ++i) {
                int lhs = @input_flatten(a, i), rhs = @input_flatten(b, i);
                if (lhs != rhs) return lhs < rhs;
            }
            return false;
        });
        ''',
        cuda_header='''
        #undef out
        #include <thrust/extrema.h>
        #include <thrust/device_ptr.h>
        #include <thrust/execution_policy.h>
        #include <thrust/device_vector.h>
        #include <thrust/sequence.h>
 
        #include <cub/cub.cuh> 
        #include <executor.h>
        ''',
        cuda_src=
        '''
            @alias(input_flatten, in0)
            @alias(indice, out)
            int dimlen = indice_shape0, dimsize = input_flatten_shape1;

            if (dimsize == 1) {
                size_t raw_allocation, d_allocation, temp_storage_bytes = 0;
                void *d_temp_storage = NULL;
                int* raw_ptr = (int*)exe.allocator->alloc(dimlen * (sizeof(int) + sizeof(input_flatten_type)), raw_allocation);

                thrust::device_ptr<int32_t> arange_ptr = thrust::device_pointer_cast(raw_ptr);
                thrust::sequence(arange_ptr, arange_ptr + dimlen);

                cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, input_flatten_p, raw_ptr + dimlen, thrust::raw_pointer_cast(arange_ptr), indice_p, dimlen);
                d_temp_storage = exe.allocator->alloc(temp_storage_bytes, d_allocation);
                cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, input_flatten_p, raw_ptr + dimlen, thrust::raw_pointer_cast(arange_ptr), indice_p, dimlen);

                exe.allocator->free(raw_ptr, dimlen * (sizeof(int) + sizeof(input_flatten_type)), raw_allocation);
                exe.allocator->free(d_temp_storage, temp_storage_bytes, d_allocation);
            } else {
                thrust::device_ptr<input_flatten_type> input_ptr = thrust::device_pointer_cast(input_flatten_p);
                thrust::device_ptr<int32_t> indice_ptr = thrust::device_pointer_cast(indice_p);

                thrust::sequence(indice_ptr, indice_ptr + dimlen);
                thrust::sort(thrust::device, indice_ptr, indice_ptr + dimlen,
                    [=] __device__ (int32_t a, int32_t b)->bool {
                        for(int i = 0; i < dimsize; ++i) {
                            input_flatten_type lhs = input_ptr[i + a * dimsize],
                                               rhs = input_ptr[i + b * dimsize];
                            if (lhs != rhs) return lhs < rhs;
                        }
                        return false;
                    });
            }
        '''
    )
    input_sorted = input_flatten[indice][:]
    
    dimlen = indice.shape[0]

    diff = jt.logical_not(jt.all(input_sorted[1:] == input_sorted[: -1], 1))
    diff = jt.concat([jt.array([False], dtype='bool'), diff], 0)
    diff = jt.array(diff, dtype = jt.int32)
    
    output, inverse = jt.code(
        [(-input_sorted.shape[0], ), (indice.shape)],
        [input_sorted.dtype, indice.dtype],
        [input_sorted, diff, indice],
        cpu_header='''
            #include <algorithm>
            @alias(input_sorted, in0)
            @alias(diff, in1)
            @alias(indice, in2)
            @alias(output, out0)
            @alias(inverse, out1)
        ''',
        cpu_src=
        f"bool return_inverse = {int(return_inverse)};" +
        '''
            int tot = -1;
            bool return_inverse = @out2(0);
            for (int i = 0; i < input_sorted_shape0; ++i) {
                if (i == 0 || @diff(i)) {
                    ++tot; @output(tot) = i;
                }
                if (return_inverse)
                    @inverse(@indice(i)) = tot;
            }
            output->set_shape({tot + 1});
        ''',
        cuda_header='''
            #undef out

            #include <thrust/extrema.h>
            #include <thrust/device_ptr.h>
            #include <thrust/execution_policy.h>
            #include <thrust/scan.h>
            #include <executor.h>

            @alias(input_sorted, in0)
            @alias(diff, in1)
            @alias(indice, in2)
            @alias(output, out0)
            @alias(inverse, out1)
        ''',
        cuda_src=
        f"bool return_inverse = {int(return_inverse)};" +
        '''
            int dimlen = input_sorted_shape0, dimsize = input_sorted_shape1;
            size_t raw_allocation;
            int* raw_ptr = (int*)exe.allocator->alloc(2 * dimlen * sizeof(int), raw_allocation);

            thrust::device_ptr<int32_t> diff_ptr = thrust::device_pointer_cast(diff_p),
                                        inverse_ptr = thrust::device_pointer_cast(inverse_p),
                                        array_ptr = thrust::device_pointer_cast(raw_ptr),
                                        sum_ptr = thrust::device_pointer_cast(raw_ptr + dimlen),
                                        indice_ptr = thrust::device_pointer_cast(indice_p);
            thrust::device_ptr<input_sorted_type> input_ptr = thrust::device_pointer_cast(input_sorted_p);

            if (return_inverse) {
                thrust::inclusive_scan(diff_ptr, diff_ptr + dimlen, sum_ptr);
                thrust::scatter(sum_ptr, sum_ptr + dimlen, indice_ptr, inverse_ptr);
            }

            thrust::sequence(array_ptr, array_ptr + dimlen);
            int num = thrust::unique(array_ptr, array_ptr + dimlen,
                [=] __device__ (int32_t a, int32_t b)->bool {
                    for(int i = 0; i < dimsize; ++i) {
                        input_sorted_type
                            lhs = input_ptr[i + a * dimsize],
                            rhs = input_ptr[i + b * dimsize];
                        if (lhs != rhs) return false;
                    }
                    return true;
                }) - array_ptr;

            cudaMemcpy(output_p, raw_ptr, sizeof(int) * num, cudaMemcpyDeviceToDevice);
            exe.allocator->free(raw_ptr, 2 * dimlen * sizeof(int), raw_allocation);
            output->set_shape({ num });
        '''
    )
    indice_shape = (output.shape[0], )
    output = input_sorted[output][:]

    new_shape = list(orig_shape[1:])
    new_shape.insert(0, -1)
    output = output.view(new_shape).transpose(dim, 0)
    if temp_shape != None:
        inverse = inverse.view(temp_shape).transpose(dim, 0)

    if return_inverse:
        if return_counts:
            counts = jt.zeros(indice_shape, dtype=jt.int32)
            jt.scatter_(counts, 0, inverse.flatten(), jt.ones(dimlen), reduce='add')
            return output, inverse, counts
        else:
            return output, inverse
    else:
        return output