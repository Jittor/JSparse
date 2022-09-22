import jittor as jt
import math

__all__ = ['spquery']

def spquery(queries: jt.Var, 
            references: jt.Var) -> jt.Var:
    q_size = queries.size()
    queries = queries.view(-1)

    indices = jt.arange(len(references), dtype=jt.int64)

    if jt.flags.use_cuda > 0:

        output = jt.code(queries.shape, jt.int64, [queries, references, indices], 
            cuda_header="""
                #include <cmath>
                #include <iostream>
                #include <vector>
                #include <cstdint>
                #include <chrono>
                #include <cstdio>
                #include <cstdlib>
                #include <assert.h>

                /** Reserved value for indicating "empty". */
                #define EMPTY_CELL (0)
                /** Max rehashing depth, and error depth. */
                #define MAX_DEPTH (100)
                #define ERR_DEPTH (-1)
                /** CUDA naive thread block size. */
                #define BLOCK_SIZE (256)
                /** CUDA multi-level thread block size = bucket size. */
                #define BUCKET_SIZE (512)
                //#define BUCKET_SIZE (1024)

                __device__ static uint64_t atomicExch(uint64_t *addr, uint64_t val) {
                    return (uint64_t)atomicExch((unsigned long long int *)addr,
                                                (unsigned long long int)val);
                }

                /** Struct of a hash function config. */
                typedef struct {
                    int rv;  // Randomized XOR value.
                    int ss;  // Randomized shift filter start position.
                } FuncConfig;

                /** Hard code hash functions and all inline helper functions for CUDA kernels'
                * use. */
                inline __device__ int do_1st_hash(const uint64_t val, const int num_buckets) {
                    return val % num_buckets;
                }

                inline __device__ int do_2nd_hash(const uint64_t val,
                                                const FuncConfig *const hash_func_configs,
                                                const int func_idx, const int size) {
                    FuncConfig fc = hash_func_configs[func_idx];
                    return ((val ^ fc.rv) >> fc.ss) % size;  // XOR function as 2nd-level hashing.
                }

                // trying to ignore EMPTY_CELL by adding 1 at make_data.
                inline __device__ uint64_t fetch_val(const uint64_t data, const int pos_width) {
                    return data >> pos_width;
                }

                inline __device__ int fetch_func(const uint64_t data, const int pos_width) {
                    return data & ((0x1 << pos_width) - 1);
                }

                inline __device__ uint64_t make_data(const uint64_t val, const int func,
                                                    const int pos_width) {
                    return (val << pos_width) ^ func;
                }

                class CuckooHashTableCuda_Multi {
                private:
                    const int _size;
                    const int _evict_bound;
                    const int _num_funcs;
                    const int _pos_width;
                    const int _num_buckets;

                    FuncConfig *_d_hash_func_configs;

                    /** Cuckoo hash function set. */
                    FuncConfig *_hash_func_configs;

                    /** Private operations. */
                    void gen_hash_funcs() {
                        // Calculate bit width of value range and table size.
                        int val_width = 8 * sizeof(uint64_t) - ceil(log2((double)_num_funcs));
                        int bucket_width = ceil(log2((double)_num_buckets));
                        int size_width = ceil(log2((double)BUCKET_SIZE));
                        // Generate randomized configurations.
                        for (int i = 0; i < _num_funcs; ++i) {  // At index 0 is a dummy function.
                            if (val_width - bucket_width <= size_width)
                                _hash_func_configs[i] = {rand(), 0};
                            else {
                                _hash_func_configs[i] = {
                                    rand(), rand() % (val_width - bucket_width - size_width + 1) +
                                                bucket_width};
                            }
                        }
                    };

                    inline uint64_t fetch_val(const uint64_t data) { return data >> _pos_width; }
                    inline int fetch_func(const uint64_t data) {
                        return data & ((0x1 << _pos_width) - 1);
                    }

                public:
                    CuckooHashTableCuda_Multi(const int size, const int evict_bound,
                                                const int num_funcs)
                        : _size(size),
                            _evict_bound(evict_bound),
                            _num_funcs(num_funcs),
                            _pos_width(ceil(log2((double)_num_funcs))),
                            _num_buckets(ceil((double)_size / BUCKET_SIZE)) {
                        srand(time(NULL));
                        _d_hash_func_configs = NULL;
                        _hash_func_configs = NULL;
                        _hash_func_configs = new FuncConfig[num_funcs];

                        gen_hash_funcs();

                        cudaMalloc((void **)&_d_hash_func_configs, _num_funcs * sizeof(FuncConfig));
                        cudaMemcpy(_d_hash_func_configs, _hash_func_configs,
                                _num_funcs * sizeof(FuncConfig), cudaMemcpyHostToDevice);
                    };
                    ~CuckooHashTableCuda_Multi() {
                        if (_hash_func_configs != NULL) delete[] _hash_func_configs;

                        if (_d_hash_func_configs != NULL) cudaFree(_d_hash_func_configs);
                    };

                    int insert_vals(const uint64_t *const keys, const uint64_t *const vals,
                                    uint64_t *d_key_buf, uint64_t *d_val_buf, uint64_t *d_key,
                                    uint64_t *d_val, const int n);

                    void lookup_vals(const uint64_t *const keys, uint64_t *const results,
                                    uint64_t *d_key, uint64_t *d_val, const int n);
                };

                __global__ void cuckooBucketKernel_Multi(
                    uint64_t *const key_buf, uint64_t *const val_buf, const int size,
                    const uint64_t *const keys, const uint64_t *const vals, const int n,
                    int *const counters, const int num_buckets);

                __global__ void cuckooInsertKernel_Multi(
                    uint64_t *const key, uint64_t *const val, const uint64_t *const key_buf,
                    const uint64_t *const val_buf, const int size,
                    const FuncConfig *const hash_func_configs, const int num_funcs,
                    const int *const counters, const int num_buckets, const int evict_bound,
                    const int pos_width, int *const rehash_requests);

                __global__ void cuckooLookupKernel_Multi(
                    const uint64_t *const keys, uint64_t *const results, const int n,
                    const uint64_t *const all_keys, const uint64_t *const all_vals,
                    const int size, const FuncConfig *const hash_func_configs,
                    const int num_funcs, const int num_buckets, const int pos_width);
                
                __global__ void cuckooBucketKernel_Multi(
                    uint64_t *const key_buf, uint64_t *const val_buf, const int size,
                    const uint64_t *const keys, const uint64_t *const vals, const int n,
                    int *const counters, const int num_buckets) {
                    // Get thread index.
                    int idx = threadIdx.x + blockIdx.x * blockDim.x;

                    // Only threads within range are active.
                    if (idx < n) {
                        // Do 1st-level hashing to get bucket id, then do atomic add to get index
                        // inside the bucket.
                        uint64_t key = keys[idx];
                        uint64_t val = vals[idx];

                        int bucket_num = do_1st_hash(key, num_buckets);
                        int bucket_ofs = atomicAdd(&counters[bucket_num], 1);

                        // Directly write the key into the table buffer.
                        if (bucket_ofs >= BUCKET_SIZE) {
                            printf("%d/%d ERROR: bucket overflow! n=%d, bucket_num=%d/%d, key=%d ", bucket_ofs, BUCKET_SIZE, n, bucket_num, num_buckets, key);
                            assert(bucket_ofs < BUCKET_SIZE);
                        } else {
                            key_buf[bucket_num * BUCKET_SIZE + bucket_ofs] = key;
                            val_buf[bucket_num * BUCKET_SIZE + bucket_ofs] = val;
                        }
                    }
                }

                __global__ void cuckooInsertKernel_Multi(
                    uint64_t *const key, uint64_t *const val, const uint64_t *const key_buf,
                    const uint64_t *const val_buf, const int size,
                    const FuncConfig *const hash_func_configs, const int num_funcs,
                    const int *const counters, const int num_buckets, const int evict_bound,
                    const int pos_width, int *const rehash_requests) {
                    // Create local cuckoo table in shared memory. Size passed in as the third
                    // kernel parameter.
                    extern __shared__ uint64_t local_key[];
                    for (int i = 0; i < num_funcs; ++i) {
                        local_key[i * BUCKET_SIZE + threadIdx.x] = EMPTY_CELL;
                    }

                    // might be useful
                    __syncthreads();

                    // Get thread index.
                    int idx = threadIdx.x + blockIdx.x * blockDim.x;
                    uint64_t cur_idx = idx;

                    // Only threads within local bucket range are active.
                    if (threadIdx.x < counters[blockIdx.x]) {
                        // Set initial conditions.
                        uint64_t cur_key = key_buf[cur_idx];
                        int cur_func = 0;
                        int evict_count = 0;

                        // Start the test-kick-and-reinsert loops.
                        do {
                        int pos = do_2nd_hash(cur_key, hash_func_configs, cur_func, BUCKET_SIZE);

                        uint64_t new_data = make_data(cur_idx + 1, cur_func, pos_width);

                        uint64_t old_idx =
                            atomicExch(&local_key[cur_func * BUCKET_SIZE + pos], new_data);

                        if (old_idx != EMPTY_CELL) {
                            cur_idx = fetch_val(old_idx, pos_width) - 1;
                            // potential overflow here. It seems that cur_idx < 0 is possible!
                            cur_key = key_buf[cur_idx];
                            cur_func = (fetch_func(old_idx, pos_width) + 1) % num_funcs;
                            evict_count++;
                        } else {
                            break;
                        }

                        } while (evict_count < num_funcs * evict_bound);

                        // If exceeds eviction bound, then needs rehashing.
                        if (evict_count >= num_funcs * evict_bound) {
                        atomicAdd(rehash_requests, 1);
                        }
                    }

                    // Every thread write its responsible local slot into the global data table.
                    __syncthreads();
                    for (int i = 0; i < num_funcs; ++i) {
                        uint64_t cur_idx = local_key[i * BUCKET_SIZE + threadIdx.x];
                        if (cur_idx == EMPTY_CELL) {
                        continue;
                        }
                        int cur_func = fetch_func(cur_idx, pos_width);
                        cur_idx = fetch_val(cur_idx, pos_width) - 1;
                        key[i * size + idx] = key_buf[cur_idx];
                        val[i * size + idx] = val_buf[cur_idx];
                    }
                }

                __global__ void cuckooLookupKernel_Multi(
                    const uint64_t *const keys, uint64_t *const results, const int n,
                    const uint64_t *const all_keys, const uint64_t *const all_vals,
                    const int size, const FuncConfig *const hash_func_configs,
                    const int num_funcs, const int num_buckets, const int pos_width) {
                    int idx = threadIdx.x + blockIdx.x * blockDim.x;

                    // Only threads within range are active.
                    if (idx < n) {
                        uint64_t key = keys[idx];
                        int bucket_num = do_1st_hash(key, num_buckets);
                        for (int i = 0; i < num_funcs; ++i) {
                        int pos = bucket_num * BUCKET_SIZE +
                                    do_2nd_hash(key, hash_func_configs, i, BUCKET_SIZE);
                        if (all_keys[i * size + pos] == key) {
                            results[idx] = all_vals[i * size + pos] + 1;
                            return;
                        }
                        }

                        // TODO(Haotian): should be a value that will not be encountered.
                        results[idx] = EMPTY_CELL;
                    }
                }

                void CuckooHashTableCuda_Multi::lookup_vals(const uint64_t *const keys,
                                                            uint64_t *d_key, uint64_t *d_val,
                                                            uint64_t *const results,
                                                            const int n) {
                    // Launch the lookup kernel.
                    cuckooLookupKernel_Multi<<<ceil((double)n / BUCKET_SIZE), BUCKET_SIZE>>>(
                        keys, results, n, d_key, d_val, _size, _d_hash_func_configs, _num_funcs,
                        _num_buckets, _pos_width);
                    }

                int CuckooHashTableCuda_Multi::insert_vals(const uint64_t *const keys,
                                                            const uint64_t *const vals,
                                                            uint64_t *d_key_buf,
                                                            uint64_t *d_val_buf, uint64_t *d_key,
                                                            uint64_t *d_val, const int n) {
                    //
                    // Phase 1: Distribute keys into buckets.
                    //

                    // Allocate GPU memory.

                    int *d_counters = NULL;

                    cudaMalloc((void **)&d_counters, _num_buckets * sizeof(int));

                    cudaMemset(d_counters, 0, _num_buckets * sizeof(int));

                    // Invoke bucket kernel.
                    cuckooBucketKernel_Multi<<<ceil((double)n / BUCKET_SIZE), BUCKET_SIZE>>>(
                        d_key_buf, d_val_buf, _size, keys, vals, n, d_counters, _num_buckets);

                    //
                    // Phase 2: Local cuckoo hashing.
                    //

                    // Allocate GPU memory.

                    cudaDeviceSynchronize();
                    int *d_rehash_requests = NULL;

                    cudaMalloc((void **)&d_rehash_requests, sizeof(int));

                    // Copy values onto GPU memory.
                    cudaMemcpy(_d_hash_func_configs, _hash_func_configs,
                                _num_funcs * sizeof(FuncConfig), cudaMemcpyHostToDevice);

                    // Invoke insert kernel. Passes shared memory table size by the third
                    // argument. Loops until no rehashing needed.

                    int rehash_count = 0;
                    do {
                        int rehash_requests = 0;
                        cudaMemset(d_rehash_requests, 0, sizeof(int));
                        cuckooInsertKernel_Multi<<<ceil((double)_size / BUCKET_SIZE), BUCKET_SIZE,
                                                _num_funcs * BUCKET_SIZE * sizeof(uint64_t)>>>(
                            d_key, d_val, d_key_buf, d_val_buf, _size, _d_hash_func_configs,
                            _num_funcs, d_counters, _num_buckets, _evict_bound, _pos_width,
                            d_rehash_requests);
                        cudaMemcpy(&rehash_requests, d_rehash_requests, sizeof(int),
                                cudaMemcpyDeviceToHost);

                        if (rehash_requests == 0) {
                        break;
                        } else {
                        rehash_count++;
                        gen_hash_funcs();
                        cudaMemcpy(_d_hash_func_configs, _hash_func_configs,
                                    _num_funcs * sizeof(FuncConfig), cudaMemcpyHostToDevice);
                        }
                    } while (rehash_count < MAX_DEPTH);

                    cudaDeviceSynchronize();

                    // Free GPU resources.

                    if (d_counters != NULL) {
                        cudaFree(d_counters);
                    }
                    if (d_rehash_requests != NULL) {
                        cudaFree(d_rehash_requests);
                    }

                    return (rehash_count < MAX_DEPTH) ? rehash_count : ERR_DEPTH;
                }
            """,
            cuda_src="""
                @alias(hash_query, in0)
                @alias(hash_target, in1)
                @alias(idx_target, in2)

                int n = hash_target_shape0;
                int n1 = hash_query_shape0;
                const int nextPow2 = pow(2, ceil(log2((double)n)));

                // When n is large, the hash values tend to be more evenly distrubuted and
                // choosing table_size to be 2 * nextPow2 typically suffices. For smaller n,
                // the effect of uneven distribution of hash values is more pronounced and
                // hence we choose table_size to be 4 * nextPow2 to reduce the chance of
                // bucket overflow.

                int table_size = (n < 2048) ? 4 * nextPow2 : 2 * nextPow2;
                if (table_size < 512) {
                    table_size = 512;
                }
                int num_funcs = 3;

                CuckooHashTableCuda_Multi in_hash_table(table_size, 8 * ceil(log2((double)n)),
                                                        num_funcs);

                int64 *key_buf_p, *val_buf_p, *key_p, *val_p;

                cudaMalloc((void **)&key_buf_p, table_size * sizeof(int64));
                cudaMalloc((void **)&val_buf_p, table_size * sizeof(int64));
                cudaMalloc((void **)&key_p, num_funcs * table_size * sizeof(int64));
                cudaMalloc((void **)&val_p, num_funcs * table_size * sizeof(int64));

                cudaMemset(key_buf_p, 0, table_size * sizeof(int64));
                cudaMemset(val_buf_p, 0, table_size * sizeof(int64));
                cudaMemset(key_p, 0, num_funcs * table_size * sizeof(int64));
                cudaMemset(val_p, 0, num_funcs * table_size * sizeof(int64));

                in_hash_table.insert_vals((uint64_t *)(hash_target_p),
                                          (uint64_t *)(idx_target_p),
                                          (uint64_t *)(key_buf_p),
                                          (uint64_t *)(val_buf_p),
                                          (uint64_t *)(key_p),
                                          (uint64_t *)(val_p), n);

                cudaMemset(out_p, 0, n1 * sizeof(int64));

                in_hash_table.lookup_vals((uint64_t *)(hash_query_p),
                                          (uint64_t *)(key_p),
                                          (uint64_t *)(val_p),
                                          (uint64_t *)(out_p), n1);

                cudaFree(key_buf_p);
                cudaFree(val_buf_p);
                cudaFree(key_p);
                cudaFree(val_p);
            """
        )

    else:
        output = jt.code(queries.shape, jt.int64, [queries, references, indices],
            cpu_header="""
                #include <cmath>
                #include <google/dense_hash_map>
                #include <iostream>
                #include <vector>
                #include <cstdint>
                #include <cstdio>
                #include <cstdlib>
            """,
            cpu_src="""
                @alias(hash_query, in0)
                @alias(hash_target, in1)
                @alias(idx_target, in2)

                int n = hash_target_shape0;
                int n1 = hash_query_shape0;

                google::dense_hash_map<int64_t, int64_t> hashmap; // Google Sparse Hash library
                hashmap.set_empty_key(0);

                for (int idx = 0; idx < n; ++ idx ) {
                    int64_t key = @hash_target(idx);
                    int64_t val = @idx_target(idx) + 1;
                    hashmap.insert(std::make_pair(key, val));
                } 

                #pragma omp parallel for
                for (int idx = 0; idx < n1; ++ idx ) {
                    int64_t key = @hash_query(idx);
                    auto iter = hashmap.find(key);
                    if (iter != hashmap.end()) {
                        @out(idx) = iter->second;
                    } else @out(idx) = 0;
                }
            """
        )
    output = (output - 1).reshape(*q_size)
    return output
    # return indices
