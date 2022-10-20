import jittor as jt 
CUDA_HEADER = r'''
#define THREADS 256
__forceinline__ __device__ int get_example_idx(int idx,
                                                   const int *ptr,
                                                   const int num_examples) {
  for (int i = 0; i < num_examples; i++) {
    if (ptr[i + 1] > idx)
      return i;
  }
  return num_examples - 1;
}

template <typename scalar_t> struct Cosine {
  static inline __device__ scalar_t dot(const scalar_t *a, const scalar_t *b,
                                        int n_a, int n_b,
                                        int size) {
    scalar_t result = 0;
    for (int i = 0; i < size; i++) {
      result += a[n_a * size + i] * b[n_b * size + i];
    }
    return result;
  }

  static inline __device__ scalar_t norm(const scalar_t *a, int n_a,
                                         int size) {
    scalar_t result = 0;
    for (int i = 0; i < size; i++) {
      result += a[n_a * size + i] * a[n_a * size + i];
    }
    return sqrt(result);
  }
};

template <typename scalar_t>
__global__ void
knn_kernel(const scalar_t *__restrict__ x, const scalar_t *__restrict__ y,
           const int *__restrict__ ptr_x, const int *__restrict__ ptr_y,
           int *__restrict__ row, int *__restrict__ col,
           const int k, const int n, const int m, const int dim,
           const int num_examples, const bool cosine) {

  const int n_y = blockIdx.x * blockDim.x + threadIdx.x;
  if (n_y >= m)
    return;

  const int example_idx = get_example_idx(n_y, ptr_y, num_examples);

  scalar_t best_dist[100];
  int best_idx[100];

  for (int e = 0; e < k; e++) {
    best_dist[e] = 1e10;
    best_idx[e] = -1;
  }

  for (int n_x = ptr_x[example_idx]; n_x < ptr_x[example_idx + 1]; n_x++) {
    scalar_t tmp_dist = 0;

    if (cosine) {
      tmp_dist = Cosine<scalar_t>::dot(x, y, n_x, n_y, dim) /
                 (Cosine<scalar_t>::norm(x, n_x, dim) *
                  Cosine<scalar_t>::norm(y, n_y, dim));
      tmp_dist = 1. - tmp_dist;
    } else {
      for (int d = 0; d < dim; d++) {
        tmp_dist += (x[n_x * dim + d] - y[n_y * dim + d]) *
                    (x[n_x * dim + d] - y[n_y * dim + d]);
      }
    }

    for (int e1 = 0; e1 < k; e1++) {
      if (best_dist[e1] > tmp_dist) {
        for (int e2 = k - 1; e2 > e1; e2--) {
          best_dist[e2] = best_dist[e2 - 1];
          best_idx[e2] = best_idx[e2 - 1];
        }
        best_dist[e1] = tmp_dist;
        best_idx[e1] = n_x;
        break;
      }
    }
  }

  for (int e = 0; e < k; e++) {
    row[n_y * k + e] = n_y;
    col[n_y * k + e] = best_idx[e];
  }
}
'''


CPU_SRC=r'''
  @alias(x,in0);
  @alias(y,in1);
  std::vector<size_t> out_vec = std::vector<size_t>();
  typedef std::vector<std::vector<float>> vec_t;

    if (!ptr_x.has_value()) { // Single example.

      vec_t pts(x_shape0);
      for (int64_t i = 0; i < x_shape0; i++) {
        pts[i].resize(x_shape1);
        for (int64_t j = 0; j < x_shape1; j++) {
          pts[i][j] = x_p[i * x_shape1 + j];
        }
      }

      typedef KDTreeVectorOfVectorsAdaptor<vec_t, float> my_kd_tree_t;

      my_kd_tree_t mat_index(x_shape1, pts, 10);
      mat_index.index->buildIndex();

      std::vector<size_t> ret_index(k);
      std::vector<float> out_dist_sqr(k);
      for (int64_t i = 0; i < y.size(0); i++) {
        size_t num_matches = mat_index.index->knnSearch(
            y_data + i * y.size(1), k, &ret_index[0], &out_dist_sqr[0]);

        for (size_t j = 0; j < num_matches; j++) {
          out_vec.push_back(ret_index[j]);
          out_vec.push_back(i);
        }
      }
    } else { // Batch-wise.

      auto ptr_x_data = ptr_x.value().data_ptr<int64_t>();
      auto ptr_y_data = ptr_y.value().data_ptr<int64_t>();

      for (int64_t b = 0; b < ptr_x.value().size(0) - 1; b++) {
        auto x_start = ptr_x_data[b], x_end = ptr_x_data[b + 1];
        auto y_start = ptr_y_data[b], y_end = ptr_y_data[b + 1];

        if (x_start == x_end || y_start == y_end)
          continue;

        vec_t pts(x_end - x_start);
        for (int64_t i = 0; i < x_end - x_start; i++) {
          pts[i].resize(x.size(1));
          for (int64_t j = 0; j < x.size(1); j++) {
            pts[i][j] = x_data[(i + x_start) * x.size(1) + j];
          }
        }

        typedef KDTreeVectorOfVectorsAdaptor<vec_t, scalar_t> my_kd_tree_t;

        my_kd_tree_t mat_index(x.size(1), pts, 10);
        mat_index.index->buildIndex();

        std::vector<size_t> ret_index(k);
        std::vector<scalar_t> out_dist_sqr(k);
        for (int64_t i = y_start; i < y_end; i++) {
          size_t num_matches = mat_index.index->knnSearch(
              y_data + i * y.size(1), k, &ret_index[0], &out_dist_sqr[0]);

          for (size_t j = 0; j < num_matches; j++) {
            out_vec.push_back(x_start + ret_index[j]);
            out_vec.push_back(i);
          }
        }
      }
    }
  });

  const int64_t size = out_vec.size() / 2;
  auto out = torch::from_blob(out_vec.data(), {size, 2},
                              x.options().dtype(torch::kLong));
  return out.t().index_select(0, torch::tensor({1, 0}));
}
'''
def knn(x, y, k,
        batch_x = None,
        batch_y = None, cosine = False,
        num_workers = 1):
    r"""Finds for each element in :obj:`y` the :obj:`k` nearest points in
    :obj:`x`.

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        y (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{M \times F}`.
        k (int): The number of neighbors.
        batch_x (LongTensor, optional): Batch vector
            (default: :obj:`None`)
        batch_y (LongTensor, optional): Batch vector
            size is batch num
            (default: :obj:`None`)
        cosine (boolean, optional): If :obj:`True`, will use the Cosine
            distance instead of the Euclidean distance to find nearest
            neighbors. (default: :obj:`False`)
        num_workers (int): Number of workers to use for computation. Has no
            effect in case :obj:`batch_x` or :obj:`batch_y` is not
            :obj:`None`, or the input lies on the GPU. (default: :obj:`1`)

    :rtype: :class:`LongTensor`
    """

    x = x.reshape(-1, 1) if x.ndim == 1 else x
    y = y.reshape(-1, 1) if y.ndim == 1 else y

    batch_size = 1
    if batch_x is not None:
        batch_size = len(batch_x)
    else:
        batch_x = jt.array([len(x)],dtype="int32")
    if batch_y is not None:
        assert batch_size == len(batch_y)
    else:
        batch_y = jt.array([len(y)],dtype="int32")

    batch_x = jt.cumsum(jt.concat([jt.zeros((1,),dtype=batch_x.dtype),batch_x],dim=0))
    batch_y = jt.cumsum(jt.concat([jt.zeros((1,),dtype=batch_y.dtype),batch_y],dim=0))

    assert jt.flags.use_cuda == 1 or not cosine, "Cosine not support cpu"
    assert x.ndim==2 and y.ndim == 2 and x.shape[1] == y.shape[1]
    if jt.flags.use_cuda == 1:
        knn_edge1 = jt.zeros((y.shape[0]*k,),dtype="int32")
        knn_edge2 = -jt.ones((y.shape[0]*k,),dtype="int32")
        
        knn_edge1,knn_edge2 = jt.code(
          inputs = [x,y,batch_x,batch_y],
          outputs = [knn_edge1,knn_edge2],
          cuda_header=CUDA_HEADER,
          cuda_src=f"""
          const int k = {k};
          const bool cosine = {'true' if cosine else 'false'};
          const int batch_size = {batch_size};
          """+r'''
          @alias(row,out0);
          @alias(col,out1);
          @alias(x,in0);
          @alias(y,in1);
          @alias(ptr_x,in2);
          @alias(ptr_y,in3);

          dim3 BLOCKS((y_shape0 + THREADS - 1) / THREADS);

          knn_kernel<float><<<BLOCKS, THREADS>>>(
              x_p, y_p,
              ptr_x_p, ptr_y_p,
              row_p, col_p, k, x_shape0,
              y_shape0, x_shape1, batch_size, cosine);
        ''')
        mask = knn_edge2>=0
        knn_edge = jt.stack([knn_edge1[mask],knn_edge2[mask]],dim=0)

    return knn_edge

def test_knn():
    jt.flags.use_cuda=1
    # x = jt.rand((10,3))
    # y = jt.rand((10,3))
    # edge = knn(x,y,k=5)
    # print(edge)
    # dist = jt.norm(y[:,None,:]-x[None,:,:],dim=-1)
    # a = jt.argsort(dist,dim=1)[0][:,:5]
    # print(a)

    x = jt.rand((100000,3))
    y = jt.rand((100000,3))
    for i in range(100):
      edge = knn(x,y,k=50)
      edge.sync()
      print(i)
    jt.sync_all(True)
    # print(edge)


if __name__ == "__main__":
    test_knn()