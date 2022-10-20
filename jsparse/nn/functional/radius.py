import jittor as jt 
import numpy as np 

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

template <typename scalar_t>
__global__ void
radius_kernel(const scalar_t *__restrict__ x, const scalar_t *__restrict__ y,
              const int *__restrict__ ptr_x,
              const int *__restrict__ ptr_y, int *__restrict__ row,
              int *__restrict__ col, const scalar_t r, const int n,
              const int m, const int dim, const int num_examples,
              const int max_num_neighbors) {

  const int n_y = blockIdx.x * blockDim.x + threadIdx.x;
  if (n_y >= m)
    return;

  int count = 0;
  const int example_idx = get_example_idx(n_y, ptr_y, num_examples);

  for (int n_x = ptr_x[example_idx]; n_x < ptr_x[example_idx + 1]; n_x++) {
    scalar_t dist = 0;
    for (int d = 0; d < dim; d++) {
      dist += (x[n_x * dim + d] - y[n_y * dim + d]) *
              (x[n_x * dim + d] - y[n_y * dim + d]);
    }

    if (dist < r) {
      row[n_y * max_num_neighbors + count] = n_y;
      col[n_y * max_num_neighbors + count] = n_x;
      count++;
    }

    if (count >= max_num_neighbors)
      break;
  }
}
'''

def radius(x, y, r,
           batch_x = None,
           batch_y = None, max_num_neighbors = 32,
           num_workers: int = 1,with_filter=True):
    r"""Finds for each element in :obj:`y` all points in :obj:`x` within
    distance :obj:`r`.

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        y (Tensor): Node feature matrix
            :math:`\mathbf{Y} \in \mathbb{R}^{M \times F}`.
        r (float): The radius.
        batch_x (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. :obj:`batch_x` needs to be sorted.
            (default: :obj:`None`)
        batch_y (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^M`, which assigns each
            node to a specific example. :obj:`batch_y` needs to be sorted.
            (default: :obj:`None`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            return for each element in :obj:`y`.
            If the number of actual neighbors is greater than
            :obj:`max_num_neighbors`, returned neighbors are picked randomly.
            (default: :obj:`32`)
        num_workers (int): Number of workers to use for computation. Has no
            effect in case :obj:`batch_x` or :obj:`batch_y` is not
            :obj:`None`, or the input lies on the GPU. (default: :obj:`1`)

    .. code-block:: python

        import torch
        from torch_cluster import radius

        x = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        batch_x = torch.tensor([0, 0, 0, 0])
        y = torch.Tensor([[-1, 0], [1, 0]])
        batch_y = torch.tensor([0, 0])
        assign_index = radius(x, y, 1.5, batch_x, batch_y)
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
    assert x.ndim==2 and y.ndim == 2 and x.shape[1] == y.shape[1]
    if jt.flags.use_cuda == 1:
        knn_edge1 = jt.zeros((y.shape[0]*max_num_neighbors,),dtype="int32")
        knn_edge2 = -jt.ones((y.shape[0]*max_num_neighbors,),dtype="int32")
        knn_edge1,knn_edge2 = jt.code(
          inputs = [x,y,batch_x,batch_y],
          outputs = [knn_edge1,knn_edge2],
          cuda_header=CUDA_HEADER,
          cuda_src=f"""
          const int k = {max_num_neighbors};
          const float radius = {r};
          const int batch_size = {batch_size};
          """+r'''
          @alias(row,out0);
          @alias(col,out1);
          @alias(x,in0);
          @alias(y,in1);
          @alias(ptr_x,in2);
          @alias(ptr_y,in3);

          dim3 BLOCKS((y_shape0 + THREADS - 1) / THREADS);

          radius_kernel<float><<<BLOCKS, THREADS>>>(
              x_p, y_p,
              ptr_x_p, ptr_y_p,
              row_p, col_p, radius*radius, x_shape0,
              y_shape0, x_shape1, batch_size, k);
        ''')
        # print(r,(knn_edge2>=0).float32().mean(),(knn_edge2>=0).sum(),knn_edge2.numel())
        if with_filter:
            mask = knn_edge2>=0
            knn_edge = jt.stack([knn_edge1[mask],knn_edge2[mask]],dim=0)
            return knn_edge
        else:
            return knn_edge2.reshape(-1,max_num_neighbors)
    else:
        assert False

def test_radius():
    jt.flags.use_cuda=1
    # x = jt.rand((10,3))
    # y = jt.rand((10,3))
    # edge = knn(x,y,k=5)
    # print(edge)
    # dist = jt.norm(y[:,None,:]-x[None,:,:],dim=-1)
    # a = jt.argsort(dist,dim=1)[0][:,:5]
    # print(a)

    import time 
    x = jt.rand((100000,3))
    y = jt.rand((100000,3))
    s = time.time()
    for i in range(100):
        edge = radius(x,y,r=0.01)
        edge.sync()
    jt.sync_all(True)
    print((time.time()-s)*10)
    # print(edge)


if __name__ == "__main__":
    test_radius()