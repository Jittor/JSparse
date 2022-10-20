import jittor as jt 
import numpy as np 

CUDA_HEADER = r'''
#define THREADS 256

template <typename scalar_t>
__global__ void fps_kernel(const scalar_t *src, const int *ptr,
                           const int *out_ptr, const int *start,
                           scalar_t *dist, int *out, int dim) {

  const int thread_idx = threadIdx.x;
  const int batch_idx = blockIdx.x;

  const int start_idx = ptr[batch_idx];
  const int end_idx = ptr[batch_idx + 1];

  __shared__ scalar_t best_dist[THREADS];
  __shared__ int best_dist_idx[THREADS];

  if (thread_idx == 0) {
    out[out_ptr[batch_idx]] = start_idx + start[batch_idx];
  }

  for (int m = out_ptr[batch_idx] + 1; m < out_ptr[batch_idx + 1]; m++) {
    __syncthreads();
    int old = out[m - 1];

    scalar_t best = (scalar_t)-1.;
    int best_idx = 0;

    for (int n = start_idx + thread_idx; n < end_idx; n += THREADS) {
      scalar_t tmp, dd = (scalar_t)0.;
      for (int d = 0; d < dim; d++) {
        tmp = src[dim * old + d] - src[dim * n + d];
        dd += tmp * tmp;
      }
      dd = min(dist[n], dd);
      dist[n] = dd;
      if (dd > best) {
        best = dd;
        best_idx = n;
      }
    }

    best_dist[thread_idx] = best;
    best_dist_idx[thread_idx] = best_idx;

    for (int i = 1; i < THREADS; i *= 2) {
      __syncthreads();
      if ((thread_idx + i) < THREADS &&
          best_dist[thread_idx] < best_dist[thread_idx + i]) {
        best_dist[thread_idx] = best_dist[thread_idx + i];
        best_dist_idx[thread_idx] = best_dist_idx[thread_idx + i];
      }
    }

    __syncthreads();
    if (thread_idx == 0) {
      out[m] = best_dist_idx[0];
    }
  }
}
'''

def fps(src, batch=None, ratio=None, random_start=True):  # noqa
    r""""A sampling algorithm from the `"PointNet++: Deep Hierarchical Feature
    Learning on Point Sets in a Metric Space"
    <https://arxiv.org/abs/1706.02413>`_ paper, which iteratively samples the
    most distant point with regard to the rest points.

    Args:
        src (Tensor): Point feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        batch (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. (default: :obj:`None`)
        ratio (float or Tensor, optional): Sampling ratio.
            (default: :obj:`0.5`)
        random_start (bool, optional): If set to :obj:`False`, use the first
            node in :math:`\mathbf{X}` as starting node. (default: obj:`True`)

    :rtype: :class:`LongTensor`


    .. code-block:: python

        import torch
        from torch_cluster import fps

        src = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        batch = torch.tensor([0, 0, 0, 0])
        index = fps(src, batch, ratio=0.5)
    """

    if ratio is None:
        ratio = 0.5 

    batch_size = 1
    if batch is not None:
        batch_size = len(batch)
    else:
        batch = jt.array([len(src)],dtype="int32")
    
    if random_start:
        start = (jt.rand((batch_size,))*batch).int32()
    else:
        start = jt.zeros((batch_size,),dtype="int32")
    src = src.flatten(1)
    out_nums = jt.ceil(batch*ratio).int32()
    if jt.flags.use_cuda == 1:
        batch_cum = jt.cumsum(jt.concat([jt.zeros((1,),dtype=batch.dtype),batch],dim=0))
        out_cum = jt.cumsum(jt.concat([jt.zeros((1,),dtype=out_nums.dtype),out_nums],dim=0))
        out_shape = out_cum[-1].item()
        dist = jt.full((src.shape[0],),5e4,dtype=src.dtype)
        out = jt.code(
            shape=(out_shape,),
            dtype="int32",
            inputs=[src,batch_cum,out_cum,start,dist],
            cuda_header=CUDA_HEADER,
            cuda_src=f'''
            const int batch_size = {batch_size};
            '''+r'''
            fps_kernel<float><<<batch_size, THREADS>>>(in0_p, in1_p,in2_p, in3_p,in4_p, out0_p, in0_shape1);
            ''')
    else:
        batch_cum = np.cumsum(jt.concat([jt.zeros((1,),dtype=batch.dtype),batch],dim=0).numpy()).tolist()
        out_cum = np.cumsum(jt.concat([jt.zeros((1,),dtype=out_nums.dtype),out_nums],dim=0).numpy()).tolist()
        out_shape = out_cum[-1]
        out = jt.zeros((out_shape,),dtype="int32")
        for b in range(batch_size):
            src_start = batch_cum[b]
            src_end = batch_cum[b+1]
            y = src[src_start:src_end]
            start_idx = start[b]
            out_start = out_cum[b]
            out_end = out_cum[b+1]

            out[out_start] = src_start+start_idx
            dist = (y-y[start_idx]).sqr().sum(dim=1)
            for i in range(out_start,out_end):
                argmax,_ = jt.argmax(dist,dim=0)
                out[i]=src_start+argmax
                dist = jt.minimum(dist,(y-y[argmax]).sqr().sum(dim=1))
    return out,out_nums


def test_fps():
    import time
    np.random.seed(0)
    src = np.random.randn(100,3).astype(np.float32)
    jt.flags.use_cuda = 1
    # src = jt.array(src)
    # batch = jt.array([30000,70000])
    # for i in range(100):
    #     out,batch_o = fps(src,batch,ratio=0.5,random_start=True)
    #     # print(out[:15],out[15:],batch_o)
    #     out.sync()
    #     batch_o.sync()
    #     print(i)
        
    jt.flags.use_cuda = 1
    src = jt.array(src)
    batch = jt.array([30,70])
    out,batch_o = fps(src,batch,ratio=0.5,random_start=True)
    print(out,batch_o)

    jt.flags.use_cuda = 0
    src = jt.array(src)
    batch = jt.array([30,70])
    out,batch_o = fps(src,batch,ratio=0.5,random_start=True)
    print(out,batch_o)


if __name__ == "__main__":
    test_fps()