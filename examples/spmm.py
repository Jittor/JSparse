
from email.header import decode_header
import time
import torch
import numpy as np

import jittor as jt
import jsparse.nn.functional as F

jt.flags.nvcc_flags += '-lcusparse'

jt.flags.use_cuda = 1


dense_data = np.array([[0, 1, 0, 0], [2, 0, 3, 0], [0, 0, 0, 4], [0, 0, 5, 0]])
size = (1,512)
dense_data = np.random.rand(*size).astype(np.float32)
dense_data = np.where(dense_data > 0.5, dense_data, 0)
mat = np.random.rand(512,256).astype(np.float32)
indices = np.nonzero(dense_data)
values = dense_data[indices]
output = F.spmm(
        rows=jt.array(indices[0]), 
        cols=jt.array(indices[1]), 
        vals=jt.array(values), 
        size=size, 
        mat=jt.array(mat), 
        cuda_spmm_alg=1)

print(output)