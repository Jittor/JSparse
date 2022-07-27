import time
import math
import numpy as np
import torch

import jittor as jt
import jittor.nn as nn
from jittor import init
from jittor.misc import _pair, _triple

from itertools import repeat
from typing import List, Tuple, Union

from JSparse import SparseTensor
from JSparse import PointTensor
from JSparse.utils import make_ntuple
from JSparse.nn import functional as F
from JSparse.nn.utils import get_kernel_offsets
from JSparse.nn.functional import Convolution

import torchsparse
from torchsparse import nn as spnn

jt.flags.use_cuda = 1

in_channels = 3
out_channels = 64
kernel_size = 3
stride = 1
dilation = 1
groups = 1
bias = False
transposed = False

kernel_size = _triple(kernel_size)
stride = _triple(stride)
dilation = _triple(dilation)
kernel_volume = int(np.prod(kernel_size))

N = 10
coords = np.random.uniform(0, 10, size=(N, 4))
feats = np.random.randn(N, 3)
labels = np.random.choice(5, N)
print(coords.shape)
print(feats.shape)

coo = jt.Var(coords)
val = jt.Var(feats)
size = (10, 10, 10)

fan = (out_channels if transposed else in_channels) * kernel_volume
std = 1 / math.sqrt(fan)
    
if kernel_volume > 1:
    weight = init.uniform([kernel_volume, in_channels, out_channels], 'float32', -std, std)
else:
    weight = init.uniform([in_channels, out_channels], 'float32')
if bias:
    bias = init.uniform([out_channels], "float32", -std, std)
else:
    bias = None


x = SparseTensor(coo, val, 1, size)
z = PointTensor(x.values, x.indices.float())

pc_hash = F.sphash(
    jt.concat([
        z.indices[:, 0].int().view(-1, 1),
        jt.floor(z.indices[:, 1:] / x.stride[0]).int() * x.stride[0]
    ], 1))
sparse_hash = F.sphash(x.indices)
idx_query = F.spquery(pc_hash, sparse_hash).int()
counts = F.spcount(idx_query, x.indices.shape[0])
z.additional_values['idx_query'][x.stride] = idx_query
z.additional_values['counts'][x.stride] = counts
inserted_values = F.spvoxelize(z.values, idx_query, counts)
new_tensor = SparseTensor(inserted_values, x.indices, x.stride, x.size, False)
new_tensor.cmaps = x.cmaps
new_tensor.kmaps = x.kmaps
print(inserted_values)

offsets = get_kernel_offsets(kernel_size=2, stride=x.stride, dilation=1)
old_hash = F.sphash(
    jt.concat([
        z.indices[:, 0].int().view(-1, 1),
        jt.floor(z.indices[:, 1:] / x.stride[0]).int() * x.stride[0]
    ], 1), offsets)
pc_hash = F.sphash(x.indices)
idx_query = F.spquery(old_hash, pc_hash).int()
weights = F.calc_ti_weights(z.indices, idx_query,
                            scale=x.stride[0]).t()
idx_query = idx_query.t()
new_values = F.spdevoxelize(x.values, idx_query, weights)

print(jt.grad(new_values, x.values))