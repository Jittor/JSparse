import jittor as jt
from jittor import nn
from numpy import kaiser

from jsparse import SparseTensor
from jsparse.nn.utils import fapply

__all__ = ['BatchNorm', 'GroupNorm']

class BatchNorm(nn.BatchNorm):
    def execute(self, input: SparseTensor) -> SparseTensor:
        return fapply(input, super().execute)

class GroupNorm(nn.GroupNorm):
    def execute(self, input: SparseTensor) -> SparseTensor:
        indices, values, stride, size = input.indices, input.values, input.stride, input.size

        batch_size = jt.max(indices[:, 0]).item() + 1
        num_channels = values.shape[1]

        n_values = jt.zeros_like(values)
        for k in range(batch_size):
            idx = indices[:, 0] == k
            b_values = values[idx]
            b_values = b_values.t().reshape(1, num_channels, -1)
            b_values = super().execute(b_values)
            b_values = b_values.reshape(num_channels, -1).t()
            n_values[idx] = b_values
        
        output = SparseTensor(values=n_values, indices=indices, stride=stride, size=size)
        output.cmaps = input.cmaps
        output.kmaps = input.kmaps
        return output

