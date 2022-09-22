
import jittor as jt
from jittor import nn

from JSparse import SparseTensor
from JSparse.nn.functional import dropout

__all__ = ['Linear', 'Dropout']

class Linear(nn.Linear):

    def execute(self, input, *args, **kwargs):
        values = super().execute(input.values, *args, **kwargs)
        output = SparseTensor(values=values, indices=input.indices, stride=input.stride, size=input.size)
        output.cmaps = input.cmaps
        output.kmaps = input.kmaps
        return output

Dropout = jt.make_module(dropout)
