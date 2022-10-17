from ast import Global
import jittor as jt
from jittor import nn

from jsparse import SparseTensor
from jsparse.nn.functional import max_pool

MaxPool = jt.make_module(max_pool)

class GlobalPool(nn.Module):
    def __init__(self,op="max"):
        super().__init__()
        self.op = op

    def execute(self,x):
        indices = x.indices[:,0]
        values = x.values
        B = indices.max().item()+1
        values = values.reindex_reduce(op=self.op,
            shape = [B,values.shape[1]],
            indexes = ["@e0(i0)","i1"],
            extras=[indices])
        return values