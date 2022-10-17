import jittor as jt
import jittor.nn as nn

from jsparse import SparseTensor
from jsparse.nn.utils import fapply

__all__ = ['dropout']

def dropout(input: SparseTensor, *args, **kwargs):
    return fapply(input, nn.dropout, *args, **kwargs)