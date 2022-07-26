from typing import Callable

import jittor as jt

from python.jsparse import SparseTensor

__all__ = ['fapply']

def fapply(input: SparseTensor, fn: Callable[..., jt.Var], *args,
           **kwargs) -> SparseTensor:
    values = fn(input.values, *args, **kwargs)
    output = SparseTensor(indices=input.indices, values=values, stride=input.stride, size=input.size)
    output.cmaps = input.cmaps
    output.kmaps = input.kmaps
    return output