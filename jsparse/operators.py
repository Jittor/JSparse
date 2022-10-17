from typing import List

import jittor as jt

from jsparse import SparseTensor
from numpy import indices

__all__ = ['cat']

def cat(inputs: List[SparseTensor]) -> SparseTensor:
    values = jt.concat([input.values for input in inputs], dim=1)
    output = SparseTensor(values=values,
                          indices=inputs[0].indices,
                          stride=inputs[0].stride)
    output.cmaps = inputs[0].cmaps
    output.kmaps = inputs[0].kmaps
    return output
