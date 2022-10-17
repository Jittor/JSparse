import jittor as jt
import jittor.nn as nn

from jsparse import SparseTensor
from jsparse.nn.utils import fapply

__all__ = ['relu', 'leaky_relu']

def relu(input: SparseTensor) -> SparseTensor:
    return fapply(input, nn.relu)


def leaky_relu(input: SparseTensor,
               scale: float = 0.01) -> SparseTensor:
    return fapply(input, nn.leaky_relu, scale=scale)