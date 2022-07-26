import jittor as jt
from jittor import nn

from python.jsparse import SparseTensor
from python.jsparse.nn.functional import relu, leaky_relu
# from python.nn.utils import fapply

__all__ = ['ReLU', 'LeakyReLU']

# class ReLU(nn.ReLU):
#     def execute(self, input: SparseTensor) -> SparseTensor:
#         return fapply(input, super().execute)

# class LeakyReLU(nn.LeakyReLU):
#     def execute(self, input: SparseTensor) -> SparseTensor:
#         return fapply(input, super().execute)

Relu = jt.make_module(relu)
ReLU = Relu
Leaky_relu = jt.make_module(leaky_relu, 2)
LeakyReLU = Leaky_relu