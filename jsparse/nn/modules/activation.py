import jittor as jt
from jittor import nn

from jsparse.nn.functional import relu, leaky_relu

__all__ = ['ReLU', 'LeakyReLU']

Relu = jt.make_module(relu)
ReLU = Relu
Leaky_relu = jt.make_module(leaky_relu, 2)
LeakyReLU = Leaky_relu