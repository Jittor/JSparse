from ast import Global
import jittor as jt
from jittor import nn

from JSparse import SparseTensor
from JSparse.nn.functional import global_avg_pool, global_max_pool, max_pool

__all__ = ['GlobalAvgPool', 'GlobalMaxPool', 'MaxPool']

GlobalAvgPool = jt.make_module(global_avg_pool)
GlobalMaxPool = jt.make_module(global_max_pool)
MaxPool = jt.make_module(max_pool)