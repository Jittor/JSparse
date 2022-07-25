from ast import Global
import jittor as jt
from jittor import nn

from python import SparseTensor
from python.nn.functional import global_avg_pool, global_max_pool

__all__ = ['GlobalAvgPool', 'GlobalMaxPool']

GlobalAvgPool = jt.make_module(global_avg_pool)
GlobalMaxPool = jt.make_module(global_max_pool)