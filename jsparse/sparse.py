from itertools import count
import numpy as np

import jittor as jt
from jittor.misc import _pair, _triple

from typing import Any, Dict, Tuple, Union

from jsparse.utils import make_ntuple, sparse_quantize, set_hash

class SparseTensor:

    def __init__(
        self, 
        values: jt.Var,
        indices: jt.Var, 
        stride: Union[int, Tuple[int, ...]]=1,
        size=None,
        voxel_size=1,
        quantize=False,
        coalesce_mode:str='sum',
        indice_manager=None,
        device=None,
    ):
        if values.ndim == 1:
            values = values.unsqueeze(1)
        assert values.ndim == 2, values.shape
        if indices.shape[1] == 3:
            indices = jt.concat([jt.zeros((indices.shape[0], 1)), indices], dim=1)
        assert indices.shape[1] == 4 and indices.ndim == 2, indices.shape
        assert indices.shape[0] == values.shape[0]
        self.size = size
        self.ndim = indices.shape[1] - 1
        self.stride = _triple(stride)
        self.voxel_size = voxel_size
        self.coalesce_mode = coalesce_mode
        self.cmaps = {}
        self.kmaps = {}

        ##########################
        # Initialize coords 
        ##########################
        if quantize:
            self.seed = 1
            for i in range(len(self.stride)):
                self.seed += i
                self.seed *= self.stride[i]
            self.hash_multiplier = jt.array([124, 119, 620, 692], dtype='int32')

            self.indices, mapping, inverse_mapping, count = \
                sparse_quantize(indices, self.hash_multiplier, self.voxel_size, return_index=True, return_inverse=True, return_count=True)
            self.inverse_mapping = inverse_mapping

            out_size = (self.indices.shape[0], values.shape[-1])

            if self.coalesce_mode == 'sum':
                self.values = jt.zeros(out_size, dtype=values.dtype).scatter_(0, inverse_mapping, values, reduce='add')
            elif self.coalesce_mode == 'average':
                self.values = jt.zeros(out_size, dtype=values.dtype).scatter_(0, inverse_mapping, values, reduce='add')
                self.values /= count
            elif self.coalesce_mode == 'sample':
                self.values = values[self.indices]
        else:
            self.indices = indices
            self.values = values

        self.indices = self.indices.int32()

    def _indices(self):
        return self.indices
    
    def _values(self):
        return self.values

    def _binary_operation(self, other, _binary_op):
        assert isinstance(other, self.__class__)
        return
        # TODO set up the indices dict
        # so that wedo not need to merge the indice group
        # which has already been merged

        # if the indices of self and other should be merged
    
    def __add__(self, other):
        output = SparseTensor(values=self.values + other.values,
                              indices=self.indices,
                              stride=self.stride)
        output.cmaps = self.cmaps
        output.kmaps = self.kmaps
        return output

        
class PointTensor:

    def __init__(self, values, indices, idx_query=None, weights=None):
        self.values = values
        self.indices = indices
        self.idx_query = idx_query if idx_query is not None else {}
        self.weights = weights if weights is not None else {}
        self.additional_values = {}
        self.additional_values['idx_query'] = {}
        self.additional_values['counts'] = {}

    def __add__(self, other):
        pt = PointTensor(self.values + other.values, self.indices, self.idx_query,
                             self.weights)
        pt.additional_values = self.additional_values
        return pt

        
