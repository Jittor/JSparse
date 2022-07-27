from itertools import count
import numpy as np

import jittor as jt
from jittor.misc import _pair, _triple

from typing import Any, Dict, Tuple, Union
from type_check import type_check

from JSparse.utils import make_ntuple, sparse_quantize, set_hash
# from .utils.quantize import sparse_quantize
# from indice_manager import IndiceManager

class SparseTensor:

    @type_check
    def __init__(
        self, 
        indices: jt.Var, 
        values: jt.Var,
        stride: Union[int, Tuple[int, ...]],
        size=None,
        quantize=True,
        voxel_size=1,
        coalesce_mode:str='sum',
        indice_manager=None,
        device=None,
    ):
        assert (values.ndim == 2)
        # self.indices = indices
        # self.values = values
        self.size = size
        self.ndim = indices.shape[1] - 1
        self.stride =_triple(stride)
        self.voxel_size = voxel_size
        self.coalesce_mode = coalesce_mode
        self.cmaps = {}
        self.kmaps = {}

        ##########################
        # Setup CoordsManager
        ##########################
        # if indice_manager is None:
        #     self.indice_manager = IndiceManager(
        #         ndim=self.ndim,

        #     )

        ##########################
        # Initialize coords 
        ##########################
        if quantize:
            self.seed = 1
            for i in range(len(self.stride)):
                self.seed += i
                self.seed *= self.stride[i]
            self.hash_multiplier = set_hash(self.ndim, self.seed)

            self.hash_num, self.indices, mapping, inverse_mapping, count = sparse_quantize(indices, self.hash_multiplier, self.voxel_size, return_index=True, return_inverse=True, return_count=True)
            self.inverse_mapping = inverse_mapping

            if len(values.shape) == 1:
                out_size = (self.indices.shape[0], )
            elif len(values.shape) == 2:
                out_size = (self.indices.shape[0], values.shape[-1])

            if self.coalesce_mode == 'sum':
                out_size = (self.indices.shape[0], values.shape[-1])
                self.values = jt.zeros(out_size, dtype=values.dtype).scatter_(0, inverse_mapping, values, reduce='add')
            elif self.coalesce_mode == 'average':
                out_size = (self.indices.shape[0], values.shape[-1])
                self.values = jt.zeros(out_size, dtype=values.dtype).scatter_(0, inverse_mapping, values, reduce='add')
                self.values /= count
            elif self.coalesce_mode == 'sample':
                self.values = values[self.indices]
        else:
            self.indices = indices
            self.values = values

        # if indice_manager is None:
        #     # TODO If set to share the indices man, use the global indices man

        #     # init the indices
        #     indice_manager = Indice


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

        
class PointTensor:

    def __init__(self, values, indices, idx_query=None, weights=None):
        self.values = values
        self.indices = indices
        self.idx_query = idx_query if idx_query is not None else {}
        self.weights = weights if weights is not None else {}
        self.additional_values = {}
        self.additional_values['idx_query'] = {}
        self.additional_values['counts'] = {}

    def detach(self):
        self.values = self.values.detach()
        self.indices = self.indices.detach()
        return self

    def __add__(self, other):
        pt = PointTensor(self.values + other.values, self.indices, self.idx_query,
                             self.weights)
        pt.additional_values = self.additional_values
        return pt

        