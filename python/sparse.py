from itertools import count
import jittor as jt
import numpy as np

from typing import Any, Dict, Tuple, Union

from utils import make_ntuple
from quantize import sparse_quantize

class SparseTensor:
    def __init__(
        self, 
        indices: jt.Var, 
        values: jt.Var,
        stride: Union[int, Tuple[int, ...], jt.NanoVector],
        size: jt.NanoVector,
        voxel_size=1,
        coalesce_mode='sum',
        indice_manager=None,
        device=None,
    ):
        assert isinstance(indices, jt.Var) and isinstance(values, jt.Var) and isinstance(size, jt.NanoVector)
        assert (values.ndim == 2)
        # self.indices = indices
        # self.values = values
        self.shape = size
        self.ndim = self.indices.shape[1] - 1
        self.stride = make_ntuple(stride, ndim=self.ndim)
        self.voxel_size = voxel_size
        self.coalesce_mode = coalesce_mode

        ##########################
        # Initialize coords 
        ##########################
        self.seed = 1
        for i in range(len(stride)):
            self.seed += i
            self.seed *= stride[i]
        self.hash_multiplier = jt.set_seed(self.seed)

        self.indices, mapping, inverse_mapping = sparse_quantize(indices, self.hash_multiplier, self.voxel_size, return_index=True, return_inverse=True)
        self.inverse_mapping = inverse_mapping
        if self.coalesce_mode == 'sum':
            out_size = (self.indices.shape[0], values.shape[-1])
            self.values = jt.zeros(out_size, dtype=values.dtype).scatter_(0, inverse_mapping, values, reduce='add')
        elif self.coalesce_mode == 'average':
            out_size = (self.indices.shape[0], values.shape[-1])
            self.values = jt.zeros(out_size, dtype=values.dtype).scatter_(0, inverse_mapping, values, reduce='add')
            self.values /= count
        elif self.coalesce_mode == 'sample':
            self.values = values[self.indices]

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
        # TODO set up the indices dict
        # so that wedo not need to merge the indice group
        # which has already been merged

        # 

        
