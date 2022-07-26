import os
import numpy as np
from typing import Union, List, Tuple

import jittor as jt
from jittor import Function

class IndiceManager:
    def __init__(
        self,
        ndim,
        # indice_map_type,
        # sparse_alorithm, # set m_hashtable_occupancy for concurrent_unordered_map 
    ):
        # if indice_map_type == 'GPU':
        #     assert(jt.has_cuda)

        self.ndim = ndim
        # self.indice_map_type = indice_map_type
        # self.sparse_algorithm = sparse_alorithm
        self.stride_key_manager = {}
        self.indice_map_manager = {}
        self.kernel_map_manager = {}
    
    def insert(self, stride, indice_key, indice_hash):
        self.stride_key_manager[stride] = indice_key
        self.indice_map_manager[indice_key] = indice_hash


# class IndiceMapManager
        
        

        