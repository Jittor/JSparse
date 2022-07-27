from itertools import repeat
from typing import List, Tuple, Union

import jittor as jt
import numpy as np

from .utils import unique1d

__all__ = ['sparse_quantize', 'set_hash']

def set_hash(ndim, seed, low=100, high=1000):
    jt.set_seed(seed)
    return jt.randint(low, high, shape=(ndim + 1,), dtype='uint64')

def hash(x: np.ndarray, multiplier: np.ndarray) -> jt.Var:
    assert x.ndim == 2, x.shape

    x = x - x.min(dim=0)
    x = x.uint64()

    h = jt.zeros(x.shape[0], dtype='uint64')
    for k in range(x.shape[1] - 1):
        h += x[:, k]
        h *= multiplier[k]
    h += x[:, -1]
    return h

def sparse_quantize(indices,
                    hash_multiplier,
                    voxel_size: Union[float, Tuple[float, ...]] = 1,
                    *,
                    return_index: bool = False,
                    return_inverse: bool = False,
                    return_count: bool = False) -> List[np.ndarray]:
    if indices.dtype.is_int() and voxel_size == 1:
        pass
    else:
        if isinstance(voxel_size, (float, int)):
            voxel_size = tuple(repeat(voxel_size, 3))
        assert isinstance(voxel_size, tuple) and len(voxel_size) == 3

        voxel_size = jt.Var(voxel_size)
        indices[:, 1:] /= voxel_size
        indices = jt.floor(indices).astype(jt.int32)

    hash_num, mapping, inverse_mapping, count = unique1d(hash(indices, hash_multiplier))
    indices = indices[mapping]

    outputs = [hash_num, indices]
    if return_index:
        outputs += [mapping]
    if return_inverse:
        outputs += [inverse_mapping]
    if return_count:
        outputs += [count]
    return outputs[0] if len(outputs) == 1 else outputs


