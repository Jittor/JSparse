from itertools import repeat
from typing import List, Tuple, Union

import jittor as jt

__all__ = ['make_ntuple', 'trunc', 'unique1d']

def make_ntuple(x: Union[int, List[int], Tuple[int, ...], jt.Var],
                ndim: int) -> Tuple[int, ...]:
    if isinstance(x, int):
        x = tuple(repeat(x, ndim))
    elif isinstance(x, list):
        x = tuple(x)
    elif isinstance(x, jt.Var):
        x = tuple(x.view(-1).numpy().tolist())

    assert isinstance(x, tuple) and len(x) == ndim, x
    return x

def trunc(x: jt.Var):
    return jt.floor(x) * (x >= 0) + jt.ceil(x) * (x < 0)

def unique1d(var, inverse_mapping=False, count=False):
    assert len(var.shape) == 1
    perm, aux = jt.argsort(var)
    mask = jt.empty(aux.shape, dtype='bool')
    mask[:1] = True
    mask[1:] = aux[1:] != aux[:-1]
    ret = (perm[mask],)
    if inverse_mapping:
        imask = jt.cumsum(mask.astype(perm.dtype)) - 1
        inv_idx = jt.empty(mask.shape, dtype=perm.dtype)
        inv_idx[perm] = imask
        ret += (inv_idx,)
    if count:
        idx = jt.concat([jt.nonzero(mask).view(-1), jt.array(mask.shape[0])])
        ret += (idx[1:] - idx[:-1],)
    return ret


