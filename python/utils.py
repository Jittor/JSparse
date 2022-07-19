from itertools import repeat
from typing import List, Tuple, Union

import jittor as jt

__all__ = ['make_ntuple', 'set_hash', 'unique1d']

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


def unique1d(var):
    assert len(var.shape) == 1
    perm, aux = jt.argsort(var)
    mask = jt.empty(aux.shape, dtype='bool')
    mask[:1] = True
    mask[1:] = aux[1:] != aux[:-1]
    ret = (aux[mask],)
    ret += (perm[mask],)
    imask = jt.cumsum(mask.astype(perm.dtype)) - 1
    inv_idx = jt.empty(mask.shape, dtype=perm.dtype)
    inv_idx[perm] = imask
    ret += (inv_idx,)
    idx = jt.concat([jt.nonzero(mask).view(-1), jt.Var(mask.shape[0])])
    ret += (idx[1:] - idx[:-1],)
    return ret


