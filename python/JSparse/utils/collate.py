from typing import Any, List

import numpy as np
import jittor

from JSparse import SparseTensor

__all__ = ['sparse_collate', 'sparse_collate_fn']

def sparse_collate(inputs: List[SparseTensor]) -> SparseTensor:
    indices, values = [], []
    stride = inputs[0].stride

    for k, x in enumerate(inputs):
        if isinstance(x.indices, np.ndarray):
            x.indices = jittor.array(x.indices)
        if isinstance(x.values, np.ndarray):
            x.values = jittor.array(x.values)

        assert isinstance(x.indices, jittor.Var), type(x.indices)
        assert isinstance(x.values, jittor.Var), type(x.values)
        assert x.stride == stride

        input_size = x.indices.shape[0]
        batch = jittor.full((input_size, 1), k, dtype=jittor.int64)

        indices.append(jittor.concat((batch, x.indices[:, 1:]), dim=1))
        values.append(x.values)

    indices = jittor.concat(indices, dim=0)
    values = jittor.concat(values, dim=0)
    output = SparseTensor(values=values, indices=indices, stride=stride)
    return output


def sparse_collate_fn(inputs: List[Any]) -> Any:
    if isinstance(inputs[0], dict):
        output = {}
        for name in inputs[0].keys():
            if isinstance(inputs[0][name], dict):
                output[name] = sparse_collate_fn([input[name] for input in inputs])
            elif isinstance(inputs[0][name], np.ndarray):
                output[name] = jittor.stack([jittor.Var(input[name]) for input in inputs], dim=0)
            elif isinstance(inputs[0][name], jittor.Var):
                output[name] = jittor.stack([input[name] for input in inputs], dim=0)
            elif isinstance(inputs[0][name], SparseTensor):
                output[name] = sparse_collate([input[name] for input in inputs])
            else:
                output[name] = [input[name] for input in inputs]
        return output
    else:
        return inputs
