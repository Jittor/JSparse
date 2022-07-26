import jittor as jt

from python.jsparse import SparseTensor

__all__ = ['global_avg_pool', 'global_max_pool']

def global_avg_pool(inputs: SparseTensor) -> jt.Var:
    batch_size = jt.max(inputs.indices[:, 0]).item() + 1
    outputs = []
    for k in range(batch_size):
        input = inputs.values[inputs.indices[:, 0] == k]
        output = jt.mean(input, dim=0)
        outputs.append(output)
    outputs = jt.stack(outputs, dim=0)
    return outputs


def global_max_pool(inputs: SparseTensor) -> jt.Var:
    batch_size = jt.max(inputs.indices[:, 0]).item() + 1
    outputs = []
    for k in range(batch_size):
        input = inputs.values[inputs.indices[:, 0] == k]
        output = jt.max(input, dim=0)[0]
        outputs.append(output)
    outputs = jt.stack(outputs, dim=0)
    return outputs