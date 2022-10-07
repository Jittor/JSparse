# JSparse

## Introduction

JSparse is a high-performance auto-differentiation library for sparse voxels computation and point cloud processing based on [TorchSparse](https://github.com/mit-han-lab/torchsparse) and [Jittor](https://github.com/Jittor/jittor).

## Installation

If you use cpu version, you need to install [Google Sparse Hash](https://github.com/sparsehash/sparsehash).

The latest JSparse can be installed by 

```bash
cd python
python setup.py install
```

## Getting Started

### Architecture

```
- JSparse
    - nn
        - functional
        - modules
    - utils
        collate/quantize/utils.py
```

You can use the modules from `JSparse/modules` .

### Sparse Tensor

Sparse tensor (`SparseTensor`) is the main data structure for point cloud, which has two data fields:

- Coordinates (`indices`): a 2D integer tensor with a shape of $N \times 4$, where the first dimension denotes the batch index, and the last three dimensions correspond to quantized $x, y, z$ coordinates.


- Features (`values`): a 2D tensor with a shape of $N \times C$, where $C$ is the number of feature channels.
Most existing datasets provide raw point cloud data with float coordinates. We can use `sparse_quantize` (provided in `JSparse.utils.quantize`) to voxelize $x, y, z$ coordinates and remove duplicates.

    You can also use the initialization method to automatically obtain the discretized features by turning on `quantize` option

    ```python
    inputs = SparseTensor(values=feats, indices=coords, voxel_size=self.voxel_size, quantize=True)
    ```
We can then use `sparse_collate_fn` (provided in `JSparse.utils.collate`) to assemble a batch of `SparseTensor`'s (and add the batch dimension to coords). Please refer to this example for more details.

### Sparse Neural Network

We finished many common modules in `JSparse.nn` such like `MaxPool`, `GlobalMaxPool`. 

The neural network interface in JSparse is similar to Jittor:

```python
import JSparse.nn as spnn
def get_conv_block(self, in_channel, out_channel, kernel_size, stride):
    return nn.Sequential(
        spnn.Conv3d(
            in_channel,
            out_channel,
            kernel_size=kernel_size,
            stride=stride,
        ),
        spnn.BatchNorm(out_channel),
        spnn.LeakyReLU(),
    )
```

You can get the usage of most of the functions and modules from the example `examples/MinkNet/classification_model40.py`.

## BenchMark

We test several networks between JSparse(v0.5.0) and TorchSparse(v1.4.0).

Because the Jittor framework is fast, inference and training are faster than PyTorch on many operators.

For `scatter` and `gather` operations in sparse convolution, we rewrite the operator that adjusts the algorithm strategy according to the number of channels, and achieves better efficiency.

We test the speed on the following model and choose the scenes from ScanNet as well.

```python
model = nn.Sequential(
    spnn.Conv3d(3, 32, 3),
    spnn.BatchNorm(32),
    spnn.ReLU(),
    spnn.Conv3d(32, 64, 3, stride=2),
    spnn.BatchNorm(64),
    spnn.ReLU(),
    spnn.Conv3d(64, 128, 3, stride=2),
    spnn.BatchNorm(128),
    spnn.ReLU(),
    spnn.Conv3d(128, 256, 3, stride=2),
    spnn.BatchNorm(256),
    spnn.ReLU(),
    spnn.Conv3d(256, 128, 3, stride=2, transposed=True),
    spnn.BatchNorm(128),
    spnn.ReLU(),
    spnn.Conv3d(128, 64, 3, stride=2, transposed=True),
    spnn.BatchNorm(64),
    spnn.ReLU(),
    spnn.Conv3d(64, 32, 3, stride=2, transposed=True),
    spnn.BatchNorm(32),
    spnn.ReLU(),
    spnn.Conv3d(32, 3, 1),
)
``` 

We finnished two versions of SparseConv(based on jittor operators and based on cuda).

We choose `batch_size = 1, input_size = 1` on RTX3080 to test every iteration's speed.

|                   | Jsparse(jittor) | Jsparse(cuda) | Torchsparse(v.1.4.0) |
|-------------------|-----------------|---------------|----------------------|
| voxel_size = 0.10 | 50.01ms         | 40.02ms       | 47.21ms              |
| voxel_size = 0.05 | 58.44ms         | 48.01ms       | 50.78ms              |
| voxel_size = 0.03 | 69.60ms         | 61.37ms       | 53.09ms              |

## Acknowledgements

The implementation and idea of JSparse refers to many open source libraries, including(but not limited to) [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine) and [TorchSparse](https://github.com/mit-han-lab/torchsparse).