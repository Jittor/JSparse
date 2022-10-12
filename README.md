# JSparse

## Introduction

JSparse is a high-performance auto-differentiation library for sparse voxels computation and point cloud processing based on [TorchSparse](https://github.com/mit-han-lab/torchsparse) and [Jittor](https://github.com/Jittor/jittor).

## Installation

If you use cpu version, you need to install [Google Sparse Hash](https://github.com/sparsehash/sparsehash), and choose the convolution algorithm with `"jittor"`.

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

Because the Jittor framework is fast, inference and training are faster than PyTorch on many operators. We also speed up `quantize` with jittor's operations and get better performance.

We test the speed on the following model and choose 10 scenes from ScanNet as the dataset.

```python
algorithm = "cuda"
# you can turn it to "jittor"
model = nn.Sequential(
    spnn.Conv3d(3, 32, 2),
    spnn.BatchNorm(32),
    spnn.ReLU(),
    spnn.Conv3d(32, 64, 3, stride=1, algorithm=algorithm),
    spnn.BatchNorm(64),
    spnn.ReLU(),
    spnn.Conv3d(64, 128, 3, stride=1, algorithm=algorithm),
    spnn.BatchNorm(128),
    spnn.ReLU(),
    spnn.Conv3d(128, 256, 2, stride=2, algorithm=algorithm),
    spnn.BatchNorm(256),
    spnn.ReLU(),
    spnn.Conv3d(256, 128, 2, stride=2, transposed=True, algorithm=algorithm),
    spnn.BatchNorm(128),
    spnn.ReLU(),
    spnn.Conv3d(128, 64, 3, stride=1, transposed=True, algorithm=algorithm),
    spnn.BatchNorm(64),
    spnn.ReLU(),
    spnn.Conv3d(64, 32, 3, stride=1, transposed=True, algorithm=algorithm),
    spnn.BatchNorm(32),
    spnn.ReLU(),
    spnn.Conv3d(32, 3, 2),
)
``` 

We finnished two versions of Sparse Convolution(completed convolution function with jittor operators or cuda).

We choose attribute `batch_size = 2, total_len = 10` and run on RTX3080 to test per iteration's speed (JSparse's version is `v0.5.0` ).

|                   | JSparse(jittor) | JSparse(cuda) | TorchSparse(v1.4.0) |
|-------------------|-----------------|---------------|----------------------|
| voxel_size = 0.50 | 26.60ms         | 20.05ms       | 33.66ms              |
| voxel_size = 0.10 | 32.34ms         | 25.15ms       | 40.40ms              |
| voxel_size = 0.02 | 86.89ms         | 81.37ms       | 87.42ms              |

We also test the same 200 scenes of ScanNet on [VMNet](https://github.com/hzykent/VMNet) on JSparse and TorchSparse.

We choose attribute `batch_size = 3, num_workers=16` and run on RTX Titan and Intel(R) Xeon(R) CPU E5-2678 v3 to test per iteration's speed.

| JSparse(cuda) | TorchSparse(v1.4.0)  |
|---------------|----------------------|
| 0.79s         | 0.92s                |

> If we ignore the initiation(`scannet.py`), and just test the speed of network, the speed of JSparse and TorchSparse is similar.

## Acknowledgements

The implementation and idea of JSparse refers to many open source libraries, including(but not limited to) [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine) and [TorchSparse](https://github.com/mit-han-lab/torchsparse).

If you use JSparse in your research, please cite our and their works by using the following BibTeX entries:

```bibtex
@article{hu2020jittor,
  title={Jittor: a novel deep learning framework with meta-operators and unified graph execution},
  author={Hu, Shi-Min and Liang, Dun and Yang, Guo-Ye and Yang, Guo-Wei and Zhou, Wen-Yang},
  journal={Science China Information Sciences},
  volume={63},
  number={222103},
  pages={1--21},
  year={2020}
}
```

```bibtex
@inproceedings{tang2022torchsparse,
  title = {{TorchSparse: Efficient Point Cloud Inference Engine}},
  author = {Tang, Haotian and Liu, Zhijian and Li, Xiuyu and Lin, Yujun and Han, Song},
  booktitle = {Conference on Machine Learning and Systems (MLSys)},
  year = {2022}
}
```

```bibtex
@inproceedings{tang2020searching,
  title = {{Searching Efficient 3D Architectures with Sparse Point-Voxel Convolution}},
  author = {Tang, Haotian and Liu, Zhijian and Zhao, Shengyu and Lin, Yujun and Lin, Ji and Wang, Hanrui and Han, Song},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year = {2020}
}
```