from pickletools import optimize
from statistics import mode
import jittor
from jittor import nn

import JSparse 
from JSparse import SparseTensor
from JSparse import nn as spnn
from JSparse.utils.quantize import sparse_quantize

import numpy as np

class RandomDataset(jittor.dataset.Dataset):

    def __init__(self, input_size: int, voxel_size: float) -> None:
        super().__init__()
        self.set_attrs(total_len = input_size)
        self.voxel_size = voxel_size

    def __getitem__(self, _: int):
        inputs = np.random.uniform(-100, 100, size=(self.total_len, 4))
        labels = np.random.choice(10, size=self.total_len)

        coords, feats = inputs[:, :], inputs
        coords -= np.min(coords, axis=0, keepdims=True)
#       coords, indices = sparse_quantize(coords, self.voxel_size, return_index=True)

        coords = jittor.Var(coords)
        feats = jittor.Var(feats)
        labels = jittor.Var(labels)
#       coords = jittor.Var(coords, dtype=jittor.int64)
#       feats = jittor.Var(feats[indices], dtype=jittor.float64)
#       labels = jittor.Var(labels[indices], dtype=jittor.int64)

        print(type(coords))
        inputs = SparseTensor(coords, feats, 1, 1)
        labels = SparseTensor(coords, labels, 1, 1)
        return inputs, labels

if __name__ == '__main__':
    np.random.seed(0)

    dataset = RandomDataset(input_size=10000, voxel_size=0.2)

    model = nn.Sequential(
        spnn.Conv3d(4, 32, 3),
        spnn.BatchNorm(32),
        spnn.ReLU(True),
        spnn.Conv3d(32, 64, 2, stride=2),
        spnn.BatchNorm(64),
        spnn.ReLU(True),
        spnn.Conv3d(64, 64, 2, stride=2, transposed=True),
        spnn.BatchNorm(64),
        spnn.ReLU(True),
        spnn.Conv3d(64, 32, 3),
        spnn.BatchNorm(32),
        spnn.ReLU(True),
        spnn.Conv3d(32, 10, 1),
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = jittor.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    lens = len(dataset)
    for batch_idx, (inputs, labels) in enumerate(dataset):
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.setp(loss)

        if batch_idx % 10 == 0:
            print('Training: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx, lens , 100. * batch_idx / lens, loss.numpy()[0]))