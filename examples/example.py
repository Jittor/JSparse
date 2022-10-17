import time
import jittor as jt
from jittor import nn

import jsparse 
from jsparse import SparseTensor
from jsparse import nn as spnn
from jsparse.utils import sparse_quantize
from jsparse.utils.collate import sparse_collate_fn

import numpy as np

class RandomDataset(jt.dataset.Dataset):

    def __init__(self, input_size: int, voxel_size: float,
                 batch_size: int = 1, total_len: int = 10000, save: bool=False) -> None:
        super().__init__()
        self.set_attrs(total_len = total_len)
        self.set_attrs(batch_size = batch_size)
        self.input_size = input_size
        self.voxel_size = voxel_size
        self.save = save
        self.data = {}

    def __getitem__(self, _: int):
        if self.save == False or _ not in self.data:
            inputs = np.random.uniform(-100, 100, size=(self.input_size, 4))
            labels = np.random.choice(10, size=self.input_size)

            coords, feats = inputs[:, :3], inputs
            coords -= np.min(coords, axis=0, keepdims=True)

            coords = jt.array(coords, dtype='int32')
            feats = jt.array(feats, dtype='float32')
            labels = jt.array(labels, dtype='int32')

            inputs = SparseTensor(feats, coords, 1, quantize=True, voxel_size=self.voxel_size)
            labels = SparseTensor(labels, coords, 1, quantize=True, voxel_size=self.voxel_size)
            assert(inputs.indices.shape == labels.indices.shape)
            if self.save == True:
                self.data[_] = {'inputs': inputs, 'labels': labels}
            return {'inputs': inputs, 'labels': labels}
        else:
            return self.data[_]

    def collate_batch(self, batch):
        return sparse_collate_fn(batch)

if __name__ == '__main__':
    t = time.perf_counter()
    np.random.seed(0)
    jt.flags.nvcc_flags += " --extended-lambda"
    jt.flags.use_cuda = 1
 #  jt.flags.amp_level=5

    dataset = RandomDataset(input_size=20000, voxel_size=0.2, batch_size=16, total_len=20000)

    model = nn.Sequential(
        spnn.Conv3d(4, 32, 3),
        spnn.BatchNorm(32),
        spnn.ReLU(),
        spnn.Conv3d(32, 64, 2, stride=2),
        spnn.BatchNorm(64),
        spnn.ReLU(),
        spnn.Conv3d(64, 64, 2, stride=2, transposed=True),
        spnn.BatchNorm(64),
        spnn.ReLU(),
        spnn.Conv3d(64, 32, 3),
        spnn.BatchNorm(32),
        spnn.ReLU(),
        spnn.Conv3d(32, 10, 1),
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = jt.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    lens = len(dataset)

    jt.sync_all(True)
    for batch_idx, pack in enumerate(dataset):
        print("batch_idx:", batch_idx)
        inputs = pack['inputs']
        labels = pack['labels']

        outputs = model(inputs)
        loss = criterion(outputs.values, labels.values)
        optimizer.step(loss)

        if batch_idx % 10 == 0:
            print('Training: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx, lens , 100. * batch_idx / lens, loss.numpy()[0]))

    jt.sync_all(True)
    print(f'cost:{(time.perf_counter() - t):.8f}s')