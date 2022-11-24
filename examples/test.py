import time
import open3d as o3d

import jittor as jt
from jittor import nn

import jsparse 
from jsparse import SparseTensor
from jsparse import nn as spnn
from jsparse.utils import sparse_quantize
from jsparse.utils.collate import sparse_collate_fn

import numpy as np

def read_ply(filename):
    ply = o3d.io.read_triangle_mesh(filename)
    points = np.array(ply.vertices)
    colors = np.array(ply.vertex_colors)
    return points, colors

def save_as_pcd(path: str, input: SparseTensor):
    point_cloud = o3d.geometry.PointCloud()
    max_type = int(jt.max(input.indices[:, 0])) + 1
    for i in range(max_type):
        mask = input.indices[:, 0] == i
        point_cloud.points = o3d.utility.Vector3dVector((input.indices[mask, 1:4]).numpy())
        maximum = jt.max(jt.max(input.indices[mask, 1:4]))
        point_cloud.colors = o3d.utility.Vector3dVector((input.indices[mask, 1:4] / maximum).numpy())
        o3d.io.write_point_cloud(str(i) + '_' + path, point_cloud)

class PlyDataset(jt.dataset.Dataset):

    def __init__(self, voxel_size: float,
                 batch_size: int = 10, total_len: int = 10, save: bool=False) -> None:
        super().__init__()
        self.set_attrs(total_len = total_len, batch_size = batch_size)
        self.voxel_size = voxel_size

        self.inputs = []
        self.labels = []
        for i in range(total_len):
            coords, labels = read_ply(str(i) + ".ply")
            feats = coords
            coords -= np.min(coords, axis=0, keepdims=True)

            coords = jt.array(coords, dtype='float32')
            feats = jt.array(feats, dtype='float32')
            labels = jt.array(labels, dtype='float32')

            self.inputs.append(SparseTensor(feats, coords, 1, quantize=True, voxel_size=self.voxel_size))
            self.labels.append(SparseTensor(labels, coords, 1, quantize=True, voxel_size=self.voxel_size))

    def __getitem__(self, _: int):
        return {'inputs': self.inputs[_], 'labels': self.labels[_]}

    def collate_batch(self, batch):
        return sparse_collate_fn(batch)

if __name__ == '__main__':
    np.random.seed(0)
    jt.flags.nvcc_flags += " --extended-lambda"
    jt.flags.use_cuda = 1

    dataset = PlyDataset(voxel_size=0.05, batch_size=10, total_len=10)

    algorithm = "cuda"
    # it can be changed to "jittor" (based on python version)

    model = nn.Sequential(
        spnn.Conv3d(3, 32, 3),
        spnn.BatchNorm(32),
        spnn.ReLU(),
        spnn.Conv3d(32, 64, 3, stride=2, algorithm=algorithm),
        spnn.BatchNorm(64),
        spnn.ReLU(),
        spnn.Conv3d(64, 128, 3, stride=2, algorithm=algorithm),
        spnn.BatchNorm(128),
        spnn.ReLU(),
        spnn.Conv3d(128, 256, 3, stride=2, algorithm=algorithm),
        spnn.BatchNorm(256),
        spnn.ReLU(),
        spnn.Conv3d(256, 128, 3, stride=2, transposed=True, algorithm=algorithm),
        spnn.BatchNorm(128),
        spnn.ReLU(),
        spnn.Conv3d(128, 64, 3, stride=2, transposed=True, algorithm=algorithm),
        spnn.BatchNorm(64),
        spnn.ReLU(),
        spnn.Conv3d(64, 32, 3, stride=2, transposed=True, algorithm=algorithm),
        spnn.BatchNorm(32),
        spnn.ReLU(),
        spnn.Conv3d(32, 3, 1),
    )

    criterion = nn.L1Loss()
    optimizer = jt.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    lens = len(dataset)

    jt.sync_all(True)
    t = time.perf_counter()

    epoch_len = 30
    for epoch in range(epoch_len):
        print("epoch: ", epoch)
        for batch_idx, pack in enumerate(dataset):
            #print("batch_idx:", batch_idx)
            inputs = pack['inputs']
            labels = pack['labels']

            outputs = model(inputs)
            save_as_pcd('b.pcd', inputs)
            loss = criterion(outputs.values, labels.values)
            optimizer.step(loss)

            #if batch_idx % 10 == 0:
            #    print('Training: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #        batch_idx, lens , 100. * batch_idx / lens, loss.numpy()[0]))
    jt.sync_all(True)
    print(f'cost of per iter:{(time.perf_counter() - t) / (epoch_len * dataset.total_len) * 1000:.8f}ms')
    # jt.profiler.stop()
    # jt.profiler.report()