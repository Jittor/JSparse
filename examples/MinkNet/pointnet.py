# Copyright (c) 2020 NVIDIA CORPORATION.
# Copyright (c) 2018-2020 Chris Choy (chrischoy@ai.stanford.edu).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.
import os
import random
import numpy as np
import glob

try:
    import h5py
except:
    print("Install h5py with `pip install h5py`")
import subprocess

import jittor
import jittor.nn as nn
from jittor.dataset import Dataset

import JSparse
from JSparse import SparseTensor, PointTensor
from JSparse import nn as spnn
from JSparse.utils.collate import sparse_collate_fn

def stack_collate_fn(list_data):
    coordinates_batch, features_batch, labels_batch = (
        jittor.stack([d["coordinates"] for d in list_data]),
        jittor.stack([d["features"] for d in list_data]),
        jittor.cat([d["label"] for d in list_data]),
    )

    return {
        "coordinates": coordinates_batch,
        "features": features_batch,
        "labels": labels_batch,
    }

# MinkowskiNet implementation of a pointnet.
#
# This network allows the number of points per batch to be arbitrary. For
# instance batch index 0 could have 500 points, batch index 1 could have 1000
# points.
class MinkowskiPointNet(nn.Module):

    def __init__(self, in_channel, out_channel, embedding_channel=1024):
        super().__init__()
        self.conv1 = nn.Sequential(
            spnn.Linear(3, 64, bias=False),
            spnn.BatchNorm(64),
            spnn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            spnn.Linear(64, 64, bias=False),
            spnn.BatchNorm(64),
            spnn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            spnn.Linear(64, 64, bias=False),
            spnn.BatchNorm(64),
            spnn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            spnn.Linear(64, 128, bias=False),
            spnn.BatchNorm(128),
            spnn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            spnn.Linear(128, embedding_channel, bias=False),
            spnn.BatchNorm(embedding_channel),
            spnn.ReLU(),
        )
        self.max_pool = spnn.GlobalMaxPool()

        self.linear1 = nn.Sequential(
            nn.Linear(embedding_channel, 512, bias=False),
            nn.BatchNorm(512),
            nn.ReLU(),
        )
        self.dp1 = nn.Dropout(0.1)
        self.linear2 = nn.Linear(512, out_channel, bias=True)

    def execute(self, x: JSparse.SparseTensor):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.max_pool(x)
        x = self.linear1(x)
        x = self.dp1(x)
        return self.linear2(x)


class CoordinateTransformation:
    def __init__(self, scale_range=(0.9, 1.1), trans=0.25, jitter=0.025, clip=0.05):
        self.scale_range = scale_range
        self.trans = trans
        self.jitter = jitter
        self.clip = clip

    def __call__(self, coords):
        if random.random() < 0.9:
            coords *= np.random.uniform(
                low=self.scale_range[0], high=self.scale_range[1], size=[1, 3]
            )
        if random.random() < 0.9:
            coords += np.random.uniform(low=-self.trans, high=self.trans, size=[1, 3])
        if random.random() < 0.7:
            coords += np.clip(
                self.jitter * (np.random.rand(len(coords), 3) - 0.5),
                -self.clip,
                self.clip,
            )
        return coords

    def __repr__(self):
        return f"Transformation(scale={self.scale_range}, translation={self.trans}, jitter={self.jitter})"


def download_modelnet40_dataset():
    if not os.path.exists("modelnet40_ply_hdf5_2048.zip"):
        print("Downloading the 2k downsampled ModelNet40 dataset...")
        subprocess.run(
            [
                "wget",
                "--no-check-certificate",
                "https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip",
            ]
        )
        subprocess.run(["unzip", "modelnet40_ply_hdf5_2048.zip"])


class ModelNet40H5(Dataset):
    def __init__(
        self,
        phase: str,
        data_root: str = "modelnet40h5",
        transform=None,
        num_points=2048,
        voxel_size: float=1
    ):
        Dataset.__init__(self)
        download_modelnet40_dataset()
        phase = "test" if phase in ["val", "test"] else "train"
        self.data, self.label = self.load_data(data_root, phase)
        self.transform = transform
        self.phase = phase
        self.num_points = num_points
        self.voxel_size = voxel_size

    def load_data(self, data_root, phase):
        data, labels = [], []
        assert os.path.exists(data_root), f"{data_root} does not exist"
        files = glob.glob(os.path.join(data_root, "ply_data_%s*.h5" % phase))
        assert len(files) > 0, "No files found"
        for h5_name in files:
            with h5py.File(h5_name) as f:
                data.extend(f["data"][:].astype("float32"))
                labels.extend(f["label"][:].astype("int32"))
        data = np.stack(data, axis=0)
        labels = np.stack(labels, axis=0)
        return data, labels

    def __getitem__(self, i: int) -> dict:
        xyz = self.data[i]
        if self.phase == "train":
            np.random.shuffle(xyz)
        if len(xyz) > self.num_points:
            xyz = xyz[: self.num_points]
        if self.transform is not None:
            xyz = self.transform(xyz)
        label = self.label[i]
        xyz = jittor.array(xyz)
        label = jittor.array(label)

        inputs = SparseTensor(xyz, xyz, 1, quantize=True, voxel_size=self.voxel_size)
        return {
            "inputs": inputs,
            "label": label,
        }

    def __len__(self):
        return self.data.shape[0]

    def __repr__(self):
        return f"ModelNet40H5(phase={self.phase}, length={len(self)}, transform={self.transform})"


if __name__ == "__main__":
    dataset = ModelNet40H5(phase="train", data_root="modelnet40_ply_hdf5_2048")

    # Use minkowski_collate_fn for pointnet
    minknet_data_loader = dataset.set_attrs(
        num_workers=4, collate_batch=sparse_collate_fn, batch_size=16,
    )

    # Network
    minkpointnet = MinkowskiPointNet(
        in_channel=3, out_channel=20, embedding_channel=1024
    )

    for i, minknet_batch in enumerate(minknet_data_loader):
        # MinkNet
        # Unlike pointnet, number of points for each point cloud do not need to be the same.
        minknet_input = SparseTensor(
            indices=minknet_batch["coordinates"], values=minknet_batch["features"]
        )
        minkpointnet(minknet_input)
        print(f"Processed batch {i}")