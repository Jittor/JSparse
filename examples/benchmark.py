from datetime import datetime
import numpy as np
import jittor as jt
from jittor import nn

from JSparse import SparseTensor
from JSparse import nn as spnn
from JSparse.utils import sparse_quantize
from JSparse.utils.collate import sparse_collate_fn

from example import RandomDataset

def dummy_train_3x3():
    model = nn.Sequential(
        spnn.Conv3d(4, 32, kernel_size=3, stride=1),
        spnn.Conv3d(32, 64, kernel_size=3, stride=1),
        spnn.Conv3d(64, 128, kernel_size=3, stride=1),
        spnn.Conv3d(128, 256, kernel_size=3, stride=1),
        spnn.Conv3d(256, 128, kernel_size=3, stride=1, transposed=True),
        spnn.Conv3d(128, 64, kernel_size=3, stride=1, transposed=True),
        spnn.Conv3d(64, 32, kernel_size=3, stride=1, transposed=True),
        spnn.Conv3d(32, 10, kernel_size=3, stride=1, transposed=True),
    )
    optimizer = jt.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print('Starting dummy_train_3x3...')
    time = datetime.now()

    dataset = RandomDataset(input_size=100000, voxel_size=0.2, batch_size=2, total_len = 10, save = True)
    with jt.profile_scope(rerun = 100) as report:
        for k, feed_dict in enumerate(dataset):
            inputs = feed_dict['inputs']
            targets = feed_dict['labels']
            outputs = model(inputs)
            loss = criterion(outputs.values, targets.values)
            optimizer.step(loss)
        print(report)

    jt.sync_all(True)
    time = datetime.now() - time
    print('Finished dummy_train_3x3 in ', time)

def dummy_train_3x1():
    model = nn.Sequential(
        spnn.Conv3d(4, 32, kernel_size=(3, 1, 3), stride=1),
        spnn.Conv3d(32, 64, kernel_size=(1, 3, 3), stride=1),
        spnn.Conv3d(64, 128, kernel_size=(3, 1, 3), stride=1),
        spnn.Conv3d(128, 256, kernel_size=(1, 3, 3), stride=1),
        spnn.Conv3d(256, 128, kernel_size=(3, 1, 3), stride=1, transposed=True),
        spnn.Conv3d(128, 64, kernel_size=(1, 3, 3), stride=1, transposed=True),
        spnn.Conv3d(64, 32, kernel_size=(3, 1, 3), stride=1, transposed=True),
        spnn.Conv3d(32, 10, kernel_size=(1, 3, 3), stride=1, transposed=True),
    )
    optimizer = jt.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print('Starting dummy_train_3x1...')
    time = datetime.now()

    dataset = RandomDataset(input_size=100000, voxel_size=0.2, batch_size=2, total_len = 10, save = True)
    with jt.profile_scope(rerun = 100) as report:
        for k, feed_dict in enumerate(dataset):
            inputs = feed_dict['inputs']
            targets = feed_dict['labels']
            outputs = model(inputs)
            loss = criterion(outputs.values, targets.values)
            optimizer.step(loss)
        print(report)

    jt.sync_all(True)
    time = datetime.now() - time
    print('Finished dummy_train_3x1 in ', time)

if __name__ == '__main__':
    jt.flags.nvcc_flags += " --extended-lambda"
    jt.flags.use_cuda = 1
    dummy_train_3x3()
    dummy_train_3x1()
