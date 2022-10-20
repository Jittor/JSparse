import jittor as jt 
from jittor import nn
import numpy as np
import glob
import os
import h5py
from jittor.dataset import Dataset 
import random
from jsparse import SparseTensor
import jsparse.nn as spnn
from tqdm import tqdm
import argparse


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

class ModelNet40H5(Dataset):
    def __init__(
        self,
        data_root = "modelnet40h5",
        transform=None,
        num_points=2048,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        voxel_size=1,
        mode="train",
    ):
        super().__init__(batch_size=batch_size, num_workers=num_workers,shuffle=shuffle)
        assert mode in ['test','train']
        if not os.path.exists(data_root):
            import wget 
            wget.download()
        # download_modelnet40_dataset()
        self.data, self.label = self.load_data(data_root, mode)
        self.transform = transform
        self.mode = mode
        self.num_points = num_points
        self.voxel_size = voxel_size
        self.total_len = len(self.data)

    def load_data(self, data_root, mode):
        print("loading data...")
        data, labels = [], []
        assert os.path.exists(data_root), f"{data_root} does not exist"
        files = glob.glob(os.path.join(data_root, f"ply_data_{mode}*.h5"))
        assert len(files) > 0, "No files found"
        for h5_name in files:
            with h5py.File(h5_name) as f:
                data.extend(f["data"][:].astype("float32"))
                labels.extend(f["label"][:].astype("int32"))
        data = np.stack(data, axis=0)
        labels = np.stack(labels, axis=0)
        print("finished data loading")
        return data, labels

    def __getitem__(self, i):
        xyz = self.data[i]
        if self.mode == "train":
            np.random.shuffle(xyz)
        if len(xyz) > self.num_points:
            xyz = xyz[: self.num_points]
        if self.transform is not None:
            xyz = self.transform(xyz)
        label = self.label[i]
         
        # voxelization
        indices = ((xyz-xyz.min(axis=0))/self.voxel_size).astype('int32')

        _,unique_idx = np.unique(indices,axis=0,return_index=True)
        indices = indices[unique_idx]
        xyz = xyz[unique_idx]
        return indices,xyz,label 

    def collate_batch(self, batch):
        indices = []
        feats = []
        labels = []
        for i,(index,feat,label) in enumerate(batch):
            index = np.concatenate([np.ones([index.shape[0],1])*i,index],axis=1)
            indices.append(index)
            feats.append(feat)
            labels.append(label)
        indices = np.concatenate(indices,axis=0)
        feats = np.concatenate(feats,axis=0)
        labels = np.concatenate(labels,axis=0)        
        tensor = SparseTensor(indices=jt.array(indices), values=jt.array(feats))
        return tensor,labels
        
class VoxelCNN(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.mlp1 = nn.Sequential(
            spnn.Linear(in_channels,64,bias=False),
            spnn.BatchNorm(64),
            spnn.ReLU()
        )
        self.convs = nn.Sequential(
            spnn.SparseConvBlock(64,128,kernel_size=3),
            spnn.SparseConvBlock(128,128,kernel_size=2,stride=2),
            spnn.SparseConvBlock(128,256,kernel_size=3),
            spnn.SparseConvBlock(256,256,kernel_size=2,stride=2),
            spnn.SparseConvBlock(256,512, kernel_size=3),
            spnn.SparseConvBlock(512,512,kernel_size=2,stride=2),
            spnn.SparseConvBlock(512,1024,kernel_size=3),
            spnn.SparseConvBlock(1024,1024,kernel_size=2,stride=2),
            spnn.SparseConvBlock(1024,1024,kernel_size=3),
        )
        self.global_pool = spnn.GlobalPool(op="max")

        self.fc = nn.Sequential(
            nn.Linear(1024,512,bias=False),
            nn.BatchNorm(512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512,256,bias=False),
            nn.BatchNorm(256),
            nn.ReLU(),
            nn.Linear(256,out_channels),
        )
    
    def execute(self,x):
        x = self.mlp1(x)
        x = self.convs(x)
        x = self.global_pool(x)
        x = self.fc(x)
        return x


def train(model,train_loader,optimizer):
    model.train()
    for i,(data,label) in tqdm(enumerate(train_loader),total=len(train_loader)):
        output = model(data)
        loss = nn.cross_entropy_loss(output,label)
        optimizer.step(loss)

def test(model,test_loader):
    model.eval()
    correct = 0
    n = 0
    for i,(data,label) in tqdm(enumerate(test_loader),total=len(test_loader)):
        output = model(data)
        pred,_ = output.argmax(dim=1)
        correct += (pred==label).sum().item()
        n += len(label)
    return correct/n

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size",type=int,default=64)
    parser.add_argument("--lr",type=float,default=0.01)
    parser.add_argument("--epochs",type=int,default=200)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--voxel_size",type=float,default=0.02)
    parser.add_argument("--data_root",type=str,default="./data/modelnet40_ply_hdf5_2048")
    parser.add_argument("--save_path",type=str,default="./model")
    return parser.parse_args()

def main():
    jt.flags.use_cuda = 1

    args = parse_args()
    # wget https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip and unzip it
    train_loader = ModelNet40H5(
        data_root=args.data_root,
        batch_size=args.batch_size,
        mode='train',
        transform=CoordinateTransformation(),
        num_workers=args.num_workers,
        shuffle=True,
        voxel_size=args.voxel_size)
    test_loader = ModelNet40H5(
        data_root=args.data_root,
        batch_size=args.batch_size,
        mode='test',
        num_workers=args.num_workers,
        voxel_size=args.voxel_size)

    model = VoxelCNN(3,40)
    optimizer = nn.SGD(model.parameters(),lr=args.lr,momentum=0.9,weight_decay=1e-4)
    scheduler = jt.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs)
    for epoch in range(args.epochs):
        train(model,train_loader,optimizer)
        scheduler.step()
        acc = test(model,test_loader)
        print(f"epoch:{epoch},acc:{acc},lr:{optimizer.lr}")
    
if __name__ == "__main__":
    main()
