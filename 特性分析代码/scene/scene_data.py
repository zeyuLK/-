import random
import sys
import scipy.io as sio
import numpy as np
import torch
from PIL import Image
import os
import torch.utils.data as data
from torchvision import transforms
from sklearn import preprocessing
#import h5py #暂时没用到，其实也是数据库处理库
#from keras.utils import np_utils
from sklearn.decomposition import PCA
# 


class Multimodal_Datasets(data.Dataset):
    def __init__(self, train=True):
        self.train = train
        # 读取数据
        dd = sio.loadmat("D:\\Study\\RMPF\\scene15.mat")

        data_view1 = dd['X1'].T
        data_view2 = dd['X2'].T
        data_view3 = dd['X3'].T
        label = dd['gt'].squeeze()

        # 归一化
        data_view1 = preprocessing.MinMaxScaler((0, 1)).fit_transform(data_view1)
        data_view2 = preprocessing.MinMaxScaler((0, 1)).fit_transform(data_view2)
        data_view3 = preprocessing.MinMaxScaler((0, 1)).fit_transform(data_view3)

        if self.train:
            self.data_view1 = data_view1
            self.data_view2 = data_view2
            self.data_view3 = data_view3
            self.labels = label
        else:
            self.data_view1 = data_view1
            self.data_view2 = data_view2
            self.data_view3 = data_view3
            self.labels = label

    def __getitem__(self, index):
        try:
            view1 = torch.from_numpy(self.data_view1[index]).float()
            view2 = torch.from_numpy(self.data_view2[index]).float()
            view3 = torch.from_numpy(self.data_view3[index]).float()

            target = self.labels[index]
            # 统一为 LongTensor，保证和 loss 兼容
            target = torch.tensor(int(target), dtype=torch.long)

            return view1, view2, view3, target
        except Exception as e:
            print(f"[Error] __getitem__ failed at index {index}: {e}")
            # 返回 None，DataLoader 会在 collate_fn 里过滤掉
            return None

    def __len__(self):
        return len(self.labels)

