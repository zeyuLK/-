import sys
import time
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
import scipy.io as sio
from sklearn import preprocessing

# 你自编的包
sys.path.append(r"D:\\Study\\RMPF\\landuse")
from landuse.loss import MultiView_all_loss
from landuse.prepare import Multi_view_dynamic_relation
from landuse.landuse_data import *
from net.metrics import get_avg_acc, get_avg_nmi, get_avg_RI, get_avg_f1

use_cuda = torch.cuda.is_available()
batch_size = 200
epoch_num1 = 60
test_batch_size = 256


# ========= Dataset ========= #
import torch.utils.data as data

class Multimodal_Datasets(data.Dataset):
    def __init__(self, train=True):
        self.train = train
        dd = sio.loadmat("D:\\Study\\RMPF\\LandUse_21.mat")

        data_view1 = dd['X'][0, 0]
        data_view2 = dd['X'][0, 1]
        data_view3 = dd['X'][0, 2]
        label = dd['Y'].squeeze()


        data_view1 = preprocessing.MinMaxScaler((0, 1)).fit_transform(data_view1)
        data_view2 = preprocessing.MinMaxScaler((0, 1)).fit_transform(data_view2)
        data_view3 = preprocessing.MinMaxScaler((0, 1)).fit_transform(data_view3)

        self.data_view1 = data_view1
        self.data_view2 = data_view2
        self.data_view3 = data_view3
        self.labels = label

    def __getitem__(self, index):
        try:
            view1 = torch.from_numpy(self.data_view1[index]).float()
            view2 = torch.from_numpy(self.data_view2[index]).float()
            view3 = torch.from_numpy(self.data_view3[index]).float()
            target = int(self.labels[index])
            target = torch.tensor(target, dtype=torch.long)
            return view1, view2, view3, target
        except Exception as e:
            print(f"[Error] __getitem__ failed at index {index}: {e}")
            return None

    def __len__(self):
        return len(self.labels)


# ========= collate_fn ========= #
def collate_fn(batch):
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)


# ========= 预训练函数 ========= #
def pre_train():
    # ---------- 数据检查 ----------
    print("Checking dataset integrity...")
    train_dataset_paired = Multimodal_Datasets(train=True)
    for i in range(len(train_dataset_paired)):
        sample = train_dataset_paired[i]
        if sample is None:
            print(f"[Error] Sample {i} is None")
        else:
            v1, v2, v3, label = sample
            if not (torch.is_tensor(v1) and torch.is_tensor(v2) and torch.is_tensor(v3) and torch.is_tensor(label)):
                print(f"[Error] Sample {i} wrong type: {[type(v1), type(v2), type(v3), type(label)]}")
    print("Dataset check done.")

    # ---------- DataLoader ----------
    trainloader_paired = DataLoader(
        train_dataset_paired, batch_size=batch_size,
        shuffle=True, num_workers=0, collate_fn=collate_fn
    )
    test_dataset = Multimodal_Datasets(train=False)
    testloader = DataLoader(
        test_dataset, batch_size=test_batch_size,
        shuffle=False, num_workers=0, collate_fn=collate_fn
    )

    print('Finish loading the data....')

    # ---------- 网络 ----------
    net = Multi_view_dynamic_relation()
    lossfunc1 = MultiView_all_loss(round=1)
    optimizer_1 = torch.optim.Adam(itertools.chain(
        net.S_net1.parameters(), net.E_net1.parameters(),
        net.S_net2.parameters(), net.E_net2.parameters(),
        net.S_net3.parameters(), net.E_net3.parameters(),
    ), lr=0.0001)

    print('-' * 10, 'Pre-Training Start', '-' * 10)

    for epoch in range(epoch_num1):
        num_batches = len(trainloader_paired)
        running_loss_1 = 0
        for batch_idx, train_data in enumerate(trainloader_paired):
            if train_data is None:
                print(f"[Warning] Batch {batch_idx} is empty, skipped.")
                continue

            inputs_x1, inputs_x2, inputs_x3, label_train = train_data
            var1 = torch.var(inputs_x1, unbiased=False)
            var2 = torch.var(inputs_x2, unbiased=False)
            var3 = torch.var(inputs_x3, unbiased=False)
            sigma = [var1, var2, var3]

            noise1 = torch.randn_like(inputs_x1) * torch.sqrt(var1)
            noise2 = torch.randn_like(inputs_x2) * torch.sqrt(var2)
            noise3 = torch.randn_like(inputs_x3) * torch.sqrt(var3)

            s_x1 = inputs_x1 + noise1
            s_x2 = inputs_x2 + noise2
            s_x3 = inputs_x3 + noise3

            optimizer_1.zero_grad()
            if use_cuda:
                inputs_x1, inputs_x2, inputs_x3, label_train, s_x1, s_x2, s_x3 = \
                    inputs_x1.cuda(), inputs_x2.cuda(), inputs_x3.cuda(), label_train.cuda(non_blocking=True), \
                    s_x1.cuda(), s_x2.cuda(), s_x3.cuda()
                net = net.cuda()

            m_s_x1, m_s_x2, m_s_x3, \
            s_x1_en, s_x2_en, s_x3_en, \
            x1_en, x2_en, x3_en, \
            x1_relation, x2_relation, x3_relation, \
            x1_en_relation, x2_en_relation, x3_en_relation, \
            x_view_relation, adj_label, adj_recover = net(
                s_x1, s_x2, s_x3, inputs_x1, inputs_x2, inputs_x3, 3
            )

            loss1 = lossfunc1(
                m_s_x1, m_s_x2, m_s_x3,
                inputs_x1, inputs_x2, inputs_x3,
                s_x1_en, s_x2_en, s_x3_en,
                x1_en, x2_en, x3_en,
                x1_relation, x2_relation, x3_relation,
                x1_en_relation, x2_en_relation, x3_en_relation,
                x_view_relation, adj_label, adj_recover,
                sigma, alpha=0.2, lam=10.0
            )

            loss1.backward()
            optimizer_1.step()
            running_loss_1 += loss1.item()

            if batch_idx % 100 == 0:
                avg_loss_1 = running_loss_1 / max(1, (batch_idx + 1))
                print(f"Epoch {epoch:2d} | Batch {batch_idx:3d}/{num_batches:3d} | Train Loss {avg_loss_1:5.4f}")

    pretrained_model_path = 'D:/Study/RMPF/model_inf/landuse_pretrained_model.pth'
    torch.save(net.state_dict(), pretrained_model_path)
    print(f'Pre-trained model saved to {pretrained_model_path}')
    print('-' * 10, 'Pre-Training End', '-' * 10)
    print('Finish pre-training....')


if __name__ == "__main__":
    pre_train()
