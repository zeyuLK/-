# 关于如何导入自编的包
import sys
sys.path.append(r"D:/Study/RMPF/reuters")
from Loss.Rp_loss import MultiView_all_loss
from net.Rp_repre import Multi_view_dynamic_relation
from Rp_data import *
from net.metrics import get_avg_acc, get_avg_nmi, get_avg_RI, get_avg_f1

# 关于GPU的选择，因为电脑只有一个GPU所以不用设置下面的GPU进程之类
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES']='0'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import sys
import time
import itertools
import numpy as np
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
import scipy.io as sio
import os

use_cuda = torch.cuda.is_available()
sys.path.append('.')
import warnings
# warnings.filterwarnings("ignore")  # 建议谨慎使用，调试时可先不忽略

# 启用代码异常检测，调试完成后可关闭以提高效率
torch.autograd.set_detect_anomaly(True)

batch_size = 256
epoch_num1 = 30
epoch_num2 = 5  # 假设后续需要，先定义好
test_batch_size = 256


def pre_train():
    train_dataset_paired = Multimodal_Datasets(train=True)
    test_dataset = Multimodal_Datasets(train=False)

    trainloader_paired = DataLoader(train_dataset_paired, batch_size,
                                    shuffle=True, num_workers=0)
    testloader = DataLoader(test_dataset, test_batch_size,
                            shuffle=False, num_workers=0)  # clustering training_set=test_set
    print('Finish loading the data....')
    net = Multi_view_dynamic_relation()
    lossfunc1 = MultiView_all_loss(round=1)
    # torch.optim.Adam 是 PyTorch 中实现 Adam 优化算法的类。
    # itertools.chain 是一个函数，用于将多个迭代器串联成一个长迭代器。
    # net.S_net1.parameters() 是一个迭代器，包含了模型 net 中 S_net1 部分的参数，
    # optimizer_1 实际上是将多个子网络（S_net 和 E_net 的不同实例）的参数组合起来进行优化。
    # lr=0.0001 是学习率
    optimizer_1 = torch.optim.Adam(itertools.chain(net.S_net1.parameters(), net.E_net1.parameters(),
                                                   net.S_net2.parameters(), net.E_net2.parameters(),
                                                   net.S_net3.parameters(), net.E_net3.parameters(),
                                                   net.S_net4.parameters(), net.E_net4.parameters(),
                                                   net.S_net5.parameters(), net.E_net5.parameters()), lr=0.0001)

    print('-' * 10, 'Pre-Training Start', '-' * 10)
    for epoch in range(epoch_num1):
        num_batches = len(trainloader_paired)
        running_loss_1, proc_size = 0, 0
        for batch_idx, train_data in enumerate(trainloader_paired):
            inputs_x1, inputs_x2, inputs_x3, inputs_x4, inputs_x5, label_train = train_data
            var1 = torch.var(inputs_x1, unbiased=False)
            var2 = torch.var(inputs_x2, unbiased=False)
            var3 = torch.var(inputs_x3, unbiased=False)
            var4 = torch.var(inputs_x4, unbiased=False)
            var5 = torch.var(inputs_x5, unbiased=False)
            sigma=[var1,var2,var3,var4,var5]

            noise1 = torch.randn_like(inputs_x1) * torch.sqrt(var1)
            noise2 = torch.randn_like(inputs_x2) * torch.sqrt(var2)
            noise3 = torch.randn_like(inputs_x3) * torch.sqrt(var3)
            noise4 = torch.randn_like(inputs_x4) * torch.sqrt(var4)
            noise5 = torch.randn_like(inputs_x5) * torch.sqrt(var5)
            s_x1 = inputs_x1 + noise1
            s_x2 = inputs_x2 + noise2
            s_x3 = inputs_x3 + noise3
            s_x4 = inputs_x4 + noise4
            s_x5 = inputs_x5 + noise5
            optimizer_1.zero_grad()
            if use_cuda:
                inputs_x1, inputs_x2, inputs_x3, inputs_x4, inputs_x5, label_train,s_x1,s_x2,s_x3,s_x4,s_x5 = inputs_x1.cuda(), inputs_x2.cuda(), inputs_x3.cuda(), inputs_x4.cuda(), inputs_x5.cuda(), label_train.cuda(non_blocking=True), s_x1.cuda(),s_x2.cuda(),s_x3.cuda(),s_x4.cuda(),s_x5.cuda()
                net = net.cuda()
                m_s_x1,m_s_x2,m_s_x3,m_s_x4,m_s_x5,\
                s_x1_en,s_x2_en,s_x3_en,s_x4_en,s_x5_en,\
               x1_en,x2_en,x3_en,x4_en,x5_en,\
               x1_relation,x2_relation,x3_relation,x4_relation,x5_relation,\
               x1_en_relation,x2_en_relation,x3_en_relation,x4_en_relation,x5_en_relation,\
               x_view_relation,adj_label,adj_recover= net(s_x1,inputs_x1,s_x2,inputs_x2,s_x3,inputs_x3, s_x4, inputs_x4,s_x5,  inputs_x5, 5)

            loss1 = lossfunc1( m_s_x1,m_s_x2,m_s_x3,m_s_x4,m_s_x5,\
                inputs_x1, inputs_x2, inputs_x3, inputs_x4, inputs_x5,\
                s_x1_en,s_x2_en,s_x3_en,s_x4_en,s_x5_en,\
               x1_en,x2_en,x3_en,x4_en,x5_en,\
               x1_relation,x2_relation,x3_relation,x4_relation,x5_relation,\
               x1_en_relation,x2_en_relation,x3_en_relation,x4_en_relation,x5_en_relation,\
               x_view_relation,adj_label,adj_recover,sigma,alpha=0.2, lam=10.0)

            loss_1 = loss1
            loss_1.backward()
            optimizer_1.step()
            running_loss_1 += loss_1.item()
            # 这行代码将 loss_1 的值（通过 .item() 方法转换为 Python 标量）累加到 running_loss_1 变量中
            # 这样，running_loss_1 就是一个累积的损失值，它会在每个 epoch 结束时被打印出来。

            proc_size += batch_idx
            if batch_idx % 100 == 0:
                if batch_idx == 0:
                    avg_loss_1 = running_loss_1
                else:
                    avg_loss_1 = running_loss_1 / 100

                print('Epoch {:2d} | Batch {:3d}/{:3d} | Train Loss_1 {:5.4f}'.
                      format(epoch, batch_idx, num_batches, avg_loss_1))
                running_loss_1, proc_size = 0, 0

    pretrained_model_path = 'D:/Study/RMPF/model_inf/reuter_pretrained_model.pth'
    torch.save(net.state_dict(), pretrained_model_path)
    print(f'Pre-trained model saved to {pretrained_model_path}')

    print('-' * 10, 'Pre-Training End', '-' * 10)
    print('Finish pre-training....')
if __name__ == "__main__":
    pre_train()