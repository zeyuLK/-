# 关于如何导入自编的包
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
epoch_num2 = 10  # 第二次训练的 epoch 数
test_batch_size = 256


def second_train():
    train_dataset_paired = Multimodal_Datasets(train=True)
    trainloader_paired = DataLoader(train_dataset_paired, batch_size,
                                    shuffle=True, num_workers=0)

    # 加载预训练模型
    net = Multi_view_dynamic_relation()
    pretrained_model_path = 'D:/Study/RMPF/model_inf/reuter_pretrained_model.pth'
    try:
        net.load_state_dict(torch.load(pretrained_model_path))
        print(f'Successfully loaded pre-trained model from {pretrained_model_path}')
    except FileNotFoundError:
        print(f'Error: The pre-trained model file {pretrained_model_path} was not found.')
        return

    if use_cuda:
        net = net.cuda()

    lossfunc2 = MultiView_all_loss()
    optimizer_2 = torch.optim.Adam(net.relation_net.parameters(), lr=0.0001)

    print('-' * 10, 'Second Training Start', '-' * 10)
    for epoch in range(epoch_num2):
        net.train()
        #为了使代码更加清晰和规范，通常在训练循环开始时显式地调用 net.train()，这样可以明确表示当前模型处于训练状态，便于代码的理解和维护。
        num_batches = len(trainloader_paired)
        running_loss_2, proc_size = 0, 0
        start_time = time.time()
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
            optimizer_2.zero_grad()
            if use_cuda:
                inputs_x1, inputs_x2, inputs_x3, inputs_x4, inputs_x5, label_train,s_x1,s_x2,s_x3,s_x4,s_x5 = inputs_x1.cuda(), inputs_x2.cuda(), inputs_x3.cuda(), inputs_x4.cuda(), inputs_x5.cuda(), label_train.cuda(non_blocking=True), s_x1.cuda(),s_x2.cuda(),s_x3.cuda(),s_x4.cuda(),s_x5.cuda()
                net = net.cuda()
                m_s_x1,m_s_x2,m_s_x3,m_s_x4,m_s_x5,\
                s_x1_en,s_x2_en,s_x3_en,s_x4_en,s_x5_en,\
               x1_en,x2_en,x3_en,x4_en,x5_en,\
               x1_relation,x2_relation,x3_relation,x4_relation,x5_relation,\
               x1_en_relation,x2_en_relation,x3_en_relation,x4_en_relation,x5_en_relation,\
               x_view_relation,adj_label,adj_recover= net(s_x1,inputs_x1,s_x2,inputs_x2,s_x3,inputs_x3, s_x4,inputs_x4, s_x5,  inputs_x5, 5)

            loss, _, loss2  = lossfunc2( m_s_x1,m_s_x2,m_s_x3,m_s_x4,m_s_x5,\
                inputs_x1, inputs_x2, inputs_x3, inputs_x4, inputs_x5,\
                s_x1_en,s_x2_en,s_x3_en,s_x4_en,s_x5_en,\
               x1_en,x2_en,x3_en,x4_en,x5_en,\
               x1_relation,x2_relation,x3_relation,x4_relation,x5_relation,\
               x1_en_relation,x2_en_relation,x3_en_relation,x4_en_relation,x5_en_relation,\
               x_view_relation,adj_label,adj_recover,sigma,alpha=0.2, lam=10.0)



            loss_2 = loss2
            loss_2.backward()
            optimizer_2.step()
            running_loss_2 += loss_2.item()

            proc_size += batch_idx
            if batch_idx % 100 == 0:
                if batch_idx == 0:
                    avg_loss_2 = running_loss_2
                else:
                    avg_loss_2 = running_loss_2 / 100
                elapsed_time = time.time() - start_time
                print('Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Loss_2 {:5.4f}'.
                      format(epoch, batch_idx, num_batches, elapsed_time * 1000 / 100, avg_loss_2))
                running_loss_2, proc_size = 0, 0
                start_time = time.time()

    # 保存第二次训练后的模型
    second_trained_model_path = 'D:/Study/RMPF/model_inf/second_trained_model.pth'
    torch.save(net.state_dict(), second_trained_model_path)
    print(f'Second trained model saved to {second_trained_model_path}')
    print('-' * 10, 'Second Training End', '-' * 10)


if __name__ == "__main__":
    # 直接进行第二次训练，从预训练模型加载参数
    second_train()
