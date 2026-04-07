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
os.environ['LOKY_MAX_CPU_COUNT'] = '16'

# 后续的代码逻辑

use_cuda = torch.cuda.is_available()
sys.path.append('.')
import warnings
# warnings.filterwarnings("ignore")  # 建议谨慎使用，调试时可先不忽略

# 启用代码异常检测，调试完成后可关闭以提高效率
torch.autograd.set_detect_anomaly(True)

batch_size = 256
epoch_num2 = 5  # 第二次训练的 epoch 数
test_batch_size = 256

def test(net):
    test_dataset = Multimodal_Datasets(train=False)
    testloader = DataLoader(test_dataset, test_batch_size,
                            shuffle=False, num_workers=0)

    if use_cuda:
        net = net.cuda()

    net.eval()  # 设置模型为评估模式
    x1_feature = []
    x2_feature = []
    x3_feature = []
    x4_feature = []
    x5_feature = []
    label_all = []

    with torch.no_grad():  # 禁用梯度计算，提高推理效率
        for data in testloader:
            inputs_x1, inputs_x2, inputs_x3, inputs_x4, inputs_x5, label_train = data
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
            if use_cuda:
                inputs_x1, inputs_x2, inputs_x3, inputs_x4, inputs_x5, label_train,s_x1,s_x2,s_x3,s_x4,s_x5 = inputs_x1.cuda(), inputs_x2.cuda(), inputs_x3.cuda(), inputs_x4.cuda(), inputs_x5.cuda(), label_train.cuda(non_blocking=True), s_x1.cuda(),s_x2.cuda(),s_x3.cuda(),s_x4.cuda(),s_x5.cuda()
                net = net.cuda()
            m_s_x1,m_s_x2,m_s_x3,m_s_x4,m_s_x5,\
                s_x1_en,s_x2_en,s_x3_en,s_x4_en,s_x5_en,\
               x1_en,x2_en,x3_en,x4_en,x5_en,\
               x1_relation,x2_relation,x3_relation,x4_relation,x5_relation,\
               x1_en_relation,x2_en_relation,x3_en_relation,x4_en_relation,x5_en_relation,\
               x_view_relation,adj_label,adj_recover= net(s_x1,s_x2,s_x3,s_x4,s_x5,inputs_x1, inputs_x2, inputs_x3, inputs_x4, inputs_x5, 5)

            x1_feature.append(x_view_relation[0 * inputs_x1.size(0):1 * inputs_x1.size(0), :])
            x2_feature.append(x_view_relation[1 * inputs_x1.size(0):2 * inputs_x1.size(0), :])
            x3_feature.append(x_view_relation[2 * inputs_x1.size(0):3 * inputs_x1.size(0), :])
            x4_feature.append(x_view_relation[3 * inputs_x1.size(0):4 * inputs_x1.size(0), :])
            x5_feature.append(x_view_relation[4 * inputs_x1.size(0):5 * inputs_x1.size(0), :])
            label_all.append(label_train)

    x_view_final = (torch.cat(x1_feature).cpu().numpy() +
                    torch.cat(x2_feature).cpu().numpy() +
                    torch.cat(x3_feature).cpu().numpy() +
                    torch.cat(x4_feature).cpu().numpy() +
                    torch.cat(x5_feature).cpu().numpy()) / 5

    label_all = torch.cat(label_all).cpu().numpy()
    label_all = label_all.reshape(len(label_all), )

    estimator = KMeans(6)
    label_pred = estimator.fit_predict(x_view_final)

    acc_avg, acc_std = get_avg_acc(label_all, [label_pred], 1)
    nmi_avg, nmi_std = get_avg_nmi(label_all, [label_pred], 1)
    ri_avg, ri_std = get_avg_RI(label_all, [label_pred], 1)
    f1_avg, f1_std = get_avg_f1(label_all, [label_pred], 1)

    print('acc_avg: {acc_avg:.4f}\t'
          'acc_std: {acc_std:.4f}\t'
          'nmi_avg: {nmi_avg:.4f}\t'
          'nmi_std: {nmi_std:.4f}\t'
          'f1_avg:{f1_avg:.4f}\t'
          'f1_std {f1_std:.4f}\t'
          'ri_avg:{ri_avg:.4f}\t'
          'ri_std: {ri_std:.4f}'.format(acc_avg=acc_avg, acc_std=acc_std, nmi_avg=nmi_avg, nmi_std=nmi_std,
                                        f1_avg=f1_avg, f1_std=f1_std, ri_avg=ri_avg, ri_std=ri_std))


if __name__ == "__main__":
    # 进行预训练
    net = Multi_view_dynamic_relation()
    sectrained_model_path = 'D:/Study/RMPF/model_inf/second_trained_model.pth'
    try:
        net.load_state_dict(torch.load(sectrained_model_path))
        print(f'Successfully loaded pre-trained model from {sectrained_model_path}')
    except FileNotFoundError:
        print(f'Error: The pre-trained model file {sectrained_model_path} was not found.')
        net = None
    # 进行测试
    if net is not None:
        test(net)
        