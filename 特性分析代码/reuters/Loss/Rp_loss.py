# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 16:43:56 2021

@author: danli
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
#一个较早期的接口，用于包装张量（torch.Tensor），并提供自动梯度计算的功能
from torch.autograd.function import Function



# import sys
# sys.path.append('.');

class MultiView_all_loss(nn.Module):
    def __init__(self,round=2):

        super(MultiView_all_loss, self).__init__()
        self.loss_fn = nn.MSELoss(reduction='mean')
        #对所有的元素求平均
        self.round=round
        
    def forward(self, m_s_x1,m_s_x2,m_s_x3,m_s_x4,m_s_x5,\
                x1,x2,x3,x4,x5,\
                s_x1_en,s_x2_en,s_x3_en,s_x4_en,s_x5_en,\
               x1_en,x2_en,x3_en,x4_en,x5_en,\
               x1_relation,x2_relation,x3_relation,x4_relation,x5_relation,\
               x1_en_relation,x2_en_relation,x3_en_relation,x4_en_relation,x5_en_relation,\
               x_view_relation,adj_label,adj_recover,sigma,alpha=0.2, lam=10.0):
       
        # ---- Step 1: 计算每个 view 的 MSE ----
        mse1 = F.mse_loss(m_s_x1, x1)
        mse2 = F.mse_loss(m_s_x2, x2)
        mse3 = F.mse_loss(m_s_x3, x3)
        mse4 = F.mse_loss(m_s_x4, x4)
        mse5 = F.mse_loss(m_s_x5, x5)
        
        # ---- Step 2: 计算每个 view 的阈值 ----
        th1, th2, th3, th4, th5 = [alpha * s for s in sigma]
        
        # ---- Step 3: 惩罚项 ----
        p1 = torch.relu(mse1 - th1)
        p2 = torch.relu(mse2 - th2)
        p3 = torch.relu(mse3 - th3)
        p4 = torch.relu(mse4 - th4)
        p5 = torch.relu(mse5 - th5)
        
        # ---- Step 4: 总 loss ----
        loss1 = (mse1 + lam * p1 +
                mse2 + lam * p2 +
                mse3 + lam * p3 +
                mse4 + lam * p4 +
                mse5 + lam * p5) / 5.0
        

        mse1_f = F.mse_loss(s_x1_en, x1_en)
        mse2_f = F.mse_loss(s_x2_en, x2_en)
        mse3_f = F.mse_loss(s_x3_en, x3_en)
        mse4_f = F.mse_loss(s_x4_en, x4_en)
        mse5_f = F.mse_loss(s_x5_en, x5_en)
        
        # ---- Step 2: 计算每个 view 的阈值 ----
       
         
        # ---- Step 4: 总 loss ----
        loss2 = (mse1_f + mse2_f + mse3_f + mse4_f + mse5_f)/5



        #学习稀疏特征表示
        N=x1_relation.size(0)
        mask_dui_jiao_0 = (torch.ones(N ,N) - torch.eye(N, N)).cuda()
        #创建一个对角线为0的矩阵
        dynamic_relation_loss= torch.FloatTensor([0]).cuda();
        #一个包含单个元素0的张量
        for view in range(1, 5 + 1):
            # 动态生成变量名
            relation_var = eval(f'x{view}_relation')
            en_relation_var = eval(f'x{view}_en_relation')
            
            # 应用掩码
            relation_var = relation_var * mask_dui_jiao_0
            en_relation_var = en_relation_var * mask_dui_jiao_0
            
            # 计算动态关系损失
            dynamic_relation_loss =dynamic_relation_loss+ (
                torch.mean(torch.pow(torch.clamp(relation_var * (relation_var >= 0.5).float() - en_relation_var * (en_relation_var >= 0.5).float(), min=0.0), 2)) +
                torch.mean(torch.pow(relation_var * (relation_var < 0.5).float() * (relation_var >= 0.2).float() - en_relation_var * (en_relation_var < 0.5).float() * (en_relation_var >= 0.2).float(), 2)) +
                torch.mean(torch.pow(torch.clamp(en_relation_var * (en_relation_var < 0.2).float() - relation_var * (relation_var < 0.2).float(), min=0.0), 2))
            )
        loss=loss1+loss2+dynamic_relation_loss
        ########################################deocder_loss
        #F.binary_cross_entropy_with_logits 是 PyTorch 中的一个函数，用于计算二元交叉熵损失。
        #adj_recover是网络预测的邻接矩阵，adj_label是真实的邻接矩阵
        loss_3 = F.binary_cross_entropy_with_logits(adj_recover, adj_label)
        
        if self.round==1:
            return loss
        else:
            total_loss=loss+loss_3
            return total_loss,loss,loss_3

