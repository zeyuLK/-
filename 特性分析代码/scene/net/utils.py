# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 16:56:12 2021

@author: danli
"""

import numpy as np
import torch
import scipy.sparse as sp

def relaition_matrix (x):
    N=x.size(0)
    tmp = torch.norm(x, 2, 1);
    x_norm = x / tmp.view(N, -1);
    x_relation=torch.mm( x_norm,x_norm.t())  
    #x_relation=x_relation*(x_relation>=0).float()
    x_relation=torch.clamp(x_relation,min=0.0)
    return x_relation

import torch

import torch

class MatrixA:
    def __init__(self, matrix1, matrix2, device, scale=10):
        # 设备 + 精度
        self.matrix1 = matrix1.to(device).float()
        self.matrix2 = matrix2.to(device).float()
        self.device = device
        self.scale = scale

        # 归一化
        norms1 = torch.norm(self.matrix1, dim=1, keepdim=True) + 1e-8
        self.matrix1_norm = self.matrix1 / norms1
        
        norms2 = torch.norm(self.matrix2, dim=1, keepdim=True) + 1e-8
        self.matrix2_norm = self.matrix2 / norms2

        # 相似度矩阵
        self.S1 = torch.mm(self.matrix1_norm, self.matrix1_norm.t())
        self.S2 = torch.mm(self.matrix2_norm, self.matrix2_norm.t())
        
        self.k = 1.1 * torch.mean(self.S1)

    def process_matrices(self):
        smooth1 = torch.sigmoid(self.scale * (self.S1 - self.k))
        smooth2 = torch.sigmoid(self.scale * (self.S2 - self.k))

        intersection = smooth1 * smooth2
        union = smooth1 + smooth2 - intersection
        jaccard = torch.where(union != 0, intersection / union, torch.zeros_like(union))

        A = torch.sigmoid(self.scale * (jaccard - 0.5))
        

        
        # ✅ 正确：非原地操作，保留梯度
        A = torch.where(A < 0.1, torch.zeros_like(A), A)

        return A

#归一化的邻接矩阵
def preprocess_graph(adj):
    D = torch.pow(adj.sum(1).float(), -1).flatten()
    D = torch.diag(D)
    adj = torch.matmul(D,adj)
    return torch_sparse_tensor(adj)


def torch_sparse_tensor(adj_all_view):
    
    idx = torch.nonzero(adj_all_view).T  
    data = adj_all_view[idx[0],idx[1]]
    #coo_a = torch.sparse_coo_tensor(idx, data, adj_all_view.shape)
    return torch.sparse_coo_tensor(idx, data, adj_all_view.shape)