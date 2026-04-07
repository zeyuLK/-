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

class MatrixA:
    def __init__(self, matrix1, matrix2, device, scale=10,k=0.5):
        self.matrix1 = matrix1.to(device).float()
        self.matrix2 = matrix2.to(device).float()
        self.scale = scale

        norms1 = torch.norm(self.matrix1, dim=1, keepdim=True) + 1e-8
        self.matrix1_normalized = self.matrix1 / norms1

        norms2 = torch.norm(self.matrix2, dim=1, keepdim=True) + 1e-8
        self.matrix2_normalized = self.matrix2 / norms2

        self.S1 = torch.mm(self.matrix1_normalized, self.matrix1_normalized.t())
        self.S2 = torch.mm(self.matrix2_normalized, self.matrix2_normalized.t())
        self.k = torch.mean(self.S1)
    def process_matrices(self):
        # 平滑激活
        smooth_matrix1 = torch.sigmoid(self.scale * (self.S1 - self.k))
        smooth_matrix2 = torch.sigmoid(self.scale * (self.S2 - self.k))

        # Jaccard 相似度
        intersection = smooth_matrix1 * smooth_matrix2
        union = smooth_matrix1 + smooth_matrix2 - intersection
        jaccard_similarity = torch.where(
            union != 0, intersection / union, torch.zeros_like(union)
        )

        # 再次平滑激活
        jaccard_similarity = torch.sigmoid(self.scale * (jaccard_similarity - 0.5))
        return jaccard_similarity

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