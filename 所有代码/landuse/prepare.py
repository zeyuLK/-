# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 15:06:20 2021

@author: danli
"""
import torch
import torch.nn as nn
from net.utils import relaition_matrix,preprocess_graph,MatrixA
from net.rmpf_network import S,E,GCNModelAE
class MatrixA:
    def __init__(self, matrix1, matrix2, device, scale=10):
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
    
class Multi_view_dynamic_relation(nn.Module):
    def __init__(self):
        super(Multi_view_dynamic_relation, self).__init__()
        #50 50

        self.S_net1 = S(20,20,20);
        self.E_net1 = E(20,30,30);
        ##########################
        self.S_net2 = S(59,59,59);
        self.E_net2 = E(59,25,30);
        #########################
        self.S_net3 = S(40,40,40);
        self.E_net3 = E(40,25,30);

        ############################
        self.relation_net=GCNModelAE(30,21,21,0)
        
        

    def forward(self,s_x1,s_x2,s_x3,x1,x2,x3,view_num):
        m_s_x1= self.S_net1(s_x1).clone();
        m_x1= self.S_net1(x1).clone();
        s_x1_en=self.E_net1(x1)
        x1_en=self.E_net1(m_x1)
        x1_relation = MatrixA(x1,x1, x1.device,scale=10).process_matrices()
        #x1_en_relation = relaition_matrix(x1_en)
        #x1_relation = relaition_matrix(x1)
        x1_en_relation = MatrixA(x1_en,x1_en, x1_en.device,scale=10).process_matrices()
        ######################################
        m_s_x2= self.S_net2(s_x2).clone();
        m_x2= self.S_net2(x2).clone();
        s_x2_en=self.E_net2(x2)
        x2_en=self.E_net2(m_x2)
        x2_relation = MatrixA(x2,x2, x2.device,scale=10).process_matrices()
        #x1_en_relation = relaition_matrix(x1_en)
        #x1_relation = relaition_matrix(x1)
        x2_en_relation = MatrixA(x2_en,x2_en, x2_en.device,scale=10).process_matrices()
        #########################################
        m_s_x3= self.S_net3(s_x3).clone();
        m_x3= self.S_net3(x3).clone();
        s_x3_en=self.E_net3(x3)
        x3_en=self.E_net3(m_x3)
        x3_relation = MatrixA(x3,x3, x3.device,scale=10).process_matrices()
        #x1_en_relation = relaition_matrix(x1_en)
        #x1_relation = relaition_matrix(x1)
        x3_en_relation = MatrixA(x3_en,x3_en, x3_en.device,scale=10).process_matrices()
        ##########################################
        
        ##########################################
        MMfeature = torch.cat([x1_en,x2_en,x3_en], dim=0) #图卷积网络的样本
        num_=x1_en.size(0)
        # adj_view=torch.zeros((view_num*num_,view_num*num_)).cuda()
        # for view in range(1,view_num+1):
        #     adj_view[(view-1)*num_:view*num_,(view-1)*num_:view*num_]=locals()['x{0}_en_relation'.format(view)]

        relations = {
                        1: x1_en_relation,
                        2: x2_en_relation,
                        3: x3_en_relation
                    }

        adj_view = torch.zeros((view_num * num_, view_num * num_)).cuda()
        for view in range(1, view_num + 1):
                adj_view[(view-1)*num_:view*num_, (view-1)*num_:view*num_] = relations[view]
        adj_view[0*num_:1*num_, 1*num_:2*num_]=adj_view[1*num_:2*num_, 0*num_:1*num_]= MatrixA(x1_en,x2_en, x1_en.device,scale=10).process_matrices()
        adj_view[0*num_:1*num_, 2*num_:3*num_]=adj_view[2*num_:3*num_, 0*num_:1*num_]= MatrixA(x1_en,x3_en, x1_en.device,scale=10).process_matrices()
        adj_view[1*num_:2*num_, 2*num_:3*num_]=adj_view[2*num_:3*num_, 1*num_:2*num_]= MatrixA(x2_en,x3_en, x1_en.device,scale=10).process_matrices()
        adj_label=adj_view                   
        x_view_relation,adj_recover=self.relation_net(MMfeature,adj_label)
       
                      

        #########################################
        
        return m_s_x1,m_s_x2,m_s_x3,\
                s_x1_en,s_x2_en,s_x3_en,\
               x1_en,x2_en,x3_en,\
               x1_relation,x2_relation,x3_relation,\
               x1_en_relation,x2_en_relation,x3_en_relation,\
               x_view_relation,adj_label,adj_recover
              