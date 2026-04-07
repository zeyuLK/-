

import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter

#selector-mask
class S(nn.Module):#选择模块
    def __init__(self, in_dim, n_hidden_1, out_dim):
        super(S, self).__init__()
        self.layer1 = nn.Sequential(#是pytorch中的一个容器，将多个模块组合在一起，依次通过每一个操作
            nn.Linear(in_dim, n_hidden_1),#全连接层，输入特征的维度映射到输出特征的维度
            nn.BatchNorm1d(n_hidden_1), #批量归一化，每个特征维度进行归一化，但特征维度不变
            nn.Sigmoid()#激活函数
            )
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_1, out_dim), nn.Sigmoid())
        self.dropout=nn.Dropout(0.1)
    def forward(self, x):#定义数据通过网络进行前向传播
        x1 = self.layer1(x)#将输入数据x通过layer1层输出x1
        mask_s = self.layer3(x1)
        return mask_s

#encoder-view-specific
class E(nn.Module):
    def __init__(self, in_dim, n_hidden_1,out_dim):
        super(E, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1), 
            nn.BatchNorm1d(n_hidden_1),
            nn.Tanh())
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1,out_dim), 
                                    nn.BatchNorm1d(out_dim))
    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        return x2

#GCN-layer
class GraphConvolution(nn.Module):#图卷积层
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()#调用父类的构造函数，将子模块和参数注册到父模块中
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        #当一个张量被声明为Parameter时，它会自动被添加到模型的参数列表中，在反向传播中自动计算梯度，并通过优化器更新权重
        #构造一个形状为in_features*out_features的张量，初始化为随机值
        self.reset_parameters()#初始化权重

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        #在pytorch中有多种初始化方法，xavier_uniform_是其中一种，初始化权重.适用于tanh激活函数和sigmoid激活函数
        #xavier_uniform_是一种比较常用的初始化方法，它会根据输入和输出的维度自动调整最合适的范围
        #xavier_uniform_初始化方法是从均匀分布中采样，均匀分布的范围是[-a,a]，a=根号(6/(in_features+out_features))

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        #dropout操作，防止过拟合，输入张量，丢弃概率，训练模式。
        #self.training 是一个布尔属性，用于指示当前模型是否处于训练模式。
        #它是一个内置属性，由 PyTorch 的 nn.Module 类管理。所以说要继承父类
        #当你调用 model.train() 时，self.training 被设置为 True；
        #当你调用 model.eval() 时，self.training 被设置为 False。
        support = torch.mm(input, self.weight)
        #torch.mm 是 PyTorch 中的矩阵乘法函数，用于计算两个矩阵的乘积
        output = torch.sparse.mm(adj, support)
        #torch.sparse.mm 是 PyTorch 中的稀疏矩阵乘法函数，用于计算两个稀疏矩阵的乘积
        #稀疏矩阵是指矩阵中大部分元素为0的矩阵，稀疏矩阵乘法是指两个稀疏矩阵相乘
        #adj是邻接矩阵，support是输入特征矩阵，output是输出特征矩阵
        output = self.act(output)
        return output

    def __repr__(self):#用于返回类的字符串表示
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
    #当你打印一个 GraphConvolution 对象时，__repr__ 方法会被调用，
    #返回的字符串将显示在控制台或日志中。
#GCN-network

class GCNModelAE(nn.Module):#GCN模型
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GCNModelAE, self).__init__()
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        #lambda x: x 是一个匿名函数，也就是说这个函数没有名字，只有参数和返回值
        #这里的意思是，act是一个激活函数，激活函数的输入是x，输出也是x，也就是说不对x做任何处理
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

    def forward(self, x, adj):
        x1 = self.gc1(x, adj)
        x2 = self.gc2(x1, adj)

        return x2, self.dc(x2)


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(MatrixA(z, z, z.device).process_matrices())
        return adj

#在 GCNModelAE 中，使用两层 GraphConvolution 和一个 InnerProductDecoder 构成了一个完整的 GCN 模型。
#其中，GraphConvolution 是 GCN 的核心层，用于学习节点的表示；
#InnerProductDecoder 是一个解码器，用于预测节点之间的关系。
#在 forward 方法中，首先通过两层 GraphConvolution 学习节点的表示，然后通过 InnerProductDecoder 预测节点之间的关系。
#在 forward 方法中，x 是节点的特征矩阵，adj 是邻接矩阵，z 是 GCN 学习到的节点表示，adj_recover 是 GCN 预测的邻接矩阵。
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




