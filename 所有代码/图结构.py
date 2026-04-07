import scipy.io as sio
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ===============================
# 1. 读取数据
# ===============================
data = sio.loadmat('dataset/Caltech101-7.mat')
X = data['X'][0]

views = [torch.tensor(X[i]).float() for i in range(len(X))]

print("视图数量:", len(views))
print("样本数:", views[0].shape[0])


# ===============================
# 2. Jaccard 方法
# ===============================
class MatrixA:
    def __init__(self, matrix1, matrix2, scale=10):
        self.matrix1 = matrix1
        self.matrix2 = matrix2
        self.scale = scale

        self.matrix1 = self.matrix1 / (torch.norm(self.matrix1, dim=1, keepdim=True) + 1e-8)
        self.matrix2 = self.matrix2 / (torch.norm(self.matrix2, dim=1, keepdim=True) + 1e-8)

        self.S1 = torch.mm(self.matrix1, self.matrix1.t())
        self.S2 = torch.mm(self.matrix2, self.matrix2.t())

        self.k = 1.1 * torch.mean(self.S1)

    def process(self):
        smooth1 = torch.sigmoid(self.scale * (self.S1 - self.k))
        smooth2 = torch.sigmoid(self.scale * (self.S2 - self.k))

        intersection = smooth1 * smooth2
        union = smooth1 + smooth2 - intersection

        jaccard = torch.where(union != 0, intersection / union, torch.zeros_like(union))

        A = torch.sigmoid(self.scale * (jaccard - 0.5))
        A[A < 0.1] = 0   # 稀疏化

        return A


# ===============================
# 3. Intersection 方法
# ===============================
class IntersectionGraph:
    def __init__(self, views, k=10):
        self.views = views
        self.k = k
        self.N = views[0].shape[0]

    def cosine(self, X):
        X = X / (torch.norm(X, dim=1, keepdim=True) + 1e-8)
        return torch.mm(X, X.t())

    def knn(self, S):
        A = torch.zeros_like(S)
        _, idx = torch.topk(S, self.k, dim=1)
        for i in range(self.N):
            A[i, idx[i]] = 1
        return A

    def construct(self):
        S_list, A_list = [], []

        for v in self.views:
            S = self.cosine(v)
            A = self.knn(S)
            S_list.append(S)
            A_list.append(A)

        A_fused = torch.zeros_like(A_list[0])

        for i in range(len(A_list)):
            for j in range(i+1, len(A_list)):
                inter = A_list[i] * A_list[j]
                weight = (S_list[i] + S_list[j]) / 2
                A_fused += inter * weight

        return A_fused


# ===============================
# 4. 构造邻接矩阵
# ===============================
A_jaccard = MatrixA(views[0], views[1]).process()
A_intersection = IntersectionGraph(views).construct()


# ===============================
# 5. 指标计算
# ===============================
def sparsity(A):
    return (A < 1e-3).float().mean().item()

def connectivity(A):
    return A.sum(dim=1).mean().item()

def information(A):
    edges = (A > 1e-3).float().sum().item()
    weight = A.sum().item()
    return edges, weight


# ===============================
# 6. 打印分析
# ===============================
def analyze(A, name):
    print(f"==== {name} ====")
    print("稀疏性:", sparsity(A))
    print("平均度:", connectivity(A))
    e, w = information(A)
    print("边数量:", e)
    print("总权重:", w)
    print()

analyze(A_jaccard, "Jaccard")
analyze(A_intersection, "Intersection")


# ===============================
# 7. 排序函数
# ===============================
def sort_matrix(A):
    idx = torch.argsort(A.sum(dim=1), descending=True)
    return A[idx][:, idx]


# ===============================
# 8. 热力图（中文）
# ===============================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
sns.heatmap(sort_matrix(A_jaccard).cpu().numpy(), cmap="viridis", xticklabels=False, yticklabels=False)
plt.title("Jaccard邻接矩阵")

plt.subplot(1, 2, 2)
sns.heatmap(sort_matrix(A_intersection).cpu().numpy(), cmap="viridis", xticklabels=False, yticklabels=False)
plt.title("邻域交集邻接矩阵")

plt.tight_layout()
plt.show()


# ===============================
# 9. 指标对比图（一个画布🔥）
# ===============================
labels = ["稀疏性", "平均度", "边数量", "总权重"]

jaccard_values = [
    sparsity(A_jaccard),
    connectivity(A_jaccard),
    information(A_jaccard)[0],
    information(A_jaccard)[1],
]

intersection_values = [
    sparsity(A_intersection),
    connectivity(A_intersection),
    information(A_intersection)[0],
    information(A_intersection)[1],
]

x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(6,4))
plt.bar(x - width/2, jaccard_values, width, label="Jaccard方法")
plt.bar(x + width/2, intersection_values, width, label="邻域交集方法")

plt.xticks(x, labels)
plt.legend()
plt.title("邻接矩阵结构指标对比")
plt.tight_layout()
plt.show()

