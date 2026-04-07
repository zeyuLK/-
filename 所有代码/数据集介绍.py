import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics.pairwise import rbf_kernel, pairwise_distances

# ===============================
# 全局参数（可调）
# ===============================
PCA_DIM = 40
HSIC_SAMPLE_SIZE = 300  # 防止内存爆炸

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ===============================
# 1️⃣ 加载数据
# ===============================
data = sio.loadmat('dataset/Caltech101-7.mat')
views = data['X'][0]
Y = data['Y'].ravel()

# One-Hot 标签（关键）
encoder = OneHotEncoder(sparse_output=False)
Y_oh = encoder.fit_transform(Y.reshape(-1, 1))


# ===============================
# 2️⃣ PCA统一子空间
# ===============================
def preprocess_views(views, dim=PCA_DIM):
    Z_views = []
    for Xv in views:
        Xv = Xv.astype(np.float64)

        # 标准化
        Xv = StandardScaler().fit_transform(Xv)

        # PCA降维
        Z = PCA(n_components=dim, random_state=42).fit_transform(Xv)

        Z_views.append(Z)
    return Z_views


# ===============================
# 3️⃣ HSIC（RBF核 + 降采样）
# ===============================
def compute_hsic(X, Y):
    n = X.shape[0]

    # 降采样（避免内存爆炸）
    if n > HSIC_SAMPLE_SIZE:
        idx = np.random.choice(n, HSIC_SAMPLE_SIZE, replace=False)
        X = X[idx]
        Y = Y[idx]
        n = X.shape[0]

    # 自适应核宽度
    dists = pairwise_distances(X)
    sigma = np.median(dists)
    gamma = 1 / (2 * sigma**2 + 1e-8)

    K = rbf_kernel(X, X, gamma=gamma)
    L = rbf_kernel(Y, Y, gamma=gamma)

    H = np.eye(n) - np.ones((n, n)) / n
    Kc = H @ K @ H
    Lc = H @ L @ H

    return np.trace(Kc @ Lc) / ((n - 1) ** 2)


# ===============================
# 4️⃣ NHSIC（归一化）
# ===============================
def compute_nhsic(X, Y):
    h_xy = compute_hsic(X, Y)
    h_xx = compute_hsic(X, X)
    h_yy = compute_hsic(Y, Y)

    return h_xy / np.sqrt(h_xx * h_yy + 1e-8)


# ===============================
# 5️⃣ 单视图重要性
# ===============================
def plot_view_importance(Z_views, Y):
    scores = [compute_nhsic(X, Y) for X in Z_views]

    plt.figure(figsize=(6, 4))
    plt.bar(range(1, len(scores)+1), scores)
    plt.xlabel("视图")
    plt.ylabel("NHSIC")
    plt.title("视图重要性（与标签相关性）")
    plt.tight_layout()
    plt.show()

    return scores


# ===============================
# 6️⃣ 视图间冗余（HSIC矩阵）
# ===============================
def compute_view_hsic_matrix(Z_views):
    n = len(Z_views)
    M = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            M[i, j] = compute_nhsic(Z_views[i], Z_views[j])

    return M


def plot_hsic_matrix(M):
    plt.figure(figsize=(7, 5))
    sns.heatmap(M, annot=True, fmt=".2f", cmap="coolwarm",
                xticklabels=[f'V{i+1}' for i in range(M.shape[0])],
                yticklabels=[f'V{i+1}' for i in range(M.shape[0])])
    plt.title("视图间相关性（HSIC）")
    plt.tight_layout()
    plt.show()


# ===============================
# 7️⃣ 互补性矩阵（核心🔥）
# ===============================
def compute_complementarity_matrix(Z_views, Y):
    n = len(Z_views)
    M = np.zeros((n, n))

    hsic_single = [compute_nhsic(X, Y) for X in Z_views]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            X1 = Z_views[i]
            X2 = Z_views[j]

            # 拼接
            X_multi = np.concatenate([X1, X2], axis=1)

            # ⭐ 关键：再降维（防止HSIC下降）
            X_multi = PCA(n_components=PCA_DIM, random_state=42).fit_transform(X_multi)

            h_multi = compute_nhsic(X_multi, Y)

            # ⭐ 互补性定义（稳定版）
            M[i, j] = h_multi - min(hsic_single[i] , hsic_single[j]) 

    return M


def plot_complementarity_matrix(M):
    plt.figure(figsize=(7, 5))
    sns.heatmap(M, annot=True, fmt=".3f", cmap="coolwarm", center=0,
                xticklabels=[f'V{i+1}' for i in range(M.shape[0])],
                yticklabels=[f'V{i+1}' for i in range(M.shape[0])])
    plt.title("互补性矩阵（信息增益）")
    plt.tight_layout()
    plt.show()


# ===============================
# 🚀 主程序
# ===============================
if __name__ == '__main__':

    # Step1：对齐
    Z_views = preprocess_views(views)

    # Step2：单视图能力
    scores = plot_view_importance(Z_views, Y_oh)

    # Step3：冗余性
    M_hsic = compute_view_hsic_matrix(Z_views)
    plot_hsic_matrix(M_hsic)

    # Step4：互补性
    M_comp = compute_complementarity_matrix(Z_views, Y_oh)
    plot_complementarity_matrix(M_comp)

    # Step5：整体互补性（加分项🔥）
    print("平均互补性:", np.mean(M_comp))