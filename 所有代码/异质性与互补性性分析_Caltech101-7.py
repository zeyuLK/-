import scipy.io as sio

data = sio.loadmat('dataset\Caltech101-7.mat')

keys_list = data.keys()

X = data['X']   # 多视图特征
views = data['X'][0]   # 取出所有视图,视图数量为6
X1 =views[0]
X2=views[1]
Y = data['Y']   # 标签
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel, pairwise_distances
from sklearn.manifold import TSNE

# ===============================
# 0. 数据预处理
# ===============================
# views = data['X'][0]
# Y = data['Y']

# 转换标签为一维
Y = Y.ravel()

# 标准化每个视图（非常关键！）
X_views = []
for v in views:
    X = v.astype(np.float64)
    X = StandardScaler().fit_transform(X)
    X_views.append(X)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import rbf_kernel, pairwise_distances


# ===============================
# 1️⃣ PCA统一子空间（核心步骤）
# ===============================
def pca_align_views(X_views, dim=40):
    Z_views = []

    for X in X_views:
        # 标准化（必须）
        X = StandardScaler().fit_transform(X)

        # PCA降维到统一维度
        pca = PCA(n_components=dim)
        Z = pca.fit_transform(X)

        Z_views.append(Z)

    return Z_views


# ===============================
# 2️⃣ MMD计算
# ===============================
def compute_mmd(X, Y):
    # 自适应核宽度（推荐）
    dists = pairwise_distances(X)
    sigma = np.median(dists)

    gamma = 1 / (2 * sigma**2)

    Kxx = rbf_kernel(X, X, gamma=gamma)
    Kyy = rbf_kernel(Y, Y, gamma=gamma)
    Kxy = rbf_kernel(X, Y, gamma=gamma)

    mmd = Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()
    return mmd


# ===============================
# 3️⃣ MMD热力图
# ===============================
def plot_mmd_heatmap(Z_views):
    n = len(Z_views)
    M = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            M[i, j] = compute_mmd(Z_views[i], Z_views[j])

    plt.figure()
    sns.heatmap(
        M,
        annot=True,
        fmt=".3f",
        cmap='viridis',
        xticklabels=[f'V{i+1}' for i in range(n)],
        yticklabels=[f'V{i+1}' for i in range(n)]
    )

    plt.title("MMD Distance Between Views (PCA-Aligned)")
    plt.xlabel("View")
    plt.ylabel("View")
    plt.show()


# ===============================
# 🚀 主流程
# ===============================
# views = data['X'][0]

# Step 1: PCA对齐
Z_views = pca_align_views(X_views, dim=40)

# Step 2: MMD热力图
plot_mmd_heatmap(Z_views)



# ===============================
# 3️⃣ t-SNE分布图
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings("ignore")  # 屏蔽无关警告
# 解决中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
def plot_tsne(X_views, Y):
    # 创建 2行3列 的大图，调整画布大小，给颜色条预留空间
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
    axes = axes.flatten()  # 展平方便索引

    # 遍历每个视图画图
    for i, X in enumerate(X_views):
        tsne = TSNE(
            n_components=2,
            perplexity=30,
            random_state=42,
            learning_rate=200
        )
        X_emb = tsne.fit_transform(X)

        # 画在第 i 个子图上
        scatter = axes[i].scatter(
            X_emb[:, 0],
            X_emb[:, 1],
            c=Y,
            s=12,
            cmap="viridis"
        )

        # 全中文标题 + 坐标轴
        axes[i].set_title(f'视图 {i+1} 的 t-SNE 分布', fontsize=14, pad=12)
        axes[i].set_xlabel('t-SNE 维度 1', fontsize=12, labelpad=8)
        axes[i].set_ylabel('t-SNE 维度 2', fontsize=12, labelpad=8)
        axes[i].grid(alpha=0.3)
        axes[i].tick_params(labelsize=10)

    # 🔧 关键修改：调整颜色条位置，避免遮挡
    # 调整子图布局，给右侧颜色条留出空间
    plt.subplots_adjust(right=0.92)
    # 创建独立的颜色条轴，位置：[左, 下, 宽, 高]，完全不遮挡子图
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    # 添加颜色条，设置标签和字体
    cbar = fig.colorbar(scatter, cax=cbar_ax)
    cbar.set_label('类别标签', fontsize=12, labelpad=10)
    cbar.ax.tick_params(labelsize=10)

    plt.tight_layout(rect=[0, 0, 0.9, 1])  # 再次优化布局，避免挤压
    plt.show()

# ===============================
# 🚀 主调用
# ===============================
plot_tsne(X_views, Y)


