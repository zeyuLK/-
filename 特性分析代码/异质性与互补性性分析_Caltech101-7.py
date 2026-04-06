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


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import rbf_kernel, pairwise_distances


# ===============================
# 1️⃣ PCA统一子空间（必须）



# ===============================
# 2️⃣ HSIC计算
# ===============================
def compute_hsic(X, Y):
    n = X.shape[0]

    # 自适应核宽度
    dists = pairwise_distances(X)
    sigma = np.median(dists)
    gamma = 1 / (2 * sigma**2 + 1e-8)

    # 核矩阵
    K = rbf_kernel(X, X, gamma=gamma)
    L = rbf_kernel(Y.reshape(-1,1), Y.reshape(-1,1), gamma=gamma)

    # 中心化
    H = np.eye(n) - np.ones((n, n)) / n
    Kc = H @ K @ H
    Lc = H @ L @ H

    hsic = np.trace(Kc @ Lc) / ((n - 1) ** 2)
    return hsic


# ===============================
# 3️⃣ 互补性矩阵（核心🔥）
# ===============================
def compute_complementarity_matrix(Z_views, Y):
    n = len(Z_views)
    M = np.zeros((n, n))

    # 先算所有单视图HSIC（避免重复算）
    hsic_single = []
    for i in range(n):
        h = compute_hsic(Z_views[i], Y)
        hsic_single.append(h)

    # 两两组合
    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            X1 = Z_views[i]
            X2 = Z_views[j]

            X_multi = np.concatenate([X1, X2], axis=1)

            h_multi = compute_hsic(X_multi, Y)

            # Δ定义
            M[i, j] = h_multi - min(hsic_single[i],hsic_single[j])

    return M, hsic_single


# ===============================
# 4️⃣ 画互补性热力图
# ===============================
def plot_complementarity_matrix(M):
    n = M.shape[0]

    plt.figure()
    sns.heatmap(
        M,
        annot=True,
        fmt=".2e",
        cmap='coolwarm',
        center=0,
        xticklabels=[f'V{i+1}' for i in range(n)],
        yticklabels=[f'V{i+1}' for i in range(n)]
    )

    plt.title("Complementarity Matrix (HSIC-based)")
    plt.xlabel("View")
    plt.ylabel("View")
    plt.show()


# ===============================
# 5️⃣ 输出最优互补组合（加分项🔥）
# ===============================
def print_top_pairs(M, top_k=5):
    n = M.shape[0]

    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            pairs.append((i, j, M[i, j]))

    # 按Δ排序
    pairs = sorted(pairs, key=lambda x: x[2], reverse=True)

    print("\nTop Complementary View Pairs:")
    for i in range(min(top_k, len(pairs))):
        vi, vj, val = pairs[i]
        print(f"View {vi+1} & View {vj+1}: Δ = {val:.6f}")


# ===============================
# 🚀 主流程（直接调用）
# ===============================
# views = data['X'][0]
# Y = data['Y']

Y = Y.ravel()



# Step2: 计算互补矩阵
M, hsic_single = compute_complementarity_matrix(X_views, Y)

# Step3: 画图
plot_complementarity_matrix(M)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel

# ===============================
# HSIC 计算函数
# ===============================
def compute_hsic(X, Y, kernel='rbf', sigma=None):
    n = X.shape[0]
    if kernel == 'rbf':
        K = rbf_kernel(X, X, gamma=1.0/(2*sigma**2)) if sigma else rbf_kernel(X, X)
        L = rbf_kernel(Y, Y, gamma=1.0/(2*sigma**2)) if sigma else rbf_kernel(Y, Y)
    else:
        K = linear_kernel(X, X)
        L = linear_kernel(Y, Y)
    H = np.eye(n) - 1.0 / n * np.ones((n, n))
    hsic_value = np.trace(K @ H @ L @ H) / (n ** 2)
    return hsic_value

# ===============================
# 绘制 6视图 HSIC 热力图
# ===============================
def plot_hsic_heatmap(X_views):
    n_views = len(X_views)
    hsic_matrix = np.zeros((n_views, n_views))

    # 计算所有视图两两之间的 HSIC
    for i in range(n_views):
        for j in range(n_views):
            hsic_matrix[i, j] = compute_hsic(X_views[i], X_views[j])

    # 画图
    plt.figure(figsize=(10, 8))
    im = plt.imshow(hsic_matrix, cmap='Purples', interpolation='nearest')

    # 设置标签
    plt.title('多视图 HSIC 相关性热力图', fontsize=16)
    plt.xticks(range(n_views), [f'视图{i+1}' for i in range(n_views)], fontsize=12)
    plt.yticks(range(n_views), [f'视图{i+1}' for i in range(n_views)], fontsize=12)

    # 显示数值
    for i in range(n_views):
        for j in range(n_views):
            plt.text(j, i, f'{hsic_matrix[i, j]:.3f}',
                    ha="center", va="center", fontsize=11, color='black')

    # 颜色条
    cbar = plt.colorbar(im)
    cbar.set_label('HSIC 值（越大相关性越强）', fontsize=12)

    plt.tight_layout()
    plt.show()

# ===============================
# 🚀 直接调用画图
# ===============================
plot_hsic_heatmap(X_views)


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel

# ===============================
# HSIC 计算函数
# ===============================
def compute_hsic(X, Y, kernel='rbf'):
    n = X.shape[0]

    # 对 X（视图）用 RBF 核
    K = rbf_kernel(X, X)

    # 对 Y（标签）用 线性核 / 独热核（标签必须这样算才对）
    # 标签是一维离散类别，用简单核即可
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            L[i, j] = 1 if Y[i] == Y[j] else 0

    H = np.eye(n) - 1.0 / n
    hsic_val = np.trace(K @ H @ L @ H) / (n ** 2)
    return hsic_val

# ===============================
# 绘制：视图 ↔ 标签 HSIC 柱状图（论文最常用）
# ===============================
def plot_view_label_hsic(X_views, Y):
    n_views = len(X_views)
    hsic_scores = []

    for v in range(n_views):
        hs = compute_hsic(X_views[v], Y)
        hsic_scores.append(hs)

    # 画图
    plt.figure(figsize=(8, 5))
    bars = plt.bar(
        [f'视图{i+1}' for i in range(n_views)],
        hsic_scores,
        color='#6a9bd1'
    )

    # 显示数值
    for bar, score in zip(bars, hsic_scores):
        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.0001,
            f'{score:.4f}',
            ha='center', fontsize=11
        )

    plt.title('各视图与标签的 HSIC 相关性', fontsize=14)
    plt.ylabel('HSIC 值（越大判别性越强）', fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# ===============================
# 🚀 运行
# ===============================
plot_view_label_hsic(X_views, Y)



import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler  # 标准化必备

# ===============================
# HSIC 计算函数
# ===============================
def compute_hsic_label(X, Y):
    n = X.shape[0]

    # 特征核矩阵
    K = rbf_kernel(X, X)

    # 标签核矩阵（同类为1）
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            L[i, j] = 1.0 if Y[i] == Y[j] else 0.0

    H = np.eye(n) - 1.0 / n
    hsic_val = np.trace(K @ H @ L @ H) / (n ** 2)
    return hsic_val

# ===============================
# ✅ 标准化 + 拼接 + PCA90% + HSIC 热力图
# ===============================
def plot_concat_standard_pca90_hsic(X_views, Y):
    n_views = len(X_views)
    hsic_matrix = np.zeros((n_views, n_views))

    for i in range(n_views):
        for j in range(n_views):
            # 1. 视图两两拼接
            concat = np.concatenate([X_views[i], X_views[j]], axis=1)

            # 2. ✅ 标准化（必须！PCA 前提）
            scaler = StandardScaler()
            concat_std = scaler.fit_transform(concat)

            # 3. ✅ PCA 自动保留 90% 信息
            pca = PCA(n_components=0.9)
            concat_pca = pca.fit_transform(concat_std)

            # 4. 计算 HSIC
            hsic_matrix[i, j] = compute_hsic_label(concat_pca, Y)

    # ================== 画图 ==================
    plt.figure(figsize=(10, 8))
    im = plt.imshow(hsic_matrix, cmap='Blues')

    plt.title('两两视图拼接+标准化+PCA(90%) 与标签 HSIC 热力图', fontsize=16)
    plt.xticks(range(n_views), [f'视图{i+1}' for i in range(n_views)], fontsize=12)
    plt.yticks(range(n_views), [f'视图{i+1}' for i in range(n_views)], fontsize=12)

    for i in range(n_views):
        for j in range(n_views):
            plt.text(j, i, f'{hsic_matrix[i, j]:.3f}', ha="center", va="center", fontsize=11)

    cbar = plt.colorbar(im)
    cbar.set_label('HSIC（越高判别性越强）', fontsize=12)
    plt.tight_layout()
    plt.show()

# ===============================
# 🚀 运行（完全正确版）
# ===============================
plot_concat_standard_pca90_hsic(X_views, Y)


import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ===============================
# HSIC 计算函数（特征 ↔ 标签）
# ===============================
def compute_hsic_label(X, Y):
    n = X.shape[0]
    K = rbf_kernel(X, X)  # 特征核矩阵

    # 标签核矩阵：同类为1，不同为0
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            L[i, j] = 1.0 if Y[i] == Y[j] else 0.0

    H = np.eye(n) - 1.0 / n
    hsic_val = np.trace(K @ H @ L @ H) / (n ** 2)
    return hsic_val

# ===============================
# 🚀 主验证代码（你要的所有组合）
# ===============================
def verify_all_hsic(X_views, Y):
    v1 = X_views[0]
    v2 = X_views[1]

    # ----------------------
    # 1. 视图1 各种情况
    # ----------------------
    hsic_v1 = compute_hsic_label(v1, Y)

    scaler1 = StandardScaler()
    v1_std = scaler1.fit_transform(v1)
    hsic_v1_std = compute_hsic_label(v1_std, Y)

    pca1 = PCA(n_components=0.9)
    v1_pca = pca1.fit_transform(v1_std)
    hsic_v1_pca = compute_hsic_label(v1_pca, Y)

    # ----------------------
    # 2. 视图2 各种情况
    # ----------------------
    hsic_v2 = compute_hsic_label(v2, Y)

    scaler2 = StandardScaler()
    v2_std = scaler2.fit_transform(v2)
    hsic_v2_std = compute_hsic_label(v2_std, Y)

    pca2 = PCA(n_components=0.9)
    v2_pca = pca2.fit_transform(v2_std)
    hsic_v2_pca = compute_hsic_label(v2_pca, Y)

    # ----------------------
    # 3. 视图1 + 视图2 拼接组合
    # ----------------------
    # 原始直接拼接
    concat_raw = np.concatenate([v1, v2], axis=1)
    hsic_concat_raw = compute_hsic_label(concat_raw, Y)

    # 分别标准化后拼接
    concat_std = np.concatenate([v1_std, v2_std], axis=1)
    hsic_concat_std = compute_hsic_label(concat_std, Y)

    # 分别PCA后拼接
    concat_pca_sep = np.concatenate([v1_pca, v2_pca], axis=1)
    hsic_concat_pca_sep = compute_hsic_label(concat_pca_sep, Y)

    # 标准化后拼接 → 再整体PCA（最标准方法）
    concat_std_pca = PCA(n_components=0.9).fit_transform(concat_std)
    hsic_concat_std_pca = compute_hsic_label(concat_std_pca, Y)

    # ===============================
    # 输出所有结果（清晰排版）
    # ===============================
    print("=" * 70)
    print("【视图1 与标签 HSIC】")
    print(f"原始视图1        : {hsic_v1:.6f}")
    print(f"标准化视图1      : {hsic_v1_std:.6f}")
    print(f"PCA(90%)视图1    : {hsic_v1_pca:.6f}")
    print("-" * 70)

    print("【视图2 与标签 HSIC】")
    print(f"原始视图2        : {hsic_v2:.6f}")
    print(f"标准化视图2      : {hsic_v2_std:.6f}")
    print(f"PCA(90%)视图2    : {hsic_v2_pca:.6f}")
    print("-" * 70)

    print("【视图1+2 拼接组合 与标签 HSIC】")
    print(f"原始直接拼接     : {hsic_concat_raw:.6f}")
    print(f"分别标准化后拼接 : {hsic_concat_std:.6f}")
    print(f"分别PCA后拼接    : {hsic_concat_pca_sep:.6f}")
    print(f"标准化拼接→整体PCA: {hsic_concat_std_pca:.6f}")
    print("=" * 70)

# ===============================
# 运行验证
# ===============================
verify_all_hsic(X_views, Y)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ===============================
# HSIC 计算函数
# ===============================
def compute_hsic_label(X, Y):
    n = X.shape[0]
    K = rbf_kernel(X, X)

    L = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            L[i, j] = 1.0 if Y[i] == Y[j] else 0.0

    H = np.eye(n) - 1.0 / n
    hsic_val = np.trace(K @ H @ L @ H) / (n ** 2)
    return hsic_val

# ===============================
# 标准化 + PCA 40维
# ===============================
def standard_pca_40d(X):
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    pca = PCA(n_components=40)
    X_pca = pca.fit_transform(X_std)
    return X_pca

# ===============================
# 1. 6个视图统一40维 → HSIC热力图
# ===============================
def plot_6view_hsic_40d(X_views, Y):
    n_views = len(X_views)
    hsic_scores = []

    for v in X_views:
        v_40d = standard_pca_40d(v)
        hsic = compute_hsic_label(v_40d, Y)
        hsic_scores.append(hsic)

    hsic_matrix = np.array([hsic_scores])

    plt.figure(figsize=(12, 3))
    im = plt.imshow(hsic_matrix, cmap='Blues', vmin=0)

    plt.xticks(range(n_views), [f'视图{i+1}' for i in range(n_views)], fontsize=12)
    plt.yticks([0], ['标签 Y'], fontsize=12)
    plt.title('6视图 标准化+PCA40维 与标签 HSIC 热力图', fontsize=14)

    for i in range(n_views):
        plt.text(i, 0, f'{hsic_scores[i]:.4f}', ha='center', va='center', fontsize=12, color='black')

    plt.colorbar(im)
    plt.tight_layout()
    plt.show()

# ===============================
# 运行
# ===============================
plot_6view_hsic_40d(X_views, Y)

# ===============================
# 2. 两两视图拼接 → 统一40维 → HSIC热力图
# ===============================
def plot_pair_concat_hsic_40d(X_views, Y):
    n_views = len(X_views)
    hsic_matrix = np.zeros((n_views, n_views))

    for i in range(n_views):
        for j in range(n_views):
            # 拼接
            concat = np.concatenate([X_views[i], X_views[j]], axis=1)
            # 统一流程：标准化 + PCA40
            concat_40d = standard_pca_40d(concat)
            # 计算HSIC
            hsic_matrix[i, j] = compute_hsic_label(concat_40d, Y)

    plt.figure(figsize=(10, 8))
    im = plt.imshow(hsic_matrix, cmap='Blues', vmin=0)

    plt.title('两两视图拼接+标准化+PCA40维 与标签 HSIC 热力图', fontsize=16)
    plt.xticks(range(n_views), [f'视图{i+1}' for i in range(n_views)], fontsize=12)
    plt.yticks(range(n_views), [f'视图{i+1}' for i in range(n_views)], fontsize=12)

    for i in range(n_views):
        for j in range(n_views):
            plt.text(j, i, f'{hsic_matrix[i, j]:.3f}', ha='center', va='center', fontsize=11)

    plt.colorbar(im, label='HSIC 值（越高判别性越强）')
    plt.tight_layout()
    plt.show()

# ===============================
# 运行
# ===============================
plot_pair_concat_hsic_40d(X_views, Y)