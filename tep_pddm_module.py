"""
TEP Dataset PDDM Module with Information Entropy
=================================================
This module implements PDDM (Pairwise Distance Distribution Matching) for 
Tennessee Eastman Process (TEP) data using Shannon entropy instead of propensity scores.

Dataset Structure:
- Column 1: faultNumber (1-20 for faulty, 0 for fault-free)
- Column 2: simulationRun (1-500)
- Column 3: sample (1-500 for training, 1-960 for testing)
- Columns 4-55: 52 TEP variables
  - xmeas_1 to xmeas_22: 22 process measurement variables
  - xmeas_23 to xmeas_41: 19 component measurements
  - xmv_1 to xmv_11: 11 operational variables

For entropy calculation and similarity, we use:
- Process variables: xmeas_1 to xmeas_22 (22 variables)
- Component measurements: xmeas_23 to xmeas_39 (17 variables, excluding 2 result variables)
Total: 39 variables for entropy computation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
from sklearn.preprocessing import StandardScaler


def compute_shannon_entropy(x: np.ndarray, n_bins: int = 30) -> float:
    """
    计算单个样本的Shannon熵
    
    Args:
        x: 样本特征向量 (n_features,)
        n_bins: 直方图分箱数量
    
    Returns:
        Shannon熵值 H(X) = -Σ p(x) * log(p(x))
    """
    # 确保是1D数组
    x = x.flatten()
    
    # 计算直方图
    hist, _ = np.histogram(x, bins=n_bins, density=True)
    
    # 归一化为概率分布
    hist = hist + 1e-10  # 避免log(0)
    bin_width = (x.max() - x.min()) / n_bins
    prob = hist * bin_width
    prob = prob / prob.sum()  # 确保和为1
    
    # 计算Shannon熵
    entropy = -np.sum(prob * np.log(prob + 1e-10))
    
    return entropy


def compute_batch_entropy(X: np.ndarray, n_bins: int = 30) -> np.ndarray:
    """
    计算批量样本的Shannon熵
    
    Args:
        X: 样本特征矩阵 (n_samples, n_features)
        n_bins: 直方图分箱数量
    
    Returns:
        entropies: 每个样本的熵值 (n_samples,)
    """
    n_samples = X.shape[0]
    entropies = np.zeros(n_samples)
    
    for i in range(n_samples):
        entropies[i] = compute_shannon_entropy(X[i], n_bins)
    
    return entropies


def entropy_based_similarity(x_i: np.ndarray, x_j: np.ndarray, n_bins: int = 30) -> float:
    """
    基于原始变量之间的相似度
    
    相似度定义: s(i,j) = S(i,j)=0.75(|h_i+h_j/2-0.5|-|h_i-h_j/2+0.5|)
    熵差越小，相似度越高
    
    Args:
        x_i: 第一个样本 (n_features,)
        x_j: 第二个样本 (n_features,)
        n_bins: 直方图分箱数量
    
    Returns:
        similarity: 相似度分数 [0, 1]
    """
    H_i = compute_shannon_entropy(x_i, n_bins)
    H_j = compute_shannon_entropy(x_j, n_bins)
    
    # 相似度 = exp(-熵差的绝对值)
    similarity = 0.75 * (np.abs((H_i + H_j) / 2 - 0.5) - np.abs((H_i - H_j) / 2 + 0.5))
    
    return similarity


def find_three_pairs_entropy(
    X: np.ndarray,
    n_bins: int = 30,
    verbose: bool = False
) ->  Tuple[np.ndarray, np.ndarray]:
    """
    对于每一批训练样本中，使用香农熵确立几个关键样本点，选择三对样本:
    
     理论：
        - 前22个操控变量和后11个操控变量（考虑到19个成分测量具有显著的时间延迟）
    
    选择策略：
        Step 1: (x_i, x_j) - 中间对
            - x_i ∈ P, x_j ∈ M
            - 熵接近均值：|H_i - H_mean| + |H_j - H_mean| 最小
        
        Step 2: (x_k, x_l) - 操控变量极端对
            - x_k: M 中与 x_i 熵差最大
            - x_l: M 中与 x_k 熵差最大
        
        Step 3: (x_m, x_n) - 过程变量极端对
            - x_m: P 中与 x_j 熵差最大
            - x_n: P 中与 x_m 熵差最大
    
    Args:
        X: 特征矩阵 (n_samples, n_features) - 仅包含33个输入变量
        n_bins: 熵计算的分箱数
        verbose: 是否打印调试信息
    
    Returns:
        three_pairs: (6, n_features) 六个样本
        indices: (6,) 在原始数据中的索引
    """
     # 计算所有样本的香农熵
    entropies = compute_batch_entropy(X, n_bins)
    H_mean = entropies.mean()
    
    # 分离过程变量（P）和操控变量（M）
    I_P = X[:,3:22]     # 过程变量（P）
    I_M = X[:,-19:-1]  # 操控变量（M）
    
    if len(I_M) < 3 or len(I_P) < 3:
        # 样本不足，随机选择
        if verbose:
            print("Warning: 样本不足，随机选择")
        indices = np.random.choice(len(X), 6, replace=False)
        return X[indices], indices
    
    H_M = entropies[I_M]  # 操控变量的熵
    H_P = entropies[I_P]  # 过程变量的熵
    
    if verbose:
        print(f"香农熵统计:")
        print(f"  总体均值: {H_mean:.4f}")
        print(f"  操控变量 M (控制组): 均值={H_M.mean():.4f}, std={H_M.std():.4f}")
        print(f"  过程变量 P (治疗组): 均值={H_P.mean():.4f}, std={H_P.std():.4f}")
        # print(f"  验证格兰杰原理: H(M)={H_M.mean():.4f} {'<' if H_M.mean() < H_P.mean() else '>'} H(P)={H_P.mean():.4f}")
    
    # ========================================
    # Step 1: 选择中间对 (x_i, x_j)
    # ========================================
    # 目标: argmin_{i∈P, j∈M} |H_i - H_mean| + |H_j - H_mean|
    
    min_dist = np.inf
    idx_i_local, idx_j_local = 0, 0
    
    for i in range(len(I_P)):
        for j in range(len(I_M)):
            dist = abs(H_P                                                                                                                          [i] - H_mean) + abs(H_M[j] - H_mean)
            if dist < min_dist:
                min_dist = dist
                idx_i_local = i
                idx_j_local = j
    
    H_i = H_P[idx_i_local]
    H_j = H_M[idx_j_local]
    
    if verbose:
        print(f"\nStep 1: 中间对")
        print(f"  x_i (过程变量): H={H_i:.4f}, |H-mean|={abs(H_i-H_mean):.4f}")
        print(f"  x_j (操控变量): H={H_j:.4f}, |H-mean|={abs(H_j-H_mean):.4f}")
    
    # ========================================
    # Step 2: 选择操控变量极端对 (x_k, x_l)
    # ========================================
    # x_k: argmax_{k∈M} |H_k - H_i|
    idx_k_local = np.argmax(np.abs(H_M - H_i))
    H_k = H_M[idx_k_local]
    # x_l: argmax_{l∈M} |H_l - H_k|（但不是 x_k 自己）
    dist_to_k = np.abs(H_M - H_k)
    dist_to_k[idx_k_local] = -1  # 排除 x_k 自己
    idx_l_local = np.argmax(dist_to_k)
    H_l = H_M[idx_l_local]
    
    if verbose:
        print(f"\nStep 2: 操控变量极端对")
        print(f"  x_k: H={H_k:.4f}, |H_k-H_i|={abs(H_k-H_i):.4f}")
        print(f"  x_l: H={H_l:.4f}, |H_l-H_k|={abs(H_l-H_k):.4f}")
    
    # ========================================
    # Step 3: 选择过程变量极端对 (x_m, x_n)
    # ========================================
    # x_m: argmax_{m∈P} |H_m - H_j|
    idx_m_local = np.argmax(np.abs(H_P - H_j))
    H_m = H_P[idx_m_local]
    
    # x_n: argmax_{n∈P} |H_n - H_m|（但不是 x_m 自己）
    dist_to_m = np.abs(H_P - H_m)
    dist_to_m[idx_m_local] = -1  # 排除 x_m 自己
    idx_n_local = np.argmax(dist_to_m)
    H_n = H_P[idx_n_local]
    
    if verbose:
        print(f"\nStep 3: 过程变量极端对")
        print(f"  x_m: H={H_m:.4f}, |H_m-H_j|={abs(H_m-H_j):.4f}")
        print(f"  x_n: H={H_n:.4f}, |H_n-H_m|={abs(H_n-H_m):.4f}")
    
    # ========================================
    # 组装三对样本
    # ========================================
    indices = np.array([
        I_P[idx_i_local],  # x_i: 过程变量中间
        I_M[idx_j_local],  # x_j: 操控变量中间
        I_M[idx_k_local],  # x_k: 操控变量极端
        I_M[idx_l_local],  # x_l: 操控变量相似
        I_P[idx_m_local],  # x_m: 过程变量极端
        I_P[idx_n_local],  # x_n: 过程变量相似
    ])
    
    three_pairs = X[indices]
    
    return three_pairs, indices


# ============================================================================
# 3. 基于熵差计算相似度
# ============================================================================

def compute_three_pairs_similarity_granger(three_pairs: np.ndarray,
                                          n_bins: int = 10) -> np.ndarray:
    """
    计算三对样本的五个相似度
    
    Args:
        three_pairs: (6, n_features) [x_i, x_j, x_k, x_l, x_m, x_n]
        n_bins: 分箱数
    
    Returns:
        similarities: (5,) [S(k,l), S(m,n), S(k,m), S(i,k), S(j,m)]
    """
    similarities = np.zeros(5)
    
    # S(k, l): 操控变量极端对的相似度
    similarities[0] = entropy_based_similarity(
        three_pairs[2], three_pairs[3], n_bins
    )
    
    # S(m, n): 过程变量极端对的相似度
    similarities[1] = entropy_based_similarity(
        three_pairs[4], three_pairs[5], n_bins
    )
    
    # S(k, m): 跨变量极端对的相似度
    similarities[2] = entropy_based_similarity(
        three_pairs[2], three_pairs[4], n_bins
    )
    
    # S(i, k): 过程中间 vs 操控极端
    similarities[3] = entropy_based_similarity(
        three_pairs[0], three_pairs[2], n_bins
    )
    
    # S(j, m): 操控中间 vs 过程极端
    similarities[4] = entropy_based_similarity(
        three_pairs[1], three_pairs[4], n_bins
    )
    
    return similarities


class PDDMNetwork(nn.Module):
    """
    PDDM网络: 严格按照提供的复杂数学公式实现。
    
    输入: 两个潜在空间样本 z_i 和 z_j。
    输出: 预测的相似度分数 S_hat (线性输出)。
    
    数学描述:
    u = |z_i - z_j|
    v = |z_i + z_j| / 2  (保留了您之前描述中的绝对值)

    u_norm = u / ||u||_2
    v_norm = v / ||v||_2
    u_1 = ReLU(W_u * u_norm + b_u)
    v_1 = ReLU(W_v * v_norm + b_v)

    u_1_norm = u_1 / ||u_1||_2
    v_1_norm = v_1 / ||v_1||_2
    h = ReLU(W_c * [u_1_norm, v_1_norm]^T + b_c)

    S_hat = W_s * h + b_s
    """
    
    def __init__(self, latent_dim: int, u_v_hidden_dim: int = 32, h_hidden_dim: int = 64):
        """
        Args:
            latent_dim: 潜在空间维度 (即 z_i 的维度，例如 16)。
            u_v_hidden_dim: 用于 u_1 和 v_1 子网络的隐藏层维度。
            h_hidden_dim: 用于 h 层的隐藏层维度 (即 W_c 的输出维度)。
        """
        super(PDDMComplexNetwork, self).__init__()
        
        # 参数 W_u, b_u 用于计算 u_1
        self.W_u = nn.Linear(latent_dim, u_v_hidden_dim)
        # 参数 W_v, b_v 用于计算 v_1
        self.W_v = nn.Linear(latent_dim, u_v_hidden_dim)
        
        # 参数 W_c, b_c 用于计算 h
        # 注意: u_1 和 v_1 拼接后维度是 2 * u_v_hidden_dim
        self.W_c = nn.Linear(2 * u_v_hidden_dim, h_hidden_dim)
        
        # 参数 W_s, b_s 用于计算最终输出 S_hat
        self.W_s = nn.Linear(h_hidden_dim, 1) # 输出维度为 1 (相似度得分)
        
        self.relu = nn.ReLU()
        
    def _normalize_vector(self, x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
        """
        计算 L2 范数并归一化向量。
        处理 L2 范数为零的情况，避免除以零。
        """
        norm = torch.norm(x, p=2, dim=dim, keepdim=True)
        return x / (norm + eps) # 加上 eps 防止除以零

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            z_i: 第一个潜在空间样本 (batch_size, latent_dim)
            z_j: 第二个潜在空间样本 (batch_size, latent_dim)
        
        Returns:
            S_hat: 预测的相似度得分 (batch_size, 1)
        """
        # 1. 计算 u 和 v
        u = torch.abs(z_i - z_j)  # u = |z_i - z_j|
        v = torch.abs(z_i + z_j) / 2.0 # v = |z_i + z_j| / 2 (根据您的公式，对和也取了绝对值)
        
        # 2. 计算 u_1
        u_norm = self._normalize_vector(u) # u / ||u||_2
        u_1 = self.relu(self.W_u(u_norm))  # u_1 = ReLU(W_u * (u/||u||_2) + b_u)
        
        # 3. 计算 v_1
        v_norm = self._normalize_vector(v) # v / ||v||_2
        v_1 = self.relu(self.W_v(v_norm))  # v_1 = ReLU(W_v * (v/||v||_2) + b_v)
        
        # 4. 计算 h
        u_1_norm = self._normalize_vector(u_1) # u_1 / ||u_1||_2
        v_1_norm = self._normalize_vector(v_1) # v_1 / ||v_1||_2
        
        # 拼接归一化后的 u_1 和 v_1
        u_v_concat = torch.cat([u_1_norm, v_1_norm], dim=-1) # [u_1/||u_1||_2, v_1/||v_1||_2]^T
        h = self.relu(self.W_c(u_v_concat)) # h = ReLU(W_c * [...]^T + b_c)
        
        # 5. 计算最终输出 S_hat
        S_hat = self.W_s(h) # S_hat = W_s * h + b_s
        
        return S_hat

def compute_pddm_loss(
    pddm_net: PDDMNetwork,
    x_i: torch.Tensor,
    x_j: torch.Tensor,
    x_k: torch.Tensor,
    x_l: torch.Tensor,
    x_m: torch.Tensor,
    x_n: torch.Tensor,
    target_similarities: Dict[str, float]
) -> torch.Tensor:
    """
    计算PDDM损失函数
    
    损失 = MSE(预测相似度, 目标相似度)
    目标相似度由信息熵计算得到
    
    Args:
        pddm_net: PDDM网络
        x_i, x_j, x_k, x_l, x_m, x_n: 六个样本 (input_dim,) 或 (batch_size, input_dim)
        target_similarities: 目标相似度字典，包含5个值
    
    Returns:
        loss: PDDM损失值
    """
    # 预测相似度
    pred_s_ij = pddm_net(x_i, x_j).squeeze()
    pred_s_kl = pddm_net(x_k, x_l).squeeze()
    pred_s_mn = pddm_net(x_m, x_n).squeeze()
    pred_s_i_kl_k = pddm_net(x_i, x_k).squeeze()
    pred_s_i_kl_l = pddm_net(x_i, x_l).squeeze()
    pred_s_i_kl = (pred_s_i_kl_k + pred_s_i_kl_l) / 2.0
    pred_s_j_mn_m = pddm_net(x_j, x_m).squeeze()
    pred_s_j_mn_n = pddm_net(x_j, x_n).squeeze()
    pred_s_j_mn = (pred_s_j_mn_m + pred_s_j_mn_n) / 2.0
    
    # 目标相似度
    device = x_i.device
    target_s_ij = torch.tensor(target_similarities['s_ij'], device=device, dtype=torch.float32)
    target_s_kl = torch.tensor(target_similarities['s_kl'], device=device, dtype=torch.float32)
    target_s_mn = torch.tensor(target_similarities['s_mn'], device=device, dtype=torch.float32)
    target_s_i_kl = torch.tensor(target_similarities['s_i_kl'], device=device, dtype=torch.float32)
    target_s_j_mn = torch.tensor(target_similarities['s_j_mn'], device=device, dtype=torch.float32)
    
    # MSE损失
    loss = (
        F.mse_loss(pred_s_ij, target_s_ij) +
        F.mse_loss(pred_s_kl, target_s_kl) +
        F.mse_loss(pred_s_mn, target_s_mn) +
        F.mse_loss(pred_s_i_kl, target_s_i_kl) +
        F.mse_loss(pred_s_j_mn, target_s_j_mn)
    )
    
    return loss


def compute_mid_point_distance(
    phi_i: torch.Tensor,
    phi_j: torch.Tensor,
    distance_type: str = 'euclidean'
) -> torch.Tensor:
    """
    计算表示层中间对的距离
    
    用于全局平衡，确保正常样本和故障样本在表示空间中的距离适中
    
    Args:
        phi_i: 正常样本的表示 (representation_dim,) 或 (batch_size, representation_dim)
        phi_j: 故障样本的表示 (representation_dim,) 或 (batch_size, representation_dim)
        distance_type: 距离类型，'euclidean'或'cosine'
    
    Returns:
        distance: 距离值
    """
    if distance_type == 'euclidean':
        # 欧氏距离
        distance = torch.norm(phi_i - phi_j, p=2)
    elif distance_type == 'cosine':
        # 余弦距离 = 1 - 余弦相似度
        cosine_sim = F.cosine_similarity(phi_i, phi_j, dim=-1)
        distance = 1.0 - cosine_sim
    else:
        raise ValueError(f"不支持的距离类型: {distance_type}")
    
    return distance


def compute_representation_similarity_loss(
    pddm_net: PDDMNetwork,
    phi_i: torch.Tensor,
    phi_j: torch.Tensor,
    phi_k: torch.Tensor,
    phi_l: torch.Tensor,
    phi_m: torch.Tensor,
    phi_n: torch.Tensor,
    target_similarities: Dict[str, float]
) -> torch.Tensor:
    """
    计算表示层的PDDM相似度损失
    
    这个函数在表示层(representation layer)而非输入层计算PDDM损失
    
    Args:
        pddm_net: PDDM网络
        phi_i, phi_j, phi_k, phi_l, phi_m, phi_n: 六个样本的表示向量
        target_similarities: 目标相似度(由输入层熵计算得到)
    
    Returns:
        loss: 表示层PDDM损失
    """
    return compute_pddm_loss(
        pddm_net, phi_i, phi_j, phi_k, phi_l, phi_m, phi_n,
        target_similarities
    )


class TEPDataPreprocessor:
    """
    TEP数据预处理器
    
    功能:
    1. 提取39个输入变量 (xmeas_1到xmeas_22 + xmeas_23到xmeas_39)
    2. 标准化特征
    3. 准备故障标签
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # TEP变量名称
        self.process_vars = [f'xmeas_{i}' for i in range(1, 23)]  # xmeas_1 to xmeas_22
        self.component_vars = [f'xmeas_{i}' for i in range(23, 40)]  # xmeas_23 to xmeas_39 (排除40,41)
        self.input_vars = self.process_vars + self.component_vars  # 总共39个变量
    
    def fit(self, df):
        """
        拟合标准化器
        
        Args:
            df: pandas DataFrame，包含TEP数据
        """
        X = df[self.input_vars].values
        self.scaler.fit(X)
        self.is_fitted = True
        return self
    
    def transform(self, df):
        """
        转换数据
        
        Args:
            df: pandas DataFrame，包含TEP数据
        
        Returns:
            X: 标准化后的输入特征 (n_samples, 39)
            fault_labels: 故障标签 (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("预处理器未拟合，请先调用fit()")
        
        X = df[self.input_vars].values
        X_scaled = self.scaler.transform(X)
        fault_labels = df['faultNumber'].values
        
        return X_scaled, fault_labels
    
    def fit_transform(self, df):
        """
        拟合并转换数据
        
        Args:
            df: pandas DataFrame，包含TEP数据
        
        Returns:
            X: 标准化后的输入特征 (n_samples, 39)
            fault_labels: 故障标签 (n_samples,)
        """
        self.fit(df)
        return self.transform(df)


def prepare_tep_data_for_pddm(
    df,
    n_bins: int = 30,
    fault_free_label: int = 0
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    准备TEP数据用于PDDM训练
    
    完整流程:
    1. 提取39个输入变量并标准化
    2. 计算信息熵并选择三对样本
    3. 计算目标相似度
    
    Args:
        df: pandas DataFrame，TEP数据
        n_bins: 熵计算分箱数
        fault_free_label: 正常工况标签
    
    Returns:
        samples: 六个样本 (6, 39)
        fault_labels: 对应的故障标签 (6,)
        target_similarities: 目标相似度字典
    """
    # 数据预处理
    preprocessor = TEPDataPreprocessor()
    X, fault_labels = preprocessor.fit_transform(df)
    
    # 选择三对样本
    x_i, x_j, x_k, x_l, x_m, x_n = find_three_pairs_entropy(
        X, fault_labels, n_bins, fault_free_label
    )
    
    # 计算目标相似度
    target_similarities = compute_three_pairs_similarity_entropy(
        x_i, x_j, x_k, x_l, x_m, x_n, n_bins
    )
    
    # 组织返回结果
    samples = np.stack([x_i, x_j, x_k, x_l, x_m, x_n], axis=0)
    
    return samples, fault_labels, target_similarities


# ==================== 使用示例 ====================

def example_usage():
    """
    使用示例 - 展示如何在TEP数据上使用PDDM
    
    注意: 这只是示例框架，实际使用时需要加载真实的TEP数据
    """
    import pandas as pd
    
    # 假设你已经加载了TEP数据
    # df = pd.read_csv('tep_data.csv')
    
    print("=" * 60)
    print("TEP PDDM模块使用示例")
    print("=" * 60)
    
    # 示例1: 数据预处理
    print("\n1. 数据预处理")
    print("-" * 60)
    print("变量配置:")
    print("  - 过程变量: xmeas_1 到 xmeas_22 (22个)")
    print("  - 成分变量: xmeas_23 到 xmeas_39 (17个)")
    print("  - 总输入维度: 39")
    print("  - 排除变量: xmeas_40, xmeas_41 (结果变量)")
    
    # 示例2: PDDM网络初始化
    print("\n2. PDDM网络初始化")
    print("-" * 60)
    input_dim = 39  # TEP输入维度
    hidden_dim = 64
    pddm_net = PDDMNetwork(input_dim, hidden_dim)
    print(f"PDDM网络结构:")
    print(f"  - 输入维度: {input_dim} × 2 = {input_dim * 2} (u和v拼接)")
    print(f"  - 隐藏层维度: {hidden_dim}")
    print(f"  - 输出维度: 1 (相似度分数)")
    print(f"  - 参数量: {sum(p.numel() for p in pddm_net.parameters())}")
    
    # 示例3: 训练流程说明
    print("\n3. 训练流程")
    print("-" * 60)
    print("步骤:")
    print("  Step 1: 数据预处理 - 提取39个输入变量，标准化")
    print("  Step 2: 计算信息熵 - 为每个样本计算Shannon熵")
    print("  Step 3: 选择三对样本 - 基于Granger因果原理")
    print("    - 中间对(x_i, x_j): 一个正常+一个故障，熵接近均值")
    print("    - 正常极端对(x_k, x_l): 两个正常工况样本")
    print("    - 故障极端对(x_m, x_n): 两个故障工况样本")
    print("  Step 4: 计算目标相似度 - 使用熵差计算5个相似度分数")
    print("  Step 5: PDDM训练 - 最小化预测相似度与目标相似度的MSE")
    
    # 示例4: 损失函数组成
    print("\n4. 总损失函数")
    print("-" * 60)
    print("Total Loss = α·Pred_Loss + β·PDDM_Loss + γ·Mid_Distance")
    print("  - Pred_Loss: 预测损失 (如分类交叉熵)")
    print("  - PDDM_Loss: 相似度匹配损失 (5个MSE项之和)")
    print("  - Mid_Distance: 中间对距离损失 (全局平衡)")
    print("  - α, β, γ: 超参数权重")
    
    print("\n" + "=" * 60)
    print("模块准备完成！")
    print("=" * 60)


if __name__ == "__main__":
    example_usage()
