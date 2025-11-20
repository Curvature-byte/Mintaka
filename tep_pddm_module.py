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


def compute_variable_entropy(X: np.ndarray, n_bins: int = 30) -> np.ndarray:
    """
    计算批量样本的Shannon熵
    
    Args:
        X: 样本特征矩阵 (n_samples, n_features)
        n_bins: 直方图分箱数量
    
    Returns:
        entropies: 每个样本的熵值 (n_samples,)
    """
    n_features = X.shape[1]
    entropies = np.zeros(n_features)
    
    for j in range(n_features):
        entropies[j] = compute_shannon_entropy(X[:,j], n_bins)
    
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
    x_i = x_i.flatten()
    x_j = x_j.flatten()
    
    # 1. 计算欧几里得距离的平方
    distance_sq = np.sum((x_i - x_j)**2)
    
    # 2. 通过高斯核 (RBF Kernel) 转换为相似度
    similarity = np.exp(-distance_sq / (2 * sigma**2))
    
    return similarity

def _find_closest_row_idx(X: np.ndarray, col_idx: int) -> int:
    """
    为指定列 (col_idx) 找到中位数，
    并返回 X 中最接近该中位数的值 所在的 "行索引"。
    """
    column_data = X[:, col_idx]
    median_val = np.median(column_data)
    
    # 找到该列中，哪个值与中位数最接近，并返回那一行的索引
    row_idx = np.argmin(np.abs(column_data - median_val))
    return int(row_idx)

# ============================================================================
# 3. 查找三对样本 
# ============================================================================
def find_three_pairs_entropy(
    X: np.ndarray,
    n_bins: int = 30,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    对于每一批训练样本中，使用香农熵确立几个关键样本点，选择三对样本:
    
    【修改后的新逻辑】:
    1.  按“列”计算熵 H(j)，而不是按“行”。
    2.  P 和 M 是“列”的集合 (P: 0-21, M: 22-32)。
    3.  算法 (Step 1-3) 用于选择6个代表性的“列索引”。
    4.  对每个选定的“列”，找到该列的中位数。
    5.  选择“行”，该行的值最接近其对应列的中位数。
    
    Args:
        X: 特征矩阵 (n_samples, n_features) 
           【重要假设】: X 仅包含33个输入变量，
           并且前22个 (0-21) 是P组, 后11个 (22-32) 是M组。
        n_bins: 熵计算的分箱数
        verbose: 是否打印调试信息
    
    Returns:
        three_pairs: (6, n_features) 六个被选中的样本(行)
        indices: (6,) 在原始数据中的行索引
    """
    n_samples, n_features = X.shape

    # 假设: X 包含 33 个输入变量 (22 P-vars, 11 M-vars)
    if n_features < 33:
         print(f"Warning: 输入 X 只有 {n_features} 个变量, 预期 33 个。将使用随机选择。")
         indices = np.random.choice(n_samples, 6, replace=False)
         return X[indices], indices
    
    # 定义P和M的 *列索引*
    # 假设 X = [xmeas_1...xmeas_22, xmv_1...xmv_11]
    P_col_indices = np.arange(0, 22) # 过程变量 (P组, 22个)
    M_col_indices = np.arange(22, 33) # 操控变量 (M组, 11个)

    # 1. 计算所有 *列* 的香农熵 (有意义的熵)
    all_col_entropies = compute_variable_entropy(X, n_bins) # Shape (n_features,)
    
    H_P = all_col_entropies[P_col_indices] # Shape (22,)
    H_M = all_col_entropies[M_col_indices] # Shape (11,)
    
    # 仅使用 P 和 M 变量的熵来计算均值
    H_mean = all_col_entropies[np.concatenate([P_col_indices, M_col_indices])].mean()

    if verbose:
        print(f"香农熵统计 (按列计算):")
        print(f"  总体均值 (P+M): {H_mean:.4f}")
        print(f"  操控变量 M (列 22-32): 均值={H_M.mean():.4f}, std={H_M.std():.4f}")
        print(f"  过程变量 P (列 0-21): 均值={H_P.mean():.4f}, std={H_P.std():.4f}")

    # ==========================================================
    # Step 1: 选择中间对 (x_i, x_j) - [选择 列]
    # ==========================================================
    min_dist = np.inf
    idx_i_local, idx_j_local = 0, 0
    
    for i in range(len(H_P)):
        for j in range(len(H_M)):
            dist = abs(H_P[i] - H_mean) + abs(H_M[j] - H_mean)
            if dist < min_dist:
                min_dist = dist
                idx_i_local = i # P组中的局部 *列* 索引
                idx_j_local = j # M组中的局部 *列* 索引

    H_i_col = H_P[idx_i_local]
    H_j_col = H_M[idx_j_local]
    
    if verbose:
        print(f"\nStep 1: 中间对 (选择 列)")
        print(f"  列 i (P组, 局部索引 {idx_i_local}): H={H_i_col:.4f}")
        print(f"  列 j (M组, 局部索引 {idx_j_local}): H={H_j_col:.4f}")

    # ==========================================================
    # Step 2: 选择操控变量极端对 (x_k, x_l) - [选择 列]
    # ==========================================================
    idx_k_local = np.argmax(np.abs(H_M - H_i_col))
    H_k_col = H_M[idx_k_local]
    
    dist_to_k = np.abs(H_M - H_k_col)
    dist_to_k[idx_k_local] = -1 # 排除自己
    idx_l_local = np.argmax(dist_to_k)
    H_l_col = H_M[idx_l_local]
    
    if verbose:
        print(f"\nStep 2: 操控变量极端对 (选择 列)")
        print(f"  列 k (M组, 局部索引 {idx_k_local}): H={H_k_col:.4f}")
        print(f"  列 l (M组, 局部索引 {idx_l_local}): H={H_l_col:.4f}")

    # ==========================================================
    # Step 3: 选择过程变量极端对 (x_m, x_n) - [选择 列]
    # ==========================================================
    idx_m_local = np.argmax(np.abs(H_P - H_j_col))
    H_m_col = H_P[idx_m_local]
    
    dist_to_m = np.abs(H_P - H_m_col)
    dist_to_m[idx_m_local] = -1 # 排除自己
    idx_n_local = np.argmax(dist_to_m)
    H_n_col = H_P[idx_n_local]
    
    if verbose:
        print(f"\nStep 3: 过程变量极端对 (选择 列)")
        print(f"  列 m (P组, 局部索引 {idx_m_local}): H={H_m_col:.4f}")
        print(f"  列 n (P组, 局部索引 {idx_n_local}): H={H_n_col:.4f}")
        
    # ==========================================================
    # Step 4: 将局部 "列索引" 转换为 全局 "列索引"
    # ==========================================================
    selected_col_indices = {
        'i': P_col_indices[idx_i_local], # 对应的全局列索引 (0-21)
        'j': M_col_indices[idx_j_local], # 对应的全局列索引 (22-32)
        'k': M_col_indices[idx_k_local],
        'l': M_col_indices[idx_l_local],
        'm': P_col_indices[idx_m_local],
        'n': P_col_indices[idx_n_local],
    }
    if verbose:
        print(f"\nStep 4: 选定的全局列索引: {selected_col_indices}")

    # ==========================================================
    # Step 5: 根据选定的 "列", 选择最接近中位数的 "行"
    # ==========================================================
    
    # 注意：这里可能会选到同一行，这是正常的。
    # 如果想避免重复，可以增加额外逻辑，但目前先按最接近中位数处理。
    selected_row_indices = [
        _find_closest_row_idx(X, selected_col_indices['i']), # 寻找 x_i 对应的行
        _find_closest_row_idx(X, selected_col_indices['j']), # 寻找 x_j 对应的行
        _find_closest_row_idx(X, selected_col_indices['k']), # 寻找 x_k 对应的行
        _find_closest_row_idx(X, selected_col_indices['l']), # 寻找 x_l 对应的行
        _find_closest_row_idx(X, selected_col_indices['m']), # 寻找 x_m 对应的行
        _find_closest_row_idx(X, selected_col_indices['n']), # 寻找 x_n 对应的行
    ]
    
    indices = np.array(selected_row_indices)
    
    if verbose:
         print(f"\nStep 5: 选定的最终行索引: {indices}")
         # 检查是否有重复
         if len(np.unique(indices)) != 6:
             print(f"Warning: 选中的行索引有重复: {indices}")

    # 组装样本 (行)
    three_pairs = X[indices] # Shape (6, n_features)
    
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
        super(PDDMNetwork, self).__init__()
        
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
    three_pairs: np.ndarray,
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
    pred_s_kl = pddm_net(three_pairs[2], three_pairs[3]).squeeze()
    pred_s_mn = pddm_net(three_pairs[4], three_pairs[5]).squeeze()
    pred_s_km = pddm_net(three_pairs[2], three_pairs[4]).squeeze()
    pred_s_ik = pddm_net(three_pairs[0], three_pairs[2]).squeeze()
    pred_s_jm = pddm_net(three_pairs[1], three_pairs[4]).squeeze()
    
    # 目标相似度
    target_similarities= compute_three_pairs_similarity_granger(three_pairs,10)
    
    target_similarities = torch.tensor(target_similarities, dtype=torch.float32)


    # MSE损失
    loss = (
        F.mse_loss(pred_s_kl, target_similarities[0]) +
        F.mse_loss(pred_s_mn, target_similarities[1]) +
        F.mse_loss(pred_s_km, target_similarities[2]) +
        F.mse_loss(pred_s_ik, target_similarities[3]) +
        F.mse_loss(pred_s_jm, target_similarities[4])
    )
    
    return loss / 5.0