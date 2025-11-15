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
    基于信息熵的相似度计算
    
    相似度定义: s(i,j) = exp(-|H(x_i) - H(x_j)|)
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
    similarity = np.exp(-np.abs(H_i - H_j))
    
    return similarity


def find_three_pairs_entropy(
    X: np.ndarray,
    fault_labels: np.ndarray,
    n_bins: int = 30,
    fault_free_label: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    基于Granger因果和信息熵选择三对样本
    
    TEP数据中:
    - fault_free_label (0): 正常工况 (类比为对照组/操纵变量M)
    - fault_labels (1-20): 故障工况 (类比为处理组/过程变量P)
    
    根据Granger因果原理: H(M) ≤ H(P)
    正常工况的熵应该小于等于故障工况的熵
    
    三步选择过程:
    Step 1: 选择中间对 (x_i, x_j) - 一个正常，一个故障，熵值接近均值
    Step 2: 选择正常工况极端对 (x_k, x_l) - 两个都是正常工况
    Step 3: 选择故障工况极端对 (x_m, x_n) - 两个都是故障工况
    
    Args:
        X: 特征矩阵 (n_samples, n_features) - 仅包含39个输入变量
        fault_labels: 故障标签 (n_samples,) - 0表示正常，1-20表示故障类型
        n_bins: 熵计算的分箱数
        fault_free_label: 正常工况标签，默认为0
    
    Returns:
        x_i, x_j: 中间对 (正常-故障混合对)
        x_k, x_l: 正常工况极端对
        x_m, x_n: 故障工况极端对
    """
    # 计算所有样本的Shannon熵
    entropies = compute_batch_entropy(X, n_bins)
    H_mean = entropies.mean()
    
    # 分离正常工况(M)和故障工况(P)
    normal_mask = (fault_labels == fault_free_label)
    fault_mask = ~normal_mask
    
    normal_indices = np.where(normal_mask)[0]
    fault_indices = np.where(fault_mask)[0]
    
    # 检查是否有足够的样本
    if len(normal_indices) < 2 or len(fault_indices) < 2:
        raise ValueError(f"需要至少2个正常样本和2个故障样本。当前: {len(normal_indices)}个正常, {len(fault_indices)}个故障")
    
    # Step 1: 选择中间对 (x_i正常, x_j故障)
    # 目标: |H_i - H_mean| + |H_j - H_mean| 最小
    min_distance = float('inf')
    best_i, best_j = None, None
    
    for i in normal_indices:
        for j in fault_indices:
            distance = np.abs(entropies[i] - H_mean) + np.abs(entropies[j] - H_mean)
            if distance < min_distance:
                min_distance = distance
                best_i, best_j = i, j
    
    x_i = X[best_i]
    x_j = X[best_j]
    H_i = entropies[best_i]
    H_j = entropies[best_j]
    
    # Step 2: 选择正常工况极端对 (x_k, x_l)
    # x_k: 与x_i熵差最大的正常样本
    # x_l: 与x_k熵差最大的正常样本
    normal_available = [idx for idx in normal_indices if idx != best_i]
    
    k_idx = max(normal_available, key=lambda idx: np.abs(entropies[idx] - H_i))
    x_k = X[k_idx]
    H_k = entropies[k_idx]
    
    normal_available = [idx for idx in normal_available if idx != k_idx]
    if len(normal_available) > 0:
        l_idx = max(normal_available, key=lambda idx: np.abs(entropies[idx] - H_k))
    else:
        # 如果只剩一个正常样本，使用它
        l_idx = k_idx
    x_l = X[l_idx]
    
    # Step 3: 选择故障工况极端对 (x_m, x_n)
    # x_m: 与x_j熵差最大的故障样本
    # x_n: 与x_m熵差最大的故障样本
    fault_available = [idx for idx in fault_indices if idx != best_j]
    
    m_idx = max(fault_available, key=lambda idx: np.abs(entropies[idx] - H_j))
    x_m = X[m_idx]
    H_m = entropies[m_idx]
    
    fault_available = [idx for idx in fault_available if idx != m_idx]
    if len(fault_available) > 0:
        n_idx = max(fault_available, key=lambda idx: np.abs(entropies[idx] - H_m))
    else:
        # 如果只剩一个故障样本，使用它
        n_idx = m_idx
    x_n = X[n_idx]
    
    return x_i, x_j, x_k, x_l, x_m, x_n


def compute_three_pairs_similarity_entropy(
    x_i: np.ndarray,
    x_j: np.ndarray,
    x_k: np.ndarray,
    x_l: np.ndarray,
    x_m: np.ndarray,
    x_n: np.ndarray,
    n_bins: int = 30
) -> Dict[str, float]:
    """
    计算三对样本之间的熵相似度
    
    返回5个相似度分数:
    - s_ij: 中间对内部相似度 (正常-故障)
    - s_kl: 正常工况对内部相似度
    - s_mn: 故障工况对内部相似度
    - s_i_kl: 中间对正常样本与正常极端对的相似度 (期望高)
    - s_j_mn: 中间对故障样本与故障极端对的相似度 (期望高)
    
    Args:
        x_i, x_j: 中间对
        x_k, x_l: 正常工况极端对
        x_m, x_n: 故障工况极端对
        n_bins: 熵计算分箱数
    
    Returns:
        similarities: 包含5个相似度分数的字典
    """
    # 计算所有样本的熵
    H_i = compute_shannon_entropy(x_i, n_bins)
    H_j = compute_shannon_entropy(x_j, n_bins)
    H_k = compute_shannon_entropy(x_k, n_bins)
    H_l = compute_shannon_entropy(x_l, n_bins)
    H_m = compute_shannon_entropy(x_m, n_bins)
    H_n = compute_shannon_entropy(x_n, n_bins)
    
    # 计算相似度 s = exp(-|H_a - H_b|)
    s_ij = np.exp(-np.abs(H_i - H_j))
    s_kl = np.exp(-np.abs(H_k - H_l))
    s_mn = np.exp(-np.abs(H_m - H_n))
    
    # 组间相似度 (使用平均熵)
    H_kl_mean = (H_k + H_l) / 2
    H_mn_mean = (H_m + H_n) / 2
    
    s_i_kl = np.exp(-np.abs(H_i - H_kl_mean))
    s_j_mn = np.exp(-np.abs(H_j - H_mn_mean))
    
    return {
        's_ij': s_ij,
        's_kl': s_kl,
        's_mn': s_mn,
        's_i_kl': s_i_kl,
        's_j_mn': s_j_mn
    }


class PDDMNetwork(nn.Module):
    """
    PDDM网络: 学习样本对之间的相似度
    
    输入: 两个样本的差值u和平均值v
    输出: 预测的相似度分数 [0, 1]
    
    网络结构:
    Input(u, v) → FC(hidden_dim) → ReLU → FC(hidden_dim) → ReLU → FC(1) → Sigmoid
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        """
        Args:
            input_dim: 输入特征维度 (对于TEP，为39维)
            hidden_dim: 隐藏层维度
        """
        super(PDDMNetwork, self).__init__()
        
        # u和v拼接后的维度是 2 * input_dim
        self.fc1 = nn.Linear(2 * input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
    
    def forward(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x_i: 第一个样本 (batch_size, input_dim) 或 (input_dim,)
            x_j: 第二个样本 (batch_size, input_dim) 或 (input_dim,)
        
        Returns:
            similarity: 预测的相似度 (batch_size, 1) 或 (1,)
        """
        # 计算差值和平均值
        u = torch.abs(x_i - x_j)  # 差值
        v = (x_i + x_j) / 2.0      # 平均值
        
        # 拼接u和v
        x = torch.cat([u, v], dim=-1)
        
        # 网络前向传播
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        
        return x


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
