import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict

# ============================================================================
# 1. 预处理工具: 计算全局因果权重 (The Teacher)
# ============================================================================
def precompute_global_weights(
    X_all: np.ndarray, 
    Y_all: np.ndarray
) -> torch.Tensor:
    """
    【核心逻辑】离线计算全局特征权重。
    基于每个变量与结果变量(XMEAS_40)的相关性(互信息近似)。
    
    Args:
        X_all: 整个训练集的特征 (N, 50)
        Y_all: 整个训练集的结果/标签 (N, 1) - 即 XMEAS_40
        
    Returns:
        weights: (50,) 归一化后的权重张量
    """
    print("正在计算全局互信息权重 (Causal Weights)...")
    n_features = X_all.shape[1]
    correlations = []
    Y_flat = Y_all.flatten()
    
    for k in range(n_features):
        # 使用皮尔逊相关系数的绝对值作为互信息的快速近似
        # 加上 1e-8 防止常数列导致的除零
        col_std = X_all[:, k].std()
        if col_std < 1e-8:
            corr = 0.0
        else:
            # corrcoef 返回矩阵 [[1, r], [r, 1]]
            corr = np.abs(np.corrcoef(X_all[:, k], Y_flat)[0, 1])
            if np.isnan(corr): 
                corr = 0.0
        correlations.append(corr)
        
    weights = np.array(correlations)
    
    # 归一化：让平均权重为 1.0，保持距离数值的量级稳定
    if weights.mean() > 0:
        weights = weights / weights.mean()
    else:
        weights = np.ones(n_features) # 如果全是噪声，退化为均匀权重
    
    # 打印 Top 5 重要变量，用于检查是否符合物理直觉
    top_indices = np.argsort(weights)[-5:][::-1]
    print(f"权重计算完成。Top 5 关键变量索引: {top_indices}")
    print(f"对应的权重值: {weights[top_indices]}")
    
    return torch.tensor(weights, dtype=torch.float32)


# ============================================================================
# 2. 表示层网络 (The Backbone/Encoder)
# ============================================================================
class BackboneNetwork(nn.Module):
    """
    你的原始表示层网络：将 50维 X 映射到 16维 Z
    结构: Linear -> ELU -> Dropout
    """
    def __init__(self, input_dim: int = 50, 
                 dim_backbone: str = '32,16', 
                 dropout: float = 0.1):
        super(BackboneNetwork, self).__init__()
        
        # 解析层维度结构，例如 '32,16' -> [32, 16]
        out_sizes = list(map(int, dim_backbone.split(',')))
        layer_sizes = [input_dim] + out_sizes
        
        self.net = nn.Sequential()
        
        # 动态构建层
        for i in range(1, len(layer_sizes)):
            # Linear
            self.net.add_module(
                f"backbone_dense{i}", 
                nn.Linear(layer_sizes[i-1], layer_sizes[i])
            )
            # Activation (ELU)
            self.net.add_module(
                f"backbone_relu{i}", 
                torch.nn.ELU()
            )
            # Dropout
            self.net.add_module(
                f"backbone_dropout{i}", 
                torch.nn.Dropout(p=dropout)
            )
            
        self.output_dim = layer_sizes[-1] # 通常是 16

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================================
# 3. PDDM 度量网络 (The Metric Learner)
# ============================================================================
class PDDMNetwork(nn.Module):
    """
    负责计算 Z 空间中两点的相似度。
    输入: z_i, z_j (来自 Backbone)
    输出: similarity score (scalar)
    """
    def __init__(self, latent_dim: int = 16, u_v_hidden_dim: int = 32, h_hidden_dim: int = 64):
        super(PDDMNetwork, self).__init__()
        
        # 这里的 latent_dim 应该等于 Backbone 的 output_dim (16)
        self.W_u = nn.Linear(latent_dim, u_v_hidden_dim)
        self.W_v = nn.Linear(latent_dim, u_v_hidden_dim)
        self.W_c = nn.Linear(2 * u_v_hidden_dim, h_hidden_dim)
        self.W_s = nn.Linear(h_hidden_dim, 1)
        self.relu = nn.ReLU()
        
    def _normalize_vector(self, x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
        norm = torch.norm(x, p=2, dim=dim, keepdim=True)
        return x / (norm + eps)

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        # 1. 构造交互特征 u (差分) 和 v (均值)
        u = torch.abs(z_i - z_j)
        v = torch.abs(z_i + z_j) / 2.0
        
        # 2. 独立的非线性映射
        u_norm = self._normalize_vector(u)
        u_1 = self.relu(self.W_u(u_norm))
        
        v_norm = self._normalize_vector(v)
        v_1 = self.relu(self.W_v(v_norm))
        
        # 3. 融合特征 h
        u_1_norm = self._normalize_vector(u_1)
        v_1_norm = self._normalize_vector(v_1)
        u_v_concat = torch.cat([u_1_norm, v_1_norm], dim=-1)
        
        h = self.relu(self.W_c(u_v_concat))
        
        # 4. 输出相似度
        S_hat = self.W_s(h)
        return S_hat


# ============================================================================
# 4. 核心逻辑: 计算加权目标相似度 (Weighted Ground Truth)
# ============================================================================
def compute_weighted_target_similarity(
    x_i: torch.Tensor, 
    x_j: torch.Tensor, 
    weights: torch.Tensor, 
    sigma: float = 1.0
) -> torch.Tensor:
    """
    基于全局因果权重，计算两个原始样本的加权物理相似度。
    
    Formula: S = exp( - sum(w * (xi - xj)^2) / 2sigma^2 )
    """
    # 计算加权平方距离
    # x_i, x_j: (batch_size, 50) or (50,)
    # weights: (50,)
    
    diff_sq = (x_i - x_j) ** 2
    weighted_dist_sq = torch.sum(weights * diff_sq, dim=-1)
    
    # 转换为相似度 (RBF Kernel)
    similarity = torch.exp(-weighted_dist_sq / (2 * sigma**2))
    
    return similarity


# ============================================================================
# 5. 训练步骤封装 (Training Step)
# ============================================================================
def pddm_train_step(
    batch_x: torch.Tensor,       # 当前 Batch 的输入 (B, 50)
    backbone: BackboneNetwork,   # 表示层
    pddm_net: PDDMNetwork,       # 度量层
    global_weights: torch.Tensor,# 全局互信息权重 (50,)
    sigma: float = 1.0,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    执行一次完整的 PDDM 训练计算。
    采样策略：随机选取 6 个样本构成 5 对组合 (保持你原本的拓扑结构)。
    """
    batch_size = batch_x.shape[0]
    
    # 1. 简单随机采样 6 个样本 (稳健策略)
    # 如果 batch 太小，允许重复
    indices = torch.randperm(batch_size)[:6]
    if len(indices) < 6:
        indices = torch.cat([indices, indices])[:6]
        
    # 选出的 6 个原始样本 X
    three_pairs_X = batch_x[indices] # (6, 50)
    
    # 2. 前向传播: X -> Z (Backbone)
    z_pairs = backbone(three_pairs_X) # (6, 16)
    
    # 3. 定义我们要比较的 5 对索引 (k-l, m-n, k-m, i-k, j-m)
    # 对应索引: i=0, j=1, k=2, l=3, m=4, n=5
    pairs_idx = [
        (2, 3), # (k, l)
        (4, 5), # (m, n)
        (2, 4), # (k, m)
        (0, 2), # (i, k)
        (1, 4)  # (j, m)
    ]
    
    loss_accum = 0.0
    
    # 4. 循环计算每一对的 Loss
    for (idx_1, idx_2) in pairs_idx:
        # A. 提取 Z 并预测相似度 (Student Prediction)
        # 输入: (1, 16) -> 输出 (1, 1)
        pred_sim = pddm_net(z_pairs[idx_1].unsqueeze(0), z_pairs[idx_2].unsqueeze(0)).squeeze()
        
        # B. 提取 X 并计算加权目标相似度 (Teacher Label)
        # 这里利用了 global_weights
        target_sim = compute_weighted_target_similarity(
            three_pairs_X[idx_1], 
            three_pairs_X[idx_2], 
            weights=global_weights, 
            sigma=sigma
        )
        
        # C. MSE Loss
        loss_accum += F.mse_loss(pred_sim, target_sim.detach()) # detach target just in case
        
    return loss_accum / 5.0


# ============================================================================
# 使用示例
# ============================================================================
if __name__ == "__main__":
    # 1. 假设数据
    N_SAMPLES = 1000
    N_FEATURES = 50
    X_dummy = np.random.randn(N_SAMPLES, N_FEATURES).astype(np.float32)
    Y_dummy = np.random.randn(N_SAMPLES, 1).astype(np.float32) # XMEAS_40

    # 2. 预计算权重 (只做一次)
    global_w = precompute_global_weights(X_dummy, Y_dummy)
    print("Global weights shape:", global_w.shape)

    # 3. 初始化模型
    device = 'cpu' # 或 'cuda'
    backbone = BackboneNetwork(input_dim=50, dim_backbone='32,16').to(device)
    pddm = PDDMNetwork(latent_dim=16).to(device)
    global_w = global_w.to(device)

    # 4. 模拟训练循环
    optimizer = torch.optim.Adam(list(backbone.parameters()) + list(pddm.parameters()), lr=0.001)
    
    # 模拟一个 Batch
    batch_tensor = torch.tensor(X_dummy[:64], dtype=torch.float32).to(device)
    
    optimizer.zero_grad()
    loss = pddm_train_step(batch_tensor, backbone, pddm, global_weights=global_w)
    loss.backward()
    optimizer.step()
    
    print(f"Batch Loss: {loss.item()}")