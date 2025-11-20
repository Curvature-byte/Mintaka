import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

# ============================================================================
# PART 1: 预处理工具 - 计算因果权重 (解决时间滞后问题)
# ============================================================================
def precompute_global_causal_weights(
    X_all: np.ndarray, 
    Y_all: np.ndarray, 
    max_lag: int = 20
) -> torch.Tensor:
    """
    【核心逻辑】离线计算全局特征权重。
    使用 '最大互相关 (Max Cross-Correlation)' 来解决时间滞后问题。
    
    Args:
        X_all: 整个训练集的特征 (N, 50) - 包含了过程变量和操控变量
        Y_all: 整个训练集的结果变量 (N, 1) - 即 XMEAS_40
        max_lag: 最大时间滞后步数 (例如 20 代表允许上游变量提前 20 个采样点影响结果)
        
    Returns:
        weights: (50,) 归一化后的权重张量
    """
    print(f"正在计算全局因果权重 (考虑最大滞后 {max_lag})...")
    n_features = X_all.shape[1]
    best_correlations = []
    Y_flat = Y_all.flatten()
    
    for k in range(n_features):
        col_data = X_all[:, k]
        max_corr = 0.0
        
        # 滑动时间窗口寻找最佳匹配 (解决时间延迟)
        # 我们只看 X 领先于 Y 的情况 (即 X 是因，Y 是果)
        for lag in range(max_lag + 1):
            if lag == 0:
                # 无滞后
                x_segment = col_data
                y_segment = Y_flat
            else:
                # X 取前 N-lag 个，Y 取后 N-lag 个 (相当于 X 平移了)
                x_segment = col_data[:-lag]
                y_segment = Y_flat[lag:]
            
            # 计算相关系数 (近似互信息)
            if x_segment.std() < 1e-8:
                corr = 0.0
            else:
                corr = np.abs(np.corrcoef(x_segment, y_segment)[0, 1])
                if np.isnan(corr): corr = 0.0
            
            if corr > max_corr:
                max_corr = corr
                
        best_correlations.append(max_corr)
        
    weights = np.array(best_correlations)
    
    # 归一化：让平均权重为 1.0，保持距离数值的量级稳定
    # 这样做的意义：不改变整体 Loss 的大小，只改变特征的相对重要性
    if weights.mean() > 1e-8:
        weights = weights / weights.mean()
    else:
        weights = np.ones(n_features) # 兜底：如果全是噪声，退化为均匀权重
    
    # 打印 Top 5 关键变量索引，用于物理验证
    top_indices = np.argsort(weights)[-5:][::-1]
    print(f"权重计算完成。")
    print(f"Top 5 关键变量索引: {top_indices}")
    print(f"对应的因果权重: {weights[top_indices]}")
    
    return torch.tensor(weights, dtype=torch.float32)


# ============================================================================
# PART 2: 模型架构 (Encoder + PDDM + Predictor)
# ============================================================================

class BackboneNetwork(nn.Module):
    """
    表示层 (Encoder): 将 50维 X 映射到 16维 Z
    """
    def __init__(self, input_dim: int = 50, dim_backbone: str = '32,16', dropout: float = 0.1):
        super(BackboneNetwork, self).__init__()
        
        out_sizes = list(map(int, dim_backbone.split(',')))
        layer_sizes = [input_dim] + out_sizes
        
        self.net = nn.Sequential()
        for i in range(1, len(layer_sizes)):
            self.net.add_module(f"dense{i}", nn.Linear(layer_sizes[i-1], layer_sizes[i]))
            self.net.add_module(f"elu{i}", nn.ELU()) # 使用 ELU 激活
            self.net.add_module(f"dropout{i}", nn.Dropout(p=dropout))
            
        self.output_dim = layer_sizes[-1] # 16

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PDDMMetricNetwork(nn.Module):
    """
    度量层: 计算 Z 空间中两点的非线性相似度
    """
    def __init__(self, latent_dim: int = 16, u_v_hidden_dim: int = 32, h_hidden_dim: int = 64):
        super(PDDMMetricNetwork, self).__init__()
        
        self.W_u = nn.Linear(latent_dim, u_v_hidden_dim)
        self.W_v = nn.Linear(latent_dim, u_v_hidden_dim)
        self.W_c = nn.Linear(2 * u_v_hidden_dim, h_hidden_dim)
        self.W_s = nn.Linear(h_hidden_dim, 1)
        self.relu = nn.ReLU()
        
    def _normalize(self, x, eps=1e-12):
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        return x / (norm + eps)

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        # 核心 PDDM 逻辑：利用差分(u)和均值(v)
        u = torch.abs(z_i - z_j)
        v = torch.abs(z_i + z_j) / 2.0
        
        u_1 = self.relu(self.W_u(self._normalize(u)))
        v_1 = self.relu(self.W_v(self._normalize(v)))
        
        concat = torch.cat([self._normalize(u_1), self._normalize(v_1)], dim=-1)
        h = self.relu(self.W_c(concat))
        
        return self.W_s(h) # 输出相似度分数


class PredictorHead(nn.Module):
    """
    预测层: 从 Z 预测 XMEAS_40 (用于内容约束 Loss_FL)
    """
    def __init__(self, latent_dim: int = 16):
        super(PredictorHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1) # 回归输出
        )
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class CausalPDDMModel(nn.Module):
    """
    整合模型: Backbone + PDDM + Predictor
    """
    def __init__(self, input_dim=50):
        super(CausalPDDMModel, self).__init__()
        self.backbone = BackboneNetwork(input_dim=input_dim)
        self.pddm_net = PDDMMetricNetwork(latent_dim=16)
        self.predictor = PredictorHead(latent_dim=16)
        
    def forward(self, x):
        z = self.backbone(x)
        y_pred = self.predictor(z)
        return z, y_pred


# ============================================================================
# PART 3: 损失计算 (Dual Loss with Causal Weights)
# ============================================================================

def compute_weighted_target_similarity(
    x_i: torch.Tensor, 
    x_j: torch.Tensor, 
    weights: torch.Tensor, 
    sigma: float = 1.0
) -> torch.Tensor:
    """
    Teacher Logic: 基于因果权重计算物理空间的"真实相似度"
    Formula: exp( - sum( w * (xi-xj)^2 ) / 2sigma^2 )
    """
    diff_sq = (x_i - x_j) ** 2
    # 关键：加权求和。只关注高权重(高因果性)的特征差异。
    weighted_dist_sq = torch.sum(weights * diff_sq, dim=-1)
    return torch.exp(-weighted_dist_sq / (2 * sigma**2))


def train_step_dual_loss(
    batch_x: torch.Tensor, 
    batch_y: torch.Tensor, # 真实的 XMEAS_40
    model: CausalPDDMModel, 
    global_weights: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    lambda_pddm: float = 1.0,
    lambda_fl: float = 1.0,
    sigma: float = 1.0
):
    """
    执行一步训练，包含两个损失函数
    """
    model.train()
    optimizer.zero_grad()
    
    # 1. Forward Pass (所有样本)
    z_all, y_pred_all = model(batch_x)
    
    # --- Loss 1: 预测损失 (L_FL) ---
    # 强迫 Z 保留能预测结果的信息 (保留稀疏样本的回归值)
    loss_fl = F.mse_loss(y_pred_all, batch_y)
    
    # --- Loss 2: 度量损失 (L_PDDM) ---
    # 随机采样 6 个索引用于构建 PDDM 对 (简单随机采样，无需复杂策略)
    batch_size = batch_x.size(0)
    indices = torch.randperm(batch_size)[:6]
    # 兜底：如果batch太小
    if len(indices) < 6: indices = torch.cat([indices, indices])[:6]
    
    z_pairs = z_all[indices]       # (6, 16)
    x_pairs = batch_x[indices]     # (6, 50)
    
    # 定义拓扑配对 (k-l, m-n, etc.)
    pairs_idx = [(2,3), (4,5), (2,4), (0,2), (1,4)]
    loss_pddm = 0.0
    
    for (idx_1, idx_2) in pairs_idx:
        # A. PDDM 网络预测 Z 空间的相似度
        # unsqueeze 用于增加 batch 维度 (1, 16)
        pred_sim = model.pddm_net(z_pairs[idx_1].unsqueeze(0), z_pairs[idx_2].unsqueeze(0)).squeeze()
        
        # B. 计算物理空间的加权目标相似度 (使用 global_weights)
        target_sim = compute_weighted_target_similarity(
            x_pairs[idx_1], x_pairs[idx_2], weights=global_weights, sigma=sigma
        )
        
        loss_pddm += F.mse_loss(pred_sim, target_sim.detach())
    
    loss_pddm = loss_pddm / 5.0
    
    # --- 总损失 ---
    total_loss = (lambda_pddm * loss_pddm) + (lambda_fl * loss_fl)
    
    total_loss.backward()
    optimizer.step()
    
    return total_loss.item(), loss_pddm.item(), loss_fl.item()


# ============================================================================
# PART 4: 主程序入口 (Main Execution)
# ============================================================================
if __name__ == "__main__":
    # 配置
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    INPUT_DIM = 50
    N_SAMPLES = 2000 # 模拟数据量
    BATCH_SIZE = 64
    
    print(f"Running on {DEVICE}")

    # ---------------------------------------------------------
    # 1. 数据准备 (模拟 TEP 数据结构)
    # 实际使用时，请加载你的 .npy 或 DataFrame
    # ---------------------------------------------------------
    # X: 假设有滞后关系，比如 X_0 是 XMEAS_40 的强因果变量
    X_dummy = np.random.randn(N_SAMPLES, INPUT_DIM).astype(np.float32)
    # 模拟一个结果 Y，它高度依赖 X 的第 0 和 第 5 列 (制造因果关系)
    Y_dummy = 3.0 * X_dummy[:, 0:1] - 2.0 * X_dummy[:, 5:6] + 0.1 * np.random.randn(N_SAMPLES, 1)
    Y_dummy = Y_dummy.astype(np.float32)
    
    # ---------------------------------------------------------
    # 2. 离线计算因果权重 (只需运行一次)
    # ---------------------------------------------------------
    # 这里会自动处理滞后，并找出第0和第5列是关键变量
    global_weights = precompute_global_causal_weights(X_dummy, Y_dummy, max_lag=5)
    global_weights = global_weights.to(DEVICE)

    # ---------------------------------------------------------
    # 3. 模型初始化
    # ---------------------------------------------------------
    model = CausalPDDMModel(input_dim=INPUT_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # ---------------------------------------------------------
    # 4. 训练循环
    # ---------------------------------------------------------
    print("\n开始训练...")
    model.train()
    
    # 简单的数据加载器模拟
    num_batches = N_SAMPLES // BATCH_SIZE
    
    for epoch in range(5): # 跑5个Epoch看看
        epoch_loss = 0
        epoch_pddm = 0
        epoch_fl = 0
        
        for i in range(num_batches):
            start_idx = i * BATCH_SIZE
            end_idx = start_idx + BATCH_SIZE
            
            batch_x = torch.tensor(X_dummy[start_idx:end_idx]).to(DEVICE)
            batch_y = torch.tensor(Y_dummy[start_idx:end_idx]).to(DEVICE)
            
            loss, l_p, l_f = train_step_dual_loss(
                batch_x, batch_y, 
                model, global_weights, optimizer,
                lambda_pddm=1.0, lambda_fl=1.0
            )
            
            epoch_loss += loss
            epoch_pddm += l_p
            epoch_fl += l_f
            
        print(f"Epoch {epoch+1}: Total={epoch_loss/num_batches:.4f} | PDDM={epoch_pddm/num_batches:.4f} | FL={epoch_fl/num_batches:.4f}")

    print("\n训练完成。模型已学会基于因果权重的表示。")