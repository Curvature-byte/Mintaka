"""
TEP Dataset Training Example with PDDM and Information Entropy
===============================================================
This script demonstrates how to integrate PDDM loss into your existing neural network
for Tennessee Eastman Process fault diagnosis.

Training Architecture:
Input (39 vars) → Representation Network (φ) → Prediction Network → Fault Classification
                         ↓
                    PDDM Loss (entropy-based)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
from tep_pddm_module import (
    PDDMNetwork,
    TEPDataPreprocessor,
    find_three_pairs_entropy,
    compute_three_pairs_similarity_entropy,
    compute_pddm_loss,
    compute_mid_point_distance,
    compute_representation_similarity_loss
)


class TEPDataset(Dataset):
    """
    TEP数据集类
    
    支持的数据格式:
    - faultNumber: 故障类型 (0=正常, 1-20=故障)
    - simulationRun: 模拟运行编号 (1-500)
    - sample: 样本编号
    - xmeas_1 to xmeas_41: 测量变量
    - xmv_1 to xmv_11: 操作变量
    """
    
    def __init__(self, df: pd.DataFrame, preprocessor: Optional[TEPDataPreprocessor] = None):
        """
        Args:
            df: TEP DataFrame
            preprocessor: 预处理器，如果为None则创建新的
        """
        if preprocessor is None:
            self.preprocessor = TEPDataPreprocessor()
            self.X, self.fault_labels = self.preprocessor.fit_transform(df)
        else:
            self.preprocessor = preprocessor
            self.X, self.fault_labels = self.preprocessor.transform(df)
        
        self.n_samples = len(self.X)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return {
            'x': torch.FloatTensor(self.X[idx]),
            'fault_label': torch.LongTensor([self.fault_labels[idx]])[0]
        }


class RepresentationNetwork(nn.Module):
    """
    表示网络 φ: X → Φ
    
    将39维输入映射到低维表示空间
    """
    
    def __init__(self, input_dim: int = 39, repr_dim: int = 128):
        """
        Args:
            input_dim: 输入维度 (TEP为39)
            repr_dim: 表示维度
        """
        super(RepresentationNetwork, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, repr_dim)
        )
        
        self.input_dim = input_dim
        self.repr_dim = repr_dim
    
    def forward(self, x):
        """
        Args:
            x: 输入特征 (batch_size, input_dim)
        
        Returns:
            phi: 表示向量 (batch_size, repr_dim)
        """
        return self.net(x)


class PredictionNetwork(nn.Module):
    """
    预测网络: Φ → Y
    
    从表示向量预测故障类型
    """
    
    def __init__(self, repr_dim: int = 128, n_classes: int = 21):
        """
        Args:
            repr_dim: 表示维度
            n_classes: 类别数 (0-20，共21类)
        """
        super(PredictionNetwork, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(repr_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )
        
        self.repr_dim = repr_dim
        self.n_classes = n_classes
    
    def forward(self, phi):
        """
        Args:
            phi: 表示向量 (batch_size, repr_dim)
        
        Returns:
            logits: 类别logits (batch_size, n_classes)
        """
        return self.net(phi)


class TEPNetWithPDDM(nn.Module):
    """
    完整的TEP故障诊断网络 + PDDM
    
    结构:
    Input → RepresentationNet → PredictionNet → Output
              ↓
           PDDM Net (学习相似度)
    """
    
    def __init__(
        self,
        input_dim: int = 39,
        repr_dim: int = 128,
        n_classes: int = 21,
        pddm_hidden_dim: int = 64
    ):
        """
        Args:
            input_dim: 输入维度
            repr_dim: 表示空间维度
            n_classes: 故障类别数
            pddm_hidden_dim: PDDM网络隐藏层维度
        """
        super(TEPNetWithPDDM, self).__init__()
        
        self.repr_net = RepresentationNetwork(input_dim, repr_dim)
        self.pred_net = PredictionNetwork(repr_dim, n_classes)
        
        # PDDM网络在表示空间中工作
        self.pddm_net = PDDMNetwork(repr_dim, pddm_hidden_dim)
        
        self.input_dim = input_dim
        self.repr_dim = repr_dim
        self.n_classes = n_classes
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征 (batch_size, input_dim)
        
        Returns:
            phi: 表示向量 (batch_size, repr_dim)
            logits: 预测logits (batch_size, n_classes)
        """
        phi = self.repr_net(x)
        logits = self.pred_net(phi)
        return phi, logits
    
    def get_representation(self, x):
        """获取表示向量"""
        return self.repr_net(x)
    
    def predict(self, x):
        """预测故障类型"""
        _, logits = self.forward(x)
        return torch.argmax(logits, dim=1)


class TEPTrainer:
    """
    TEP训练器 - 集成PDDM损失
    """
    
    def __init__(
        self,
        model: TEPNetWithPDDM,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        alpha: float = 1.0,  # 预测损失权重
        beta: float = 0.5,   # PDDM损失权重
        gamma: float = 0.1,  # 中间距离损失权重
        n_bins: int = 30     # 熵计算分箱数
    ):
        """
        Args:
            model: TEP网络模型
            device: 计算设备
            alpha: 预测损失权重
            beta: PDDM损失权重
            gamma: 中间距离损失权重
            n_bins: 信息熵分箱数
        """
        self.model = model.to(device)
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.n_bins = n_bins
        
        # 损失函数
        self.criterion_pred = nn.CrossEntropyLoss()
        
        # 优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
    
    def compute_total_loss(
        self,
        X_batch: torch.Tensor,
        fault_labels_batch: torch.Tensor,
        X_full: np.ndarray,
        fault_labels_full: np.ndarray
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算总损失
        
        Args:
            X_batch: 批次输入 (batch_size, 39)
            fault_labels_batch: 批次标签 (batch_size,)
            X_full: 完整数据集 (用于选择三对样本)
            fault_labels_full: 完整标签
        
        Returns:
            total_loss: 总损失
            loss_dict: 各项损失的字典
        """
        # 前向传播
        phi_batch, logits_batch = self.model(X_batch)
        
        # 1. 预测损失
        pred_loss = self.criterion_pred(logits_batch, fault_labels_batch)
        
        # 2. PDDM损失 (每个epoch随机选择一组三对样本)
        try:
            # 从完整数据集中选择三对样本
            x_i, x_j, x_k, x_l, x_m, x_n = find_three_pairs_entropy(
                X_full, fault_labels_full, self.n_bins, fault_free_label=0
            )
            
            # 计算目标相似度 (基于输入空间的熵)
            target_similarities = compute_three_pairs_similarity_entropy(
                x_i, x_j, x_k, x_l, x_m, x_n, self.n_bins
            )
            
            # 转换为tensor并获取表示
            x_i_t = torch.FloatTensor(x_i).unsqueeze(0).to(self.device)
            x_j_t = torch.FloatTensor(x_j).unsqueeze(0).to(self.device)
            x_k_t = torch.FloatTensor(x_k).unsqueeze(0).to(self.device)
            x_l_t = torch.FloatTensor(x_l).unsqueeze(0).to(self.device)
            x_m_t = torch.FloatTensor(x_m).unsqueeze(0).to(self.device)
            x_n_t = torch.FloatTensor(x_n).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                phi_i = self.model.get_representation(x_i_t).squeeze(0)
                phi_j = self.model.get_representation(x_j_t).squeeze(0)
                phi_k = self.model.get_representation(x_k_t).squeeze(0)
                phi_l = self.model.get_representation(x_l_t).squeeze(0)
                phi_m = self.model.get_representation(x_m_t).squeeze(0)
                phi_n = self.model.get_representation(x_n_t).squeeze(0)
            
            # 在表示空间计算PDDM损失
            pddm_loss = compute_representation_similarity_loss(
                self.model.pddm_net,
                phi_i, phi_j, phi_k, phi_l, phi_m, phi_n,
                target_similarities
            )
            
            # 3. 中间对距离损失
            mid_distance = compute_mid_point_distance(phi_i, phi_j, distance_type='euclidean')
            
        except Exception as e:
            # 如果选择样本失败 (比如数据不足)，跳过PDDM损失
            print(f"Warning: 无法计算PDDM损失: {e}")
            pddm_loss = torch.tensor(0.0, device=self.device)
            mid_distance = torch.tensor(0.0, device=self.device)
        
        # 总损失
        total_loss = (
            self.alpha * pred_loss +
            self.beta * pddm_loss +
            self.gamma * mid_distance
        )
        
        loss_dict = {
            'total': total_loss.item(),
            'pred': pred_loss.item(),
            'pddm': pddm_loss.item(),
            'mid_dist': mid_distance.item()
        }
        
        return total_loss, loss_dict
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        X_full: np.ndarray,
        fault_labels_full: np.ndarray
    ) -> Dict[str, float]:
        """
        训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
            X_full: 完整训练数据 (用于PDDM)
            fault_labels_full: 完整训练标签
        
        Returns:
            avg_losses: 平均损失字典
        """
        self.model.train()
        
        total_losses = {'total': 0.0, 'pred': 0.0, 'pddm': 0.0, 'mid_dist': 0.0}
        n_batches = 0
        
        for batch in train_loader:
            X_batch = batch['x'].to(self.device)
            fault_labels_batch = batch['fault_label'].to(self.device)
            
            # 计算损失
            loss, loss_dict = self.compute_total_loss(
                X_batch, fault_labels_batch,
                X_full, fault_labels_full
            )
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 累积损失
            for key in total_losses:
                total_losses[key] += loss_dict[key]
            n_batches += 1
        
        # 计算平均损失
        avg_losses = {key: val / n_batches for key, val in total_losses.items()}
        
        return avg_losses
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        评估模型
        
        Args:
            test_loader: 测试数据加载器
        
        Returns:
            metrics: 评估指标字典
        """
        self.model.eval()
        
        correct = 0
        total = 0
        test_loss = 0.0
        
        with torch.no_grad():
            for batch in test_loader:
                X_batch = batch['x'].to(self.device)
                fault_labels_batch = batch['fault_label'].to(self.device)
                
                # 前向传播
                _, logits = self.model(X_batch)
                
                # 计算损失
                loss = self.criterion_pred(logits, fault_labels_batch)
                test_loss += loss.item()
                
                # 计算准确率
                preds = torch.argmax(logits, dim=1)
                correct += (preds == fault_labels_batch).sum().item()
                total += fault_labels_batch.size(0)
        
        accuracy = correct / total
        avg_loss = test_loss / len(test_loader)
        
        return {
            'accuracy': accuracy,
            'loss': avg_loss
        }
    
    def fit(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        X_train_full: np.ndarray,
        fault_labels_train_full: np.ndarray,
        n_epochs: int = 50
    ):
        """
        完整训练流程
        
        Args:
            train_loader: 训练数据加载器
            test_loader: 测试数据加载器
            X_train_full: 完整训练数据
            fault_labels_train_full: 完整训练标签
            n_epochs: 训练轮数
        """
        print("=" * 80)
        print("开始训练 TEP 故障诊断网络 (带PDDM)")
        print("=" * 80)
        print(f"模型配置:")
        print(f"  - 输入维度: {self.model.input_dim}")
        print(f"  - 表示维度: {self.model.repr_dim}")
        print(f"  - 类别数: {self.model.n_classes}")
        print(f"损失权重:")
        print(f"  - α (预测): {self.alpha}")
        print(f"  - β (PDDM): {self.beta}")
        print(f"  - γ (中间距离): {self.gamma}")
        print(f"训练参数:")
        print(f"  - Epochs: {n_epochs}")
        print(f"  - 设备: {self.device}")
        print(f"  - 优化器: {type(self.optimizer).__name__}")
        print("=" * 80)
        
        for epoch in range(n_epochs):
            # 训练
            train_losses = self.train_epoch(
                train_loader, X_train_full, fault_labels_train_full
            )
            
            # 评估
            test_metrics = self.evaluate(test_loader)
            
            # 打印进度
            print(f"Epoch [{epoch+1}/{n_epochs}]")
            print(f"  Train - Total: {train_losses['total']:.4f}, "
                  f"Pred: {train_losses['pred']:.4f}, "
                  f"PDDM: {train_losses['pddm']:.4f}, "
                  f"Mid: {train_losses['mid_dist']:.4f}")
            print(f"  Test  - Loss: {test_metrics['loss']:.4f}, "
                  f"Acc: {test_metrics['accuracy']:.4f}")
        
        print("=" * 80)
        print("训练完成！")
        print("=" * 80)


# ==================== 主函数 ====================

def main():
    """
    主训练流程
    
    使用说明:
    1. 加载你的TEP数据 (CSV或其他格式)
    2. 使用TEPDataset创建数据集
    3. 使用TEPTrainer训练模型
    """
    
    print("TEP PDDM 训练示例")
    print("=" * 80)
    print("请按以下步骤操作:")
    print()
    print("1. 准备数据:")
    print("   import pandas as pd")
    print("   train_df = pd.read_csv('your_tep_train_data.csv')")
    print("   test_df = pd.read_csv('your_tep_test_data.csv')")
    print()
    print("2. 创建数据集:")
    print("   preprocessor = TEPDataPreprocessor()")
    print("   train_dataset = TEPDataset(train_df, preprocessor)")
    print("   test_dataset = TEPDataset(test_df, preprocessor)")
    print()
    print("3. 创建DataLoader:")
    print("   train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)")
    print("   test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)")
    print()
    print("4. 创建模型和训练器:")
    print("   model = TEPNetWithPDDM(input_dim=39, repr_dim=128, n_classes=21)")
    print("   trainer = TEPTrainer(model, alpha=1.0, beta=0.5, gamma=0.1)")
    print()
    print("5. 训练:")
    print("   X_train, labels_train = preprocessor.transform(train_df)")
    print("   trainer.fit(train_loader, test_loader, X_train, labels_train, n_epochs=50)")
    print()
    print("=" * 80)
    print()
    print("关键参数说明:")
    print("  - input_dim=39: TEP输入维度 (22过程+17成分)")
    print("  - repr_dim=128: 表示空间维度 (可调)")
    print("  - n_classes=21: 故障类型数 (0-20)")
    print("  - alpha=1.0: 预测损失权重")
    print("  - beta=0.5: PDDM损失权重 (调大增强相似度约束)")
    print("  - gamma=0.1: 中间距离损失权重 (调大增强全局平衡)")
    print("  - n_bins=30: 熵计算分箱数 (根据数据分布调整)")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
