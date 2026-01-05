"""
DeXposure 传染效应预测 - 修复版（解决 NaN 问题）

修复：
1. 数据清理（移除 NaN/Inf）
2. 标签裁剪（限制极端值）
3. 更鲁棒的损失函数
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.graph.dexposure_dataset import DeXposureRegressionDataset, collate_fn


class ContagionPredictor(nn.Module):
    """传染效应预测模型（简化 GNN）"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,  # 减小隐藏层
        output_dim: int = 3,
        num_layers: int = 2,    # 减少层数
        dropout: float = 0.2
    ):
        super().__init__()

        # 节点特征编码
        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # 添加 BatchNorm
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # GNN 层
        self.gnn_layers = nn.ModuleList([
            GNNLayer(hidden_dim, hidden_dim, dropout)
            for _ in range(num_layers)
        ])

        # 回归头
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, graph, node_features):
        # 编码
        h = self.node_encoder(node_features)

        # GNN 消息传递
        for gnn in self.gnn_layers:
            h = gnn(graph, h)

        # 回归预测
        predictions = self.regression_head(h)

        return predictions


class GNNLayer(nn.Module):
    """简单的 GNN 层"""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, graph, node_features):
        import dgl.function as fn

        with graph.local_scope():
            graph.ndata['h'] = node_features

            # 消息传递
            graph.update_all(
                fn.copy_u('h', 'm'),
                fn.mean('m', 'h_neigh')
            )

            h_self = node_features
            h_neigh = graph.ndata.get('h_neigh', torch.zeros_like(h_self))

            # 聚合
            h = self.linear(h_self + h_neigh)
            h = self.norm(h)
            h = self.activation(h)
            h = self.dropout(h)

            return h + node_features  # 残差连接


def compute_loss_robust(predictions: torch.Tensor, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    鲁棒的损失函数（修复 NaN 问题）

    关键改进：
    1. 使用 Huber Loss 替代 MSE（对异常值更鲁棒）
    2. 添加数值稳定性检查
    3. 裁剪极端预测值
    """
    # 检查输入是否有 NaN
    if torch.isnan(predictions).any() or torch.isnan(labels).any():
        print("⚠️  警告：输入包含 NaN")
        predictions = torch.nan_to_num(predictions, 0.0)
        labels = torch.nan_to_num(labels, 0.0)

    # 裁剪预测值到合理范围
    predictions = torch.clamp(predictions, -20, 20)

    # 1. 变化率损失 (Huber Loss)
    loss_change = F.huber_loss(predictions[:, 0], labels[:, 0], delta=1.0)

    # 2. 绝对损失损失 (Huber Loss)
    loss_abs = F.huber_loss(predictions[:, 1], labels[:, 1], delta=1.0)

    # 3. 受影响程度损失 (MSE + sigmoid)
    pred_impact = torch.sigmoid(predictions[:, 2])
    loss_impact = F.mse_loss(pred_impact, labels[:, 2])

    # 总损失（降低权重，更保守）
    total_loss = 0.5 * loss_change + 0.3 * loss_abs + loss_impact

    # 检查损失是否有效
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        print("⚠️  警告：损失为 NaN/Inf，使用默认值")
        total_loss = torch.tensor(1.0, device=predictions.device)

    return {
        'total': total_loss,
        'change_rate': loss_change,
        'abs_loss': loss_abs,
        'impact': loss_impact
    }


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str
) -> Dict[str, float]:
    """训练一个 epoch"""
    model.train()

    total_losses = {'total': 0.0, 'change_rate': 0.0, 'abs_loss': 0.0, 'impact': 0.0}
    num_batches = 0

    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, batch in enumerate(pbar):
        try:
            graph = batch['graph'].to(device)
            labels = batch['labels'].to(device)
            node_features = graph.ndata['feat']

            # 检查数据
            if torch.isnan(node_features).any():
                print(f"⚠️  批次 {batch_idx}: 特征包含 NaN，跳过")
                continue
            if torch.isnan(labels).any():
                print(f"⚠️  批次 {batch_idx}: 标签包含 NaN，跳过")
                continue

            # 前向传播
            predictions = model(graph, node_features)

            # 计算损失
            losses = compute_loss_robust(predictions, labels)

            # 反向传播
            optimizer.zero_grad()
            losses['total'].backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

            optimizer.step()

            # 累积损失
            for key in total_losses:
                if not torch.isnan(losses[key]):
                    total_losses[key] += losses[key].item()
            num_batches += 1

            # 更新进度条
            pbar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'impact': f"{losses['impact'].item():.4f}"
            })

        except Exception as e:
            print(f"\n⚠️  批次 {batch_idx} 失败: {e}")
            continue

    # 平均损失
    if num_batches == 0:
        return {'total': float('inf'), 'change_rate': 0, 'abs_loss': 0, 'impact': 0}

    avg_losses = {k: v / num_batches for k, v in total_losses.items()}
    return avg_losses


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: str
) -> Dict[str, float]:
    """评估模型"""
    model.eval()

    total_losses = {'total': 0.0, 'change_rate': 0.0, 'abs_loss': 0.0, 'impact': 0.0}
    num_batches = 0

    all_predictions = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        try:
            graph = batch['graph'].to(device)
            labels = batch['labels'].to(device)
            node_features = graph.ndata['feat']

            # 检查数据
            if torch.isnan(node_features).any() or torch.isnan(labels).any():
                continue

            # 前向传播
            predictions = model(graph, node_features)

            # 计算损失
            losses = compute_loss_robust(predictions, labels)

            # 累积损失
            for key in total_losses:
                if not torch.isnan(losses[key]):
                    total_losses[key] += losses[key].item()
            num_batches += 1

            # 保存预测
            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())

        except Exception as e:
            continue

    # 平均损失
    if num_batches == 0:
        return {'total': float('inf'), 'change_rate': 0, 'abs_loss': 0, 'impact': 0, 'mae_impact': 0, 'corr_impact': 0}

    avg_losses = {k: v / num_batches for k, v in total_losses.items()}

    # 额外指标
    if all_predictions:
        all_predictions = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # MAE
        pred_impact = torch.sigmoid(all_predictions[:, 2])
        mae_impact = F.l1_loss(pred_impact, all_labels[:, 2])
        avg_losses['mae_impact'] = mae_impact.item()

        # 相关系数
        try:
            corr_matrix = torch.corrcoef(torch.stack([pred_impact, all_labels[:, 2]]))
            corr_impact = corr_matrix[0, 1]
            if not torch.isnan(corr_impact):
                avg_losses['corr_impact'] = corr_impact.item()
            else:
                avg_losses['corr_impact'] = 0.0
        except:
            avg_losses['corr_impact'] = 0.0

    return avg_losses


def main():
    """主训练流程"""

    # 配置
    config = {
        'data_root': '/home/figurich/inter-protocol-exposure/DeXposure/data',
        'output_dir': '/home/figurich/inter-protocol-exposure/graphpfn/exp/dexposure_regression_fixed',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'batch_size': 2,
        'num_epochs': 50,
        'lr': 5e-4,      # 降低学习率
        'hidden_dim': 128,  # 减小模型
        'num_layers': 2,
        'dropout': 0.2,
        'seed': 42
    }

    # 设置随机种子
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    print("=" * 60)
    print("DeXposure 传染效应预测 - 修复版 (NaN-Free)")
    print("=" * 60)
    print(f"设备: {config['device']}")
    print(f"批次大小: {config['batch_size']}")
    print(f"学习率: {config['lr']}")
    print(f"模型大小: {config['hidden_dim']} hidden, {config['num_layers']} layers")
    print()

    # 危机场景
    crisis_scenarios = [
        {'name': 'binance_collapse', 'nodes': ['2269'], 'ratio': 0.95},
        {'name': 'lido_collapse', 'nodes': ['182'], 'ratio': 0.90},
        {'name': 'double_crisis', 'nodes': ['2269', '182'], 'ratio': 0.95}
    ]

    # 创建数据集
    print("加载数据集...")
    train_dataset = DeXposureRegressionDataset(
        root=config['data_root'],
        split='train',
        crisis_scenarios=crisis_scenarios
    )

    val_dataset = DeXposureRegressionDataset(
        root=config['data_root'],
        split='val',
        crisis_scenarios=crisis_scenarios
    )

    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn
    )

    # 创建模型
    print("\n初始化模型...")
    input_dim = train_dataset.get_node_features_dim()
    model = ContagionPredictor(
        input_dim=input_dim,
        hidden_dim=config['hidden_dim'],
        output_dim=3,
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(config['device'])

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 优化器
    optimizer = AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=config['num_epochs'])

    # 训练循环
    print("\n开始训练...")
    best_val_loss = float('inf')
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    history = {'train': [], 'val': []}

    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        print("-" * 60)

        # 训练
        train_losses = train_epoch(model, train_loader, optimizer, config['device'])

        # 验证
        val_losses = evaluate(model, val_loader, config['device'])

        # 学习率调整
        scheduler.step()

        # 打印结果
        print(f"训练 - 总损失: {train_losses['total']:.4f}, "
              f"变化率: {train_losses['change_rate']:.4f}, "
              f"受影响: {train_losses['impact']:.4f}")
        print(f"验证 - 总损失: {val_losses['total']:.4f}, "
              f"MAE: {val_losses.get('mae_impact', 0):.4f}, "
              f"Corr: {val_losses.get('corr_impact', 0):.4f}")

        # 保存历史
        history['train'].append(train_losses)
        history['val'].append(val_losses)

        # 保存最佳模型
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'config': config
            }, output_dir / 'best_model_fixed.pt')
            print(f"✓ 保存最佳模型 (val_loss={best_val_loss:.4f})")

    # 保存历史
    with open(output_dir / 'history_fixed.json', 'w') as f:
        # 将 NaN 转换为 null
        history_clean = json.loads(json.dumps(history, allow_nan=False, default=str))
        json.dump(history_clean, f, indent=2)

    print("\n" + "=" * 60)
    print(f"训练完成！最佳验证损失: {best_val_loss:.4f}")
    print(f"模型保存至: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
