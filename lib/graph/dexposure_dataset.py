"""
DeXposure 数据集 - GraphPFN 适配器
支持节点级回归任务：预测传染效应
"""

from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np
import torch
import dgl
from sklearn.model_selection import train_test_split

from lib.graph.dexposure_temporal import DeXposureTemporalLoader


class DeXposureRegressionDataset:
    """
    DeXposure 回归数据集

    任务：给定 t 时刻的图（可能包含破产节点），预测 t+1 时刻的节点变化
    """

    def __init__(
        self,
        root: str,
        split: str = "train",  # 'train', 'val', 'test'
        crisis_scenarios: Optional[List[Dict]] = None,
        prediction_window: int = 1,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42
    ):
        """
        Args:
            root: DeXposure 数据目录
            split: 数据集划分 ('train', 'val', 'test')
            crisis_scenarios: 危机场景列表，每个场景包含 {'nodes': [...], 'ratio': 0.9}
            prediction_window: 预测窗口（周）
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            seed: 随机种子
        """
        self.root = Path(root)
        self.split = split
        self.prediction_window = prediction_window

        # 加载时序数据
        self.loader = DeXposureTemporalLoader(
            data_path=str(self.root / "historical-network_week_2025-07-01.json"),
            meta_path=str(self.root / "meta_df.csv"),
            prediction_window=prediction_window
        )

        # 生成数据样本
        self.samples = self._generate_samples(crisis_scenarios)

        # 划分数据集
        self._split_dataset(val_ratio, test_ratio, seed)

        print(f"\n数据集统计:")
        print(f"  - 总样本数: {len(self.samples)}")
        print(f"  - {split} 集样本数: {len(self.indices)}")

    def _generate_samples(self, crisis_scenarios: Optional[List[Dict]] = None) -> List[Dict]:
        """
        生成训练样本

        策略：
        1. 正常时序对（无破产）
        2. 模拟破产场景（如果提供）
        """
        samples = []
        num_snapshots = len(self.loader.dates) - self.prediction_window

        print(f"\n生成训练样本...")

        # 1. 正常时序对
        print(f"  生成正常时序对...")
        for t in range(num_snapshots):
            try:
                graph_t, graph_t1, labels, _ = self.loader.get_temporal_pair(t)
                samples.append({
                    'graph_t': graph_t,
                    'graph_t1': graph_t1,
                    'labels': labels,
                    'scenario': 'normal',
                    'time_idx': t
                })
            except Exception as e:
                print(f"    ⚠️  跳过时刻 {t}: {e}")

        print(f"    ✓ 生成了 {len(samples)} 个正常样本")

        # 2. 危机场景
        if crisis_scenarios:
            print(f"  生成危机场景样本...")
            for scenario in crisis_scenarios:
                fail_nodes = scenario.get('nodes', [])
                fail_ratio = scenario.get('ratio', 0.9)
                scenario_name = scenario.get('name', f"crisis_{fail_nodes}")

                for t in range(num_snapshots):
                    try:
                        graph_t, graph_t1, labels, _ = self.loader.get_temporal_pair(
                            t,
                            inject_failure=fail_nodes,
                            failure_ratio=fail_ratio
                        )
                        samples.append({
                            'graph_t': graph_t,
                            'graph_t1': graph_t1,
                            'labels': labels,
                            'scenario': scenario_name,
                            'time_idx': t
                        })
                    except Exception as e:
                        print(f"    ⚠️  跳过场景 {scenario_name} @ {t}: {e}")

            print(f"    ✓ 生成了 {len(samples) - num_snapshots} 个危机样本")

        return samples

    def _split_dataset(self, val_ratio: float, test_ratio: float, seed: int):
        """划分数据集"""
        n_samples = len(self.samples)
        indices = np.arange(n_samples)

        # 先划分训练+验证 vs 测试
        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=test_ratio,
            random_state=seed
        )

        # 再划分训练 vs 验证
        val_size = val_ratio / (1 - test_ratio)
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_size,
            random_state=seed
        )

        if self.split == 'train':
            self.indices = train_idx
        elif self.split == 'val':
            self.indices = val_idx
        elif self.split == 'test':
            self.indices = test_idx
        else:
            raise ValueError(f"Unknown split: {self.split}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        返回一个样本

        Returns:
            {
                'graph': DGLGraph,
                'labels': torch.Tensor (num_nodes, 3),
                'scenario': str,
                'time_idx': int
            }
        """
        sample_idx = self.indices[idx]
        sample = self.samples[sample_idx]

        return {
            'graph': sample['graph_t'],
            'labels': sample['labels'],
            'scenario': sample['scenario'],
            'time_idx': sample['time_idx']
        }

    def get_node_features_dim(self) -> int:
        """获取节点特征维度"""
        sample = self.samples[0]
        return sample['graph_t'].ndata['feat'].shape[1]

    def get_label_dim(self) -> int:
        """获取标签维度"""
        return 3  # (变化率, 绝对损失, 受影响程度)


def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """
    批次整理函数

    由于不同样本的图大小可能不同，需要用 DGL 的 batch 功能
    """
    graphs = [item['graph'] for item in batch]
    labels = [item['labels'] for item in batch]

    # 批量图
    batched_graph = dgl.batch(graphs)

    # 批量标签（拼接）
    batched_labels = torch.cat(labels, dim=0)

    return {
        'graph': batched_graph,
        'labels': batched_labels,
        'scenarios': [item['scenario'] for item in batch],
        'time_indices': [item['time_idx'] for item in batch]
    }


# ============ 使用示例 ============

if __name__ == "__main__":
    # 定义危机场景
    crisis_scenarios = [
        {
            'name': 'binance_collapse',
            'nodes': ['2269'],  # Binance
            'ratio': 0.95
        },
        {
            'name': 'lido_collapse',
            'nodes': ['182'],   # Lido
            'ratio': 0.90
        },
        {
            'name': 'double_crisis',
            'nodes': ['2269', '182'],  # Binance + Lido
            'ratio': 0.95
        }
    ]

    # 创建数据集
    train_dataset = DeXposureRegressionDataset(
        root="/home/figurich/inter-protocol-exposure/DeXposure/data",
        split="train",
        crisis_scenarios=crisis_scenarios,
        prediction_window=1
    )

    val_dataset = DeXposureRegressionDataset(
        root="/home/figurich/inter-protocol-exposure/DeXposure/data",
        split="val",
        crisis_scenarios=crisis_scenarios,
        prediction_window=1
    )

    print(f"\n特征维度: {train_dataset.get_node_features_dim()}")
    print(f"标签维度: {train_dataset.get_label_dim()}")

    # 测试一个样本
    sample = train_dataset[0]
    print(f"\n样本示例:")
    print(f"  图: {sample['graph'].num_nodes()} 节点, {sample['graph'].num_edges()} 边")
    print(f"  标签形状: {sample['labels'].shape}")
    print(f"  场景: {sample['scenario']}")

    # 测试批次整理
    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        train_dataset,
        batch_size=2,
        collate_fn=collate_fn,
        shuffle=True
    )

    batch = next(iter(dataloader))
    print(f"\n批次示例:")
    print(f"  批量图: {batch['graph'].num_nodes()} 节点 (合并)")
    print(f"  批量标签: {batch['labels'].shape}")
    print(f"  场景: {batch['scenarios']}")
