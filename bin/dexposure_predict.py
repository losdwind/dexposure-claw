"""
DeXposure 传染效应预测 - 预测和可视化工具

用途：
1. 加载训练好的模型
2. 模拟任意协议破产
3. 预测传染效应
4. 可视化结果
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Optional
import json

import torch
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.graph.dexposure_temporal import DeXposureTemporalLoader


class ContagionPredictorWrapper:
    """传染效应预测器"""

    def __init__(
        self,
        model_path: str,
        data_root: str,
        device: str = 'cpu'
    ):
        self.device = device

        # 加载数据加载器
        self.loader = DeXposureTemporalLoader(
            data_path=f"{data_root}/historical-network_week_2025-07-01.json",
            meta_path=f"{data_root}/meta_df.csv",
            prediction_window=1
        )

        # 加载模型
        checkpoint = torch.load(model_path, map_location=device)
        config = checkpoint['config']

        # 从修复版导入模型（包含 BatchNorm）
        from bin.dexposure_train_fixed import ContagionPredictor as Model
        self.model = Model(
            input_dim=13,  # 固定特征维度
            hidden_dim=config['hidden_dim'],
            output_dim=3,
            num_layers=config['num_layers'],
            dropout=config['dropout']
        ).to(device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"✓ 模型加载成功 (epoch {checkpoint['epoch']}, val_loss={checkpoint['val_loss']:.4f})")

    @torch.no_grad()
    def predict(
        self,
        time_idx: int = 0,
        fail_nodes: Optional[List[str]] = None,
        fail_ratio: float = 0.95
    ) -> Dict:
        """
        预测传染效应

        Args:
            time_idx: 时间索引
            fail_nodes: 破产节点ID列表
            fail_ratio: 破产损失比例

        Returns:
            {
                'predictions': np.ndarray (num_nodes, 3),
                'node_ids': List[str],
                'ground_truth': np.ndarray (num_nodes, 3),
                'scenario': str
            }
        """
        # 获取时序图对
        graph_t, graph_t1, labels, node_ids = self.loader.get_temporal_pair(
            t=time_idx,
            inject_failure=fail_nodes,
            failure_ratio=fail_ratio
        )

        # 预测
        graph_t = graph_t.to(self.device)
        node_features = graph_t.ndata['feat']

        predictions = self.model(graph_t, node_features)

        # 应用 sigmoid 到 impact 维度
        predictions_np = predictions.cpu().numpy()
        predictions_np[:, 2] = torch.sigmoid(predictions[:, 2]).cpu().numpy()

        return {
            'predictions': predictions_np,
            'node_ids': node_ids,
            'ground_truth': labels.numpy(),
            'scenario': f"fail_{fail_nodes}_ratio_{fail_ratio}" if fail_nodes else "normal",
            'graph': graph_t
        }

    def get_top_affected_nodes(
        self,
        predictions: np.ndarray,
        node_ids: List[str],
        top_k: int = 20,
        metric: str = 'impact'  # 'impact', 'change_rate', 'abs_loss'
    ) -> pd.DataFrame:
        """
        获取最受影响的节点

        Args:
            predictions: 预测结果 (num_nodes, 3)
            node_ids: 节点ID列表
            top_k: 返回前 k 个节点
            metric: 排序指标

        Returns:
            DataFrame with columns: [node_id, change_rate, abs_loss, impact_score, rank]
        """
        metric_idx = {'change_rate': 0, 'abs_loss': 1, 'impact': 2}[metric]

        # 按指标排序
        if metric == 'change_rate':
            # 变化率：选择最负的（损失最大）
            sorted_indices = np.argsort(predictions[:, metric_idx])[:top_k]
        else:
            # 其他指标：选择最大的
            sorted_indices = np.argsort(predictions[:, metric_idx])[::-1][:top_k]

        results = []
        for rank, idx in enumerate(sorted_indices, 1):
            node_id = node_ids[idx]

            # 获取节点信息
            node_info = self.loader.get_node_info(node_id)

            results.append({
                'rank': rank,
                'node_id': node_id,
                'name': node_info.get('name', 'Unknown'),
                'category': node_info.get('category', 'Unknown'),
                'change_rate': predictions[idx, 0],
                'abs_loss_log': predictions[idx, 1],
                'impact_score': predictions[idx, 2]
            })

        return pd.DataFrame(results)

    def visualize_contagion(
        self,
        predictions: np.ndarray,
        node_ids: List[str],
        output_path: Optional[str] = None
    ):
        """
        可视化传染效应

        生成：
        1. 受影响程度分布图
        2. Top 20 受影响节点柱状图
        3. 传染网络图
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            print("⚠️  需要安装 matplotlib 和 seaborn 进行可视化")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. 受影响程度分布
        ax = axes[0, 0]
        impact_scores = predictions[:, 2]
        ax.hist(impact_scores, bins=50, alpha=0.7, color='red', edgecolor='black')
        ax.set_xlabel('Impact Score', fontsize=12)
        ax.set_ylabel('Number of Nodes', fontsize=12)
        ax.set_title('Distribution of Impact Scores', fontsize=14, fontweight='bold')
        ax.axvline(0.5, color='black', linestyle='--', label='Threshold=0.5')
        ax.legend()

        # 2. 变化率分布
        ax = axes[0, 1]
        change_rates = predictions[:, 0]
        ax.hist(change_rates, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax.set_xlabel('Change Rate', fontsize=12)
        ax.set_ylabel('Number of Nodes', fontsize=12)
        ax.set_title('Distribution of Change Rates', fontsize=14, fontweight='bold')
        ax.axvline(0, color='black', linestyle='--', label='No Change')
        ax.legend()

        # 3. Top 20 受影响节点
        ax = axes[1, 0]
        top_affected = self.get_top_affected_nodes(predictions, node_ids, top_k=20)
        y_pos = np.arange(len(top_affected))
        ax.barh(y_pos, top_affected['impact_score'], color='crimson', alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{row['rank']}. {row['name'][:15]}" for _, row in top_affected.iterrows()],
                           fontsize=9)
        ax.set_xlabel('Impact Score', fontsize=12)
        ax.set_title('Top 20 Most Affected Protocols', fontsize=14, fontweight='bold')
        ax.invert_yaxis()

        # 4. 统计摘要
        ax = axes[1, 1]
        ax.axis('off')

        num_affected_10 = (impact_scores > 0.1).sum()
        num_affected_50 = (impact_scores > 0.5).sum()
        num_affected_90 = (impact_scores > 0.9).sum()

        summary_text = f"""
        传染效应统计摘要

        总节点数: {len(node_ids):,}

        受影响节点数:
          • 轻微影响 (>10%): {num_affected_10:,} ({num_affected_10/len(node_ids)*100:.1f}%)
          • 中等影响 (>50%): {num_affected_50:,} ({num_affected_50/len(node_ids)*100:.1f}%)
          • 严重影响 (>90%): {num_affected_90:,} ({num_affected_90/len(node_ids)*100:.1f}%)

        变化率统计:
          • 平均变化率: {change_rates.mean():.2%}
          • 最大损失: {change_rates.min():.2%}
          • 最大增长: {change_rates.max():.2%}

        受影响程度:
          • 平均受影响程度: {impact_scores.mean():.2%}
          • 最大受影响程度: {impact_scores.max():.2%}
        """

        ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                verticalalignment='center')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✓ 可视化保存至: {output_path}")
        else:
            plt.show()

        return fig


def main():
    parser = argparse.ArgumentParser(description="DeXposure 传染效应预测")
    parser.add_argument('--model', type=str, required=True, help='模型路径')
    parser.add_argument('--data_root', type=str,
                       default='/home/figurich/inter-protocol-exposure/DeXposure/data',
                       help='数据目录')
    parser.add_argument('--fail_nodes', type=str, nargs='+', default=None,
                       help='破产节点ID（例如: 2269 182）')
    parser.add_argument('--fail_ratio', type=float, default=0.95,
                       help='破产损失比例')
    parser.add_argument('--time_idx', type=int, default=0,
                       help='时间索引')
    parser.add_argument('--output', type=str, default=None,
                       help='输出路径')
    parser.add_argument('--device', type=str, default='cpu',
                       help='设备 (cpu/cuda)')

    args = parser.parse_args()

    print("=" * 60)
    print("DeXposure 传染效应预测工具")
    print("=" * 60)

    # 创建预测器
    predictor = ContagionPredictorWrapper(
        model_path=args.model,
        data_root=args.data_root,
        device=args.device
    )

    # 进行预测
    print(f"\n场景设置:")
    print(f"  破产节点: {args.fail_nodes or '无（正常情况）'}")
    print(f"  破产损失比例: {args.fail_ratio:.0%}")
    print(f"  时间索引: {args.time_idx}")
    print()

    result = predictor.predict(
        time_idx=args.time_idx,
        fail_nodes=args.fail_nodes,
        fail_ratio=args.fail_ratio
    )

    # 显示 Top 20 受影响节点
    print("\n" + "=" * 60)
    print("最受影响的协议 (Top 20)")
    print("=" * 60)

    top_affected = predictor.get_top_affected_nodes(
        result['predictions'],
        result['node_ids'],
        top_k=20
    )

    print(top_affected.to_string(index=False))

    # 可视化
    output_path = args.output or f"contagion_prediction_{args.time_idx}.png"
    predictor.visualize_contagion(
        result['predictions'],
        result['node_ids'],
        output_path=output_path
    )

    # 保存预测结果
    result_path = Path(output_path).with_suffix('.json')
    with open(result_path, 'w') as f:
        json.dump({
            'scenario': result['scenario'],
            'time_idx': args.time_idx,
            'num_nodes': len(result['node_ids']),
            'statistics': {
                'mean_impact': float(result['predictions'][:, 2].mean()),
                'max_impact': float(result['predictions'][:, 2].max()),
                'num_affected_50': int((result['predictions'][:, 2] > 0.5).sum()),
                'num_affected_90': int((result['predictions'][:, 2] > 0.9).sum())
            },
            'top_affected': top_affected.to_dict('records')
        }, f, indent=2)

    print(f"\n✓ 预测结果保存至: {result_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
