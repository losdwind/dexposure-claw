"""
DeXposure 时序图数据加载器 - 用于传染效应预测
支持回归任务：预测节点在危机中的受影响程度
"""

import json
import pandas as pd
import dgl
import torch
import numpy as np
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional


class DeXposureTemporalLoader:
    """
    DeXposure 时序数据加载器

    用途：
    1. 加载历史网络快照序列
    2. 构建时序图对
    3. 支持手动注入"节点破产"场景
    4. 预测传染效应（回归任务）
    """

    def __init__(
        self,
        data_path: str,
        meta_path: str,
        prediction_window: int = 1,  # 预测未来几周
        auto_download: bool = True  # 自动下载数据集
    ):
        """
        Args:
            data_path: historical-network_week_xxx.json 路径
            meta_path: meta_df.csv 路径
            prediction_window: 预测窗口（默认1周）
            auto_download: 数据文件不存在时是否自动下载
        """
        self.data_path = Path(data_path)
        self.meta_path = Path(meta_path)
        self.prediction_window = prediction_window

        # 检查数据文件是否存在
        self._check_data_files(auto_download)

        # 加载数据
        print(f"加载数据: {self.data_path}")
        with open(self.data_path) as f:
            self.network_data = json.load(f)['data']

        # 加载元数据（将ID转换为字符串以匹配节点ID）
        self.meta_df = pd.read_csv(self.meta_path)
        self.meta_df['id'] = self.meta_df['id'].astype(str)
        self.meta_dict = self.meta_df.set_index('id').to_dict('index')

        self.dates = sorted(self.network_data.keys())
        print(f"✓ 加载了 {len(self.dates)} 个时间快照")
        print(f"  日期范围: {self.dates[0]} → {self.dates[-1]}")

    def _build_node_features(self, node: Dict) -> np.ndarray:
        """
        构建节点特征向量（修复 NaN 问题）

        特征包括：
        1. 节点总规模 (log scale)
        2. 资产种类数量
        3. 资产多样性 (熵)
        4. 最大单一资产占比
        5. 节点类别 (one-hot: CEX, DEX, Lending, etc.)
        """
        node_id = node['id']
        size = node['size']
        comp = node['composition']

        # 基础统计特征（添加数值稳定性）
        log_size = np.log1p(max(size, 0))  # 确保非负
        num_assets = len(comp)

        # 资产多样性（Shannon熵）- 修复 NaN 问题
        diversity = 0.0
        max_concentration = 0.0
        
        if num_assets > 0:
            values = np.array(list(comp.values()))
            # 过滤负值和 NaN
            values = np.maximum(values, 0)
            values = np.nan_to_num(values, 0.0)
            
            total = values.sum()
            if total > 1e-10:  # 避免除以接近零的数
                proportions = values / total
                # 使用更稳定的熵计算
                # 避免 log(0) 产生 -inf
                proportions_safe = np.maximum(proportions, 1e-10)
                diversity = -np.sum(proportions * np.log(proportions_safe))
                # 裁剪到合理范围
                diversity = np.clip(diversity, 0, 10)
                max_concentration = proportions.max()

        # 类别特征 (从 meta_df 获取)
        category = self.meta_dict.get(str(node_id), {}).get('category', 'Unknown')
        category_features = self._encode_category(category)

        features = np.array([
            log_size,
            num_assets,
            diversity,
            max_concentration,
            *category_features
        ], dtype=np.float32)

        # 最后检查：移除 NaN 和 Inf
        features = np.nan_to_num(features, nan=0.0, posinf=100.0, neginf=-100.0)

        return features

    def _encode_category(self, category: str) -> List[float]:
        """编码协议类别为 one-hot"""
        categories = ['CEX', 'Lending', 'Liquid Staking', 'DEX', 'Bridge',
                     'CDP', 'Restaking', 'Chain', 'Unknown']
        one_hot = [1.0 if cat == category else 0.0 for cat in categories]
        return one_hot

    def get_temporal_pair(
        self,
        t: int,
        inject_failure: Optional[List[str]] = None,
        failure_ratio: float = 0.9  # 破产节点保留10%资产
    ) -> Tuple[dgl.DGLGraph, dgl.DGLGraph, torch.Tensor]:
        """
        获取时序图对：(t 时刻图, t+1 时刻图, 变化标签)

        Args:
            t: 时间索引
            inject_failure: 要注入破产的节点ID列表（例如 ['2269'] 代表 Binance）
            failure_ratio: 破产导致的资产损失比例（默认90%）

        Returns:
            graph_t: t 时刻的图
            graph_t1: t+1 时刻的图（真实值，用于验证）
            labels: 每个节点的变化标签（回归目标）
                - shape: (num_nodes, 3)
                - [:, 0]: TVL 变化率 (delta_size / size_t)
                - [:, 1]: 绝对损失金额 (log scale)
                - [:, 2]: 受影响程度 [0, 1]
        """
        if t + self.prediction_window >= len(self.dates):
            raise ValueError(f"时间索引超出范围: {t} + {self.prediction_window} >= {len(self.dates)}")

        date_t = self.dates[t]
        date_t1 = self.dates[t + self.prediction_window]

        snapshot_t = self.network_data[date_t]
        snapshot_t1 = self.network_data[date_t1]

        # 获取两个时间点的所有节点ID（取交集，确保一致）
        nodes_t = snapshot_t['nodes']
        nodes_t1 = snapshot_t1['nodes']

        # 过滤掉 None 的节点ID
        node_ids_t = set(n['id'] for n in nodes_t if n['id'] is not None)
        node_ids_t1 = set(n['id'] for n in nodes_t1 if n['id'] is not None)

        # 使用两个时间点共同的节点，并转换为字符串
        common_node_ids = sorted(str(nid) for nid in (node_ids_t & node_ids_t1))

        if len(common_node_ids) < len(node_ids_t) * 0.9:
            print(f"  ⚠️  警告: 只有 {len(common_node_ids)}/{len(node_ids_t)} 个节点在两个时间点都存在")

        # 构建 t 时刻的图（只包含共同节点）
        graph_t, node_ids_list, sizes_t = self._build_graph_snapshot(
            snapshot_t,
            common_node_ids=common_node_ids,
            inject_failure=inject_failure,
            failure_ratio=failure_ratio
        )

        # 构建 t+1 时刻的图（只包含共同节点）
        graph_t1, _, sizes_t1 = self._build_graph_snapshot(
            snapshot_t1,
            common_node_ids=common_node_ids,
            inject_failure=None  # t+1 不注入破产
        )

        # 计算变化标签（回归目标）
        labels = self._compute_labels(sizes_t, sizes_t1)

        return graph_t, graph_t1, labels, node_ids_list

    def _build_graph_snapshot(
        self,
        snapshot: Dict,
        common_node_ids: Optional[List[str]] = None,
        inject_failure: Optional[List[str]] = None,
        failure_ratio: float = 0.9
    ) -> Tuple[dgl.DGLGraph, List[str], np.ndarray]:
        """构建单个时间点的图快照"""
        nodes = snapshot['nodes']
        links = snapshot['links']

        # 如果指定了 common_node_ids，只保留这些节点
        if common_node_ids is not None:
            node_id_set = set(common_node_ids)
            nodes = [n for n in nodes if n['id'] is not None and str(n['id']) in node_id_set]
            # 按照 common_node_ids 的顺序排序
            node_id_to_node = {str(n['id']): n for n in nodes}
            nodes = [node_id_to_node[nid] for nid in common_node_ids if nid in node_id_to_node]

        # 节点ID列表（统一转换为字符串）
        node_ids = [str(n['id']) for n in nodes if n['id'] is not None]
        id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}

        # 注入破产场景
        if inject_failure:
            nodes = self._inject_node_failure(nodes, inject_failure, failure_ratio)

        # 提取节点特征和规模
        node_features = []
        node_sizes = []

        for node in nodes:
            features = self._build_node_features(node)
            node_features.append(features)
            node_sizes.append(node['size'])

        # 构建边
        src_nodes = []
        dst_nodes = []
        edge_weights = []

        for link in links:
            source = link['source']
            target = link['target']

            # 转换为字符串并检查
            if source is None or target is None:
                continue

            source_str = str(source)
            target_str = str(target)

            if source_str not in id_to_idx or target_str not in id_to_idx:
                continue

            src_idx = id_to_idx[source_str]
            dst_idx = id_to_idx[target_str]
            weight = link['size']

            src_nodes.append(src_idx)
            dst_nodes.append(dst_idx)
            edge_weights.append(weight)

        # 创建双向图（无向图）
        graph = dgl.graph(
            (src_nodes + dst_nodes, dst_nodes + src_nodes),
            num_nodes=len(nodes)
        )

        # 添加特征
        graph.ndata['feat'] = torch.FloatTensor(node_features)
        graph.ndata['size'] = torch.FloatTensor(node_sizes)

        # 添加边权重
        edge_weights_bi = edge_weights + edge_weights
        graph.edata['weight'] = torch.FloatTensor(edge_weights_bi)

        return graph, node_ids, np.array(node_sizes)

    def _inject_node_failure(
        self,
        nodes: List[Dict],
        failure_ids: List[str],
        failure_ratio: float
    ) -> List[Dict]:
        """
        注入节点破产场景

        模拟节点失败：
        - 将指定节点的 size 减少到原来的 (1 - failure_ratio)
        - composition 中的资产按比例减少
        """
        nodes_copy = [node.copy() for node in nodes]

        for node in nodes_copy:
            if str(node['id']) in failure_ids:
                # 破产！资产大幅缩水
                node['size'] *= (1 - failure_ratio)

                # composition 也按比例缩减
                if node['composition']:
                    node['composition'] = {
                        k: v * (1 - failure_ratio)
                        for k, v in node['composition'].items()
                    }

                print(f"  💥 注入破产: 节点 {node['id']} 资产损失 {failure_ratio*100:.0f}%")

        return nodes_copy

    def _compute_labels(
        self,
        sizes_t: np.ndarray,
        sizes_t1: np.ndarray
    ) -> torch.Tensor:
        """
        计算回归标签

        Returns:
            labels: (num_nodes, 3)
                - [:, 0]: 相对变化率
                - [:, 1]: 绝对损失 (log scale)
                - [:, 2]: 受影响程度 [0, 1]
        """
        # 避免除零
        sizes_t_safe = np.maximum(sizes_t, 1.0)

        # 1. 相对变化率
        delta_ratio = (sizes_t1 - sizes_t) / sizes_t_safe

        # 2. 绝对损失（取负值的 log）
        abs_loss = sizes_t - sizes_t1
        log_abs_loss = np.log1p(np.maximum(abs_loss, 0))

        # 3. 受影响程度 [0, 1]
        # 如果损失超过50%，算严重受影响
        impact_score = np.clip(-delta_ratio, 0, 1)

        labels = np.stack([delta_ratio, log_abs_loss, impact_score], axis=1)
        return torch.FloatTensor(labels)

    def get_node_info(self, node_id: str) -> Dict:
        """获取节点的详细信息"""
        meta = self.meta_dict.get(str(node_id), {})
        return {
            'id': node_id,
            'name': meta.get('name', 'Unknown'),
            'category': meta.get('category', 'Unknown')
        }

    def get_all_temporal_pairs(
        self,
        inject_failure: Optional[List[str]] = None
    ) -> List[Tuple]:
        """获取所有可用的时序图对"""
        pairs = []
        num_pairs = len(self.dates) - self.prediction_window

        print(f"\n构建 {num_pairs} 个时序图对...")
        for t in range(num_pairs):
            try:
                pair = self.get_temporal_pair(t, inject_failure=inject_failure)
                pairs.append(pair)
            except Exception as e:
                print(f"  ⚠️  跳过第 {t} 个图对: {e}")

        print(f"✓ 成功构建 {len(pairs)} 个图对\n")
        return pairs

    def _check_data_files(self, auto_download: bool = True):
        """
        检查数据文件是否存在,如果不存在则提示下载或自动下载

        Args:
            auto_download: 是否自动运行下载脚本
        """
        missing_files = []

        # 检查数据文件
        if not self.data_path.exists():
            missing_files.append(str(self.data_path))

        if not self.meta_path.exists():
            missing_files.append(str(self.meta_path))

        if not missing_files:
            return  # 所有文件都存在

        # 有文件缺失
        print("\n" + "="*60)
        print("❌ 数据文件缺失!")
        print("="*60)
        for f in missing_files:
            print(f"  - {f}")

        print("\n数据集大小约 1.2GB,请确保网络连接稳定\n")

        if auto_download:
            # 自动下载
            print("正在尝试自动下载数据集...\n")

            # 获取数据目录
            data_dir = self.data_path.parent

            # 调用下载脚本
            try:
                script_path = Path(__file__).parent.parent.parent / "bin" / "download_dataset.py"
                result = subprocess.run(
                    [sys.executable, str(script_path), "--data-dir", str(data_dir)],
                    capture_output=True,
                    text=True
                )

                if result.returncode == 0:
                    print("\n✓ 数据集下载成功!")
                else:
                    print(f"\n✗ 自动下载失败:")
                    print(result.stderr)
                    print("\n请手动运行以下命令下载数据集:")
                    print(f"  python {script_path} --data-dir {data_dir}")
                    sys.exit(1)
            except Exception as e:
                print(f"\n✗ 自动下载出错: {e}")
                print("\n请手动运行以下命令下载数据集:")
                script_path = Path(__file__).parent.parent.parent / "bin" / "download_dataset.py"
                print(f"  python {script_path} --data-dir {data_dir}")
                sys.exit(1)
        else:
            # 不自动下载,只提示
            script_path = Path(__file__).parent.parent.parent / "bin" / "download_dataset.py"
            print("请运行以下命令下载数据集:")
            print(f"  python {script_path} --data-dir {self.data_path.parent}\n")
            sys.exit(1)


# ============ 使用示例 ============

if __name__ == "__main__":
    loader = DeXposureTemporalLoader(
        data_path="/home/figurich/inter-protocol-exposure/DeXposure/data/historical-network_week_2025-07-01.json",
        meta_path="/home/figurich/inter-protocol-exposure/DeXposure/data/meta_df.csv",
        prediction_window=1
    )

    # 示例1：正常情况下的预测
    print("=" * 60)
    print("示例1: 正常情况")
    print("=" * 60)
    graph_t, graph_t1, labels = loader.get_temporal_pair(t=0)
    print(f"图 t:   {graph_t.num_nodes()} 节点, {graph_t.num_edges()} 边")
    print(f"图 t+1: {graph_t1.num_nodes()} 节点, {graph_t1.num_edges()} 边")
    print(f"标签形状: {labels.shape}")
    print(f"平均变化率: {labels[:, 0].mean():.4f}")
    print(f"受影响节点数 (>10%): {(labels[:, 2] > 0.1).sum()}")

    # 示例2：模拟 Binance 破产
    print("\n" + "=" * 60)
    print("示例2: 模拟 Binance (2269) 破产")
    print("=" * 60)
    binance_info = loader.get_node_info('2269')
    print(f"破产节点: {binance_info}")

    graph_t, graph_t1, labels = loader.get_temporal_pair(
        t=0,
        inject_failure=['2269'],  # Binance
        failure_ratio=0.95  # 损失95%资产
    )

    print(f"\n预测结果:")
    print(f"  受严重影响的节点数 (>50%): {(labels[:, 2] > 0.5).sum()}")
    print(f"  最大损失率: {labels[:, 0].min():.2%}")
    print(f"  平均受影响程度: {labels[:, 2].mean():.4f}")

    # 找出最受影响的前10个节点
    top_affected = torch.argsort(labels[:, 2], descending=True)[:10]
    print(f"\n最受影响的前10个节点:")
    for i, idx in enumerate(top_affected):
        # 这里需要从 graph 获取节点ID，简化起见先打印索引
        impact = labels[idx, 2].item()
        loss_ratio = labels[idx, 0].item()
        print(f"  {i+1}. 节点 #{idx}: 受影响程度={impact:.2%}, 损失率={loss_ratio:.2%}")
