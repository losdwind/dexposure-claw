#!/usr/bin/env python3
"""
DeXposure 数据转换脚本

将 DeXposure 时序快照转换为 GraphLand 格式，便于持久化存储和重用。

用法:
    # 转换单个时间快照
    python bin/convert_dexposure.py --time-idx 0

    # 转换所有时间快照
    python bin/convert_dexposure.py --all

    # 使用随机划分策略
    python bin/convert_dexposure.py --all --split-strategy random

    # 转换到指定目录
    python bin/convert_dexposure.py --all --output-dir data/dexposure_graphland
"""

import argparse
import json
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import yaml
from lib.graph.dexposure_adapter import DeXposureToGraphPFNAdapter


def save_graphland_format(graph_data: dict, output_dir: Path):
    """
    将 GraphData 保存为 GraphLand 格式

    GraphLand 格式包含：
    - info.yaml: 数据集元信息
    - features.csv: 节点特征
    - targets.csv: 回归标签
    - edgelist.csv: 边列表
    - split_masks_RL.csv: 数据集划分掩码

    Args:
        graph_data: GraphData 字典
        output_dir: 输出目录
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"  保存到 {output_dir}...")

    # 1. 保存 info.yaml
    info = {
        'dataset_name': graph_data['name'],
        'task': 'regression',
        'metric': 'r2',
        'num_features_names': [
            f'feat_{i}' for i in range(graph_data['num_features'].shape[1])
        ],
        'cat_features_names': [],
        'proportion_features_names': [],
        'target_name': 'tvl_change_rate',
        'graph_is_directed': False,
        'graph_is_weighted': True,
        'has_unlabeled_nodes': False,
        'has_nans_in_num_features': False
    }

    with open(output_dir / 'info.yaml', 'w') as f:
        yaml.dump(info, f, default_flow_style=False)

    # 2. 保存 features.csv
    features_df = pd.DataFrame(
        graph_data['num_features'],
        index=range(graph_data['graph'].num_nodes())
    )
    features_df.to_csv(output_dir / 'features.csv')

    # 3. 保存 targets.csv
    targets_df = pd.DataFrame(graph_data['labels'])
    targets_df.to_csv(output_dir / 'targets.csv', header=['target'])

    # 4. 保存 edgelist.csv
    src, dst = graph_data['graph'].edges()
    edgelist_df = pd.DataFrame({
        'source': src.numpy(),
        'target': dst.numpy()
    })
    edgelist_df.to_csv(output_dir / 'edgelist.csv', index=False)

    # 5. 保存 split_masks_RL.csv
    masks_df = pd.DataFrame({
        'train': graph_data['masks']['train'],
        'val': graph_data['masks']['val'],
        'test': graph_data['masks']['test']
    })
    masks_df.to_csv(output_dir / 'split_masks_RL.csv', index=False)

    print(f"  ✓ 保存完成")


def main():
    parser = argparse.ArgumentParser(
        description='转换 DeXposure 数据为 GraphLand 格式',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # 数据路径参数
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/historical-network_week_2025-07-01.json',
        help='DeXposure JSON 文件路径'
    )
    parser.add_argument(
        '--meta-path',
        type=str,
        default='data/meta_df.csv',
        help='元数据 CSV 文件路径'
    )

    # 输出参数
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/dexposure_graphland',
        help='输出目录'
    )

    # 转换参数
    parser.add_argument(
        '--time-idx',
        type=int,
        default=0,
        help='时间索引（用于转换单个快照）'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='转换所有时间快照'
    )
    parser.add_argument(
        '--split-strategy',
        type=str,
        choices=['temporal', 'random'],
        default='temporal',
        help='数据集划分策略'
    )
    parser.add_argument(
        '--label-idx',
        type=int,
        default=0,
        choices=[0, 1, 2],
        help='回归标签索引 (0=变化率, 1=绝对损失, 2=受影响程度)'
    )

    args = parser.parse_args()

    # 验证数据路径
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"❌ 错误: 找不到数据文件 {args.data_path}")
        sys.exit(1)

    meta_path = Path(args.meta_path)
    if not meta_path.exists():
        print(f"❌ 错误: 找不到元数据文件 {args.meta_path}")
        sys.exit(1)

    # 获取总快照数
    print(f"读取数据文件: {data_path}")
    with open(data_path) as f:
        network_data = json.load(f)['data']
        num_snapshots = len(network_data)

    print(f"时间快照总数: {num_snapshots}")
    print(f"输出目录: {args.output_dir}")
    print(f"划分策略: {args.split_strategy}")
    print(f"回归标签: {['变化率', '绝对损失', '受影响程度'][args.label_idx]}")
    print()

    # 转换数据
    if args.all:
        # 转换所有时间快照
        print(f"开始转换所有 {num_snapshots} 个时间快照...\n")

        success_count = 0
        for idx in range(num_snapshots):
            try:
                print(f"[{idx+1}/{num_snapshots}] 转换时间快照 {idx}...")

                adapter = DeXposureToGraphPFNAdapter(
                    data_path=str(data_path),
                    meta_path=str(meta_path),
                    time_idx=idx,
                    split_strategy=args.split_strategy,
                    label_idx=args.label_idx
                )

                graph_data = adapter.convert()

                output_path = Path(args.output_dir) / f'week_{idx:02d}'
                save_graphland_format(graph_data, output_path)

                success_count += 1
                print()

            except Exception as e:
                print(f"  ❌ 转换失败: {e}")
                print()
                continue

        print(f"\n✓ 转换完成!")
        print(f"  成功: {success_count}/{num_snapshots}")
        print(f"  输出目录: {args.output_dir}")

    else:
        # 转换单个快照
        if args.time_idx >= num_snapshots:
            print(f"❌ 错误: time_idx {args.time_idx} 超出范围 [0, {num_snapshots-1}]")
            sys.exit(1)

        print(f"转换时间快照 {args.time_idx}...\n")

        adapter = DeXposureToGraphPFNAdapter(
            data_path=str(data_path),
            meta_path=str(meta_path),
            time_idx=args.time_idx,
            split_strategy=args.split_strategy,
            label_idx=args.label_idx
        )

        graph_data = adapter.convert()

        output_path = Path(args.output_dir) / f'week_{args.time_idx:02d}'
        save_graphland_format(graph_data, output_path)

        print(f"\n✓ 转换完成!")
        print(f"  输出目录: {output_path}")


if __name__ == '__main__':
    main()
