#!/usr/bin/env python3
"""
快速测试 DeXposure 数据适配器

只测试核心功能，避免内存问题
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("测试 DeXposure 数据适配器...")
print()

try:
    # 测试 1: 导入模块
    print("✓ 测试 1: 导入模块")
    from lib.graph.dexposure_adapter import DeXposureToGraphPFNAdapter
    print("  成功导入 DeXposureToGraphPFNAdapter")
    print()

    # 测试 2: 检查数据文件
    print("✓ 测试 2: 检查数据文件")
    data_path = Path("data/historical-network_week_2025-07-01.json")
    meta_path = Path("data/meta_df.csv")

    if not data_path.exists():
        print(f"  警告: 找不到 {data_path}")
        print(f"  请确保 DeXposure 数据文件存在")
        sys.exit(1)

    if not meta_path.exists():
        print(f"  警告: 找不到 {meta_path}")
        print(f"  请确保元数据文件存在")
        sys.exit(1)

    print(f"  数据文件: {data_path} ✓")
    print(f"  元数据文件: {meta_path} ✓")
    print()

    # 测试 3: 创建适配器（不实际转换数据）
    print("✓ 测试 3: 适配器初始化")
    print("  创建适配器实例...")
    adapter = DeXposureToGraphPFNAdapter(
        data_path=str(data_path),
        meta_path=str(meta_path),
        time_idx=0
    )
    print("  适配器创建成功 ✓")
    print()

    # 测试 4: 测试数据加载
    print("✓ 测试 4: 数据加载")
    print("  加载时序图对...")
    graph_t, graph_t1, labels_all, _ = adapter.loader.get_temporal_pair(0)

    print(f"  graph_t 节点数: {graph_t.num_nodes()}")
    print(f"  graph_t 边数: {graph_t.num_edges()}")
    print(f"  graph_t1 节点数: {graph_t1.num_nodes()}")
    print(f"  graph_t1 边数: {graph_t1.num_edges()}")
    print(f"  标签形状: {labels_all.shape}")
    print(f"  标签范围: [{labels_all.min():.4f}, {labels_all.max():.4f}]")
    print()

    # 测试 5: 测试时序编码
    print("✓ 测试 5: 时序编码")
    import torch
    features = graph_t.ndata['feat']
    print(f"  原始特征维度: {features.shape[1]}")

    features_encoded = adapter._add_temporal_encoding(
        features,
        time_idx=0,
        num_total_times=len(adapter.loader.dates)
    )
    print(f"  编码后特征维度: {features_encoded.shape[1]}")
    print(f"  增加 {features_encoded.shape[1] - features.shape[1]} 维时序特征 ✓")
    print()

    # 测试 6: 测试掩码创建
    print("✓ 测试 6: 数据集划分掩码")
    masks_temporal = adapter._create_masks(
        num_nodes=graph_t.num_nodes(),
        split_strategy='temporal',
        time_idx=0,
        num_total_times=len(adapter.loader.dates)
    )
    print(f"  Temporal 策略:")
    print(f"    训练集: {masks_temporal['train'].sum()}")
    print(f"    验证集: {masks_temporal['val'].sum()}")
    print(f"    测试集: {masks_temporal['test'].sum()}")
    print()

    masks_random = adapter._create_masks(
        num_nodes=graph_t.num_nodes(),
        split_strategy='random',
        time_idx=0,
        num_total_times=len(adapter.loader.dates)
    )
    print(f"  Random 策略:")
    print(f"    训练集: {masks_random['train'].sum()}")
    print(f"    验证集: {masks_random['val'].sum()}")
    print(f"    测试集: {masks_random['test'].sum()}")
    print()

    print("=" * 60)
    print("🎉 所有核心功能测试通过!")
    print("=" * 60)
    print()
    print("适配器功能正常，可以进行数据转换。")
    print()
    print("使用示例:")
    print("  from lib.graph.data import load_data")
    print("  dataset = load_data('data/', time_idx=0)")
    print()

except Exception as e:
    print(f"\n❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
