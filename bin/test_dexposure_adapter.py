#!/usr/bin/env python3
"""
测试 DeXposure 数据适配器

验证：
1. 数据加载功能
2. GraphData 格式正确性
3. GraphDataset 创建
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from lib.graph.data import load_data, GraphDataset
from lib.graph.util import Setting


def test_load_data():
    """测试 1: 数据加载功能"""
    print("=" * 60)
    print("测试 1: 数据加载功能")
    print("=" * 60)

    try:
        # 加载第 0 周的数据
        print("\n加载 DeXposure 数据 (time_idx=0)...")
        dataset = load_data(
            path='data/',
            time_idx=0,
            split_strategy='temporal',
            label_idx=0  # 使用变化率作为标签
        )

        print("\n✓ 数据加载成功!")
        print(f"  数据集名称: {dataset['name']}")
        print(f"  节点数: {dataset['graph'].num_nodes()}")
        print(f"  边数: {dataset['graph'].num_edges()}")
        print(f"  特征维度: {dataset['num_features'].shape}")
        print(f"  标签形状: {dataset['labels'].shape}")
        print(f"  训练集大小: {dataset['masks']['train'].sum()}")
        print(f"  验证集大小: {dataset['masks']['val'].sum()}")
        print(f"  测试集大小: {dataset['masks']['test'].sum()}")

        # 验证数据格式
        assert 'name' in dataset, "缺少 'name' 字段"
        assert 'graph' in dataset, "缺少 'graph' 字段"
        assert 'labels' in dataset, "缺少 'labels' 字段"
        assert 'masks' in dataset, "缺少 'masks' 字段"
        assert 'num_features' in dataset, "缺少 'num_features' 字段"
        assert 'cat_features' in dataset, "缺少 'cat_features' 字段"
        assert 'ratio_features' in dataset, "缺少 'ratio_features' 字段"

        print("\n✓ GraphData 格式验证通过!")
        return True

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_graph_dataset():
    """测试 2: GraphDataset 创建"""
    print("\n" + "=" * 60)
    print("测试 2: GraphDataset 创建")
    print("=" * 60)

    try:
        print("\n创建 GraphDataset 对象...")
        dataset = GraphDataset.from_dir(
            path='data/',
            setting='transductive',
            time_idx=0
        )

        print("\n✓ GraphDataset 创建成功!")
        print(f"  任务类型: {dataset.task.type_}")
        print(f"  是否回归任务: {dataset.task.is_regression}")
        print(f"  是否分类任务: {dataset.task.is_classification}")
        print(f"  是否转导式学习: {dataset.task.is_transductive}")
        print(f"  节点数: {dataset.data['graph'].num_nodes()}")

        # 验证 GraphDataset 属性
        assert hasattr(dataset, 'data'), "缺少 'data' 属性"
        assert hasattr(dataset, 'task'), "缺少 'task' 属性"
        assert dataset.task.is_regression, "任务类型应该是回归"

        print("\n✓ GraphDataset 验证通过!")
        return True

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_snapshots():
    """测试 3: 多个时间快照加载"""
    print("\n" + "=" * 60)
    print("测试 3: 多个时间快照加载")
    print("=" * 60)

    try:
        print("\n加载多个时间快照...")

        for time_idx in [0, 1, 2]:
            print(f"\n  时间快照 {time_idx}:")
            dataset = load_data(
                path='data/',
                time_idx=time_idx,
                split_strategy='temporal'
            )
            print(f"    名称: {dataset['name']}")
            print(f"    节点数: {dataset['graph'].num_nodes()}")
            print(f"    特征维度: {dataset['num_features'].shape[1]}")

        print("\n✓ 多个时间快照加载成功!")
        return True

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_different_labels():
    """测试 4: 不同回归标签"""
    print("\n" + "=" * 60)
    print("测试 4: 不同回归标签")
    print("=" * 60)

    try:
        print("\n测试三种不同的回归标签...")

        label_names = ['变化率', '绝对损失', '受影响程度']
        for label_idx in range(3):
            print(f"\n  标签 {label_idx} ({label_names[label_idx]}):")
            dataset = load_data(
                path='data/',
                time_idx=0,
                label_idx=label_idx
            )

            labels = dataset['labels']
            print(f"    标签形状: {labels.shape}")
            print(f"    标签范围: [{labels.min():.4f}, {labels.max():.4f}]")
            print(f"    标签均值: {labels.mean():.4f}")
            print(f"    标签标准差: {labels.std():.4f}")

        print("\n✓ 不同回归标签测试成功!")
        return True

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_split_strategies():
    """测试 5: 不同划分策略"""
    print("\n" + "=" * 60)
    print("测试 5: 不同划分策略")
    print("=" * 60)

    try:
        print("\n测试两种划分策略...")

        for strategy in ['random', 'temporal']:
            print(f"\n  策略: {strategy}")
            dataset = load_data(
                path='data/',
                time_idx=0,
                split_strategy=strategy
            )

            masks = dataset['masks']
            train_size = masks['train'].sum()
            val_size = masks['val'].sum()
            test_size = masks['test'].sum()
            total = train_size + val_size + test_size

            print(f"    训练集: {train_size} ({100*train_size/total:.1f}%)")
            print(f"    验证集: {val_size} ({100*val_size/total:.1f}%)")
            print(f"    测试集: {test_size} ({100*test_size/total:.1f}%)")

        print("\n✓ 不同划分策略测试成功!")
        return True

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "█" * 60)
    print("█" + " " * 58 + "█")
    print("█" + "  DeXposure 数据适配器测试套件".center(56) + "█")
    print("█" + " " * 58 + "█")
    print("█" * 60)

    tests = [
        ("数据加载功能", test_load_data),
        ("GraphDataset 创建", test_graph_dataset),
        ("多个时间快照加载", test_multiple_snapshots),
        ("不同回归标签", test_different_labels),
        ("不同划分策略", test_split_strategies),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n❌ 测试 '{test_name}' 发生异常: {e}")
            results.append((test_name, False))

    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)

    for test_name, success in results:
        status = "✓ 通过" if success else "❌ 失败"
        print(f"  {test_name}: {status}")

    passed = sum(1 for _, success in results if success)
    total = len(results)

    print(f"\n总计: {passed}/{total} 测试通过")

    if passed == total:
        print("\n🎉 所有测试通过!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} 个测试失败")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
