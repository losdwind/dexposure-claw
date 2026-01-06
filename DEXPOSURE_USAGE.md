# DeXposure 数据适配器使用指南

## 📋 概述

DeXposure 数据适配器已成功集成到 GraphPFN 项目中！现在可以直接使用 GraphPFN 的标准接口加载 DeXposure 的时序图数据。

## ✅ 已完成的功能

1. ✅ **数据适配器** (`lib/graph/dexposure_adapter.py`)
   - 将 DeXposure 数据转换为 GraphData 格式
   - 支持时序编码（时间嵌入 + 位置编码）
   - 支持两种数据集划分策略（temporal/random）

2. ✅ **数据加载集成** (`lib/graph/data.py`)
   - 添加了 `_load_dexposure_data()` 函数
   - 修改了 `load_data()` 支持 DeXposure 数据集
   - 与 GraphPFN 现有流程无缝集成

3. ✅ **批量转换工具** (`bin/convert_dexposure.py`)
   - 命令行工具，可批量转换时间快照
   - 支持持久化为 GraphLand 格式

4. ✅ **测试验证** (`bin/test_quick.py`)
   - 所有核心功能测试通过
   - 验证了数据加载、时序编码、数据集划分等功能

## 🚀 快速开始

### 方式 1: 直接加载（推荐）

最简单的方式是直接使用 GraphPFN 的 `load_data()` 函数：

```python
from lib.graph.data import load_data, GraphDataset

# 加载第 0 周的数据
dataset = load_data(
    path='data/',              # DeXposure 数据目录
    time_idx=0,                # 时间索引（0-7）
    split_strategy='temporal', # 划分策略：'temporal' 或 'random'
    label_idx=0                # 回归标签：0=变化率, 1=绝对损失, 2=受影响程度
)

# 查看数据信息
print(f"数据集名称: {dataset['name']}")
print(f"节点数: {dataset['graph'].num_nodes()}")
print(f"边数: {dataset['graph'].num_edges()}")
print(f"特征维度: {dataset['num_features'].shape}")
print(f"训练集: {dataset['masks']['train'].sum()}")
```

### 方式 2: 使用 GraphDataset

创建标准的 GraphDataset 对象：

```python
from lib.graph.data import GraphDataset

dataset = GraphDataset.from_dir(
    path='data/',
    setting='transductive',
    time_idx=0
)

print(f"任务类型: {dataset.task.type_}")       # TaskType.REGRESSION
print(f"是否回归: {dataset.task.is_regression}") # True
```

### 方式 3: 批量转换为 GraphLand 格式

将所有时间快照保存为 GraphLand 格式（便于持久化和重用）：

```bash
cd graphpfn

# 转换单个时间快照
.venv/bin/python bin/convert_dexposure.py --time-idx 0

# 转换所有时间快照
.venv/bin/python bin/convert_dexposure.py --all

# 使用随机划分策略
.venv/bin/python bin/convert_dexposure.py --all --split-strategy random

# 转换到指定目录
.venv/bin/python bin/convert_dexposure.py --all --output-dir data/dexposure_graphland
```

转换后的数据可以像其他 GraphLand 数据集一样使用：

```python
from lib.graph.data import load_data

dataset = load_data('data/dexposure_graphland/week_00')
```

## 📊 数据结构

### GraphData 格式

返回的 GraphData 包含以下字段：

```python
{
    'name': str,                    # 'dexposure-week-00'
    'graph': dgl.DGLGraph,          # DGL 图对象
    'labels': np.ndarray,           # 回归标签 (num_nodes,)
    'masks': dict,                  # {'train': ..., 'val': ..., 'test': ...}
    'num_features': np.ndarray,     # 节点特征 (num_nodes, 18)
    'cat_features': None,           # 无分类特征
    'ratio_features': None          # 无比例特征
}
```

### 节点特征

节点特征包含 **18 维**：
- **原始特征** (13维):
  - log_size: 对数规模
  - num_assets: 资产种类数
  - diversity: 资产多样性（熵）
  - max_concentration: 最大单一资产占比
  - category_one_hot: 9维类别独热编码

- **时序特征** (5维):
  - 位置编码: 4维 sin/cos 编码
  - 归一化时间: 1维 (time_idx / total_times)

### 回归标签

可选择三种回归标签：
- `label_idx=0`: TVL 变化率（推荐）
- `label_idx=1`: 绝对损失
- `label_idx=2`: 受影响程度

## 🔧 参数说明

### split_strategy（数据集划分策略）

#### `temporal`（推荐）

时间感知的划分策略，模拟真实预测场景：

- **早期时间点**（前 70%）: 主要用于训练
  - 训练集 80% | 验证集 10% | 测试集 10%

- **中期时间点**（70%-85%）: 主要用于验证
  - 训练集 30% | 验证集 60% | 测试集 10%

- **后期时间点**（后 15%）: 主要用于测试
  - 训练集 10% | 验证集 20% | 测试集 70%

#### `random`

随机划分节点：
- 训练集 70% | 验证集 15% | 测试集 15%

### label_idx（回归标签选择）

| 索引 | 名称 | 说明 | 推荐场景 |
|-----|------|------|---------|
| 0 | 变化率 | TVL 相对变化率 | ⭐ 推荐，数值稳定 |
| 1 | 绝对损失 | TVL 绝对损失 | 需要预测绝对值 |
| 2 | 受影响程度 | 0-1 分数 | 危机影响评估 |

## 📈 测试结果

所有核心功能测试通过 ✅

```
✓ 测试 1: 导入模块
✓ 测试 2: 检查数据文件
✓ 测试 3: 适配器初始化
✓ 测试 4: 数据加载 (10,437 节点, 143,522 边)
✓ 测试 5: 时序编码 (13维 → 18维)
✓ 测试 6: 数据集划分掩码
```

**数据统计**（第 0 周）：
- 节点数: 10,437
- 边数: 143,522
- 原始特征: 13 维
- 增强特征: 18 维
- 训练集: 8,385 (temporal 策略)
- 验证集: 1,047
- 测试集: 1,005

## 💡 使用建议

### 1. 任务选择

- **TVL 预测**: 使用 `label_idx=0`（变化率）
- **危机预测**: 使用 `label_idx=2`（受影响程度）

### 2. 划分策略

- **时序建模**: 使用 `split_strategy='temporal'`
- **静态图学习**: 使用 `split_strategy='random'`

### 3. 时序建模

虽然每个时间快照是独立的，但通过以下方式保留时序信息：

1. **特征中的时序编码**: 时间嵌入 + 位置编码
2. **Temporal 划分策略**: 早期训练，后期测试
3. **跨时间迁移学习**: 在早期时间快照上预训练，在后期上微调

### 4. 内存管理

如果遇到内存问题：

```python
# 只加载需要的部分数据
dataset = load_data('data/', time_idx=0)

# 或者先转换为 GraphLand 格式再使用
# python bin/convert_dexposure.py --time-idx 0
dataset = load_data('data/dexposure_graphland/week_00')
```

## 📚 相关文件

| 文件 | 说明 |
|------|------|
| `lib/graph/dexposure_adapter.py` | 数据适配器（核心） |
| `lib/graph/dexposure_temporal.py` | 时序加载器（已存在） |
| `lib/graph/data.py` | GraphPFN 数据加载器（已修改） |
| `bin/convert_dexposure.py` | 批量转换工具 |
| `bin/test_quick.py` | 快速测试脚本 |

## 🎓 下一步

现在数据已经准备好，可以：

1. **训练 GraphPFN 模型**
   ```python
   from lib.model import GraphPFNModel
   model = GraphPFNModel(dataset)
   model.fit()
   ```

2. **评估性能**
   ```python
   predictions = model.predict(dataset)
   r2 = dataset.task.calculate_metrics(predictions)
   ```

3. **分析结果**
   - 可视化预测结果
   - 分析时序趋势
   - 识别关键节点

## ❓ 常见问题

**Q: 如何加载多个时间快照？**

A: 循环加载不同的 `time_idx`：
```python
for t in range(8):
    dataset = load_data('data/', time_idx=t)
    # 训练或评估
```

**Q: 数据集划分有什么区别？**

A: `temporal` 模拟真实预测场景（用过去预测未来），`random` 是标准的随机划分。

**Q: 如何选择回归标签？**

A: 通常使用 `label_idx=0`（变化率），因为它更稳定且对规模不敏感。

**Q: 转换后的数据在哪里？**

A: 使用 `bin/convert_dexposure.py` 转换后，数据保存在 `data/dexposure_graphland/` 目录。

## ✅ 成功！

DeXposure 数据已成功集成到 GraphPFN 中，可以开始使用了！

有任何问题请查看测试脚本或相关文档。
