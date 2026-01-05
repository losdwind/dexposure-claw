# DeXposure 数据集说明文档

本文档详细说明 DeXposure 项目中各个数据文件的结构、用途和相互关系。

## 📁 目录结构

```
data/
├── historical-network_week_2025-07-01.json   # 主要时序网络数据 (2025年)
├── historical-network_week_2020-03-30.json   # 主要时序网络数据 (2020年)
├── meta_df.csv                               # 协议元数据
├── mapping/                                  # 映射文件目录
│   ├── id_to_info.json                       # ID 到协议信息映射
│   ├── rev_map.json                          # 反向映射
│   └── token_to_protocol.json                # Token 到协议映射
└── network_data/                             # 网络数据目录
    ├── filtered_edges_ftx.csv                # FTX 危机相关边数据
    ├── filtered_edges_terra.csv              # Terra 危机相关边数据
    ├── filtered_graph_data.csv               # 过滤后的完整图数据
    ├── filtered_nodes_ftx.csv                # FTX 危机相关节点数据
    └── filtered_nodes_terra.csv              # Terra 危机相关节点数据
```

## 📊 主要数据文件

### 1. historical-network_week_*.json

**文件说明**: 时序网络快照数据,记录多个时间点的 DeFi 协议网络状态

**数据结构**:
```json
{
  "data": {
    "YYYY-MM-DD": {                    // 时间点(周)
      "nodes": [                        // 节点列表
        {
          "id": "2269",                // 协议ID (字符串)
          "size": 1234567.89,          // 总资产规模 (USD)
          "composition": {             // 资产组成
            "BTC": 100000.0,
            "ETH": 50000.0,
            "USDT": 30000.0
          }
        }
      ],
      "links": [                        // 边列表(协议间关系)
        {
          "source": "2269",            // 源协议ID
          "target": "182",             // 目标协议ID (可能为null)
          "size": 50000.0,             // 关系强度(资金流)
          "composition": {             // 资产组成
            "ETH": 50000.0
          }
        }
      ]
    }
  }
}
```

**字段说明**:
- `id`: 协议唯一标识符,与 `meta_df.csv` 中的 ID 对应
- `size`: 协议总资产规模,单位 USD
- `composition`: 协议持有的各类加密资产及其数量
- `source/target`: 边的源节点和目标节点,target 可能为 null (表示外部资产)

**数据规模**:
- `historical-network_week_2025-07-01.json`:
  - 时间点数: 8 周
  - 节点数: ~10,550 个协议
  - 边数: ~73,506 条关系
  - 文件大小: 76MB

- `historical-network_week_2020-03-30.json`:
  - 文件大小: 1.1GB
  - 更早期的历史数据

### 2. meta_df.csv

**文件说明**: 协议元数据,提供协议的基本信息

**数据结构**:
```csv
id,name,category
2269,Binance CEX,CEX
182,Lido,Liquid Staking
1599,AAVE V3,Lending
```

**字段说明**:
- `id`: 协议唯一标识符
- `name`: 协议名称
- `category`: 协议类别

**协议类别**:
- `CEX`: 中心化交易所
- `Liquid Staking`: 流动性质押
- `Lending`: 借贷协议
- `Bridge`: 跨链桥
- `CDP`: 抵押债务位置
- `Restaking`: 再质押
- `DEX`: 去中心化交易所
- `Chain`: 区块链网络

## 🗺️ mapping 目录

### id_to_info.json

**文件说明**: 协议 ID 到详细信息的映射

**数据结构**:
```json
{
  "2269": {
    "name": "Binance CEX",
    "category": "Trading & Exchanges"
  },
  "182": {
    "name": "Lido",
    "category": "Asset Management"
  }
}
```

**用途**: 用于查询协议的名称和类别信息

### rev_map.json

**文件说明**: 反向映射表

**用途**: 用于协议 ID 的反向查询和转换

### token_to_protocol.json

**文件说明**: Token (加密货币) 到持有协议的映射

**数据结构**:
```json
{
  "ETH": {
    "182": {"id": "182", "frequency": 76972},      // Lido 持有 ETH
    "2269": {"id": "2269", "frequency": 123456}    // Binance 持有 ETH
  },
  "USDT": {
    "2269": {"id": "2269", "frequency": 54321}
  }
}
```

**字段说明**:
- `frequency`: 该协议持有该 token 的频率或数量

**用途**:
- 追踪特定资产在哪些协议中被持有
- 分析资产集中度和风险传播路径

## 🌐 network_data 目录

### filtered_graph_data.csv

**文件说明**: 过滤后的完整时序图数据

**数据结构**:
```csv
time,source,target,size
2018-04-23,Ethereum,TokenStore,25136.074390000023
2018-04-30,Ethereum,TokenStore,190389.93129999994
2018-05-07,TokenStore,Ethereum,111523.01886999997
```

**字段说明**:
- `time`: 时间戳
- `source`: 源协议名称
- `target`: 目标协议名称
- `size`: 资金流规模 (USD)

**数据规模**: 1,258,298 条记录

### filtered_edges_ftx.csv / filtered_nodes_ftx.csv

**文件说明**: FTX 危机相关的网络数据

**edges 数据结构**:
```csv
source,target,size
protocol_a,protocol_b,123456.78
```

**nodes 数据结构**:
```csv
time,id,size,type
2022-10-03,DogePup,4440.38,"Infrastructure, Services & Financial Products"
```

**用途**: 用于分析 FTX 崩盘事件对 DeFi 系统的影响

**数据规模**:
- 边: 67,001 条
- 节点: 16,184 个

### filtered_edges_terra.csv / filtered_nodes_terra.csv

**文件说明**: Terra/Luna 危机相关的网络数据

**结构与用途**: 同 FTX 数据,用于分析 Terra 危机传染

**数据规模**:
- 边: 60,765 条
- 节点: 11,406 个

## 🔗 数据关系图

```
                    ┌─────────────────────────┐
                    │  meta_df.csv            │
                    │  (协议元数据)            │
                    └───────────┬─────────────┘
                                │ ID 映射
                                ▼
    ┌──────────────────────────────────────────────┐
    │  historical-network_week_*.json              │
    │  (时序网络快照)                               │
    │  ┌──────────┐        ┌──────────┐           │
    │  │ nodes    │───────▶│ links    │           │
    │  │ (协议)    │        │ (关系)    │           │
    │  └──────────┘        └──────────┘           │
    └──────────────────────────────────────────────┘
                                │
                                │ 引用
                                ▼
    ┌──────────────────────────────────────────────┐
    │  mapping/                                     │
    │  ├─ id_to_info.json      (ID → 信息)         │
    │  ├─ rev_map.json         (反向映射)          │
    │  └─ token_to_protocol.json (Token → 协议)    │
    └──────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────┐
    │  network_data/                                │
    │  ├─ filtered_graph_data.csv  (完整时序图)    │
    │  ├─ filtered_edges_ftx.csv    (FTX危机)      │
    │  ├─ filtered_nodes_ftx.csv    (FTX危机)      │
    │  ├─ filtered_edges_terra.csv  (Terra危机)    │
    │  └─ filtered_nodes_terra.csv  (Terra危机)    │
    └──────────────────────────────────────────────┘
```

## 📈 数据用途示例

### 1. 时序图分析
使用 `historical-network_week_*.json` 构建:
- 协议间的关系网络图
- 时序演化分析
- 危机传播路径追踪

### 2. 协议特征构建
结合 `meta_df.csv` 和 `id_to_info.json`:
- 节点特征: 协议规模、资产组成、类别
- 边特征: 资金流强度、资产类型

### 3. 危机案例研究
使用 `network_data/filtered_*_ftx.csv` 和 `*_terra.csv`:
- FTX 崩盘传染分析
- Terra/Luna 脱锚影响评估
- 压力测试和风险建模

### 4. 资产追踪
使用 `token_to_protocol.json`:
- 特定资产(如 ETH, BTC)的持有人分布
- 资产集中度分析
- 连锁风险评估

## 🔧 数据处理建议

### 数据清洗
```python
# 处理空值和异常值
nodes = [n for n in snapshot['nodes'] if n['id'] is not None]
links = [l for l in snapshot['links'] if l['target'] is not None]

# 处理负值资产
composition = {k: max(0, v) for k, v in node['composition'].items()}
```

### 特征工程
```python
# 节点特征示例
features = [
    log(size),                      # 对数规模
    len(composition),               # 资产种类数
    diversity_index(composition),   # 资产多样性
    max_concentration(composition), # 最大单一资产占比
    category_one_hot(category)      # 类别独热编码
]
```

### 图构建
```python
import dgl

# 创建 DGL 图
g = dgl.graph((src_nodes, dst_nodes), num_nodes=len(nodes))
g.ndata['feat'] = node_features
g.ndata['size'] = node_sizes
g.edata['weight'] = edge_weights
```

## 📝 数据更新

- **频率**: 每周更新一次
- **时间范围**: 2018年 至 2025年7月
- **最新版本**: v1.0.0 (2025-07-01)

## ⚠️ 注意事项

1. **数据完整性**:
   - 部分历史数据可能缺失早期时间点
   - 某些协议的 `target` 字段可能为 `null`

2. **数据质量**:
   - 资产规模存在估计误差
   - 部分数据来自链上推断,可能有偏差

3. **使用建议**:
   - 建议先使用 `historical-network_week_2025-07-01.json` 进行开发测试
   - 完整训练时再使用 `historical-network_week_2020-03-30.json`

## 📚 参考资料

- 数据下载: [DATASET.md](./DATASET.md)
- 代码实现: [lib/graph/dexposure_temporal.py](./lib/graph/dexposure_temporal.py)
- 使用示例: [bin/dexposure_train_fixed.py](./bin/dexposure_train_fixed.py)

---

**最后更新**: 2025-01-05
**数据版本**: v1.0.0
