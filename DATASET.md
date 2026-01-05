# DeXposure 数据集下载指南

本项目的 DeXposure 数据集文件较大(约 1.2GB),不包含在 git 仓库中。使用前需要先下载数据集。

## 数据集文件

### 主要数据文件 (必需)
- `historical-network_week_2025-07-01.json` (~76MB) - 历史网络快照数据(2025年)
- `historical-network_week_2020-03-30.json` (~1.1GB) - 历史网络快照数据(2020年)
- `meta_df.csv` (~128KB) - 协议元数据

### mapping 目录 (可选)
- `mapping/id_to_info.json` (~457KB) - ID到信息的映射
- `mapping/rev_map.json` (~49KB) - 反向映射
- `mapping/token_to_protocol.json` (~2.5MB) - Token到协议的映射

### network_data 目录 (可选)
- `network_data/filtered_edges_ftx.csv` (~3.2MB) - FTX相关边数据
- `network_data/filtered_edges_terra.csv` (~2.9MB) - Terra相关边数据
- `network_data/filtered_graph_data.csv` (~59MB) - 过滤后的图数据
- `network_data/filtered_nodes_ftx.csv` (~990KB) - FTX相关节点数据
- `network_data/filtered_nodes_terra.csv` (~698KB) - Terra相关节点数据

**总计:** 约 1.2GB

## 自动下载

当你运行训练或预测脚本时,如果数据集文件不存在,程序会自动尝试下载:

```bash
# 训练脚本(会自动下载)
python bin/dexposure_train_fixed.py

# 微调脚本(会自动下载)
python bin/dexposure_finetune_graphpfn.py

# 预测脚本(会自动下载)
python bin/dexposure_predict.py
```

## 手动下载

如果自动下载失败,可以手动运行下载脚本:

```bash
# 下载数据集到当前目录
python bin/download_dataset.py

# 下載到指定目录
python bin/download_dataset.py --data-dir /path/to/data

# 强制重新下载(覆盖已存在的文件)
python bin/download_dataset.py --force

# 仅下载特定文件
python bin/download_dataset.py --files meta_df.csv
```

## 下载数据源

数据集文件从以下位置下载:

- GitHub Releases: `https://github.com/losdwind/graph-dexposure/releases/download/v1.0.0/`

## 手动上传数据集到 GitHub Releases

如果 GitHub Releases 中还没有数据集文件,需要手动上传:

1. 准备数据集文件:
   ```bash
   cd /home/figurich/inter-protocol-exposure/DeXposure/data
   ```

2. 在 GitHub 上创建 Release:
   - 访问: `https://github.com/losdwind/graph-dexposure/releases/new`
   - Tag: `v1.0.0`
   - Title: `DeXposure Dataset v1.0.0`

3. 上传文件:
   - `historical-network_week_2025-07-01.json`
   - `meta_df.csv`

4. 发布 Release

## 代码中使用

数据集会自动在以下位置查找:

```python
from lib.graph.dexposure_dataset import DeXposureRegressionDataset

# 数据集会自动从 data_path 加载
# 如果文件不存在,会自动提示下载
dataset = DeXposureRegressionDataset(
    root="/path/to/data",  # 数据目录
    split="train"
)
```

## 禁用自动下载

如果不想自动下载,可以在代码中设置 `auto_download=False`:

```python
from lib.graph.dexposure_temporal import DeXposureTemporalLoader

loader = DeXposureTemporalLoader(
    data_path="...",
    meta_path="...",
    auto_download=False  # 禁用自动下载
)
```

## 常见问题

### Q: 下载失败怎么办?

A: 可以尝试以下方法:
1. 检查网络连接
2. 检查 GitHub Releases 是否存在
3. 手动下载数据集文件并放到正确的目录
4. 使用 `--force` 参数强制重新下载

### Q: 如何更改数据集路径?

A: 在代码中修改 `root` 参数或 `data_path` 参数:

```python
dataset = DeXposureRegressionDataset(
    root="/your/custom/path",  # 自定义路径
    split="train"
)
```

### Q: 数据集文件太大,可以只下载部分吗?

A: 可以使用 `--files` 参数仅下载需要的文件:

```bash
# 仅下载元数据(快速)
python bin/download_dataset.py --files meta_df.csv
```

### Q: 如何验证下载的文件是否完整?

A: 下载脚本会自动检查文件大小。如果添加了 MD5 校验,也会进行 MD5 验证。
