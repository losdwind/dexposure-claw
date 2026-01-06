#!/bin/bash
# GPU服务器上运行DeXposure实验的脚本

set -e

echo "========================================"
echo "DeXposure GPU服务器运行脚本"
echo "========================================"

# 1. 检查并安装依赖
echo "[1/6] 检查Python依赖..."

pip3 show torch > /dev/null 2>&1 || {
    echo "  正在安装PyTorch (CUDA 12.1)..."
    pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121
}

pip3 show dgl > /dev/null 2>&1 || {
    echo "  正在安装DGL..."
    pip3 install --no-cache-dir dgl -f https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html
}

for pkg in ijson scikit-learn pandas matplotlib seaborn ipywidgets; do
    pip3 show $pkg > /dev/null 2>&1 || {
        echo "  正在安装 $pkg..."
        pip3 install --no-cache-dir $pkg
    }
done

echo "  ✓ 所有依赖已安装"

# 2. 检查代码更新
echo "[2/6] 检查代码更新..."
cd /root/graph-dexposure
git pull origin main
echo "  ✓ 代码已是最新"

# 3. 检查数据文件
echo "[3/6] 检查数据文件..."
if [ ! -f "data/historical-network_week_2025-07-01.json" ]; then
    echo "  ⚠️  警告: 数据文件不存在，请先上传数据文件"
    echo "     期望路径: data/historical-network_week_2025-07-01.json"
    echo "     期望路径: data/meta_df.csv"
else
    echo "  ✓ 数据文件存在"
fi

# 4. 检查checkpoint
echo "[4/6] 检查模型checkpoint..."
if [ ! -f "checkpoints/graphpfn-v1.ckpt" ]; then
    echo "  ⚠️  警告: GraphPFN checkpoint不存在"
    echo "     期望路径: checkpoints/graphpfn-v1.ckpt"
else
    echo "  ✓ Checkpoint存在 ($(du -h checkpoints/graphpfn-v1.ckpt | cut -f1))"
fi

# 5. 创建输出目录
echo "[5/6] 创建输出目录..."
mkdir -p output/dexposure_graphpfn_link
echo "  ✓ 输出目录已创建"

# 6. 启动Jupyter Lab
echo "[6/6] 启动Jupyter Lab..."
echo "  ========================================"
echo "  Jupyter Lab 将在后台启动"
echo "  访问地址: http://localhost:8889"
echo "  ========================================"

# 停止已存在的jupyter进程
pkill -f "jupyter-lab" || true
sleep 2

# 启动Jupyter Lab
nohup jupyter-lab \
    --ip=0.0.0.0 \
    --port=8889 \
    --no-browser \
    --allow-root \
    --NotebookApp.token='' \
    --NotebookApp.password='' \
    --NotebookApp.allow_origin='*' \
    --NotebookApp.disable_check_xsrf=True \
    > /root/jupyter.log 2>&1 &

echo "  ✓ Jupyter Lab 已启动"
echo ""
echo "等待服务启动..."
sleep 5

# 检查服务状态
if pgrep -f "jupyter-lab" > /dev/null; then
    echo "  ✓ Jupyter Lab 运行中"
    echo ""
    echo "========================================"
    echo "访问方式:"
    echo "========================================"
    echo "方法1: SSH隧道"
    echo "  在本地运行: ssh -L 8889:localhost:8889 gpu-server"
    echo "  然后访问: http://localhost:8889"
    echo ""
    echo "方法2: 直接访问"
    echo "  浏览器访问: http://$(hostname -I | awk '{print $1}'):8889"
    echo ""
    echo "日志查看: tail -f /root/jupyter.log"
    echo "========================================"
else
    echo "  ✗ 启动失败，请检查日志: /root/jupyter.log"
    exit 1
fi
