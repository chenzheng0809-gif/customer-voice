#!/usr/bin/env bash
# 一键跑通全流程。任何一步失败立即退出（set -e），避免"看起来跑完了"。
set -euo pipefail

cd "$(dirname "$0")"
ROOT="$(cd .. && pwd)"
PY="$ROOT/.venv/bin/python"

# HF 镜像 + 工作区缓存（国内访问 huggingface.co 会超时）
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-$ROOT/.hf_cache}"

MODE="${1:-full}"

echo "==> 1/4 生成数据"
PYTHONPATH=src "$PY" src/build_dataset.py --n 600 --seed 42

echo "==> 2/4 建库入库"
PYTHONPATH=src "$PY" src/build_db.py

echo "==> 3/4 训练与评测"
if [ "$MODE" = "fast" ]; then
  PYTHONPATH=src "$PY" src/train_eval.py --fast
else
  PYTHONPATH=src "$PY" src/train_eval.py
  PYTHONPATH=src "$PY" src/final_train.py --epochs 2
fi

echo "==> 4/4 OCR 演示"
PYTHONPATH=src "$PY" src/ocr_pipeline.py --demo | head -20

echo
echo "完成。启动看板："
echo "  PYTHONPATH=src $PY src/server.py   # http://127.0.0.1:8765/"