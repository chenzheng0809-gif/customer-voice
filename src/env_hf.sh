# Hugging Face 环境变量
# 用法：source src/env_hf.sh
#
# 国内访问 huggingface.co 会超时，这里默认走镜像。
# 如果你能直连 HF，把 HF_ENDPOINT 那一行注释掉即可。
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# 模型缓存放项目上一级，避免写系统默认的 ~/.cache/huggingface
# （该目录在部分环境下没有写权限，会导致加载模型报 PermissionError）
export HF_HOME="${HF_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/../.hf_cache}"