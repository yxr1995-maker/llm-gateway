#!/bin/bash
# llm-gateway 启动脚本（供 launchd 调用）
# 假设仓库克隆在 ~/llm-gateway，且已创建 .venv
set -e
cd "$(dirname "$0")/../.."
exec .venv/bin/python -m app.main
