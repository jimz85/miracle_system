#!/bin/bash
# Miracle-Kronos 生产入口脚本
# 用法: ./run_miracle_kronos.sh [audit|live]
# audit模式: 不真实下单，只扫描
# live模式: 真实交易（OKX_FLAG=0时）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 加载环境变量
if [ -f "$HOME/.hermes/.env" ]; then
    source "$HOME/.hermes/.env"
fi

MODE="${1:-audit}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Miracle-Kronos starting in ${MODE} mode"

if [ "$MODE" = "live" ]; then
    if [ "${OKX_FLAG:-1}" = "0" ]; then
        echo "⚠️  LIVE TRADING MODE - Real money at risk!"
        python3 miracle_kronos.py --mode live
    else
        echo "⚠️  OKX_FLAG=1 (simulation), forcing audit mode"
        python3 miracle_kronos.py --mode audit
    fi
else
    python3 miracle_kronos.py --mode audit
fi
