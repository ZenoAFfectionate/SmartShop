#!/bin/bash
# 串行算法对照实验编排器：等 GPU0 的 GRPO 基线跑完 → Dr.GRPO → GSPO
# 用法: nohup ./run_serial_ab.sh > serial_orchestration.log 2>&1 &
#
# 依赖:
#   - GPU0 GRPO 基线正在从断点续训 (nohup 父进程 PID 由 GRPO_PID 指定)
#   - run_rl_ab.sh 已含: Ray agent 就绪等待 / 定向端口 / GPU 隔离 / 算法切换
#
# 路径约定：项目内路径从脚本自身位置推导，不硬编码；盘外资源（数据盘上的
# slime-runs/ 与 micromamba/）统一收敛到 BASE_DIR 单一变量。
#
# 可覆盖的环境变量:
#   SHOP_ROOT / SLIME_DIR / LOG_DIR / BASE_DIR / MAMBA_ROOT_PREFIX /
#   SFT_ROOT / GRPO_RUN / SFT_TURN_DATA / GRPO_PID
set -u

# ── 路径推导（项目内相对化）──────────────────────────────────────
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SLIME_DIR="${SLIME_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"      # <项目根>/slime
SHOP_ROOT="${SHOP_ROOT:-$(cd "${SLIME_DIR}/.." && pwd)}"          # 项目根
LOG_DIR="${LOG_DIR:-${SHOP_ROOT}/result/rl}"

# ── 盘外资源（数据盘；换机器/换盘只需覆盖 BASE_DIR）──────────────
BASE_DIR="${BASE_DIR:-/hdd/kemove}"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-${BASE_DIR}/micromamba}"

SFT_ROOT="${SFT_ROOT:-${BASE_DIR}/slime-runs/qwen35_2b_shop_sft_20260829_225338}"
GRPO_RUN="${GRPO_RUN:-${BASE_DIR}/slime-runs/qwen35_2b_shop_rl_20260830_195438}"
SFT_TURN_DATA="${SFT_TURN_DATA:-${SHOP_ROOT}/data/prepared/turn_examples.jsonl}"
GRPO_PID="${GRPO_PID:-1810107}"

say() { echo "[编排 $(date '+%m-%d %H:%M:%S')] $*"; }

# ─── 阶段 0: 等待 GPU0 GRPO 基线完成 ─────────────────────────────
say "等待 GPU0 GRPO 基线完成 (PID ${GRPO_PID}, 断点续训自 rollout_49)..."
while kill -0 "${GRPO_PID}" 2>/dev/null; do
  sleep 120
done
say "GRPO 基线进程已退出"

# 等收尾写盘, 并验证完成标志 (第100轮保存点 hf/rollout_99)
sleep 60
if [ -d "${GRPO_RUN}/hf/rollout_99" ]; then
  say "GRPO 基线完整完成 ✓ (hf/rollout_99 存在)"
else
  say "警告: ${GRPO_RUN}/hf/rollout_99 不存在, GRPO 可能未完整结束"
  say "train.log 尾部: $(tail -3 "${GRPO_RUN}/train.log" 2>/dev/null | tr '\n' ' ')"
  say "继续执行对照实验 (基线已有 checkpoint, 不受影响)"
fi

# 等 Ray/GPU 资源完全释放
sleep 120

cd "${SLIME_DIR}"

# ─── 阶段 1: Dr. GRPO (GPU1, grpo + 去 std 归一化) ────────────────
say "启动 Dr.GRPO (GPU1)..."
BASE_DIR="${BASE_DIR}" \
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX}" \
SLIME_DIR="${SLIME_DIR}" \
HF_CHECKPOINT="${SFT_ROOT}/hf/rollout_0" \
REF_MODEL_PATH="${SFT_ROOT}/checkpoints" \
RUN_ROOT="${BASE_DIR}/slime-runs/qwen35_2b_shop_rl_drgrpo" \
RAY_TEMP_DIR="${BASE_DIR}/ray/rl_drgrpo" \
RAY_GCS_PORT=6380 RAY_DASHBOARD_PORT=8266 ADAPTER_PORT=18081 \
CUDA_VISIBLE_DEVICES=1 \
ADVANTAGE_ESTIMATOR=grpo DISABLE_GRPO_STD_NORMALIZATION=1 \
SFT_TURN_DATA="${SFT_TURN_DATA}" \
./examples/ShopSimulator/run_rl_ab.sh > "${LOG_DIR}/rl_drgrpo.log" 2>&1
say "Dr.GRPO 退出 (code=$?)"

sleep 180  # 等资源释放

# ─── 阶段 2: GSPO (GPU2, 优势序列级估计器) ────────────────────────
say "启动 GSPO (GPU2)..."
BASE_DIR="${BASE_DIR}" \
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX}" \
SLIME_DIR="${SLIME_DIR}" \
HF_CHECKPOINT="${SFT_ROOT}/hf/rollout_0" \
REF_MODEL_PATH="${SFT_ROOT}/checkpoints" \
RUN_ROOT="${BASE_DIR}/slime-runs/qwen35_2b_shop_rl_gspo" \
RAY_TEMP_DIR="${BASE_DIR}/ray/rl_gspo" \
RAY_GCS_PORT=6381 RAY_DASHBOARD_PORT=8267 ADAPTER_PORT=18082 \
CUDA_VISIBLE_DEVICES=2 \
ADVANTAGE_ESTIMATOR=gspo \
SFT_TURN_DATA="${SFT_TURN_DATA}" \
./examples/ShopSimulator/run_rl_ab.sh > "${LOG_DIR}/rl_gspo.log" 2>&1
say "GSPO 退出 (code=$?)"

say "全部串行实验完成 (GRPO 已在 ${GRPO_RUN}, Dr.GRPO → ${LOG_DIR}/rl_drgrpo.log, GSPO → ${LOG_DIR}/rl_gspo.log)"
