#!/bin/bash
# 串行链：等 REINFORCE++ 训练结束 → 评测 REINFORCE++ → 训练 CISPO(GPU1) → 评测 CISPO
#
# 说明：CISPO 沿用与其他算法相同的 eps_clip=0.2 / eps_clip_high=0.28（来自
# shop_rl.json），以做严格受控对照（唯一变量=算法）。slime 建议的 CISPO 规范
# 单边参数(1.0/4.0)留作后续超参消融，见 TODO E2。
#
# 路径约定：项目内路径从脚本自身位置推导，不硬编码；盘外资源（数据盘上的
# slime-runs/ 与 micromamba/）统一收敛到 BASE_DIR 单一变量。
#
# 用法: nohup ./run_chain_rpp_cispo.sh > chain_rpp_cispo.log 2>&1 &
#
# 可覆盖的环境变量:
#   SHOP_ROOT / SLIME_DIR / LOG_DIR / BASE_DIR / MAMBA_ROOT_PREFIX /
#   RAY_BIN / PY / SFT_ROOT / RPP_PID
set -u

# ── 路径推导（项目内相对化）──────────────────────────────────────
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SLIME_DIR="${SLIME_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"      # <项目根>/slime
SHOP_ROOT="${SHOP_ROOT:-$(cd "${SLIME_DIR}/.." && pwd)}"          # 项目根
LOG_DIR="${LOG_DIR:-${SHOP_ROOT}/result/rl}"

# ── 盘外资源（数据盘；换机器/换盘只需覆盖 BASE_DIR）──────────────
BASE_DIR="${BASE_DIR:-/hdd/kemove}"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-${BASE_DIR}/micromamba}"
RAY_BIN="${RAY_BIN:-${MAMBA_ROOT_PREFIX}/envs/slime/bin/ray}"
PY="${PY:-${MAMBA_ROOT_PREFIX}/envs/slime/bin/python}"

SFT_ROOT="${SFT_ROOT:-${BASE_DIR}/slime-runs/qwen35_2b_shop_sft_20260829_225338}"
RPP_PID="${RPP_PID:-2588500}"

say() { echo "[chain $(date '+%m-%d %H:%M:%S')] $*"; }

# 评测某个已训练好的算法（默认端口 + 默认 GPU，已验证可用）。
# 自动选取该 run 最新导出的 HF rollout 权重，避免硬编码 step 序号。
eval_run() {
  local NAME="$1"
  local HF_DIR="${BASE_DIR}/slime-runs/qwen35_2b_shop_rl_${NAME}/hf"
  local LAST_ID
  LAST_ID="$(ls "${HF_DIR}" 2>/dev/null | grep -oE '[0-9]+' | sort -n | tail -1)"
  if [[ -z "${LAST_ID}" ]]; then
    say "错误: ${HF_DIR} 下没有任何 rollout 导出，跳过评测 ${NAME}"
    return 1
  fi
  say "=== 评测 ${NAME} 开始 (HF: rollout_${LAST_ID}) ==="
  ( cd "${SLIME_DIR}" && env BASE_DIR="${BASE_DIR}" \
      MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX}" \
      SLIME_DIR="${SLIME_DIR}" \
      EVAL_CHECKPOINT="${BASE_DIR}/slime-runs/qwen35_2b_shop_rl_${NAME}/checkpoints" \
      HF_CHECKPOINT="${HF_DIR}/rollout_${LAST_ID}" \
      RUN_ROOT="${BASE_DIR}/slime-runs/eval_rl_${NAME}" \
      RAY_TEMP_DIR="${BASE_DIR}/ray/eval_${NAME}" \
      ./examples/ShopSimulator/run_eval.sh > "${LOG_DIR}/eval_rl_${NAME}.log" 2>&1 )
  say "=== 评测 ${NAME} 结束 (code=$?) ==="
  ( cd "${SLIME_DIR}" && "${PY}" -m examples.ShopSimulator.summarize_eval \
      --run-root "${BASE_DIR}/slime-runs/eval_rl_${NAME}" \
      >> "${LOG_DIR}/eval_rl_${NAME}.log" 2>&1 )
  say "=== 评测 ${NAME} summarize 完成 ==="
}

# 0) 等待 REINFORCE++ 训练结束
say "等待 REINFORCE++ 训练结束 (PID ${RPP_PID})..."
while kill -0 "${RPP_PID}" 2>/dev/null; do
  sleep 60
done
say "REINFORCE++ 训练已结束，清理 Ray 环境"
"${RAY_BIN}" stop --force >/dev/null 2>&1 || true
sleep 60

# 1) 评测 REINFORCE++
eval_run rpp
"${RAY_BIN}" stop --force >/dev/null 2>&1 || true
sleep 60

# 2) 训练 CISPO（GPU1；超参与其他算法一致，仅切换 estimator）
say "=== 训练 CISPO 开始 (GPU1) ==="
( cd "${SLIME_DIR}" && env BASE_DIR="${BASE_DIR}" \
    MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX}" \
    SLIME_DIR="${SLIME_DIR}" \
    HF_CHECKPOINT="${SFT_ROOT}/hf/rollout_0" \
    REF_MODEL_PATH="${SFT_ROOT}/checkpoints" \
    RUN_ROOT="${BASE_DIR}/slime-runs/qwen35_2b_shop_rl_cispo" \
    RAY_TEMP_DIR="${BASE_DIR}/ray/rl_cispo" \
    CUDA_VISIBLE_DEVICES=1 \
    ADVANTAGE_ESTIMATOR=cispo \
    ./examples/ShopSimulator/run_rl_ab.sh > "${LOG_DIR}/rl_cispo.log" 2>&1 )
say "=== 训练 CISPO 结束 (code=$?) ==="
"${RAY_BIN}" stop --force >/dev/null 2>&1 || true
sleep 60

# 3) 评测 CISPO
eval_run cispo
"${RAY_BIN}" stop --force >/dev/null 2>&1 || true
say "=== 全部完成：REINFORCE++ 与 CISPO 训练+评测均结束 ==="
