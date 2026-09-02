#!/usr/bin/env bash
# Native Slime SFT over every row in a prepared ShopSimulator turn dataset.
#
# 断点续训（resume）：
#   默认单 rollout（全部数据一遍，结束才存 checkpoint），中断后无档可续。
#   想要可恢复的 SFT，启动时加 SFT_ROLLOUT_CHUNK=<N>（例如 512）：数据被切成
#   DATA_ROWS/N 个 rollout，每个 rollout 结束都保存 Megatron checkpoint
#   （含数据位置）。中断后用同一命令 + RESUME=1 RUN_ROOT=<原目录> 重跑：
#     - Megatron 从 latest_checkpointed_iteration.txt 恢复权重与训练进度；
#     - 脚本按 iteration/chunk 自动推算已完成 rollout 数并传 --start-rollout-id，
#       slime 据此跳过已完成 rollout 并恢复数据 buffer 位置；
#   注意：当前保存配置为 --no-save-optim/--no-save-rng（Adam 动量与 shuffle
#   顺序不保存），恢复后优化器状态从零开始——权重级续训，非逐位精确复现。

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SLIME_DIR="${SLIME_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
SHOP_ROOT="${SHOP_ROOT:-$(cd "${SLIME_DIR}/.." && pwd)}"
BASE_DIR="${BASE_DIR:-${HOME}}"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-${HOME}/micromamba}"
MEGATRON_DIR="${MEGATRON_DIR:-${BASE_DIR}/Megatron-LM}"
SLIME_PYTHON="${SLIME_PYTHON:-${MAMBA_ROOT_PREFIX}/envs/slime/bin/python}"
SLIME_BIN="${SLIME_BIN:-$(dirname "${SLIME_PYTHON}")}"
RAY_BIN="${RAY_BIN:-${SLIME_BIN}/ray}"
source "${SLIME_DIR}/scripts/models/qwen3.5-0.8B.sh"

HF_CHECKPOINT="${HF_CHECKPOINT:-${BASE_DIR}/models/Qwen3.5-0.8B}"
REF_MODEL_PATH="${REF_MODEL_PATH:-${BASE_DIR}/models/Qwen3.5-0.8B_torch_dist}"
PROMPT_DATA="${FULL_DATA:-${SHOP_ROOT}/data/prepared/turn_examples.jsonl}"
NUM_GPUS="${NUM_GPUS:-1}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-12288}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"
NUM_DATA_PASSES="${NUM_DATA_PASSES:-1}"

for required in "${SLIME_PYTHON}" "${MEGATRON_DIR}" "${HF_CHECKPOINT}" "${REF_MODEL_PATH}" "${PROMPT_DATA}"; do
  [[ -e "${required}" ]] || { echo "Required input does not exist: ${required}" >&2; exit 2; }
done

DATA_ROWS="$(awk 'NF {count++} END {print count + 0}' "${PROMPT_DATA}")"
if (( DATA_ROWS <= 0 )); then
  echo "SFT data is empty: ${PROMPT_DATA}" >&2
  exit 2
fi
if (( GLOBAL_BATCH_SIZE <= 0 || DATA_ROWS % GLOBAL_BATCH_SIZE != 0 )); then
  echo "Data rows (${DATA_ROWS}) must be divisible by GLOBAL_BATCH_SIZE (${GLOBAL_BATCH_SIZE})" >&2
  exit 2
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
RUNS_ROOT="${RUNS_ROOT:-${BASE_DIR}/slime-runs}"
RUN_ROOT="${RUN_ROOT:-${RUNS_ROOT}/qwen35_2b_shop_sft_${STAMP}}"
RAY_TEMP_DIR="${RAY_TEMP_DIR:-${BASE_DIR}/ray/sft}"

RESUME="${RESUME:-0}"
SFT_ROLLOUT_CHUNK="${SFT_ROLLOUT_CHUNK:-0}"   # >0 时按 N 条样本切一个 rollout，产生中途 checkpoint

if [[ -e "${RUN_ROOT}" && "${RESUME}" != "1" ]]; then
  echo "Refusing to overwrite existing RUN_ROOT: ${RUN_ROOT}" >&2
  echo "续训用法：RESUME=1 RUN_ROOT=${RUN_ROOT}（加 SFT_ROLLOUT_CHUNK=512 才有中途存档可续）" >&2
  exit 2
fi

# ── rollout 切分与恢复点推算 ─────────────────────────────────────
ROLLOUT_BATCH_SIZE="${DATA_ROWS}"
NUM_ROLLOUT="${NUM_DATA_PASSES}"
START_ROLLOUT_ID_ARGS=()
if (( SFT_ROLLOUT_CHUNK > 0 )); then
  if (( SFT_ROLLOUT_CHUNK > DATA_ROWS || DATA_ROWS % SFT_ROLLOUT_CHUNK != 0 )); then
    echo "SFT_ROLLOUT_CHUNK (${SFT_ROLLOUT_CHUNK}) 必须能整除数据行数 (${DATA_ROWS})" >&2
    exit 2
  fi
  ROLLOUT_BATCH_SIZE="${SFT_ROLLOUT_CHUNK}"
  NUM_ROLLOUT="$(( NUM_DATA_PASSES * DATA_ROWS / SFT_ROLLOUT_CHUNK ))"
fi
if [[ "${RESUME}" == "1" ]]; then
  if [[ ! -f "${RUN_ROOT}/checkpoints/latest_checkpointed_iteration.txt" ]]; then
    echo "RESUME=1 但找不到 checkpoint：${RUN_ROOT}/checkpoints/latest_checkpointed_iteration.txt" >&2
    echo "（原训练若未用 SFT_ROLLOUT_CHUNK 切分，则没有中途存档，只能删除 RUN_ROOT 重跑）" >&2
    exit 2
  fi
  LAST_ITER="$(cat "${RUN_ROOT}/checkpoints/latest_checkpointed_iteration.txt")"
  if (( SFT_ROLLOUT_CHUNK > 0 )); then
    STEPS_PER_ROLLOUT=$(( SFT_ROLLOUT_CHUNK / GLOBAL_BATCH_SIZE ))
    COMPLETED_ROLLOUTS=$(( LAST_ITER / STEPS_PER_ROLLOUT ))
    START_ROLLOUT_ID_ARGS=(--start-rollout-id "${COMPLETED_ROLLOUTS}")
    echo "RESUME: 从 iteration ${LAST_ITER} 恢复 = 已完成 ${COMPLETED_ROLLOUTS}/${NUM_ROLLOUT} 个 rollout，从第 $((COMPLETED_ROLLOUTS + 1)) 个继续"
  else
    START_ROLLOUT_ID_ARGS=(--start-rollout-id 0)   # 单 rollout 模式：进度由 Megatron iteration 恢复
    echo "RESUME: 单 rollout 模式，从 iteration ${LAST_ITER} 恢复权重与数据进度"
  fi
fi
mkdir -p "${RUN_ROOT}/checkpoints" "${RUN_ROOT}/hf" "${RAY_TEMP_DIR}"

CKPT_ARGS=(
  --hf-checkpoint "${HF_CHECKPOINT}"
  --ref-load "${REF_MODEL_PATH}"
  --load "${RUN_ROOT}/checkpoints"
  --save "${RUN_ROOT}/checkpoints"
  --save-interval 1
  --save-hf "${RUN_ROOT}/hf/rollout_{rollout_id}"
  --no-save-optim
  --no-save-rng
)

SFT_ARGS=(
  --rollout-function-path slime.rollout.sft_rollout.generate_rollout
  --prompt-data "${PROMPT_DATA}"
  --input-key messages
  --metadata-key metadata
  --tool-key tools
  --rollout-shuffle
  --num-rollout "${NUM_ROLLOUT}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --loss-type sft_loss
  --loss-mask-type qwen3_5
  --calculate-per-token-loss
  --disable-compute-advantages-and-returns
  --debug-train-only
  "${START_ROLLOUT_ID_ARGS[@]}"
)

PERF_ARGS=(
  --tensor-model-parallel-size 1
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
  --log-probs-chunk-size "${LOG_PROBS_CHUNK_SIZE:-512}"
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr "${LR:-1e-5}"
  --lr-decay-style cosine
  --min-lr "${MIN_LR:-1e-6}"
  --lr-warmup-fraction "${LR_WARMUP_FRACTION:-0.05}"
  --weight-decay "${WEIGHT_DECAY:-0.1}"
  --adam-beta1 0.9
  --adam-beta2 0.98
  --clip-grad 1.0
  --use-distributed-optimizer
)

TRAIN_ARGS=(
  --actor-num-nodes 1
  --actor-num-gpus-per-node "${NUM_GPUS}"
  "${MODEL_ARGS[@]}"
  "${CKPT_ARGS[@]}"
  "${SFT_ARGS[@]}"
  "${PERF_ARGS[@]}"
  "${OPTIMIZER_ARGS[@]}"
  --seed "${SEED:-1234}"
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)

# B1 训练监控（wandb / tensorboard 备选）：SFT 无 rollout loop，不启用
# multi-turn / passrate 日志。命名规范同 run_rl.sh。
if [[ "${USE_WANDB}" == "1" ]]; then
  TRAIN_ARGS+=(
    --use-wandb
    --wandb-mode "${WANDB_MODE:-offline}"
    --wandb-dir "${WANDB_DIR:-${RUNS_ROOT}/wandb}"
    --wandb-project "${WANDB_PROJECT:-smartshop}"
    --wandb-group "sft"
    --wandb-run-name "${RUN_ROOT##*/}"
    --disable-wandb-random-suffix
    --wandb-always-use-train-step
  )
  [[ -n "${WANDB_KEY:-}" ]] && TRAIN_ARGS+=(--wandb-key "${WANDB_KEY}")
  [[ -n "${WANDB_HOST:-}" ]] && TRAIN_ARGS+=(--wandb-host "${WANDB_HOST}")
  [[ -n "${WANDB_TEAM:-}" ]] && TRAIN_ARGS+=(--wandb-team "${WANDB_TEAM}")
fi
if [[ "${USE_TENSORBOARD:-0}" == "1" ]]; then
  TRAIN_ARGS+=(
    --use-tensorboard
    --tb-project-name "${TENSORBOARD_DIR:-${RUNS_ROOT}/tensorboard}"
    --tb-experiment-name "sft"
  )
fi

{
  printf '%q ' "${SLIME_PYTHON}" -u train.py "${TRAIN_ARGS[@]}"
  printf '\n'
} >"${RUN_ROOT}/train_command.txt"

if [[ "${CHECK_ONLY:-0}" == "1" ]]; then
  printf 'PROMPT_DATA=%s\nDATA_ROWS=%s\nNUM_DATA_PASSES=%s\nRUN_ROOT=%s\nROLLOUT_BATCH_SIZE=%s\nNUM_ROLLOUT=%s\nRESUME=%s\n' \
    "${PROMPT_DATA}" "${DATA_ROWS}" "${NUM_DATA_PASSES}" "${RUN_ROOT}" "${ROLLOUT_BATCH_SIZE}" "${NUM_ROLLOUT}" "${RESUME}"
  exit 0
fi

command -v nvidia-smi >/dev/null || { echo "nvidia-smi is required" >&2; exit 2; }
[[ -x "${RAY_BIN}" ]] || { echo "ray is required: ${RAY_BIN}" >&2; exit 2; }
if "${RAY_BIN}" status >/dev/null 2>&1; then
  echo "An existing Ray cluster is running; stop it first." >&2
  exit 2
fi

export PYTHONUNBUFFERED=1
export CUDA_HOME="${CUDA_HOME:-${MAMBA_ROOT_PREFIX}/envs/slime}"
export PATH="${CUDA_HOME}/bin:${HOME}/.local/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-/usr/local/cuda/lib64}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost,${MASTER_ADDR}}"
export no_proxy="${no_proxy:-${NO_PROXY}}"
export PYTHONPATH="${MEGATRON_DIR}:${SLIME_DIR}:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
if [[ ! "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export OMP_NUM_THREADS=16
fi

RAY_STARTED=0
cleanup() {
  if (( RAY_STARTED == 1 )); then
    "${RAY_BIN}" stop --force >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

"${RAY_BIN}" start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" \
  --disable-usage-stats --dashboard-host=127.0.0.1 --dashboard-port=8265 \
  --temp-dir "${RAY_TEMP_DIR}"
RAY_STARTED=1

RUNTIME_ENV_JSON="$("${SLIME_PYTHON}" -c 'import json, os; keys=("PYTHONPATH","PATH","CUDA_HOME","LD_LIBRARY_PATH","MASTER_ADDR","NO_PROXY","no_proxy","CUDA_DEVICE_MAX_CONNECTIONS","PYTORCH_CUDA_ALLOC_CONF","OMP_NUM_THREADS","NVTE_DEBUG","NVTE_DEBUG_LEVEL","WANDB_MODE","WANDB_API_KEY","WANDB_BASE_URL","TENSORBOARD_DIR"); print(json.dumps({"env_vars": {key: os.environ[key] for key in keys if key in os.environ}}))')"

cd "${SLIME_DIR}"
"${RAY_BIN}" job submit --address=http://127.0.0.1:8265 \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- "${SLIME_PYTHON}" -u train.py "${TRAIN_ARGS[@]}" 2>&1 | tee "${RUN_ROOT}/train.log"

printf 'SFT complete. HF exports: %s\nMegatron checkpoints: %s\n' \
  "${RUN_ROOT}/hf" "${RUN_ROOT}/checkpoints"
