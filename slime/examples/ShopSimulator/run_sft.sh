#!/usr/bin/env bash
# Native Slime SFT over every row in a prepared ShopSimulator turn dataset.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SLIME_DIR="${SLIME_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
BASE_DIR="${BASE_DIR:-${HOME}}"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-${HOME}/micromamba}"
MEGATRON_DIR="${MEGATRON_DIR:-${BASE_DIR}/Megatron-LM}"
SLIME_PYTHON="${SLIME_PYTHON:-${MAMBA_ROOT_PREFIX}/envs/slime/bin/python}"
SLIME_BIN="${SLIME_BIN:-$(dirname "${SLIME_PYTHON}")}"
RAY_BIN="${RAY_BIN:-${SLIME_BIN}/ray}"
source "${SLIME_DIR}/scripts/models/qwen3.5-0.8B.sh"

HF_CHECKPOINT="${HF_CHECKPOINT:-${BASE_DIR}/models/Qwen3.5-0.8B}"
REF_MODEL_PATH="${REF_MODEL_PATH:-${BASE_DIR}/models/Qwen3.5-0.8B_torch_dist}"
PROMPT_DATA="${FULL_DATA:-${BASE_DIR}/slime-runs/shop_sft_512/prepared/turn_examples.jsonl}"
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
if [[ -e "${RUN_ROOT}" ]]; then
  echo "Refusing to overwrite existing RUN_ROOT: ${RUN_ROOT}" >&2
  exit 2
fi
mkdir -p "${RUN_ROOT}/checkpoints" "${RUN_ROOT}/hf" "${RAY_TEMP_DIR}"

CKPT_ARGS=(
  --hf-checkpoint "${HF_CHECKPOINT}"
  --ref-load "${REF_MODEL_PATH}"
  --load "${RUN_ROOT}/checkpoints"
  --save "${RUN_ROOT}/checkpoints"
  --save-interval "${NUM_DATA_PASSES}"
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
  --num-rollout "${NUM_DATA_PASSES}"
  --rollout-batch-size "${DATA_ROWS}"
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --loss-type sft_loss
  --loss-mask-type qwen3_5
  --calculate-per-token-loss
  --disable-compute-advantages-and-returns
  --debug-train-only
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

{
  printf '%q ' "${SLIME_PYTHON}" -u train.py "${TRAIN_ARGS[@]}"
  printf '\n'
} >"${RUN_ROOT}/train_command.txt"

if [[ "${CHECK_ONLY:-0}" == "1" ]]; then
  printf 'PROMPT_DATA=%s\nDATA_ROWS=%s\nNUM_DATA_PASSES=%s\nRUN_ROOT=%s\n' \
    "${PROMPT_DATA}" "${DATA_ROWS}" "${NUM_DATA_PASSES}" "${RUN_ROOT}"
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

RUNTIME_ENV_JSON="$("${SLIME_PYTHON}" -c 'import json, os; keys=("PYTHONPATH","PATH","CUDA_HOME","LD_LIBRARY_PATH","MASTER_ADDR","NO_PROXY","no_proxy","CUDA_DEVICE_MAX_CONNECTIONS","PYTORCH_CUDA_ALLOC_CONF","OMP_NUM_THREADS","NVTE_DEBUG","NVTE_DEBUG_LEVEL"); print(json.dumps({"env_vars": {key: os.environ[key] for key in keys if key in os.environ}}))')"

cd "${SLIME_DIR}"
"${RAY_BIN}" job submit --address=http://127.0.0.1:8265 \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- "${SLIME_PYTHON}" -u train.py "${TRAIN_ARGS[@]}" 2>&1 | tee "${RUN_ROOT}/train.log"

printf 'SFT complete. HF exports: %s\nMegatron checkpoints: %s\n' \
  "${RUN_ROOT}/hf" "${RUN_ROOT}/checkpoints"
