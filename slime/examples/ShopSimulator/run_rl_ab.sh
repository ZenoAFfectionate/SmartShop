#!/usr/bin/env bash
# One-epoch online GRPO over rl_500, initialized from an explicitly supplied SFT checkpoint.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SLIME_DIR="${SLIME_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
BASE_DIR="${BASE_DIR:-${HOME}}"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-${HOME}/micromamba}"
MEGATRON_DIR="${MEGATRON_DIR:-${BASE_DIR}/Megatron-LM}"
SLIME_PYTHON="${SLIME_PYTHON:-${MAMBA_ROOT_PREFIX}/envs/slime/bin/python}"
SLIME_BIN="${SLIME_BIN:-$(dirname "${SLIME_PYTHON}")}"
RAY_BIN="${RAY_BIN:-${SLIME_BIN}/ray}"
PI_BIN="${PI_BIN:-$(command -v pi || true)}"
source "${SLIME_DIR}/scripts/models/qwen3.5-0.8B.sh"

RL_CONFIG="${RL_CONFIG:-${SCRIPT_DIR}/config/shop_rl.json}"
SHOP_ENV_URL="${SHOP_ENV_URL:-http://127.0.0.1:5000/api/shop_agent}"
: "${HF_CHECKPOINT:?set HF_CHECKPOINT to the SFT Hugging Face export}"
: "${REF_MODEL_PATH:?set REF_MODEL_PATH to the matching SFT Megatron checkpoint directory}"
[[ -f "${RL_CONFIG}" ]] || { echo "RL config does not exist: ${RL_CONFIG}" >&2; exit 2; }

eval "$("${SLIME_PYTHON}" - "${RL_CONFIG}" <<'PY'
import json
import shlex
import sys

config = json.load(open(sys.argv[1], encoding="utf-8"))
training = config["training"]
rollout = config["rollout"]
concurrency = config["concurrency"]
values = {
    "PROMPT_DATA": config["datasets"]["rl"]["path"],
    "NUM_EPOCH": training["num_epoch"],
    "ROLLOUT_BATCH_SIZE": training["rollout_batch_size"],
    "GLOBAL_BATCH_SIZE": training["global_batch_size"],
    "N_SAMPLES_PER_PROMPT": training["n_samples_per_prompt"],
    "NUM_STEPS_PER_ROLLOUT": training["num_steps_per_rollout"],
    "MICRO_BATCH_SIZE": training["micro_batch_size"],
    "LR": training["learning_rate"],
    "LR_DECAY_STYLE": training["lr_decay_style"],
    "KL_COEF": training["kl_loss_coefficient"],
    "KL_TYPE": training["kl_loss_type"],
    "ENTROPY_COEF": training["entropy_coefficient"],
    "EPS_CLIP": training["eps_clip"],
    "EPS_CLIP_HIGH": training["eps_clip_high"],
    "CLIP_GRAD": training["gradient_clip"],
    "SEED": training["seed"],
    "MAX_CONTEXT_LEN": rollout["max_context_len"],
    "MAX_RESPONSE_LEN": rollout["max_response_len"],
    "MAX_MODEL_TURNS": rollout["max_model_turns"],
    "ROLLOUT_SEED": rollout["seed"],
    "TEMPERATURE": rollout["temperature"],
    "TOP_P": rollout["top_p"],
    "TOP_K": rollout["top_k"],
    "KEEP_ACT_RESULTS": rollout["context_keep_shop_act_results"],
    "ROLLOUT_TIMEOUT_SEC": rollout["wall_clock_timeout_seconds"],
    "GENERATE_PATH": rollout["custom_generate_function_path"],
    "REWARD_PATH": rollout["reward_post_process_path"],
    "FILTER_PATH": rollout["sample_filter_path"],
    "SHOP_ENV_CAPACITY": concurrency["shop_env_capacity"],
    "ROLLOUT_NUM_ENGINES": concurrency["rollout_num_engines"],
}
for key, value in values.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
)"

if [[ "${PROMPT_DATA}" != /* ]]; then
  PROMPT_DATA="${SLIME_DIR}/${PROMPT_DATA}"
fi
[[ -n "${PI_BIN}" && -x "${PI_BIN}" ]] || { echo "pi is required; set PI_BIN to its executable" >&2; exit 2; }
for required in "${SLIME_PYTHON}" "${MEGATRON_DIR}" "${HF_CHECKPOINT}" "${REF_MODEL_PATH}" "${PROMPT_DATA}"; do
  [[ -e "${required}" ]] || { echo "Required input does not exist: ${required}" >&2; exit 2; }
done

DATA_ROWS="$(awk 'NF {count++} END {print count + 0}' "${PROMPT_DATA}")"
if (( DATA_ROWS <= 0 || DATA_ROWS % ROLLOUT_BATCH_SIZE != 0 )); then
  echo "RL rows (${DATA_ROWS}) must be positive and divisible by rollout_batch_size (${ROLLOUT_BATCH_SIZE})" >&2
  exit 2
fi
if (( GLOBAL_BATCH_SIZE != ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT / NUM_STEPS_PER_ROLLOUT )); then
  echo "global_batch_size does not match Slime rollout arithmetic" >&2
  exit 2
fi
if (( SHOP_ENV_CAPACITY % ROLLOUT_NUM_ENGINES != 0 )); then
  echo "ShopSimulator capacity must divide evenly across rollout engines" >&2
  exit 2
fi
NUM_ROLLOUTS=$((DATA_ROWS / ROLLOUT_BATCH_SIZE))
TOTAL_ROLLOUTS=$((NUM_ROLLOUTS * NUM_EPOCH))
SGLANG_SERVER_CONCURRENCY=$((SHOP_ENV_CAPACITY / ROLLOUT_NUM_ENGINES))
# C2: checkpoint every N rollouts (25 by default) so dev-based model selection
# has candidates to choose from; the final rollout always saves too.
SAVE_INTERVAL="${SAVE_INTERVAL:-25}"
if (( SAVE_INTERVAL <= 0 )); then
  echo "SAVE_INTERVAL must be positive" >&2
  exit 2
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
RUNS_ROOT="${RUNS_ROOT:-${BASE_DIR}/slime-runs}"
RUN_ROOT="${RUN_ROOT:-${RUNS_ROOT}/qwen35_2b_shop_rl_${STAMP}}"
RAY_TEMP_DIR="${RAY_TEMP_DIR:-${BASE_DIR}/ray/rl}"
# 多实验并行隔离：每个实验用独立的 Ray 端口与 adapter 端口
RAY_GCS_PORT="${RAY_GCS_PORT:-6379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
RAY_ADDRESS="http://127.0.0.1:${RAY_DASHBOARD_PORT}"
# GPU 隔离：限定本实验可见的 GPU（如 CUDA_VISIBLE_DEVICES=1）
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
ADAPTER_PUBLIC_HOST="${ADAPTER_PUBLIC_HOST:-127.0.0.1}"
ADAPTER_BIND_HOST="${ADAPTER_BIND_HOST:-0.0.0.0}"
ADAPTER_PORT="${ADAPTER_PORT:-18080}"
# 算法切换：grpo / gspo / cispo / reinforce_plus_plus[_baseline] / ppo
ADVANTAGE_ESTIMATOR="${ADVANTAGE_ESTIMATOR:-grpo}"
# Dr. GRPO：置 1 则去掉 GRPO 的 std 归一化（消除难度偏差）
DISABLE_GRPO_STD_NORMALIZATION="${DISABLE_GRPO_STD_NORMALIZATION:-0}"
if [[ -e "${RUN_ROOT}" ]]; then
  echo "Refusing to overwrite existing RUN_ROOT: ${RUN_ROOT}" >&2
  exit 2
fi
mkdir -p "${RUN_ROOT}/checkpoints" "${RUN_ROOT}/hf" "${RUN_ROOT}/rollout_dumps" "${RAY_TEMP_DIR}"

TRAIN_ARGS=(
  --actor-num-nodes 1
  --actor-num-gpus-per-node 1
  "${MODEL_ARGS[@]}"
  --hf-checkpoint "${HF_CHECKPOINT}"
  --ref-load "${REF_MODEL_PATH}"
  --load "${RUN_ROOT}/checkpoints"
  --save "${RUN_ROOT}/checkpoints"
  --save-interval "${SAVE_INTERVAL}"
  --save-hf "${RUN_ROOT}/hf/rollout_{rollout_id}"
  --no-save-optim
  --no-save-rng
  --custom-generate-function-path "${GENERATE_PATH}"
  --custom-reward-post-process-path "${REWARD_PATH}"
  --rollout-sample-filter-path "${FILTER_PATH}"
  --prompt-data "${PROMPT_DATA}"
  --input-key prompt
  --label-key label
  --metadata-key metadata
  --num-epoch "${NUM_EPOCH}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
  --rollout-max-context-len "${MAX_CONTEXT_LEN}"
  --rollout-max-response-len "${MAX_RESPONSE_LEN}"
  --rollout-temperature "${TEMPERATURE}"
  --rollout-top-p "${TOP_P}"
  --rollout-top-k "${TOP_K}"
  --rollout-seed "${ROLLOUT_SEED}"
  --num-steps-per-rollout "${NUM_STEPS_PER_ROLLOUT}"
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --micro-batch-size "${MICRO_BATCH_SIZE}"
  --balance-data
  --save-debug-rollout-data "${RUN_ROOT}/rollout_dumps/rollout_{rollout_id}.pt"
  --loss-type policy_loss
  --advantage-estimator "${ADVANTAGE_ESTIMATOR}"
  --use-kl-loss
  --kl-loss-coef "${KL_COEF}"
  --kl-loss-type "${KL_TYPE}"
  --entropy-coef "${ENTROPY_COEF}"
  --eps-clip "${EPS_CLIP}"
  --eps-clip-high "${EPS_CLIP_HIGH}"
  --optimizer adam
  --lr "${LR}"
  --lr-decay-style "${LR_DECAY_STYLE}"
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
  --clip-grad "${CLIP_GRAD}"
  --use-distributed-optimizer
  --tensor-model-parallel-size 1
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-12288}"
  --rollout-num-gpus 1
  --rollout-num-gpus-per-engine 1
  --sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY}"
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.55}"
  --sglang-tool-call-parser qwen3_coder
  --sglang-reasoning-parser qwen3
  --sglang-enable-deterministic-inference
  --seed "${SEED}"
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
  --loss-mask-type qwen3_5
  --colocate
)

# 可选：Dr. GRPO —— 去掉 advantage 的 std 归一化
if [[ "${DISABLE_GRPO_STD_NORMALIZATION}" == "1" ]]; then
  TRAIN_ARGS+=(--disable-grpo-std-normalization)
fi

# REINFORCE++ 变体要求开启 advantage 归一化（slime 强制校验，缺失会 assert 失败）
case "${ADVANTAGE_ESTIMATOR}" in
  reinforce_plus_plus|reinforce_plus_plus_baseline)
    TRAIN_ARGS+=(--normalize-advantages)
    ;;
esac

{
  printf '%q ' "${SLIME_PYTHON}" -u train.py "${TRAIN_ARGS[@]}"
  printf '\n'
} >"${RUN_ROOT}/train_command.txt"

if [[ "${CHECK_ONLY:-0}" == "1" ]]; then
  printf 'PROMPT_DATA=%s\nDATA_ROWS=%s\nNUM_ROLLOUTS=%s\nRUN_ROOT=%s\n' \
    "${PROMPT_DATA}" "${DATA_ROWS}" "${TOTAL_ROLLOUTS}" "${RUN_ROOT}"
  exit 0
fi

command -v nvidia-smi >/dev/null || { echo "nvidia-smi is required" >&2; exit 2; }
[[ -x "${RAY_BIN}" ]] || { echo "ray is required: ${RAY_BIN}" >&2; exit 2; }
# 只检查本实验自己端口上的 Ray 集群，避免误判其他实验的集群
if RAY_ADDRESS="${RAY_ADDRESS}" "${RAY_BIN}" status >/dev/null 2>&1; then
  echo "An existing Ray cluster is running on ${RAY_ADDRESS}; stop it first." >&2
  exit 2
fi

export PYTHONUNBUFFERED=1
export PI_BIN
export CUDA_HOME="${CUDA_HOME:-${MAMBA_ROOT_PREFIX}/envs/slime}"
export PATH="${CUDA_HOME}/bin:${HOME}/.local/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-/usr/local/cuda/lib64}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export SHOP_ENV_URL SHOP_MAX_TURNS="${MAX_MODEL_TURNS}"
export SHOP_CONTEXT_KEEP_ACT_RESULTS="${KEEP_ACT_RESULTS}"
export SHOP_ROLLOUT_TIMEOUT_SEC="${ROLLOUT_TIMEOUT_SEC}"
export SHOP_REQUIRE_NONZERO_VARIANCE_PER_ROLLOUT=0
export ADAPTER_PUBLIC_HOST ADAPTER_BIND_HOST ADAPTER_PORT
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost,${MASTER_ADDR},${ADAPTER_PUBLIC_HOST}}"
export no_proxy="${no_proxy:-${NO_PROXY}}"
export PYTHONPATH="${MEGATRON_DIR}:${SLIME_DIR}:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
if [[ ! "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export OMP_NUM_THREADS=16
fi

RAY_STARTED=0
cleanup() {
  # ray stop --force 是全局命令，会杀死机器上所有 Ray 集群（包括其他实验的）。
  # 并行模式下默认不自动 stop，改为提示手动清理，避免误伤其他实验。
  if (( RAY_STARTED == 1 )); then
    if [[ "${RAY_KEEP_CLUSTER:-0}" == "1" ]]; then
      echo "实验退出。Ray 集群保留（RAY_KEEP_CLUSTER=1），如需停止请手动执行：" >&2
      echo "  RAY_ADDRESS=${RAY_ADDRESS} ${RAY_BIN} stop --force" >&2
    else
      echo "警告: ray stop 将停止本机所有 Ray 集群！" >&2
      "${RAY_BIN}" stop --force >/dev/null 2>&1 || true
    fi
  fi
}
trap cleanup EXIT INT TERM

"${RAY_BIN}" start --head --node-ip-address "${MASTER_ADDR}" --num-gpus 1 \
  --port "${RAY_GCS_PORT}" --disable-usage-stats \
  --dashboard-host=127.0.0.1 --dashboard-port="${RAY_DASHBOARD_PORT}" \
  --temp-dir "${RAY_TEMP_DIR}"
RAY_STARTED=1
if [[ -n "${CUDA_VISIBLE_DEVICES}" ]]; then
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (本实验可见 GPU)"
fi

RUNTIME_ENV_JSON="$("${SLIME_PYTHON}" -c 'import json, os; keys=("PYTHONPATH","PATH","CUDA_HOME","LD_LIBRARY_PATH","MASTER_ADDR","NO_PROXY","no_proxy","CUDA_VISIBLE_DEVICES","CUDA_DEVICE_MAX_CONNECTIONS","PYTORCH_CUDA_ALLOC_CONF","OMP_NUM_THREADS","SHOP_ENV_URL","SHOP_MAX_TURNS","SHOP_CONTEXT_KEEP_ACT_RESULTS","SHOP_ROLLOUT_TIMEOUT_SEC","SHOP_REQUIRE_NONZERO_VARIANCE_PER_ROLLOUT","ADAPTER_PUBLIC_HOST","ADAPTER_BIND_HOST","ADAPTER_PORT","PI_BIN"); print(json.dumps({"env_vars": {key: os.environ[key] for key in keys if key in os.environ}}))')"

cd "${SLIME_DIR}"

# 等待 Ray dashboard agent 就绪（ray start 后 agent 需要数秒启动，
# 过早 submit 会报 "No available agent to submit job" 500 错误）
AGENT_READY=0
for i in $(seq 1 36); do
  if "${RAY_BIN}" job list --address="${RAY_ADDRESS}" >/dev/null 2>&1; then
    echo "Ray job agent 就绪（等待了 $((i*5)) 秒）"
    AGENT_READY=1
    break
  fi
  echo "等待 Ray job agent 就绪... ($((i*5))s)"
  sleep 5
done
if (( AGENT_READY != 1 )); then
  echo "错误: Ray job agent 180 秒内未就绪，放弃 submit" >&2
  exit 3
fi

"${RAY_BIN}" job submit --address="${RAY_ADDRESS}" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- "${SLIME_PYTHON}" -u train.py "${TRAIN_ARGS[@]}" 2>&1 | tee "${RUN_ROOT}/train.log"

printf 'RL complete. HF exports: %s\nMegatron checkpoints: %s\n' \
  "${RUN_ROOT}/hf" "${RUN_ROOT}/checkpoints"
