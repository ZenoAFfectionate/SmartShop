#!/usr/bin/env bash
# Slime eval-only run on official_test_200, one rollout per task by default.

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

EVAL_CONFIG="${EVAL_CONFIG:-${SCRIPT_DIR}/config/shop_eval_official_k1.yaml}"
SHOP_ENV_URL="${SHOP_ENV_URL:-http://127.0.0.1:5000/api/shop_agent}"
SHOP_ENV_CAPACITY="${SHOP_ENV_CAPACITY:-20}"
MAX_MODEL_TURNS="${MAX_MODEL_TURNS:-40}"
MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-16384}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-2048}"
ROLLOUT_SEED="${ROLLOUT_SEED:-42}"
SEED="${SEED:-1234}"
: "${EVAL_CHECKPOINT:?set EVAL_CHECKPOINT to a Hugging Face or Megatron checkpoint directory}"
HF_CHECKPOINT="${HF_CHECKPOINT:-${EVAL_CHECKPOINT}}"
# Overridable so non-default EVAL_CONFIGs can point at their own dataset
# (e.g. shop_dev.yaml -> dev.jsonl for full development-set evaluation).
PROMPT_DATA="${PROMPT_DATA:-${SCRIPT_DIR}/data/tasks_v2/official_test_200.jsonl}"

[[ -n "${PI_BIN}" && -x "${PI_BIN}" ]] || { echo "pi is required; set PI_BIN to its executable" >&2; exit 2; }
for required in "${SLIME_PYTHON}" "${MEGATRON_DIR}" "${EVAL_CHECKPOINT}" "${HF_CHECKPOINT}" "${EVAL_CONFIG}" "${PROMPT_DATA}"; do
  [[ -e "${required}" ]] || { echo "Required input does not exist: ${required}" >&2; exit 2; }
done

STAMP="$(date +%Y%m%d_%H%M%S)"
RUNS_ROOT="${RUNS_ROOT:-${BASE_DIR}/slime-runs}"
RUN_ROOT="${RUN_ROOT:-${RUNS_ROOT}/qwen35_2b_shop_eval_${STAMP}}"
RAY_TEMP_DIR="${RAY_TEMP_DIR:-${BASE_DIR}/ray/eval}"
ADAPTER_PUBLIC_HOST="${ADAPTER_PUBLIC_HOST:-127.0.0.1}"
ADAPTER_BIND_HOST="${ADAPTER_BIND_HOST:-0.0.0.0}"
ADAPTER_PORT="${ADAPTER_PORT:-18080}"
# Overridable so a second eval can run alongside an existing Ray cluster
# (e.g. while an RL run holds the default 6379/8265 ports).
RAY_GCS_PORT="${RAY_GCS_PORT:-6379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
RAY_ADDRESS="http://127.0.0.1:${RAY_DASHBOARD_PORT}"
if [[ -e "${RUN_ROOT}" ]]; then
  echo "Refusing to overwrite existing RUN_ROOT: ${RUN_ROOT}" >&2
  exit 2
fi
mkdir -p "${RUN_ROOT}/rollout_dumps" "${RAY_TEMP_DIR}"

TRAIN_ARGS=(
  --actor-num-nodes 1
  --actor-num-gpus-per-node 1
  "${MODEL_ARGS[@]}"
  --hf-checkpoint "${HF_CHECKPOINT}"
  --ref-load "${EVAL_CHECKPOINT}"
  --load "${EVAL_CHECKPOINT}"
  --num-rollout 0
  --eval-interval 1
  --lr-decay-iters 1
  --eval-config "${EVAL_CONFIG}"
  --save-debug-rollout-data "${RUN_ROOT}/rollout_dumps/rollout_{rollout_id}.pt"
  --prompt-data "${PROMPT_DATA}"
  --input-key prompt
  --label-key label
  --metadata-key metadata
  --rollout-batch-size 1
  --n-samples-per-prompt 1
  --global-batch-size 1
  --micro-batch-size 1
  --rollout-max-context-len "${MAX_CONTEXT_LEN}"
  --rollout-max-response-len "${MAX_RESPONSE_LEN}"
  --rollout-seed "${ROLLOUT_SEED}"
  --custom-generate-function-path examples.ShopSimulator.generate.generate
  --tensor-model-parallel-size 1
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1
  --rollout-num-gpus 1
  --rollout-num-gpus-per-engine 1
  --sglang-server-concurrency "${SHOP_ENV_CAPACITY}"
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.55}"
  --sglang-tool-call-parser qwen3_coder
  --sglang-reasoning-parser qwen3
  --sglang-enable-deterministic-inference
  --seed "${SEED}"
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --attention-softmax-in-fp32
  --attention-backend flash
  --loss-mask-type qwen3_5
  --colocate
)

# B1 评测留档（可选）：单 step job，wandb/tensorboard 仅作结果存档与曲线。
# 命名规范同 run_rl.sh。
if [[ "${USE_WANDB:-0}" == "1" ]]; then
  TRAIN_ARGS+=(
    --use-wandb
    --wandb-mode "${WANDB_MODE:-offline}"
    --wandb-dir "${WANDB_DIR:-${RUNS_ROOT}/wandb}"
    --wandb-project "${WANDB_PROJECT:-smartshop}"
    --wandb-group "eval"
    --wandb-run-name "${RUN_ROOT##*/}"
    --disable-wandb-random-suffix
  )
  [[ -n "${WANDB_KEY:-}" ]] && TRAIN_ARGS+=(--wandb-key "${WANDB_KEY}")
  [[ -n "${WANDB_HOST:-}" ]] && TRAIN_ARGS+=(--wandb-host "${WANDB_HOST}")
  [[ -n "${WANDB_TEAM:-}" ]] && TRAIN_ARGS+=(--wandb-team "${WANDB_TEAM}")
fi
if [[ "${USE_TENSORBOARD:-0}" == "1" ]]; then
  TRAIN_ARGS+=(
    --use-tensorboard
    --tb-project-name "${TENSORBOARD_DIR:-${RUNS_ROOT}/tensorboard}"
    --tb-experiment-name "eval"
  )
fi

{
  printf '%q ' "${SLIME_PYTHON}" -u train.py "${TRAIN_ARGS[@]}"
  printf '\n'
} >"${RUN_ROOT}/eval_command.txt"

if [[ "${CHECK_ONLY:-0}" == "1" ]]; then
  printf 'EVAL_CHECKPOINT=%s\nEVAL_CONFIG=%s\nRUN_ROOT=%s\n' \
    "${EVAL_CHECKPOINT}" "${EVAL_CONFIG}" "${RUN_ROOT}"
  exit 0
fi

command -v nvidia-smi >/dev/null || { echo "nvidia-smi is required" >&2; exit 2; }
[[ -x "${RAY_BIN}" ]] || { echo "ray is required: ${RAY_BIN}" >&2; exit 2; }
if RAY_ADDRESS="${RAY_ADDRESS}" "${RAY_BIN}" status >/dev/null 2>&1; then
  echo "An existing Ray cluster is running; stop it first." >&2
  exit 2
fi

export PYTHONUNBUFFERED=1
export PI_BIN
export CUDA_HOME="${CUDA_HOME:-${MAMBA_ROOT_PREFIX}/envs/slime}"
export PATH="${CUDA_HOME}/bin:${HOME}/.local/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-/usr/local/cuda/lib64}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export SHOP_ENV_URL SHOP_MAX_TURNS="${MAX_MODEL_TURNS}"
export SHOP_CONTEXT_KEEP_ACT_RESULTS="${SHOP_CONTEXT_KEEP_ACT_RESULTS:-3}"
export SHOP_ROLLOUT_TIMEOUT_SEC="${SHOP_ROLLOUT_TIMEOUT_SEC:-600}"
# D5 non-RL baseline: optional ReAct-style system prompt override. Set
# SHOP_SYSTEM_PROMPT (literal) or SHOP_SYSTEM_PROMPT_FILE (path) before
# launching to evaluate the same checkpoint under different instructions.
if [[ -n "${SHOP_SYSTEM_PROMPT:-}" ]]; then export SHOP_SYSTEM_PROMPT; fi
if [[ -n "${SHOP_SYSTEM_PROMPT_FILE:-}" ]]; then export SHOP_SYSTEM_PROMPT_FILE; fi
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
  # ray stop --force 是全局命令，并行评测时会误杀其他实验的 Ray 集群。
  # RAY_KEEP_CLUSTER=1 时保留集群，改为提示手动清理。
  if (( RAY_STARTED == 1 )); then
    if [[ "${RAY_KEEP_CLUSTER:-0}" == "1" ]]; then
      echo "评测退出。Ray 集群保留（RAY_KEEP_CLUSTER=1），如需停止：" >&2
      echo "  RAY_ADDRESS=${RAY_ADDRESS} ${RAY_BIN} stop --force" >&2
    else
      RAY_ADDRESS="${RAY_ADDRESS}" "${RAY_BIN}" stop --force >/dev/null 2>&1 || true
    fi
  fi
}
trap cleanup EXIT INT TERM

"${RAY_BIN}" start --head --node-ip-address "${MASTER_ADDR}" --num-gpus 1 \
  --port "${RAY_GCS_PORT}" --disable-usage-stats \
  --dashboard-host=127.0.0.1 --dashboard-port="${RAY_DASHBOARD_PORT}" \
  --temp-dir "${RAY_TEMP_DIR}"
RAY_STARTED=1

RUNTIME_ENV_JSON="$("${SLIME_PYTHON}" -c 'import json, os; keys=("PYTHONPATH","PATH","CUDA_HOME","LD_LIBRARY_PATH","MASTER_ADDR","NO_PROXY","no_proxy","CUDA_VISIBLE_DEVICES","CUDA_DEVICE_MAX_CONNECTIONS","PYTORCH_CUDA_ALLOC_CONF","OMP_NUM_THREADS","SHOP_ENV_URL","SHOP_MAX_TURNS","SHOP_CONTEXT_KEEP_ACT_RESULTS","SHOP_ROLLOUT_TIMEOUT_SEC","SHOP_SYSTEM_PROMPT","SHOP_SYSTEM_PROMPT_FILE","ADAPTER_PUBLIC_HOST","ADAPTER_BIND_HOST","ADAPTER_PORT","PI_BIN","WANDB_MODE","WANDB_API_KEY","WANDB_BASE_URL","TENSORBOARD_DIR"); print(json.dumps({"env_vars": {key: os.environ[key] for key in keys if key in os.environ}}))')"

# 等待 Ray job agent 就绪（ray start 后 agent 需要数秒启动，
# 过早 submit 会报 "No available agent to submit job" 500 错误）
AGENT_READY=0
for i in $(seq 1 36); do
  if RAY_ADDRESS="${RAY_ADDRESS}" "${RAY_BIN}" job list >/dev/null 2>&1; then
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

cd "${SLIME_DIR}"
# 提交评测 job：agent 未完全就绪时 submit 会报 500 "No available agent"，
# dashboard 就绪(job list 成功)不代表 agent runtime 已注册，故带重试
SUBMIT_OK=0
for i in $(seq 1 10); do
  # || true 防 set -e：submit 失败时靠 JOB_ID 判断，而非让脚本直接终止
  SUBMIT_OUT="$("${RAY_BIN}" job submit --address="${RAY_ADDRESS}" \
      --runtime-env-json="${RUNTIME_ENV_JSON}" --no-wait \
      -- "${SLIME_PYTHON}" -u train.py "${TRAIN_ARGS[@]}" 2>&1 || true)"
  echo "${SUBMIT_OUT}" | tail -4 >&2
  # { grep || true; } 防 pipefail：grep 无匹配时退出码 1 会经管道触发 set -e
  JOB_ID="$(printf '%s' "${SUBMIT_OUT}" | { grep -oE 'raysubmit_[A-Za-z0-9]+' || true; } | tail -1)"
  if [[ -n "${JOB_ID}" ]]; then
    SUBMIT_OK=1
    break
  fi
  echo "job submit 失败（第 $i/10 次，agent 可能未就绪），15 秒后重试..." >&2
  sleep 15
done
if (( SUBMIT_OK != 1 )); then
  echo "错误: job submit 重试 10 次均失败" >&2
  exit 4
fi
echo "评测 job 已提交: ${JOB_ID}"

# 跟踪 job 日志直至结束（--follow 阻塞到 job 完成）。
# || true：ray CLI 在 job 正常结束后仍可能以非零码退出（WebSocket 关闭被
# 判为异常），在 set -euo pipefail 下会误杀脚本、跳过下方 status 判定与
# summarize（2026-09-02 DAPO 评测误报 code=1 的根因）。成败以 job status 为准。
"${RAY_BIN}" job logs --address="${RAY_ADDRESS}" --follow "${JOB_ID}" 2>&1 | tee "${RUN_ROOT}/eval.log" || true

# 以 job 最终状态判定成败（job 运行失败不重试，只有提交失败才重试）。
# 注意：job status 输出含多行横幅，需匹配关键词而非盲取最后一行。
STATUS="$("${RAY_BIN}" job status "${JOB_ID}" --address="${RAY_ADDRESS}" 2>/dev/null | grep -oE 'SUCCEEDED|FAILED|STOPPED|DEAD' | head -1)"
echo "评测 job 最终状态: ${STATUS}" >&2
case "${STATUS}" in
  SUCCEEDED) : ;;
  *) echo "错误: 评测 job 未成功 (status=${STATUS:-UNKNOWN})" >&2; exit 5 ;;
esac

"${SLIME_PYTHON}" -m examples.ShopSimulator.utils summarize --run-root "${RUN_ROOT}"
printf 'Evaluation complete. Results: %s\n' "${RUN_ROOT}/eval_results.json"
