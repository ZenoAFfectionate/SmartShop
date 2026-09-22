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
# AF_UNIX socket 路径 ≤107 字节：ray 会在 temp_dir/session_<42 字符>/sockets/
# dash_MetricsHead 下建 socket，temp_dir 过长（>40 字符）会让 dashboard 的
# metrics 模块起不来、agent 永远不就绪。算法名越长越容易触顶
# 的根因；名字越长的算法越容易触发，属隐性路径长度炸弹）。
if [[ ${#RAY_TEMP_DIR} -gt 40 ]]; then
  echo "错误: RAY_TEMP_DIR 长度 ${#RAY_TEMP_DIR} > 40 字符，AF_UNIX socket 路径会超限导致 dashboard 无法启动：${RAY_TEMP_DIR}" >&2
  echo "请换用更短的目录（如 /home/kemove/ray/e_xxx）后重试。" >&2
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
  --seed "${SEED}"
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --attention-softmax-in-fp32
  --attention-backend flash
  --loss-mask-type qwen3_5
  --colocate
)

# SGLang deterministic inference（与 run_rl.sh 同一开关）：评测保留默认开以
# 保证可复现（k=1 单次采样需同输入同输出）；设 SGLANG_DETERMINISTIC_INFERENCE=0
# 可启用 radix cache 加速（注意会让评测结果不再严格可复现）。
if [[ "${SGLANG_DETERMINISTIC_INFERENCE:-1}" == "1" ]]; then
  TRAIN_ARGS+=(--sglang-enable-deterministic-inference)
fi

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
# 关键路径预检：配错只会在 job **运行中**才暴露（rollout 起不来 / import 失败），
# 与 CUDA_HOME 同类隐蔽。这里提前 fail fast。
[[ -x "${SLIME_PYTHON}" ]] || { echo "错误: SLIME_PYTHON 不可执行: ${SLIME_PYTHON}" >&2; exit 2; }
[[ -d "${MEGATRON_DIR}" ]] || { echo "错误: MEGATRON_DIR 不存在: ${MEGATRON_DIR}" >&2; exit 2; }
# 注：PI_BIN 的可执行校验已在参数构建阶段完成（见文件上方 "pi is required"）。
if RAY_ADDRESS="${RAY_ADDRESS}" "${RAY_BIN}" status >/dev/null 2>&1; then
  echo "An existing Ray cluster is running; stop it first." >&2
  exit 2
fi

export PYTHONUNBUFFERED=1
export PI_BIN
# CUDA_HOME 必须含 nvcc：SGLang/flashinfer 的 CUDA graph JIT 在**运行中**才编译，
# 路径错误会让 server 启动阶段崩溃，且错误埋在 SGLang 日志里难以定位
# （2026-09-22 实测：MAMBA_ROOT_PREFIX 指向 /home/... 而实际 env 在 /hdd/...，
# 默认值 ${MAMBA_ROOT_PREFIX}/envs/slime 下没有 nvcc）。故改为探测 + 前置校验。
if [[ -z "${CUDA_HOME:-}" ]]; then
  for _CUDA_CAND in "${BASE_DIR}/cuda" /usr/local/cuda "${MAMBA_ROOT_PREFIX}/envs/slime"; do
    if [[ -x "${_CUDA_CAND}/bin/nvcc" ]]; then
      CUDA_HOME="${_CUDA_CAND}"
      break
    fi
  done
fi
if [[ -z "${CUDA_HOME:-}" || ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  echo "错误: 找不到可用的 nvcc（CUDA_HOME=${CUDA_HOME:-未设置}）；SGLang/flashinfer 的 JIT 编译会失败，请显式设置 CUDA_HOME=/usr/local/cuda" >&2
  exit 2
fi
export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${HOME}/.local/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-/usr/local/cuda/lib64}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export SHOP_ENV_URL SHOP_MAX_TURNS="${MAX_MODEL_TURNS}"
export SHOP_CONTEXT_KEEP_ACT_RESULTS="${SHOP_CONTEXT_KEEP_ACT_RESULTS:-3}"
# R5 结构化记忆（与 run_rl.sh 同一开关，保证训练/评测上下文一致）。
export SHOP_CONTEXT_STRUCTURED_MEMORY="${SHOP_CONTEXT_STRUCTURED_MEMORY:-1}"
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

# GPU 占用预检：目标卡已有进程时启动必然 OOM（与 run_rl.sh/run_sft.sh 同款防护）。
# 放在 CHECK_ONLY 之后：dry-run 不应被本机占用情况绑架。
_EVAL_GPU_INDEX="${CUDA_VISIBLE_DEVICES:-0}"
_EVAL_GPU_INDEX="${_EVAL_GPU_INDEX%%,*}"
_EVAL_GPU_USED="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${_EVAL_GPU_INDEX}" 2>/dev/null | tr -d ' ')"
if [[ "${_EVAL_GPU_USED}" =~ ^[0-9]+$ ]] && (( _EVAL_GPU_USED > 2000 )); then
  echo "错误: GPU ${_EVAL_GPU_INDEX} 已被占用（${_EVAL_GPU_USED} MiB）；如有正在运行的实验请先停止" >&2
  exit 2
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
      # 等待 GCS 进程真正退出：stop 返回时端口释放是异步的，不等待会让 chain 里
      # 紧接着的下一个实验 ray start 撞"端口被占用"（2026-09-04 run_rl chain 实测）。
      local _w=0
      while pgrep -f -- "temp_dir=${RAY_TEMP_DIR:-__none__}" >/dev/null 2>&1; do
        (( _w >= 30 )) && break
        sleep 2; _w=$(( _w + 2 ))
      done
    fi
  fi
}
trap cleanup EXIT INT TERM

"${RAY_BIN}" start --head --node-ip-address "${MASTER_ADDR}" --num-gpus 1 \
  --port "${RAY_GCS_PORT}" --disable-usage-stats \
  --dashboard-host=127.0.0.1 --dashboard-port="${RAY_DASHBOARD_PORT}" \
  --temp-dir "${RAY_TEMP_DIR}"
RAY_STARTED=1

# 等待 dashboard/agent 就绪（同 run_rl.sh/run_sft.sh）：ray start 返回只代表 head
# 进程已拉起，agent 注册完成前 submit 会 500。此处理失败只告警不退出，因为下方
# submit 自带 10 次重试兜底。
_RAY_DEADLINE=$(( SECONDS + 120 ))
until curl -sf -m 3 "http://127.0.0.1:${RAY_DASHBOARD_PORT}/api/version" >/dev/null 2>&1; do
  if (( SECONDS >= _RAY_DEADLINE )); then
    echo "警告: Ray dashboard 120s 内未就绪，仍继续提交（submit 有 10 次重试兜底）" >&2
    break
  fi
  sleep 3
done

RUNTIME_ENV_JSON="$("${SLIME_PYTHON}" -c 'import json, os; keys=("PYTHONPATH","PATH","CUDA_HOME","LD_LIBRARY_PATH","MASTER_ADDR","NO_PROXY","no_proxy","CUDA_VISIBLE_DEVICES","CUDA_DEVICE_MAX_CONNECTIONS","PYTORCH_CUDA_ALLOC_CONF","OMP_NUM_THREADS","SHOP_ENV_URL","SHOP_MAX_TURNS","SHOP_CONTEXT_KEEP_ACT_RESULTS","SHOP_CONTEXT_STRUCTURED_MEMORY","SHOP_ROLLOUT_TIMEOUT_SEC","SHOP_SYSTEM_PROMPT","SHOP_SYSTEM_PROMPT_FILE","ADAPTER_PUBLIC_HOST","ADAPTER_BIND_HOST","ADAPTER_PORT","PI_BIN","WANDB_MODE","WANDB_API_KEY","WANDB_BASE_URL","TENSORBOARD_DIR"); print(json.dumps({"env_vars": {key: os.environ[key] for key in keys if key in os.environ}}))')"

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
  # --working-dir 必须显式指定：不指定时 job 的 cwd 取决于 ray head 进程的启动
  # 目录（随调用位置漂移），曾导致 can't open train.py（与 run_rl.sh 同因）。
  SUBMIT_OUT="$("${RAY_BIN}" job submit --address="${RAY_ADDRESS}" \
      --working-dir "${SLIME_DIR}" \
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
# summarize。成败以 job status 为准。
"${RAY_BIN}" job logs --address="${RAY_ADDRESS}" --follow "${JOB_ID}" 2>&1 | tee "${RUN_ROOT}/eval.log" || true

# 以 job 最终状态判定成败（job 运行失败不重试，只有提交失败才重试）。
# 注意：job status 输出含多行横幅，需匹配关键词而非盲取最后一行。
# || true 防 set -e/pipefail：状态查询偶发失败时不应让整个评测以假失败收场。
# 状态查询返回空时重试 3 次；仍为空则用 **rollout dump** 兜底判定——
# 2026-09-22 实测：日志已打印 "Job 'xxx' succeeded"、dump 93MB 已落盘，
# status 查询却连续为空，评测被误判失败、summarize 被跳过（真实结果丢失）。
# 注意仅"查询为空"才兜底；status 明确为 FAILED/STOPPED/DEAD 时仍按失败处理。
STATUS=""
for _i in 1 2 3; do
  STATUS="$("${RAY_BIN}" job status "${JOB_ID}" --address="${RAY_ADDRESS}" 2>/dev/null | { grep -oE 'SUCCEEDED|FAILED|STOPPED|DEAD' || true; } | head -1)"
  [[ -n "${STATUS}" ]] && break
  echo "job status 查询为空（第 ${_i}/3 次），5s 后重试..." >&2
  sleep 5
done
echo "评测 job 最终状态: ${STATUS:-<查询为空>}" >&2

DUMP_FILE="$(ls -1 "${RUN_ROOT}"/rollout_dumps/rollout_eval_*.pt 2>/dev/null | head -1)"
DUMP_BYTES=0
[[ -n "${DUMP_FILE}" ]] && DUMP_BYTES="$(stat -c%s "${DUMP_FILE}" 2>/dev/null || echo 0)"
if (( DUMP_BYTES > 1048576 )); then DUMP_OK=1; else DUMP_OK=0; fi

case "${STATUS}" in
  SUCCEEDED) : ;;
  "")
    if (( DUMP_OK == 1 )); then
      echo "警告: job status 查询为空，但已发现完整 rollout dump（${DUMP_BYTES} bytes），按成功继续" >&2
    else
      echo "错误: 评测 job 状态未知且未发现 rollout dump" >&2
      exit 5
    fi ;;
  *)
    echo "错误: 评测 job 未成功 (status=${STATUS})" >&2
    exit 5 ;;
esac

"${SLIME_PYTHON}" -m examples.ShopSimulator.utils summarize --run-root "${RUN_ROOT}"
printf 'Evaluation complete. Results: %s\n' "${RUN_ROOT}/eval_results.json"
