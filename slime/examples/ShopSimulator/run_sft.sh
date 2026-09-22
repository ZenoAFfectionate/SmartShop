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
# 动态 batch 每 microbatch 的 token 上限。SFT 必须计算**全词表 logits**，
# 峰值显存 ≈ max_tokens × vocab(151936) × 2B：8192→2.5GB、16384→5GB、
# 32768→10GB（叠加激活/优化器后 48GB 卡必然 OOM，2026-09-22 实测）。
# 默认保持历史的 8192；显存充裕时可用 env 覆盖，但请对照上表。
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-8192}"
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
if [[ "${USE_WANDB:-0}" == "1" ]]; then
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
# 关键路径预检：配错只会在 job **运行中**才暴露（import 失败等），与 CUDA_HOME
# 同类隐蔽。SFT 不启动 agent，故无需校验 PI_BIN。这里提前 fail fast。
[[ -x "${SLIME_PYTHON}" ]] || { echo "错误: SLIME_PYTHON 不可执行: ${SLIME_PYTHON}" >&2; exit 2; }
[[ -d "${MEGATRON_DIR}" ]] || { echo "错误: MEGATRON_DIR 不存在: ${MEGATRON_DIR}" >&2; exit 2; }

# GPU 占用预检：目标卡上已有进程时启动必然 OOM（与 max-tokens-per-gpu 过大会
# 造成 OOM 并列为两类显存故障），与 run_rl.sh 的 check_gpu_free 同款防护。
# 放在 CHECK_ONLY 之后：dry-run 只验证参数生成，不应被本机占用情况绑架。
_SFT_GPU_INDEX="${CUDA_VISIBLE_DEVICES:-0}"
_SFT_GPU_INDEX="${_SFT_GPU_INDEX%%,*}"
_SFT_GPU_USED="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${_SFT_GPU_INDEX}" 2>/dev/null | tr -d ' ')"
if [[ "${_SFT_GPU_USED}" =~ ^[0-9]+$ ]] && (( _SFT_GPU_USED > 2000 )); then
  echo "错误: GPU ${_SFT_GPU_INDEX} 已被占用（${_SFT_GPU_USED} MiB）；如有正在运行的训练请先停止" >&2
  exit 2
fi
# AF_UNIX socket 路径 ≤107 字符：temp_dir + session_<42 字符> + /sockets/
# dash_MetricsHead 必须放得下，temp_dir 过长会让 dashboard 起不来、agent
# 永远不就绪（2026-09-05 RL 评测实测）。放在 CHECK_ONLY 之后：dry-run 只
# 验证参数生成，不应被本地临时路径长度绑架。
if [[ ${#RAY_TEMP_DIR} -gt 40 ]]; then
  echo "错误: RAY_TEMP_DIR 长度 ${#RAY_TEMP_DIR} > 40 字符，会导致 dashboard 无法启动：${RAY_TEMP_DIR}" >&2
  exit 2
fi
if "${RAY_BIN}" status >/dev/null 2>&1; then
  echo "An existing Ray cluster is running; stop it first." >&2
  exit 2
fi

export PYTHONUNBUFFERED=1
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
    # 等待 GCS 进程真正退出：stop 返回时端口释放是异步的，不等待会让紧接着的
    # 下一个实验 ray start 撞"端口被占用"（2026-09-04 run_rl chain 实测）。
    local _w=0
    while pgrep -f -- "temp_dir=${RAY_TEMP_DIR:-__none__}" >/dev/null 2>&1; do
      (( _w >= 30 )) && break
      sleep 2; _w=$(( _w + 2 ))
    done
  fi
}
trap cleanup EXIT INT TERM

"${RAY_BIN}" start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" \
  --disable-usage-stats --dashboard-host=127.0.0.1 --dashboard-port=8265 \
  --temp-dir "${RAY_TEMP_DIR}"
RAY_STARTED=1

# 等待 dashboard/agent 就绪：ray start 返回只代表 head 进程已拉起，
# dashboard agent 注册完成前 job submit 会 500（No available agent）。
# （与 run_rl.sh 同一防护；2026-09-03 曾在 RL 侧遇到同类失败。）
_RAY_DEADLINE=$(( SECONDS + 120 ))
until curl -sf -m 3 "http://127.0.0.1:8265/api/version" >/dev/null 2>&1; do
  if (( SECONDS >= _RAY_DEADLINE )); then
    echo "错误: Ray dashboard 在 120s 内未就绪，放弃提交 SFT 训练" >&2
    exit 2
  fi
  sleep 3
done

RUNTIME_ENV_JSON="$("${SLIME_PYTHON}" -c 'import json, os; keys=("PYTHONPATH","PATH","CUDA_HOME","LD_LIBRARY_PATH","MASTER_ADDR","NO_PROXY","no_proxy","CUDA_DEVICE_MAX_CONNECTIONS","PYTORCH_CUDA_ALLOC_CONF","OMP_NUM_THREADS","NVTE_DEBUG","NVTE_DEBUG_LEVEL","WANDB_MODE","WANDB_API_KEY","WANDB_BASE_URL","TENSORBOARD_DIR"); print(json.dumps({"env_vars": {key: os.environ[key] for key in keys if key in os.environ}}))')"

cd "${SLIME_DIR}"
# --working-dir 必须显式指定：不指定时 job 的 cwd 取决于 ray head 进程的启动
# 目录（随调用位置漂移），曾导致 can't open train.py。提交阶段的网关错误
# （500/504 等）可重试；job 已开始运行后的失败不重试（避免重跑整场训练）。
_SUBMIT_OK=0
for _ATTEMPT in 1 2 3; do
  if "${RAY_BIN}" job submit --address=http://127.0.0.1:8265 \
    --working-dir "${SLIME_DIR}" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- "${SLIME_PYTHON}" -u train.py "${TRAIN_ARGS[@]}" 2>&1 | tee "${RUN_ROOT}/train.log"; then
    _SUBMIT_OK=1
  else
    _RC=$?
  fi
  (( _SUBMIT_OK == 1 )) && break
  if grep -qE "No available agent|status code 5[0-9][0-9]" "${RUN_ROOT}/train.log" 2>/dev/null; then
    echo "Ray agent 未就绪（第 ${_ATTEMPT}/3 次，code=${_RC:-?}），20s 后重试..." >&2
    sleep 20
  else
    echo "错误: SFT ray job 运行失败 (code=${_RC:-?})，不重试" >&2
    break
  fi
done
if (( _SUBMIT_OK != 1 )); then
  echo "错误: SFT 训练提交失败（见 ${RUN_ROOT}/train.log）" >&2
  exit 2
fi

printf 'SFT complete. HF exports: %s\nMegatron checkpoints: %s\n' \
  "${RUN_ROOT}/hf" "${RUN_ROOT}/checkpoints"
