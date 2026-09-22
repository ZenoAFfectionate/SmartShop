#!/usr/bin/env bash
# One-epoch online RL over rl_500 from an SFT checkpoint, with algorithm switching
# and automatic post-training evaluation.
#
# 用法:
#   bash run_rl.sh                                      # 默认 GRPO，训练完自动评测
#   ALGORITHM=cispo bash run_rl.sh                      # 单算法
#   ALGORITHM=all nohup bash run_rl.sh > rl_all.log 2>&1 &   # 串行跑全部六种算法
#   ALGORITHM=grpo AUTO_EVAL=0 bash run_rl.sh           # 只训练不评测
#
# ALGORITHM 可选值:
#   grpo          标准 GRPO（默认）
#   dr_grpo       Dr.GRPO：GRPO 去除 advantage 的 std 归一化
#   gspo          GSPO：序列级优势估计器
#   dapo          DAPO 组合：token-level loss + 动态采样(零方差组过滤) + 去 std 归一化
#   reinforce_pp  REINFORCE++
#   cispo         CISPO
#   all           依次执行以上全部（每个算法 = 训练 → 评测 → 下一个）
#
# 其余超参全部来自 config/shop_rl.json（与既有六组实验严格一致，唯一变量=算法）。
# 可覆盖变量:
#   AUTO_EVAL(默认1 训练后自动评测) / SAVE_INTERVAL(默认25) /
#   OVER_SAMPLING_BATCH_SIZE(默认10, DAPO 过采样批) / RESUME(1=断点续训) /
#   DAPO_DYNAMIC_SAMPLING(默认0 过渡模式不开启动态采样; 1=恢复 over-sampling+非零方差过滤) /
#   START_ROLLOUT_ID / LOAD_FROM / CHECK_ONLY(1=只打印命令不执行) /
#   RAY_GCS_PORT(默认6379) / RAY_DASHBOARD_PORT(默认8265) / ADAPTER_PORT(默认18080) /
#   RAY_TEMP_DIR(默认 BASE_DIR/ray/rl_<算法>) /
#   RUN_ROOT 显式指定时仅用于单算法模式（all 模式按算法自动命名）
#
# 多实验并行配方（一卡一算法）：
#   # 终端 A（GPU0）:
#   CUDA_VISIBLE_DEVICES=0 ALGORITHM=grpo bash run_rl.sh
#   # 终端 B（GPU1）: 三类端口必须错开（temp-dir 已按算法自动独立）
# 多卡并行（每卡一个独立 ray 集群 + 一个独立 env 实例）：
#   # 0) 启动/补齐 env 实例池（每实例 20 槽位 = 一个实验的并发 candidate 数）
#   bash ShopSimulator/start_server.sh 3
#   # 1) 启动前预检（只读：不启动服务、不创建 RUN_ROOT）
#   PREFLIGHT=1 CUDA_VISIBLE_DEVICES=1 ALGORITHM=cispo RUN_TAG=_mem2 \
#     RAY_GCS_PORT=6380 RAY_DASHBOARD_PORT=8266 RAY_DASHBOARD_AGENT_PORT=52366 \
#     RAY_MIN_WORKER_PORT=20000 RAY_MAX_WORKER_PORT=29999 \
#     RAY_METRICS_EXPORT_PORT=44218 ADAPTER_PORT=18082 \
#     SHOP_ENV_URL=http://127.0.0.1:5001/api/shop_agent \
#     RAY_TEMP_DIR=/hdd/kemove/ray/e1 bash run_rl.sh
#   # 2) 预检通过后去掉 PREFLIGHT=1 正式启动
#   端口错开配方（实验序号 i）：gcs 6379+i / dashboard 8265+i / agent 52365+i /
#     worker (i+1)0000-((i+2)0000-1) / metrics 44217+i / adapter 18081+i /
#     env 5000+i / RAY_TEMP_DIR ${BASE_DIR}/ray/e<i>
#   CUDA_VISIBLE_DEVICES=1 ALGORITHM=cispo RAY_GCS_PORT=6380 \
#     RAY_DASHBOARD_PORT=8266 ADAPTER_PORT=18081 bash run_rl.sh
# 原理与约束:
#   - CUDA_VISIBLE_DEVICES 同时限定 Ray 资源注册与训练/推理进程可见的卡；
#   - ShopSimulator server 的环境池（默认 20 槽）为所有实验共享，两个 RL 同时
#     跑会分抢容量导致吞吐减半，必要时下调各实验并发；
#   - 清理只作用于本集群（按 GCS 端口 + temp-dir 精确匹配进程），绝不使用全局
#     ray stop --force —— 它会误杀机器上其他实验的 Ray 集群（血泪教训）。

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
# checkpoint every N rollouts (25 by default) so dev-based model selection has
# candidates to choose from; the final rollout always saves too.
SAVE_INTERVAL="${SAVE_INTERVAL:-25}"
if (( SAVE_INTERVAL <= 0 )); then
  echo "SAVE_INTERVAL must be positive" >&2
  exit 2
fi
OVER_SAMPLING_BATCH_SIZE="${OVER_SAMPLING_BATCH_SIZE:-10}"

# ShopSimulator env 自动修复与训练中守护（详见 env 预检函数块注释）
ENV_WATCHDOG="${ENV_WATCHDOG:-1}"
ENV_WATCHDOG_INTERVAL="${ENV_WATCHDOG_INTERVAL:-300}"
GPU_INDEX="${GPU_INDEX:-0}"
SHOPSIM_PYTHON="${SHOPSIM_PYTHON:-${HOME}/.conda/envs/shopsim/bin/python}"
SHOPSIM_JAVA_HOME="${SHOPSIM_JAVA_HOME:-}"
SHOP_ENV_DIR="${SHOP_ENV_DIR:-}"
SHOP_ENV_LOG="${SHOP_ENV_LOG:-}"
if [[ -z "${SHOP_ENV_DIR}" && -d "${SLIME_DIR}/../ShopSimulator/shop_env/shop_env" ]]; then
  SHOP_ENV_DIR="${SLIME_DIR%/slime}/ShopSimulator/shop_env/shop_env"
fi
if [[ -z "${SHOP_ENV_LOG}" && -n "${SHOP_ENV_DIR}" ]]; then
  SHOP_ENV_LOG="${SHOP_ENV_DIR%/shop_env/shop_env}/result/shopsim_server.log"
fi

AUTO_EVAL="${AUTO_EVAL:-1}"
ALGORITHM="${ALGORITHM:-grpo}"
ALL_ALGOS=(grpo dr_grpo gspo dapo reinforce_pp cispo)
case " ${ALL_ALGOS[*]} all " in
  *" ${ALGORITHM} "*) : ;;
  *) echo "未知 ALGORITHM: ${ALGORITHM}（可选: ${ALL_ALGOS[*]} | all）" >&2; exit 2 ;;
esac

STAMP="$(date +%Y%m%d_%H%M%S)"
RUNS_ROOT="${RUNS_ROOT:-${BASE_DIR}/slime-runs}"
# Ray 集群端口：并行多实验时必须各自错开（见头部并行配方）。
RAY_GCS_PORT="${RAY_GCS_PORT:-6379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
ADAPTER_PUBLIC_HOST="${ADAPTER_PUBLIC_HOST:-127.0.0.1}"
ADAPTER_BIND_HOST="${ADAPTER_BIND_HOST:-0.0.0.0}"
ADAPTER_PORT="${ADAPTER_PORT:-18080}"
# 项目内路径（评测日志落盘位置）
SHOP_ROOT="${SHOP_ROOT:-$(cd "${SLIME_DIR}/.." && pwd)}"
LOG_DIR="${LOG_DIR:-${SHOP_ROOT}/result/rl}"

# B1 训练监控：USE_WANDB=1 注入 wandb 参数组（默认 offline 模式——本地落盘、
# 事后 wandb sync 上传，服务器无外网也可用）；USE_TENSORBOARD=1 落本地 tb
# 事件文件作为无账号备选。命名规范 {project}/{group=算法名}/{run name=RUN_ROOT
# 名}：group 聚合使六算法对照曲线同图，run name 用 RUN_ROOT 名保证唯一。
# WANDB_API_KEY / WANDB_BASE_URL 需在启动前 export（在线模式才需要）。
USE_WANDB="${USE_WANDB:-0}"
USE_TENSORBOARD="${USE_TENSORBOARD:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-smartshop}"
WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_DIR="${WANDB_DIR:-${RUNS_ROOT}/wandb}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${RUNS_ROOT}/tensorboard}"
WANDB_KEY="${WANDB_KEY:-}"
WANDB_HOST="${WANDB_HOST:-}"
WANDB_TEAM="${WANDB_TEAM:-}"

# 保留用户显式传入的 RUN_ROOT（仅单算法模式；all 模式按算法自动命名并忽略之）
USER_RUN_ROOT="${RUN_ROOT:-}"

# RUN_TAG：同一算法多组对照实验的目录后缀（如 _r12r32 表示启用 R1-2 分解
# 优势 + R3-2 行为分的重训组）。同时作用于训练 RUN_ROOT、评测 RUN_ROOT、
# 评测日志与 Ray temp-dir，避免覆盖无后缀的历史结果。
RUN_TAG="${RUN_TAG:-}"
if [[ -n "${RUN_TAG}" ]] && [[ ! "${RUN_TAG}" =~ ^_[A-Za-z0-9_-]+$ ]]; then
  echo "RUN_TAG 必须匹配 _[A-Za-z0-9_-]+（如 _r12r32），当前: ${RUN_TAG}" >&2
  exit 2
fi

# ── 算法 → advantage estimator 映射 ──────────────────────────────
estimator_for() {
  case "$1" in
    grpo|dr_grpo|dapo) echo "grpo" ;;
    gspo) echo "gspo" ;;
    reinforce_pp) echo "reinforce_plus_plus" ;;
    cispo) echo "cispo" ;;
  esac
}

say() { echo "[rl $(date '+%m-%d %H:%M:%S')] $*"; }

# ── 并行安全的 Ray 集群清理 ──────────────────────────────────────
# 只杀"本集群"的 daemon（gcs_server 按 GCS 端口匹配、其余组件按 temp-dir 匹配）。
# 绝不使用全局 ray stop --force —— 它会误杀机器上其他实验的 Ray 集群
# （CHECK_ONLY 触发清理时会误杀同名端口上正在运行的训练集群）。
# Ray worker（ray:: 进程）不含上述标识，但由 raylet 托管，raylet 退出后自行终止。
stop_own_ray() {
  pkill -TERM -f -- "gcs_server_port=${RAY_GCS_PORT}" 2>/dev/null || true
  pkill -TERM -f -- "temp_dir=${RAY_TEMP_DIR:-__undefined__}" 2>/dev/null || true
  sleep 5
  pkill -KILL -f -- "gcs_server_port=${RAY_GCS_PORT}" 2>/dev/null || true
  pkill -KILL -f -- "temp_dir=${RAY_TEMP_DIR:-__undefined__}" 2>/dev/null || true
  if [[ -n "${RAY_TEMP_DIR:-}" ]]; then
    rm -rf "${RAY_TEMP_DIR}"/session_* 2>/dev/null || true
  fi
  # 等待 GCS 进程真正退出：KILL 后端口释放是异步的，不等待会让 chain 里
  # 下一个算法 ray start 时撞 "GCS 端口被占"。
  local _wait=0
  while pgrep -f -- "gcs_server_port=${RAY_GCS_PORT}" >/dev/null 2>&1; do
    if (( _wait >= 30 )); then
      echo "警告: GCS 端口 ${RAY_GCS_PORT} 的进程 30s 内未退出，后续 ray start 可能失败" >&2
      return 0
    fi
    sleep 2
    _wait=$(( _wait + 2 ))
  done
}

# ── ShopSimulator env 预检与守护 ─────────────────────────────────
# 异常终止的 pi 会话（超时 SIGKILL / 崩溃）不会归还
# env 槽位，20 个槽位被泄漏耗尽后所有 reset 失败 → reward 全零 → 训练
# 空转 18 小时。以下预检 + watchdog 与 env 侧新增的 release_session /
# release_stale API（pack_api.py）共同构成防线：
#   启动前: ensure_env_ready —— 服务不通则自动拉起，残留会话自动清理，
#           槽位异常则拒绝训练（fail-fast）；
#   训练中: start_env_watchdog —— 周期回收 idle 超过阈值的泄漏槽位
#           （幂等，活跃会话每轮交互远快于阈值，不会误伤）。
env_call() {  # $1 = JSON body
  curl -s -m 15 -X POST "${SHOP_ENV_URL}" -H 'Content-Type: application/json' -d "$1" 2>/dev/null
}

env_field() {  # $1 = status JSON, $2 = field name
  "${SLIME_PYTHON}" -c 'import json,sys; print(json.load(sys.stdin)["result"][sys.argv[1]])' "$2" <<<"$1" 2>/dev/null
}

start_env_server() {
  # 按原命令行拉起 pack_api.py：cwd 必须在 shop_env 目录；JAVA 必须用
  # shopsim 环境自带的 JDK（${SHOPSIM_JAVA_HOME}，默认其 lib/jvm）——
  # 系统 Java 8 会因 Anserini 类版本过旧（需要 11+）启动即崩。
  [[ -d "${SHOP_ENV_DIR}" && -x "${SHOPSIM_PYTHON}" ]] || return 1
  local shopsim_bin jvm_home
  shopsim_bin="$(dirname "${SHOPSIM_PYTHON}")"
  jvm_home="${SHOPSIM_JAVA_HOME:-${shopsim_bin}/../lib/jvm}"
  [[ -d "${jvm_home}" ]] || return 1
  (
    cd "${SHOP_ENV_DIR}" || exit 1
    JAVA_HOME="${jvm_home}" PATH="${shopsim_bin}:${PATH}" \
      nohup "${SHOPSIM_PYTHON}" pack_api.py >> "${SHOP_ENV_LOG}" 2>&1 &
  )
}

ensure_env_ready() {
  local status cap free active waited
  status="$(env_call '{"action":"status"}')"
  if [[ -z "${status}" ]]; then
    say "env 预检: ShopSimulator 服务(${SHOP_ENV_URL})无响应，尝试自动拉起"
    if ! start_env_server; then
      echo "错误: 无法自动启动 ShopSimulator env（需要 SHOP_ENV_DIR=${SHOP_ENV_DIR:-<未探测到>} 与 SHOPSIM_PYTHON=${SHOPSIM_PYTHON} 有效）；请手动启动后重试" >&2
      return 1
    fi
    waited=0
    until status="$(env_call '{"action":"status"}')" && [[ -n "${status}" ]]; do
      sleep 3
      waited=$((waited + 3))
      if (( waited >= 120 )); then
        echo "错误: ShopSimulator env 拉起后 120s 内仍未就绪，请检查日志: ${SHOP_ENV_LOG}" >&2
        return 1
      fi
    done
    say "env 预检: ShopSimulator 已自动拉起并就绪"
  fi
  cap="$(env_field "${status}" capacity)"
  free="$(env_field "${status}" free)"
  active="$(env_field "${status}" active)"
  [[ -n "${active}" ]] || active=0
  if [[ -z "${cap}" || -z "${free}" ]]; then
    echo "错误: env status 响应异常: ${status}" >&2
    return 1
  fi
  if (( active > 0 )); then
    say "env 预检: 发现 ${active} 个残留会话（上次异常退出遗留），自动 release_all 清理"
    env_call '{"action":"release_all"}' >/dev/null
    status="$(env_call '{"action":"status"}')"
    cap="$(env_field "${status}" capacity)"
    free="$(env_field "${status}" free)"
    active="$(env_field "${status}" active)"
  fi
  if (( cap <= 0 )) || (( free < cap )); then
    echo "错误: env 槽位异常 capacity=${cap} free=${free} active=${active:-?}；请检查 ${SHOP_ENV_LOG}" >&2
    return 1
  fi
  say "env 预检通过: ${free}/${cap} 槽位空闲"
}

check_gpu_free() {
  # 物理设备 = CUDA_VISIBLE_DEVICES 的首个值（与 run_sft.sh / run_eval.sh 同语义）。
  # 旧写法用 GPU_INDEX（默认 0）会永远查 GPU 0 → 三卡并行时第二、三张卡被误报
  # "已被占用"（2026-09-22 三卡并行实测）。
  local _phys="${CUDA_VISIBLE_DEVICES:-0}"
  _phys="${_phys%%,*}"
  local used
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${_phys}" 2>/dev/null | tr -d ' ')"
  if ! [[ "${used}" =~ ^[0-9]+$ ]]; then
    echo "错误: 无法读取 GPU ${_phys} 状态（nvidia-smi）" >&2
    return 1
  fi
  if (( used > 2000 )); then
    echo "错误: GPU ${_phys} 已被占用（${used} MiB）；如有正在运行的训练请先停止" >&2
    return 1
  fi
}

# ── 并行前置检查（PREFLIGHT=1 调用；只读）────────────────────────────
preflight_one() {
  local algo="${1:-?}" problems=0
  local gpu="${CUDA_VISIBLE_DEVICES:-0}"
  gpu="${gpu%%,*}"
  echo "── PREFLIGHT（${algo}）──"
  # 1) GPU（按可见设备，与 check_gpu_free 同语义）
  local used
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${gpu}" 2>/dev/null | tr -d ' ')"
  if [[ "${used}" =~ ^[0-9]+$ ]] && (( used <= 2000 )); then
    echo "  ✓ GPU ${gpu} 空闲"
  else
    echo "  ✗ GPU ${gpu} 已占用 ${used:-读取失败} MiB"
    problems=$(( problems + 1 ))
  fi
  # 2) 本实例端口（agent 端口是并行时最容易撞的一个）
  local pair kind port holder
  for pair in "GCS:${RAY_GCS_PORT}" "dashboard:${RAY_DASHBOARD_PORT}" \
              "agent:${RAY_DASHBOARD_AGENT_PORT:-52365}" \
              "metrics:${RAY_METRICS_EXPORT_PORT:-44217}" \
              "adapter:${ADAPTER_PORT}"; do
    kind="${pair%%:*}"
    port="${pair#*:}"
    holder="$(ss -tlnp 2>/dev/null | grep -E "[:.]${port}[[:space:]]" | grep -oE 'users:\(\("[^"]+' | head -1 | sed 's/users:((\"//')"
    if [[ -n "${holder}" ]]; then
      echo "  ✗ ${kind} 端口 ${port} 被占（${holder}）→ 与其他实验错开"
      problems=$(( problems + 1 ))
    else
      echo "  ✓ ${kind} 端口 ${port} 空闲"
    fi
  done
  # worker 段抽查
  local wmin="${RAY_MIN_WORKER_PORT:-10000}" wmax="${RAY_MAX_WORKER_PORT:-19999}" wbusy=0
  for port in $(seq "${wmin}" "$(( wmin + 15 ))"); do
    if ss -tln 2>/dev/null | grep -qE "[:.]${port}[[:space:]]"; then wbusy=$(( wbusy + 1 )); fi
  done
  if (( wbusy > 0 )); then
    echo "  ✗ worker 段 ${wmin}-${wmax} 有 ${wbusy} 个端口被占 → 换段"
    problems=$(( problems + 1 ))
  else
    echo "  ✓ worker 段 ${wmin}-${wmax} 空闲"
  fi
  # 3) env 实例容量
  local body cap
  body="$(curl -s -m 4 "${SHOP_ENV_URL}" -X POST -H 'Content-Type: application/json' \
    -d '{"action":"status"}' 2>/dev/null)"
  cap="$(printf '%s' "${body}" | "${SLIME_PYTHON}" -c 'import json,sys
try:
    d = json.load(sys.stdin)["result"]
    print(d["capacity"], d["free"])
except Exception:
    print("", "")' 2>/dev/null)"
  set -- ${cap}
  if [[ -z "${1:-}" ]]; then
    echo "  ✗ env ${SHOP_ENV_URL} 无响应；多实例可用：bash ShopSimulator/start_server.sh 3"
    problems=$(( problems + 1 ))
  else
    echo "  ✓ env ${SHOP_ENV_URL} capacity=${1} free=${2}"
  fi
  # 4) RUN_ROOT 残留
  if [[ -e "${RUN_ROOT}" ]]; then
    echo "  ✗ RUN_ROOT 已存在（换 RUN_TAG 或删旧目录）：${RUN_ROOT}"
    problems=$(( problems + 1 ))
  else
    echo "  ✓ RUN_ROOT 可用：${RUN_ROOT}"
  fi
  # 5) ray temp-dir 残留进程
  local cnt
  cnt="$(pgrep -cf "temp_dir=${RAY_TEMP_DIR}([[:space:]]|$)" 2>/dev/null || true)"
  cnt="${cnt:-0}"
  if [[ "${cnt}" =~ ^[0-9]+$ ]] && (( cnt > 0 )); then
    echo "  ✗ temp-dir ${RAY_TEMP_DIR} 仍有 ${cnt} 个 ray 进程 → pkill -f 'temp_dir=${RAY_TEMP_DIR}'"
    problems=$(( problems + 1 ))
  else
    echo "  ✓ temp-dir ${RAY_TEMP_DIR} 无残留进程"
  fi
  if (( problems == 0 )); then
    echo "── PREFLIGHT 通过（0 问题）──"
    return 0
  fi
  echo "── PREFLIGHT 发现 ${problems} 个问题，请先处理 ──"
  return 2
}

start_env_watchdog() {
  [[ "${ENV_WATCHDOG}" == "1" ]] || return 0
  local max_idle=$(( ROLLOUT_TIMEOUT_SEC + 300 ))
  (
    while :; do
      sleep "${ENV_WATCHDOG_INTERVAL}"
      curl -s -m 15 -X POST "${SHOP_ENV_URL}" -H 'Content-Type: application/json' \
        -d "{\"action\":\"release_stale\",\"max_idle_seconds\":${max_idle}}" >/dev/null 2>&1 || true
    done
  ) >/dev/null 2>&1 &
  ENV_WATCHDOG_PID=$!
  say "env watchdog 已启动 (pid=${ENV_WATCHDOG_PID}, 每 ${ENV_WATCHDOG_INTERVAL}s 回收 idle>${max_idle}s 的泄漏槽位)"
}

stop_env_watchdog() {
  if [[ -n "${ENV_WATCHDOG_PID:-}" ]] && (( ENV_WATCHDOG_PID > 0 )); then
    kill "${ENV_WATCHDOG_PID}" 2>/dev/null || true
    pkill -TERM -P "${ENV_WATCHDOG_PID}" 2>/dev/null || true
    ENV_WATCHDOG_PID=0
  fi
}

# ── 训练单个算法（$1=算法标签）───────────────────────────────────
train_one() {
  local ALGO="$1"
  local ESTIMATOR
  ESTIMATOR="$(estimator_for "${ALGO}")"
  # 单算法模式尊重用户显式 RUN_ROOT；all 模式按算法自动命名。
  # RUN_TAG（如 _r12r32）用于同一算法的多组对照实验，避免覆盖历史 run/eval。
  local RUN_ROOT="${USER_RUN_ROOT:-${RUNS_ROOT}/qwen35_2b_shop_rl_${ALGO}${RUN_TAG}}"
  # Ray 会话目录按算法独立，是并行清理精确匹配的依据之一。
  # e_ 前缀：temp_dir 会拼进 AF_UNIX socket 路径（≤107 字符硬限制），
  # 长算法名 + 长 BASE_DIR 会撑爆它（详见 run_eval.sh 内同名校验）。
  local RAY_TEMP_DIR="${RAY_TEMP_DIR:-${BASE_DIR}/ray/e_${ALGO}}"

  # ── PREFLIGHT=1：并行编排前置检查（只读；不启动服务、不创建 RUN_ROOT）──
  # 检查 GPU 语义 / 端口段冲突（尤其 dashboard agent 随机撞端口）/ env 实例容量
  # / RUN_ROOT 残留 / ray temp-dir 残留。必须放在 Refusing 与 mkdir 之前，
  # 否则 RUN_ROOT 残留时预检看不到它，且预检本身会把目录创建出来。
  if [[ "${PREFLIGHT:-0}" == "1" ]]; then
    preflight_one "${ALGO}"
    return $?
  fi

  if [[ -e "${RUN_ROOT}" ]] && [[ "${RESUME:-0}" != "1" ]]; then
    echo "Refusing to overwrite existing RUN_ROOT: ${RUN_ROOT} (set RESUME=1 to resume from its checkpoints)" >&2
    return 2
  fi
  if [[ "${RESUME:-0}" == "1" ]] && [[ -f "${RUN_ROOT}/checkpoints/latest_checkpointed_iteration.txt" ]]; then
    say "RESUME=1：将从 $(cat "${RUN_ROOT}/checkpoints/latest_checkpointed_iteration.txt") 号 checkpoint 自动续训（权重+rollout_id+数据位置）"
  fi
  mkdir -p "${RUN_ROOT}/checkpoints" "${RUN_ROOT}/hf" "${RUN_ROOT}/rollout_dumps" "${RAY_TEMP_DIR}"

  local TRAIN_ARGS=(
    --actor-num-nodes 1
    --actor-num-gpus-per-node 1
    "${MODEL_ARGS[@]}"
    --hf-checkpoint "${HF_CHECKPOINT}"
    --ref-load "${REF_MODEL_PATH}"
    --load "${LOAD_FROM:-${RUN_ROOT}/checkpoints}"
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
    --advantage-estimator "${ESTIMATOR}"
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
    # 动态 batch 每 microbatch 的 token 上限。actor 训练同样要算全词表 logits
    # （与 SFT 同一条 loss 路径），峰值显存 ≈ max_tokens × vocab(151936) × 2B：
    # 16384→5GB、32768→10GB——SFT 侧已实测 32768 必 OOM，故默认取 chain 实测
    # 安全的 16384；仍可用 MAX_TOKENS_PER_GPU 覆盖。单条超长样本会独立成批
    # 不受此上限截断。
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-16384}"
    --rollout-num-gpus 1
    --rollout-num-gpus-per-engine 1
    --sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY}"
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.55}"
    --sglang-tool-call-parser qwen3_coder
    --sglang-reasoning-parser qwen3
    --seed "${SEED}"
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
    --loss-mask-type qwen3_5
    --colocate
  )

  # SGLang deterministic inference（默认开，保证同输入同输出、评测可复现）。
  # 注意：该模式下 radix cache（跨请求前缀缓存）仅在 fa3/triton/ascend
  # attention backend 受支持，其余 backend 会被 SGLang 静默禁用——表现为
  # rollout 指标 prefix_cache_hit_rate 恒为 0。RL 训练 rollout 用 temp=1.0
  # 采样，deterministic 的收益有限；若要启用 radix cache 加速 rollout，
  # 设 SGLANG_DETERMINISTIC_INFERENCE=0（训练与评测可分别取舍）。
  if [[ "${SGLANG_DETERMINISTIC_INFERENCE:-1}" == "1" ]]; then
    TRAIN_ARGS+=(--sglang-enable-deterministic-inference)
  fi

  # ── 恢复语义 ──────────────────────────────────────────────────
  # 默认不传 --start-rollout-id：slime 会自动从 --load 的 checkpoint 恢复
  # （新训练=0；续训=checkpoint 的 iteration，RL 下 iteration ≡ rollout_id，
  #   且 buffer 数据位置一并恢复）。仅当用户显式指定 START_ROLLOUT_ID 时
  #   才覆盖（例如故意从某个 rollout 重放）。
  if [[ -n "${START_ROLLOUT_ID:-}" ]]; then
    TRAIN_ARGS+=(--start-rollout-id "${START_ROLLOUT_ID}")
  fi

  # ── 算法专属开关（唯一变量=算法，其余与 config 严格一致）───────
  case "${ALGO}" in
    dr_grpo)
      TRAIN_ARGS+=(--disable-grpo-std-normalization)
      ;;
    dapo)
      # DAPO 组合：token-level loss + 去 std 归一化。
      # 动态采样（over-sampling + 非零方差组过滤）默认关闭过渡模式：
      # SFT policy 在 rl_500 上全零分时，过滤器会把所有组丢弃导致无限
      # 重采死锁（policy 得不到更新→永远全零→永远凑不齐）。
      # 等 policy 能拿到非零 reward 后，用 DAPO_DYNAMIC_SAMPLING=1 恢复。
      TRAIN_ARGS+=(--calculate-per-token-loss --disable-grpo-std-normalization)
      if [[ "${DAPO_DYNAMIC_SAMPLING:-0}" == "1" ]]; then
        TRAIN_ARGS+=(
          --over-sampling-batch-size "${OVER_SAMPLING_BATCH_SIZE}"
          --dynamic-sampling-filter-path examples.ShopSimulator.utils.check_reward_nonzero_std_grouped
        )
      fi
      ;;
    reinforce_pp)
      # slime 强制校验：reinforce_plus_plus 系 estimator 必须开 advantage 归一化
      TRAIN_ARGS+=(--normalize-advantages)
      ;;
  esac

  # R1-2 多维分解 advantage + R3-2 行为过程奖励：CISPO（当前最优 estimator）
  # 为对照载体默认开启；其余算法默认关闭。两者都作用于训练 advantage，
  # 不改变环境 reward 与评测指标。可用环境变量覆盖（如关闭：=0）。
  if [[ "${ALGO}" == "cispo" ]]; then
    export SHOP_DECOMPOSED_ADVANTAGE_WEIGHT="${SHOP_DECOMPOSED_ADVANTAGE_WEIGHT:-0.5}"
    export SHOP_BEHAVIOR_DELTA="${SHOP_BEHAVIOR_DELTA:-0.05}"
  else
    export SHOP_DECOMPOSED_ADVANTAGE_WEIGHT="${SHOP_DECOMPOSED_ADVANTAGE_WEIGHT:-0}"
    export SHOP_BEHAVIOR_DELTA="${SHOP_BEHAVIOR_DELTA:-0}"
  fi

  # ── B1 训练监控接线（见变量区注释）────────────────────────────
  # agent 专属日志：--log-multi-turn（多轮过程）与 --log-passrate（pass@n）
  # 均按 rollout step 记录；--log-reward-category 需 dict 型 reward（会破坏
  # dump 兼容性），失败模式统计走 metadata["error_kind"] 离线路径（R7-5）。
  if [[ "${USE_WANDB}" == "1" ]]; then
    TRAIN_ARGS+=(
      --use-wandb
      --wandb-mode "${WANDB_MODE}"
      --wandb-dir "${WANDB_DIR}"
      --wandb-project "${WANDB_PROJECT}"
      --wandb-group "rl_${ALGO}"
      --wandb-run-name "${RUN_ROOT##*/}"
      --disable-wandb-random-suffix
      --wandb-always-use-train-step
      --log-multi-turn
      --log-passrate
    )
    [[ -n "${WANDB_KEY}" ]] && TRAIN_ARGS+=(--wandb-key "${WANDB_KEY}")
    [[ -n "${WANDB_HOST}" ]] && TRAIN_ARGS+=(--wandb-host "${WANDB_HOST}")
    [[ -n "${WANDB_TEAM}" ]] && TRAIN_ARGS+=(--wandb-team "${WANDB_TEAM}")
  fi
  if [[ "${USE_TENSORBOARD}" == "1" ]]; then
    TRAIN_ARGS+=(
      --use-tensorboard
      --tb-project-name "${TENSORBOARD_DIR}"
      --tb-experiment-name "rl_${ALGO}"
    )
  fi

  {
    printf '%q ' "${SLIME_PYTHON}" -u train.py "${TRAIN_ARGS[@]}"
    printf '\n'
  } >"${RUN_ROOT}/train_command.txt"

  if [[ "${CHECK_ONLY:-0}" == "1" ]]; then
    printf 'ALGORITHM=%s\nESTIMATOR=%s\nPROMPT_DATA=%s\nDATA_ROWS=%s\nRUN_ROOT=%s\n' \
      "${ALGO}" "${ESTIMATOR}" "${PROMPT_DATA}" "${DATA_ROWS}" "${RUN_ROOT}"
    return 0
  fi

  # ── 启动前环境预检：env 服务/槽位/GPU，异常先修复、修不好则拒绝训练 ──
  # AF_UNIX socket 路径 ≤107 字符：temp_dir + session_<42 字符> + /sockets/
  # dash_MetricsHead 必须放得下，temp_dir 过长会让 dashboard 起不来、agent
  # 永远不就绪。放在 CHECK_ONLY
  # 之后：dry-run 只验证参数生成，不应被本地临时路径长度绑架。
  if [[ ${#RAY_TEMP_DIR} -gt 40 ]]; then
    echo "错误: RAY_TEMP_DIR 长度 ${#RAY_TEMP_DIR} > 40 字符，会导致 dashboard 无法启动：${RAY_TEMP_DIR}" >&2
    return 2
  fi
  ensure_env_ready || return 2
  check_gpu_free || return 2

  if RAY_ADDRESS="http://127.0.0.1:${RAY_DASHBOARD_PORT}" "${RAY_BIN}" status >/dev/null 2>&1; then
    echo "An existing Ray cluster is running; stop it first." >&2
    return 2
  fi

  RAY_STARTED=0
  cleanup() {
    stop_env_watchdog
    if (( RAY_STARTED == 1 )); then
      stop_own_ray
    fi
  }
  trap cleanup EXIT INT TERM

  # 多集群并行时还需隔离三类内部端口，否则 dashboard agent 会随机撞上已占端口，
  # 以 "address already in use" 静默失败 → job submit 500 / No available agent
  # （2026-09-22 三卡并行实测，错误只出现在 ray dashboard 内部日志里）。默认值
  # 保持单实例行为不变。
  if ! "${RAY_BIN}" start --head --node-ip-address "${MASTER_ADDR}" --num-gpus 1 \
    --port "${RAY_GCS_PORT}" \
    --disable-usage-stats --dashboard-host=127.0.0.1 --dashboard-port="${RAY_DASHBOARD_PORT}" \
    --dashboard-agent-listen-port "${RAY_DASHBOARD_AGENT_PORT:-52365}" \
    --min-worker-port "${RAY_MIN_WORKER_PORT:-10000}" \
    --max-worker-port "${RAY_MAX_WORKER_PORT:-19999}" \
    --metrics-export-port "${RAY_METRICS_EXPORT_PORT:-44217}" \
    --temp-dir "${RAY_TEMP_DIR}"; then
    echo "错误: ray start 失败（常见原因：GCS/dashboard/agent/worker 端口被其他实验占用，或 temp-dir ${RAY_TEMP_DIR} 有残留；并行时需错开全部端口段）" >&2
    return 2
  fi
  RAY_STARTED=1
  start_env_watchdog

  # 等待 dashboard/agent 就绪：ray start 返回只代表 head 进程已拉起，
  # dashboard agent 注册完成前 job submit 会 500（No available agent）。
  local _deadline=$(( SECONDS + 120 ))
  until curl -sf -m 3 "http://127.0.0.1:${RAY_DASHBOARD_PORT}/api/version" >/dev/null 2>&1; do
    if (( SECONDS >= _deadline )); then
      echo "错误: Ray dashboard 在 120s 内未就绪，放弃提交 ${ALGO}" >&2
      return 2
    fi
    sleep 3
  done

  RUNTIME_ENV_JSON="$("${SLIME_PYTHON}" -c 'import json, os; keys=("PYTHONPATH","PATH","CUDA_HOME","LD_LIBRARY_PATH","MASTER_ADDR","NO_PROXY","no_proxy","CUDA_VISIBLE_DEVICES","CUDA_DEVICE_MAX_CONNECTIONS","PYTORCH_CUDA_ALLOC_CONF","OMP_NUM_THREADS","SHOP_ENV_URL","SHOP_MAX_TURNS","SHOP_CONTEXT_KEEP_ACT_RESULTS","SHOP_CONTEXT_STRUCTURED_MEMORY","SHOP_ROLLOUT_TIMEOUT_SEC","SHOP_REQUIRE_NONZERO_VARIANCE_PER_ROLLOUT","SHOP_CAPTURE_ROLLOUT_EVENTS","SHOP_DECOMPOSED_ADVANTAGE_WEIGHT","SHOP_BEHAVIOR_DELTA","SHOP_REWARD_METRIC","WANDB_MODE","WANDB_API_KEY","WANDB_BASE_URL","TENSORBOARD_DIR","ADAPTER_PUBLIC_HOST","ADAPTER_BIND_HOST","ADAPTER_PORT","PI_BIN"); print(json.dumps({"env_vars": {key: os.environ[key] for key in keys if key in os.environ}}))')"

  cd "${SLIME_DIR}"
  # 提交失败（如 agent 未就绪的 500）必须传播出去：tee 会吞掉管道退出码，
  # 之前链式脚本据此把失败当成功继续跑后续算法。仅对"提交阶段"错误
  # （500 / No available agent）重试；job 已开始运行后的失败不重试（避免重跑）。
  local _submit_ok=0 _attempt _rc
  # 多实例并行时 dashboard agent 注册明显更慢（实测 3×20s 窗口不够），
  # 放宽到 20 次（最长约 6.7 分钟）；仅对"提交阶段"错误重试。
  for _attempt in $(seq 1 20); do
    set -o pipefail
    # --working-dir 必须显式指定：不指定时 job 的 cwd 取决于 ray head 进程
    # 的启动目录（随 chain 启动位置漂移），曾导致 can't open train.py。
    if "${RAY_BIN}" job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
      --working-dir "${SLIME_DIR}" \
      --runtime-env-json="${RUNTIME_ENV_JSON}" \
      -- "${SLIME_PYTHON}" -u train.py "${TRAIN_ARGS[@]}" 2>&1 | tee "${RUN_ROOT}/train.log"; then
      _submit_ok=1
    else
      # $? 必须在 else 分支内捕获：if 语句整体退出码恒为 0，
      # 在 fi 之后取值只会得到 0，把失败显示成成功。
      _rc=$?
    fi
    set +o pipefail
    if (( _submit_ok == 1 )); then
      break
    fi
    # 提交阶段的网关错误（500/504 等）视为 agent 未就绪，可重试；
    # job 已开始运行后的失败不重试（避免重跑整场训练）。
    if grep -qE "No available agent|status code 5[0-9][0-9]" "${RUN_ROOT}/train.log" 2>/dev/null; then
      echo "Ray agent 未就绪（第 ${_attempt}/20 次，code=${_rc}），20s 后重试..." >&2
      sleep 20
    else
      echo "错误: ${ALGO} ray job 运行失败 (code=${_rc})，不重试" >&2
      break
    fi
  done
  if (( _submit_ok != 1 )); then
    return 2
  fi

  trap - EXIT INT TERM
  cleanup
  LAST_TRAIN_ROOT="${RUN_ROOT}"
  say "训练 ${ALGO} 完成。HF exports: ${RUN_ROOT}/hf；checkpoints: ${RUN_ROOT}/checkpoints"
}

# ── 评测单个算法（$1=算法标签；自动选最新 rollout 导出）──────────
eval_one() {
  local ALGO="$1"
  # 与本次训练实际使用的 RUN_ROOT 对齐（支持用户显式 RUN_ROOT 的场景）
  local TRAIN_ROOT="${LAST_TRAIN_ROOT:-${RUNS_ROOT}/qwen35_2b_shop_rl_${ALGO}${RUN_TAG}}"
  local HF_DIR="${TRAIN_ROOT}/hf"
  local LAST_ID
  # || true 防 set -e+pipefail：HF 目录不存在/为空时 grep 无匹配会直接杀死
  # 整个脚本，而不是走下面的友好报错分支。
  LAST_ID="$(ls "${HF_DIR}" 2>/dev/null | { grep -oE '[0-9]+' || true; } | sort -n | tail -1)"
  if [[ -z "${LAST_ID}" ]]; then
    say "错误: ${HF_DIR} 下没有任何 rollout 导出，跳过评测 ${ALGO}"
    return 1
  fi
  say "评测 ${ALGO} 开始 (HF: rollout_${LAST_ID})"
  local EVAL_RC=0
  ( cd "${SLIME_DIR}" && env BASE_DIR="${BASE_DIR}" \
      MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX}" \
      SLIME_DIR="${SLIME_DIR}" \
      SLIME_PYTHON="${SLIME_PYTHON}" \
      MEGATRON_DIR="${MEGATRON_DIR}" \
      PI_BIN="${PI_BIN}" \
      EVAL_CHECKPOINT="${TRAIN_ROOT}/checkpoints" \
      HF_CHECKPOINT="${HF_DIR}/rollout_${LAST_ID}" \
      RUN_ROOT="${RUNS_ROOT}/eval_rl_${ALGO}${RUN_TAG}" \
      RAY_TEMP_DIR="${BASE_DIR}/ray/e_${ALGO}${RUN_TAG}" \
      ./examples/ShopSimulator/run_eval.sh > "${LOG_DIR}/eval_rl_${ALGO}${RUN_TAG}.log" 2>&1 ) || EVAL_RC=$?
  if [[ "${EVAL_RC}" != "0" ]]; then
    # 护栏：评测失败时严禁对可能存在的旧 rollout dump 做 summarize——
    # 那会把历史结果冒充本次输出。
    say "评测 ${ALGO} 失败 (code=${EVAL_RC})，跳过 summarize 以防误用旧结果"
    return "${EVAL_RC}"
  fi
  ( cd "${SLIME_DIR}" && "${SLIME_PYTHON}" -m examples.ShopSimulator.utils summarize \
      --run-root "${RUNS_ROOT}/eval_rl_${ALGO}${RUN_TAG}" \
      >> "${LOG_DIR}/eval_rl_${ALGO}${RUN_TAG}.log" 2>&1 ) || true
  say "评测 ${ALGO} 结束 (code=${EVAL_RC})，结果: ${RUNS_ROOT}/eval_rl_${ALGO}${RUN_TAG}/eval_results.json"
  return "${EVAL_RC}"
}

# ── 主流程 ────────────────────────────────────────────────────────
command -v nvidia-smi >/dev/null || { echo "nvidia-smi is required" >&2; exit 2; }
[[ -x "${RAY_BIN}" ]] || { echo "ray is required: ${RAY_BIN}" >&2; exit 2; }
# 关键路径预检：配错只会在 job **运行中**才暴露（rollout 起不来 / import 失败），
# 与 CUDA_HOME 同类隐蔽。这里提前 fail fast。
[[ -x "${SLIME_PYTHON}" ]] || { echo "错误: SLIME_PYTHON 不可执行: ${SLIME_PYTHON}" >&2; exit 2; }
[[ -d "${MEGATRON_DIR}" ]] || { echo "错误: MEGATRON_DIR 不存在: ${MEGATRON_DIR}" >&2; exit 2; }
# 注：PI_BIN 的可执行校验已在参数构建阶段完成（见文件上方 "pi is required"）。

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
export SHOP_CONTEXT_KEEP_ACT_RESULTS="${KEEP_ACT_RESULTS}"
# R5 结构化记忆：历史 shop_act 结果压成候选摘要行（默认 1；置 0 回退旧占位符）。
export SHOP_CONTEXT_STRUCTURED_MEMORY="${SHOP_CONTEXT_STRUCTURED_MEMORY:-1}"
export SHOP_ROLLOUT_TIMEOUT_SEC="${ROLLOUT_TIMEOUT_SEC}"
export SHOP_REQUIRE_NONZERO_VARIANCE_PER_ROLLOUT=0
# 默认开启采样轨迹落盘：事件流 + 每轮上下文快照会写入 rollout_dumps 的 metadata。
# 置 0 可关闭以缩小落盘体积。
export SHOP_CAPTURE_ROLLOUT_EVENTS="${SHOP_CAPTURE_ROLLOUT_EVENTS:-1}"
export ADAPTER_PUBLIC_HOST ADAPTER_BIND_HOST ADAPTER_PORT
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost,${MASTER_ADDR},${ADAPTER_PUBLIC_HOST}}"
export no_proxy="${no_proxy:-${NO_PROXY}}"
export PYTHONPATH="${MEGATRON_DIR}:${SLIME_DIR}:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
if [[ ! "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export OMP_NUM_THREADS=16
fi

if [[ "${ALGORITHM}" == "all" ]]; then
  if [[ -n "${USER_RUN_ROOT}" ]]; then
    say "all 模式按算法自动命名 RUN_ROOT，忽略显式传入的 RUN_ROOT=${USER_RUN_ROOT}"
  fi
  USER_RUN_ROOT=""
  declare -A STATUS
  for ALGO in "${ALL_ALGOS[@]}"; do
    say "════ 算法 ${ALGO}：训练开始 ════"
    if train_one "${ALGO}"; then
      if [[ "${AUTO_EVAL}" == "1" ]] && [[ "${CHECK_ONLY:-0}" != "1" ]]; then
        sleep 60   # 等 Ray/GPU 资源完全释放
        eval_one "${ALGO}" && STATUS[${ALGO}]="train+eval OK" || STATUS[${ALGO}]="train OK / eval FAIL"
      else
        STATUS[${ALGO}]="train OK (未评测)"
      fi
    else
      STATUS[${ALGO}]="train FAIL"
    fi
    # CHECK_ONLY=1 不得做任何 Ray 清理；正常结束时只清理本算法自己的集群
    if [[ "${CHECK_ONLY:-0}" != "1" ]]; then
      stop_own_ray
      sleep 60
    fi
  done
  say "════ 全部完成，汇总 ════"
  for ALGO in "${ALL_ALGOS[@]}"; do
    printf '  %-12s %s\n' "${ALGO}" "${STATUS[${ALGO}]}"
  done
else
  # 单算法模式：训练失败直接退出（此前会被吞掉、评测继续跑、chain 记 code=0）
  if ! train_one "${ALGORITHM}"; then
    say "训练 ${ALGORITHM} 失败（见上方错误与 ${LOG_DIR} 下日志）"
    exit 2
  fi
  if [[ "${AUTO_EVAL}" == "1" ]] && [[ "${CHECK_ONLY:-0}" != "1" ]]; then
    sleep 60
    eval_one "${ALGORITHM}" || exit 3
  fi
fi
