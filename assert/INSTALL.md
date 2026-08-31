# 📦 安装与配置指南

本文件说明如何从 clone 本仓库开始，完成环境初始化与配置，以便运行 [README.md](../README.md)「快速开始」中的四个实验。实验本身的运行命令（教师采集 → SFT → GRPO → 评测）见 README.md。

## 环境要求

训练与评测需要 **Linux + NVIDIA GPU**（CUDA 12.9、PyTorch 2.11、SGLang、Megatron-LM）。macOS / Windows 无法运行训练流程，本地无 GPU 环境仅可浏览代码与结果。

当前启动配置面向单机单卡 NVIDIA GPU；历史运行使用 84 GB 显存的 Pro 6000D，训练侧 `MAX_TOKENS_PER_GPU=12288`。更小显存配置需要重新调整 token budget、batch size 和 SGLang 显存比例，本仓库尚未验证。

多机/多卡扩展相关变量（当前单卡流程无需设置，本仓库未验证多机组合）：`shop_env_capacity`（`config/shop_rl.json` 的 `concurrency` 段，必须与 ShopSimulator 环境池预创建数量一致且能被 `rollout_num_engines` 整除）、`ADAPTER_PUBLIC_HOST` / `ADAPTER_BIND_HOST` / `ADAPTER_PORT`（`run_rl.sh` / `run_eval.sh`，多 worker 时需按进程分配不同端口）。

### 0. 克隆仓库并定义路径

```bash
git clone https://github.com/Piucente/pi-slime-shopsimulator.git
cd pi-slime-shopsimulator

export PROJECT_ROOT="$PWD"
export THIRD_PARTY_ROOT=/absolute/path/to/pi-slime-work
export SLIME_DIR="$PROJECT_ROOT/slime"
export MAMBA_ROOT_PREFIX="$THIRD_PARTY_ROOT/micromamba"
export MAMBA_EXE=/root/.local/bin/micromamba
export BASE_DIR="$THIRD_PARTY_ROOT"

mkdir -p "$THIRD_PARTY_ROOT"
```

`THIRD_PARTY_ROOT` 用于放置 micromamba、SGLang、Megatron-LM 和 ShopSimulator；不要把这些运行环境目录提交到本仓库。非 root 用户应把 `MAMBA_EXE` 改为自己的 micromamba 安装位置。

### 1. 安装 Pi 与 Slime 训练环境

先准备 Node.js `>=22.19.0`，再安装本项目使用的 Pi 版本：

```bash
node --version
npm --version
npm install --global @earendil-works/pi-coding-agent@0.84.2

export PI_BIN="$(command -v pi)"
"$PI_BIN" --version
```

然后运行修改后的 Slime 安装脚本。它会创建或复用名为 `slime` 的 micromamba 环境，在 `THIRD_PARTY_ROOT` 下检出脚本固定的 SGLang 与 Megatron-LM revision，并安装 CUDA 12.9、PyTorch 2.11 和相应依赖：

```bash
export SLIME_DIR="$PROJECT_ROOT/slime"
export BASE_DIR="$THIRD_PARTY_ROOT"
export MAMBA_ROOT_PREFIX="$THIRD_PARTY_ROOT/micromamba"
export MAMBA_EXE=/root/.local/bin/micromamba

bash "$SLIME_DIR/build_conda.sh"

export SLIME_PYTHON="$MAMBA_ROOT_PREFIX/envs/slime/bin/python"
export MEGATRON_DIR="$THIRD_PARTY_ROOT/Megatron-LM"

"$SLIME_PYTHON" -c 'import ray, sglang, torch; print(torch.__version__, torch.version.cuda)'
```

`build_conda.sh` 会下载依赖、编译 CUDA 扩展并修改它检出的 SGLang/Megatron-LM 工作树，耗时较长。后续命令都应继续使用这里的 `SLIME_PYTHON` 和 `MEGATRON_DIR`。

### 2. 获取、打补丁并启动 ShopSimulator

```bash
export SHOP_SIM_DIR="$THIRD_PARTY_ROOT/ShopSimulator"

git clone https://github.com/ShopAgent-Team/ShopSimulator.git "$SHOP_SIM_DIR"
git -C "$SHOP_SIM_DIR" checkout 51bb26012cee31aea7ac26177c5ffe807026ac07
git -C "$SHOP_SIM_DIR" apply --check "$PROJECT_ROOT/assert/shopsimulator-slime-integration.patch"
git -C "$SHOP_SIM_DIR" apply "$PROJECT_ROOT/assert/shopsimulator-slime-integration.patch"

"$SLIME_PYTHON" -m pip install -r "$SHOP_SIM_DIR/shop_env/requirements.runtime.txt"
```

在单独终端启动服务，并在教师采集、GRPO 和评测期间保持运行：

```bash
cd "$SHOP_SIM_DIR/shop_env"
"$SLIME_PYTHON" shop_env/pack_api.py
```

在另一个终端确认 20-slot 环境池可用：

```bash
curl -sS -X POST http://127.0.0.1:5000/api/shop_agent \
  -H 'content-type: application/json' \
  --data '{"action":"status"}'
```

### 3. 准备 Qwen3.5-0.8B 的两种 checkpoint

四个实验使用同一份基础模型。先下载 [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B)，再转换出训练侧使用的 Megatron `torch_dist` checkpoint：

```bash
mkdir -p "$THIRD_PARTY_ROOT/models"
export HF_CLI="$MAMBA_ROOT_PREFIX/envs/slime/bin/hf"
export BASE_HF_CHECKPOINT="$THIRD_PARTY_ROOT/models/Qwen3.5-0.8B"
export BASE_MEGATRON_CHECKPOINT="$THIRD_PARTY_ROOT/models/Qwen3.5-0.8B_torch_dist"

"$HF_CLI" download Qwen/Qwen3.5-0.8B --local-dir "$BASE_HF_CHECKPOINT"

cd "$SLIME_DIR"
source scripts/models/qwen3.5-0.8B.sh
PYTHONPATH="$MEGATRON_DIR:$SLIME_DIR" "$SLIME_PYTHON" tools/convert_hf_to_torch_dist.py "${MODEL_ARGS[@]}" --hf-checkpoint "$BASE_HF_CHECKPOINT" --save "$BASE_MEGATRON_CHECKPOINT"
```

HF checkpoint 提供 tokenizer 和 SGLang rollout 权重，Megatron checkpoint 提供训练权重；SFT、RL 和评测启动器需要成对传入匹配的两种格式。

---

## 🔧 与上游的差异

### ShopSimulator 相对基线的补丁

本仓库不复制或重新分发 ShopSimulator 源码，只提供一个相对于指定上游提交的补丁，用于构建与 Slime 集成的环境。

- 上游仓库：<https://github.com/ShopAgent-Team/ShopSimulator>
- 固定补丁基线：`51bb26012cee31aea7ac26177c5ffe807026ac07`
- 补丁对应的测试版本：`3ab366b2982e9ffa59957086d0845f955ef2245b`
- 补丁文件：[`assert/shopsimulator-slime-integration.patch`](shopsimulator-slime-integration.patch)
- SHA-256：`c75973456391f8f7784e0823bde2eae5cd6f4ae519dae51d43ed5c8b86c19dac`

补丁是上述两个提交之间的完整 Git diff，包含 50 个文件的变化。除功能修改外，它还删除了上游提交中误跟踪的 Python 缓存和运行日志，因此体积约为 382 KiB。

#### 并发 rollout 与会话隔离

- 将 Flask API 改为线程化服务，并为环境池和单个环境增加锁。
- 启动时预创建 20 个环境；这些环境共享只读的 `SimServer` 数据，避免重复加载商品和目标数据。
- 为每条 rollout 引入独立的 `rollout_session_id`，将任务索引 `idx` 与会话标识分离。
- `reset` 分配空闲环境，`interact` 校验环境和会话是否匹配，任务结束后自动释放环境和会话状态。
- 完善 `release_one`、`release_all`，并增加 `status`，可查看容量、空闲环境和活动会话。
- 环境分配后发生异常时自动回收资源，避免环境池泄漏。

这些修改用于防止 Slime 并发采样时不同 rollout 共享或覆盖交互状态。

#### 可复现的价格与价格约束

- 商品价格不再依赖进程级全局随机状态，而是根据商品 ASIN 确定性生成。
- 目标中的 `price_upper` 根据 ASIN 和 instruction 确定性生成。
- 同一任务在不同服务进程、环境实例和多次启动中使用相同价格数据，避免 `r_price` 与 `r_hard` 因服务端随机性漂移。

#### 文本环境适配

- 修正自定义字符串会话下任务索引的传递，确保目标仍由请求中的 `idx` 选择。
- 增加单会话状态释放接口，只清理该 rollout 的可变状态。
- 文本 API 的空图像占位改用 NumPy，去除该路径对 Torch 的非必要依赖。

#### 运行依赖与仓库清理

- 新增 `shop_env/requirements.runtime.txt`，记录本项目使用的 Python 运行依赖版本。
- 新增 `.gitignore`，忽略 Python 缓存、日志、PID 和本地编辑器状态。
- 从版本控制快照中删除已提交的 `__pycache__`、`.pyc` 和 `shop_agent.log` 等运行产物。

#### 补丁校验与使用

完整的 ShopSimulator 获取、基线检出、补丁校验/应用、运行依赖安装和服务启动命令见上文“第 2 步 获取、打补丁并启动 ShopSimulator”。应用前请同时核对上述基线 revision 和补丁 SHA-256；不要在其他 ShopSimulator 版本上强制应用。

#### 许可边界

该补丁只描述本项目对指定 ShopSimulator 快照所做的差异，不包含完整上游源码，也不替代或变更上游项目的许可条款。使用者应自行查看上游仓库当前的许可与使用条件，并确保其获取、使用和分发行为拥有相应授权。

### Slime 相对固定官方基线的修改

本节固定以本项目当时 fork Slime 时使用的官方提交为比较基准，不跟随 Slime 上游后续更新：

- 官方仓库：<https://github.com/THUDM/slime>
- 固定官方基线：`624b824a898ab0ec1fcb4d373004c7f3852bf515`（2026-08-21，`[NFC] Add observability subfolder (#2298)`）
- 本仓库中的修改后源码：[`slime/`](../slime/)

后续即使官方仓库发生更新，也不应在未重新审查兼容性和冲突的情况下替换上述基线。下面列出的内容均指当前 `slime/` 相对该固定提交的修改。

#### 新增 ShopSimulator 实验示例

固定官方基线中没有 `examples/ShopSimulator/`。本项目新增了该目录及以下能力：

- 新增 `pi_harness.py`、`shop_extension.ts`、`generate.py` 和 `common.py`，将 Pi 多轮工具调用、ShopSimulator HTTP API、Slime rollout 和 reward 计算连接起来。
- 新增 `collect_sft.py` 与 `prepare_sft.py`，用于采集教师轨迹并转换为 turn-level SFT 数据。
- 新增 `run_sft.sh`，提供 Qwen3.5-0.8B 的 SFT 训练入口。
- 新增 `run_rl.sh` 与 `config/shop_rl.json`，提供基于 ShopSimulator 在线 rollout 的 GRPO 训练入口。
- 新增 `run_eval.sh`、`config/shop_eval_official_k1.yaml` 与 `summarize_eval.py`，提供单次 rollout 评测及指标汇总入口。
- 新增 `data/tasks_v2/` 下的 `sft`、`rl`、`dev`、`official_test` 四个任务池，以及实验入口使用的 `sft_512`、`rl_500` 和 `official_test_200` 数据文件。

#### Qwen3.5-0.8B 与 checkpoint 兼容

- 新增 `scripts/models/qwen3.5-0.8B.sh`，补充 Qwen3.5-0.8B 在 Megatron 中使用的模型结构参数。
- 调整 `tools/convert_hf_to_torch_dist.py`：仅在 Megatron 参数解析器尚未注册时添加 `--use-gated-attention` 和 `--padded-vocab-size`，避免新版本 Megatron 因重复参数定义退出，同时保持旧版本兼容。

#### Qwen3.5 loss mask 与空 think 块

- 修正 `slime/utils/mask_utils.py` 对 Qwen3.5 chat template 的处理。
- thinking 关闭时，模板注入的完整空块 `<think>\n\n</think>\n\n` 被视为 prompt，不参与 loss。
- thinking 开启时，只屏蔽模板注入的 `<think>\n` 前缀；模型生成的 reasoning 内容继续参与训练。
- 该修改不改变 token 序列，只修正训练 mask 的起点，并继续检查文本 tokenization 与 `apply_chat_template(..., tokenize=True)` 的结果一致。

#### 多轮 adapter 的终止状态

- 在 `slime/agent/adapters/common.py` 中暴露每个 session 已完成并写入 trajectory 的真实模型 turn 数。
- 为 turn 上限增加结构化的 `turn_limit` 终止原因，供 ShopSimulator rollout 区分正常达到上限与其他 429 或基础设施异常。
- session 打开、完成或丢弃时清理 turn 计数与终止原因，避免复用 session id 时残留旧状态。

#### 安装与测试兼容

- 调整 `build_conda.sh`，支持在无交互 AutoDL 会话中直接初始化或复用 micromamba，避免依赖 shell 启动脚本，并可复用已存在的 `slime` 环境。
- 更新 adapter 测试，验证真实 turn 数、`turn_limit` 原因及 session 清理。
- CPU-only agent rollout 测试仅在本机确实没有安装 `transformers` 时注入 stub，避免覆盖已经可用的真实包。
