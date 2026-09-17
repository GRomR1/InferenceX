<div align="center">

[English](./local-benchmarking.md) | **中文**

</div>

# 自定义加速卡的本地基准测试

在厂商加速卡主机上完全本地地运行 InferenceX 固定序列方法学 —— 无 GitHub
Actions、无 Slurm、不上传 InferenceX 服务器。厂商容器只负责提供
OpenAI 兼容 API；宿主 python 是基准测试客户端；所有产物（吞吐、延迟、功耗）
都归一化为规范 `agg_*.json` 格式，并且可以与已发布的 InferenceX 基线离线对比。

已验证环境：1x MetaX C500（65 GB），镜像
`cr.metax-tech.com/public-ai-release/maca/vllm-metax:0.23.0-maca.ai3.8.0.103-torch2.10-py312-ubuntu22.04-amd64`，
MACA 3.8.1.3，`mx-smi` 2.3.4。MetaX 是参考实现；该工作流可推广到下一个厂商
（Ascend `npu-smi`、Gaudi `hpu-smi`、Moore Threads `mtsmi` 等）：在
`benchmarks/benchmark_lib.sh` 中新增遥测分支并添加 launcher 即可。逐厂商清单见
agent 技能
[`.agents/skills/custom-gpu-bench/SKILL.md`](../.agents/skills/custom-gpu-bench/SKILL.md)。

| 文件 | 角色 |
| --- | --- |
| `benchmarks/local/run_local_sweep.sh` | 编排器：并发扫描 + 聚合 + 对比 |
| `benchmarks/single_node/fixed_seq_len/qwen3.8_int8_c500.sh` | Launcher：docker 启动 vLLM（或复用已有服务）并运行客户端 |
| `benchmarks/benchmark_lib.sh` | 公共库：GPU 监控（`metax`/`mx-smi` 分支）、客户端执行、就绪轮询 |
| `infx/results/fixed_sequence.py` | 把原始客户端结果归一化为 `agg_*.json`（规范格式） |
| `infx/results/collect_results.py` | 把所有点打包为一个 `agg_<run>.json`（列表） |
| `infx/results/local_compare.py` | run 对 baseline 文件的离线对比 |
| `infx/results/fetch_baselines.py` | 一次性从公开 API 导出已发布结果到 baseline JSON |

## 1. 一次性环境准备

### 1.1 宿主 python 客户端

基准测试客户端（`python3 -m infx.bench_serving.benchmark_serving`）和所有结果
处理器都**在宿主**执行，而不是容器内。创建 venv：

```bash
python3 -m venv ~/.venvs/inferencex-bench
~/.venvs/inferencex-bench/bin/pip install \
    "huggingface_hub[cli]" transformers aiohttp tqdm numpy requests pytest pyyaml
```

下文所有 `python3` 均指该 venv（置于 `PATH` 最前）。

### 1.2 模型权重

`MODEL` 使用宿主绝对路径即可完全绕过 HuggingFace
（launcher 中的 `[[ "$MODEL" != /* ]]`）。权重可用任意方式下载，例如：

```bash
~/.venvs/inferencex-bench/bin/hf download Qwen/Qwen3-0.6B \
    --local-dir /home/$USER/metax-vllm/models/Qwen3-0.6B
```

### 1.3 对比用基线（需联网，一次性）

在任意联网机器上，把已发布的 InferenceX 结果导出为本地 JSON 文件：

```bash
cd InferenceX
# 前端模型名；可用 /api/v1/availability 查询
python3 -m infx.results.fetch_baselines \
    --model "Qwen-3.5-397B-A17B" \
    --model-prefix qwen3.5 \
    --isl 8192 --osl 1024 \
    --out-dir ~/inferencex-local-bench/baselines
```

导出器按 (hardware, framework, precision) 组合各写一个文件 ——
`<model>_<isl>x<osl>_<hw>_<fw>_<prec>.json` —— 内容为规范格式的点列表，
按并发排序。可用可重复的 `--hardware <hw>` / `--framework <fw>` 参数取子集。
`--model-prefix` 决定导出点写入的 `infmax_model_prefix` 身份（本地 run 必须用
同一值才能匹配；缺省使用行自身的 model slug）；公开模型名查询
`https://inferencex.semianalysis.com/api/v1/availability`。

## 2. 运行 sweep

### 2.1 命令

```bash
export PATH=~/.venvs/inferencex-bench/bin:$PATH
cd InferenceX

MODEL=/home/$USER/metax-vllm/models/Qwen3.8-27B-Uncensored-INT8 \
RUNNER_TYPE=metax-c500 MODEL_PREFIX=qwen3.8 FRAMEWORK=vllm PRECISION=int8 \
TP=1 ISL=8192 OSL=1024 RANDOM_RANGE_RATIO=0.0 CONC_LIST="4 8 16 32 64" \
IMAGE=cr.metax-tech.com/public-ai-release/maca/vllm-metax:0.23.0-maca.ai3.8.0.103-torch2.10-py312-ubuntu22.04-amd64 \
BASELINES_DIR=~/inferencex-local-bench/baselines \
bash benchmarks/local/run_local_sweep.sh
```

对 `CONC_LIST` 的每个取值，编排器：
1. 运行 launcher（拉起带所需参数的 vLLM docker 容器，等待 `/health`，运行基准
   客户端，删除容器）；
2. 归一化该点：`python3 -m infx.results.fixed_sequence`；
3. 收尾时把 `agg_bmk_*.json` 汇总进 `agg_<RUN_NAME>.json`
   （`infx.results.collect_results`），并在设置 `BASELINES_DIR` 时写出
   `comparison.md`（`infx.results.local_compare`）。

launcher 失败或聚合失败都会把该并发记入失败点集合，sweep 以 1 退出；复用的
`RESULTS_DIR` 不会泄漏陈旧点位（运行前删除上一次的 `agg_bmk_*.json` 与 `agg/`
批量目录）。

全部产物位于 `~/inferencex-local-bench/results/<RUN_NAME>/`
（`RESULTS_DIR`/`LOCAL_BENCH_HOME` 可改位置）。

### 2.2 编排器参数（必填）

| 变量 | 说明 |
| --- | --- |
| `MODEL` | 权重的宿主绝对路径（以只读方式挂载到容器同路径） |
| `RUNNER_TYPE` | 结果中的硬件名，如 `metax-c500`（写入 `hw` 字段） |
| `MODEL_PREFIX` | 模型前缀，用于基线匹配（必须等于基线中的 `infmax_model_prefix`，如 `qwen3.5`） |
| `FRAMEWORK` | 引擎：`vllm` / `sglang` / `trt` |
| `PRECISION` | 精度：`int8` / `fp8` / `bf16` / `fp4` … |
| `TP` | 张量并行（每节点加速卡数） |
| `ISL`、`OSL` | 固定输入/输出序列长度（合成负载；`RANDOM_RANGE_RATIO=0.0` = 无变差） |
| `IMAGE` | 厂商引擎镜像 tag（写入结果 `image` 字段并用于 docker 启动） |

### 2.3 编排器参数（可选）

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `CONC_LIST` | `1 2 4 8 16 32 64` | 并发点列表（每个点 = 一次独立服务器运行） |
| `RANDOM_RANGE_RATIO` | `0.0` | 长度随机变差比例（0 = 严格固定） |
| `SPEC_DECODING` | `none` | 投机解码方法（结果字段；apples-to-apples 对比请保持一致值） |
| `DISAGG` | `false` | 分离式 prefill/decode |
| `EP_SIZE`、`DP_ATTENTION`、`PP_SIZE`、`DCP_SIZE`、`PCP_SIZE` | `1/false/1/1/1` | 并行拓扑（写入 agg JSON 并参与校验） |
| `RUN_NAME` | `local_<prefix>_<prec>_<runner>_<stamp>` | run 名（文件与目录名） |
| `RESULTS_DIR` | `~/inferencex-local-bench/results/$RUN_NAME` | 结果写入位置 |
| `BASELINES_DIR` | (空) | `comparison.md` 用的 baseline JSON 目录 |
| `LAUNCHER` | `.../fixed_seq_len/qwen3.8_int8_c500.sh` | launcher 路径（可用自己的） |
| `EXISTING_SERVER_PORT`、`OPENAI_API_KEY` | (空) | 「服务已在运行」模式 —— 见 §2.5 |
| `PORT` | `8888` | 服务器端口（容器映射） |

### 2.4 Launcher 参数（原样透传）

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `METAX_VLLM_IMAGE` | 调用方 `IMAGE` | docker 启动的镜像 tag（仅作显式覆盖） |
| `METAX_GPU_MEMORY_UTILIZATION` | `0.92` | `--gpu-memory-utilization`。若 GPU 上已有其他服务，降到*空闲*内存的份额（vLLM 检查：空闲 >= GMU x 总量） |
| `METAX_MAX_NUM_SEQS` | (无) | `--max-num-seqs`（KV 总量较小时建议限流） |
| `METAX_EXTRA_VLLM_ARGS` | (空) | 任意 vLLM 参数，如 `--enforce-eager`（MetaX 上不加可能卡死 —— 见 §6） |
| `METAX_CONTAINER_NAME` | `inferencex-bench-<pid>` | 容器名 |
| `SERVER_LOG` | `<RESULTS_DIR>/server_c<conc>.log` | 服务器日志 |
| `RUN_EVAL` | (非 true) | 基准测试后运行 lm-eval |

Launcher 内固定服务器参数：`--host 0.0.0.0 --port 8888
--tensor-parallel-size $TP --max-model-len $((ISL+OSL+20)) --dtype bfloat16
--trust-remote-code --no-enable-prefix-caching`。客户端：`--request-rate inf`、
`--ignore-eos`、`--num-warmups 2xCONC`、`--num-prompts 10xCONC`、
`--percentile-metrics ttft,tpot,itl,e2el`。

### 2.5 「基准测试已在运行的服务」模式

若 vLLM 已在服务该模型（例如你的工作容器），可直接对它测试而不新起服务：

```bash
# OPENAI_API_KEY 仅在服务器带 --api-key 启动时需要
EXISTING_SERVER_PORT=8011 \
OPENAI_API_KEY=opencode-key \
MODEL=<tokenizer 用的权重路径> \
bash benchmarks/local/run_local_sweep.sh
```

就绪检查轮询该端口的 `/health`；设置 `EXISTING_SERVER_PID` 可同时监管服务器
进程，否则等待受 `EXISTING_SERVER_TIMEOUT`（默认 3600 秒）约束。注意：服务器
参数（prefix caching、MTP、max-model-len、max-num-seqs）会影响数字 —— 只有当
服务器配置与参考一致时，该模式才能与 InferenceX recipe 对比。

## 3. 生成的结果

`~/inferencex-local-bench/results/<RUN_NAME>/` 的布局：

```
bmk_<prefix>_<prec>_<runner>_c<conc>_gpus_<tp>.json   # 原始客户端结果（逐点）
agg_bmk_...c<conc>_gpus_<tp>.json                     # 规范点（逐点）
agg_local_....json                                     # 全部点列表（collect_results）
gpu_metrics.csv                                        # mx-smi 遥测（timestamp,index,power_w）
gpu_metrics_identity.txt                               # mx-smi 快照（版本/设备）
power_validation_...c<conc>_gpus_<tp>.json             # 逐点能耗审计
server_c<conc>.log                                     # 逐点服务器日志
agg/                                                   # agg_bmk_*.json 副本（供 collect）
comparison.md                                          # 设置 BASELINES_DIR 时生成
```

### 3.1 原始 JSON（`bmk_*.json`）

客户端写入：`duration`、`benchmark_start_time_unix` /
`benchmark_end_time_unix`（功耗积分窗口）、`completed`、`total_input_tokens`、
`total_output_tokens`、`request_throughput`、`output_throughput`、
`total_token_throughput`、`mean/median/std_ttft_ms`、`p90/p99/p99.9_ttft_ms`，
`tpot`、`itl`、`e2el` 同理，`max_concurrency`、`model_id`、`tokenizer_id`、
`benchmark_outcome`（`status: passed|failed`、`requested`、`completed`、
`failed`、`max_failure_rate`）。

### 3.2 规范 JSON（`agg_bmk_*.json`）—— 主要产物

配置字段（结果以其为键）：

| 字段 | 示例 | 含义 |
| --- | --- | --- |
| `hw` | `metax-c500` | 硬件（`RUNNER_TYPE`） |
| `framework`、`precision`、`spec_decoding`、`disagg` | `vllm`、`int8`、`none`、`false` | 配置 |
| `isl`、`osl`、`conc` | `8192`、`1024`、`16` | 负载点 |
| `tp`、`ep`、`pp`、`dcp_size`、`pcp_size`、`dp_attention` | `1`、`1`、… | 拓扑 |
| `image` | `cr.metax-tech.com/...` | 引擎镜像 |
| `infmax_model_prefix` | `qwen3.8` | 匹配用的模型身份 |
| `benchmark_outcome.status` | `passed` | 失败请求比例（5%）门槛 |

指标（所有延迟均为**秒**；tok/s 均为每 GPU）：

| 字段 | 含义 | 更优 |
| --- | --- | --- |
| `tput_per_gpu` | （输入+输出）每 GPU 每秒令牌 | 越高 |
| `output_tput_per_gpu` | 每 GPU 生成令牌/秒 | 越高 |
| `input_tput_per_gpu` | 每 GPU 输入令牌/秒 | 越高 |
| `median_ttft`、`p99_ttft` | 首令牌时间，中位/p99 | 越低 |
| `median_tpot`、`p99_tpot` | 每输出令牌时间 | 越低 |
| `median_intvty`、`p99_intvty` | 1000/TPOT —— 每用户令牌/秒（TPOT 倒数） | 越高 |
| `mean_e2el` 等 | 端到端延迟 | 更低 |
| `power_valid` | 1 —— 功耗遥测通过校验，0 —— 未通过（见 sidecar） | 1 |
| `avg_power_w`、`avg_total_gpu_power_w` | 平均功耗，单 GPU / 整机，W | 更低 |
| `p75_power_w`、`p90_power_w`、`p75/p90_total_gpu_power_w` | 功耗分位 | 更低 |
| `total_gpu_energy_j` | 基准窗口内积分能量，J | 更低 |
| `joules_per_input_token`、`joules_per_output_token`、`joules_per_total_token`、`joules_per_successful_query` | 每令牌/每请求能耗，J | 更低 |

`power_validation_*.json` —— 完整审计：基准窗口、每 GPU 样本数、最大采样间隔、
无效原因（空列表 = 有效）、每 GPU 积分。

### 3.3 `agg_<RUN_NAME>.json`

run 全部点的数组（`collect_results` 输出）。这是 `local_compare` 消费的格式，
也是需要历史分析时可按 `InferenceX-app/packages/db` schema 载入本地数据库的
格式。

## 4. 查看结果

快速逐点扫描：

```bash
cd ~/inferencex-local-bench/results/<RUN_NAME>
python3 - <<'PY'
import json, glob
for f in sorted(glob.glob("agg_bmk_*.json")):
    d = json.load(open(f))
    print(d["conc"], d["hw"], d["tput_per_gpu"], d["output_tput_per_gpu"],
          d["median_ttft"], d["median_tpot"], d["power_valid"])
PY
```

检查要点：
- 每个点 `benchmark_outcome.status == "passed"`；
- `power_valid == 1`（否则原因见 `power_validation_*.json`）；
- `tput_per_gpu` 随 `conc` 增长至饱和、`median_tpot` 缓慢上升 —— 正常曲线
  形态；
- `server_c*.log` 无 vLLM 错误、`non-default args` 合理；
- `gpu_metrics.csv` 连续（最大间隔 < 3 秒，否则功耗标记为
  `sampling_gap_exceeded`）。

## 5. 与其他测量对比

### 5.1 离线对比（无网络、无数据库）

```bash
cd InferenceX
python3 -m infx.results.local_compare \
    --run ~/inferencex-local-bench/results/<RUN_NAME>/agg_<RUN_NAME>.json \
    --baselines ~/inferencex-local-bench/baselines \
    --out ~/inferencex-local-bench/results/<RUN_NAME>/comparison.md
```

`comparison.md` —— 每个点一张 markdown 表：run 值、各 baseline 值，以及 §3.2
指标的 `Δ`（绝对）与 `Δ%`（相对 baseline）。run 列头与每个 baseline 列标签都
显示硬件与并行度（如 `metax-c500 tp1` 对 `h100 tp8`）—— `tp` 刻意不作为匹配
键：吞吐指标按每 GPU 计，因此 1 卡加速卡可以与多卡基线对比。点按
`(model prefix, framework, precision, spec_decoding, disagg, isl, osl, conc)`
匹配。退出码：`0` —— 有匹配，`1` —— 软性「无匹配」（模型/点不同），`2` ——
参数/IO 错误。

要让对比真正匹配，run 必须与基线共享：`MODEL_PREFIX`（= 基线导出时的
`infmax_model_prefix`）、`FRAMEWORK`、`PRECISION`、`ISL`/`OSL`（因此 §2.1 用
ISL=8192/OSL=1024 —— 已发布点）、`SPEC_DECODING=none`。

### 5.2 「run 对自身」/ 两个本地 run

同格式任意两个文件都可作 baseline：

```bash
mkdir -p ~/inferencex-local-bench/baselines_self
cp <old_run>/agg_bmk_*.json ~/inferencex-local-bench/baselines_self/
python3 -m infx.results.local_compare \
    --run <new_run>/agg_<run>.json \
    --baselines ~/inferencex-local-bench/baselines_self \
    --name "new-run" --out <new_run>/comparison.md
```

典型场景：A/B 服务器参数（`--enforce-eager` 对图模式）、换镜像、换
`max-num-seqs`。

### 5.3 替代：官方公开数据库对比

若日后需要官方公开仪表盘：结果经 CI 进入 `InferenceX-app`（PostgreSQL）——
那需要 PR（`configs/*-master.yaml` 中的 recipe、`runners.yaml`、
`perf-changelog.yaml` 记录），见 `docs/results-and-ingestion.md`。本地
`local_compare` 与 `fetch_baselines` 在不走 CI 的情况下提供同样的对比效果。

## 6. 已知坑（MetaX / vllm-metax 0.23.0）

1. **`--enforce-eager`**：不加时 EngineCore 可能在 `Maximum concurrency for ...`
   之后卡死（99% CPU、无日志）。请检查 `health`，卡死时加
   `METAX_EXTRA_VLLM_ARGS="--enforce-eager"`。MTP 生产容器上图模式可用 —— 该坑
   出现在短 max-model-len，但每套配置都应复核。
2. **`--disable-log-requests`** 该 fork 不接受 —— 不要加。
3. **显存**：vLLM 要求 `空闲 >= GMU × 总量`。若 GPU 上有其他服务（如工作 27B
   占用 65 GB 中的约 59 GB），把 `METAX_GPU_MEMORY_UTILIZATION` 降到空闲份额
   （验证时用的 0.07）。
4. **镜像 ENTRYPOINT** 是 `/opt/conda/bin/vllm`；launcher 传 `--entrypoint ""`
   和完整 `vllm serve ...`，而不是裸的 `serve`。
5. **GPU 设备**：容器需要 `--device /dev/mxcd --device /dev/dri`
   （runc，无 nvidia 运行时）。
6. **客户端依赖**在宿主而非容器内 —— §1.1 的 venv 必须置于 `PATH` 最前
   （缺依赖时编排器自行报错退出并给出提示）。
7. **`mx-smi`** 只能写文件（`-o`），`-t` 与 `-o` 不兼容；间隔为毫秒
   （`-l <ms>`）。raw CSV 到 `timestamp,index,power_w` 的归一化在
   `stop_gpu_monitor`（`metax` 分支）中完成。
8. CI 的 `nodes:N` label 不变量对本地 run 不适用（不用 Slurm）；点位迁移到
   CI recipe 时见 AGENTS.md。

## 7. 快速冒烟命令（smoke）

在简单模型上的最小冒烟运行（每点 1–2 分钟）：

```bash
export PATH=~/.venvs/inferencex-bench/bin:$PATH
cd InferenceX
MODEL=$HOME/metax-vllm/models/Qwen3-0.6B \
RUNNER_TYPE=metax-c500 MODEL_PREFIX=qwen3 FRAMEWORK=vllm PRECISION=bfloat16 \
TP=1 ISL=1024 OSL=512 RANDOM_RANGE_RATIO=0.0 CONC_LIST="1 2" \
IMAGE=cr.metax-tech.com/public-ai-release/maca/vllm-metax:0.23.0-maca.ai3.8.0.103-torch2.10-py312-ubuntu22.04-amd64 \
METAX_GPU_MEMORY_UTILIZATION=0.07 METAX_MAX_NUM_SEQS=8 \
METAX_EXTRA_VLLM_ARGS="--enforce-eager" \
bash benchmarks/local/run_local_sweep.sh
```

预期结果（参考值，GPU 被占用的 C500）：10/10 与 20/20 请求、
`power_valid=1`、c1 点约 184 total tok/s/GPU、c2 约 281；平均功耗 85–90 W。
