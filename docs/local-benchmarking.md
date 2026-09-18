<div align="center">

**English** | [中文](./local-benchmarking_zh.md)

</div>

# Local Benchmarking on Custom Accelerators

Run the InferenceX fixed-sequence methodology fully locally on a vendor
accelerator host — no GitHub Actions, no Slurm, no upload to the InferenceX
server. The vendor container only serves the OpenAI-compatible API; the host
python is the benchmark client; every artifact (throughput, latencies, power)
is normalized into the canonical `agg_*.json` shape, and results can be
compared offline against published InferenceX baselines.

Validated on: 1x MetaX C500 (65 GB), image
`cr.metax-tech.com/public-ai-release/maca/vllm-metax:0.23.0-maca.ai3.8.0.103-torch2.10-py312-ubuntu22.04-amd64`,
MACA 3.8.1.3, `mx-smi` 2.3.4. MetaX is the reference implementation; the
workflow generalizes to the next vendor (Ascend `npu-smi`, Gaudi `hpu-smi`,
Moore Threads `mtsmi`, ...) by adding a telemetry branch to
`benchmarks/benchmark_lib.sh` and a launcher. The agent skill
[`.agents/skills/custom-gpu-bench/SKILL.md`](../.agents/skills/custom-gpu-bench/SKILL.md)
holds the per-vendor checklist.

| File | Role |
| --- | --- |
| `benchmarks/local/run_local_sweep.sh` | Orchestrator: concurrency sweep + aggregation + comparison |
| `benchmarks/single_node/fixed_seq_len/qwen3.8_int8_c500.sh` | Launcher: starts vLLM in docker (or hits a live server) and runs the client |
| `benchmarks/benchmark_lib.sh` | Shared library: GPU monitoring (`metax`/`mx-smi` branch), client run, readiness polling |
| `infx/results/fixed_sequence.py` | Normalizes the raw client result into `agg_*.json` (canonical format) |
| `infx/results/collect_results.py` | Bundles all points into one `agg_<run>.json` (list) |
| `infx/results/local_compare.py` | Offline comparison of a run against baseline files |
| `infx/results/fetch_baselines.py` | One-shot export of published InferenceX results from the public API into baseline JSON |

## 1. One-time environment setup

### 1.1 Host python client

The benchmark client (`python3 -m infx.bench_serving.benchmark_serving`) and
all result processors run **on the host**, not in the container. Create a
venv:

```bash
python3 -m venv ~/.venvs/inferencex-bench
~/.venvs/inferencex-bench/bin/pip install \
    "huggingface_hub[cli]" transformers aiohttp tqdm numpy requests pytest pyyaml
```

Everywhere below, `python3` means this venv (first on `PATH`).

### 1.2 Model weights

An absolute host path in `MODEL` bypasses HuggingFace entirely
(`[[ "$MODEL" != /* ]]` in the launcher). Download the weights by any
convenient means, e.g.:

```bash
~/.venvs/inferencex-bench/bin/hf download Qwen/Qwen3-0.6B \
    --local-dir /home/$USER/metax-vllm/models/Qwen3-0.6B
```

### 1.3 Baselines for comparison (needs internet, once)

On any internet-connected machine, export published InferenceX results into
local JSON files:

```bash
cd InferenceX
# Frontend model name; list them at /api/v1/availability
python3 -m infx.results.fetch_baselines \
    --model "Qwen-3.5-397B-A17B" \
    --model-prefix qwen3.5 \
    --isl 8192 --osl 1024 \
    --out-dir ~/inferencex-local-bench/baselines
```

The exporter writes one file per (hardware, framework, precision)
combination — `<model>_<isl>x<osl>_<hw>_<fw>_<prec>.json` — containing a
list of points in the canonical format, sorted by concurrency. Subset with
repeated `--hardware <hw>` / `--framework <fw>` flags. `--model-prefix`
stamps the `infmax_model_prefix` identity the local run must use to match
these baselines (without it the exporter uses the row's own model slug);
query the public model names at
`https://inferencex.semianalysis.com/api/v1/availability`.

## 2. Running the sweep

### 2.1 Command

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

For every value in `CONC_LIST` the orchestrator:
1. runs the launcher (it starts the vLLM docker container with the needed
   flags, waits for `/health`, runs the benchmark client, removes the
   container);
2. normalizes the point: `python3 -m infx.results.fixed_sequence`;
3. at the end bundles `agg_bmk_*.json` into `agg_<RUN_NAME>.json`
   (`infx.results.collect_results`) and, if `BASELINES_DIR` is set, writes
   `comparison.md` (`infx.results.local_compare`).

A launcher failure or an aggregation failure records that concurrency in the
failed-points set and the sweep exits 1; a reused `RESULTS_DIR` never leaks
stale points (previous `agg_bmk_*.json` files and the `agg/` batch directory
are dropped before the run).

Everything lands in `~/inferencex-local-bench/results/<RUN_NAME>/`
(`RESULTS_DIR`/`LOCAL_BENCH_HOME` override the location).

### 2.2 Orchestrator parameters (required)

| Variable | Description |
| --- | --- |
| `MODEL` | Absolute host path to the weights (mounted read-only into the container at the same path) |
| `RUNNER_TYPE` | Hardware name for the result, e.g. `metax-c500` (lands in the `hw` field) |
| `MODEL_PREFIX` | Model prefix for baseline matching (must equal `infmax_model_prefix` in the baselines, e.g. `qwen3.5`) |
| `FRAMEWORK` | Engine: `vllm` / `sglang` / `trt` |
| `PRECISION` | Precision: `int8` / `fp8` / `bf16` / `fp4` ... |
| `TP` | Tensor parallelism (accelerators per node) |
| `ISL`, `OSL` | Fixed input/output sequence lengths (synthetic load; `RANDOM_RANGE_RATIO=0.0` = no variation) |
| `IMAGE` | Vendor engine image tag — the result `image` provenance in **both** modes. Docker-launch mode runs it; existing-server mode records it (use the tag the running server was started from) |

### 2.3 Orchestrator parameters (optional)

| Variable | Default | Description |
| --- | --- | --- |
| `CONC_LIST` | `1 2 4 8 16 32 64` | Concurrency points (each point = a separate server run) |
| `RANDOM_RANGE_RATIO` | `0.0` | Fraction of random length variation (0 = strictly fixed) |
| `SPEC_DECODING` | `none` | Speculative decoding method (result field; compare equal values for apples-to-apples) |
| `DISAGG` | `false` | Disaggregated prefill/decode |
| `EP_SIZE`, `DP_ATTENTION`, `PP_SIZE`, `DCP_SIZE`, `PCP_SIZE` | `1/false/1/1/1` | Parallelism topology (lands in the agg JSON and is validated) |
| `RUN_NAME` | `local_<prefix>_<prec>_<runner>_<stamp>` | Run name (file and directory names) |
| `RESULTS_DIR` | `~/inferencex-local-bench/results/$RUN_NAME` | Where to write results |
| `BASELINES_DIR` | (empty) | Baseline JSON directory for `comparison.md` |
| `LAUNCHER` | `.../fixed_seq_len/qwen3.8_int8_c500.sh` | Path to the launcher (yours is fine) |
| `EXISTING_SERVER_PORT`, `OPENAI_API_KEY` | (empty) | "Server already running" mode — see §2.5 |
| `PORT` | `8888` | Server port (container mapping) |

### 2.4 Launcher parameters (passed through as-is)

| Variable | Default | Description |
| --- | --- | --- |
| `METAX_VLLM_IMAGE` | caller's `IMAGE` | Image tag for the docker run (explicit override only) |
| `METAX_GPU_MEMORY_UTILIZATION` | `0.92` | `--gpu-memory-utilization`. If another server already occupies the GPU, lower it to a fraction of the *free* memory (vLLM check: free >= GMU x total) |
| `METAX_MAX_NUM_SEQS` | (none) | `--max-num-seqs` (useful to cap when total KV capacity is small) |
| `METAX_EXTRA_VLLM_ARGS` | (empty) | Arbitrary vLLM flags, e.g. `--enforce-eager` (on MetaX the engine can stall without it — see §6) |
| `METAX_CONTAINER_NAME` | `inferencex-bench-<pid>` | Container name |
| `SERVER_LOG` | `<RESULTS_DIR>/server_c<conc>.log` | Server log |
| `RUN_EVAL` | (not true) | Run lm-eval after the benchmark |

Fixed server flags in the launcher: `--host 0.0.0.0 --port 8888
--tensor-parallel-size $TP --max-model-len $((ISL+OSL+20)) --dtype bfloat16
--trust-remote-code --no-enable-prefix-caching`. Client: `--request-rate inf`,
`--ignore-eos`, `--num-warmups 2xCONC`, `--num-prompts 10xCONC`,
`--percentile-metrics ttft,tpot,itl,e2el`.

### 2.5 "Benchmark an already-running server" mode

If vLLM already serves the model (e.g. your working container), benchmark it
without starting a new one:

```bash
# OPENAI_API_KEY is only needed if the server was started with --api-key.
# IMAGE is still required: it records the engine-image provenance of the
# result (use the tag the running server was started from).
EXISTING_SERVER_PORT=8011 \
OPENAI_API_KEY=<your-api-key> \
IMAGE=<tag-the-running-server-was-started-from> \
MODEL=<path-to-weights-for-the-tokenizer> \
bash benchmarks/local/run_local_sweep.sh
```

Readiness polls `/health` on the port; set `EXISTING_SERVER_PID` to also
supervise the server process, otherwise the wait is bounded by
`EXISTING_SERVER_TIMEOUT` (default 3600 s). Note: server flags (prefix
caching, MTP, max-model-len, max-num-seqs) affect the numbers — this mode is
only comparable to InferenceX recipes when the server configuration matches
the reference one.

## 3. Generated results

Layout of `~/inferencex-local-bench/results/<RUN_NAME>/`:

```
bmk_<prefix>_<prec>_<runner>_c<conc>_gpus_<tp>.json   # raw client result (per point)
agg_bmk_...c<conc>_gpus_<tp>.json                     # canonical point (per point)
agg_local_....json                                     # list of all points (collect_results)
gpu_metrics.csv                                        # mx-smi telemetry (timestamp,index,power_w)
gpu_metrics_identity.txt                               # mx-smi snapshot (versions/devices)
power_validation_...c<conc>_gpus_<tp>.json             # per-point energy audit
server_c<conc>.log                                     # per-point server log
agg/                                                   # copies of agg_bmk_*.json (for collect)
comparison.md                                          # when BASELINES_DIR is set
```

### 3.1 Raw JSON (`bmk_*.json`)

What the client writes: `duration`, `benchmark_start_time_unix` /
`benchmark_end_time_unix` (the window for power integration), `completed`,
`total_input_tokens`, `total_output_tokens`, `request_throughput`,
`output_throughput`, `total_token_throughput`, `mean/median/std_ttft_ms`,
`p90/p99/p99.9_ttft_ms`, likewise for `tpot`, `itl`, `e2el`,
`max_concurrency`, `model_id`, `tokenizer_id`, `benchmark_outcome`
(`status: passed|failed`, `requested`, `completed`, `failed`,
`max_failure_rate`).

### 3.2 Canonical JSON (`agg_bmk_*.json`) — the main artifact

Configuration fields (results are keyed on these):

| Field | Example | Meaning |
| --- | --- | --- |
| `hw` | `metax-c500` | Hardware (`RUNNER_TYPE`) |
| `framework`, `precision`, `spec_decoding`, `disagg` | `vllm`, `int8`, `none`, `false` | Configuration |
| `isl`, `osl`, `conc` | `8192`, `1024`, `16` | Load point |
| `tp`, `ep`, `pp`, `dcp_size`, `pcp_size`, `dp_attention` | `1`, `1`, ... | Topology |
| `image` | `cr.metax-tech.com/...` | Engine image |
| `infmax_model_prefix` | `qwen3.8` | Model identity for matching |
| `benchmark_outcome.status` | `passed` | Gate on the failed-request share (5%) |

Metrics (all latencies in **seconds**; tokens/s are per GPU):

| Field | Meaning | Better |
| --- | --- | --- |
| `tput_per_gpu` | (input+output) tokens/s per GPU | higher |
| `output_tput_per_gpu` | generated tokens/s per GPU | higher |
| `input_tput_per_gpu` | input tokens/s per GPU | higher |
| `median_ttft`, `p99_ttft` | Time-to-First-Token, median/p99 | lower |
| `median_tpot`, `p99_tpot` | Time-Per-Output-Token | lower |
| `median_intvty`, `p99_intvty` | 1000/TPOT — tokens/s per user (inverted TPOT) | higher |
| `mean_e2el` and others | End-to-end latency | lower |
| `power_valid` | 1 — power telemetry passed validation, 0 — not (see sidecar) | 1 |
| `avg_power_w`, `avg_total_gpu_power_w` | Mean power, 1 GPU / whole node, W | lower |
| `p75_power_w`, `p90_power_w`, `p75/p90_total_gpu_power_w` | Power percentiles | lower |
| `total_gpu_energy_j` | Integrated energy over the benchmark window, J | lower |
| `joules_per_input_token`, `joules_per_output_token`, `joules_per_total_token`, `joules_per_successful_query` | Energy per token/request, J | lower |

`power_validation_*.json` — the full audit: benchmark window, sample count
per GPU, max sampling gap, invalidity reasons (empty list = valid), per-GPU
integrals.

### 3.3 `agg_<RUN_NAME>.json`

The array of all points of the run (output of `collect_results`). This is the
format `local_compare` consumes and the format a local database can ingest
per the `InferenceX-app/packages/db` schema when historical analytics are
needed.

## 4. Inspecting results

Quick per-point scan:

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

What to look at:
- `benchmark_outcome.status == "passed"` in every point;
- `power_valid == 1` (otherwise the reasons are in `power_validation_*.json`);
- `tput_per_gpu` grows with `conc` until saturation and `median_tpot` climbs
  slowly — the normal curve shape;
- no vLLM errors in `server_c*.log`, sane `non-default args`;
- `gpu_metrics.csv` is continuous (max gap < 3 s, otherwise power is marked
  `sampling_gap_exceeded`).

## 5. Comparing against other measurements

### 5.1 Offline comparison (no network, no database)

```bash
cd InferenceX
python3 -m infx.results.local_compare \
    --run ~/inferencex-local-bench/results/<RUN_NAME>/agg_<RUN_NAME>.json \
    --baselines ~/inferencex-local-bench/baselines \
    --out ~/inferencex-local-bench/results/<RUN_NAME>/comparison.md
```

`comparison.md` — one markdown table per point: the run's values, each
baseline's values, and `Δ` (absolute) and `Δ%` (relative to the baseline) for
the §3.2 metrics. The run column header and every baseline column label show
the hardware and parallelism (e.g. `metax-c500 tp1` vs `h100 tp8`), because
`tp` is deliberately not a matching key — throughput metrics are per-GPU, so
a 1x accelerator is comparable against multi-GPU baselines. Points are
matched on `(model prefix, framework, precision, spec_decoding, disagg, isl,
osl, conc)`. Exit codes: `0` — at least one match, `1` — soft "nothing
matched" (model/point differ), `2` — argument/IO error.

For the comparison to actually match, the run must share with the baselines:
`MODEL_PREFIX` (= the `infmax_model_prefix` the baselines were exported
with), `FRAMEWORK`, `PRECISION`, `ISL`/`OSL` (hence ISL=8192/OSL=1024 in
§2.1 — the published points), `SPEC_DECODING=none`.

### 5.2 "Run against itself" / two local runs

Any two files in the same format work as baselines:

```bash
mkdir -p ~/inferencex-local-bench/baselines_self
cp <old_run>/agg_bmk_*.json ~/inferencex-local-bench/baselines_self/
python3 -m infx.results.local_compare \
    --run <new_run>/agg_<run>.json \
    --baselines ~/inferencex-local-bench/baselines_self \
    --name "new-run" --out <new_run>/comparison.md
```

Typical scenarios: A/B a server flag (`--enforce-eager` vs graphs), change
the image, change `max-num-seqs`.

### 5.3 Alternative: the official public-database comparison

If the official public dashboard is ever needed, results land in
`InferenceX-app` (PostgreSQL) through CI — that requires a PR (recipe in
`configs/*-master.yaml`, `runners.yaml`, a `perf-changelog.yaml` entry), see
[`results-and-ingestion.md`](./results-and-ingestion.md). The local
`local_compare` + `fetch_baselines` give the same comparative effect without
CI.

## 6. Known pitfalls (MetaX / vllm-metax 0.23.0)

1. **`--enforce-eager`**: without it the engine core can stall (99% CPU, no
   logs) after `Maximum concurrency for ...`. Check `/health` and, when it
   stalls, add `METAX_EXTRA_VLLM_ARGS="--enforce-eager"`. On the production
   MTP container graphs work — the stall showed up on short max-model-len,
   but re-verify per configuration.
2. **`--disable-log-requests`** is not accepted by the fork — do not add it.
3. **Memory**: vLLM requires `free >= GMU x total`. If another server is
   alive on the GPU (e.g. a working 27B occupying ~59 GB of 65), set
   `METAX_GPU_MEMORY_UTILIZATION` below the free fraction (validation used
   0.07).
4. **Image ENTRYPOINT** is `/opt/conda/bin/vllm`; the launcher passes
   `--entrypoint ""` plus the full `vllm serve ...` command, not a bare
   `serve`.
5. **GPU devices**: the container needs `--device /dev/mxcd --device
   /dev/dri` (runc, no nvidia runtime).
6. **Client dependencies** live on the host, not in the container — the
   §1.1 venv must be first on `PATH` (the orchestrator exits with a hint
   when an import is missing).
7. **`mx-smi`** streams CSV only to a file (`-o`); `-t` is incompatible with
   `-o`; the interval is in milliseconds (`-l <ms>`). Normalization of the
   raw stream to `timestamp,index,power_w` happens in `stop_gpu_monitor`
   (`metax` branch of `benchmarks/benchmark_lib.sh`).
8. The CI `nodes:N` label invariant does not apply to local runs (no
   Slurm), but when a point moves into a CI recipe — see `AGENTS.md`.

## 7. Smoke check command

Minimal smoke on a simple model (1–2 min per point):

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

Expected outcome (reference, C500 with a busy GPU): 10/10 and 20/20
requests, `power_valid=1`, ~184 total tok/s per GPU at c1 and ~281 at c2,
mean power 85–90 W.
