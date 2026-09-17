# Custom GPU Benchmark Execution and Integration Plan

## Context
The objective is to enable independent, isolated (air-gapped) benchmarking of LLM inference on alternative accelerator architectures (non-NVIDIA, non-AMD, e.g., MetaX, Intel Gaudi, Huawei Ascend, Moore Threads) using the InferenceX methodology. This allows producing 100% apples-to-apples comparable throughput, latency, and power metrics against published NVIDIA/AMD baselines, generating standard InferenceX JSON artifacts, and loading them into a local or upstream database without requiring GitHub Actions runners.

## Implementation status (MetaX C500, vllm, fully local)

The plan is implemented and validated end-to-end on a single MetaX C500 host
(vllm-metax 0.23.0 / MACA image). Everything runs locally; nothing is
uploaded to the InferenceX server.

- **Phases 2 & 3** — `benchmarks/benchmark_lib.sh` gained a `metax` branch in
  `start_gpu_monitor`/`stop_gpu_monitor`: streams `mx-smi --show-board-power`
  to a raw CSV, then normalizes it to `timestamp,index,power_w` so the generic
  power aggregator validates it (`power_valid=1`). The launcher
  `benchmarks/single_node/fixed_seq_len/qwen3.8_int8_c500.sh` either starts a
  dedicated docker container (`--device /dev/mxcd --device /dev/dri`,
  `--entrypoint ""`, image ENTRYPOINT is already `/opt/conda/bin/vllm`) or
  benchmarks an already-running server via `EXISTING_SERVER_PORT`
  (+ `OPENAI_API_KEY` for auth).
- **Phases 3–4** — `benchmarks/local/run_local_sweep.sh` sweeps `CONC_LIST`,
  normalizes each point with `python3 -m infx.results.fixed_sequence`, and
  batch-aggregates with `python3 -m infx.results.collect_results`. The
  benchmark client (host python) needs `aiohttp huggingface_hub numpy tqdm
  transformers` on `PATH` (see `~/.venvs/inferencex-bench`).
- **Phase 5** — baselines come from the public API export
  (`python3 -m infx.results.fetch_baselines --model <name> --isl .. --osl ..
  --out-dir <dir>`); offline comparison is
  `python3 -m infx.results.local_compare --run <agg.json> --baselines <dir>
  --out <report.md>`.
- **MetaX notes** — the vllm-metax fork rejects `--disable-log-requests`;
  pass extra flags via `METAX_EXTRA_VLLM_ARGS` (validation used
  `--enforce-eager`); when another server already occupies the GPU, lower
  `METAX_GPU_MEMORY_UTILIZATION` to fit the free memory.
- **Validated** — Qwen3-0.6B, ISL 1024 / OSL 512, conc 1 and 2: 10/10 and
  20/20 requests completed, `power_valid=1`, agg JSONs and a markdown
  comparison report were produced (see `~/inferencex-local-bench/results/`).

<details><summary>中文</summary>

MetaX C500 实现已完成并端到端验证：`benchmark_lib.sh` 自动识别 `mx-smi` 并归一化
功耗 CSV（`power_valid=1`）；`qwen3.8_int8_c500.sh` 在本机 docker 中启动 vllm
（或通过 `EXISTING_SERVER_PORT` 复用已有服务）；`run_local_sweep.sh` 本地完成
并发扫描、`fixed_sequence` 聚合与 `collect_results` 批量汇总；`local_compare.py`
与 `fetch_baselines.py` 支持离线对比已发布的 InferenceX 基线。

</details>

## Approach

### Phase 1: Isolated Environment & Container Image Preparation
Run entirely offline without GitHub Actions or internet access:
1. **Model Weights Staging:**
   - Pre-download model weights (e.g., `Qwen/Qwen2.5-72B-Instruct` or FP8 quantized checkpoints) to local storage.
   - Use absolute paths in the `$MODEL` environment variable (e.g. `/data/models/qwen2.5-72b-instruct`). Scripts in InferenceX check `[[ "$MODEL" != /* ]]` before invoking `hf download`; an absolute path bypasses any HuggingFace network requests.
2. **Container Image Selection:**
   - Vendor-specific runtime image is required because LLM serving depends on low-level device drivers and vendor kernel libraries (e.g. MetaX MACA, Huawei CANN, Intel Habana OneAPI).
   - Ensure the image contains an LLM serving engine implementing OpenAI-compatible REST API (`vllm serve` or vendor fork like `vllm-ascend`, `habana-ai/vllm`).
   - Package the image as a tarball (`docker save <image> -o image.tar`) and transfer it to the isolated host, then run `docker load -i image.tar`.
3. **InferenceX Codebase Mounting:**
   - Clone or copy this InferenceX repository onto the host and mount it into the container at `/workspace/InferenceX`.
   - Ensure container Python has required client dependencies: `pip install pydantic pandas datasets aiohttp requests`.

### Phase 2: Hardware Telemetry Adaptation (Optional Power Metrics)
1. **Extend GPU Monitoring in `benchmarks/benchmark_lib.sh`:**
   - Locate `start_gpu_monitor()` (lines 181-213). Currently detects `nvidia-smi` and `amd-smi`.
   - Add a vendor branch for the target accelerator (e.g., `mx-smi` for MetaX, `npu-smi` for Ascend, `xpu-smi` for Intel):
     ```bash
     elif command -v <vendor>-smi &>/dev/null; then
         GPU_MONITOR_VENDOR="<vendor>"
         <vendor>-smi ... > "$output" &
         GPU_MONITOR_PID=$!
     ```
   - If vendor power telemetry is skipped or unavailable, `benchmark_lib.sh` cleanly defaults `GPU_MONITOR_VENDOR=""` and skips power logging without failing the throughput benchmark.

### Phase 3: Standalone Benchmark Script Formulation
Create a dedicated standalone benchmark runner `benchmarks/single_node/fixed_seq_len/<model>_<precision>_<custom_hw>.sh`:
1. **Source Library:** `source "$(dirname "$0")/../../benchmark_lib.sh"`.
2. **Define Fixed-Sequence Parameters:**
   - Input Sequence Length: `ISL=4096`
   - Output Sequence Length: `OSL=2048`
   - Random variation: `RANDOM_RANGE_RATIO=0.0`
   - Concurrency sweep list: `CONC_LIST="1 2 4 8 16 32 64"`
   - Tensor Parallelism: `TP=<device_count>`
   - Result prefix: `RESULT_FILENAME="bmk_${MODEL_PREFIX}_${PRECISION}_${CUSTOM_HW}_c${CONC}_gpus_${TP}"`
3. **Launch Engine Server:**
   - Execute the vendor serving command in background redirecting to `$SERVER_LOG`:
     ```bash
     vllm serve "$MODEL" --host 0.0.0.0 --port "$PORT" \
       --tensor-parallel-size "$TP" \
       --max-model-len 8192 \
       --gpu-memory-utilization 0.92 \
       --no-enable-prefix-caching > "$SERVER_LOG" 2>&1 &
     SERVER_PID=$!
     ```
   - Call `wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"`.
4. **Execute Standard Load Client:**
   - Invoke `run_benchmark_serving` (which calls `infx.bench_serving.benchmark_serving`):
     ```bash
     run_benchmark_serving \
       --model "$MODEL" \
       --port "$PORT" \
       --backend vllm \
       --input-len "$ISL" \
       --output-len "$OSL" \
       --random-range-ratio "$RANDOM_RANGE_RATIO" \
       --num-prompts "$((CONC * 10))" \
       --max-concurrency "$CONC" \
       --result-filename "$RESULT_FILENAME" \
       --result-dir /workspace/results/ \
       --trust-remote-code
     ```
   - Ensure `--request-rate inf` and `--ignore-eos` remain active (default inside `benchmark_lib.sh`).
5. **Enforce Speculative Decoding Parity (if MTP/Draft model evaluated):**
   - For speculative runs, do not rely on prompt luck. Match the Synthetic Acceptance Length from `golden_al_distribution/*.yaml` for the model family.

### Phase 4: Output Processing into Canonical InferenceX JSON
1. **Transform Raw Benchmark Output:**
   - Execute `infx.results.fixed_sequence`:
     ```bash
     export RUNNER_TYPE="<custom_hw>"
     export FRAMEWORK="vllm"
     export PRECISION="fp8"
     export SPEC_DECODING="none"
     export DISAGG="false"
     export MODEL_PREFIX="qwen3.5"
     export IMAGE="<vendor_image_tag>"
     export ISL=4096
     export OSL=2048
     export TP=<device_count>
     export EP_SIZE=1
     export DP_ATTENTION=false
     export RESULT_FILENAME="$RESULT_FILENAME"
     
     python3 -m infx.results.fixed_sequence
     ```
   - This parses the raw client output, validates request outcomes, and generates `agg_<RESULT_FILENAME>.json`.
2. **Batch Aggregation:**
   - Run `python3 -m infx.results.collect_results /workspace/results/ <custom_run_name>` to combine all concurrency points into `agg_<custom_run_name>.json`.

### Phase 5: Result Comparison and Ingestion
1. **Local Comparison Mode (Air-gapped / Private):**
   - Load `agg_<custom_run_name>.json` alongside benchmark JSONs extracted from InferenceX.
   - Compare key standardized metrics:
     - `tput_per_gpu` (Total tokens/sec divided by active accelerator count).
     - `output_tput_per_gpu` (Generated tokens/sec per accelerator).
     - `ttft_median`, `ttft_p90`, `ttft_p99` (Time to First Token in seconds).
     - `tpot_median`, `tpot_p90` (Time Per Output Token in seconds).
     - `intvty_median` ($1000 / \text{TPOT}$ in tokens/sec/user).
     - `joules_per_token` / `tokens_per_joule` (if power monitoring was enabled).
2. **Upstream Contribution to Public InferenceX Database:**
   - GitHub Actions is required **only** for official upstream merge to SemiAnalysis public dashboard, because SemiAnalysis CI verifies reproducibility on trusted physical clusters:
     - Define runner labels in `configs/runners.yaml`.
     - Add model recipe under `configs/<vendor>-master.yaml`.
     - Append entry to `perf-changelog.yaml` referencing the PR.
     - CI triggers `run-sweep.yml`, uploads `results_bmk` artifacts, and dispatches to `InferenceX-app` for database ingestion.
   - For private clusters without GitHub access, export `agg_*.json` into any local PostgreSQL/ClickHouse/SQLite database conforming to the `InferenceX-app/packages/db` schema.

## Critical Files & Anchors
- `benchmarks/benchmark_lib.sh`: `run_benchmark_serving()` (lines 576-811) and `start_gpu_monitor()` (lines 165-213) contain the exact client flags, port polling, and telemetry collectors.
- `infx/bench_serving/benchmark_serving.py`: `sample_random_requests()` (lines 100-250) and main benchmark client logic executing the synthetic workload.
- `infx/results/fixed_sequence.py`: `build_result()` (lines 33-178) and `process_result()` (lines 301-328) normalize raw outputs into `agg_*.json`.
- `configs/runners.yaml`: Schema of hardware runner descriptors and DRAM capacity definitions.
- `golden_al_distribution/README.md`: Reference specification for synthetic acceptance length during speculative decoding benchmarks.

## Verification
1. **Isolated Dry Run Test:**
   - Execute a 1-prompt sanity run on local GPU:
     ```bash
     python3 -m infx.bench_serving.benchmark_serving \
       --model /data/models/qwen2.5-72b-instruct \
       --backend vllm \
       --base-url "http://0.0.0.0:8000" \
       --dataset-name random \
       --random-input-len 128 \
       --random-output-len 64 \
       --num-prompts 2 \
       --max-concurrency 1 \
       --save-result \
       --result-filename test_run.json
     ```
   - Verify `test_run.json` contains valid numeric `total_token_throughput`, `mean_ttft_ms`, and `mean_tpot_ms`.
2. **Schema & Result Normalization Test:**
   - Run `python3 -m infx.results.fixed_sequence` with required environment variables.
   - Verify generated `agg_test_run.json` contains `tput_per_gpu`, `hw`, `conc`, and normalized latencies.

## Assumptions & Contingencies
- **Assumed API:** The non-NVIDIA/non-AMD GPU serving engine provides an OpenAI-compatible HTTP interface (`/v1/completions` or `/v1/chat/completions`). If the engine is proprietary without OpenAI API, write a lightweight FastAPI adapter forwarding to the proprietary runtime.
- **Power Monitoring:** If vendor SMI does not support CSV streaming or energy accumulators, run without power metrics (`REQUIRE_POWER=0`); throughput and latency metrics remain 100% valid and comparable.
- **Speculative Decoding:** If testing non-speculative baseline, set `SPEC_DECODING=none`. If speculative decoding is tested, ensure the engine supports synthetic acceptance length override, or compare only non-speculative baselines for direct hardware parity.
