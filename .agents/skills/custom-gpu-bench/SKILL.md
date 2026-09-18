---
name: custom-gpu-bench
description: Add and run InferenceX benchmarks on a non-NVIDIA/non-AMD accelerator (MetaX C500, ZhenWu, Hygon, Intel Gaudi, Ascend, Moore Threads, ...) fully locally, without GitHub Actions or Slurm. Use when the user asks to add a new custom/vendor GPU to InferenceX, benchmark vllm/sglang on a vendor chip, probe a vendor SMI (mx-smi, npu-smi, xpu-smi) for power telemetry, run a local air-gapped sweep, or compare local results against published InferenceX baselines.
---

# Custom GPU Local Benchmarking

Runs the InferenceX fixed-sequence methodology on a vendor accelerator host: vendor
container serves, the host python is the benchmark client, and everything
(throughput, latencies, power) is normalized into canonical `agg_*.json` — no GHA,
no Slurm, no upload. MetaX C500 (mx-smi, vllm-metax) is the reference implementation
end-to-end; the checklist below generalizes to the next vendor.

## When to reach for this

- "Add <vendor> GPU to the benchmarks", "run InferenceX locally on <chip>",
  "benchmark vllm on our MetaX/ZhenWu/Hygon box", "compare our chip against
  NVIDIA/AMD baselines".
- Validating a new vendor image or model before it goes into
  `configs/runners.yaml` and the public sweeps.

## Reference layout

| Piece | File |
| --- | --- |
| Vendor telemetry (`metax` branch) | `benchmarks/benchmark_lib.sh` → `start_gpu_monitor` / `stop_gpu_monitor` |
| Single-point launcher (docker or existing server) | `benchmarks/single_node/fixed_seq_len/qwen3.8_int8_c500.sh` |
| Sweep orchestrator (conc loop, aggregation, comparison) | `benchmarks/local/run_local_sweep.sh` |
| Offline comparison vs baseline JSONs | `infx/results/local_compare.py` + `utils/test_local_compare.py` |
| Export published rows as baselines (internet once) | `infx/results/fetch_baselines.py` (+ `utils/test_fetch_baselines.py`) |
| Contributor guide (EN/ZH, indexed) | `docs/local-benchmarking.md` / `docs/local-benchmarking_zh.md` |

Results live under `~/inferencex-local-bench/results/<run>/`; host client venv is
`~/.venvs/inferencex-bench` (needs `aiohttp huggingface_hub numpy tqdm transformers`;
`fetch_baselines`/tests add `pytest pyyaml`).

## Adding a new vendor GPU

1. **Probe the vendor SMI on the host first** (as a normal user, not just root):
   - Find the tool (`mx-smi`, `npu-smi`, `xpu-smi`, ...) and a way to stream
     timestamped power/util to CSV. MetaX quirks: `mx-smi -o file` must be combined
     with a `--show-*` command, `-l` takes milliseconds, and `-t` is rejected
     together with `-o`.
   - If streaming exists: add a vendor branch to `start_gpu_monitor` in
     `benchmark_lib.sh` (mirror the `metax`/`amd` branches). If the vendor header is
     not understood by the power aggregator, stream to `<out>_raw.csv` and normalize
     at `stop_gpu_monitor` time to exactly `timestamp,index,power_w` — the generic
     `infx/results/power` pipeline then works unchanged (verify with one
     `integrate_power()` call, expect `power_valid=True`).
   - If streaming does not exist: leave `GPU_MONITOR_VENDOR=""` (throughput/latency
     stay 100% valid; power is best-effort) and note `REQUIRE_POWER=0` in the run.
2. **Launcher:** copy `qwen3.8_int8_c500.sh` to
   `<model>_<precision>_<sku>.sh`. It sources `benchmark_lib.sh`, checks
   `MODEL TP CONC ISL OSL RANDOM_RANGE_RATIO RESULT_FILENAME EP_SIZE`, starts the
   vendor container (or an existing server via `EXISTING_SERVER_PORT` +
   `OPENAI_API_KEY`), calls `wait_for_server_ready` + `run_benchmark_serving`, and
   `stop_gpu_monitor` before `exit "$BMK_EXIT"`. Vendor-specific knobs are env
   vars (`<VENDOR>_VLLM_IMAGE`, `<VENDOR>_GPU_MEMORY_UTILIZATION`,
   `<VENDOR>_EXTRA_VLLM_ARGS`) — keep the CI contract (`check_env_vars`,
   `start_gpu_monitor`, `run_benchmark_serving`) untouched.
3. **Sweep:** drive it with `benchmarks/local/run_local_sweep.sh`
   (`MODEL RUNNER_TYPE MODEL_PREFIX FRAMEWORK PRECISION TP ISL OSL IMAGE CONC_LIST`;
   optional `BASELINES_DIR` → automatic `local_compare` report). It exports the
   per-point `RESULT_FILENAME=bmk_<prefix>_<precision>_<runner>_c<conc>_gpus_<N>`,
   runs `infx.results.fixed_sequence` per point, then
   `infx.results.collect_results` over `agg_bmk_*.json` only.
4. **Upstream later (not local):** only when publishing — add the runner label to
   `configs/runners.yaml`, the recipe to the vendor master YAML, and a
   `perf-changelog.yaml` entry (append-only, byte-sensitive).

## Docker gotchas (learned on MetaX)

- GPU access is plain device binds, not a runtime: MetaX is
  `--device /dev/mxcd --device /dev/dri`. Check how the existing serving container
  on the host was started (`docker inspect ... HostConfig.Devices`) and copy it.
- Vendor vllm images often set `ENTRYPOINT [".../vllm"]`: launch with
  `--entrypoint ""` and a full `vllm serve ...` command, or you get
  `vllm vllm serve ...`.
- Forks may drop upstream flags (`--disable-log-requests` was rejected by
  vllm-metax 0.23); when a flag dies, put it behind
  `<VENDOR>_EXTRA_VLLM_ARGS` (a space-separated string env var) instead of hard
  coding it in the launcher.
- Engine startup can hang after "GPU KV cache size" while capturing CUDA graphs;
  `--enforce-eager` is the validated workaround for local runs.
- If another server already occupies the GPU, size
  `<VENDOR>_GPU_MEMORY_UTILIZATION` against the *free* memory
  (vllm: `free < gmu × total`), not the total.

## Bash gotcha

`VAR=()` clobbers an inherited string env var: read it into a `_STR` temp first,
then `read -r -a VAR <<< "$VAR_STR"`. This bit `METAX_EXTRA_VLLM_ARGS`.

## Validation recipe (before any big run)

1. Small local model first (e.g. Qwen3-0.6B via `hf download`), short sweep:
   `ISL=1024 OSL=512 CONC_LIST="1 2"`.
2. Assert per point: `benchmark_outcome.status == "passed"`, all numeric metrics
   present in `agg_*.json`, and (with telemetry) `power_validation_*.json`
   `power_valid == true` with `reasons == []`.
3. Run the comparison twice: against a self-baseline copy (deltas must be 0) and
   against the published baselines (`fetch_baselines` output) to exercise the
   no-match path. `local_compare` matches on
   (model prefix, framework, precision, spec decoding, disagg, isl, osl, conc) —
   pick ISL/OSL that exist in the published data (check
   `/api/v1/availability`) or the comparison is a no-match by construction.
   Export the baselines with `--model-prefix <the run's MODEL_PREFIX>` so the
   stamped `infmax_model_prefix` can actually equal the run's identity (the
   API's own model slug rarely does).
4. `uvx ruff check infx && uvx ruff format --check infx` plus the new/affected
   pytest files before declaring done.
