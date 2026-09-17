#!/usr/bin/env bash

# Fully local fixed-sequence sweep: no GitHub Actions, no Slurm, no result
# upload. Launches the server per point, benchmarks it, normalizes each point
# with infx.results.fixed_sequence, batch-aggregates with
# infx.results.collect_results, and (optionally) compares against baseline
# JSON files with infx.results.local_compare.
#
# Usage:
#   MODEL=/abs/path/to/weights \
#   RUNNER_TYPE=metax-c500 MODEL_PREFIX=qwen3.8 FRAMEWORK=vllm PRECISION=int8 \
#   TP=1 ISL=4096 OSL=2048 CONC_LIST="1 2 4 8 16 32 64" \
#   IMAGE=cr.metax-tech.com/.../vllm-metax:0.23.0-... \
#   bash benchmarks/local/run_local_sweep.sh

set -euo pipefail

LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$LOCAL_DIR/../.." && pwd)"
LAUNCHER="${LAUNCHER:-$REPO_ROOT/benchmarks/single_node/fixed_seq_len/qwen3.8_int8_c500.sh}"

for var in MODEL RUNNER_TYPE MODEL_PREFIX FRAMEWORK PRECISION TP ISL OSL IMAGE; do
    if [[ -z "${!var:-}" ]]; then
        echo "Error: $var is required" >&2
        exit 1
    fi
done

SPEC_DECODING="${SPEC_DECODING:-none}"
DISAGG="${DISAGG:-false}"
RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-0.0}"
CONC_LIST="${CONC_LIST:-1 2 4 8 16 32 64}"
EP_SIZE="${EP_SIZE:-1}"
DP_ATTENTION="${DP_ATTENTION:-false}"
PP_SIZE="${PP_SIZE:-1}"
DCP_SIZE="${DCP_SIZE:-1}"
PCP_SIZE="${PCP_SIZE:-1}"
STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_NAME="${RUN_NAME:-local_${MODEL_PREFIX}_${PRECISION}_${RUNNER_TYPE}_${STAMP}}"
RESULTS_DIR="${RESULTS_DIR:-${LOCAL_BENCH_HOME:-$HOME/inferencex-local-bench}/results/${RUN_NAME}}"
BASELINES_DIR="${BASELINES_DIR:-}"
# The launcher and the result processor run in child processes.
export SPEC_DECODING DISAGG RANDOM_RANGE_RATIO CONC_LIST EP_SIZE DP_ATTENTION
export PP_SIZE DCP_SIZE PCP_SIZE RUN_NAME RESULTS_DIR BASELINES_DIR

# The benchmark client and the result processors run on the host, so the
# host python (not the container's) needs the client dependencies.
python3 -c "import aiohttp, huggingface_hub, numpy, tqdm, transformers" 2>/dev/null || {
    echo "Error: host python lacks benchmark client deps (aiohttp, huggingface_hub, numpy, tqdm, transformers)." >&2
    echo "Create a venv with them and re-run with its bin/ first on PATH." >&2
    exit 1
}

mkdir -p "$RESULTS_DIR"
cd "$RESULTS_DIR"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export GPU_METRICS_CSV="$RESULTS_DIR/gpu_metrics.csv"
export RESULT_DIR="$RESULTS_DIR"

GPU_TOTAL=$((TP * PP_SIZE * PCP_SIZE))
echo "Run: $RUN_NAME"
echo "Launcher: $LAUNCHER"
echo "Results: $RESULTS_DIR"
echo "Concurrencies: $CONC_LIST"

FAILED_POINTS=()
for CONC in $CONC_LIST; do
    export CONC
    export RESULT_FILENAME="bmk_${MODEL_PREFIX}_${PRECISION}_${RUNNER_TYPE}_c${CONC}_gpus_${GPU_TOTAL}"
    export SERVER_LOG="$RESULTS_DIR/server_c${CONC}.log"
    echo "=== Point CONC=$CONC RESULT_FILENAME=$RESULT_FILENAME"
    if ! bash "$LAUNCHER"; then
        echo "=== Point CONC=$CONC failed" >&2
        FAILED_POINTS+=("$CONC")
        continue
    fi
    python3 -m infx.results.fixed_sequence ||
        echo "=== Aggregation failed for $RESULT_FILENAME" >&2
done

# Batch aggregation over the canonical per-point results only.
AGG_DIR="$RESULTS_DIR/agg"
mkdir -p "$AGG_DIR"
if ls agg_bmk_*.json >/dev/null 2>&1; then
    cp -f agg_bmk_*.json "$AGG_DIR/"
else
    echo "Warning: no per-point agg_bmk_*.json to aggregate" >&2
fi
python3 -m infx.results.collect_results "$AGG_DIR" "$RUN_NAME"

if [[ -n "${BASELINES_DIR:-}" ]]; then
    python3 -m infx.results.local_compare \
        --run "$RESULTS_DIR/agg_${RUN_NAME}.json" \
        --baselines "$BASELINES_DIR" \
        --out "$RESULTS_DIR/comparison.md"
    echo "Comparison: $RESULTS_DIR/comparison.md"
fi

if (( ${#FAILED_POINTS[@]} > 0 )); then
    echo "Completed with failed points: ${FAILED_POINTS[*]}" >&2
    exit 1
fi
echo "Sweep complete: $RESULTS_DIR"
