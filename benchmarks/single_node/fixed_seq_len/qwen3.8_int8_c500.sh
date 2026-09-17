#!/usr/bin/env bash

# Qwen3.8-27B W8A8 (INT8) on MetaX C500 via vllm (MACA), single node, TP1.
#
# Local (air-gapped) mode: sources benchmark_lib.sh and either launches a
# dedicated docker container for the server (default) or benchmarks an
# already-running server when EXISTING_SERVER_PORT is set. No Slurm, no GHA.
#
# Required environment: MODEL (absolute host path), TP, CONC, ISL, OSL,
# RANDOM_RANGE_RATIO, RESULT_FILENAME, EP_SIZE. Docker-launch mode additionally
# requires IMAGE (the vendor engine image tag recorded in the results).
# Optional: PORT (default 8888), RESULT_DIR, SERVER_LOG, EXISTING_SERVER_PORT,
# EXISTING_SERVER_PID (supervise an external server; unset means poll /health
# only, bounded by EXISTING_SERVER_TIMEOUT, default 3600), OPENAI_API_KEY
# (auth for an external server), METAX_VLLM_IMAGE (docker image override;
# defaults to the caller-supplied IMAGE), METAX_GPU_MEMORY_UTILIZATION
# (default 0.92), METAX_MAX_NUM_SEQS, METAX_EXTRA_VLLM_ARGS,
# METAX_CONTAINER_NAME.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    CONC \
    ISL \
    OSL \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME \
    EP_SIZE

if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi

SERVER_LOG="${SERVER_LOG:-$(pwd)/server.log}"
RESULT_DIR="${RESULT_DIR:-$(pwd)}"
PORT="${PORT:-8888}"
MAX_SEQ_LEN=$((ISL + OSL + 20))

mx-smi 2>/dev/null || true

echo "TP: $TP, EP_SIZE: $EP_SIZE, CONC: $CONC, ISL: $ISL, OSL: $OSL, MAX_SEQ_LEN: $MAX_SEQ_LEN"

start_gpu_monitor

# Idempotent cleanup for both the explicit stop below and the failure paths
# (server death or readiness timeout exits the script before it would
# otherwise reach stop_gpu_monitor). stop_gpu_monitor is itself a no-op once
# the stream has been stopped.
METAX_CONTAINER_STARTED=0
METAX_CONTAINER_NAME="${METAX_CONTAINER_NAME:-inferencex-bench-$BASHPID}"
_on_exit() {
    set +e
    stop_gpu_monitor
    if [[ "$METAX_CONTAINER_STARTED" -eq 1 ]]; then
        docker rm -f "$METAX_CONTAINER_NAME" >/dev/null 2>&1
    fi
    return 0
}
trap '_on_exit' EXIT

METAX_GPU_MEMORY_UTILIZATION="${METAX_GPU_MEMORY_UTILIZATION:-0.92}"
METAX_EXTRA_VLLM_ARGS_STR="${METAX_EXTRA_VLLM_ARGS:-}"
METAX_EXTRA_VLLM_ARGS=()
if [[ -n "$METAX_EXTRA_VLLM_ARGS_STR" ]]; then
    read -r -a METAX_EXTRA_VLLM_ARGS <<< "$METAX_EXTRA_VLLM_ARGS_STR"
fi
if [[ -n "${METAX_MAX_NUM_SEQS:-}" ]]; then
    METAX_EXTRA_VLLM_ARGS+=(--max-num-seqs "$METAX_MAX_NUM_SEQS")
fi

if [[ -n "${EXISTING_SERVER_PORT:-}" ]]; then
    # Benchmark a server that already serves $MODEL; its lifecycle is owned by
    # the caller. Either supervise a real PID (EXISTING_SERVER_PID) or poll
    # /health only, bounded by a timeout.
    PORT="$EXISTING_SERVER_PORT"
    if ! : > "$SERVER_LOG" 2>/dev/null; then
        SERVER_LOG="/tmp/inferencex-server-$$.log"
        : > "$SERVER_LOG"
    fi
    READY_ARGS=(--port "$PORT" --server-log "$SERVER_LOG")
    if [[ -n "${EXISTING_SERVER_PID:-}" ]]; then
        READY_ARGS+=(--server-pid "$EXISTING_SERVER_PID")
    else
        EXISTING_SERVER_TIMEOUT="${EXISTING_SERVER_TIMEOUT:-3600}"
        READY_ARGS+=(--no-supervision --timeout "$EXISTING_SERVER_TIMEOUT")
    fi
    wait_for_server_ready "${READY_ARGS[@]}" || {
        echo "Error: existing server on port $PORT did not become ready" >&2
        exit 1
    }
    echo "Using existing server on port $PORT"
else
    check_env_vars IMAGE
    # IMAGE is the caller-owned engine image and the provenance recorded in
    # the results; METAX_VLLM_IMAGE is an explicit override only.
    METAX_VLLM_IMAGE="${METAX_VLLM_IMAGE:-$IMAGE}"
    set -x
    docker run --rm --name "$METAX_CONTAINER_NAME" \
        --device /dev/mxcd --device /dev/dri \
        -v "$MODEL:$MODEL":ro \
        -p "$PORT:8888" \
        --entrypoint "" \
        "$METAX_VLLM_IMAGE" \
        vllm serve "$MODEL" \
            --host 0.0.0.0 \
            --port 8888 \
            --tensor-parallel-size "$TP" \
            --max-model-len "$MAX_SEQ_LEN" \
            --gpu-memory-utilization "$METAX_GPU_MEMORY_UTILIZATION" \
            --dtype bfloat16 \
            --trust-remote-code \
            --no-enable-prefix-caching \
            "${METAX_EXTRA_VLLM_ARGS[@]}" \
            > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
    METAX_CONTAINER_STARTED=1
    set +x

    wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID" || {
        echo "Error: server failed to become ready on port $PORT" >&2
        exit 1
    }
fi

set +e
set -x
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
    --result-dir "$RESULT_DIR" \
    --trust-remote-code
BMK_EXIT=$?
set +x
set -e

if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

stop_gpu_monitor
exit "$BMK_EXIT"
