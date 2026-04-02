#!/usr/bin/env bash
# Launcher: runs all 15 training jobs (5 conditions × 3 seeds) in batches of 4.
# Resource limits: CPU_THREADS=2 per process, max 4 parallel → ~6-8 threads active.
# GPU: shared across processes (models are small, 4GB VRAM is sufficient).

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
LOG_DIR="$RESULTS_DIR/logs"
mkdir -p "$LOG_DIR"

CONDA_ENV="de"
BATCH_SIZE=4        # max parallel jobs at once
CPU_THREADS=2       # torch threads per process

run_job() {
    local cond=$1
    local seed=$2
    local logfile="$LOG_DIR/cond${cond}_seed${seed}.log"
    echo "[$(date '+%H:%M:%S')] Starting Cond${cond} Seed${seed} → $logfile"
    CPU_THREADS=$CPU_THREADS conda run -n "$CONDA_ENV" \
        python "$SCRIPT_DIR/training/train.py" \
        --condition "$cond" --seed "$seed" \
        --results_dir results \
        > "$logfile" 2>&1 &
    echo $!
}

# Build job list: all 5 conditions × 3 seeds
jobs_cond=(1 1 1 2 2 2 3 3 3 4 4 4 5 5 5)
jobs_seed=(0 1 2 0 1 2 0 1 2 0 1 2 0 1 2)
total=${#jobs_cond[@]}

pids=()
job_idx=0

while [ $job_idx -lt $total ]; do
    # Fill up to BATCH_SIZE slots
    while [ ${#pids[@]} -lt $BATCH_SIZE ] && [ $job_idx -lt $total ]; do
        cond=${jobs_cond[$job_idx]}
        seed=${jobs_seed[$job_idx]}
        pid=$(run_job "$cond" "$seed")
        pids+=("$pid")
        job_idx=$((job_idx + 1))
    done

    # Wait for any one to finish
    for i in "${!pids[@]}"; do
        pid=${pids[$i]}
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "[$(date '+%H:%M:%S')] Job PID $pid finished."
            unset 'pids[$i]'
            pids=("${pids[@]}")  # re-index
            break
        fi
    done
    sleep 5
done

# Wait for all remaining
for pid in "${pids[@]}"; do
    wait "$pid" && echo "[$(date '+%H:%M:%S')] PID $pid done." || echo "[$(date '+%H:%M:%S')] PID $pid exited with error."
done

echo ""
echo "=== All training jobs complete ==="
echo "Results in: $RESULTS_DIR"
