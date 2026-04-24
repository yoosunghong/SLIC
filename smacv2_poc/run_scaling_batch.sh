#!/usr/bin/env bash
# Launch scaling-study training runs.
#
# Default: first batch = 3v3_balanced × {C1, C3} × seeds {0,1,2} launched
# in parallel. Once these finish, re-run with BATCH=6v6_balanced etc.
#
# Usage:
#   bash run_scaling_batch.sh                  # default: 3v3_balanced
#   BATCH=6v6_balanced bash run_scaling_batch.sh
#   BATCH=9v9_balanced PARALLEL_PER_COND=2 bash run_scaling_batch.sh
#
# Env vars:
#   BATCH              config_name from configs/scaling.yaml (default: 3v3_balanced)
#   CONDITIONS         space-separated list of conditions (default: "1 3")
#   SEEDS              space-separated list of seeds (default: "0 1 2")
#   PYTHON             python interpreter (default: anaconda paper env)
#   PARALLEL_PER_COND  launch seeds concurrently inside each condition (default: 1=yes)

set -u

BATCH="${BATCH:-3v3_balanced}"
CONDITIONS="${CONDITIONS:-1 3}"
SEEDS="${SEEDS:-0 1 2}"
PYTHON="${PYTHON:-/c/Users/PC/anaconda3/envs/paper/python.exe}"
PARALLEL_PER_COND="${PARALLEL_PER_COND:-1}"

cd "$(dirname "$0")"
mkdir -p results_scaling/logs

echo "[batch=$BATCH] conditions=[$CONDITIONS] seeds=[$SEEDS] parallel=$PARALLEL_PER_COND"

for cond in $CONDITIONS; do
    echo "== Launching config=$BATCH cond=$cond seeds=$SEEDS =="
    pids=()
    for seed in $SEEDS; do
        tag="${BATCH}_cond${cond}_seed${seed}"
        log="results_scaling/logs/${tag}.stdout.log"
        echo "  -> $tag  (log: $log)"
        "$PYTHON" -u training/train_scaling.py \
            --config_name "$BATCH" --condition "$cond" --seed "$seed" \
            > "$log" 2>&1 &
        pids+=($!)
        if [ "$PARALLEL_PER_COND" -ne 1 ]; then
            wait "${pids[-1]}"
        fi
    done
    if [ "$PARALLEL_PER_COND" -eq 1 ]; then
        echo "  waiting on ${#pids[@]} seeds for cond=$cond..."
        for pid in "${pids[@]}"; do
            wait "$pid"
        done
    fi
    echo "== Finished config=$BATCH cond=$cond =="
done

echo "[batch=$BATCH] all runs complete."
