#!/usr/bin/env bash
# Run 9v9_balanced conditions 1 and 3, seeds 0/1/2 each — all 6 in parallel.
#
# Resource budget (this machine):
#   RAM  : ~108 GB available  (6v6 3-seed baseline = 20 GB → ~10 GB/run assumed)
#   CPU  : 32 cores  → 4 threads/process × 6 = 24 cores used
#   VRAM : ~48 GB    → vram_fraction=0.14 × 6 = 84 % of 49 GB card

set -e
cd "$(dirname "$0")"

CONFIG=9v9_balanced
LOG_DIR=results_scaling/logs
SCRIPT=training/train_scaling.py
CPU_THREADS=4
VRAM_FRAC=0.14

mkdir -p "$LOG_DIR"

echo "[$(date '+%T')] Starting $CONFIG  cond=1,3  seeds=0 1 2  (6 jobs)"

pids=()
tags=()

for cond in 1 3; do
    for seed in 0 1 2; do
        tag="${CONFIG}_cond${cond}_seed${seed}"
        out="$LOG_DIR/${tag}.stdout.log"
        err="$LOG_DIR/${tag}.stderr.log"

        CPU_THREADS=$CPU_THREADS \
        SC2PATH="$HOME/StarCraftII" \
        conda run -n smac python -u "$SCRIPT" \
            --config_name "$CONFIG" \
            --condition   "$cond" \
            --seed        "$seed" \
            --vram_fraction "$VRAM_FRAC" \
            > "$out" 2> "$err" &

        pids+=($!)
        tags+=("$tag")
        echo "  -> $tag  PID=$!  log: $out"
    done
done

echo "  waiting on ${#pids[@]} jobs..."
failed=0
for i in "${!pids[@]}"; do
    pid=${pids[$i]}
    tag=${tags[$i]}
    if wait "$pid"; then
        echo "  [OK]   $tag (PID $pid) done"
    else
        code=$?
        echo "  [FAIL] $tag (PID $pid) exit $code"
        failed=$((failed + 1))
    fi
done

if [ "$failed" -eq 0 ]; then
    echo "[$(date '+%T')] $CONFIG — all 6 jobs complete."
else
    echo "[$(date '+%T')] $CONFIG — $failed job(s) failed."
    exit 1
fi
