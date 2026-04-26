#!/usr/bin/env bash
# Run 6v6_balanced condition 1, seeds 0/1/2 in parallel.
# Uses python -u to disable stdout buffering so logs appear immediately.

set -e
cd "$(dirname "$0")"

CONFIG=6v6_balanced
COND=1
LOG_DIR=results_scaling/logs
SCRIPT=training/train_scaling.py

mkdir -p "$LOG_DIR"

echo "[$(date '+%T')] Starting $CONFIG cond=$COND seeds=0 1 2"

pids=()
for seed in 0 1 2; do
    tag="${CONFIG}_cond${COND}_seed${seed}"
    out="$LOG_DIR/${tag}.stdout.log"
    err="$LOG_DIR/${tag}.stderr.log"
    python -u "$SCRIPT" --config_name "$CONFIG" --condition "$COND" --seed "$seed" \
        > "$out" 2> "$err" &
    pids+=($!)
    echo "  -> $tag  PID=$!  log: $out"
done

echo "  waiting on ${#pids[@]} seeds..."
failed=0
for i in "${!pids[@]}"; do
    pid=${pids[$i]}
    if wait "$pid"; then
        echo "  [OK] seed $i (PID $pid) done"
    else
        echo "  [FAIL] seed $i (PID $pid) exit $?"
        failed=$((failed + 1))
    fi
done

if [ "$failed" -eq 0 ]; then
    echo "[$(date '+%T')] $CONFIG cond=$COND — all seeds complete."
else
    echo "[$(date '+%T')] $CONFIG cond=$COND — $failed seed(s) failed."
    exit 1
fi
