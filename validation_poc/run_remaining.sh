#!/usr/bin/env bash
# Orchestrator: launches remaining training jobs in batches of 4.
# Call this after batch 1 (cond1 all seeds + cond2 seed0) is already running.
# Waits for each batch slot to free before launching next job.

PYTHON="C:/ProgramData/anaconda3/envs/de/python.exe"
TRAIN="$(dirname "$0")/training/train.py"
LOGS="$(dirname "$0")/results/logs"
mkdir -p "$LOGS"

BATCH_LIMIT=4
CPU_THREADS=2

# Remaining jobs after batch 1
remaining_conds=(2 2 3 3 3 4 4 4 5 5 5)
remaining_seeds=(1 2 0 1 2 0 1 2 0 1 2)
total=${#remaining_conds[@]}

pids=()

launch_job() {
    local cond=$1 seed=$2
    local logfile="$LOGS/cond${cond}_seed${seed}.log"
    echo "[$(date '+%H:%M:%S')] Launching Cond${cond} Seed${seed} → $(basename $logfile)"
    CPU_THREADS=$CPU_THREADS "$PYTHON" "$TRAIN" \
        --condition "$cond" --seed "$seed" --results_dir results \
        > "$logfile" 2>&1 &
    echo $!
}

wait_for_slot() {
    # Wait until active PIDs < BATCH_LIMIT
    while true; do
        new_pids=()
        for pid in "${pids[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                new_pids+=("$pid")
            else
                echo "[$(date '+%H:%M:%S')] Job PID $pid finished."
            fi
        done
        pids=("${new_pids[@]}")
        if [ ${#pids[@]} -lt $BATCH_LIMIT ]; then
            break
        fi
        sleep 10
    done
}

cd "$(dirname "$0")"

for i in "${!remaining_conds[@]}"; do
    wait_for_slot
    cond=${remaining_conds[$i]}
    seed=${remaining_seeds[$i]}
    pid=$(launch_job "$cond" "$seed")
    pids+=("$pid")
done

# Wait for all to finish
echo "[$(date '+%H:%M:%S')] All jobs launched, waiting for completion..."
for pid in "${pids[@]}"; do
    wait "$pid" 2>/dev/null
done

echo ""
echo "=== All 15 training runs complete ==="
echo "Run analysis next:"
echo "  cd $(dirname $0)"
echo "  $PYTHON analysis/analyze_attention.py --results_dir results"
