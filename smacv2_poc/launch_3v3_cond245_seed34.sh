#!/usr/bin/env bash
# 3v3 conditions 2, 4, 5 × seeds 3, 4 — 6 jobs in parallel.
set -euo pipefail

cd "$(dirname "$0")"

mkdir -p results_scaling/logs results_scaling/launchers

ts="$(date +%Y%m%d_%H%M%S)"
launch_log="results_scaling/launchers/launch_3v3_cond245_seed34_${ts}.log"

source "$HOME/anaconda3/etc/profile.d/conda.sh"

echo "[launcher] start $(date -Is)" | tee "$launch_log"

launch_one() {
    local cond="$1"
    local seed="$2"
    local tag="3v3_balanced_cond${cond}_seed${seed}"
    local out="results_scaling/logs/${tag}.stdout.log"
    local err="results_scaling/logs/${tag}.stderr.log"
    local pidfile="results_scaling/logs/${tag}.pid"

    echo "[launcher] launching ${tag}" | tee -a "$launch_log"

    CPU_THREADS=2 SC2PATH="$HOME/StarCraftII" nohup conda run -n smac python -u training/train_scaling.py \
        --config_name 3v3_balanced \
        --condition "$cond" \
        --seed "$seed" \
        --vram_fraction 0.14 \
        > "$out" 2> "$err" &

    local pid="$!"
    echo "$pid" > "$pidfile"
    echo "[launcher] ${tag} pid=${pid}  log: ${out}" | tee -a "$launch_log"
}

for cond in 2 4 5; do
    for seed in 3 4; do
        launch_one "$cond" "$seed"
    done
done

echo "[launcher] all 6 jobs launched $(date -Is)" | tee -a "$launch_log"
echo "launch log: $launch_log"
