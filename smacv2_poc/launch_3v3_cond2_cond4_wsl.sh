#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

mkdir -p results_scaling/logs results_scaling/launchers

ts="$(date +%Y%m%d_%H%M%S)"
launch_log="results_scaling/launchers/launch_3v3_cond2_34_cond4_3_wsl_${ts}.log"

source "$HOME/miniconda3/etc/profile.d/conda.sh"

echo "[launcher] start $(date -Is)" | tee "$launch_log"

launch_one() {
    local cond="$1"
    local seed="$2"
    local tag="3v3_balanced_cond${cond}_seed${seed}"
    local out="results_scaling/logs/${tag}.stdout.log"
    local err="results_scaling/logs/${tag}.stderr.log"
    local pidfile="results_scaling/logs/${tag}.wsl.pid"

    echo "[launcher] launching ${tag}" | tee -a "$launch_log"
    echo "[launcher] stdout=${out}" | tee -a "$launch_log"
    echo "[launcher] stderr=${err}" | tee -a "$launch_log"

    CPU_THREADS=2 SC2PATH="$HOME/StarCraftII" nohup conda run -n base python -u training/train_scaling.py \
        --config_name 3v3_balanced \
        --condition "$cond" \
        --seed "$seed" \
        --vram_fraction 0.30 \
        > "$out" 2> "$err" &

    local pid="$!"
    echo "$pid" > "$pidfile"
    echo "[launcher] ${tag} pid=${pid}" | tee -a "$launch_log"
}

launch_one 2 3
launch_one 2 4
launch_one 4 3

echo "[launcher] all launched $(date -Is)" | tee -a "$launch_log"
echo "$launch_log"
