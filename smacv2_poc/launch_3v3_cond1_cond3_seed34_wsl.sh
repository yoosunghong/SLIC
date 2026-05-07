#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

mkdir -p results_scaling/logs results_scaling/launchers

ts="$(date +%Y%m%d_%H%M%S)"
launch_log="results_scaling/launchers/launch_3v3_cond1_cond3_seed34_wsl_${ts}.log"

source "$HOME/miniconda3/etc/profile.d/conda.sh"

echo "[launcher] start $(date -Is)" | tee "$launch_log"

latest_checkpoint() {
    local tag="$1"
    local ckpt_dir="results_scaling/checkpoints/${tag}"
    local latest=""
    local latest_step=-1

    shopt -s nullglob
    for ckpt in "$ckpt_dir"/step*.pt; do
        local name="${ckpt##*/}"
        local step="${name#step}"
        step="${step%.pt}"
        if [[ "$step" =~ ^[0-9]+$ ]] && (( step > latest_step )); then
            latest_step="$step"
            latest="$ckpt"
        fi
    done
    shopt -u nullglob

    printf '%s' "$latest"
}

launch_one() {
    local cond="$1"
    local seed="$2"
    local tag="3v3_balanced_cond${cond}_seed${seed}"
    local out="results_scaling/logs/${tag}.stdout.log"
    local err="results_scaling/logs/${tag}.stderr.log"
    local pidfile="results_scaling/logs/${tag}.wsl.pid"
    local resume
    resume="$(latest_checkpoint "$tag")"

    local args=(
        python -u training/train_scaling.py
        --config_name 3v3_balanced
        --condition "$cond"
        --seed "$seed"
        --vram_fraction 0.22
    )

    if [[ -n "$resume" ]]; then
        args+=(--resume "$resume")
    fi

    echo "[launcher] launching ${tag}" | tee -a "$launch_log"
    if [[ -n "$resume" ]]; then
        echo "[launcher] resume=${resume}" | tee -a "$launch_log"
    else
        echo "[launcher] resume=<none>" | tee -a "$launch_log"
    fi
    echo "[launcher] stdout=${out}" | tee -a "$launch_log"
    echo "[launcher] stderr=${err}" | tee -a "$launch_log"

    CPU_THREADS=2 SC2PATH="$HOME/StarCraftII" nohup conda run -n base "${args[@]}" \
        > "$out" 2> "$err" &

    local pid="$!"
    echo "$pid" > "$pidfile"
    echo "[launcher] ${tag} pid=${pid}" | tee -a "$launch_log"
}

launch_one 1 3
launch_one 1 4
launch_one 3 3
launch_one 3 4

echo "[launcher] all launched $(date -Is)" | tee -a "$launch_log"
echo "$launch_log"
