#!/usr/bin/env bash
set -euo pipefail

cd /mnt/c/Users/PC/Documents/GitHub/de_paper/smacv2_poc
PYTHON_BIN="/home/yoosung/miniconda3/bin/python"

stamp="$(date +%Y%m%d_%H%M%S)"
launcher_log="results_scaling/launchers/resume_3v3_cond1_seed34_noearly_${stamp}.log"

{
  echo "[launcher] start $(date -Is)"
  echo "[launcher] script=$0"
} > "${launcher_log}"

launch_one() {
  local seed="$1"
  local resume="$2"
  local tag="3v3_balanced_cond1_seed${seed}"
  local stdout="results_scaling/logs/${tag}.resume_noearly.stdout.log"
  local stderr="results_scaling/logs/${tag}.resume_noearly.stderr.log"

  {
    echo "[launcher] launching ${tag}"
    echo "[launcher] resume=${resume}"
    echo "[launcher] disable_early_stop=true"
    echo "[launcher] stdout=${stdout}"
    echo "[launcher] stderr=${stderr}"
  } >> "${launcher_log}"

  CPU_THREADS=2 nohup "${PYTHON_BIN}" training/train_scaling.py \
    --config_name 3v3_balanced \
    --condition 1 \
    --seed "${seed}" \
    --resume "${resume}" \
    --disable_early_stop \
    --vram_fraction 0.30 \
    >> "${stdout}" 2>> "${stderr}" &

  echo "[launcher] ${tag} pid=$!" >> "${launcher_log}"
}

launch_one 3 "results_scaling/checkpoints/3v3_balanced_cond1_seed3/step1244672.pt"
launch_one 4 "results_scaling/checkpoints/3v3_balanced_cond1_seed4/step1617920.pt"

echo "[launcher] all launched $(date -Is)" >> "${launcher_log}"
cat "${launcher_log}"
