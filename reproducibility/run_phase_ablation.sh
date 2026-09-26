#!/bin/bash
# Capacity- and initialisation-matched invariant-readout interpolation:
# lambda=0,0.5,1.  Usage: WORKERS=2 THREADS=1 bash run_phase_ablation.sh
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
WORKERS=${WORKERS:-2}
THREADS=${THREADS:-1}
EPOCHS=${EPOCHS:-8}
export PY EPOCHS THREADS
mkdir -p results
printf '%s\n' 0 0.5 1 | while read lam; do
  for seed in 0 1 2; do
    printf '%s %s\n' "$lam" "$seed"
  done
done | xargs -n2 -P "$WORKERS" sh -c '
  lam=$1; seed=$2
  "$PY" cnn_phase_ablation.py --readout_lambda "$lam" --seed "$seed" --epochs "$EPOCHS" --threads "$THREADS"
' sh
echo "PHASE_ABLATION_DONE ($(find results -name 'cnn_phase_lambda*_g0.01_s*.json' ! -name '*quick*' | wc -l | tr -d ' ') records)"
