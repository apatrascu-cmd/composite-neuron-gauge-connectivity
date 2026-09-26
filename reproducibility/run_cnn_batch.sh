#!/bin/bash
# Runs the 17 CIFAR-10 jobs reported in the paper (5 configurations x 3 seeds at gamma = 1e-2, plus the
# gamma = 1e-3 and 1e-1 points for the adaptive configuration at seed 0), WORKERS jobs at a time.
# Results: results/cnn_<config>_g<gamma>_s<seed>.json ; per-job logs results/cnn_*.log
# Usage: [PY=python3] [WORKERS=5] [THREADS=3] [EPOCHS=8] bash run_cnn_batch.sh
cd "$(dirname "$0")"
PY=${PY:-python3}; WORKERS=${WORKERS:-5}; THREADS=${THREADS:-3}; EPOCHS=${EPOCHS:-8}
jobs=()
for s in 0 1 2; do for c in baseline semicomp adaptive adaptive_girl adaptive_phase; do jobs+=("--config $c --gamma 1e-2 --seed $s"); done; done
jobs+=("--config adaptive --gamma 1e-3 --seed 0" "--config adaptive --gamma 1e-1 --seed 0")
echo "[$(date '+%H:%M:%S')] starting ${#jobs[@]} jobs, $WORKERS workers x $THREADS threads, $EPOCHS epochs, python=$PY"
printf '%s\n' "${jobs[@]}" | xargs -P "$WORKERS" -I{} sh -c "$PY cnn_cifar10_composite.py {} --threads $THREADS --epochs $EPOCHS > /dev/null 2>&1"
echo "[$(date '+%H:%M:%S')] BATCH_DONE ($(ls results/cnn_*_s?.json 2>/dev/null | wc -l | tr -d ' ') result files)"
