#!/bin/bash
# Reproduce the four seed-0 records that predated the code-hash field.
# The corrected repository keeps only these reruns and the already hashed seed-1/2 records.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
THREADS=${THREADS:-2}
EPOCHS=${EPOCHS:-8}
mkdir -p results
for cfg in baseline semicomp adaptive adaptive_phase; do
    tag="cnn_${cfg}_g0.01_s0"
    echo "[$(date +%H:%M:%S)] START $tag"
    "$PY" cnn_cifar10_composite.py --config "$cfg" --gamma 1e-2 --seed 0 --threads "$THREADS" --epochs "$EPOCHS" > "results/${tag}.log" 2>&1
    echo "[$(date +%H:%M:%S)] END $tag"
done
echo CNN_PROVENANCE_V2_DONE
