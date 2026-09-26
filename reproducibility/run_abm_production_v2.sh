#!/bin/bash
# Corrected production runs.  Run through pyscf_launch.sh after the pilot passes.
set -u
cd "$(dirname "$0")"
mkdir -p results

PY=${PYTHON:-python3}
COMMON=(--N 1000 --seeds 5 --T1 800 --T2 400 --sigma_omega 0.05 --eta_J 0.005 --theta_rule normalised --eta_theta 0.05 --eta_omega 0.005 --dt 0.05 --flat_tolerance 0.01 --models gauge standard quenched)

run() {
    tag=$1
    shift
    echo "[$(date +%H:%M:%S)] START $tag $*"
    "$PY" abm_composite_gauge.py --tag "$tag" "${COMMON[@]}" "$@" > "results/abm_${tag}.log" 2>&1
    rc=$?
    echo "[$(date +%H:%M:%S)] END $tag rc=$rc"
    return $rc
}

run main_v2 --delta_max 3.141592653589793 || exit $?
run mild_v2 --delta_max 1.5707963267948966 || exit $?
echo PRODUCTION_V2_DONE
