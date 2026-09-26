#!/bin/bash
# Corrected-model pilot and sensitivity checks.  Run through pyscf_launch.sh.
set -u
cd "$(dirname "$0")"
mkdir -p results

PY=${PYTHON:-python3}
COMMON=(--N 250 --T1 400 --T2 200 --sigma_omega 0.05 --eta_J 0.005 --delta_max 3.141592653589793 --flat_tolerance 0.01)

run() {
    tag=$1
    shift
    echo "[$(date +%H:%M:%S)] START $tag $*"
    "$PY" abm_composite_gauge.py --tag "$tag" "${COMMON[@]}" "$@" > "results/abm_${tag}.log" 2>&1
    rc=$?
    echo "[$(date +%H:%M:%S)] END $tag rc=$rc"
    return $rc
}

run pilot_v2       --seeds 2 --models gauge standard quenched --theta_rule normalised --eta_theta 0.05 --eta_omega 0.005 --dt 0.05 || exit $?
run scan_energy_v2 --seeds 1 --models gauge standard          --theta_rule energy     --eta_theta 0.05 --eta_omega 0.005 --dt 0.05 || exit $?
run scan_etaomega_v2 --seeds 1 --models gauge standard        --theta_rule normalised --eta_theta 0.05 --eta_omega 0.01  --dt 0.05 || exit $?
run scan_dt025_v2  --seeds 1 --models gauge standard          --theta_rule normalised --eta_theta 0.05 --eta_omega 0.005 --dt 0.025 || exit $?

echo VALIDATION_V2_DONE
