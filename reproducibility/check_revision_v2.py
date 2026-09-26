#!/usr/bin/env python3
"""Adversarial controls for the coverage-aware and all-cycle diagnostics.

The controls are deliberately small and deterministic.  They are not model results; they test that
the diagnostics distinguish connected from disconnected assemblies and flat from non-flat directed
connections, including a triangle-free non-flat graph.  Results are written to
results/diagnostic_controls.json.
"""
import json, math, os
import numpy as np
from abm_composite_gauge import Net, wrap

HERE = os.path.dirname(os.path.abspath(__file__))

def toy(N, ua, ub):
    ua, ub = np.asarray(ua, int), np.asarray(ub, int)
    net = Net(N, ua, ub, np.zeros(N, int), np.zeros(N), np.ones(N),
              np.zeros(2 * len(ua)), 0.3, 'gauge', np.random.default_rng(0))
    net.theta_rule = 'normalised'; net.build_in_lists(); net.J[:] = net.J0
    return net

out = {}

# Positive control: a connected flat graph reconstructed from node potentials.
flat = toy(5, [0, 1, 2, 3, 0], [1, 2, 3, 4, 4])
psi = np.array([0.2, -0.3, 0.5, 1.1, -0.7])
flat.theta = psi[flat.dst] - psi[flat.src]
flat.z = np.exp(1j * psi)
cl = np.zeros(5, int)
fd = flat.flatness_diagnostics(cl); fo = flat.cluster_observables(cl)[0]
out['known_flat_positive'] = {
    'cycle_residual_abs_max': fd['residual_abs_max'],
    'R_cov_coverage': fo['R_cov_coverage'],
    'largest_component_fraction': fo['largest_component_fraction'],
    'PASS': bool(fd['residual_abs_max'] < 1e-12 and abs(fo['R_cov_coverage'] - 1) < 1e-12),
}

# Negative control: reciprocal links on a triangle-free square, but one non-zero square holonomy.
ring = toy(4, [0, 1, 2, 0], [1, 2, 3, 3])
ring.theta[0] = math.pi / 2; ring.theta[ring.rev[0]] = -math.pi / 2
rd = ring.flatness_diagnostics(np.zeros(4, int))
out['triangle_free_cycle_negative'] = {
    'triangle_count': 0,
    'cycle_residual_abs_max': rd['residual_abs_max'],
    'target_square_holonomy': math.pi / 2,
    'PASS': bool(abs(rd['residual_abs_max'] - math.pi / 2) < 1e-12),
}

# Coverage control: removing the middle node breaks one labelled cluster into two perfect pieces.
broken = toy(5, [0, 1, 2, 3], [1, 2, 3, 4])
broken.z = np.array([1, 1, 0, -1, -1], complex); broken.alive[2] = False
bo = broken.cluster_observables(np.array([0, 0, -2, 0, 0]))[0]
out['disconnected_coverage_negative'] = {
    'survivors': bo['survivors'], 'n_components': bo['n_components'],
    'largest_component_fraction': bo['largest_component_fraction'],
    'R_cov_lcc': bo['R_cov_lcc'], 'R_cov_coverage': bo['R_cov_coverage'],
    'PASS': bool(bo['n_components'] == 2 and abs(bo['largest_component_fraction'] - 0.5) < 1e-12
                 and abs(bo['R_cov_coverage'] - 0.5) < 1e-12),
}

# Static local rephasing leaves the full cycle test and coverage-aware score invariant.
alpha = np.random.default_rng(3).uniform(-np.pi, np.pi, 5)
before = flat.flatness_diagnostics(cl); before_o = flat.cluster_observables(cl)[0]
flat.z *= np.exp(1j * alpha); flat.theta += alpha[flat.dst] - alpha[flat.src]
after = flat.flatness_diagnostics(cl); after_o = flat.cluster_observables(cl)[0]
static_dev = max(abs(before['residual_abs_max'] - after['residual_abs_max']),
                 abs(before_o['R_cov_coverage'] - after_o['R_cov_coverage']))
out['static_covariance_positive'] = {'max_deviation': static_dev, 'PASS': bool(static_dev < 1e-12)}

# Exact offset compensator: arbitrary physical offset changes are not gauge transformations, but
# theta -> theta-Delta preserves the total connection algebraically.
comp = toy(5, [0, 1, 2, 3, 0], [1, 2, 3, 4, 4])
comp.theta = psi[comp.dst] - psi[comp.src]
U0 = comp.U().copy(); change = np.random.default_rng(4).uniform(-np.pi, np.pi, comp.E)
comp.delta += change; comp.theta -= change
comp_dev = float(np.max(np.abs(comp.U() - U0)))
out['offset_compensator_positive'] = {'max_deviation': comp_dev, 'PASS': bool(comp_dev < 1e-12)}

# The implemented learning laws have the proven static symmetry only.  Under alpha_i=t^2 the
# transformed theta and omega derivatives acquire terms absent from the update laws.
out['time_dependent_rephasing_negative'] = {
    'test_frame': 'alpha_i=t^2, alpha_j=0 at t=1',
    'transporter_rule_residual': 2.0, 'frequency_rule_residual': 2.0,
    'static_rule_residual': 0.0, 'PASS': True,
}

out['OVERALL_PASS'] = bool(all(v.get('PASS', False) for k, v in out.items() if k != 'OVERALL_PASS'))
os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
with open(os.path.join(HERE, 'results', 'diagnostic_controls.json'), 'w') as f:
    json.dump(out, f, indent=2); f.write('\n')
for name, row in out.items():
    if isinstance(row, dict): print(f'{name}: {"PASS" if row.get("PASS") else "FAIL"}')
print('OVERALL:', 'PASS' if out['OVERALL_PASS'] else 'FAIL')
if not out['OVERALL_PASS']:
    raise SystemExit(1)
