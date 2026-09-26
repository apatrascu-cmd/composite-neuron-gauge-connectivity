#!/usr/bin/env python3
"""Deterministic checks for the gauge-covariant hierarchy construction."""
import json
import os

import numpy as np

from abm_composite_gauge import Net, wrap
from abm_hierarchy import boundary_support, promote

HERE = os.path.dirname(os.path.abspath(__file__))

ua = np.array([0, 1, 0, 3, 4, 3, 2], int)
ub = np.array([1, 2, 2, 4, 5, 5, 3], int)
n = 6
net = Net(n, ua, ub, np.array([0, 0, 0, 1, 1, 1]), np.zeros(n), np.ones(n),
          np.zeros(2 * len(ua)), 0.3, 'gauge', np.random.default_rng(0))
net.theta_rule = 'normalised'
net.build_in_lists()
psi = np.array([0.2, -0.4, 0.7, -0.1, 0.5, 1.0])
net.theta = psi[net.dst] - psi[net.src]
# Give the only inter-composite edge an additional physical phase.
net.theta[6] += 0.8
net.theta[net.rev[6]] -= 0.3
net.z = np.exp(1j * psi)
pos = np.column_stack((np.arange(n), np.zeros(n)))
groups = np.array([0, 0, 0, 1, 1, 1])
cl = groups.copy()

p0 = promote(net, pos, groups, cl, {0, 1}, 0.5)
pairs0, phase0, _ = boundary_support(net, p0)
roots = p0['roots']

alpha = np.array([0.3, -0.2, 0.9, -0.7, 0.4, 0.1])
net.z = np.exp(1j * alpha) * net.z
net.theta = net.theta + alpha[net.dst] - alpha[net.src]
p1 = promote(net, pos, groups, cl, {0, 1}, 0.5)
pairs1, phase1, _ = boundary_support(net, p1)

state_dev = max(abs(p1['Z'][c] - np.exp(1j * alpha[roots[c]]) * p0['Z'][c])
                for c in range(2))
link_dev = 0.0
for c, d in [(0, 1), (1, 0)]:
    expected = phase0[(c, d)] + alpha[roots[c]] - alpha[roots[d]]
    link_dev = max(link_dev, abs(float(wrap(phase1[(c, d)] - expected))))

out = {
    'promoted_state_covariance_max_deviation': float(state_dev),
    'coarse_link_covariance_max_deviation': float(link_dev),
    'support_invariant': pairs0 == pairs1 == {(0, 1)},
    'PASS': bool(state_dev < 1e-12 and link_dev < 1e-12 and pairs0 == pairs1),
}
os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
with open(os.path.join(HERE, 'results', 'hierarchy_controls.json'), 'w') as f:
    json.dump(out, f, indent=2)
    f.write('\n')
print(json.dumps(out, indent=2))
print('OVERALL:', 'PASS' if out['PASS'] else 'FAIL')
if not out['PASS']:
    raise SystemExit(1)
