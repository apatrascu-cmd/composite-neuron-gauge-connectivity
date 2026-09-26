#!/usr/bin/env python3
"""Deterministic positive and negative controls for the phase-routing claim.

This verifies the representation symmetry of the routing layer itself.  It is
deliberately not an image-adversarial-robustness test: those perturbations act
in a different space and remain an empirical negative result in the paper.
"""
import json
import os

import numpy as np
import torch

from cnn_cifar10_composite import CosRouting, PhaseRouting


HERE = os.path.dirname(os.path.abspath(__file__))
torch.manual_seed(1701)
rng = np.random.default_rng(1701)

batch, n_in, n_out = 7, 20, 6
x = torch.randn(batch, n_in, dtype=torch.float64)
phase = PhaseRouting(n_in, n_out).double().eval()
cosine = CosRouting(n_in, n_out).double().eval()

with torch.no_grad():
    y0 = phase(x)
    m = phase.m
    alpha = torch.tensor(rng.uniform(-np.pi, np.pi, m), dtype=torch.float64)
    c = torch.complex(x[:, :m], x[:, m:2*m])
    c1 = c * torch.exp(1j * alpha)[None, :]
    x1 = torch.cat((c1.real, c1.imag), dim=1)

    # Local input rephasing accompanied by the transporter transformation.
    theta0 = phase.theta.detach().clone()
    phase.theta.copy_(theta0 - alpha[None, :])
    y_cov = phase(x1)
    phase.theta.copy_(theta0)

    # The modulus read-out also removes one common phase without a link update.
    beta = torch.tensor(0.731, dtype=torch.float64)
    cg = c * torch.exp(1j * beta)
    xg = torch.cat((cg.real, cg.imag), dim=1)
    y_global = phase(xg)

    # Negative controls: a local rephasing without transporter compensation and
    # the real cosine layer under the same paired rotation should both change.
    y_uncomp = phase(x1)
    c0 = cosine(x)
    c1out = cosine(x1)

    # The matched interpolation used in the training ablation changes the
    # transformation property without changing the expected squared output.
    # Make the two quadratures have exactly equal empirical second moments so
    # the normalisation identity is checked without Monte-Carlo tolerance.
    yr = torch.randn(4096, dtype=torch.float64)
    yi = torch.randn(4096, dtype=torch.float64)
    yi = yi * torch.sqrt(torch.mean(yr * yr) / torch.mean(yi * yi))
    beta2 = torch.tensor(0.731, dtype=torch.float64)
    yrr = torch.cos(beta2) * yr - torch.sin(beta2) * yi
    yii = torch.sin(beta2) * yr + torch.cos(beta2) * yi
    moment_means = []
    for lam in [0.0, 0.5, 1.0]:
        scale = 2.0 / (1.0 + lam)
        moment_means.append(float(torch.mean(scale * (yr * yr + lam * yi * yi))))
    elliptic0 = torch.sqrt(2.0 * yr * yr + 1e-8)
    elliptic0_rot = torch.sqrt(2.0 * yrr * yrr + 1e-8)
    elliptic1 = torch.sqrt(yr * yr + yi * yi + 1e-8)
    elliptic1_rot = torch.sqrt(yrr * yrr + yii * yii + 1e-8)

cov_dev = float(torch.max(torch.abs(y_cov - y0)))
global_dev = float(torch.max(torch.abs(y_global - y0)))
uncomp_change = float(torch.max(torch.abs(y_uncomp - y0)))
cosine_change = float(torch.max(torch.abs(c1out - c0)))
moment_spread = float(max(moment_means) - min(moment_means))
elliptic0_change = float(torch.max(torch.abs(elliptic0_rot - elliptic0)))
elliptic1_dev = float(torch.max(torch.abs(elliptic1_rot - elliptic1)))
out = {
    'local_rephasing_with_transporter_max_deviation': cov_dev,
    'common_rephasing_modulus_max_deviation': global_dev,
    'local_rephasing_without_transporter_max_change': uncomp_change,
    'cosine_layer_paired_rotation_max_change': cosine_change,
    'elliptic_readout_second_moment_spread': moment_spread,
    'elliptic_lambda0_common_phase_max_change': elliptic0_change,
    'elliptic_lambda1_common_phase_max_deviation': elliptic1_dev,
    'PASS': bool(cov_dev < 1e-10 and global_dev < 1e-10 and
                 uncomp_change > 1e-4 and cosine_change > 1e-4 and
                 moment_spread < 1e-10 and elliptic0_change > 1e-4 and
                 elliptic1_dev < 1e-10),
}
os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
with open(os.path.join(HERE, 'results', 'phase_routing_controls.json'), 'w') as f:
    json.dump(out, f, indent=2)
    f.write('\n')
print(json.dumps(out, indent=2))
print('OVERALL:', 'PASS' if out['PASS'] else 'FAIL')
if not out['PASS']:
    raise SystemExit(1)
