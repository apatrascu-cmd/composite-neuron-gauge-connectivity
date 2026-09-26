#!/usr/bin/env python3
"""
Static and transient partial compositeness: the cluster elimination used in the paper.

Model: rate units tau dV/dt = -V + J V + I.  A composite cluster C of n units, all-to-all coupled with strength Lambda,
is read out through its mean V_C; an elementary unit i couples to the cluster with strength eps_i in both directions
(the mixing matrix [[-1, n eps_i], [eps_i, -(1-(n-1)Lambda)]] of the paper).  At a fixed point, eliminating the cluster
by a Schur complement gives J_eff(0) = J_EE + g_C eps eps^T, where
      g_C = n / (1 - (n-1) Lambda),       Lambda_c = 1/(n-1).
For transients the elimination is non-local in time: in Laplace frequency s,
      J_eff(s) = J_EE + n eps eps^T / (1-(n-1)Lambda + tau s).
Checks: (1) the Schur complement equals J + g_C eps eps^T to machine precision (random instances);
        (2) the fixed point of the reduced system equals the fixed point of the full system;
        (3) the naive reduction (composite = plain mean of its members, no g_C) gives a DIFFERENT fixed point (negative control);
        (4) exact transient poles and the frequency-dependent memory kernel.  The static Schur
            complement is exact at zero frequency; the static local time is not an exact pole.
Prints PASS/FAIL and writes results/reduction_dynamics.json.  Pure numpy.
"""
import json, os
import numpy as np
runs = {'random_fixed_point_checks': [], 'transient_curve': []}
rng = np.random.default_rng(0); ok = True
for trial in range(5):
    nE, n = rng.integers(2, 6), rng.integers(3, 9); Lam = rng.uniform(0.0, 0.8) / (n - 1)
    eps = rng.normal(size=nE) * 0.3; JEE = rng.normal(size=(nE, nE)) * 0.1
    JCC = Lam * (np.ones((n, n)) - np.eye(n)); JEC = np.outer(eps, np.ones(n)); JCE = JEC.T
    Jfull = np.block([[JEE, JEC], [JCE, JCC]]); I = rng.normal(size=nE + n)
    gC = n / (1 - (n - 1) * Lam)
    Jred = JEE + JEC @ np.linalg.solve(np.eye(n) - JCC, JCE)                # exact Schur complement
    d1 = np.abs(Jred - (JEE + gC * np.outer(eps, eps))).max()
    Vfull = np.linalg.solve(np.eye(nE + n) - Jfull, I)                       # fixed point of the full system
    Ired = I[:nE] + JEC @ np.linalg.solve(np.eye(n) - JCC, I[nE:])           # = I_i + g_C eps_i I_C when I_C is uniform; general case here
    Vred = np.linalg.solve(np.eye(nE) - Jred, Ired)
    d2 = np.abs(Vred - Vfull[:nE]).max()
    Jnaive = JEE + np.outer(eps, eps) * n                                    # 'composite = plain mean', ignores the cluster's own recurrence
    Vnaive = np.linalg.solve(np.eye(nE) - Jnaive, I[:nE] + eps * I[nE:].sum())
    d3 = np.abs(Vnaive - Vfull[:nE]).max()
    passed = d1 < 1e-12 and d2 < 1e-10 and d3 > 1e-3
    ok &= passed
    runs['random_fixed_point_checks'].append({'trial': trial, 'n': int(n), 'n_elementary': int(nE),
        'Lambda': float(Lam), 'g_C': float(gC), 'schur_max_deviation': float(d1),
        'fixed_point_max_deviation': float(d2), 'naive_fixed_point_max_deviation': float(d3),
        'PASS': bool(passed)})
    print(f'trial {trial}: n={n} nE={nE} Lambda={Lam:.4f} (Lambda_c={1/(n-1):.4f}) g_C={gC:.4f} | Schur dev {d1:.1e} | fixed-point dev {d2:.1e} | naive-reduction dev {d3:.2e} (must be large) -> {"PASS" if passed else "FAIL"}')
print('slow composite mode: eigenvalues of the mixing matrix [[-1, n eps],[eps, -(1-(n-1)Lambda)]] for n=6, eps=0.2.')
n, e = 6, 0.2
Lstar = (1 - n * e ** 2) / (n - 1)      # det = 0: the exact linear pair acquires a zero pole
print(f'  marginal point Lambda* = (1 - n eps^2)/(n-1) = {Lstar:.4f} < Lambda_c = 1/(n-1) = {1/(n-1):.4f}; there g_C eps^2 = {n/(1-(n-1)*Lstar)*e**2:.6f} (= 1)')
for Lam in [0.0, 0.05, 0.1, 0.14, 0.15, 0.151, Lstar, 0.19]:
    M = np.array([[-1.0, n * e], [e, -(1 - (n - 1) * Lam)]]); ev = np.linalg.eigvals(M); gC = n / (1 - (n - 1) * Lam)
    evs = np.sort(ev.real); stable = bool(evs[-1] < -1e-12)
    static_local = 1 / (1 - gC * e**2) if abs(1 - gC * e**2) > 1e-12 else None
    exact_slow = -1 / evs[-1] if stable else None
    a = 1 - (n - 1) * Lam
    low_frequency = ((1 + n * e**2 / a**2) / (1 - gC * e**2)
                     if abs(1 - gC * e**2) > 1e-12 else None)
    print(f'  Lambda={Lam:.4f}: eigenvalues {np.round(evs, 6)}   '
          f'static-local tau={static_local} exact slow tau={exact_slow} '
          f'({"stable" if stable else "unstable or marginal"})')
    runs['transient_curve'].append({'Lambda': Lam, 'a': a, 'eigenvalues_over_tau': evs.tolist(),
        'static_local_tau_over_tau': static_local, 'first_order_low_frequency_tau_over_tau': low_frequency,
        'exact_slow_tau_over_tau': exact_slow, 'stable': stable})
ok &= abs(n / (1 - (n - 1) * Lstar) * e ** 2 - 1) < 1e-12
runs['n'] = n; runs['epsilon'] = e; runs['Lambda_star'] = Lstar; runs['Lambda_c'] = 1/(n-1)
runs['interpretation'] = ('The Schur complement is exact for fixed points. Exact elimination at Laplace '
    'frequency s gives J_eff(s)=J_EE+n eps eps^T/(a+tau s), hence a memory kernel. '
    'The static-local time is not an exact pole; the exact slow time is -1/lambda_plus.')
runs['OVERALL_PASS'] = bool(ok)
os.makedirs(os.path.join(os.path.dirname(__file__), 'results'), exist_ok=True)
with open(os.path.join(os.path.dirname(__file__), 'results', 'reduction_dynamics.json'), 'w') as f:
    json.dump(runs, f, indent=2); f.write('\n')
print('OVERALL:', 'PASS' if ok else 'FAIL')
