#!/usr/bin/env python3
"""
Symbolic checks (sympy) of the gauge structure used in the paper.
 (1) The transporter learning rule of the ORIGINAL manuscript, dA_ij/dt = -eta d/dA_ij |D_t V_i - (d_t - i q_i A_ij) V_j|^2,
     has an identically vanishing gradient (the A_ij V_j term appears in both pieces and cancels): the rule never updates
     the link it is written for.  This is why it was replaced.
 (2) The NEW rule: E = sum_ij J_ij |z_i - U_ij z_j|^2 with U_ij = exp(i(theta_ij + delta_ij)) gives
     dE/dtheta_ij = 2 J_ij Im(conj(z_i) U_ij z_j), which is not identically zero.
 (3) Static local phase invariance: E, the link correlation m_ij = conj(zhat_i) U_ij zhat_j, the triangle holonomy and the
     frequency-adaptation drive are invariant under z_k -> exp(i alpha_k) z_k, theta_ij -> theta_ij + alpha_i - alpha_j;
     the composite operator V_C^(i) = (1/|C|) sum_k U_ik z_k is covariant (picks up exp(i alpha_i)); the raw mean is not.
Prints PASS/FAIL for every item.
"""
import sympy as sp
ok = True
def report(name, cond):
    global ok; ok &= bool(cond); print(f'  {name}: {"PASS" if cond else "FAIL"}')
print('(1) original rule')
t = sp.symbols('t', real=True); q1 = sp.symbols('q1', real=True)
V = [sp.Function(f'V{k}')(t) for k in range(3)]
a = {k: sp.symbols(f'a1{k}', real=True) for k in range(3)}; b = {k: sp.symbols(f'b1{k}', real=True) for k in range(3)}
A = {k: a[k] + sp.I * b[k] for k in range(3)}
DtV1 = sp.diff(V[0], t) - sp.I * q1 * sum(A[k] * V[k] for k in range(3))
j = 1; resid = DtV1 - (sp.diff(V[j], t) - sp.I * q1 * A[j] * V[j]); loss = resid * sp.conjugate(resid)
g = (sp.simplify(sp.diff(loss, a[j])), sp.simplify(sp.diff(loss, b[j])))
report('gradient of the original objective w.r.t. A_1j is identically zero', g == (0, 0))
print('(2) new rule')
r = sp.symbols('r0:3', positive=True); ph = sp.symbols('phi0:3', real=True); th = {}; de = {}; J = {}
links = [(0, 1), (1, 2), (2, 0), (1, 0), (2, 1), (0, 2)]
for (i, k) in links:
    th[(i, k)] = sp.symbols(f'theta{i}{k}', real=True); de[(i, k)] = sp.symbols(f'delta{i}{k}', real=True); J[(i, k)] = sp.symbols(f'J{i}{k}', positive=True)
z = [r[k] * sp.exp(sp.I * ph[k]) for k in range(3)]
U = {l: sp.exp(sp.I * (th[l] + de[l])) for l in links}
E = sum(J[l] * sp.Abs(z[l[0]] - U[l] * z[l[1]]) ** 2 for l in links)
E = sp.expand(sum(J[l] * ((z[l[0]] - U[l] * z[l[1]]) * sp.conjugate(z[l[0]] - U[l] * z[l[1]])) for l in links))
l = (0, 1)
grad = sp.simplify(sp.diff(E, th[l]))
formula = sp.simplify(2 * J[l] * sp.im(sp.conjugate(z[0]) * U[l] * z[1]))
syms = [r[0], r[1], ph[0], ph[1], th[l], de[l], J[l]]
f = sp.lambdify(syms, grad - formula, 'numpy')
import numpy as np, itertools
pts = np.random.default_rng(1).uniform(0.2, 3.0, size=(20, 7))
report('dE/dtheta_01 = 2 J_01 Im(conj(z_0) U_01 z_1)  (numerically at 20 random points)', max(abs(complex(f(*pt))) for pt in pts) < 1e-12)
report('new gradient is not identically zero', sp.simplify(grad.subs({th[l]: 0.3, de[l]: 0.2, ph[0]: 0.1, ph[1]: 0.7, r[0]: 1, r[1]: 1, J[l]: 1})) != 0)
print('(3) static local phase invariance / covariance')
al = sp.symbols('alpha0:3', real=True)
sub = {ph[k]: ph[k] + al[k] for k in range(3)}
sub.update({th[(i, k)]: th[(i, k)] + al[i] - al[k] for (i, k) in links})
report('energy E invariant', sp.simplify(E.subs(sub, simultaneous=True) - E) == 0)
m01 = sp.exp(-sp.I * ph[0]) * U[(0, 1)] * sp.exp(sp.I * ph[1])
report('link correlation m_01 invariant', sp.simplify(m01.subs(sub, simultaneous=True) - m01) == 0)
hol = th[(0, 1)] + de[(0, 1)] + th[(1, 2)] + de[(1, 2)] + th[(2, 0)] + de[(2, 0)]
report('triangle holonomy invariant', sp.simplify(hol.subs(sub, simultaneous=True) - hol) == 0)
drive = sum(J[(0, k)] * sp.im(sp.exp(-sp.I * ph[0]) * U[(0, k)] * sp.exp(sp.I * ph[k])) for k in (1, 2))
report('frequency-adaptation drive on unit 0 invariant', sp.simplify(drive.subs(sub, simultaneous=True) - drive) == 0)
VC0 = (U[(0, 1)] * z[1] + U[(0, 2)] * z[2]) / 2
report('composite operator V_C^(0) covariant (factor exp(i alpha_0))', sp.simplify(VC0.subs(sub, simultaneous=True) - sp.exp(sp.I * al[0]) * VC0) == 0)
raw = (z[1] + z[2]) / 2
report('raw mean NOT covariant (changes under an inhomogeneous rephasing)', sp.simplify(raw.subs(sub, simultaneous=True) - sp.exp(sp.I * al[0]) * raw) != 0)
print('OVERALL:', 'PASS' if ok else 'FAIL')
