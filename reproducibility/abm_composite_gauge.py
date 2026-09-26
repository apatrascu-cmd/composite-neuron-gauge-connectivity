#!/usr/bin/env python3
"""
Agent-based simulation of composite clustering with gauge-covariant adaptive connectivity.

Every neuron is an agent that owns: a complex state z_i = r_i e^{i phi_i}, a natural frequency
omega_i, a charge (adaptation gain) q_i, and, for each incoming link j -> i, a coupling weight
J_ij >= 0 and a transporter phase theta_ij.  Signals arriving from j carry a fixed transmission
phase offset delta_ij (heterogeneous axonal/dendritic delays).  Local update rules (all quantities
on the right-hand side belong to i or to its direct neighbours):

  dz_i/dt      = (mu + i omega_i - |z_i|^2) z_i + sum_j J_ij (U_ij z_j - z_i) + noise,   U_ij = exp(i(theta_ij + delta_ij))
  dtheta_ij/dt = -2 eta_theta q_i Im( conj(zhat_i) U_ij zhat_j )             (gradient descent on the normalised covariant mismatch; theta_rule='energy' multiplies by J_ij)
  dJ_ij/dt     =  eta_J ( J0 Re( conj(zhat_i) U_ij zhat_j ) - J_ij ),  J_ij clipped to [0, J0]   (covariant Hebbian rule)

Models compared (same graph, same delays, same seeds):
  'gauge'    : transporters learned (the proposed model)
  'standard' : theta_ij = 0 fixed (ordinary adaptive network; the control)
  'quenched' : theta_ij random and fixed (gauge-glass control: transporters without learning)

Outputs (results/abm_*.json, figures/): coupling-defined candidate clusters, adjusted Rand index
against the planted frequency groups, coverage-aware raw and transported coherences, a complete
cycle-basis consistency test (reciprocal two-cycles plus fundamental cycles), and three perturbation
tests (offset redraw, neuron removal, frequency step).  Run with --quick for a smoke test.
PASS/FAIL controls: static gauge covariance, non-vacuity of the transporter rule, an exact algebraic
offset compensator, and positive/negative controls for the coverage and cycle diagnostics.
"""
import argparse, hashlib, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------- graph
def random_geometric_graph(N, k_target, rng):
    """Random geometric graph in the unit square; radius chosen for mean degree ~ k_target."""
    pos = rng.random((N, 2))
    r = np.sqrt(k_target / (np.pi * N))
    d2 = ((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1)
    A = (d2 < r * r) & ~np.eye(N, dtype=bool)
    iu = np.triu_indices(N, 1)
    m = A[iu]
    ua, ub = iu[0][m], iu[1][m]          # undirected edges a<b
    return pos, ua, ub

def union_find_components(N, a, b):
    parent = np.arange(N)
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i, j in zip(a, b):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    roots = np.array([find(i) for i in range(N)])
    _, labels = np.unique(roots, return_inverse=True)
    return labels

def triangles(N, ua, ub):
    nbr = [set() for _ in range(N)]
    for a, b in zip(ua, ub):
        nbr[a].add(b); nbr[b].add(a)
    tri = []
    for a in range(N):
        for b in nbr[a]:
            if b <= a: continue
            for c in nbr[a] & nbr[b]:
                if c > b: tri.append((a, b, c))
    return np.array(tri, dtype=int).reshape(-1, 3)

def adjusted_rand_index(x, y):
    x = np.asarray(x); y = np.asarray(y)
    cx, xi = np.unique(x, return_inverse=True); cy, yi = np.unique(y, return_inverse=True)
    M = np.zeros((len(cx), len(cy)))
    np.add.at(M, (xi, yi), 1)
    comb = lambda n: n * (n - 1) / 2.0
    sum_ij = comb(M).sum(); a = comb(M.sum(1)).sum(); b = comb(M.sum(0)).sum(); n = comb(len(x))
    exp = a * b / n if n > 0 else 0.0
    mx = 0.5 * (a + b)
    return (sum_ij - exp) / (mx - exp) if mx != exp else 1.0


def cluster_purity(cl, groups):
    """size-weighted mean over detected clusters of the fraction of members belonging to the cluster's majority hidden group"""
    labs = [c for c in np.unique(cl) if c >= 0]
    if not labs: return None
    pur, siz = [], []
    for c in labs:
        gg = groups[cl == c]; pur.append(np.bincount(gg).max() / len(gg)); siz.append(len(gg))
    return float(np.average(pur, weights=siz))


def clusters_per_group(cl, groups, G):
    """mean number of detected clusters whose majority group is g, over the G hidden groups (1 = each group is one cluster)"""
    labs = [c for c in np.unique(cl) if c >= 0]
    if not labs: return None
    maj = [int(np.argmax(np.bincount(groups[cl == c]))) for c in labs]
    return float(np.mean([maj.count(g) for g in range(G)]))

def omega_cluster_spread(omega, cl, groups, G):
    """Diagnostic for where the natural-frequency spread ends up after adaptation.

    Returns (within_cluster_sd, cluster_mean_sd_within_group):
      within_cluster_sd            mean over detected clusters of the s.d. of omega inside the cluster;
      cluster_mean_sd_within_group mean over hidden groups of the s.d., over the clusters that intersect
                                   the group, of the mean omega of that cluster-group cell.
    A within-group spread that is large while within_cluster_sd is small means adaptation pulled each
    cluster to its own frequency and the group was split, not that adaptation failed."""
    labs = [c for c in np.unique(cl) if c >= 0]
    if not labs: return None, None
    within = [float(omega[cl == c].std()) for c in labs if (cl == c).sum() > 1]
    cellsd = []
    for g in range(G):
        ms = [float(omega[(cl == c) & (groups == g)].mean()) for c in labs if ((cl == c) & (groups == g)).sum() > 1]
        if len(ms) > 1: cellsd.append(float(np.std(ms)))
    return (float(np.mean(within)) if within else None), (float(np.mean(cellsd)) if cellsd else None)

def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi

# ----------------------------------------------------------------------------- network object
class Net:
    def __init__(self, N, ua, ub, groups, omega, q, delta, J0, model, rng, quenched_theta=None):
        self.N = N
        # directed edges: for each undirected (a,b): a<-b (dst=a, src=b) and b<-a
        self.dst = np.concatenate([ua, ub]); self.src = np.concatenate([ub, ua])
        self.und = np.concatenate([np.arange(len(ua)), np.arange(len(ua))])  # undirected index of each directed edge
        self.rev = np.concatenate([np.arange(len(ua)) + len(ua), np.arange(len(ua))])  # index of the reverse edge
        self.E = len(self.dst)
        self.groups = groups; self.omega = omega.copy(); self.q = q; self.J0 = J0; self.model = model
        self.delta = delta.copy()                       # transmission phase offsets (fixed by the environment)
        self.J = np.full(self.E, J0)
        if model == 'gauge':
            self.theta = np.zeros(self.E)
        elif model == 'quenched':
            self.theta = quenched_theta.copy()
        else:
            self.theta = np.zeros(self.E)
        self.z = 0.1 * (rng.standard_normal(N) + 1j * rng.standard_normal(N))
        self.alive = np.ones(N, dtype=bool)

    def U(self):
        return np.exp(1j * (self.theta + self.delta))

    def step(self, dt, mu, eta_theta, eta_J, D, rng, eta_omega=0.0):
        """One synchronous Euler--Maruyama step.

        Every derivative is evaluated from the state, transporter, coupling and frequency at the
        beginning of the step.  Updates are assigned only after all derivatives are formed.
        """
        z = self.z; dst, src = self.dst, self.src
        U = self.U()
        Jm = self.J * self.alive[src] * self.alive[dst]
        transported = U * z[src]
        coup = Jm * (transported - z[dst])
        I = np.bincount(dst, weights=coup.real, minlength=self.N) + 1j * np.bincount(dst, weights=coup.imag, minlength=self.N)
        zhat = z / (np.abs(z) + 1e-12)
        m = np.conj(zhat[dst]) * U * zhat[src]          # covariant phase correlation on each link
        # Local derivatives, all evaluated at the previous-step state.
        dtheta = np.zeros(self.E)
        if self.model == 'gauge' and eta_theta > 0:
            w_theta = Jm if getattr(self, 'theta_rule', 'energy') == 'energy' else 1.0   # 'energy': gradient of sum J|z_i-Uz_j|^2 ; 'normalised': gradient of sum |zhat_i-U zhat_j|^2
            dtheta = -2.0 * eta_theta * self.q[dst] * w_theta * m.imag
        dJ = eta_J * (self.J0 * m.real - self.J)
        domega = (eta_omega * np.bincount(dst, weights=Jm * m.imag, minlength=self.N)
                  if eta_omega > 0 else np.zeros(self.N))
        dz = (mu + 1j * self.omega - np.abs(z) ** 2) * z + I
        noise = np.sqrt(2 * D * dt) * (rng.standard_normal(self.N) + 1j * rng.standard_normal(self.N))
        self.theta = self.theta + dt * dtheta
        self.J = np.clip(self.J + dt * dJ, 0.0, self.J0)
        self.omega = self.omega + dt * domega
        self.z = z + dt * dz + noise
        self.z[~self.alive] = 0.0

    # ------------------------------------------------------------- diagnostics
    def J_sym(self):
        return 0.5 * (self.J + self.J[self.rev])[: self.E // 2]

    def detect_clusters(self, thr_frac=0.5, min_size=5):
        Js = self.J_sym()
        ua, ub = self.dst[: self.E // 2], self.src[: self.E // 2]
        keep = (Js > thr_frac * self.J0) & self.alive[ua] & self.alive[ub]
        lab = union_find_components(self.N, ua[keep], ub[keep])
        sizes = np.bincount(lab)
        cl = -np.ones(self.N, dtype=int)   # -1 = elementary (intrinsic) neuron
        k = 0
        for c in np.argsort(-sizes):
            if sizes[c] < min_size: break
            cl[lab == c] = k; k += 1
        cl[~self.alive] = -2
        return cl

    def _strong_components(self, members, thr_frac):
        """Connected components of surviving members using strong undirected support edges."""
        members = np.asarray(members, dtype=int)
        if len(members) == 0:
            return []
        in_set = np.zeros(self.N, dtype=bool); in_set[members] = True
        M = self.E // 2; ua, ub = self.dst[:M], self.src[:M]
        keep = ((self.J_sym() > thr_frac * self.J0) & in_set[ua] & in_set[ub]
                & self.alive[ua] & self.alive[ub])
        adj = {int(i): [] for i in members if self.alive[i]}
        for e in np.flatnonzero(keep):
            a, b = int(ua[e]), int(ub[e]); adj[a].append((b, int(e))); adj[b].append((a, int(self.rev[e])))
        comps = []
        unseen = set(adj)
        while unseen:
            root = min(unseen); stack = [root]; unseen.remove(root); nodes = []
            while stack:
                i = stack.pop(); nodes.append(i)
                for j, _ in adj[i]:
                    if j in unseen: unseen.remove(j); stack.append(j)
            comps.append(np.array(sorted(nodes), dtype=int))
        return sorted(comps, key=len, reverse=True)

    def _transport_component(self, component, thr_frac):
        """Transport all states to one root along a strong-edge spanning tree."""
        component = np.asarray(component, dtype=int)
        if len(component) == 0:
            return 0.0, {}, set()
        in_set = np.zeros(self.N, dtype=bool); in_set[component] = True
        phi = self.theta + self.delta; M = self.E // 2
        ua, ub = self.dst[:M], self.src[:M]
        keep = ((self.J_sym() > thr_frac * self.J0) & in_set[ua] & in_set[ub]
                & self.alive[ua] & self.alive[ub])
        adj = {int(i): [] for i in component}
        for e in np.flatnonzero(keep):
            a, b = int(ua[e]), int(ub[e])
            # e is a<-b; rev[e] is b<-a.
            adj[a].append((b, int(e), int(e)))
            adj[b].append((a, int(self.rev[e]), int(e)))
        root = int(component[0]); psi = {root: 0.0}; tree_und = set(); queue = [root]
        while queue:
            i = queue.pop(0)
            for j, directed_e, und_e in adj[i]:
                if j not in psi:
                    # phi_(i<-j) = psi_i - psi_j on a flat connection.
                    psi[j] = psi[i] - phi[directed_e]
                    tree_und.add(und_e); queue.append(j)
        ks = np.array(sorted(psi), dtype=int)
        zhat = self.z / (np.abs(self.z) + 1e-12)
        transported = np.array([np.exp(-1j * psi[int(k)]) * zhat[k] for k in ks])
        return float(abs(transported.mean())) if len(transported) else 0.0, psi, tree_und

    def cluster_observables(self, cl, thr_frac=0.5):
        """Coverage-aware observables for every labelled cluster.

        R_cov_lcc is the transported coherence inside the largest surviving strong component.
        R_cov_coverage = R_cov_lcc times the fraction of surviving members in that component;
        this conservative score cannot report a disconnected assembly as fully coherent.
        """
        zhat = self.z / (np.abs(self.z) + 1e-12)
        rows = []
        for c in range(max(int(cl.max()) + 1, 0)):
            members = np.where((cl == c) & self.alive)[0]
            comps = self._strong_components(members, thr_frac)
            n = len(members); lcc = comps[0] if comps else np.array([], dtype=int)
            R_lcc, _, _ = self._transport_component(lcc, thr_frac)
            coverage = len(lcc) / n if n else 0.0
            component_scores = [self._transport_component(comp, thr_frac)[0] for comp in comps]
            rows.append({
                'cluster': c, 'survivors': n, 'n_components': len(comps),
                'largest_component_fraction': float(coverage),
                'R_raw_all': float(abs(zhat[members].mean())) if n else 0.0,
                'R_cov_lcc': float(R_lcc),
                'R_cov_coverage': float(R_lcc * coverage),
                'R_cov_component_weighted': (float(np.average(component_scores, weights=[len(x) for x in comps]))
                                             if comps else 0.0),
            })
        return rows

    def flatness_diagnostics(self, cl, thr_frac=0.5, tolerance=1e-2):
        """Complete consistency test on the strong subgraph of every candidate cluster.

        A directed U(1) connection is flat iff reciprocal two-cycles and a fundamental cycle
        basis vanish.  Tree potentials are reconstructed once per cluster; residuals on every
        reciprocal edge and every non-tree edge then span all directed cycle constraints.
        """
        phi = self.theta + self.delta; M = self.E // 2
        ua, ub = self.dst[:M], self.src[:M]; Js = self.J_sym()
        all_residuals, per_cluster, flat_units, total_units = [], [], 0, 0
        for c in range(max(int(cl.max()) + 1, 0)):
            members = np.where((cl == c) & self.alive)[0]; total_units += len(members)
            comps = self._strong_components(members, thr_frac)
            residuals = []
            for comp in comps:
                _, psi, tree_und = self._transport_component(comp, thr_frac)
                in_set = np.zeros(self.N, dtype=bool); in_set[comp] = True
                keep = np.flatnonzero((Js > thr_frac * self.J0) & in_set[ua] & in_set[ub])
                for e in keep:
                    residuals.append(abs(float(wrap(phi[e] + phi[self.rev[e]]))))
                    if int(e) not in tree_und:
                        predicted = psi[int(ua[e])] - psi[int(ub[e])]
                        residuals.append(abs(float(wrap(phi[e] - predicted))))
            r = np.asarray(residuals, dtype=float)
            rmax = float(r.max()) if len(r) else 0.0
            if rmax <= tolerance: flat_units += len(members)
            all_residuals.extend(r.tolist())
            per_cluster.append({'cluster': c, 'n_constraints': int(len(r)),
                                'mean': float(r.mean()) if len(r) else 0.0,
                                'q95': float(np.quantile(r, 0.95)) if len(r) else 0.0,
                                'max': rmax, 'approximately_flat': bool(rmax <= tolerance)})
        r = np.asarray(all_residuals, dtype=float)
        return {
            'tolerance': tolerance, 'n_constraints': int(len(r)),
            'residual_abs_mean': float(r.mean()) if len(r) else None,
            'residual_abs_median': float(np.median(r)) if len(r) else None,
            'residual_abs_q95': float(np.quantile(r, 0.95)) if len(r) else None,
            'residual_abs_max': float(r.max()) if len(r) else None,
            'fraction_constraints_below_tolerance': float(np.mean(r <= tolerance)) if len(r) else None,
            'fraction_candidate_units_certified': float(flat_units / total_units) if total_units else None,
            'hist_log_edges': np.geomspace(1e-8, np.pi, 41).tolist(),
            'hist_log_counts': np.histogram(np.maximum(r, 1e-8), bins=np.geomspace(1e-8, np.pi, 41))[0].tolist() if len(r) else [0] * 40,
            'per_cluster': per_cluster,
        }

    def build_in_lists(self):
        self._in = {}
        for e in range(self.E):
            self._in.setdefault(self.dst[e], []).append(e)

    def holonomies(self, tri, cl, edge_index):
        """loop holonomy of the total connection (theta+delta) around every triangle a<-b<-c<-a, and the
        reversibility defect (theta+delta)_ij + (theta+delta)_ji on every undirected edge."""
        phi = self.theta + self.delta
        if len(tri) == 0:
            return np.array([], dtype=float), np.array([], dtype=bool), wrap(phi + phi[self.rev])[: self.E // 2]
        e_ab = edge_index[(tri[:, 0], tri[:, 1])]; e_bc = edge_index[(tri[:, 1], tri[:, 2])]; e_ca = edge_index[(tri[:, 2], tri[:, 0])]
        Phi = wrap(phi[e_ab] + phi[e_bc] + phi[e_ca])
        within = (cl[tri[:, 0]] >= 0) & (cl[tri[:, 0]] == cl[tri[:, 1]]) & (cl[tri[:, 1]] == cl[tri[:, 2]])
        defect = wrap(phi + phi[self.rev])[: self.E // 2]
        return Phi, within, defect

def edge_index_map(N, dst, src):
    """dense lookup (dst, src) -> directed edge id, via a dict-backed array for the triangle evaluation."""
    idx = {}
    for e, (i, j) in enumerate(zip(dst, src)):
        idx[(i, j)] = e
    class EI:
        def __getitem__(self, ij):
            i, j = ij
            return np.array([idx[(a, b)] for a, b in zip(i, j)])
    return EI()

# ----------------------------------------------------------------------------- experiment
def run_seed(seed, P, log):
    rng = np.random.default_rng(seed)
    N = P['N']
    pos, ua, ub = random_geometric_graph(N, P['k'], rng)
    groups = rng.integers(0, P['G'], N)
    Omega = P['dOmega'] * (np.arange(P['G']) - (P['G'] - 1) / 2.0)
    omega = Omega[groups] + P['sigma_omega'] * rng.standard_normal(N)
    q = np.ones(N)
    E2 = 2 * len(ua)
    delta = P['delta_max'] * rng.uniform(-1, 1, E2)
    quenched = rng.uniform(-np.pi, np.pi, E2)
    tri = triangles(N, ua, ub)
    out = {'seed': seed, 'N': N, 'E': len(ua), 'n_triangles': int(len(tri)), 'mean_degree': 2 * len(ua) / N, 'models': {}}
    nets = {}
    for model in P['models']:
        net = Net(N, ua, ub, groups, omega, q, delta, P['J0'], model, np.random.default_rng(seed + 1), quenched)
        net.build_in_lists()
        nets[model] = net
        net.theta_rule = P.get('theta_rule', 'energy')   # 'normalised' (paper, default) or 'energy' (J-weighted)
    eidx = edge_index_map(N, nets[P['models'][0]].dst, nets[P['models'][0]].src)
    dt = P['dt']; nA = int(P['T1'] / dt); rec_every = int(P['rec_every'] / dt)
    # ---------------- phase A: self-organisation
    series = {m: [] for m in P['models']}
    t0 = time.time()
    for model, net in nets.items():
        rng_dyn = np.random.default_rng(seed + 7)
        for s in range(nA):
            net.step(dt, P['mu'], P['eta_theta'], P['eta_J'], P['D'], rng_dyn, P['eta_omega'])
            if s == int(0.9 * nA): net._theta_snap = net.theta.copy()
            if s % rec_every == 0:
                zhat = net.z / (np.abs(net.z) + 1e-12)
                Rg = [abs(zhat[groups == g].mean()) for g in range(P['G'])]
                # A planted frequency group need not be connected.  Only its raw all-member
                # coherence is recorded; transported coherence is reserved for connected,
                # coupling-defined candidate clusters and carries an explicit coverage factor.
                series[model].append([s * dt] + Rg + [float(net.J.mean() / P['J0'])])
        log(f'  seed {seed} {model:9s} phase A done ({time.time()-t0:.0f} s)')
    if len(tri):
        delta_hol = wrap(delta[eidx[(tri[:, 0], tri[:, 1])]] + delta[eidx[(tri[:, 1], tri[:, 2])]] + delta[eidx[(tri[:, 2], tri[:, 0])]])
    else:
        delta_hol = np.array([], dtype=float)
    out['delay_holonomy_abs_mean'] = float(np.abs(delta_hol).mean()) if len(delta_hol) else None
    out['delay_holonomy_hist'] = np.histogram(np.abs(delta_hol), bins=30, range=(0, np.pi))[0].tolist()
    for model, net in nets.items():
        cl = net.detect_clusters(P['thr_frac'], P['min_size'])
        obs = net.cluster_observables(cl, P['thr_frac'])
        R_raw = np.array([x['R_raw_all'] for x in obs])
        R_cov = np.array([x['R_cov_coverage'] for x in obs])
        R_cov_lcc = np.array([x['R_cov_lcc'] for x in obs])
        coverage = np.array([x['largest_component_fraction'] for x in obs])
        n_components = np.array([x['n_components'] for x in obs])
        flat = net.flatness_diagnostics(cl, P['thr_frac'], P['flat_tolerance'])
        Phi, within, defect = net.holonomies(tri, cl, eidx)
        det = cl >= 0
        M = net.E // 2; ua0, ub0 = net.dst[:M], net.src[:M]
        within_edge = (cl[ua0] >= 0) & (cl[ua0] == cl[ub0])
        between_edge = (cl[ua0] >= 0) & (cl[ub0] >= 0) & (cl[ua0] != cl[ub0])
        res = {
            'n_clusters': int(cl.max() + 1), 'frac_composite': float(det.mean()), 'frac_elementary': float((cl == -1).mean()),
            'ARI_vs_groups': float(adjusted_rand_index(cl[det], groups[det])) if det.sum() > 1 else 0.0,
            'R_raw_mean': float(R_raw.mean()) if len(R_raw) else 0.0, 'R_cov_mean': float(R_cov.mean()) if len(R_cov) else 0.0,
            'R_cov_lcc_mean': float(R_cov_lcc.mean()) if len(R_cov_lcc) else 0.0,
            'largest_component_fraction_mean': float(coverage.mean()) if len(coverage) else 0.0,
            'n_components_mean': float(n_components.mean()) if len(n_components) else 0.0,
            'R_raw_per_cluster': R_raw.tolist(), 'R_cov_per_cluster': R_cov.tolist(),
            'cluster_observables': obs, 'cluster_sizes': np.bincount(cl[det]).tolist() if det.any() else [],
            'cluster_purity': cluster_purity(cl, groups), 'clusters_per_group_mean': clusters_per_group(cl, groups, P['G']),
            'J_mean_over_J0': float(net.J.mean() / P['J0']),
            'J_within_groups_over_J0': float(net.J[groups[net.dst] == groups[net.src]].mean() / P['J0']),
            'J_between_groups_over_J0': float(net.J[groups[net.dst] != groups[net.src]].mean() / P['J0']),
            'holonomy_abs_mean_within': float(np.abs(Phi[within]).mean()) if within.any() else None,
            'holonomy_abs_mean_between': float(np.abs(Phi[~within]).mean()) if (~within).any() else None,
            'holonomy_abs_median_within': float(np.median(np.abs(Phi[within]))) if within.any() else None,
            'holonomy_abs_median_between': float(np.median(np.abs(Phi[~within]))) if (~within).any() else None,
            'n_tri_within': int(within.sum()), 'n_tri_between': int((~within).sum()),
            'holonomy_hist_within': np.histogram(np.abs(Phi[within]), bins=30, range=(0, np.pi))[0].tolist(),
            'holonomy_hist_between': np.histogram(np.abs(Phi[~within]), bins=30, range=(0, np.pi))[0].tolist(),
            'reversibility_defect_abs_mean': float(np.abs(defect).mean()),
            'reversibility_defect_abs_mean_within': float(np.abs(defect[within_edge]).mean()) if within_edge.any() else None,
            'reversibility_defect_abs_mean_between': float(np.abs(defect[between_edge]).mean()) if between_edge.any() else None,
            'cycle_residual_abs_mean': flat['residual_abs_mean'],
            'cycle_residual_abs_median': flat['residual_abs_median'],
            'cycle_residual_abs_q95': flat['residual_abs_q95'],
            'cycle_residual_abs_max': flat['residual_abs_max'],
            'cycle_constraints_below_tolerance': flat['fraction_constraints_below_tolerance'],
            'candidate_units_certified_flat': flat['fraction_candidate_units_certified'],
            'cycle_basis_diagnostics': flat,
            'omega_within_groups_sd_final': float(np.mean([net.omega[groups == g].std() for g in range(P['G'])])),
            'omega_within_clusters_sd_final': omega_cluster_spread(net.omega, cl, groups, P['G'])[0],
            'omega_cluster_means_sd_within_groups_final': omega_cluster_spread(net.omega, cl, groups, P['G'])[1],
            'omega_group_means_final': [float(net.omega[groups == g].mean()) for g in range(P['G'])],
            'transporter_drift_rate_within': (lambda w: float(w.mean()) if w.size else None)(
                (np.abs(wrap(net.theta - net._theta_snap)) / (0.1 * P['T1']))[(cl[net.dst] == cl[net.src]) & (cl[net.dst] >= 0)]),
            'series': series[model],
        }
        # ---------------- gauge-covariance control (exact invariance of every reported invariant)
        if model == 'gauge':
            alpha = np.random.default_rng(99).uniform(-np.pi, np.pi, N)
            z_s, th_s = net.z.copy(), net.theta.copy()
            net.z = np.exp(1j * alpha) * net.z
            net.theta = net.theta + alpha[net.dst] - alpha[net.src]
            obs2 = net.cluster_observables(cl, P['thr_frac'])
            R_raw2 = np.array([x['R_raw_all'] for x in obs2])
            R_cov2 = np.array([x['R_cov_coverage'] for x in obs2])
            flat2 = net.flatness_diagnostics(cl, P['thr_frac'], P['flat_tolerance'])
            Phi2, _, defect2 = net.holonomies(tri, cl, eidx)
            zhat = net.z / (np.abs(net.z) + 1e-12); U = net.U()
            m2 = np.conj(zhat[net.dst]) * U * zhat[net.src]
            zhat0 = z_s / (np.abs(z_s) + 1e-12); U0 = np.exp(1j * (th_s + net.delta))
            m0 = np.conj(zhat0[net.dst]) * U0 * zhat0[net.src]
            dev = max(np.abs(R_cov2 - R_cov).max() if len(R_cov) else 0.0,
                      np.abs(wrap(Phi2 - Phi)).max() if len(Phi) else 0.0,
                      np.abs(wrap(defect2 - defect)).max() if len(defect) else 0.0,
                      abs((flat2['residual_abs_max'] or 0.0) - (flat['residual_abs_max'] or 0.0)),
                      np.abs(m2 - m0).max())
            res['gauge_covariance_control'] = {'max_deviation': float(dev), 'PASS': bool(dev < 1e-9),
                                               'raw_coherence_changed_by': float(np.abs(R_raw2 - R_raw).max()) if len(R_raw) else 0.0}
            net.z, net.theta = z_s, th_s
            # non-vacuity control: the learning-rule gradient is not identically zero
            g = 2 * net.q[net.dst] * m0.imag
            res['learning_rule_nonvacuous'] = {'rms_gradient': float(np.sqrt((g ** 2).mean())), 'PASS': bool(np.abs(g).max() > 1e-6)}
        out['models'][model] = res
        log(f'  seed {seed} {model:9s}: clusters={res["n_clusters"]} composite={res["frac_composite"]:.2f} purity={res["cluster_purity"]} ARI={res["ARI_vs_groups"]:.3f} Rraw={res["R_raw_mean"]:.3f} Rcov*coverage={res["R_cov_mean"]:.3f} '
            f'cycle_q95={res["cycle_residual_abs_q95"]} cycle_max={res["cycle_residual_abs_max"]}')
    # ---------------- phase B: perturbation tests (each from a copy of the organised state)
    import copy
    nB = int(P['T2'] / dt)
    for model in P['models']:
        base = nets[model]
        cl0 = base.detect_clusters(P['thr_frac'], P['min_size'])
        obs0 = base.cluster_observables(cl0, P['thr_frac'])
        R0_raw = np.array([x['R_raw_all'] for x in obs0])
        R0_cov = np.array([x['R_cov_coverage'] for x in obs0])
        tests = {}
        # B1 delay shock: all transmission offsets redrawn
        net = copy.deepcopy(base); rng_dyn = np.random.default_rng(seed + 11)
        old_delta = base.delta.copy()
        new_delta = P['delta_max'] * np.random.default_rng(seed + 5).uniform(-1, 1, net.E)
        net.delta = new_delta.copy()
        # Known-true positive: the algebraic compensator leaves the total connection unchanged.
        compensator = copy.deepcopy(base)
        compensator.delta = new_delta.copy(); compensator.theta = compensator.theta - (new_delta - old_delta)
        compensator_dev = float(np.max(np.abs(compensator.U() - base.U())))
        rec = []; t_rec = None
        for s in range(nB):
            net.step(dt, P['mu'], P['eta_theta'], P['eta_J'], P['D'], rng_dyn, P['eta_omega'])
            if s % rec_every == 0:
                oo = net.cluster_observables(cl0, P['thr_frac'])
                Rr = np.array([x['R_raw_all'] for x in oo]); Rc = np.array([x['R_cov_coverage'] for x in oo])
                rec.append([s * dt, float(Rr.mean()), float(Rc.mean()), float(net.J.mean() / P['J0'])])
        oo = net.cluster_observables(cl0, P['thr_frac'])
        Rr = np.array([x['R_raw_all'] for x in oo]); Rc = np.array([x['R_cov_coverage'] for x in oo])
        cov_lcc = np.array([x['R_cov_lcc'] for x in oo]); coverage_after = np.array([x['largest_component_fraction'] for x in oo])
        cl1 = net.detect_clusters(P['thr_frac'], P['min_size'])
        Rs = [r[1] for r in rec]; i_min = int(np.argmin(Rs))      # recovery = first return to 90 % of the pre-shock value AFTER the minimum
        t_rec = next((rec[i][0] for i in range(i_min, len(rec)) if Rs[i] >= 0.9 * R0_raw.mean()), None)
        Rcs = [r[2] for r in rec]; i_minc = int(np.argmin(Rcs))   # same with the covariant (transported) coherence
        t_rec_cov = next((rec[i][0] for i in range(i_minc, len(rec)) if Rcs[i] >= 0.9 * R0_cov.mean()), None)
        tests['delay_shock'] = {'R_raw_before': float(R0_raw.mean()), 'R_raw_after': float(Rr.mean()), 'R_cov_after': float(Rc.mean()),
                                'R_raw_min_during': float(min(r[1] for r in rec)), 'recovery_time_raw_0.9': t_rec,
                                'J_after_over_J0': float(net.J.mean() / P['J0']), 'J_within_groups_after_over_J0': float(net.J[groups[net.dst] == groups[net.src]].mean() / P['J0']),
                                'clusters_after': int(cl1.max() + 1), 'ARI_after_vs_before': float(adjusted_rand_index(cl1[(cl0 >= 0)], cl0[(cl0 >= 0)])),
                                'R_cov_before': float(R0_cov.mean()), 'R_cov_min_during': float(min(r[2] for r in rec)), 'recovery_time_cov_0.9': t_rec_cov,
                                'frac_clusters_collapsed': float(np.mean(Rr < 0.5)) if len(Rr) else None,
                                'frac_clusters_collapsed_cov': float(np.mean(Rc < 0.5)) if len(Rc) else None,
                                'ARI_after_vs_groups': float(adjusted_rand_index(cl1[cl1 >= 0], groups[cl1 >= 0])) if (cl1 >= 0).sum() > 1 else 0.0,
                                'purity_after': cluster_purity(cl1, groups),
                                'largest_component_fraction_after': float(coverage_after.mean()) if len(coverage_after) else None,
                                'R_cov_lcc_after': float(cov_lcc.mean()) if len(cov_lcc) else None,
                                'exact_compensator_max_deviation': compensator_dev,
                                'series': rec}
        if model == 'gauge':
            frozen = copy.deepcopy(base); frozen.delta = new_delta.copy(); rng_frozen = np.random.default_rng(seed + 11)
            for _ in range(nB):
                frozen.step(dt, P['mu'], 0.0, P['eta_J'], P['D'], rng_frozen, P['eta_omega'])
            frozen_obs = frozen.cluster_observables(cl0, P['thr_frac'])
            frozen_cl = frozen.detect_clusters(P['thr_frac'], P['min_size'])
            tests['delay_shock']['frozen_R_cov_after'] = float(np.mean([x['R_cov_coverage'] for x in frozen_obs])) if frozen_obs else None
            tests['delay_shock']['frozen_ARI_after_vs_before'] = float(adjusted_rand_index(frozen_cl[cl0 >= 0], cl0[cl0 >= 0]))
        # B2 neuron removal
        net = copy.deepcopy(base); rng_dyn = np.random.default_rng(seed + 13)
        kill = np.random.default_rng(seed + 3).random(N) < P['remove_frac']
        net.alive[kill] = False; net.z[kill] = 0
        cl_s = cl0.copy(); cl_s[kill] = -2
        for s in range(nB):
            net.step(dt, P['mu'], P['eta_theta'], P['eta_J'], P['D'], rng_dyn, P['eta_omega'])
        oo = net.cluster_observables(cl_s, P['thr_frac'])
        Rr = np.array([x['R_raw_all'] for x in oo]); Rc = np.array([x['R_cov_coverage'] for x in oo])
        coverage_removal = np.array([x['largest_component_fraction'] for x in oo])
        tests['removal'] = {'removed_frac': float(kill.mean()), 'R_raw_before': float(R0_raw.mean()), 'R_raw_after_survivors': float(Rr.mean()),
                            'R_cov_after_survivors': float(Rc.mean()), 'frac_clusters_collapsed': float(np.mean(Rr < 0.5)) if len(Rr) else None,
                            'R_cov_before': float(R0_cov.mean()), 'frac_clusters_collapsed_cov': float(np.mean(Rc < 0.5)) if len(Rc) else None,
                            'largest_component_fraction_after': float(coverage_removal.mean()) if len(coverage_removal) else None}
        # B3 frequency step of hidden group 0
        net = copy.deepcopy(base); rng_dyn = np.random.default_rng(seed + 17)
        half = (groups == 0) & (np.random.default_rng(seed + 19).random(N) < 0.5)   # a random half of hidden group 0 is detuned
        net.omega = base.omega + P['freq_step'] * half
        target = [c for c in range(cl0.max() + 1) if np.mean(groups[cl0 == c] == 0) > 0.5]
        t_rel = None; Rmin = 1.0
        for s in range(nB):
            net.step(dt, P['mu'], P['eta_theta'], P['eta_J'], P['D'], rng_dyn, P['eta_omega'])
            if s % rec_every == 0 and target:
                Rr = np.array([x['R_raw_all'] for x in net.cluster_observables(cl0, P['thr_frac'])])
                Rt = float(np.mean(Rr[target])); Rmin = min(Rmin, Rt)
                if t_rel is None and s * dt > 5 * P['rec_every'] and Rt >= 0.9 * float(np.mean(R0_raw[target])): t_rel = s * dt
        R0t = float(np.mean(R0_raw[target])) if target else None
        Rr_end = np.array([x['R_raw_all'] for x in net.cluster_observables(cl0, P['thr_frac'])])
        Rt_end = float(np.mean(Rr_end[target])) if target else None
        tests['frequency_step'] = {'n_target_clusters': len(target), 'n_shifted': int(half.sum()), 'R_raw_before': R0t, 'R_raw_min_during': Rmin,
                                   'R_raw_end': Rt_end, 'R_raw_end_over_before': (Rt_end / R0t) if (R0t and Rt_end is not None) else None, 'relock_time_raw_0.9': t_rel,
                                   'omega_shifted_final_minus_group_mean': float(np.mean(net.omega[half]) - np.mean(net.omega[(groups == 0) & ~half])) if half.any() and (~half & (groups == 0)).any() else None}
        out['models'][model]['tests'] = tests
        log(f'  seed {seed} {model:9s} tests: shock R_raw {tests["delay_shock"]["R_raw_before"]:.3f}->{tests["delay_shock"]["R_raw_after"]:.3f} '
            f'(cov {tests["delay_shock"]["R_cov_after"]:.3f}) rec={tests["delay_shock"]["recovery_time_raw_0.9"]}; removal R {tests["removal"]["R_raw_after_survivors"]:.3f}; '
            f'relock={tests["frequency_step"]["relock_time_raw_0.9"]}')
    # figure data for the first seed
    if seed == P['seeds'][0]:
        out['figure_data'] = {'pos': pos.tolist(), 'groups': groups.tolist(),
                              'clusters': {m: nets[m].detect_clusters(P['thr_frac'], P['min_size']).tolist() for m in P['models']},
                              'ua': ua.tolist(), 'ub': ub.tolist(), 'J_sym': {m: nets[m].J_sym().tolist() for m in P['models']}}
    return out

def aggregate(runs, models):
    agg = {}
    keys = ['n_clusters', 'frac_composite', 'ARI_vs_groups', 'R_raw_mean', 'R_cov_mean', 'R_cov_lcc_mean',
            'largest_component_fraction_mean', 'n_components_mean', 'J_within_groups_over_J0', 'J_between_groups_over_J0',
            'holonomy_abs_mean_within', 'holonomy_abs_mean_between', 'holonomy_abs_median_within', 'holonomy_abs_median_between',
            'reversibility_defect_abs_mean', 'reversibility_defect_abs_mean_within', 'reversibility_defect_abs_mean_between',
            'cycle_residual_abs_mean', 'cycle_residual_abs_median', 'cycle_residual_abs_q95', 'cycle_residual_abs_max',
            'cycle_constraints_below_tolerance', 'candidate_units_certified_flat',
            'omega_within_groups_sd_final', 'omega_within_clusters_sd_final',
                 'omega_cluster_means_sd_within_groups_final', 'transporter_drift_rate_within', 'cluster_purity', 'clusters_per_group_mean']
    tkeys = {'delay_shock': ['R_raw_after', 'R_cov_after', 'R_raw_min_during', 'recovery_time_raw_0.9', 'J_within_groups_after_over_J0', 'ARI_after_vs_before', 'frac_clusters_collapsed', 'R_cov_min_during', 'recovery_time_cov_0.9', 'frac_clusters_collapsed_cov', 'ARI_after_vs_groups', 'purity_after', 'largest_component_fraction_after', 'R_cov_lcc_after', 'exact_compensator_max_deviation', 'frozen_R_cov_after', 'frozen_ARI_after_vs_before'],
             'removal': ['R_raw_after_survivors', 'R_cov_after_survivors', 'frac_clusters_collapsed', 'frac_clusters_collapsed_cov', 'largest_component_fraction_after'],
             'frequency_step': ['R_raw_min_during', 'R_raw_end_over_before', 'relock_time_raw_0.9', 'omega_shifted_final_minus_group_mean']}
    def ms(vals):
        v = [x for x in vals if x is not None]
        return {'mean': float(np.mean(v)) if v else None, 'sd': float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, 'n': len(v), 'n_none': len(vals) - len(v)}
    for m in models:
        agg[m] = {k: ms([r['models'][m][k] for r in runs]) for k in keys}
        for t, ks in tkeys.items():
            for k in ks:
                agg[m][f'{t}/{k}'] = ms([r['models'][m]['tests'][t].get(k) for r in runs])
    return agg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true'); ap.add_argument('--seeds', type=int, default=5); ap.add_argument('--N', type=int, default=None)
    ap.add_argument('--tag', default='main'); ap.add_argument('--delta_max', type=float, default=np.pi)
    ap.add_argument('--T1', type=float, default=800.0); ap.add_argument('--T2', type=float, default=400.0); ap.add_argument('--sigma_omega', type=float, default=0.05)
    ap.add_argument('--dt', type=float, default=0.05); ap.add_argument('--flat_tolerance', type=float, default=1e-2)
    ap.add_argument('--eta_theta', type=float, default=0.05); ap.add_argument('--theta_rule', choices=['energy', 'normalised'], default='normalised'); ap.add_argument('--models', nargs='+', default=['gauge', 'standard', 'quenched']); ap.add_argument('--eta_J', type=float, default=0.005); ap.add_argument('--eta_omega', type=float, default=0.01)
    a = ap.parse_args()
    P = dict(schema_version=2, N=a.N or 1000, k=12, G=4, dOmega=2.0, sigma_omega=a.sigma_omega, mu=1.0, J0=0.3, eta_theta=a.eta_theta, eta_J=a.eta_J, eta_omega=a.eta_omega, D=0.005, dt=a.dt,
             T1=a.T1, T2=a.T2, rec_every=2.0, thr_frac=0.5, min_size=5, delta_max=a.delta_max, remove_frac=0.2, freq_step=0.5,
             models=a.models, seeds=list(range(a.seeds)), theta_rule=a.theta_rule, flat_tolerance=a.flat_tolerance,
             code_sha256=hashlib.sha256(open(os.path.abspath(__file__), 'rb').read()).hexdigest())
    if a.quick:
        P.update(N=a.N or 250, T1=min(a.T1, 400.0), T2=min(a.T2, 200.0), seeds=[0]); a.tag = 'quick' if a.tag == 'main' else a.tag
    os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
    logf = open(os.path.join(HERE, 'results', f'abm_{a.tag}.log'), 'a')
    def log(s):
        line = f'[{time.strftime("%H:%M:%S")}] {s}'; print(line, flush=True); logf.write(line + '\n'); logf.flush()
    log(f'params {json.dumps(P)}')
    runs = []; t0 = time.time()
    for i, seed in enumerate(P['seeds']):
        runs.append(run_seed(seed, P, log))
        el = time.time() - t0; log(f'seed {seed} done, elapsed {el:.0f} s, ETA {el/(i+1)*(len(P["seeds"])-i-1):.0f} s')
        json.dump({'params': P, 'runs': runs, 'aggregate': aggregate(runs, P['models'])}, open(os.path.join(HERE, 'results', f'abm_{a.tag}.json'), 'w'))
    agg = aggregate(runs, P['models'])
    log('AGGREGATE (mean, sd over seeds):')
    for m in P['models']:
        for k, v in agg[m].items():
            log(f'  {m:9s} {k:45s} {v["mean"]} +- {v["sd"]} (n={v["n"]}, none={v["n_none"]})')
    ok = all(r['models']['gauge']['gauge_covariance_control']['PASS'] and r['models']['gauge']['learning_rule_nonvacuous']['PASS'] for r in runs)
    log(f'CONTROLS: gauge-covariance + non-vacuity: {"PASS" if ok else "FAIL"}')
    log('DONE')

if __name__ == '__main__':
    main()
