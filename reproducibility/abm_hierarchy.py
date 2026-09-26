#!/usr/bin/env python3
"""Gauge-covariant hierarchical extension of the composite-neuron ABM.

Level 0 is the corrected gauge arm from ``abm_composite_gauge.py``.  Only
candidate clusters whose complete cycle diagnostics pass the declared
tolerance are promoted to level-1 agents.  Their states are transported to a
cluster root before averaging.  Fine boundary links are dressed by the two
intra-cluster transports, so each resulting level-1 link transforms as a U(1)
transporter between the two cluster roots.

The experiment asks whether an explicit next information scale resolves the
fragmentation of planted frequency groups.  It compares:

  * local_gauge / local_standard: boundary-derived coarse support only;
  * frequency_gauge / frequency_standard: the same boundary support plus a
    sparse k-nearest graph in the observable cluster-mean frequency;
  * random_gauge: the same number of added links placed randomly;
  * oracle_gauge: a positive control whose added links may use planted labels.

The planted labels are never used by the proposed frequency arm.  The tested
values k=2,4,8 and the label-free selection rule were fixed in a development
pilot before the held-out seeds: select the largest k whose gauge result meets
both declared cycle-consistency thresholds, otherwise retain local support.
The fixed k=4 topology is also used for equal-count random and planted-label
oracle controls.  Every outcome is retained.  A gain shared by the gauge and
standard frequency arms is attributed to the new information topology, not to
gauge transport.
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np

from abm_composite_gauge import (
    Net,
    adjusted_rand_index,
    cluster_purity,
    clusters_per_group,
    random_geometric_graph,
)

HERE = os.path.dirname(os.path.abspath(__file__))


def fine_organisation(seed, p, log):
    """Reproduce the corrected level-0 gauge arm through the organisation phase."""
    rng = np.random.default_rng(seed)
    pos, ua, ub = random_geometric_graph(p['N'], p['k'], rng)
    groups = rng.integers(0, p['G'], p['N'])
    centres = p['dOmega'] * (np.arange(p['G']) - (p['G'] - 1) / 2.0)
    omega = centres[groups] + p['sigma_omega'] * rng.standard_normal(p['N'])
    q = np.ones(p['N'])
    delta = p['delta_max'] * rng.uniform(-1, 1, 2 * len(ua))
    # Consume the same draw as the parent production code before Net construction.
    _ = rng.uniform(-np.pi, np.pi, 2 * len(ua))
    net = Net(p['N'], ua, ub, groups, omega, q, delta, p['J0'], 'gauge',
              np.random.default_rng(seed + 1))
    net.theta_rule = 'normalised'
    net.build_in_lists()
    dyn = np.random.default_rng(seed + 7)
    nsteps = int(p['T0'] / p['dt'])
    observe_from = max(0, nsteps - int(p['observation_window'] / p['dt']))
    observe_every = max(1, int(p['observation_every'] / p['dt']))
    snapshots = []
    for s in range(nsteps):
        net.step(p['dt'], p['mu'], p['eta_theta0'], p['eta_J0'], p['D0'], dyn,
                 p['eta_omega0'])
        if s >= observe_from and (s - observe_from) % observe_every == 0:
            snapshots.append((net.z / (np.abs(net.z) + 1e-12)).copy())
    cl = net.detect_clusters(p['thr_frac'], p['min_size'])
    flat = net.flatness_diagnostics(cl, p['thr_frac'], p['flat_tolerance'])
    certified = {row['cluster'] for row in flat['per_cluster'] if row['approximately_flat']}
    log('fine seed {}: candidates={} certified={} assigned={:.3f}'.format(
        seed, int(cl.max() + 1), len(certified), float(np.mean(np.isin(cl, list(certified))))))
    return net, pos, ua, ub, groups, cl, flat, certified, np.asarray(snapshots)


def promote(net, pos, groups, cl, certified, thr_frac, snapshots=None):
    """Promote certified level-0 clusters and retain their root-frame potentials."""
    zhat = net.z / (np.abs(net.z) + 1e-12)
    labels = sorted(certified)
    node_to_super = -np.ones(net.N, dtype=int)
    members = []
    roots = []
    psi_by_node = {}
    Z = []
    omega = []
    xy = []
    majority = []
    sizes = []
    zseries = []
    for cnew, cold in enumerate(labels):
        mem = np.flatnonzero(cl == cold)
        comps = net._strong_components(mem, thr_frac)
        if len(comps) != 1 or len(comps[0]) != len(mem):
            raise RuntimeError('certified candidate is not one strong component')
        _, psi, _ = net._transport_component(mem, thr_frac)
        if len(psi) != len(mem):
            raise RuntimeError('transport tree did not cover certified candidate')
        root = min(psi)
        transported = np.array([np.exp(-1j * psi[int(i)]) * zhat[i] for i in mem])
        zc = transported.mean()
        if abs(zc) < 1e-10:
            raise RuntimeError('promoted state has vanishing amplitude')
        node_to_super[mem] = cnew
        members.append(mem)
        roots.append(root)
        psi_by_node.update({int(i): float(psi[int(i)]) for i in mem})
        Z.append(zc)
        if snapshots is not None and len(snapshots):
            zseries.append(np.mean(snapshots[:, mem] *
                                   np.exp(-1j * np.array([psi[int(i)] for i in mem]))[None, :], axis=1))
        omega.append(float(net.omega[mem].mean()))
        xy.append(pos[mem].mean(axis=0))
        g = groups[mem]
        majority.append(int(np.argmax(np.bincount(g))))
        sizes.append(len(mem))
    return {
        'node_to_super': node_to_super,
        'members': members,
        'roots': roots,
        'psi': psi_by_node,
        'Z': np.asarray(Z, complex),
        'omega': np.asarray(omega, float),
        'pos': np.asarray(xy, float),
        'majority': np.asarray(majority, int),
        'sizes': np.asarray(sizes, int),
        'Z_series': (np.asarray(zseries, complex).T if zseries else np.empty((0, len(members)), complex)),
    }


def scale_information(promoted):
    """Gauge-covariant pair relation and gauge-invariant affinities at level 1.

    The observable mean frequency supplies a frame-independent scale feature.  The final-window
    complex correlation supplies a covariant phase: M_cd -> g_c M_cd g_d^{-1}.  Hidden labels are
    used only after construction to describe the information content, never to form this matrix.
    """
    omega = promoted['omega']
    domega = np.abs(omega[:, None] - omega[None, :])
    frequency_affinity = 1.0 / (1.0 + domega)
    X = promoted['Z_series']
    if len(X):
        X = X / (np.abs(X) + 1e-12)
        M = X.T @ np.conj(X) / len(X)
        temporal_affinity = np.abs(M)
        phase = np.angle(M)
    else:
        M = promoted['Z'][:, None] * np.conj(promoted['Z'][None, :])
        temporal_affinity = np.abs(M) / (np.abs(promoted['Z'])[:, None] * np.abs(promoted['Z'])[None, :] + 1e-12)
        phase = np.angle(M)
    np.fill_diagonal(frequency_affinity, 0.0)
    np.fill_diagonal(temporal_affinity, 0.0)
    return {'frequency_affinity': frequency_affinity,
            'temporal_affinity': temporal_affinity,
            'phase': phase}


def boundary_support(net, promoted):
    """Return coarse boundary pairs and root-dressed directed phases."""
    M = net.E // 2
    ua, ub = net.dst[:M], net.src[:M]
    phi = net.theta + net.delta
    nts = promoted['node_to_super']
    psi = promoted['psi']
    terms = {}
    pairs = set()
    for e, (a0, b0) in enumerate(zip(ua, ub)):
        a, b = int(a0), int(b0)
        ca, cb = int(nts[a]), int(nts[b])
        if ca < 0 or cb < 0 or ca == cb:
            continue
        pair = tuple(sorted((ca, cb)))
        pairs.add(pair)
        phase_ab = -psi[a] + float(phi[e]) + psi[b]       # ca <- cb
        er = int(net.rev[e])
        phase_ba = -psi[b] + float(phi[er]) + psi[a]      # cb <- ca
        terms.setdefault((ca, cb), []).append(float(net.J[e]) * np.exp(1j * phase_ab))
        terms.setdefault((cb, ca), []).append(float(net.J[er]) * np.exp(1j * phase_ba))
    directed_phase = {}
    directed_strength = {}
    for key, vals in terms.items():
        s = np.sum(vals)
        directed_phase[key] = float(np.angle(s)) if abs(s) else 0.0
        directed_strength[key] = float(abs(s) / len(vals)) if vals else 0.0
    return pairs, directed_phase, directed_strength


def knn_frequency_pairs(omega, k):
    pairs = set()
    n = len(omega)
    for i in range(n):
        order = np.argsort(np.abs(omega - omega[i]) + (np.arange(n) == i) * 1e12)
        for j in order[:min(k, n - 1)]:
            pairs.add(tuple(sorted((int(i), int(j)))))
    return pairs


def oracle_pairs(omega, majority, k):
    pairs = set()
    for i in range(len(omega)):
        eligible = np.flatnonzero((majority == majority[i]) & (np.arange(len(omega)) != i))
        if len(eligible) == 0:
            continue
        order = eligible[np.argsort(np.abs(omega[eligible] - omega[i]))]
        for j in order[:min(k, len(order))]:
            pairs.add(tuple(sorted((int(i), int(j)))))
    return pairs


def random_added_pairs(n, boundary, nadd, rng):
    available = [(i, j) for i in range(n) for j in range(i + 1, n) if (i, j) not in boundary]
    if nadd >= len(available):
        return set(available)
    take = rng.choice(len(available), size=nadd, replace=False)
    return {available[int(i)] for i in take}


def support_arrays(pairs, boundary_phase, information_phase, rng):
    ordered = sorted(pairs)
    ua = np.asarray([x[0] for x in ordered], int)
    ub = np.asarray([x[1] for x in ordered], int)
    delta = np.empty(2 * len(ordered), float)
    for e, (a, b) in enumerate(ordered):
        # Existing physical boundaries retain their root-dressed fine connection.  A newly
        # proposed scale-level link is initialised from the covariant phase of the observed
        # composite relation.  The adaptive transporter may subsequently change it.
        delta[e] = boundary_phase.get((a, b), float(information_phase[a, b]))
        delta[e + len(ordered)] = boundary_phase.get((b, a), float(information_phase[b, a]))
    return ua, ub, delta


def fine_partition_metrics(promoted, groups):
    labels = -np.ones(len(groups), int)
    for c, mem in enumerate(promoted['members']):
        labels[mem] = c
    mask = labels >= 0
    return {
        'assigned_fraction': float(mask.mean()),
        'n_modules': int(len(promoted['members'])),
        'ARI_vs_groups': float(adjusted_rand_index(labels[mask], groups[mask])) if mask.sum() > 1 else None,
        'purity': cluster_purity(labels, groups),
        'clusters_per_group': clusters_per_group(labels, groups, int(groups.max() + 1)),
        'mean_module_size': float(np.mean(promoted['sizes'])) if len(promoted['sizes']) else None,
    }


def mapped_metrics(coarse_labels, promoted, groups):
    labels = -np.ones(len(groups), int)
    for c, mem in enumerate(promoted['members']):
        labels[mem] = int(coarse_labels[c])
    mask = labels >= 0
    sizes = np.bincount(labels[mask]) if mask.any() else np.array([], int)
    return {
        'assigned_fraction': float(mask.mean()),
        'n_modules': int(len(np.unique(coarse_labels))),
        'ARI_vs_groups': float(adjusted_rand_index(labels[mask], groups[mask])) if mask.sum() > 1 else None,
        'purity': cluster_purity(labels, groups),
        'clusters_per_group': clusters_per_group(labels, groups, int(groups.max() + 1)),
        'mean_module_size': float(sizes.mean()) if len(sizes) else None,
        'max_module_size': int(sizes.max()) if len(sizes) else None,
    }


def covariance_control(net, labels, thr_frac, flat_tolerance):
    z0, th0 = net.z.copy(), net.theta.copy()
    zhat0 = z0 / (np.abs(z0) + 1e-12)
    m0 = np.conj(zhat0[net.dst]) * net.U() * zhat0[net.src]
    f0 = net.flatness_diagnostics(labels, thr_frac, flat_tolerance)
    alpha = np.random.default_rng(9182).uniform(-np.pi, np.pi, net.N)
    net.z = np.exp(1j * alpha) * net.z
    net.theta = net.theta + alpha[net.dst] - alpha[net.src]
    zhat1 = net.z / (np.abs(net.z) + 1e-12)
    m1 = np.conj(zhat1[net.dst]) * net.U() * zhat1[net.src]
    f1 = net.flatness_diagnostics(labels, thr_frac, flat_tolerance)
    dev = max(float(np.max(np.abs(m1 - m0))) if len(m0) else 0.0,
              abs((f1['residual_abs_max'] or 0.0) - (f0['residual_abs_max'] or 0.0)))
    net.z, net.theta = z0, th0
    return {'max_deviation': dev, 'PASS': bool(dev < 1e-9)}


def run_coarse(seed, name, model, support, delta, promoted, groups, p, log):
    ua, ub = support
    net = Net(len(promoted['members']), ua, ub, promoted['majority'], promoted['omega'],
              np.ones(len(promoted['members'])), delta, p['J1'], model,
              np.random.default_rng(seed + 3001))
    net.theta_rule = 'normalised'
    net.build_in_lists()
    net.z = promoted['Z'].copy()
    dyn = np.random.default_rng(seed + 4001)
    nsteps = int(p['T1_coarse'] / p['dt'])
    every = max(1, int(p['record_every'] / p['dt']))
    series = []
    for s in range(nsteps):
        net.step(p['dt'], p['mu'], p['eta_theta1'], p['eta_J1'], p['D1'], dyn,
                 p['eta_omega1'])
        if s % every == 0 or s == nsteps - 1:
            cc = net.detect_clusters(p['thr_frac'], 1)
            mm = mapped_metrics(cc, promoted, groups)
            series.append([float(s * p['dt']), mm['ARI_vs_groups'], mm['n_modules'],
                           float(net.J.mean() / p['J1']) if len(net.J) else 0.0])
    cc = net.detect_clusters(p['thr_frac'], 1)
    result = mapped_metrics(cc, promoted, groups)
    result['n_supernodes'] = int(net.N)
    result['n_support_edges'] = int(len(ua))
    result['support_same_group_fraction'] = (float(np.mean(promoted['majority'][ua] == promoted['majority'][ub]))
                                             if len(ua) else None)
    result['mean_J_over_J1'] = float(net.J.mean() / p['J1']) if len(net.J) else None
    result['merge_factor'] = float(net.N / result['n_modules']) if result['n_modules'] else None
    flat = net.flatness_diagnostics(cc, p['thr_frac'], p['flat_tolerance'])
    result['cycle_residual_abs_q95'] = flat['residual_abs_q95']
    result['cycle_residual_abs_max'] = flat['residual_abs_max']
    result['candidate_units_certified_flat'] = flat['fraction_candidate_units_certified']
    result['covariance_control'] = covariance_control(net, cc, p['thr_frac'], p['flat_tolerance'])
    result['series'] = series
    log('  {:24s} modules={:3d} ARI={:.3f} purity={:.3f} support_hom={:.3f}'.format(
        name, result['n_modules'], result['ARI_vs_groups'], result['purity'],
        result['support_same_group_fraction'] if result['support_same_group_fraction'] is not None else -1.0))
    return result


def run_seed(seed, p, log):
    net, pos, ua0, ub0, groups, cl, flat, certified, snapshots = fine_organisation(seed, p, log)
    promoted = promote(net, pos, groups, cl, certified, p['thr_frac'], snapshots)
    if len(promoted['members']) < 2:
        raise RuntimeError('fewer than two certified composites; no hierarchy can be tested')
    boundary, boundary_phase, _ = boundary_support(net, promoted)
    info = scale_information(promoted)
    base = fine_partition_metrics(promoted, groups)
    out = {
        'seed': seed,
        'level0': base,
        'n_boundary_pairs': len(boundary),
        'n_certified_composites': len(promoted['members']),
        'scale_information': {
            'n_observations': int(len(promoted['Z_series'])),
            'frequency_affinity_same_group_mean': float(info['frequency_affinity'][(promoted['majority'][:, None] == promoted['majority'][None, :]) & ~np.eye(len(promoted['majority']), dtype=bool)].mean()),
            'frequency_affinity_different_group_mean': float(info['frequency_affinity'][promoted['majority'][:, None] != promoted['majority'][None, :]].mean()),
            'temporal_affinity_same_group_mean': float(info['temporal_affinity'][(promoted['majority'][:, None] == promoted['majority'][None, :]) & ~np.eye(len(promoted['majority']), dtype=bool)].mean()),
            'temporal_affinity_different_group_mean': float(info['temporal_affinity'][promoted['majority'][:, None] != promoted['majority'][None, :]].mean()),
        },
        'arms': {},
    }

    def run_arm(name, model, pairs, phase_seed):
        u, v, d = support_arrays(pairs, boundary_phase, info['phase'], np.random.default_rng(phase_seed))
        out['arms'][name] = run_coarse(seed, name, model, (u, v), d, promoted, groups, p, log)

    run_arm('local_gauge', 'gauge', boundary, seed + 5001)
    run_arm('local_standard', 'standard', boundary, seed + 5001)
    for k in p['coarse_k_values']:
        feature_added = knn_frequency_pairs(promoted['omega'], k) - boundary
        feature = boundary | feature_added
        run_arm('frequency_gauge_k{}'.format(k), 'gauge', feature, seed + 6000 + k)
        run_arm('frequency_standard_k{}'.format(k), 'standard', feature, seed + 6000 + k)
        if k == p['control_k']:
            random_added = random_added_pairs(len(promoted['members']), boundary, len(feature_added),
                                              np.random.default_rng(seed + 7000 + k))
            oracle_added = oracle_pairs(promoted['omega'], promoted['majority'], k) - boundary
            run_arm('random_gauge_k{}'.format(k), 'gauge', boundary | random_added, seed + 8000 + k)
            run_arm('oracle_gauge_k{}'.format(k), 'gauge', boundary | oracle_added, seed + 9000 + k)
            out['primary_support'] = {
                'feature_added_edges': len(feature_added),
                'random_added_edges': len(random_added),
                'oracle_added_edges': len(oracle_added),
            }
    # Gauge-defined scale selection: choose the densest tested information graph for which
    # the complete cycle diagnostic still certifies at least the declared fraction of promoted
    # units and the q95 residual is below tolerance.  No planted label enters this rule.
    admissible = []
    for k in p['coarse_k_values']:
        row = out['arms']['frequency_gauge_k{}'.format(k)]
        if ((row['cycle_residual_abs_q95'] is not None and
             row['cycle_residual_abs_q95'] <= p['flat_tolerance']) and
                (row['candidate_units_certified_flat'] is not None and
                 row['candidate_units_certified_flat'] >= p['scale_certified_fraction'])):
            admissible.append(k)
    selected_k = max(admissible) if admissible else 0
    gname = 'frequency_gauge_k{}'.format(selected_k) if selected_k else 'local_gauge'
    sname = 'frequency_standard_k{}'.format(selected_k) if selected_k else 'local_standard'
    out['scale_selection'] = {
        'rule': 'largest k with gauge q95 <= flat_tolerance and certified fraction >= scale_certified_fraction',
        'admissible_k': admissible,
        'selected_k': selected_k,
        'abstained_to_local_support': bool(not admissible),
        'gauge_arm': gname,
        'matched_standard_arm': sname,
        'gauge_metrics': {key: out['arms'][gname][key] for key in
                          ['ARI_vs_groups', 'purity', 'n_modules', 'clusters_per_group',
                           'cycle_residual_abs_q95', 'candidate_units_certified_flat']},
        'matched_standard_metrics': {key: out['arms'][sname][key] for key in
                                     ['ARI_vs_groups', 'purity', 'n_modules', 'clusters_per_group',
                                      'cycle_residual_abs_q95', 'candidate_units_certified_flat']},
    }
    log('  scale selection: admissible={} selected={} gauge_ARI={:.3f} standard_ARI={:.3f}'.format(
        admissible, selected_k, out['arms'][gname]['ARI_vs_groups'], out['arms'][sname]['ARI_vs_groups']))
    return out


def aggregate(runs):
    names = sorted(runs[0]['arms'])
    keys = ['assigned_fraction', 'n_modules', 'ARI_vs_groups', 'purity', 'clusters_per_group',
            'mean_module_size', 'max_module_size', 'n_supernodes', 'n_support_edges',
            'support_same_group_fraction', 'mean_J_over_J1', 'merge_factor',
            'cycle_residual_abs_q95', 'cycle_residual_abs_max', 'candidate_units_certified_flat']

    def ms(vals):
        vals = [v for v in vals if v is not None]
        return {'mean': float(np.mean(vals)) if vals else None,
                'sd': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                'n': len(vals)}

    out = {'level0': {k: ms([r['level0'][k] for r in runs]) for k in
                      ['assigned_fraction', 'n_modules', 'ARI_vs_groups', 'purity',
                       'clusters_per_group', 'mean_module_size']}, 'arms': {}}
    for name in names:
        out['arms'][name] = {k: ms([r['arms'][name][k] for r in runs]) for k in keys}
        out['arms'][name]['covariance_PASS'] = bool(all(r['arms'][name]['covariance_control']['PASS']
                                                        for r in runs if 'gauge' in name)) if 'gauge' in name else None
    fixed = 'frequency_gauge_k4'
    fair = 'frequency_standard_k4'
    random = 'random_gauge_k4'
    oracle = 'oracle_gauge_k4'
    if fixed in names:
        out['paired_fixed_k4_control'] = {
            'gauge_minus_level0_ARI': ms([r['arms'][fixed]['ARI_vs_groups'] - r['level0']['ARI_vs_groups'] for r in runs]),
            'gauge_minus_standard_ARI': ms([r['arms'][fixed]['ARI_vs_groups'] - r['arms'][fair]['ARI_vs_groups'] for r in runs]),
            'gauge_minus_random_ARI': ms([r['arms'][fixed]['ARI_vs_groups'] - r['arms'][random]['ARI_vs_groups'] for r in runs]),
            'oracle_minus_level0_ARI': ms([r['arms'][oracle]['ARI_vs_groups'] - r['level0']['ARI_vs_groups'] for r in runs]),
        }
    out['scale_selected'] = {
        'nonzero_selection_fraction': ms([float(r['scale_selection']['selected_k'] > 0) for r in runs]),
        'selected_k': ms([r['scale_selection']['selected_k'] for r in runs]),
        'gauge_ARI': ms([r['scale_selection']['gauge_metrics']['ARI_vs_groups'] for r in runs]),
        'standard_same_k_ARI': ms([r['scale_selection']['matched_standard_metrics']['ARI_vs_groups'] for r in runs]),
        'gauge_purity': ms([r['scale_selection']['gauge_metrics']['purity'] for r in runs]),
        'gauge_n_modules': ms([r['scale_selection']['gauge_metrics']['n_modules'] for r in runs]),
        'gauge_cycle_q95': ms([r['scale_selection']['gauge_metrics']['cycle_residual_abs_q95'] for r in runs]),
        'standard_same_k_cycle_q95': ms([r['scale_selection']['matched_standard_metrics']['cycle_residual_abs_q95'] for r in runs]),
        'gauge_certified_fraction': ms([r['scale_selection']['gauge_metrics']['candidate_units_certified_flat'] for r in runs]),
        'standard_same_k_certified_fraction': ms([r['scale_selection']['matched_standard_metrics']['candidate_units_certified_flat'] for r in runs]),
        'gauge_minus_level0_ARI': ms([r['scale_selection']['gauge_metrics']['ARI_vs_groups'] - r['level0']['ARI_vs_groups'] for r in runs]),
        'gauge_minus_standard_same_k_ARI': ms([r['scale_selection']['gauge_metrics']['ARI_vs_groups'] - r['scale_selection']['matched_standard_metrics']['ARI_vs_groups'] for r in runs]),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='hierarchy_main')
    ap.add_argument('--N', type=int, default=1000)
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--seed_start', type=int, default=10)
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--T0', type=float, default=800.0)
    ap.add_argument('--T1_coarse', type=float, default=800.0)
    a = ap.parse_args()
    p = {
        'schema_version': 2,
        'N': a.N,
        'k': 12,
        'G': 4,
        'dOmega': 2.0,
        'sigma_omega': 0.05,
        'mu': 1.0,
        'J0': 0.3,
        'J1': 0.3,
        'eta_theta0': 0.05,
        'eta_J0': 0.005,
        'eta_omega0': 0.005,
        'eta_theta1': 0.025,
        'eta_J1': 0.0025,
        'eta_omega1': 0.0025,
        'D0': 0.005,
        'D1': 0.001,
        'dt': 0.05,
        'T0': a.T0,
        'T1_coarse': a.T1_coarse,
        'record_every': 10.0,
        'observation_window': 100.0,
        'observation_every': 1.0,
        'thr_frac': 0.5,
        'min_size': 5,
        'delta_max': float(np.pi),
        'flat_tolerance': 0.01,
        'coarse_k_values': [2, 4, 8],
        'control_k': 4,
        'scale_certified_fraction': 0.95,
        'seeds': list(range(a.seed_start, a.seed_start + a.seeds)),
        'code_sha256': hashlib.sha256(open(os.path.abspath(__file__), 'rb').read()).hexdigest(),
        'base_code_sha256': hashlib.sha256(open(os.path.join(HERE, 'abm_composite_gauge.py'), 'rb').read()).hexdigest(),
    }
    if a.quick:
        p.update(N=min(a.N, 250), T0=min(a.T0, 400.0), T1_coarse=min(a.T1_coarse, 400.0), seeds=[0])
        if a.tag == 'hierarchy_main':
            a.tag = 'hierarchy_quick'
    os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
    log_path = os.path.join(HERE, 'results', 'abm_{}.log'.format(a.tag))
    result_path = os.path.join(HERE, 'results', 'abm_{}.json'.format(a.tag))
    logf = open(log_path, 'w')

    def log(msg):
        line = '[{}] {}'.format(time.strftime('%H:%M:%S'), msg)
        print(line, flush=True)
        logf.write(line + '\n')
        logf.flush()

    log('params {}'.format(json.dumps(p, sort_keys=True)))
    runs = []
    start = time.time()
    for i, seed in enumerate(p['seeds']):
        runs.append(run_seed(seed, p, log))
        payload = {'params': p, 'runs': runs, 'aggregate': aggregate(runs)}
        with open(result_path, 'w') as f:
            json.dump(payload, f, indent=2)
            f.write('\n')
        elapsed = time.time() - start
        log('seed {} done, elapsed {:.1f}s, ETA {:.1f}s'.format(
            seed, elapsed, elapsed / (i + 1) * (len(p['seeds']) - i - 1)))
    agg = aggregate(runs)
    log('FIXED_K4_CONTROL {}'.format(json.dumps(agg.get('paired_fixed_k4_control', {}), sort_keys=True)))
    all_cov = all(v.get('covariance_PASS') in (True, None) for v in agg['arms'].values())
    log('CONTROLS hierarchy static covariance: {}'.format('PASS' if all_cov else 'FAIL'))
    log('DONE')


if __name__ == '__main__':
    main()
