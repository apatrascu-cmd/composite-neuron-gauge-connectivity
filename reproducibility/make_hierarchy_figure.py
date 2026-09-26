#!/usr/bin/env python3
"""Render the held-out hierarchical scale-selection diagnostics from archived JSON."""
import argparse
import json
import os
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'manuscript', 'figures')


def mean_sd(vals):
    vals = np.asarray(vals, float)
    return vals.mean(), vals.std(ddof=1) if len(vals) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='hierarchy_main')
    args = ap.parse_args()
    with open(os.path.join(HERE, 'results', 'abm_{}.json'.format(args.tag))) as f:
        data = json.load(f)
    runs = data['runs']
    if not runs:
        raise SystemExit('HIERARCHY_FIGURE_FAIL: no completed runs')
    os.makedirs(OUT, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.7))
    ax = axes[0, 0]
    counts = Counter(int(r['scale_selection']['selected_k']) for r in runs)
    ks = sorted(set([0] + list(data['params']['coarse_k_values'])))
    ax.bar([str(k) for k in ks], [counts.get(k, 0) for k in ks], color='#4477AA')
    ax.set_xlabel('label-free selected k (0 = local support)')
    ax.set_ylabel('held-out seeds')
    ax.set_title('(a) Scale selected without planted labels')

    ax = axes[0, 1]
    labels = ['fine', 'selected\ngauge', 'same k,\nno transporter', 'random\nk=4', 'oracle\nk=4']
    vals = [
        [r['level0']['ARI_vs_groups'] for r in runs],
        [r['scale_selection']['gauge_metrics']['ARI_vs_groups'] for r in runs],
        [r['scale_selection']['matched_standard_metrics']['ARI_vs_groups'] for r in runs],
        [r['arms']['random_gauge_k4']['ARI_vs_groups'] for r in runs],
        [r['arms']['oracle_gauge_k4']['ARI_vs_groups'] for r in runs],
    ]
    for i, yy in enumerate(vals):
        x = i + np.linspace(-0.08, 0.08, len(yy))
        ax.scatter(x, yy, s=24, color='#228833', alpha=0.85, zorder=3)
        m, s = mean_sd(yy)
        ax.errorbar(i, m, yerr=s, color='black', marker='_', markersize=15, capsize=4, zorder=4)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(-0.02, 1.05)
    ax.set_ylabel('ARI against planted groups')
    ax.set_title('(b) Recovery and controls')

    colors = {'gauge': '#AA3377', 'standard': '#777777'}
    ax = axes[1, 0]
    for kind, label in [('gauge', 'gauge transporter'), ('standard', 'no transporter')]:
        means, sds = [], []
        for k in data['params']['coarse_k_values']:
            y = [r['arms']['frequency_{}_k{}'.format(kind, k)]['cycle_residual_abs_q95'] for r in runs]
            m, s = mean_sd(y); means.append(m); sds.append(s)
        ax.errorbar(data['params']['coarse_k_values'], means, yerr=sds, marker='o', capsize=3,
                    color=colors[kind], label=label)
    ax.axhline(data['params']['flat_tolerance'], color='#CC3311', linestyle='--', linewidth=1,
               label='flatness threshold')
    ax.set_yscale('log')
    ax.set_xlabel('information-neighbour count k')
    ax.set_ylabel('complete-cycle residual q95 (rad)')
    ax.set_title('(c) Global consistency across scale')
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    for kind, label in [('gauge', 'gauge transporter'), ('standard', 'no transporter')]:
        means, sds = [], []
        for k in data['params']['coarse_k_values']:
            y = [r['arms']['frequency_{}_k{}'.format(kind, k)]['candidate_units_certified_flat'] for r in runs]
            m, s = mean_sd(y); means.append(m); sds.append(s)
        ax.errorbar(data['params']['coarse_k_values'], means, yerr=sds, marker='o', capsize=3,
                    color=colors[kind], label=label)
    ax.axhline(data['params']['scale_certified_fraction'], color='#CC3311', linestyle='--', linewidth=1,
               label='selection threshold')
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel('information-neighbour count k')
    ax.set_ylabel('promoted units in certified modules')
    ax.set_title('(d) Certification across scale')
    ax.legend(fontsize=7)

    fig.tight_layout()
    for ext in ('pdf', 'png'):
        path = os.path.join(OUT, 'fig5_hierarchy.' + ext)
        fig.savefig(path, dpi=180 if ext == 'png' else None, bbox_inches='tight')
        print(path)
    plt.close(fig)


if __name__ == '__main__':
    main()
