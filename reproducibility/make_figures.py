#!/usr/bin/env python3
"""Generate the figures of the manuscript from the simulation result files.
Usage: python3 make_figures.py [--abm_tag main_v2] [--seed_index 0]
Writes ../manuscript/figures/fig1_schematic.pdf, fig2_network.pdf,
fig3_cycle_flatness.pdf and fig4_timeseries.pdf (and .png previews)."""
import json, os, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Ellipse, Patch
from matplotlib.collections import LineCollection
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, '..', 'manuscript', 'figures'); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({'font.size': 9, 'axes.labelsize': 9, 'legend.fontsize': 8, 'pdf.fonttype': 42})

def save(fig, name):
    fig.savefig(os.path.join(FIG, name + '.pdf'), bbox_inches='tight'); fig.savefig(os.path.join(FIG, name + '.png'), dpi=160, bbox_inches='tight'); plt.close(fig)
    print('wrote', name)

def fig1_schematic():
    fig, ax = plt.subplots(figsize=(6.6, 3.4)); ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis('off')
    def cluster(cx, cy, n, r, label, col):
        ax.add_patch(Ellipse((cx, cy), 2.6 * r, 2.2 * r, fc=col, ec='none', alpha=0.18))
        ang = 2 * np.pi * np.arange(n) / n; xs = cx + r * np.cos(ang); ys = cy + 0.85 * r * np.sin(ang)
        for i in range(n):
            for j in range(i + 1, n):
                ax.plot([xs[i], xs[j]], [ys[i], ys[j]], color=col, lw=0.6, alpha=0.6, zorder=1)
        for x, y in zip(xs, ys): ax.add_patch(Circle((x, y), 0.14, fc='white', ec=col, lw=1.4, zorder=3))
        ax.text(cx, cy - 1.15 * r - 0.25, label, ha='center', va='top', fontsize=8.5)
        return xs, ys
    xa, ya = cluster(2.3, 3.1, 6, 1.0, 'candidate composite $C_1$\n(connected strong-coupling subgraph;\ncertified when every cycle residual is small;\ntransported operator $Z_{C_1}$)', 'tab:blue')
    xb, yb = cluster(7.6, 3.1, 5, 0.85, 'composite cluster $C_2$', 'tab:green')
    # elementary units
    for (x, y, lab) in [(5.0, 1.0, 'elementary unit $i$\n$\\epsilon_i$: mixing with $C_1$'), (5.0, 4.3, 'elementary unit $k$')]:
        ax.add_patch(Circle((x, y), 0.16, fc='white', ec='tab:red', lw=1.6, zorder=3)); ax.text(x, y - 0.32, lab, ha='center', va='top', fontsize=8)
    # links with transporter labels
    def link(p, q, text, dy=0.18, col='k'):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=9, lw=1.0, color=col, shrinkA=6, shrinkB=6, zorder=2))
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + dy, text, ha='center', fontsize=7.5, color=col)
    link((xa[0], ya[0]), (5.0, 1.0), '$J_{ij}\\,e^{i(\\theta_{ij}+\\delta_{ij})}$', dy=0.22, col='tab:purple')
    link((5.0, 1.0), (xb[2], yb[2]), '$J$, $\\theta$, $\\delta$', dy=0.22, col='tab:purple')
    link((xa[1], ya[1]), (5.0, 4.3), '$\\theta_{ki}$ adapts', dy=0.2, col='tab:purple')
    link((5.0, 4.3), (xb[1], yb[1]), '$\\delta$ fixed', dy=0.2, col='tab:purple')
    ax.text(5.0, 2.55, 'local phase transformation:\n$z_i\\to e^{i\\alpha_i}z_i$, $\\theta_{ij}\\to\\theta_{ij}+\\alpha_i-\\alpha_j$\n(offsets $\\delta_{ij}$ do not change)', ha='center', va='center', fontsize=8,
            bbox=dict(boxstyle='round', fc='white', ec='0.6'))
    save(fig, 'fig1_schematic')

def fig2_network(run, J0):
    fd = run['figure_data']; pos = np.array(fd['pos']); groups = np.array(fd['groups']); ua = np.array(fd['ua']); ub = np.array(fd['ub'])
    models = [m for m in ['gauge', 'standard'] if m in fd['clusters']]
    fig, axes = plt.subplots(1, 1 + len(models), figsize=(3.2 * (1 + len(models)), 3.4))
    cmap = plt.get_cmap('viridis')
    def draw(ax, colors, Js, title):
        strong = Js > 0.5 * J0
        segs = np.stack([pos[ua], pos[ub]], axis=1)
        ax.add_collection(LineCollection(segs[~strong], colors='0.85', linewidths=0.3, zorder=1))
        ax.add_collection(LineCollection(segs[strong], colors='0.35', linewidths=0.6, zorder=2))
        ax.scatter(pos[:, 0], pos[:, 1], c=colors, s=14, zorder=3, edgecolors='k', linewidths=0.2)
        ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect('equal'); ax.autoscale()
    Jref = np.array(fd['J_sym'][models[0]])
    group_cmap = plt.get_cmap('Set1')
    draw(axes[0], [group_cmap(g) for g in groups], Jref, '(a) planted frequency groups')
    axes[0].legend(handles=[Patch(facecolor=group_cmap(g), edgecolor='k', linewidth=0.2, label=f'group {g}')
                            for g in sorted(np.unique(groups))], loc='upper right', fontsize=6, frameon=True)
    for k, m in enumerate(models):
        cl = np.array(fd['clusters'][m]); Js = np.array(fd['J_sym'][m])
        n_cl = int(cl.max() + 1); comp = float((cl >= 0).mean())
        norm = Normalize(vmin=0, vmax=max(n_cl - 1, 1))
        cols = ['0.75' if c < 0 else cmap(norm(c)) for c in cl]
        draw(axes[k + 1], cols, Js, f'({"bc"[k]}) {"gauge network" if m == "gauge" else "delay-only network"}\n{n_cl} candidates, {comp:.0%} of units assigned')
        sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
        cb = fig.colorbar(sm, ax=axes[k + 1], fraction=0.046, pad=0.02)
        cb.set_label('candidate-cluster ID', fontsize=7); cb.ax.tick_params(labelsize=6)
    save(fig, 'fig2_network')

def fig3_cycle_flatness(run, tag):
    """Complete flatness diagnostics: reciprocal two-cycles plus a fundamental cycle basis."""
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.75))
    colors = {'gauge': 'tab:blue', 'standard': 'tab:orange', 'quenched': 'tab:green'}
    labels = {'gauge': 'gauge', 'standard': 'delay-only', 'quenched': 'quenched'}
    models = [m for m in ['gauge', 'standard', 'quenched'] if m in run['models']]
    tol = None
    for m in models:
        d = run['models'][m]['cycle_basis_diagnostics']
        edges = np.asarray(d['hist_log_edges'], float); counts = np.asarray(d['hist_log_counts'], float)
        centers = np.sqrt(edges[:-1] * edges[1:]); n = counts.sum()
        if n:
            axes[0].step(centers, counts, where='mid', lw=1.4,
                         color=colors[m], label=f'{labels[m]} ($n={int(n)}$ constraints)')
        tol = d['tolerance']
    axes[0].axvline(tol, color='k', ls=':', lw=1, label=f'tolerance {tol:g} rad')
    axes[0].set_xscale('log'); axes[0].set_yscale('log')
    axes[0].set_xlim(1e-8, np.pi); axes[0].set_xlabel('absolute cycle-consistency residual (rad)')
    axes[0].set_ylabel('number of constraints (log scale)'); axes[0].set_title('(a) all independent cycle constraints', fontsize=9)
    axes[0].legend(fontsize=6.2)
    x = np.arange(len(models)); w = 0.34
    pass_fraction = [run['models'][m]['cycle_constraints_below_tolerance'] for m in models]
    certified = [run['models'][m]['candidate_units_certified_flat'] for m in models]
    axes[1].bar(x - w/2, pass_fraction, width=w, color=[colors[m] for m in models], alpha=0.8,
                label='constraints below tolerance')
    axes[1].bar(x + w/2, certified, width=w, color=[colors[m] for m in models], alpha=0.35, hatch='//',
                label='candidate units in certified clusters')
    axes[1].set_xticks(x, [labels[m] for m in models], rotation=15)
    axes[1].set_ylim(0, 1.05); axes[1].set_ylabel('fraction')
    axes[1].set_title('(b) constraint and cluster-level certification', fontsize=9)
    axes[1].legend(fontsize=6.2, loc='upper right')
    save(fig, 'fig3_cycle_flatness')

def fig4_timeseries(run, G):
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.15))
    ax = axes[0]
    for m, ls, lab, col in [('gauge', '-', 'gauge', 'tab:blue'), ('standard', '--', 'delay-only', 'tab:orange'), ('quenched', ':', 'quenched', 'tab:green')]:
        if m not in run['models']: continue
        S = np.array(run['models'][m]['series'], float)
        raw_group_mean = S[:, 1:1 + G].mean(axis=1)
        ax.plot(S[:, 0], raw_group_mean, ls, color=col, lw=1.2, label=lab)
        mean_label = 'pale curves: mean $J/J_0$' if m == 'gauge' else '_nolegend_'
        ax.plot(S[:, 0], S[:, 1 + G], ls, color=col, lw=0.8, alpha=0.45, label=mean_label)
    ax.set_xlabel('time'); ax.set_ylabel('raw group coherence / mean $J/J_0$')
    ax.set_title('(a) organisation phase', fontsize=9); ax.legend(fontsize=5.9, ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.22)); ax.set_ylim(0, 1.05)
    ax = axes[1]
    for m, ls, lab in [('gauge', '-', 'gauge'), ('standard', '--', 'delay-only')]:
        if m not in run['models']: continue
        S = np.array(run['models'][m]['tests']['delay_shock']['series'], float)
        ax.plot(S[:, 0], S[:, 1], ls, color='tab:gray', lw=1, label=f'{lab}: raw')
        ax.plot(S[:, 0], S[:, 2], ls, color='tab:blue' if m == 'gauge' else 'tab:orange', lw=1.2,
                label=f'{lab}: cov. $\\times$ coverage')
    ax.set_xlabel('time after the offset redraw'); ax.set_ylabel('mean cluster coherence'); ax.set_title('(b) offset redraw', fontsize=9); ax.legend(fontsize=6.1, ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.22)); ax.set_ylim(0, 1.05)
    fig.subplots_adjust(bottom=0.30, wspace=0.30)
    save(fig, 'fig4_timeseries')

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--abm_tag', default='main_v2'); ap.add_argument('--seed_index', type=int, default=0); a = ap.parse_args()
    fig1_schematic()
    f = os.path.join(HERE, 'results', f'abm_{a.abm_tag}.json')
    if os.path.exists(f):
        d = json.load(open(f)); run = d['runs'][a.seed_index]; P = d['params']
        fig2_network(run, P['J0']); fig3_cycle_flatness(run, a.abm_tag); fig4_timeseries(run, P['G'])
    else:
        print('no', f, '- only the schematic was drawn')
