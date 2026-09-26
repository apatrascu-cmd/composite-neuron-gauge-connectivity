#!/usr/bin/env python3
"""Generate every number of the manuscript from the result JSON files.
Writes ../manuscript/tables/numbers.tex, the LaTeX tables, and matching plain CSV tables.
Usage: python3 make_tables.py [--abm_tag main_v2] [--abm_mild_tag mild_v2]
Every macro name is letters only (LaTeX); the mapping key -> macro is listed in MACROS below."""
import csv, json, os, glob, re, argparse, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'manuscript', 'tables')
os.makedirs(OUT, exist_ok=True)

def fmt(x, nd=2):
    if x is None: return '--'
    return f'{x:.{nd}f}'

def ms(v):
    """aggregate entries may be dicts {'mean','sd','n',...} or lists [mean, sd, n]"""
    if isinstance(v, dict): return v.get('mean'), v.get('sd'), v.get('n')
    if isinstance(v, (list, tuple)): return (list(v) + [None, None, None])[:3]
    return v, None, None

def pm(v, nd=2):
    m, s, n = ms(v)
    if m is None: return '--'
    return f'{m:.{nd}f}' if (s is None or n in (None, 1)) else f'{m:.{nd}f} $\\pm$ {s:.{nd}f}'

def sci(x):
    if x is None: return '--'
    a, b = f'{x:.1e}'.split('e')
    return f'{a}\\times 10^{{{int(b)}}}'

def sci_pm(v):
    m, s, n = ms(v)
    if m is None: return '--'
    if s is None or n in (None, 1): return '$' + sci(m) + '$'
    return '$' + sci(m) + ' \\pm ' + sci(s) + '$'

def plain_label(label):
    """Readable CSV label corresponding to a LaTeX table-row label."""
    replacements = {
        '$R^{\\mathrm{raw}}$': 'R_raw',
        '$R^{\\mathrm{cov}}_{\\mathrm{covg}}$': 'R_cov_coverage_weighted',
        '$R^{\\mathrm{cov}}$': 'R_cov',
        '$J/J_0$': 'J/J0',
        '$\\omega$': 'omega',
        '\\%': '%',
        's.d.\\ ': 's.d. ',
    }
    for old, new in replacements.items():
        label = label.replace(old, new)
    return label.replace('$', '').replace('{', '').replace('}', '').replace('\\mathrm', '').replace('\\', '')

MODELS = [('gauge', 'Gauge network'), ('standard', 'Delay-only network'), ('quenched', 'Quenched transporters')]
# (aggregate key, macro suffix, table label, decimals)
MAIN_ROWS = [
    ('n_clusters', 'Nclusters', 'number of clusters', 1),
    ('frac_composite', 'FracComposite', 'fraction of composite units', 2),
    ('ARI_vs_groups', 'ARI', 'ARI clusters vs hidden groups', 2),
    ('cluster_purity', 'Purity', 'cluster purity with respect to the hidden groups', 2),
    ('clusters_per_group_mean', 'ClustersPerGroup', 'clusters per hidden group', 1),
    ('R_raw_mean', 'Rraw', 'raw cluster coherence $R^{\\mathrm{raw}}$', 2),
    ('R_cov_mean', 'Rcov', 'coverage-weighted transported coherence $R^{\\mathrm{cov}}_{\\mathrm{covg}}$', 2),
    ('R_cov_lcc_mean', 'RcovLcc', 'transported coherence inside the largest component', 2),
    ('largest_component_fraction_mean', 'Coverage', 'largest-component fraction', 2),
    ('n_components_mean', 'Ncomponents', 'strong components per candidate cluster', 2),
    ('J_within_groups_over_J0', 'JWithin', 'mean $J/J_0$ within groups', 2),
    ('J_between_groups_over_J0', 'JBetween', 'mean $J/J_0$ between groups', 2),
    ('cycle_residual_abs_median', 'CycleMed', 'median cycle-basis residual (rad)', 4),
    ('cycle_residual_abs_q95', 'CycleQ', '95th percentile cycle-basis residual (rad)', 4),
    ('cycle_residual_abs_max', 'CycleMax', 'maximum cycle-basis residual (rad)', 4),
    ('cycle_constraints_below_tolerance', 'CyclePass', 'fraction of cycle constraints below tolerance', 3),
    ('candidate_units_certified_flat', 'FlatUnits', 'fraction of candidate units in certified clusters', 3),
    ('reversibility_defect_abs_mean_within', 'RevWithin', 'mean reciprocal two-cycle defect inside clusters (rad)', 4),
    ('omega_within_groups_sd_final', 'OmegaSd', 'final s.d.\\ of $\\omega$ within groups', 3),
    ('omega_within_clusters_sd_final', 'OmegaClusterSd', 'final s.d.\\ of $\\omega$ within clusters', 3),
    ('omega_cluster_means_sd_within_groups_final', 'OmegaClusterMeanSd', 's.d.\\ of cluster-mean $\\omega$ inside a group', 3),
    ('transporter_drift_rate_within', 'Drift', 'transporter drift inside clusters (rad per unit time)', 4),
]
TEST_ROWS = [
    ('delay_shock/R_raw_after', 'ShockRaw', 'offset redraw: $R^{\\mathrm{raw}}$ after', 2),
    ('delay_shock/R_cov_after', 'ShockCov', 'offset redraw: coverage-weighted $R^{\\mathrm{cov}}$ after', 2),
    ('delay_shock/R_cov_lcc_after', 'ShockCovLcc', 'offset redraw: transported coherence in largest component', 2),
    ('delay_shock/largest_component_fraction_after', 'ShockCoverage', 'offset redraw: largest-component fraction', 2),
    ('delay_shock/R_raw_min_during', 'ShockRawMin', 'offset redraw: minimum $R^{\\mathrm{raw}}$ during', 2),
    ('delay_shock/recovery_time_raw_0.9', 'ShockRec', 'offset redraw: time to 90\\% recovery', 0),
    ('delay_shock/ARI_after_vs_before', 'ShockARI', 'offset redraw: ARI of clusters after vs before', 2),
    ('delay_shock/R_cov_min_during', 'ShockCovMin', 'offset redraw: minimum $R^{\\mathrm{cov}}$ during', 2),
    ('delay_shock/recovery_time_cov_0.9', 'ShockRecCov', 'offset redraw: time to 90\\% covariant recovery', 0),
    ('delay_shock/purity_after', 'ShockPurityAfter', 'offset redraw: cluster purity after', 2),
    ('delay_shock/frac_clusters_collapsed', 'ShockCollapsed', 'offset redraw: fraction of clusters with $R^{\\mathrm{raw}}<0.5$ after', 2),
    ('delay_shock/frac_clusters_collapsed_cov', 'ShockCollapsedCov', 'offset redraw: fraction of clusters with $R^{\\mathrm{cov}}<0.5$ after', 2),
    ('delay_shock/exact_compensator_max_deviation', 'CompensatorDev', 'exact compensator: maximum connection deviation', 2),
    ('delay_shock/frozen_R_cov_after', 'FrozenShockCov', 'offset redraw with learned transporters frozen: $R^{\\mathrm{cov}}$', 2),
    ('delay_shock/frozen_ARI_after_vs_before', 'FrozenShockARI', 'offset redraw with learned transporters frozen: ARI', 2),
    ('removal/R_raw_after_survivors', 'RemovalRaw', 'removal: $R^{\\mathrm{raw}}$ of survivors', 2),
    ('removal/R_cov_after_survivors', 'RemovalCov', 'removal: $R^{\\mathrm{cov}}$ of survivors', 2),
    ('removal/frac_clusters_collapsed', 'RemovalCollapsed', 'removal: fraction of clusters with $R^{\\mathrm{raw}}<0.5$', 2),
    ('removal/frac_clusters_collapsed_cov', 'RemovalCollapsedCov', 'removal: fraction of clusters with $R^{\\mathrm{cov}}<0.5$', 2),
    ('removal/largest_component_fraction_after', 'RemovalCoverage', 'removal: largest-component fraction', 2),
    ('frequency_step/R_raw_min_during', 'FreqRawMin', 'frequency step: minimum $R^{\\mathrm{raw}}$ during', 2),
    ('frequency_step/R_raw_end_over_before', 'FreqRawEnd', 'frequency step: $R^{\\mathrm{raw}}$ end / before', 2),
    ('frequency_step/relock_time_raw_0.9', 'FreqRelock', 'frequency step: re-locking time', 0),
    ('frequency_step/omega_shifted_final_minus_group_mean', 'FreqResidual', 'frequency step: residual $\\omega$ offset', 2),
]

def abm_tables(tag, macros, suffix='', write_tables=True):
    f = os.path.join(HERE, 'results', f'abm_{tag}.json')
    if not os.path.exists(f):
        print('no', f, '- placeholder macros written')
        for key, suf, lab, nd in MAIN_ROWS + TEST_ROWS:
            for m, _ in MODELS: macros[f'ABM{suffix}{m}{suf}'] = '--'; macros[f'ABM{suffix}{m}{suf}Sd'] = '--'
        for mk in ['N', 'Seeds', 'Tone', 'Ttwo', 'Degree', 'Groups', 'DOmega', 'SigmaOmega', 'Mu', 'Jzero', 'EtaTheta', 'EtaJ', 'EtaOmega', 'Noise', 'Dt', 'MinSize', 'ThrFrac', 'FlatTolerance', 'RemoveFrac', 'FreqStep', 'DeltaMax', 'ThetaRule', 'Edges', 'Triangles', 'Controls']:
            macros[f'ABM{suffix}{mk}'] = '--'
        return
    d = json.load(open(f)); P = d['params']; agg = d['aggregate']; runs = d['runs']
    models = [m for m in MODELS if m[0] in agg]
    for key, suf, lab, nd in MAIN_ROWS + TEST_ROWS:      # arms absent from this run file get '--' so the manuscript always compiles
        for m, _ in MODELS:
            if m not in agg: macros[f'ABM{suffix}{m}{suf}'] = '--'; macros[f'ABM{suffix}{m}{suf}Sd'] = '--'
    macros[f'ABM{suffix}N'] = str(P['N']); macros[f'ABM{suffix}Seeds'] = str(len(P['seeds'])); macros[f'ABM{suffix}Tone'] = fmt(P['T1'], 0); macros[f'ABM{suffix}Ttwo'] = fmt(P['T2'], 0)
    for k, mk in [('k', 'Degree'), ('G', 'Groups'), ('dOmega', 'DOmega'), ('sigma_omega', 'SigmaOmega'), ('mu', 'Mu'), ('J0', 'Jzero'), ('eta_theta', 'EtaTheta'), ('eta_J', 'EtaJ'), ('eta_omega', 'EtaOmega'), ('D', 'Noise'), ('dt', 'Dt'), ('thr_frac', 'ThrFrac'), ('flat_tolerance', 'FlatTolerance'), ('min_size', 'MinSize'), ('remove_frac', 'RemoveFrac'), ('freq_step', 'FreqStep')]:
        v = P.get(k); macros[f'ABM{suffix}{mk}'] = (fmt(v, 3).rstrip('0').rstrip('.') if isinstance(v, float) else str(v))
    macros[f'ABM{suffix}DeltaMax'] = ('\\pi' if abs(P['delta_max'] - 3.141592653589793) < 1e-9 else fmt(P['delta_max'], 3))
    macros[f'ABM{suffix}ThetaRule'] = P.get('theta_rule', 'energy')
    macros[f'ABM{suffix}Edges'] = str(runs[0]['E']); macros[f'ABM{suffix}Triangles'] = str(runs[0]['n_triangles'])
    for key, suf, lab, nd in MAIN_ROWS + TEST_ROWS:
        for m, _ in models:
            v = agg[m].get(key); mean, sd, n = ms(v)
            if key == 'delay_shock/exact_compensator_max_deviation':
                macros[f'ABM{suffix}{m}{suf}'] = sci(mean); macros[f'ABM{suffix}{m}{suf}Sd'] = sci(sd)
            else:
                macros[f'ABM{suffix}{m}{suf}'] = fmt(mean, nd); macros[f'ABM{suffix}{m}{suf}Sd'] = fmt(sd, nd)
    def ratio(key, nd=1):
        try:
            a = ms(agg['gauge'][key])[0]; b = ms(agg['standard'][key])[0]
            return fmt(a / b, nd) if (a is not None and b) else '--'
        except Exception:
            return '--'
    macros[f'ABM{suffix}CompositeRatio'] = ratio('frac_composite')
    macros[f'ABM{suffix}ClustersPerGroupRatio'] = ratio('clusters_per_group_mean')
    macros[f'ABM{suffix}RelockRatio'] = ratio('frequency_step/relock_time_raw_0.9')
    if all(m in agg for m in ('gauge', 'standard')):
        paired_ari = [r['models']['gauge']['ARI_vs_groups'] - r['models']['standard']['ARI_vs_groups']
                      for r in runs if all(m in r['models'] for m in ('gauge', 'standard'))]
        macros[f'ABM{suffix}GaugeMinusStandardARI'] = fmt(st.mean(paired_ari), 3) if paired_ari else '--'
        macros[f'ABM{suffix}GaugeMinusStandardARISd'] = fmt(st.stdev(paired_ari), 3) if len(paired_ari) > 1 else '--'
    else:
        macros[f'ABM{suffix}GaugeMinusStandardARI'] = '--'
        macros[f'ABM{suffix}GaugeMinusStandardARISd'] = '--'
    ctrl = all(r['models']['gauge']['gauge_covariance_control']['PASS']
               and r['models']['gauge']['learning_rule_nonvacuous']['PASS']
               and r['models']['gauge']['tests']['delay_shock']['exact_compensator_max_deviation'] < 1e-9
               for r in runs) if 'gauge' in agg else False
    macros[f'ABM{suffix}Controls'] = 'PASS' if ctrl else 'FAIL'
    if not write_tables:
        print('ABM macros only (no table) for', tag, 'suffix', suffix)
        return
    for name, rows in [('abm_main_table', MAIN_ROWS), ('abm_tests_table', TEST_ROWS)]:
        with open(os.path.join(OUT, f'{name}{suffix}.tex'), 'w') as o:
            o.write(f'% generated by si_code/make_tables.py from results/abm_{tag}.json\n')
            o.write('\\begin{tabular}{@{}l' + 'c' * len(models) + '@{}}\n\\toprule\nQuantity & ' + ' & '.join(l for _, l in models) + ' \\\\\n\\midrule\n')
            for key, suf, lab, nd in rows:
                cells = [sci_pm(agg[m].get(key)) if key == 'delay_shock/exact_compensator_max_deviation'
                         else pm(agg[m].get(key), nd) for m, _ in models]
                if all(c == '--' for c in cells): continue      # quantity not measured by the run that produced this file
                o.write(lab + ' & ' + ' & '.join(cells) + ' \\\\\n')
            o.write('\\bottomrule\n\\end{tabular}\n')
        with open(os.path.join(OUT, f'{name}{suffix}.csv'), 'w', newline='') as c:
            w = csv.writer(c)
            header = ['quantity']
            for _, model_label in models:
                header.extend([model_label + ' mean', model_label + ' sd', model_label + ' n'])
            w.writerow(header)
            for key, suf, lab, nd in rows:
                values = [ms(agg[m].get(key)) for m, _ in models]
                if all(mean is None for mean, sd, n in values):
                    continue
                row = [plain_label(lab)]
                for mean, sd, n in values:
                    row.extend([mean, sd, n])
                w.writerow(row)
    print('ABM tables written for', tag, 'models', [m for m, _ in models], 'seeds', len(P['seeds']))


SCAN_TAGS = ['pilot_v2', 'scan_energy_v2', 'scan_etaomega_v2', 'scan_dt025_v2']

def purity_from_figure_data(run, model):
    import numpy as np
    fd = run.get('figure_data', {}); cl = np.array(fd['clusters'][model]); g = np.array(fd['groups'])
    labs = [c for c in np.unique(cl) if c >= 0]
    if not labs: return None
    pur = [np.bincount(g[cl == c]).max() / (cl == c).sum() for c in labs]; siz = [(cl == c).sum() for c in labs]
    return float(np.average(pur, weights=siz))

def scan_table(macros):
    """Corrected-model sensitivity checks: one row per setting and arm."""
    rows = []
    for tag in SCAN_TAGS:
        f = os.path.join(HERE, 'results', f'abm_{tag}.json')
        if not os.path.exists(f): continue
        d = json.load(open(f)); P = d['params']; r = d['runs'][0]
        for m, lab in MODELS:
            if m not in r['models']: continue
            rm = r['models'][m]
            rows.append((tag, P.get('theta_rule', 'energy'), P['dt'], P['eta_theta'], P['eta_omega'], lab,
                         rm['frac_composite'], rm['R_cov_mean'], rm['largest_component_fraction_mean'],
                         rm['cycle_residual_abs_q95'], rm['cycle_constraints_below_tolerance'],
                         rm['candidate_units_certified_flat'], rm['tests']['delay_shock']['ARI_after_vs_before'],
                         rm['tests']['delay_shock'].get('frozen_R_cov_after')))
    if not rows: return
    with open(os.path.join(OUT, 'abm_scan_table.tex'), 'w') as o:
        o.write('% generated by si_code/make_tables.py from corrected v2 sensitivity runs\n')
        o.write('\\begin{tabular}{@{}lllccclccccccc@{}}\n\\toprule\nrun & rule & $\\Delta t$ & $\\eta_\\theta$ & $\\eta_\\omega$ & arm & composite & $R^{\\mathrm{cov}}_{\\mathrm{covg}}$ & coverage & cycle q95 & constraints pass & certified units & ARI redraw & frozen $R^{\\mathrm{cov}}_{\\mathrm{covg}}$ \\\\\n\\midrule\n')
        for tag, rule, dt, et, ew, lab, fc, rc, cov, q95, cpass, flat, ari, frozen in rows:
            tex_tag = tag.replace('_', r'\_')
            o.write(f'{tex_tag} & {rule} & {dt:g} & {et:g} & {ew:g} & {lab} & {fc:.2f} & {rc:.2f} & {cov:.2f} & {fmt(q95,4)} & {fmt(cpass,3)} & {fmt(flat,3)} & {ari:.2f} & {fmt(frozen,2)} \\\\\n')
        o.write('\\bottomrule\n\\end{tabular}\n')
    macros['ABMScanRuns'] = str(len(set(r[0] for r in rows)))
    print('scan table written with', len(rows), 'rows')

CNN_CONFIGS = [('baseline', 'Baseline (no regulariser)'), ('semicomp', 'Semi-composite (fixed uniform mixing)'), ('adaptive', 'Adaptive semi-composite (learned mixing)'),
               ('adaptive_girl', 'Adaptive + cosine routing layer'), ('adaptive_phase', 'Adaptive + phase routing layer')]

def cnn_tables(macros):
    files = sorted(glob.glob(os.path.join(HERE, 'results', 'cnn_*.json')))
    for k, v in [('CNNepochs', '[pending]'), ('CNNeps', '[pending]'), ('CNNnadv', '[pending]')]: macros.setdefault(k, v)
    if not files:
        for name in ['cnn_table', 'cnn_gamma_table']:
            open(os.path.join(OUT, name + '.tex'), 'w').write('\\begin{tabular}{@{}l@{}}\\toprule pending \\\\ \\bottomrule\\end{tabular}\n')
        print('no CNN results yet'); return
    rows = {}
    for f in files:
        if os.path.basename(f).startswith('cnn_phase_lambda') or f.endswith('_quick.json'):
            continue
        m = re.match(r'cnn_(.+)_g([0-9.e-]+)_s(\d+)\.json', os.path.basename(f))
        if not m: continue
        cfg, gamma, seed = m.group(1), float(m.group(2)), int(m.group(3))
        rows.setdefault((cfg, gamma), []).append(json.load(open(f)))
    def agg(lst, key):
        vals = [r[key] for r in lst if r.get(key) is not None]
        if not vals: return None, None, 0
        return st.mean(vals), (st.stdev(vals) if len(vals) > 1 else None), len(vals)
    def cell(lst, key, scale=100, nd=1):
        m, s, n = agg(lst, key)
        if m is None: return '--'
        return f'{m*scale:.{nd}f}' if s is None else f'{m*scale:.{nd}f} $\\pm$ {s*scale:.{nd}f}'
    any_r = next(iter(rows.values()))[0]
    macros['CNNepochs'] = str(any_r.get('epochs', '')); macros['CNNeps'] = str(any_r.get('eps', '')); macros['CNNnadv'] = str(any_r.get('n_adv', ''))
    main_gamma = 0.01
    cnn_csv_rows = []
    with open(os.path.join(OUT, 'cnn_table.tex'), 'w') as o:
        o.write('% generated by si_code/make_tables.py from results/cnn_*.json\n\\begin{tabular}{@{}lccccccc@{}}\n\\toprule\nConfiguration & seeds & inference params & total trainable & clean (\\%) & FGSM (\\%) & PGD-10 (\\%) & compactness \\\\\n\\midrule\n')
        for cfg, lab in CNN_CONFIGS:
            lst = rows.get((cfg, main_gamma))
            if not lst:
                o.write(lab + ' & 0 & -- & -- & -- & -- & -- & -- \\\\\n')
                cnn_csv_rows.append([lab, 0, '', '', '', '', '', '', '', '', '', ''])
                continue
            n_in = 64 if cfg in ('adaptive_girl', 'adaptive_phase') else 2304
            extra = 0 if cfg == 'baseline' else 4 * n_in + (0 if cfg == 'semicomp' else 24 * 4)
            total_trainable = lst[0]['n_params'] + extra
            o.write(f"{lab} & {len(lst)} & {lst[0]['n_params']:,} & {total_trainable:,} & {cell(lst,'clean_test_acc')} & {cell(lst,'fgsm_test_acc')} & {cell(lst,'pgd10_test_acc')} & {cell(lst,'cluster_compactness',1,2)} \\\\\n")
            clean = agg(lst, 'clean_test_acc'); fgsm = agg(lst, 'fgsm_test_acc')
            pgd = agg(lst, 'pgd10_test_acc'); compact = agg(lst, 'cluster_compactness')
            cnn_csv_rows.append([lab, len(lst), lst[0]['n_params'], total_trainable,
                                 clean[0] * 100, None if clean[1] is None else clean[1] * 100,
                                 fgsm[0] * 100, None if fgsm[1] is None else fgsm[1] * 100,
                                 pgd[0] * 100, None if pgd[1] is None else pgd[1] * 100,
                                 compact[0], compact[1]])
            mac = re.sub(r'[^A-Za-z]', '', cfg)
            for key, suf in [('clean_test_acc', 'Clean'), ('fgsm_test_acc', 'FGSM'), ('pgd10_test_acc', 'PGD')]:
                m, s, n = agg(lst, key); macros[f'CNN{mac}{suf}'] = fmt(m*100 if m is not None else None, 1); macros[f'CNN{mac}{suf}Sd'] = fmt(s*100 if s is not None else None, 1)   # 0.0 is a value, not a missing entry
            m, s, n = agg(lst, 'cluster_compactness'); macros[f'CNN{mac}Compact'] = fmt(m, 2)
            macros[f'CNN{mac}Seeds'] = str(n)
            macros[f'CNN{mac}InferenceParams'] = f"{lst[0]['n_params']:,}"
            macros[f'CNN{mac}TrainableParams'] = f"{total_trainable:,}"
        o.write('\\bottomrule\n\\end{tabular}\n')
    with open(os.path.join(OUT, 'cnn_table.csv'), 'w', newline='') as c:
        w = csv.writer(c)
        w.writerow(['configuration', 'seeds', 'inference parameters', 'total trainable parameters',
                    'clean accuracy mean (%)', 'clean accuracy sd (%)',
                    'FGSM accuracy mean (%)', 'FGSM accuracy sd (%)',
                    'PGD-10 accuracy mean (%)', 'PGD-10 accuracy sd (%)',
                    'compactness mean', 'compactness sd'])
        w.writerows(cnn_csv_rows)
    # derived differences quoted in the text, so that no arithmetic is done by hand in the manuscript
    def diff(a, b, nd=1):
        try: return fmt(float(macros[a]) - float(macros[b]), nd)
        except Exception: return '--'
    macros['CNNphaseOverBaseline'] = diff('CNNadaptivephaseClean', 'CNNbaselineClean')
    macros['CNNphaseOverAdaptive'] = diff('CNNadaptivephaseClean', 'CNNadaptiveClean')
    macros['CNNphaseOverCosine'] = diff('CNNadaptivephaseClean', 'CNNadaptivegirlClean')
    macros['CNNadaptiveBelowSemicomp'] = diff('CNNsemicompClean', 'CNNadaptiveClean')
    macros['CNNadaptiveBelowBaseline'] = diff('CNNbaselineClean', 'CNNadaptiveClean')
    for cfg, key in [('baseline', 'CNNbaseParams'), ('adaptive_phase', 'CNNroutedParams')]:
        lst = rows.get((cfg, main_gamma)); macros[key] = f"{lst[0]['n_params']:,}" if lst else '--'
    gamma_csv_rows = []
    with open(os.path.join(OUT, 'cnn_gamma_table.tex'), 'w') as o:
        o.write('% generated by si_code/make_tables.py from results/cnn_adaptive_g*_s0.json\n\\begin{tabular}{@{}lcccc@{}}\n\\toprule\n$\\gamma$ & clean (\\%) & FGSM (\\%) & PGD-10 (\\%) & compactness \\\\\n\\midrule\n')
        for (cfg, gamma), lst in sorted(rows.items()):
            if cfg != 'adaptive': continue
            l0 = [r for r in lst if r.get('seed', 0) == 0] or lst[:1]
            o.write(f"{gamma:g} & {cell(l0,'clean_test_acc')} & {cell(l0,'fgsm_test_acc')} & {cell(l0,'pgd10_test_acc')} & {cell(l0,'cluster_compactness',1,2)} \\\\\n")
            gamma_csv_rows.append([gamma,
                                   agg(l0, 'clean_test_acc')[0] * 100,
                                   agg(l0, 'fgsm_test_acc')[0] * 100,
                                   agg(l0, 'pgd10_test_acc')[0] * 100,
                                   agg(l0, 'cluster_compactness')[0]])
            nm = {0.001: 'Lo', 0.01: 'Mid', 0.1: 'Hi'}.get(gamma)
            if nm:
                macros[f'CNNgamma{nm}'] = f'{gamma:g}'
                macros[f'CNNgamma{nm}Clean'] = fmt(agg(l0, 'clean_test_acc')[0] * 100, 1)
                macros[f'CNNgamma{nm}FGSM'] = fmt(agg(l0, 'fgsm_test_acc')[0] * 100, 1)
                macros[f'CNNgamma{nm}Compact'] = fmt(agg(l0, 'cluster_compactness')[0], 2)
        o.write('\\bottomrule\n\\end{tabular}\n')
    with open(os.path.join(OUT, 'cnn_gamma_table.csv'), 'w', newline='') as c:
        w = csv.writer(c)
        w.writerow(['gamma', 'clean accuracy (%)', 'FGSM accuracy (%)', 'PGD-10 accuracy (%)', 'compactness'])
        w.writerows(gamma_csv_rows)
    print('CNN tables written from', len(files), 'files; configs:', sorted(set(k[0] for k in rows)))

def analytical_macros(macros):
    f = os.path.join(HERE, 'results', 'reduction_dynamics.json')
    if os.path.exists(f):
        d = json.load(open(f)); curve = {round(x['Lambda'], 3): x for x in d['transient_curve']}
        for lam, name in [(0.10, 'Ten'), (0.14, 'Fourteen')]:
            x = curve[round(lam, 3)]
            macros[f'ReductionStatic{name}'] = fmt(x['static_local_tau_over_tau'], 2)
            macros[f'ReductionLowFrequency{name}'] = fmt(x['first_order_low_frequency_tau_over_tau'], 2)
            macros[f'ReductionExact{name}'] = fmt(x['exact_slow_tau_over_tau'], 2)
    f = os.path.join(HERE, 'results', 'diagnostic_controls.json')
    if os.path.exists(f):
        d = json.load(open(f)); macros['DiagnosticControls'] = 'PASS' if d.get('OVERALL_PASS') else 'FAIL'

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--abm_tag', default='main_v2'); ap.add_argument('--abm_mild_tag', default='mild_v2'); ap.add_argument('--abm_diag_tag', default=''); ap.add_argument('--code_url', default='https://github.com/apatrascu-cmd/composite-neuron-gauge-connectivity')
    a = ap.parse_args()
    macros = {'CodeURL': a.code_url}
    abm_tables(a.abm_tag, macros)
    abm_tables(a.abm_mild_tag, macros, suffix='mild')
    if a.abm_diag_tag: abm_tables(a.abm_diag_tag, macros, suffix='diag', write_tables=False)
    scan_table(macros); cnn_tables(macros); analytical_macros(macros)
    with open(os.path.join(OUT, 'numbers.tex'), 'w') as o:
        o.write('% generated by si_code/make_tables.py -- do not edit; every number in the manuscript comes from here\n')
        for k, v in sorted(macros.items()):
            assert re.fullmatch(r'[A-Za-z]+', k), k
            o.write(f'\\newcommand{{\\{k}}}{{{v}}}\n')
    print('numbers.tex:', len(macros), 'macros')
