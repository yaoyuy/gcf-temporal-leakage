"""Generate LaTeX tables from real round-3 results. No hand-typed numbers.
Inputs: exp/out/r3/summary.csv (means/SD over seeds, chosen hyper-parameters, n_seeds),
        analysis/info_r3.json, analysis/wilcoxon_r3.json (both written by collect_r3.py from exp/out/r3).
The round-2 version of this script is kept as make_tables_r2.py."""
import os
ROOT = os.environ.get('GCF_ROOT', '.')  # project root containing exp/out/r2..r5 and paper/
import json, pandas as pd, numpy as np
from scipy.stats import kendalltau
R = ROOT
S = pd.read_csv(f'{R}/exp/out/r3/summary.csv', dtype={'hparam': str})
INFO = json.load(open(f'{R}/paper/analysis/info_r3.json'))
WX = json.load(open(f'{R}/paper/analysis/wilcoxon_r3.json'))
DS = ['ml-1m', 'amz-games', 'amz-music', 'gowalla']
DSN = {'ml-1m': 'MovieLens-1M', 'amz-games': 'Amazon Video Games', 'amz-music': 'Amazon Musical Instruments', 'gowalla': 'Gowalla (15-core)'}
DSS = {'ml-1m': 'ML-1M', 'amz-games': 'Video Games', 'amz-music': 'Musical Instr.', 'gowalla': 'Gowalla'}
PR = ['random', 'loo', 'global_temporal']; PRN = {'random': 'Random', 'loo': 'LOO', 'global_temporal': 'GT'}
M = ['MostPop', 'ItemKNN', 'EASE', 'BPR-MF', 'LightGCN', 'SimGCL']
CAP_MIN = 590  # seconds; a run that reached the 10-min cap
# runs that stopped at the 10-min cap in round 3 (checked from res_*.json 'sec' >= 590 s; all five seeds in both cells)
CAPPED = {('amz-games', 'random', 'BPR-MF'), ('amz-games', 'loo', 'BPR-MF')}
def row(ds, p, m): return S[(S.dataset == ds) & (S.protocol == p) & (S.model == m)].iloc[0]
def v(ds, p, m, c): return float(row(ds, p, m)[c])
def ns(ds, p, m, pop):
    """True if m is not the best and its difference from the best is not significant (Holm p >= .05)."""
    for w in WX:
        if (w['dataset'], w['protocol'], w['pop'], w['vs']) == (ds, p, pop, m): return not w['sig']
    return False
def nseeds(ds, p, m): return int(row(ds, p, m)['n_seeds'])
WORD = {3: 'three', 4: 'four'}
def seed_note(protos=PR):
    by = {}
    for ds in DS:
        for p in protos:
            for m in M:
                n = nseeds(ds, p, m)
                if n < 5: by.setdefault(n, []).append(f"{DSS[ds]} {PRN[p]} {m}")
    return '; '.join(f"{WORD[n]} seeds: " + ', '.join(c) for n, c in sorted(by.items(), reverse=True))

def metric_table(col, pop, name, label, caption, sigmarks, protos=PR, datasets=DS, dec=4):
    L = [r'\begin{table}[t]', r'\centering', r'\caption{' + caption + '}', r'\label{' + label + '}', r'\scriptsize', r'\setlength{\tabcolsep}{1.0pt}',
         r'\begin{tabular}{@{}l' + 'c' * len(M) + '@{}}', r'\toprule', 'Protocol & ' + ' & '.join(M) + r' \\', r'\midrule']
    used = set()
    for ds in datasets:
        for j, p in enumerate(protos):
            means = {m: v(ds, p, m, f'{pop}_{col}_mean') for m in M}
            order = sorted(M, key=lambda m: -means[m]); best, second = order[0], order[1]
            cells = []
            for m in M:
                s = f'{means[m]:.{dec}f}'; sd = v(ds, p, m, f'{pop}_{col}_std')
                s = f'{s}$_{{\\pm ' + f'{sd:.{dec}f}'.lstrip('0') + '}$' if sd > 0 else s
                if m == best: s = r'\textbf{' + s + '}'
                elif m == second: s = r'\underline{' + s + '}'
                sup = []
                if (ds, p, m) in CAPPED: sup.append('a'); used.add('a')
                if sigmarks and m != best and ns(ds, p, m, pop): sup.append('b'); used.add('b')
                if nseeds(ds, p, m) < 5: sup.append('c'); used.add('c')
                if sup: s += r'\textsuperscript{' + ','.join(sup) + '}'
                cells.append(s)
            if j == 0: L.append(r'\multicolumn{' + str(len(M) + 1) + r'}{@{}l}{\textit{' + DSN[ds] + r'}} \\')
            L.append(PRN[p] + ' & ' + ' & '.join(cells) + r' \\')
        if ds != datasets[-1]: L.append(r'\addlinespace')
    L += [r'\bottomrule', r'\end{tabular}']
    if 'a' in used: L.append(r'\footnotetext[a]{All five runs stopped at the 10-min training cap while validation NDCG@20 was still rising slowly (Sect.~\ref{sec:threats}); these values are lower bounds}')
    if 'b' in used: L.append(r'\footnotetext[b]{Difference from the best model of the row not significant: paired two-sided Wilcoxon signed-rank test on per-user NDCG@20, Holm-adjusted $p\ge0.05$ (Sect.~\ref{sec:metrics})}')
    if 'c' in used: L.append(r'\footnotetext[c]{Fewer than five seeds: the remaining runs were not started because the preset time budget of the session, set to stay within the free GPU quota, had been used up (' + seed_note(protos) + ')}')
    L += [r'\end{table}']
    open(name, 'w').write('\n'.join(L) + '\n')

metric_table('NDCG@20', 'warm', 'tab_ndcg.tex', 'tab:ndcg',
             r'NDCG@20 on warm test interactions (user and item seen in training): mean over seeds (five unless marked), with the SD across seeds as subscript. Best per row in bold, second best underlined. MostPop, ItemKNN and EASE are deterministic under LOO and GT and were run once, so no SD is shown', True)
metric_table('Recall@20', 'warm', 'tab_recall.tex', 'tab:recall',
             r'Recall@20 on warm test interactions (mean over seeds, SD as subscript). Same conventions as Table~\ref{tab:ndcg}; significance was tested for NDCG@20 only', False)
metric_table('NDCG@20', 'full', 'tab_full.tex', 'tab:full',
             r'NDCG@20 under the global temporal split on the full test population: all raw test users and interactions, with cold users scoring 0 and cold items counted as misses. Five decimals are shown because the values are small; otherwise the conventions of Table~\ref{tab:ndcg} apply, and significance marks refer to Wilcoxon tests on this population. Under the random and LOO splits full and warm values differ by at most 0.0002 (not shown)', True, protos=['global_temporal'], dec=5)

# chosen hyper-parameters (seed 1, validation NDCG@20)
def fmt(h):
    if h in ('', 'nan') or h is None: return '--'
    if 'e-' in h:
        a, b = h.split('e-'); return (f'${a}\\cdot10^{{-{int(b)}}}$' if a != '1' else f'$10^{{-{int(b)}}}$')
    return f'{int(h):,}'.replace(',', '{,}')
L = [r'\begin{table}[t]', r'\centering',
     r'\caption{Hyperparameters selected on seed~1 by validation NDCG@20 (grids in Table~\ref{tab:hparams}). Values at an end of their grid are marked with an asterisk}',
     r'\label{tab:lr}', r'\footnotesize', r'\begin{tabular}{@{}llccccc@{}}', r'\toprule',
     r'Dataset & Protocol & ItemKNN $k$ & EASE $\lambda$ & BPR-MF lr & LightGCN lr & SimGCL lr \\', r'\midrule']
EDGE = {'ItemKNN': {'50', '500'}, 'EASE_ml-1m': {'100', '5000'}, 'EASE': {'10', '1000'}}
edges = []
for ds in DS:
    for j, p in enumerate(PR):
        c = []
        for m in ['ItemKNN', 'EASE', 'BPR-MF', 'LightGCN', 'SimGCL']:
            h = str(row(ds, p, m)['hparam']); s = fmt(h)
            if m in ('ItemKNN', 'EASE'):
                e = EDGE['EASE_ml-1m' if (m == 'EASE' and ds == 'ml-1m') else m]
                if h in e: s += '*'; edges.append((ds, p, m, h))
            c.append(s)
        L.append((DSS[ds] if j == 0 else '') + ' & ' + PRN[p] + ' & ' + ' & '.join(c) + r' \\')
    if ds != DS[-1]: L.append(r'\addlinespace')
L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
open('tab_lr.tex', 'w').write('\n'.join(L) + '\n')

# split statistics (seed-1 sizes; Leak/Future/Cold means over the five seeds)
L = [r'\begin{table}[t]', r'\centering', r"\caption{Split statistics (seed~1 sizes; validation and test counts after removing interactions whose user or item is absent from the training set). Leak: share of training interactions that occur after the same user's first test interaction. Future: share of training interactions later than the global 80\% time quantile. Cold: share of raw test interactions whose user or item is unseen in training. Leak, Future and Cold are means over the five seeds (they vary only under the random protocol)}", r'\label{tab:splits}', r'\scriptsize',
     r'\begin{tabular}{@{}lrrrrrrr@{}}', r'\toprule', r'Protocol & Train & Val. & Test & Test users & Leak (\%) & Future (\%) & Cold (\%) \\', r'\midrule']
for ds in DS:
    for j, p in enumerate(PR):
        x = INFO[f'{ds}|{p}']; s1 = x['1']
        lk = np.mean([x[s]['train_after_user_first_test_pct'] for s in x]); cd = np.mean([x[s]['test_cold_rows_pct'] for s in x]); fu = np.mean([x[s]['train_after_global_cut_pct'] for s in x])
        if j == 0: L.append(r'\multicolumn{8}{@{}l}{\textit{' + DSN[ds] + r'}} \\')
        L.append(f"{PRN[p]} & {s1['train']:,} & {s1['val']:,} & {s1['test']:,} & {s1['test_users']:,} & ".replace(',', '{,}') + f"{lk:.2f} & {fu:.2f} & {cd:.2f}" + r' \\')
    if ds != DS[-1]: L.append(r'\addlinespace')
L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
open('tab_splits.tex', 'w').write('\n'.join(L) + '\n')

# numbers quoted in the text
pairs = [('random', 'global_temporal'), ('loo', 'global_temporal'), ('random', 'loo')]
txt = {'grid_edges': edges}
for ds in DS:
    for pop in ('warm', 'full'):
        for c in ['NDCG@20', 'Recall@20']:
            for a, b in pairs:
                txt[f'{ds}|{pop}|{c}|tau|{a}-{b}'] = round(kendalltau([v(ds, a, m, f'{pop}_{c}_mean') for m in M], [v(ds, b, m, f'{pop}_{c}_mean') for m in M])[0], 3)
        for p in PR:
            for c in ['NDCG@20', 'Recall@20']:
                o = sorted(M, key=lambda m: -v(ds, p, m, f'{pop}_{c}_mean'))
                txt[f'{ds}|{pop}|{c}|order|{p}'] = ' > '.join(f"{m} {v(ds, p, m, f'{pop}_{c}_mean'):.4f}" for m in o)
            txt[f'{ds}|{pop}|gapLG-EASE%|{p}'] = round(100 * (v(ds, p, 'LightGCN', f'{pop}_NDCG@20_mean') / v(ds, p, 'EASE', f'{pop}_NDCG@20_mean') - 1), 1)
    for p in PR:
        x = INFO[f'{ds}|{p}']
        for k in ('train_after_user_first_test_pct', 'train_after_global_cut_pct', 'test_cold_rows_pct', 'val_cold_rows_pct'):
            vals = [x[s][k] for s in x]; txt[f'{ds}|{p}|{k}'] = [round(float(np.mean(vals)), 2), min(vals), max(vals)]
json.dump(txt, open('derived_numbers.json', 'w'), indent=0)
for k, x in txt.items(): print(k, x)
