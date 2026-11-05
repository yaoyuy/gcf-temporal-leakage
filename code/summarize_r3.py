"""Round-3 summary. Reads out/r3/gcf-r3-*/ (res_*.json, peruser_*.npz, lr_choice_new_*.json).
Writes out/r3/summary.csv and out/r3/stats.md.
- metrics: NDCG@20, Recall@20, NDCG@10, Recall@10 in two populations: warm (= r2 definition) and full (all test users,
  cold items/users count as misses / 0).
- ItemKNN k / EASE lambda chosen on seed 1 by validation NDCG@20 (base_choice_<ds>.json); MostPop/ItemKNN/EASE on loo/global_temporal were run for seed 1 only (deterministic split + deterministic model, r2 std = 0);
  they are replicated to seeds 2-5 and flagged n_seeds_run=1.
- Kendall tau (scipy, tau-b) between protocol rankings of model means, per dataset, over models present in both protocols.
- Paired Wilcoxon signed-rank (two-sided, zero_method='wilcox') on per-user NDCG@20: best model (by mean) vs every other model,
  per dataset x protocol x population, Holm correction within each dataset x protocol x population family.
  Pairing unit: loo/global_temporal -> user (per-user value averaged over the seeds both models have; test users identical across
  seeds); random -> (seed, user) pairs over the common seeds (test users differ by seed).
"""
import os
ROOT = os.environ.get('GCF_ROOT', '.')  # project root containing exp/out/r2..r5 and paper/
import json, glob, os, re, csv, numpy as np
from scipy.stats import kendalltau, wilcoxon
R = os.environ.get('R3_DIR', os.path.join(ROOT, 'exp/out/r3'))
MODELS = ['MostPop', 'ItemKNN', 'EASE', 'BPR-MF', 'LightGCN', 'SimGCL', 'UltraGCN']
PROTOS = ['random', 'loo', 'global_temporal']
LR_R2 = {}  # chosen lr for BPR-MF/LightGCN (round 2) and SimGCL/UltraGCN (round 3 grid)
exec(re.search(r"LR_R2 = (\{.*?\n\})", open(os.path.join(ROOT, 'exp/driver_v4.py')).read(), re.S).group(0))
lrnew, basech = {}, {}
for p in glob.glob(R + '/gcf-r3-new-*/**/*lr_choice_new_*.json', recursive=True):
    ds = re.search(r'lr_choice_new_(.*)\.json', p).group(1); lrnew[ds] = {k: v[1] for k, v in json.load(open(p)).items()}
for p in glob.glob(R + '/gcf-r3-basetune-*/**/*base_choice_*.json', recursive=True):
    ds = re.search(r'base_choice_(.*)\.json', p).group(1); basech[ds] = {k: v[1] for k, v in json.load(open(p)).items()}
def chosen(ds, pr, m):
    if m in ('BPR-MF', 'LightGCN'): return LR_R2[ds][f'{pr}/{m}']
    if m in ('ItemKNN', 'EASE'): return basech.get(ds, {}).get(f'{pr}/{m}')
    if m == 'MostPop': return None
    return lrnew.get(ds, {}).get(f'{pr}/{m}')
res, npz, info, nrun = {}, {}, {}, {}
files = []
for pat in ('basetune', 'rerun', 'new'): files += glob.glob(R + f'/gcf-r3-{pat}-*/**/*res_*.json', recursive=True)
for p in sorted(files):
    k = re.sub(r'^out_', '', os.path.basename(p))[:-5]
    m = re.match(r'res_(.+?)_(random|loo|global_temporal)_s(\d)_(MostPop|ItemKNN|EASE|BPR-MF|LightGCN|SimGCL|UltraGCN)(?:_(?:lr|k|lam)(.+))?$', k)
    if not m: continue
    ds, pr, s, mo, par = m.groups(); s = int(s)
    d = json.load(open(p)).get(pr, {})
    if d.get('info') and s == 1: info[(ds, pr)] = d['info']
    if mo not in d or par != chosen(ds, pr, mo): continue
    f = glob.glob(os.path.join(os.path.dirname(p), f'*peruser_{k[4:]}_{pr}_{mo}.npz'))
    reps = [1, 2, 3, 4, 5] if (mo in ('MostPop', 'ItemKNN', 'EASE') and pr != 'random') else [s]
    for ss in reps:
        res[(ds, pr, mo, ss)] = d[mo]
        if f: npz[(ds, pr, mo, ss)] = f[0]
    nrun[(ds, pr, mo)] = nrun.get((ds, pr, mo), 0) + 1
METS = ['NDCG@20', 'Recall@20', 'NDCG@10', 'Recall@10']
rows, mean = [], {}
for ds in sorted({k[0] for k in res}):
    for pr in PROTOS:
        for mo in MODELS:
            seeds = sorted(s for (a, b, c, s) in res if (a, b, c) == (ds, pr, mo))
            if not seeds: continue
            row = {'dataset': ds, 'protocol': pr, 'model': mo, 'n_seeds': len(seeds), 'n_seeds_run': nrun[(ds, pr, mo)],
                   'hparam': chosen(ds, pr, mo) or '', 'seeds': ' '.join(map(str, seeds))}
            for pop, pre in (('warm', ''), ('full', 'full_')):
                for me in METS:
                    v = np.array([res[(ds, pr, mo, s)][pre + me] for s in seeds])
                    row[f'{pop}_{me}_mean'] = v.mean(); row[f'{pop}_{me}_std'] = v.std(ddof=1) if len(v) > 1 else 0.
                    mean[(ds, pr, mo, pop, me)] = v.mean()
            inf = info.get((ds, pr)) or {}
            for c in ('test_users_all', 'test_users_wu', 'test_users_warm', 'test_rows_all', 'test_rows_warm', 'test_cold_rows_pct'):
                row[c] = inf.get(c, '')
            rows.append(row)
os.makedirs(R, exist_ok=True)
if rows:
    with open(R + '/summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
def holm(ps):
    o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for r, i in enumerate(o): run = max(run, min(1, (m - r) * ps[i])); adj[i] = run
    return adj
def peruser(ds, pr, mo, seeds, pop):
    key = 'warm_NDCG@20' if pop == 'warm' else 'wu_NDCG@20'
    out = {}
    for s in seeds:
        z = np.load(npz[(ds, pr, mo, s)]); v = z[key]; u = z['users']
        keep = ~np.isnan(v); out[s] = dict(zip(u[keep].tolist(), v[keep].tolist()))
    return out
L = ['# Round 3 statistics', '', 'Generated by exp/summarize_r3.py from out/r3/gcf-r3-*/ (raw logs alongside).', '',
     'Populations: **warm** = test interactions with user and item seen in train (identical to the round-2 metric); '
     '**full** = all test interactions, cold items/users count as misses (0).', '']
L += ['## Test population sizes', '', '| dataset | protocol | test users (all) | warm users (in train) | users with >=1 warm item | test rows (all) | warm rows | cold rows % |', '|---|---|---|---|---|---|---|---|']
for (ds, pr), inf in sorted(info.items()):
    if inf: L.append(f"| {ds} | {pr} | {inf.get('test_users_all')} | {inf.get('test_users_wu')} | {inf.get('test_users_warm')} | {inf.get('test_rows_all')} | {inf.get('test_rows_warm')} | {inf.get('test_cold_rows_pct')} |")
for pop in ('warm', 'full'):
    L += ['', f'## Mean NDCG@20 ({pop}), mean ± std over seeds (n)', '', '| dataset | protocol | ' + ' | '.join(MODELS) + ' |', '|' + '---|' * (len(MODELS) + 2)]
    for r in rows:
        pass
    for ds in sorted({r['dataset'] for r in rows}):
        for pr in PROTOS:
            cells = []
            for mo in MODELS:
                rr = [r for r in rows if (r['dataset'], r['protocol'], r['model']) == (ds, pr, mo)]
                cells.append(f"{rr[0][f'{pop}_NDCG@20_mean']:.4f}±{rr[0][f'{pop}_NDCG@20_std']:.4f} ({rr[0]['n_seeds']})" if rr else 'missing')
            L.append(f'| {ds} | {pr} | ' + ' | '.join(cells) + ' |')
L += ['', '## Chosen hyper-parameters (seed 1, validation NDCG@20)', '', '| dataset | protocol | ItemKNN k | EASE λ | BPR-MF lr (r2) | LightGCN lr (r2) | SimGCL lr | UltraGCN lr |', '|---|---|---|---|---|---|---|---|']
for ds in sorted({r['dataset'] for r in rows}):
    for pr in PROTOS:
        L.append(f'| {ds} | {pr} | ' + ' | '.join(str(chosen(ds, pr, m) or '-') for m in ['ItemKNN', 'EASE', 'BPR-MF', 'LightGCN', 'SimGCL', 'UltraGCN']) + ' |')
r2info = {}
for p in glob.glob(os.path.join(ROOT, 'exp/out/r2/gcf-r2-*/**/*res_*_s1_*.json'), recursive=True):
    mm = re.search(r'res_(.+?)_(random|loo|global_temporal)_s1_', os.path.basename(p))
    if mm:
        dd = json.load(open(p)).get(mm.group(2), {}).get('info')
        if dd: r2info[(mm.group(1), mm.group(2))] = dd
L += ['', '## Split statistics, round 3 vs round 2 (seed 1; Gowalla re-prepared with second-resolution timestamps in round 3)', '',
      '| dataset | protocol | stat | r2 | r3 |', '|---|---|---|---|---|']
for (ds, pr), inf in sorted(info.items()):
    o = r2info.get((ds, pr), {})
    for c in ('train', 'val', 'test', 'users', 'items', 'test_users', 'test_cold_rows_pct', 'val_cold_rows_pct', 'train_after_user_first_test_pct', 'train_after_global_cut_pct'):
        a, b = o.get(c, 'n/a'), inf.get(c, 'n/a')
        if ds == 'gowalla' or a != b: L.append(f'| {ds} | {pr} | {c} | {a} | {b} |{" changed" if a != b else ""}')
L += ['', '## Kendall τ between protocol rankings (NDCG@20 means)', '', '| dataset | population | pair | τ | p | n models |', '|---|---|---|---|---|---|']
for ds in sorted({r['dataset'] for r in rows}):
    for pop in ('warm', 'full'):
        for a, b in (('random', 'global_temporal'), ('loo', 'global_temporal'), ('random', 'loo')):
            ms = [m for m in MODELS if (ds, a, m, pop, 'NDCG@20') in mean and (ds, b, m, pop, 'NDCG@20') in mean]
            if len(ms) < 3: continue
            t, p = kendalltau([mean[(ds, a, m, pop, 'NDCG@20')] for m in ms], [mean[(ds, b, m, pop, 'NDCG@20')] for m in ms])
            L.append(f'| {ds} | {pop} | {a} vs {b} | {t:.3f} | {p:.3g} | {len(ms)} |')
L += ['', '## Paired Wilcoxon (NDCG@20 per user), best model vs others, Holm-adjusted within each dataset × protocol × population', '',
      '| dataset | protocol | population | best | vs | mean diff | n pairs | p (raw) | p (Holm) | sig@0.05 |', '|---|---|---|---|---|---|---|---|---|---|']
for ds in sorted({r['dataset'] for r in rows}):
    for pr in PROTOS:
        for pop in ('warm', 'full'):
            ms = [m for m in MODELS if (ds, pr, m, pop, 'NDCG@20') in mean]
            if len(ms) < 2: continue
            best = max(ms, key=lambda m: mean[(ds, pr, m, pop, 'NDCG@20')]); tests = []
            for o in ms:
                if o == best: continue
                sb = {s for (a, b, c, s) in npz if (a, b, c) == (ds, pr, best)}; so = {s for (a, b, c, s) in npz if (a, b, c) == (ds, pr, o)}
                cs = sorted(sb & so)
                if not cs: continue
                A, B = peruser(ds, pr, best, cs, pop), peruser(ds, pr, o, cs, pop)
                if pr == 'random':
                    x = [A[s][u] for s in cs for u in A[s] if u in B[s]]; y = [B[s][u] for s in cs for u in A[s] if u in B[s]]
                else:
                    us = sorted(set.intersection(*[set(A[s]) for s in cs], *[set(B[s]) for s in cs]))
                    x = [np.mean([A[s][u] for s in cs]) for u in us]; y = [np.mean([B[s][u] for s in cs]) for u in us]
                x, y = np.array(x), np.array(y)
                try: p = wilcoxon(x, y, zero_method='wilcox', alternative='two-sided').pvalue
                except ValueError: p = 1.0
                tests.append([o, float((x - y).mean()), len(x), p])
            if not tests: continue
            adj = holm(np.array([t[3] for t in tests]))
            for t, pa in zip(tests, adj):
                L.append(f'| {ds} | {pr} | {pop} | {best} | {t[0]} | {t[1]:+.4f} | {t[2]} | {t[3]:.3g} | {pa:.3g} | {"yes" if pa < .05 else "no"} |')
miss = [(ds, pr, mo) for ds in sorted({r['dataset'] for r in rows}) for pr in PROTOS for mo in MODELS
        if not any((r['dataset'], r['protocol'], r['model']) == (ds, pr, mo) for r in rows)]
L += ['', '## Missing / partial', '']
L += [f'- missing: {m}' for m in miss] or ['- none missing']
L += [f"- {r['dataset']} {r['protocol']} {r['model']}: {r['n_seeds']} seeds ({r['seeds']})" for r in rows if r['n_seeds'] < 5]
open(R + '/stats.md', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L[:60])); print('rows', len(rows))
