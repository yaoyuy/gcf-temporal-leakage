"""Why does E1 condition a (EASE >= LightGCN on ml-1m/gowalla) differ from the full GT table (LightGCN >= EASE)?
Same training data (a = GT train), so the differences are (i) users: only evaluation users E (~50%), (ii) items: held-out half H
of their warm test items, with own-future F masked, (iii) LightGCN seeds (new R4 runs). This script evaluates the FULL-GT per-user
NDCG@20 (round-3 LightGCN over its seeds; R4 extended-grid EASE) restricted to the E users, and E1-a on the same users.
Writes out/r4/e1a_decompose.md and .json"""
import os
ROOT = os.environ.get('GCF_ROOT', '.')  # project root containing exp/out/r2..r5 and paper/
import glob, re, json, numpy as np, pandas as pd
from scipy.stats import wilcoxon
E = os.path.join(ROOT, 'exp'); R3 = f'{E}/out/r3'; R4 = f'{E}/out/r4'
LR = {'ml-1m': '5e-3', 'gowalla': '1e-3', 'amz-music': '1e-3', 'amz-games': '1e-3'}
def load(name):
    if name == 'ml-1m': return pd.read_csv(f'{E}/data/ml-1m/ratings.dat', sep='::', engine='python', names=['u','i','r','t']).drop(columns='r')
    return pd.read_csv(f'{E}/data/{name}.csv', usecols=['u','i','t'])
def avg(files, key='users_raw', raw=None):
    acc = {}
    for f in files:
        z = np.load(f); us = z[key] if raw is None else raw[z['users']]
        for u, v in zip(us, z['warm_NDCG@20']):
            if not np.isnan(v): acc.setdefault(str(u), []).append(float(v))
    return {u: np.mean(v) for u, v in acc.items()}
J = {}
L = ['# E1 condition a vs full GT: decomposition (exp/e1a_decompose.py)', '', '| dataset | population | n users | LightGCN | EASE | LightGCN−EASE | Wilcoxon p (raw) |', '|---|---|---|---|---|---|---|']
for ds in ['ml-1m', 'gowalla', 'amz-music', 'amz-games']:
    d = load(ds).sort_values('t', kind='stable'); tr = d.iloc[:int(.8*len(d))]; raw = np.array([str(u) for u in sorted(set(tr.u))])
    lg_full = avg(sorted(glob.glob(f'{R3}/gcf-r3-*/out_peruser_{ds}_global_temporal_s*_LightGCN_lr{LR[ds]}_global_temporal_LightGCN.npz')), raw=raw)
    ea_full = avg(glob.glob(f'{R4}/gcf-r4-cpu-gt-{ds}/out_peruser_gtcpu_{ds}_s1_gt_EASE.npz'))
    lg_a = avg(sorted(glob.glob(f'{R4}/gcf-r4-e1-{ds}/out_peruser_e1_{ds}_s*_cond_a_LightGCN.npz')))
    ea_a = avg(glob.glob(f'{R4}/gcf-r4-cpu-e1-{ds}/out_peruser_e1cpu_{ds}_s1_cond_a_EASE.npz'))
    Eu = sorted(set(lg_a) & set(ea_a))
    for name, A, B, us in (('full GT, all warm users', lg_full, ea_full, sorted(set(lg_full) & set(ea_full))),
                           ('full GT, E users only', lg_full, ea_full, [u for u in Eu if u in lg_full and u in ea_full]),
                           ('E1-a (E users, H items, F masked)', lg_a, ea_a, Eu)):
        x = np.array([A[u] for u in us]); y = np.array([B[u] for u in us]); dd = x - y; dd = dd[dd != 0]
        J.setdefault(ds, {})[name.split(' (')[0].replace(', ', '|')] = dict(n=len(us), lgcn=float(x.mean()), ease=float(y.mean()), diff=float(x.mean() - y.mean()), p=float(wilcoxon(dd).pvalue))
        L.append(f'| {ds} | {name} | {len(us)} | {x.mean():.4f} | {y.mean():.4f} | {x.mean()-y.mean():+.4f} | {wilcoxon(dd).pvalue:.3g} |')
open(f'{R4}/e1a_decompose.md', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L)); json.dump(J, open(f'{R4}/e1a_decompose.json', 'w'), indent=1)
