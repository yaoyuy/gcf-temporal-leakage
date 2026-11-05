"""Round-4 summary.  usage: python summarize_r4.py [--posthoc-only]
env: R4_DIR (default out/r4; the local smoke test uses out/r4_smoke/kw/out), R3_DIR (default out/r3), INCLUDE_SMOKE=1.
Reads every res_*.json (+ peruser_*.npz next to it) below R4_DIR and writes R4_DIR/summary_r4.csv and R4_DIR/stats_r4.md.
Also computes, with NO new training, a users-only population alignment of round 3 (random/LOO per-user NDCG@20 restricted to
the warm GT test users) from out/r3 per-user files; written to R4_DIR/posthoc_r3_aligned.md.

Tables (-> paper):
 T-E1  controlled leakage: NDCG@20 per condition a/b/c/f x model; gap = LightGCN - EASE (and BPR-MF - EASE);
       DiD = mean over eval users of [gap_u(c) - gap_u(a)] with 95% user-bootstrap CI (2000 resamples); paired Wilcoxon
       LightGCN vs EASE per condition with rank-biserial r, Holm within dataset.
 T-RF  GT no-refit vs refit on train+val (warm = round-3 test rows; warmtv = rows warm w.r.t. train+val); Kendall tau.
 T-AL  random / LOO: round-3 population ('warm') vs GT-aligned population; Kendall tau of each vs the GT ranking.
 T-E2  LightGCN leak path: NDCG@20 per K x variant; delta(variant - neither) with user-bootstrap CI.
 T-E3  filter-then-split (FTS) vs split-then-filter (STF) GT: population sizes, NDCG@20 per model incl. time-aware
       baselines, Kendall tau(FTS ranking, STF ranking).
Seeds: closed-form/count models are deterministic given the split (run once, n_seeds_run=1); trained models report the
actual seed count. Per-user pairing uses raw user ids (users_raw) and averages a model's per-user values over its seeds."""
import os
ROOT = os.environ.get('GCF_ROOT', '.')  # project root containing exp/out/r2..r5 and paper/
import json, glob, os, re, csv, sys, numpy as np, pandas as pd
from scipy.stats import kendalltau, wilcoxon
E = os.path.join(ROOT, 'exp')
R = os.environ.get('R4_DIR', f'{E}/out/r4'); R3 = os.environ.get('R3_DIR', f'{E}/out/r3'); os.makedirs(R, exist_ok=True)
MET = ['NDCG@20', 'Recall@20', 'NDCG@10', 'Recall@10']
ORDER = ['MostPop', 'RecentPop', 'ItemKNN', 'ItemKNN-TD', 'EASE', 'EASE-TD', 'BPR-MF', 'LightGCN', 'SimGCL']
rs = np.random.default_rng(0)

# ---------------------------------------------------------------- load round-4 results
recs = []   # dict(exp, ds, setting, model, seed, metrics{pop_metric: v}, npz{pop: path}, param)
def npz_for(resp, tag, key, model):
    f = glob.glob(os.path.join(os.path.dirname(resp), f'*peruser_{tag}_{key}_{model}.npz'))
    return f[0] if f else None
stf_choice = {}
for p in glob.glob(R + '/**/*lr_choice_stf_*.json', recursive=True):
    ds = re.search(r'lr_choice_stf_(.*)\.json', p).group(1); stf_choice[ds] = {m: v[1] for m, v in json.load(open(p)).items()}
for p in sorted(glob.glob(R + '/**/*res_*.json', recursive=True)):
    try: d = json.load(open(p))
    except Exception: continue
    tag = re.sub(r'^(out_)?res_', '', os.path.basename(p))[:-5]
    if tag.startswith('smoke') and not os.environ.get('INCLUDE_SMOKE'): continue
    exp, ds, seed = d.get('exp'), d.get('dataset'), d.get('seed')
    for key, block in d.items():
        if not isinstance(block, dict) or key in ('construction', 'info', 'gt_reference'): continue
        for model, m in block.items():
            if model == 'info' or not isinstance(m, dict): continue
            if tag.startswith('stf_'):   # keep only the chosen lr for the stf lr grid
                mm = re.search(r'_lr([0-9e\-]+)$', tag)
                if mm and stf_choice.get(ds, {}).get(model) not in (None, mm.group(1)): continue
            base = dict(exp=exp, ds=ds, model=model, seed=seed, tag=tag, param=m.get('param', m.get('lr', '')))
            if exp == 'e1': recs.append(dict(base, setting=key.replace('cond_', ''), met={k: m.get(k) for k in MET}, npz=npz_for(p, tag, key, model)))
            elif exp == 'e2': recs.append(dict(base, setting=key, met={k: m.get(k) for k in MET}, npz=npz_for(p, tag, key, model)))
            elif exp == 'aligned':
                if key.endswith('_aligned'): recs.append(dict(base, setting=key, met={k: m.get(f'aligned_{k}') for k in MET}, npz=npz_for(p, tag, key, model)))
                else: recs.append(dict(base, setting=key, met={k: m.get(f'warm_{k}') for k in MET}, npz=npz_for(p, tag, key, model)))
            elif exp == 'gtrefit':
                if key == 'norefit': recs.append(dict(base, setting='norefit', met={k: m.get(k) for k in MET}, npz=npz_for(p, tag, key, model)))
                elif key == 'refit': recs.append(dict(base, setting='refit', met={k: m.get(f'warm_{k}') for k in MET}, npz=npz_for(p, tag, key, model)))
                elif key == 'refit_warmtv': recs.append(dict(base, setting='refit_warmtv', met={k: m.get(f'warmtv_{k}') for k in MET}, npz=npz_for(p, tag, key, model)))
            elif exp == 'gt':
                recs.append(dict(base, setting='gt', met={k: m.get(k) for k in MET}, npz=npz_for(p, tag, key, model)))
                recs.append(dict(base, setting='gt_full', met={k: m.get(f'full_{k}') for k in MET}, npz=None))
# dedup (exp, ds, setting, model, seed): GPU and CPU kernels never overlap by design; keep the first and report duplicates
seen, dup, R4 = set(), [], []
for r in recs:
    k = (r['exp'], r['ds'], r['setting'], r['model'], r['seed'])
    if k in seen: dup.append(k); continue
    seen.add(k); R4.append(r)
def group():
    g = {}
    for r in R4: g.setdefault((r['exp'], r['ds'], r['setting'], r['model']), []).append(r)
    return g
G = group()
rows = []
for (exp, ds, st, mo), rr in sorted(G.items()):
    row = dict(exp=exp, dataset=ds, setting=st, model=mo, n_seeds=len(rr), seeds=' '.join(str(r['seed']) for r in sorted(rr, key=lambda r: r['seed'])),
               param=';'.join(sorted({str(r['param']) for r in rr})))
    for me in MET:
        v = np.array([r['met'][me] for r in rr if r['met'].get(me) is not None], dtype=float)
        row[f'{me}_mean'] = float(np.nanmean(v)) if len(v) else np.nan; row[f'{me}_std'] = float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.
    rows.append(row)
if rows:
    with open(R + '/summary_r4.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
M = {(r['exp'], r['dataset'], r['setting'], r['model']): r for r in rows}
def mean(exp, ds, st, mo, me='NDCG@20'):
    r = M.get((exp, ds, st, mo)); return r[f'{me}_mean'] if r else np.nan
def cell(exp, ds, st, mo):
    r = M.get((exp, ds, st, mo))
    return f"{r['NDCG@20_mean']:.4f}±{r['NDCG@20_std']:.4f} ({r['n_seeds']})" if r else '–'
def peruser(exp, ds, st, mo, pop='warm', me='NDCG@20'):
    """raw user id -> value averaged over the seeds available"""
    acc = {}
    for r in G.get((exp, ds, st, mo), []):
        if not r['npz']: continue
        z = np.load(r['npz']); key = f'{pop}_{me}'
        if key not in z: key = [k for k in z.files if k.endswith(me)][0]
        for u, v in zip(z['users_raw'], z[key]):
            if not np.isnan(v): acc.setdefault(u, []).append(float(v))
    return {u: float(np.mean(v)) for u, v in acc.items()}
def boot_ci(x, B=2000):
    x = np.asarray(x, float)
    if len(x) < 2: return (np.nan, np.nan)
    bs = x[rs.integers(0, len(x), (B, len(x)))].mean(1); return tuple(np.percentile(bs, [2.5, 97.5]))
def wil(x, y):
    d = np.asarray(x) - np.asarray(y); d = d[d != 0]
    if len(d) < 1: return 1.0, 0.0
    p = wilcoxon(d, zero_method='wilcox').pvalue
    rk = pd.Series(np.abs(d)).rank().values; rpos, rneg = rk[d > 0].sum(), rk[d < 0].sum()
    return float(p), float((rpos - rneg) / (rpos + rneg))
def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for r, i in enumerate(o): run = max(run, min(1, (m - r) * ps[i])); adj[i] = run
    return adj
def tau(a, b):
    ms = [m for m in a if m in b and not np.isnan(a[m]) and not np.isnan(b[m])]
    if len(ms) < 3: return np.nan, len(ms)
    return float(kendalltau([a[m] for m in ms], [b[m] for m in ms]).statistic), len(ms)
dss = sorted({r['dataset'] for r in rows})
J = {'e1': {}, 'e2': {}, 'refit': {}, 'aligned': {}, 'e3': {}}   # machine-readable copy for paper/analysis/make_tables_r4.py
L = ['# Round 4 statistics', '', f'Generated by exp/summarize_r4.py from {R}. NDCG@20 = mean ± std over seeds (n). '
     'Closed-form / count models are deterministic (n = 1 run).', '']

# ---------------------------------------------------------------- T-E1
L += ['## T-E1 controlled leakage (fixed GT test set: held-out items of evaluation users)', '',
      'a = GT train; b = + other users\' test-period rows (cross-user); c = b + own future rows (same-user); f = own future only. Equal size.', '']
for ds in dss:
    cons = [c for c in 'abcf' if any(k[:3] == ('e1', ds, c) for k in M)]
    if not cons: continue
    ms = [m for m in ORDER if any(('e1', ds, c, m) in M for c in cons)]
    L += [f'### {ds}', '', '| cond | ' + ' | '.join(ms) + ' | LightGCN−EASE | BPR-MF−EASE |', '|' + '---|' * (len(ms) + 3)]
    for c in cons:
        L.append(f'| {c} | ' + ' | '.join(cell('e1', ds, c, m) for m in ms) +
                 f" | {mean('e1', ds, c, 'LightGCN') - mean('e1', ds, c, 'EASE'):+.4f} | {mean('e1', ds, c, 'BPR-MF') - mean('e1', ds, c, 'EASE'):+.4f} |")
    L += ['', '| cond | LightGCN vs EASE: mean user diff | 95% CI | Wilcoxon p (Holm) | rank-biserial r | DiD vs a | DiD 95% CI | n users |', '|---|---|---|---|---|---|---|---|']
    pa_l, pa_e = peruser('e1', ds, 'a', 'LightGCN'), peruser('e1', ds, 'a', 'EASE'); out = []
    for c in cons:
        pl, pe = peruser('e1', ds, c, 'LightGCN'), peruser('e1', ds, c, 'EASE'); us = sorted(set(pl) & set(pe))
        if not us: continue
        d = np.array([pl[u] - pe[u] for u in us]); p, rb = wil([pl[u] for u in us], [pe[u] for u in us])
        u2 = [u for u in us if u in pa_l and u in pa_e]; dd = np.array([(pl[u] - pe[u]) - (pa_l[u] - pa_e[u]) for u in u2])
        out.append([c, d.mean(), boot_ci(d), p, rb, dd.mean() if len(dd) else np.nan, boot_ci(dd) if c != 'a' else (np.nan, np.nan), len(us)])
    if out:
        adj = holm([o[3] for o in out])
        for o, pa in zip(out, adj):
            L.append(f'| {o[0]} | {o[1]:+.4f} | [{o[2][0]:+.4f}, {o[2][1]:+.4f}] | {pa:.3g} | {o[4]:+.3f} | {o[5]:+.4f} | [{o[6][0]:+.4f}, {o[6][1]:+.4f}] | {o[7]} |')
            J['e1'].setdefault(ds, {})[o[0]] = dict(diff=float(o[1]), ci=[float(o[2][0]), float(o[2][1])], p_holm=float(pa), r=float(o[4]),
                                                   did=float(o[5]), did_ci=[float(o[6][0]), float(o[6][1])], n_users=int(o[7]))
    L.append('')
# ---------------------------------------------------------------- T-RF
L += ['## T-RF GT without vs with refit on train+val', '']
for ds in dss:
    ms = [m for m in ORDER if ('gtrefit', ds, 'norefit', m) in M]
    if not ms: continue
    a = {m: mean('gtrefit', ds, 'norefit', m) for m in ms}; b = {m: mean('gtrefit', ds, 'refit', m) for m in ms}; c = {m: mean('gtrefit', ds, 'refit_warmtv', m) for m in ms}
    L += [f'### {ds}', '', '| model | no refit | refit (same test rows) | refit (rows warm wrt train+val) |', '|---|---|---|---|']
    for m in ms: L.append(f"| {m} | {cell('gtrefit', ds, 'norefit', m)} | {cell('gtrefit', ds, 'refit', m)} | {cell('gtrefit', ds, 'refit_warmtv', m)} |")
    t1, n1 = tau(a, b); J['refit'][ds] = dict(tau=t1, n=n1, best_norefit=max(a, key=a.get), best_refit=max(b, key=lambda k: (b[k] if not np.isnan(b[k]) else -1)))
    L += ['', f'Kendall τ(no refit, refit) = {t1:.3f} over {n1} models; best: {max(a, key=a.get)} → {max(b, key=lambda k: (b[k] if not np.isnan(b[k]) else -1))}', '']
# ---------------------------------------------------------------- T-AL
L += ['## T-AL population-aligned random / LOO (users warm under GT, items in GT train)', '']
for ds in dss:
    ms = [m for m in ORDER if ('aligned', ds, 'random', m) in M or ('aligned', ds, 'loo', m) in M]
    if not ms: continue
    gt = {m: mean('gtrefit', ds, 'norefit', m) for m in ms}
    if all(np.isnan(v) for v in gt.values()): gt = {m: mean('gt', ds, 'gt', m) for m in ms}
    L += [f'### {ds}', '', '| model | random (r3 pop.) | random (aligned) | loo (r3 pop.) | loo (aligned) | GT |', '|---|---|---|---|---|---|']
    for m in ms: L.append(f"| {m} | {cell('aligned', ds, 'random', m)} | {cell('aligned', ds, 'random_aligned', m)} | {cell('aligned', ds, 'loo', m)} | {cell('aligned', ds, 'loo_aligned', m)} | {gt[m]:.4f} |")
    for pr in ('random', 'loo'):
        t0, n0 = tau({m: mean('aligned', ds, pr, m) for m in ms}, gt); t1, n1 = tau({m: mean('aligned', ds, pr + '_aligned', m) for m in ms}, gt)
        L.append(f'- τ({pr}, GT): r3 population {t0:.3f}, aligned {t1:.3f} (n models {n1})')
        J['aligned'].setdefault(ds, {})[pr] = dict(tau_r3pop=t0, tau_aligned=t1, n=n1)
    L.append('')
# ---------------------------------------------------------------- T-E2
L += ['## T-E2 graph-propagation leakage (LightGCN; leak set = e1-c leak rows)', '',
      'neither: adj=train, loss=train | adj: adj=train+L, loss=train | both: adj=train+L, loss=train+L | loss: adj=train, loss=train+L', '']
for ds in dss:
    sets = sorted({k[2] for k in M if k[0] == 'e2' and k[1] == ds and k[3] == 'LightGCN'})
    if not sets: continue
    Ks = sorted({int(s.split('_K')[1]) for s in sets}); vs = ['neither', 'adj', 'both', 'loss']
    L += [f'### {ds}', '', '| K | ' + ' | '.join(vs) + ' | Δadj [95% CI] | Δboth [95% CI] | Δloss [95% CI] |', '|' + '---|' * 8]
    for K in Ks:
        base = peruser('e2', ds, f'neither_K{K}', 'LightGCN'); cells, deltas = [], []
        for v in vs: cells.append(cell('e2', ds, f'{v}_K{K}', 'LightGCN'))
        for v in ('adj', 'both', 'loss'):
            pv = peruser('e2', ds, f'{v}_K{K}', 'LightGCN'); us = sorted(set(pv) & set(base))
            if not us: deltas.append('–'); continue
            d = np.array([pv[u] - base[u] for u in us]); lo, hi = boot_ci(d); deltas.append(f'{d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]')
            J['e2'].setdefault(ds, {})[f'{v}_K{K}'] = dict(delta=float(d.mean()), ci=[float(lo), float(hi)], n_users=len(us))
        L.append(f'| {K} | ' + ' | '.join(cells) + ' | ' + ' | '.join(deltas) + ' |')
    refs = [m for m in ORDER if ('e2', ds, 'neither_ref', m) in M]
    for m in refs: L.append(f"- {m} (no graph): neither {cell('e2', ds, 'neither_ref', m)} → both {cell('e2', ds, 'both_ref', m)}")
    L.append('')
# ---------------------------------------------------------------- T-E3
L += ['## T-E3 filter-then-split (FTS, rounds 2-3 data) vs split-then-filter (STF), GT warm NDCG@20; time-aware baselines', '']
r3sum = {}
if os.path.exists(R3 + '/summary.csv'):
    for r in csv.DictReader(open(R3 + '/summary.csv')):
        if r['protocol'] == 'global_temporal': r3sum[(r['dataset'], r['model'])] = float(r['warm_NDCG@20_mean'])
st3 = json.load(open(f'{E}/data/e3/stats_e3.json')) if os.path.exists(f'{E}/data/e3/stats_e3.json') else {}
for base in sorted({d.replace('-stf', '') for d in dss if ('gt', d, 'gt', 'EASE') in M or d.endswith('-stf')}):
    stf = base + '-stf'; ms = [m for m in ORDER if ('gt', base, 'gt', m) in M or ('gt', stf, 'gt', m) in M or (base, m) in r3sum]
    if not ms: continue
    fts = {m: (mean('gt', base, 'gt', m) if ('gt', base, 'gt', m) in M else r3sum.get((base, m), np.nan)) for m in ms}
    sv = {m: mean('gt', stf, 'gt', m) for m in ms}
    L += [f'### {base}', '', '| model | FTS GT | STF GT |', '|---|---|---|']
    for m in ms: L.append(f"| {m} | {'–' if np.isnan(fts[m]) else f'{fts[m]:.4f}'}{'' if ('gt', base, 'gt', m) in M or np.isnan(fts[m]) else ' (r3)'} | {cell('gt', stf, 'gt', m)} |")
    t, n = tau(fts, sv); s = st3.get(stf, {}); J['e3'][base] = dict(tau=t, n=n, fts_r3={m: ('gt', base, 'gt', m) not in M for m in ms})
    L += ['', f'Kendall τ(FTS, STF) = {t:.3f} over {n} models. STF test rows warm: {s.get("test_warm_rows", "?")} of {s.get("test_rows", "?")} '
          f'(user-cold {s.get("test_usercold_rows_pct", "?")}%, item-cold {s.get("test_itemcold_rows_pct", "?")}%).', '']
# ---------------------------------------------------------------- bookkeeping
L += ['## Seeds actually run / missing', '']
L += [f"- {r['exp']} {r['dataset']} {r['setting']} {r['model']}: n={r['n_seeds']} ({r['seeds']})" for r in rows if r['n_seeds'] < 3 and r['model'] in ('BPR-MF', 'LightGCN', 'SimGCL')] or ['- all trained-model cells have ≥3 seeds']
if dup: L += ['', f'- duplicate records ignored: {len(dup)} (first kept)']
if rows: open(R + '/stats_r4.md', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L[:80]))
if rows: json.dump(J, open(R + '/stats_r4.json', 'w'), indent=1, default=float)
print('rows', len(rows), 'records', len(R4))

# ---------------------------------------------------------------- post-hoc users-only alignment of round 3 (no training)
def posthoc():
    sys.path.insert(0, E)
    LR = {}; exec(re.search(r"LR_R2 = (\{.*?\n\})", open(f'{E}/driver_v4.py').read(), re.S).group(0), {}, LR); LR = LR['LR_R2']
    out = ['# Post-hoc users-only population alignment of round 3 (no new training)', '',
           'Round-3 per-user warm NDCG@20 for random / LOO restricted to users that are warm GT test users (in GT train with ≥1 warm GT test row). '
           'Items are NOT restricted (needs the aligned re-run). Split reconstruction: same code and seeds as run_v4.py.', '']
    for ds in ('ml-1m', 'gowalla', 'amz-music', 'amz-games'):
        bc = glob.glob(f'{R3}/gcf-r3-basetune-{ds}/**/*base_choice_{ds}.json', recursive=True)
        if not bc: out.append(f'- {ds}: round-3 base_choice not available yet; skipped'); continue
        bc = {k: v[1] for k, v in json.load(open(bc[0])).items()}
        df = (pd.read_csv(f'{E}/data/ml-1m/ratings.dat', sep='::', engine='python', names=['u', 'i', 'r', 't']).drop(columns='r') if ds == 'ml-1m'
              else pd.read_csv(f'{E}/data/{ds}.csv', usecols=['u', 'i', 't']))
        g = df.sort_values('t', kind='stable'); n = len(g); gtr, gte = g.iloc[:int(.8 * n)], g.iloc[int(.9 * n):]
        GU = set(gte[gte.u.isin(set(gtr.u)) & gte.i.isin(set(gtr.i))].u)
        res = {}
        for pr in ('random', 'loo', 'global_temporal'):
            for m in ('MostPop', 'ItemKNN', 'EASE', 'BPR-MF', 'LightGCN'):
                suf = ('' if m == 'MostPop' else f"_k{bc.get(f'{pr}/ItemKNN')}" if m == 'ItemKNN' else f"_lam{bc.get(f'{pr}/EASE')}" if m == 'EASE'
                       else f"_lr{LR[ds][f'{pr}/{m}']}")
                vals_all, vals_al = [], []
                for s in (1, 2, 3, 4, 5):
                    f = glob.glob(f'{R3}/gcf-r3-*/**/*peruser_{ds}_{pr}_s{s}_{m}{suf}_{pr}_{m}.npz', recursive=True)
                    if not f and m in ('MostPop', 'ItemKNN', 'EASE') and pr != 'random':
                        if s > 1: continue
                    if not f: continue
                    z = np.load(f[0]); v = z['warm_NDCG@20']
                    if pr == 'random':
                        rng = np.random.default_rng(s); p = rng.permutation(len(df)); users = np.array(sorted(set(df.iloc[p[:int(.8 * len(df))]].u)))
                    elif pr == 'loo':
                        d2 = df.sort_values(['u', 't'], kind='stable'); rk = d2.groupby('u').cumcount(ascending=False); users = np.array(sorted(set(d2[rk >= 2].u)))
                    else: users = np.array(sorted(set(gtr.u)))
                    raw = users[z['users']]; keep = np.array([u in GU for u in raw]) & ~np.isnan(v)
                    vals_all.append(np.nanmean(v)); vals_al.append(v[keep].mean() if keep.any() else np.nan)
                if vals_all: res[(pr, m)] = (np.mean(vals_all), np.mean(vals_al), len(vals_all))
        if not res: out.append(f'- {ds}: no round-3 per-user files found'); continue
        ms = [m for m in ('MostPop', 'ItemKNN', 'EASE', 'BPR-MF', 'LightGCN') if ('global_temporal', m) in res]
        out += [f'## {ds} (warm GT test users: {len(GU)})', '', '| model | random (all) | random (GT users) | loo (all) | loo (GT users) | GT | seeds (rnd/loo/GT) |', '|---|---|---|---|---|---|---|']
        for m in ms:
            r_, l_, g_ = res.get(('random', m), (np.nan,) * 3), res.get(('loo', m), (np.nan,) * 3), res[('global_temporal', m)]
            out.append(f'| {m} | {r_[0]:.4f} | {r_[1]:.4f} | {l_[0]:.4f} | {l_[1]:.4f} | {g_[0]:.4f} | {r_[2]}/{l_[2]}/{g_[2]} |')
        gt = {m: res[('global_temporal', m)][0] for m in ms}
        for pr in ('random', 'loo'):
            a = {m: res[(pr, m)][0] for m in ms if (pr, m) in res}; b = {m: res[(pr, m)][1] for m in ms if (pr, m) in res}
            out.append(f'- τ({pr}, GT): all users {tau(a, gt)[0]:.3f}, GT-warm users only {tau(b, gt)[0]:.3f}; best {max(a, key=a.get)} → {max(b, key=b.get)} (GT best {max(gt, key=gt.get)})')
        out.append('')
    open(R + '/posthoc_r3_aligned.md', 'w').write('\n'.join(out) + '\n'); print('\n'.join(out))
if '--posthoc' in sys.argv or '--posthoc-only' in sys.argv: posthoc()
