"""Round-5 summary: merges round-4 (out/r4) and round-5 (out/r5) results on the same protocol and the same test sets.
usage: python summarize_r5.py      env: R4_DIR, R5_DIR (default out/r4, out/r5; the local smoke uses other dirs), INCLUDE_SMOKE=1
Writes R5_DIR/summary_r45.csv, R5_DIR/stats_r45.md, R5_DIR/stats_r45.json. Never writes into R4_DIR.

Merge rules
 - Trained models (BPR-MF, LightGCN, SimGCL): R4 and R5 seeds are pooled per cell (exp, dataset, setting, model). The same
   (cell, seed) twice -> the R4 record is kept and the duplicate is listed. R5 only adds seed numbers / conditions R4 did not run.
 - Closed-form / count models (MostPop, ItemKNN, EASE, RecentPop, EASE-TD, ItemKNN-TD) are deterministic given the split. Where
   R5 re-ran a cell with the extended grid (a superset of the R4 grid, chosen by the same validation NDCG@20), the R5 record
   REPLACES the R4 one in every table; the R4 value/param are kept in columns param_r4 / NDCG@20_r4 and in table T-GRID.
 - STF: only the lr chosen on seed 1 enters (R4 choice for BPR-MF/LightGCN, R5 choice for SimGCL).
 - Same-test-set check: the e1/e2 leak construction, gt/stf population info and aligned split info must be identical between
   R4 and R5 for the same dataset (and split seed); any mismatch is printed at the top of stats_r45.md.
Statistics: NDCG@20 mean ± std over seeds (n = actual seeds), 95% t-interval over seeds (n >= 2); per-user tests on
per-user NDCG@20 averaged over each model's seeds: paired Wilcoxon signed-rank + rank-biserial r, Holm within dataset/table,
95% user-bootstrap CI (2000 resamples, rng seed 0) of the mean user difference."""
import os
ROOT = os.environ.get('GCF_ROOT', '.')  # project root containing exp/out/r2..r5 and paper/
import json, glob, os, re, csv, sys, numpy as np, pandas as pd
from scipy.stats import kendalltau, wilcoxon, t as tdist
E = os.path.join(ROOT, 'exp')
R4 = os.environ.get('R4_DIR', f'{E}/out/r4'); R5 = os.environ.get('R5_DIR', f'{E}/out/r5'); R3 = f'{E}/out/r3'
os.makedirs(R5, exist_ok=True)
MET = ['NDCG@20', 'Recall@20', 'NDCG@10', 'Recall@10']
ORDER = ['MostPop', 'RecentPop', 'ItemKNN', 'ItemKNN-TD', 'EASE', 'EASE-TD', 'BPR-MF', 'LightGCN', 'SimGCL']
TRAINED = ('BPR-MF', 'LightGCN', 'SimGCL')
rs = np.random.default_rng(0)

def load(root, src):
    recs, infos = [], []
    stf_choice = {}
    for p in glob.glob(root + '/**/*lr_choice_stf_*.json', recursive=True):
        if re.search(r'/gcf-r[45]-smoke', p) or '_smoke' in p: continue
        mm = re.search(r'lr_choice_stf_(simgcl_)?(.*)\.json', p); ds = mm.group(2)
        stf_choice.setdefault(ds, {}).update({m: v[1] for m, v in json.load(open(p)).items()})
    for p in sorted(glob.glob(root + '/**/*res_*.json', recursive=True)):
        if '/r4_smoke' in p or '/r5_smoke' in p or re.search(r'/gcf-r[45]-smoke', p): continue   # smoke kernels: subsampled data, never results
        try: d = json.load(open(p))
        except Exception: continue
        tag = re.sub(r'^(out_)?res_', '', os.path.basename(p))[:-5]
        if tag.startswith('smoke') and not os.environ.get('INCLUDE_SMOKE'): continue
        exp, ds, seed = d.get('exp'), d.get('dataset'), d.get('seed')
        for ik in ('construction', 'info', 'gt_reference'):
            if ik in d: infos.append((src, exp, ds, seed, ik, tag, d[ik]))
        for key, block in d.items():
            if not isinstance(block, dict) or key in ('construction', 'info', 'gt_reference'): continue
            if 'info' in block and isinstance(block['info'], dict): infos.append((src, exp, ds, seed, key, tag, block['info']))
            for model, m in block.items():
                if model == 'info' or not isinstance(m, dict): continue
                if tag.startswith('stf_'):
                    mm = re.search(r'_lr([0-9e\-]+)$', tag)
                    if mm and stf_choice.get(ds, {}).get(model) not in (None, mm.group(1)): continue
                def npz(k=key): f = glob.glob(os.path.join(os.path.dirname(p), f'*peruser_{tag}_{k}_{model}.npz')); return f[0] if f else None
                base = dict(src=src, exp=exp, ds=ds, model=model, seed=seed, tag=tag, param=m.get('param', m.get('lr', '')),
                            edge=m.get('at_grid_edge'), val_grid=m.get('val_grid'))
                if exp in ('e1', 'e2'): recs.append(dict(base, setting=key.replace('cond_', '') if exp == 'e1' else key, met={k: m.get(k) for k in MET}, npz=npz()))
                elif exp == 'aligned':
                    pre = 'aligned_' if key.endswith('_aligned') else 'warm_'
                    recs.append(dict(base, setting=key, met={k: m.get(pre + k) for k in MET}, npz=npz()))
                elif exp == 'gtrefit':
                    pre = {'norefit': '', 'refit': 'warm_', 'refit_warmtv': 'warmtv_'}.get(key)
                    if pre is not None: recs.append(dict(base, setting=key, met={k: m.get(pre + k) for k in MET}, npz=npz()))
                elif exp == 'gt':
                    recs.append(dict(base, setting='gt', met={k: m.get(k) for k in MET}, npz=npz()))
                    recs.append(dict(base, setting='gt_full', met={k: m.get(f'full_{k}') for k in MET}, npz=None))
    return recs, infos
r4, i4 = load(R4, 'r4'); r5, i5 = load(R5, 'r5')
# ---------------------------------------------------------------- merge
key = lambda r: (r['exp'], r['ds'], r['setting'], r['model'], r['seed'])
r5cf = {key(r): r for r in r5 if r['model'] not in TRAINED}
replaced = {}   # cell key (no seed) -> r4 record replaced by r5
ALL, seen, dup = [], set(), []
for r in r4:
    k = key(r)
    if r['model'] not in TRAINED and k in r5cf: replaced[k[:4]] = r; continue
    if k in seen: dup.append(('r4', k)); continue
    seen.add(k); ALL.append(r)
for r in r5:
    k = key(r)
    if k in seen: dup.append(('r5', k)); continue
    seen.add(k); ALL.append(r)
G = {}
for r in ALL: G.setdefault((r['exp'], r['ds'], r['setting'], r['model']), []).append(r)
# ---------------------------------------------------------------- same-test-set check
mism = []
def norm(x): return json.dumps(x, sort_keys=True, default=str)
ref = {}
for src, exp, ds, seed, ik, tag, info in i4 + i5:
    if exp in ('e1', 'e2') and ik == 'construction': k = ('leak', ds)
    elif exp == 'gt' and ik == 'info': k = ('gt', ds)
    elif exp == 'aligned' and ik in ('random', 'loo'): k = ('aligned', ds, ik, seed if ik == 'random' else 1)
    elif exp == 'aligned' and ik == 'gt_reference': k = ('aligned_ref', ds)
    else: continue
    if tag.startswith('smoke'): continue
    if k not in ref: ref[k] = (src, tag, norm(info))
    elif ref[k][2] != norm(info): mism.append(f'{k}: {ref[k][0]}:{ref[k][1]} vs {src}:{tag}')
n_checked = len(ref)
# ---------------------------------------------------------------- rows
def tci(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    if len(v) < 2: return (np.nan, np.nan)
    h = tdist.ppf(.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)); return (v.mean() - h, v.mean() + h)
rows = []
for (exp, ds, st, mo), rr in sorted(G.items()):
    rr = sorted(rr, key=lambda r: r['seed'])
    row = dict(exp=exp, dataset=ds, setting=st, model=mo, n_seeds=len(rr), seeds=' '.join(str(r['seed']) for r in rr),
               seeds_r4=' '.join(str(r['seed']) for r in rr if r['src'] == 'r4'), seeds_r5=' '.join(str(r['seed']) for r in rr if r['src'] == 'r5'),
               grid='r5ext' if (exp, ds, st, mo) in replaced or (mo not in TRAINED and any(r['src'] == 'r5' for r in rr)) else 'r4',
               param=';'.join(sorted({str(r['param']) for r in rr})), at_grid_edge=';'.join(sorted({str(r['edge']) for r in rr if r['edge'] is not None})))
    old = replaced.get((exp, ds, st, mo))
    row['param_r4'] = '' if old is None else str(old['param']); row['NDCG@20_r4'] = '' if old is None else old['met'].get('NDCG@20')
    for me in MET:
        v = np.array([r['met'][me] for r in rr if r['met'].get(me) is not None], dtype=float)
        row[f'{me}_mean'] = float(np.nanmean(v)) if len(v) else np.nan; row[f'{me}_std'] = float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.
    lo, hi = tci([r['met'].get('NDCG@20') for r in rr if r['met'].get('NDCG@20') is not None]); row['NDCG@20_ci95_lo'] = lo; row['NDCG@20_ci95_hi'] = hi
    rows.append(row)
cols = list(rows[0].keys()) if rows else []
with open(R5 + '/summary_r45.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
M = {(r['exp'], r['dataset'], r['setting'], r['model']): r for r in rows}
def mean(exp, ds, st, mo, me='NDCG@20'):
    r = M.get((exp, ds, st, mo)); return r[f'{me}_mean'] if r else np.nan
def cell(exp, ds, st, mo):
    r = M.get((exp, ds, st, mo))
    return f"{r['NDCG@20_mean']:.4f}±{r['NDCG@20_std']:.4f} ({r['n_seeds']})" if r else '–'
_pu = {}
def peruser(exp, ds, st, mo, pop='warm', me='NDCG@20'):
    k = (exp, ds, st, mo, pop, me)
    if k in _pu: return _pu[k]
    acc = {}
    for r in G.get((exp, ds, st, mo), []):
        if not r['npz']: continue
        z = np.load(r['npz']); kk = f'{pop}_{me}'
        if kk not in z: kk = [x for x in z.files if x.endswith(me)][0]
        for u, v in zip(z['users_raw'], z[kk]):
            if not np.isnan(v): acc.setdefault(u, []).append(float(v))
    _pu[k] = {u: float(np.mean(v)) for u, v in acc.items()}; return _pu[k]
def boot_ci(x, B=2000):
    x = np.asarray(x, float)
    if len(x) < 2: return (np.nan, np.nan)
    bs = x[rs.integers(0, len(x), (B, len(x)))].mean(1); return tuple(np.percentile(bs, [2.5, 97.5]))
def wil(d):
    d = np.asarray(d, float); d = d[d != 0]
    if len(d) < 1: return 1.0, 0.0
    p = wilcoxon(d, zero_method='wilcox').pvalue
    rk = pd.Series(np.abs(d)).rank().values; rpos, rneg = rk[d > 0].sum(), rk[d < 0].sum()
    return float(p), float((rpos - rneg) / (rpos + rneg))
def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for r, i in enumerate(o): run = max(run, min(1, (m - r) * ps[i])); adj[i] = run
    return adj
def paired(a, b):
    us = sorted(set(a) & set(b)); d = np.array([a[u] - b[u] for u in us])
    if not len(d): return None
    p, rb = wil(d); return dict(diff=float(d.mean()), ci=[float(x) for x in boot_ci(d)], p=p, r=rb, n=len(us), d=d, us=us)
def tau(a, b):
    ms = [m for m in a if m in b and not np.isnan(a[m]) and not np.isnan(b[m])]
    if len(ms) < 3: return np.nan, len(ms)
    return float(kendalltau([a[m] for m in ms], [b[m] for m in ms]).statistic), len(ms)
f4 = lambda x: '–' if x is None or (isinstance(x, float) and np.isnan(x)) else f'{x:+.4f}'
dss = sorted({r['dataset'] for r in rows})
J = {'e1': {}, 'e2': {}, 'e3': {}, 'grid': {}, 'refit': {}, 'aligned': {}, 'checks': {'mismatches': mism, 'n_checked': n_checked, 'duplicates': len(dup)}}
L = ['# Rounds 4 + 5 merged statistics', '',
     f'Generated by exp/summarize_r5.py from {R4} (R4) + {R5} (R5). NDCG@20 = mean ± std over seeds (n = actual seeds). '
     'Closed-form / count models are deterministic (n = 1); cells re-run with the R5 extended grid replace the R4 value (see T-GRID).', '',
     f'**Same-test-set check:** {n_checked} construction/population records compared between R4 and R5; '
     + ('all identical.' if not mism else f'{len(mism)} MISMATCHES: ' + '; '.join(mism[:20])),
     f'Duplicate (cell, seed) records ignored: {len(dup)}' + (f' (e.g. {dup[:3]})' if dup else ''), '']
# ---------------------------------------------------------------- T-E1
L += ['## T-E1 controlled leakage (fixed GT test set)', '',
      'a = GT train; b = + other users\' test-period rows (cross-user); c = b + own future rows; f = own future only. Equal size.',
      'Per-user tests: model vs EASE in each condition (Wilcoxon, Holm within dataset over all model×condition tests); '
      'DiD = mean over users of [gap_u(cond) − gap_u(a)], 95% user-bootstrap CI, Wilcoxon on the per-user DiD (Holm within dataset).', '']
for ds in dss:
    cons = [c for c in 'abcf' if any(k[:3] == ('e1', ds, c) for k in M)]
    if not cons: continue
    ms = [m for m in ORDER if any(('e1', ds, c, m) in M for c in cons)]
    L += [f'### {ds}', '', '| cond | ' + ' | '.join(ms) + ' |', '|' + '---|' * (len(ms) + 1)]
    for c in cons: L.append(f'| {c} | ' + ' | '.join(cell('e1', ds, c, m) for m in ms) + ' |')
    tests = []
    for mo in [m for m in TRAINED if m in ms]:
        pa = paired(peruser('e1', ds, 'a', mo), peruser('e1', ds, 'a', 'EASE'))
        for c in cons:
            pc = paired(peruser('e1', ds, c, mo), peruser('e1', ds, c, 'EASE'))
            if not pc: continue
            did = None
            if c != 'a' and pa:
                ga = dict(zip(pa['us'], pa['d'])); gc = dict(zip(pc['us'], pc['d'])); us = sorted(set(ga) & set(gc))
                dd = np.array([gc[u] - ga[u] for u in us])
                if len(dd): p2, r2 = wil(dd); did = dict(did=float(dd.mean()), ci=[float(x) for x in boot_ci(dd)], p=p2, r=r2, n=len(us))
            tests.append((mo, c, pc, did))
    if tests:
        adj = holm([t[2]['p'] for t in tests]); dl = [t for t in tests if t[3]]; adj2 = dict(zip([(t[0], t[1]) for t in dl], holm([t[3]['p'] for t in dl]))) if dl else {}
        L += ['', '| model | cond | n seeds | model − EASE | 95% CI | Wilcoxon p (Holm) | r | DiD vs a | DiD 95% CI | DiD p (Holm) | n users |', '|---|---|---|---|---|---|---|---|---|---|---|']
        for (mo, c, pc, did), pa_ in zip(tests, adj):
            ns = M[('e1', ds, c, mo)]['n_seeds']
            L.append(f"| {mo} | {c} | {ns} | {pc['diff']:+.4f} | [{pc['ci'][0]:+.4f}, {pc['ci'][1]:+.4f}] | {pa_:.3g} | {pc['r']:+.3f} | "
                     + (f"{did['did']:+.4f} | [{did['ci'][0]:+.4f}, {did['ci'][1]:+.4f}] | {adj2[(mo, c)]:.3g}" if did else '– | – | –') + f" | {pc['n']} |")
            J['e1'].setdefault(ds, {}).setdefault(mo, {})[c] = dict(n_seeds=ns, diff=pc['diff'], ci=pc['ci'], p_holm=float(pa_), r=pc['r'], n_users=pc['n'],
                **({'did': did['did'], 'did_ci': did['ci'], 'did_p_holm': float(adj2[(mo, c)])} if did else {}))
    L.append('')
# ---------------------------------------------------------------- T-E2
L += ['## T-E2 graph-propagation leakage (LightGCN; leak set = e1-c leak rows)', '',
      'neither: adj=train, loss=train | adj: adj=train+L | both: adj+loss | loss: loss only. Δ = variant − neither per user '
      '(seed-averaged), 95% user-bootstrap CI, Wilcoxon, Holm within dataset.', '']
for ds in dss:
    sets = sorted({k[2] for k in M if k[0] == 'e2' and k[1] == ds and k[3] == 'LightGCN'})
    if not sets: continue
    Ks = sorted({int(s.split('_K')[1]) for s in sets}); vs = ['neither', 'adj', 'both', 'loss']
    L += [f'### {ds}', '', '| K | ' + ' | '.join(vs) + ' |', '|' + '---|' * 5]
    for K in Ks: L.append(f'| {K} | ' + ' | '.join(cell('e2', ds, f'{v}_K{K}', 'LightGCN') for v in vs) + ' |')
    tests = []
    for K in Ks:
        base = peruser('e2', ds, f'neither_K{K}', 'LightGCN')
        for v in ('adj', 'both', 'loss'):
            pv = peruser('e2', ds, f'{v}_K{K}', 'LightGCN')
            if pv and base: r = paired(pv, base); tests.append((K, v, r)) if r else None
    if tests:
        adj = holm([t[2]['p'] for t in tests])
        L += ['', '| K | variant | n seeds | Δ vs neither | 95% CI | Wilcoxon p (Holm) | r | n users |', '|---|---|---|---|---|---|---|---|']
        for (K, v, r), pa_ in zip(tests, adj):
            ns = min(M[('e2', ds, f'{v}_K{K}', 'LightGCN')]['n_seeds'], M[('e2', ds, f'neither_K{K}', 'LightGCN')]['n_seeds'])
            L.append(f"| {K} | {v} | {ns} | {r['diff']:+.4f} | [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}] | {pa_:.3g} | {r['r']:+.3f} | {r['n']} |")
            J['e2'].setdefault(ds, {})[f'{v}_K{K}'] = dict(n_seeds=ns, delta=r['diff'], ci=r['ci'], p_holm=float(pa_), r=r['r'], n_users=r['n'])
    for m in [m for m in ORDER if ('e2', ds, 'neither_ref', m) in M]:
        r = paired(peruser('e2', ds, 'both_ref', m), peruser('e2', ds, 'neither_ref', m))
        L.append(f"- {m} (no graph): neither {cell('e2', ds, 'neither_ref', m)} → both {cell('e2', ds, 'both_ref', m)}"
                 + (f"; Δ {r['diff']:+.4f} [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}], Wilcoxon p {r['p']:.3g}" if r else ''))
    L.append('')
# ---------------------------------------------------------------- T-E3
L += ['## T-E3 filter-then-split (FTS) vs split-then-filter (STF), GT warm NDCG@20; time-aware baselines', '',
      'ML-1M has no STF version: our pipeline applies no k-core filter to ML-1M (used as released; GroupLens kept only users with '
      '≥ 20 ratings before release and no unfiltered ML-1M exists), so FTS and STF coincide for it.', '']
r3sum = {}
if os.path.exists(R3 + '/summary.csv'):
    for r in csv.DictReader(open(R3 + '/summary.csv')):
        if r['protocol'] == 'global_temporal': r3sum[(r['dataset'], r['model'])] = float(r['warm_NDCG@20_mean'])
st3 = json.load(open(f'{E}/data/e3/stats_e3.json')) if os.path.exists(f'{E}/data/e3/stats_e3.json') else {}
for base in sorted({d.replace('-stf', '') for d in dss if d.endswith('-stf')}):
    stf = base + '-stf'; ms = [m for m in ORDER if ('gt', base, 'gt', m) in M or ('gt', stf, 'gt', m) in M or (base, m) in r3sum]
    fts = {m: (mean('gt', base, 'gt', m) if ('gt', base, 'gt', m) in M else r3sum.get((base, m), np.nan)) for m in ms}
    sv = {m: mean('gt', stf, 'gt', m) for m in ms}
    L += [f'### {base}', '', '| model | FTS GT | STF GT (mean±std (n)) | STF 95% CI over seeds |', '|---|---|---|---|']
    for m in ms:
        r = M.get(('gt', stf, 'gt', m))
        L.append(f"| {m} | {'–' if np.isnan(fts[m]) else f'{fts[m]:.4f}'}{'' if ('gt', base, 'gt', m) in M or np.isnan(fts[m]) else ' (r3)'} | {cell('gt', stf, 'gt', m)} | "
                 + (f"[{r['NDCG@20_ci95_lo']:.4f}, {r['NDCG@20_ci95_hi']:.4f}]" if r and r['n_seeds'] > 1 else '–') + ' |')
    t_, n_ = tau(fts, sv)
    valid = {m: v for m, v in sv.items() if not np.isnan(v)}
    if valid:
        best = max(valid, key=valid.get); pb = peruser('gt', stf, 'gt', best); tests = []
        for m in valid:
            if m == best: continue
            r = paired(pb, peruser('gt', stf, 'gt', m))
            if r: tests.append((m, r))
        L += ['', f'Kendall τ(FTS, STF) = {t_:.3f} over {n_} models. STF best = {best}; paired per-user tests best vs each (Holm):', '',
              '| vs | diff | 95% CI | p (Holm) | r | n users |', '|---|---|---|---|---|---|']
        if tests:
            for (m, r), pa_ in zip(tests, holm([x[1]['p'] for x in tests])):
                L.append(f"| {m} | {r['diff']:+.4f} | [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}] | {pa_:.3g} | {r['r']:+.3f} | {r['n']} |")
                J['e3'].setdefault(base, {}).setdefault('tests', {})[m] = dict(diff=r['diff'], ci=r['ci'], p_holm=float(pa_), r=r['r'], n=r['n'])
        J['e3'].setdefault(base, {}).update(tau=t_, n=n_, best=best, stf={m: dict(mean=v, n=M[('gt', stf, 'gt', m)]['n_seeds']) for m, v in valid.items()})
    s = st3.get(stf, {})
    L += ['', f'STF test rows warm: {s.get("test_warm_rows", "?")} of {s.get("test_rows", "?")} (user-cold {s.get("test_usercold_rows_pct", "?")}%, item-cold {s.get("test_itemcold_rows_pct", "?")}%).', '']
# ---------------------------------------------------------------- T-GRID
L += ['## T-GRID closed-form grids extended in R5 (validation NDCG@20 picks the value; R5 grid ⊇ R4 grid)', '',
      '| exp | dataset | setting | model | R4 param | R5 param | R5 still at edge | NDCG@20 R4 | NDCG@20 R5 |', '|---|---|---|---|---|---|---|---|---|']
for k, old in sorted(replaced.items()):
    r = M.get(k)
    if not r or k[2] in ('gt_full',): continue
    L.append(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {old['param']} | {r['param']} | {r['at_grid_edge'] or '–'} | {old['met'].get('NDCG@20', float('nan')):.4f} | {r['NDCG@20_mean']:.4f} |")
    J['grid'].setdefault(k[1], {})[f'{k[0]}/{k[2]}/{k[3]}'] = dict(param_r4=old['param'], param_r5=r['param'], edge_r5=r['at_grid_edge'],
                                                                   ndcg_r4=old['met'].get('NDCG@20'), ndcg_r5=r['NDCG@20_mean'])
edge_left = [r for r in rows if r['grid'] == 'r5ext' and 'True' in str(r['at_grid_edge']) and r['setting'] not in ('gt_full',)]
L += ['', 'R5 cells still at a grid endpoint: ' + (', '.join(f"{r['exp']}/{r['dataset']}/{r['setting']}/{r['model']}={r['param']}" for r in edge_left) or 'none'),
      '(EASE-TD / ItemKNN-TD half-life 1e9 = no time decay, i.e. the plain model; that endpoint cannot be extended further.)', '']
# ---------------------------------------------------------------- T-RF / T-AL (same as R4, with merged records)
L += ['## T-RF GT without vs with refit on train+val', '']
for ds in dss:
    ms = [m for m in ORDER if ('gtrefit', ds, 'norefit', m) in M]
    if not ms: continue
    a = {m: mean('gtrefit', ds, 'norefit', m) for m in ms}; b = {m: mean('gtrefit', ds, 'refit', m) for m in ms}
    L += [f'### {ds}', '', '| model | no refit | refit (same test rows) | refit (rows warm wrt train+val) |', '|---|---|---|---|']
    for m in ms: L.append(f"| {m} | {cell('gtrefit', ds, 'norefit', m)} | {cell('gtrefit', ds, 'refit', m)} | {cell('gtrefit', ds, 'refit_warmtv', m)} |")
    t1, n1 = tau(a, b); J['refit'][ds] = dict(tau=t1, n=n1); L += ['', f'Kendall τ(no refit, refit) = {t1:.3f} over {n1} models', '']
L += ['## T-AL population-aligned random / LOO', '']
for ds in dss:
    ms = [m for m in ORDER if ('aligned', ds, 'random', m) in M or ('aligned', ds, 'loo', m) in M]
    if not ms: continue
    gt = {m: mean('gtrefit', ds, 'norefit', m) for m in ms}
    if all(np.isnan(v) for v in gt.values()): gt = {m: mean('gt', ds, 'gt', m) for m in ms}
    L += [f'### {ds}', '', '| model | random (r3 pop.) | random (aligned) | loo (r3 pop.) | loo (aligned) | GT |', '|---|---|---|---|---|---|']
    for m in ms: L.append(f"| {m} | {cell('aligned', ds, 'random', m)} | {cell('aligned', ds, 'random_aligned', m)} | {cell('aligned', ds, 'loo', m)} | {cell('aligned', ds, 'loo_aligned', m)} | {gt[m]:.4f} |")
    for pr in ('random', 'loo'):
        t0, n0 = tau({m: mean('aligned', ds, pr, m) for m in ms}, gt); t1, n1 = tau({m: mean('aligned', ds, pr + '_aligned', m) for m in ms}, gt)
        L.append(f'- τ({pr}, GT): r3 population {t0:.3f}, aligned {t1:.3f} (n models {n1})'); J['aligned'].setdefault(ds, {})[pr] = dict(tau_r3pop=t0, tau_aligned=t1, n=n1)
    L.append('')
# ---------------------------------------------------------------- seeds
L += ['## Seeds per trained-model cell (R4 + R5)', '', '| exp | dataset | setting | model | n | seeds R4 | seeds R5 |', '|---|---|---|---|---|---|---|']
for r in rows:
    if r['model'] in TRAINED and r['setting'] not in ('gt_full',) and r['exp'] in ('e1', 'e2', 'gt'):
        L.append(f"| {r['exp']} | {r['dataset']} | {r['setting']} | {r['model']} | {r['n_seeds']} | {r['seeds_r4'] or '–'} | {r['seeds_r5'] or '–'} |")
open(R5 + '/stats_r45.md', 'w').write('\n'.join(L) + '\n')
json.dump(J, open(R5 + '/stats_r45.json', 'w'), indent=1, default=float)
print('\n'.join(L[:60])); print('rows', len(rows), 'records', len(ALL), 'r4', len(r4), 'r5', len(r5), 'replaced', len(replaced), 'dup', len(dup), 'mismatch', len(mism))
