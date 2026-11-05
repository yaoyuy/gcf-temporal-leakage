"""Round-3 driver (runs on Kaggle). usage: python driver_v4.py DATASET DATA_DIR MODE
MODE basetune: MostPop + ItemKNN k grid {50,100,200,500} + EASE lambda grid (per dataset), seed 1 by validation NDCG@20;
             random split seeds 2-5 rerun with the chosen value (loo/global_temporal deterministic -> seed 1 reused).
MODE base  : MostPop/ItemKNN/EASE. random: seeds 1-5; loo & global_temporal: seed 1 only (splits and models are deterministic,
             r2 std = 0 for these; summariser replicates seed 1 for seeds 2-5 and marks it).
MODE rerun : BPR-MF & LightGCN re-run with the lr chosen in round 2 (lr_choice_<ds>.json, embedded below), seed-major order.
MODE new   : SimGCL & UltraGCN. seed 1: small lr grid per protocol, pick by best validation NDCG@20 (as driver_v3);
             then seeds 2.. with the chosen lr, seed-major order.
env: SEEDS (default 1,2,3,4,5), DEADLINE_MIN (stop launching new runs after this many minutes), MAX_MIN (per-run cap, 10)."""
import os, sys, json, subprocess, time
W = os.environ.get("KW", "/kaggle/working"); OUT = W + "/out"; os.makedirs(OUT, exist_ok=True)
DATASET, DATA_DIR, MODE = sys.argv[1], sys.argv[2], sys.argv[3]
SEEDS = [int(s) for s in os.environ.get('SEEDS', '1,2,3,4,5').split(',')]
DEADLINE = float(os.environ.get('DEADLINE_MIN', '1e9')) * 60
PROTOS = os.environ.get('PROTOS', 'random,loo,global_temporal').split(',')
LR_R2 = {  # out/r2/**/lr_choice_<ds>.json (round 2, chosen by validation NDCG@20 on seed 1)
 'ml-1m':     {'random/BPR-MF': '1e-3', 'random/LightGCN': '5e-3', 'loo/BPR-MF': '1e-3', 'loo/LightGCN': '5e-3', 'global_temporal/BPR-MF': '1e-3', 'global_temporal/LightGCN': '5e-3'},
 'gowalla':   {'random/BPR-MF': '1e-3', 'random/LightGCN': '5e-3', 'loo/BPR-MF': '1e-3', 'loo/LightGCN': '5e-3', 'global_temporal/BPR-MF': '1e-3', 'global_temporal/LightGCN': '1e-3'},
 'amz-music': {'random/BPR-MF': '1e-3', 'random/LightGCN': '1e-3', 'loo/BPR-MF': '1e-3', 'loo/LightGCN': '1e-3', 'global_temporal/BPR-MF': '1e-3', 'global_temporal/LightGCN': '1e-3'},
 'amz-games': {'random/BPR-MF': '5e-3', 'random/LightGCN': '5e-3', 'loo/BPR-MF': '5e-3', 'loo/LightGCN': '5e-3', 'global_temporal/BPR-MF': '5e-3', 'global_temporal/LightGCN': '1e-3'},
}
GRID = {'SimGCL': ['1e-3', '5e-3'],      # SELFRec default 1e-3
        'UltraGCN': ['1e-4', '1e-3']}    # official configs use 1e-3 (ML-1M, Amazon-Books, Yelp) and 1e-4 (Gowalla)
NEW_MODELS = os.environ.get('NEW_MODELS', 'SimGCL,UltraGCN').split(',')
T0 = time.time()
def late(): return time.time() - T0 > DEADLINE
def run(seed, proto, models, lr, tag, extra=None):
    env = dict(os.environ, DATASET=DATASET, DATA_DIR=DATA_DIR, OUT_DIR=OUT, LR=lr, BS='8192',
               MAX_MIN=os.environ.get('MAX_MIN', '10'), TAG=tag, **(extra or {}))
    t = time.time(); r = subprocess.run([sys.executable, W + '/run_v4.py', str(seed), '400', proto, models], env=env, cwd=W)
    print(f'## {tag} exit {r.returncode} {round(time.time()-t,1)}s elapsed {round((time.time()-T0)/60,1)}min', flush=True)
    try: return json.load(open(f'{OUT}/res_{tag}.json'))
    except Exception: return None
KNN_GRID = ['50', '100', '200', '500']
LAM_GRID = ['100', '200', '500', '1000', '2000', '5000'] if DATASET == 'ml-1m' else ['10', '25', '50', '100', '200', '500', '1000']
# EASE lambda scaled to dataset density: ML-1M is dense (Steck 2019 used 200-500 on ML-20M); the Amazon/Gowalla sets are
# far sparser (item degrees ~10^1-10^2), so the grid is shifted down. Both grids include the round-2 value 500.
if MODE == 'basetune':
    choice = {}
    for proto in PROTOS:   # seed 1: grid by validation NDCG@20
        if late(): print('## DEADLINE stop', flush=True); break
        run(1, proto, 'MostPop', '1e-3', f'{DATASET}_{proto}_s1_MostPop')
        for m, grid, var, short in (('ItemKNN', KNN_GRID, 'KNN_K', 'k'), ('EASE', LAM_GRID, 'EASE_LAM', 'lam')):
            best = (-1, None)
            for v in grid:
                r = run(1, proto, m, '1e-3', f'{DATASET}_{proto}_s1_{m}_{short}{v}', {var: v})
                if r and m in r.get(proto, {}): best = max(best, (r[proto][m]['val_NDCG@20'], v))
            choice[f'{proto}/{m}'] = best; print('## CHOICE', proto, m, best, flush=True)
            json.dump(choice, open(f'{OUT}/base_choice_{DATASET}.json', 'w'), indent=1)
    if 'random' in PROTOS:   # loo / global_temporal are deterministic: seed-1 results are reused for seeds 2-5
        for s in SEEDS:
            if s == 1: continue
            run(s, 'random', 'MostPop', '1e-3', f'{DATASET}_random_s{s}_MostPop')
            for m, var, short in (('ItemKNN', 'KNN_K', 'k'), ('EASE', 'EASE_LAM', 'lam')):
                v = choice[f'random/{m}'][1]
                if v: run(s, 'random', m, '1e-3', f'{DATASET}_random_s{s}_{m}_{short}{v}', {var: v})
elif MODE == 'base':
    for proto in PROTOS:
        for s in (SEEDS if proto == 'random' else [1]):
            if late(): print('## DEADLINE stop', flush=True); break
            run(s, proto, 'MostPop,ItemKNN,EASE', '1e-3', f'{DATASET}_{proto}_s{s}_base')
elif MODE == 'rerun':
    for s in SEEDS:
        for proto in PROTOS:
            for m in ['BPR-MF', 'LightGCN']:
                if late(): print('## DEADLINE stop before', s, proto, m, flush=True); continue
                lr = LR_R2[DATASET][f'{proto}/{m}']
                run(s, proto, m, lr, f'{DATASET}_{proto}_s{s}_{m}_lr{lr}')
elif MODE == 'new':
    choice = {}
    if 1 in SEEDS:
        for proto in PROTOS:
            for m in NEW_MODELS:
                best = (-1, None)
                for lr in GRID[m]:
                    if late(): print('## DEADLINE stop in grid', proto, m, lr, flush=True); continue
                    r = run(1, proto, m, lr, f'{DATASET}_{proto}_s1_{m}_lr{lr}')
                    if r and m in r.get(proto, {}): v = max(x[2] for x in r[proto][m]['val_log']); best = max(best, (v, lr))
                choice[f'{proto}/{m}'] = best
                print('## CHOICE', proto, m, best, flush=True)
                json.dump(choice, open(f'{OUT}/lr_choice_new_{DATASET}.json', 'w'), indent=1)
    else:
        choice = json.loads(os.environ['LR_CHOICE_NEW'])
    for s in SEEDS:
        if s == 1: continue
        for proto in PROTOS:
            for m in NEW_MODELS:
                lr = choice.get(f'{proto}/{m}', (None, None))[1]
                if lr is None: continue
                if late(): print('## DEADLINE stop before', s, proto, m, flush=True); continue
                run(s, proto, m, lr, f'{DATASET}_{proto}_s{s}_{m}_lr{lr}')
elif MODE == 'smoke':
    run(1, 'global_temporal', 'MostPop,ItemKNN,EASE,BPR-MF,LightGCN,SimGCL,UltraGCN', '1e-3', f'{DATASET}_smoke',
        {'MAX_MIN': '0.5', 'SIMGCL_EPOCHS': '2'})
print('ALL DONE', DATASET, MODE, round(time.time() - T0), 's', flush=True)
