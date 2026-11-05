"""Round-5 driver = driver_v5.py (round 4, unchanged code paths) + R5 modes e1plan, stfx, alignedcpu_ext, smoke5.
Round-4 driver (runs inside one Kaggle kernel, or locally with KW=<dir>). usage: python driver_v5.py DATASET DATA_DIR MODE
Each unit is one subprocess of run_v5.py (one experiment x one seed); exit code and wall time are logged as '## <tag> exit <rc>'.
MODE
  e1        GPU: BPR-MF, LightGCN (+ SimGCL if E1_MODELS says so) on conditions E1_CONDS, seeds SEEDS
  e1cpu     CPU: MostPop, ItemKNN, EASE on e1 conditions a,b,c,f (deterministic -> seed 1) + e2 references (neither/both)
  e2        GPU: LightGCN leak-path variants over E2_LAYERS, seeds SEEDS
  refit     GPU: gtrefit for BPR-MF, LightGCN, seeds SEEDS
  aligned   GPU: aligned re-evaluation of BPR-MF, LightGCN with the round-2 lr, random + loo, seeds SEEDS
  alignedcpu CPU: aligned re-evaluation of MostPop/ItemKNN/EASE with the ROUND-3 chosen k / lambda (R3_BASE env, injected by
            make_r4.py), random seeds SEEDS, loo seed 1
  gtcpu     CPU: gt with MostPop, ItemKNN, EASE, RecentPop, EASE-TD, ItemKNN-TD (seed 1) + gtrefit closed form (FTS data only)
  stf       GPU: gt on a split-then-filter file: seed 1 lr grid {1e-3,5e-3} for BPR-MF and LightGCN chosen by best validation
            NDCG@20, then seeds 2.. with the chosen lr
env: SEEDS, DEADLINE_MIN (no new unit after this), MAX_MIN (per-run cap, default 10), E1_CONDS, E1_MODELS, E2_LAYERS,
     E2_LOSS_LAYERS, R3_BASE (json {"random/EASE": "1000", ...}), R3_LR_NEW (json {"global_temporal/SimGCL": "1e-3"})"""
import os, sys, json, subprocess, time
W = os.environ.get('KW', '/kaggle/working'); OUT = W + '/out'; os.makedirs(OUT, exist_ok=True)
RUN = os.environ.get('RUN_V5', W + '/run_v5.py')
DATASET, DATA_DIR, MODE = sys.argv[1], os.path.abspath(sys.argv[2]), sys.argv[3]
FAILED = []
SEEDS = [int(s) for s in os.environ.get('SEEDS', '1,2,3,4,5').split(',')]
DEADLINE = float(os.environ.get('DEADLINE_MIN', '1e9')) * 60; T0 = time.time()
BASE_DS = DATASET.replace('-stf', '')
LR_R2 = {  # round 2, chosen by validation NDCG@20 on seed 1 (identical to driver_v4.py)
 'ml-1m':     {'random/BPR-MF': '1e-3', 'random/LightGCN': '5e-3', 'loo/BPR-MF': '1e-3', 'loo/LightGCN': '5e-3', 'global_temporal/BPR-MF': '1e-3', 'global_temporal/LightGCN': '5e-3'},
 'gowalla':   {'random/BPR-MF': '1e-3', 'random/LightGCN': '5e-3', 'loo/BPR-MF': '1e-3', 'loo/LightGCN': '5e-3', 'global_temporal/BPR-MF': '1e-3', 'global_temporal/LightGCN': '1e-3'},
 'amz-music': {'random/BPR-MF': '1e-3', 'random/LightGCN': '1e-3', 'loo/BPR-MF': '1e-3', 'loo/LightGCN': '1e-3', 'global_temporal/BPR-MF': '1e-3', 'global_temporal/LightGCN': '1e-3'},
 'amz-games': {'random/BPR-MF': '5e-3', 'random/LightGCN': '5e-3', 'loo/BPR-MF': '5e-3', 'loo/LightGCN': '5e-3', 'global_temporal/BPR-MF': '5e-3', 'global_temporal/LightGCN': '1e-3'},
}
LRg = LR_R2.get(BASE_DS, {})
R3_LR_NEW = json.loads(os.environ.get('R3_LR_NEW', '{}')); R3_BASE = json.loads(os.environ.get('R3_BASE', '{}'))
def chosen(r, key):   # pass the validation-chosen k / lambda of an earlier unit on as one-value grids (saves CPU time)
    ex = {}
    try:
        ex['KNN_GRID'] = str(int(r[key]['ItemKNN']['param'])); ex['EASE_GRID'] = str(r[key]['EASE']['param'])
    except Exception as e: print('## no earlier choice to pass on, grids re-run:', repr(e), flush=True)
    return ex
def late(): return time.time() - T0 > DEADLINE
def run(exp, seed, models, tag, conds='', extra=None):
    if late(): print('## DEADLINE skip', tag, flush=True); return None
    env = dict(os.environ, DATASET=DATASET, DATA_DIR=DATA_DIR, OUT_DIR=OUT, TAG=tag, MAX_MIN=os.environ.get('MAX_MIN', '10'),
               BS='8192', LR_BPR=LRg.get('global_temporal/BPR-MF', '1e-3'), LR_LGCN=LRg.get('global_temporal/LightGCN', '1e-3'),
               LR_SIMGCL=R3_LR_NEW.get('global_temporal/SimGCL', '1e-3'))
    env.update({k: str(v) for k, v in (extra or {}).items()})
    t = time.time(); r = subprocess.run([sys.executable, RUN, exp, str(seed), models, conds], env=env, cwd=W)
    print(f'## {tag} exit {r.returncode} {round(time.time() - t, 1)}s elapsed {round((time.time() - T0) / 60, 1)}min', flush=True)
    if r.returncode: FAILED.append(tag)
    try: return json.load(open(f'{OUT}/res_{tag}.json'))
    except Exception: return None
if MODE == 'e1':
    for s in SEEDS:
        run('e1', s, os.environ.get('E1_MODELS', 'BPR-MF,LightGCN'), f'e1_{DATASET}_s{s}', os.environ.get('E1_CONDS', 'a,b,c'))
elif MODE == 'e1cpu':
    r = run('e1', 1, 'MostPop,ItemKNN,EASE', f'e1cpu_{DATASET}_s1', 'a,b,c,f')
    run('e2', 1, 'MostPop,ItemKNN,EASE', f'e2cpu_{DATASET}_s1', 'neither,both', chosen(r, 'cond_a'))   # same k / lambda as e1-a
elif MODE == 'e2':
    for s in SEEDS:
        run('e2', s, 'LightGCN', f'e2_{DATASET}_s{s}', os.environ.get('E2_VARIANTS', 'neither,adj,both,loss'))
elif MODE == 'refit':
    for s in SEEDS:
        run('gtrefit', s, os.environ.get('REFIT_MODELS', 'BPR-MF,LightGCN'), f'refit_{DATASET}_s{s}')
elif MODE == 'aligned':
    for s in SEEDS:
        for proto in ('random', 'loo'):
            run('aligned', s, 'BPR-MF,LightGCN', f'aligned_{DATASET}_{proto}_s{s}', proto,
                {'LR_BPR': LRg[f'{proto}/BPR-MF'], 'LR_LGCN': LRg[f'{proto}/LightGCN']})
elif MODE == 'alignedcpu':
    for proto in ('random', 'loo'):
        k, lam = R3_BASE.get(f'{proto}/ItemKNN'), R3_BASE.get(f'{proto}/EASE')
        assert k and lam, f'round-3 choice missing for {proto}: rebuild with make_r4.py after round 3 is fetched'
        for s in (SEEDS if proto == 'random' else [1]):
            run('aligned', s, 'MostPop,ItemKNN,EASE', f'alignedcpu_{DATASET}_{proto}_s{s}', proto, {'KNN_GRID': k, 'EASE_GRID': lam})
elif MODE == 'gtcpu':
    r = run('gt', 1, 'MostPop,ItemKNN,EASE,RecentPop,EASE-TD,ItemKNN-TD', f'gtcpu_{DATASET}_s1')
    if not DATASET.endswith('-stf'): run('gtrefit', 1, 'MostPop,ItemKNN,EASE', f'refitcpu_{DATASET}_s1', '', chosen(r, 'gt'))
elif MODE == 'stf':
    choice = {}
    if 1 in SEEDS:
        best = {'BPR-MF': (-1, None), 'LightGCN': (-1, None)}
        for lr in ('1e-3', '5e-3'):
            r = run('gt', 1, 'BPR-MF,LightGCN', f'stf_{DATASET}_s1_lr{lr}', '', {'LR_BPR': lr, 'LR_LGCN': lr})
            for m in best:
                if r and m in r.get('gt', {}):
                    v = max((x[2] for x in r['gt'][m]['val_log'] if isinstance(x[0], int)), default=-1); best[m] = max(best[m], (v, lr))
        choice = {m: v[1] for m, v in best.items()}; print('## CHOICE', best, flush=True)
        json.dump(best, open(f'{OUT}/lr_choice_stf_{DATASET}.json', 'w'), indent=1)
    else:
        choice = json.loads(os.environ['STF_LR'])
    for s in SEEDS:
        if s == 1: continue
        for m, lr in choice.items():
            if lr: run('gt', s, m, f'stf_{DATASET}_s{s}_{m}_lr{lr}', '', {'LR_BPR': lr, 'LR_LGCN': lr})
elif MODE == 'smoke':   # used by make_r4.py smoke kernel / local smoke; tiny everything
    ex = {'EPOCHS': '10', 'PATIENCE': '2', 'MAX_MIN': '0.5', 'SIMGCL_EPOCHS': '2', 'E2_LAYERS': '0,2', 'E2_LOSS_LAYERS': '2',
          'SUBSAMPLE_USERS': os.environ.get('SMOKE_USERS', '1500'), 'EASE_GRID': '500,5000', 'KNN_GRID': '50,200', 'HL_GRID': '0.1,1e9', 'RECENT_GRID': '0.1,1'}
    run('e1', 1, 'MostPop,EASE,BPR-MF,LightGCN,SimGCL', f'smoke_e1_{DATASET}', 'a,b,c,f', ex)
    run('e2', 1, 'EASE,LightGCN', f'smoke_e2_{DATASET}', '', ex)
    run('aligned', 1, 'EASE,BPR-MF,LightGCN', f'smoke_aligned_{DATASET}', 'random,loo', ex)
    run('gtrefit', 1, 'EASE,BPR-MF,LightGCN', f'smoke_refit_{DATASET}', '', ex)
    run('gt', 1, 'MostPop,ItemKNN,EASE,RecentPop,EASE-TD,ItemKNN-TD,BPR-MF,LightGCN', f'smoke_gt_{DATASET}', '', ex)
    # one full-size GT LightGCN run (round-2 lr, 10-min cap) to re-measure T4 speed against the round-3 logs
    run('gt', 1, 'LightGCN', f'smoke_gtfull_{DATASET}', '', {'MAX_MIN': '3'})
elif MODE == 'e1plan':   # R5: explicit (seed, conditions) list, E1_PLAN="4:a,b,c,f;5:a,b,c,f;1:f;2:f;3:f"
    for item in os.environ['E1_PLAN'].split(';'):
        s, conds = item.split(':'); s = int(s)
        run('e1', s, os.environ.get('E1_MODELS', 'BPR-MF,LightGCN'), f'e1_{DATASET}_s{s}_{conds.replace(",", "")}', conds)
elif MODE == 'stfx':   # R5: STF extra seeds for BPR-MF/LightGCN with the R4 lr choice (STF_LR json) + SimGCL (lr grid on seed 1)
    choice = json.loads(os.environ['STF_LR'])
    for s in [int(x) for x in os.environ.get('STF_SEEDS', '3,4,5').split(',') if x]:
        for m, lr in choice.items():
            run('gt', s, m, f'stf_{DATASET}_s{s}_{m}_lr{lr}', '', {'LR_BPR': lr, 'LR_LGCN': lr})
    sg = [int(x) for x in os.environ.get('SIMGCL_SEEDS', '').split(',') if x]
    if sg:
        best = (-1, None)
        for lr in ('1e-3', '5e-3'):   # same lr grid and selection rule as round 3 (driver_v4 'new') and the R4 stf mode
            r = run('gt', 1, 'SimGCL', f'stf_{DATASET}_s1_SimGCL_lr{lr}', '', {'LR_SIMGCL': lr})
            if r and 'SimGCL' in r.get('gt', {}):
                v = max((x[2] for x in r['gt']['SimGCL']['val_log'] if isinstance(x[0], int)), default=-1); best = max(best, (v, lr))
        print('## CHOICE SimGCL', best, flush=True); json.dump({'SimGCL': best}, open(f'{OUT}/lr_choice_stf_simgcl_{DATASET}.json', 'w'), indent=1)
        for s in sg:
            if s != 1 and best[1]: run('gt', s, 'SimGCL', f'stf_{DATASET}_s{s}_SimGCL_lr{best[1]}', '', {'LR_SIMGCL': best[1]})
elif MODE == 'alignedcpu_ext':   # R5: random/LOO ItemKNN with the extended k grid (KNN_GRID env), tuned on seed 1 per protocol as
    # round 3 did, then the chosen k for the other random seeds; EASE/MostPop are not re-run (their round-3 values are interior)
    for proto in ('random', 'loo'):
        r = run('aligned', 1, 'ItemKNN', f'alignedx_{DATASET}_{proto}_s1', proto)
        try: k = str(int(r[proto]['ItemKNN']['param']))
        except Exception as e: print('## no chosen k', proto, repr(e), flush=True); continue
        if proto == 'random':
            for s in SEEDS:
                if s != 1: run('aligned', s, 'ItemKNN', f'alignedx_{DATASET}_{proto}_s{s}', proto, {'KNN_GRID': k})
elif MODE == 'smoke5':   # R5 smoke: tiny versions of every R5 path (subsampled users, few epochs, short grids)
    ex = {'EPOCHS': '10', 'PATIENCE': '2', 'MAX_MIN': '0.5', 'SIMGCL_EPOCHS': '2', 'SUBSAMPLE_USERS': os.environ.get('SMOKE_USERS', '1500'),
          'EASE_GRID': '500,5000', 'KNN_GRID': '20,200', 'HL_GRID': '0.01,1e9', 'RECENT_GRID': '0.005,1'}
    run('e1', 4, 'BPR-MF,LightGCN,SimGCL', f'smoke5_e1_{DATASET}_s4_abcf', 'a,b,c,f', ex)
    run('e1', 1, 'BPR-MF,LightGCN', f'smoke5_e1_{DATASET}_s1_f', 'f', ex)
    run('e1', 1, 'MostPop,ItemKNN,EASE', f'smoke5_e1cpu_{DATASET}', 'a,b,c,f', ex)
    run('e2', 2, 'LightGCN', f'smoke5_e2_{DATASET}', '', dict(ex, E2_LAYERS='3', E2_LOSS_LAYERS='3'))
    run('gt', 3, 'MostPop,ItemKNN,EASE,RecentPop,EASE-TD,ItemKNN-TD,BPR-MF,LightGCN,SimGCL', f'smoke5_gt_{DATASET}', '', ex)
    run('aligned', 1, 'ItemKNN', f'smoke5_alignedx_{DATASET}', 'random,loo', ex)
    if os.environ.get('SMOKE_FULL', '1') == '1':   # one full-size GT LightGCN unit to re-measure T4 speed (3-min cap)
        run('gt', 1, 'LightGCN', f'smoke5_gtfull_{DATASET}', '', {'MAX_MIN': '3'})
print('ALL DONE', DATASET, MODE, round(time.time() - T0), 's', 'FAILED_UNITS', FAILED, flush=True)
sys.exit(1 if FAILED else 0)
