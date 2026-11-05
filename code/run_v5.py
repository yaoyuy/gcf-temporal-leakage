"""run_v5.py  (round 4 supplementary experiments; the models are the round-3 code of run_v4.py, unchanged in their math).
One process = one experiment x one seed x a list of conditions x a list of models.

usage: python run_v5.py EXP SEED MODELS [CONDS]
  EXP    e1 | e2 | aligned | gtrefit | gt
  MODELS comma list from MostPop, ItemKNN, EASE, BPR-MF, LightGCN, SimGCL, RecentPop, EASE-TD, ItemKNN-TD
  CONDS  e1: a,b,c,f (default a,b,c)   e2: neither,adj,both,loss (default all)   aligned: random,loo   others: ignored
env: DATASET, DATA_DIR, OUT_DIR, TAG, EPOCHS(400) PATIENCE(5) MAX_MIN(0=none) BS(8192)
     LR_BPR, LR_LGCN, LR_SIMGCL (default 1e-3)  LAYERS (LightGCN K, default 3)  E2_LAYERS (default 0,1,2,3,4)
     E2_LOSS_LAYERS (K values at which the loss-only variant is run; default all)
     EASE_GRID, KNN_GRID, RECENT_GRID, HL_GRID (comma lists; validation NDCG@20 picks the value; a one-value list fixes it)
     E1_SPLIT_SEED (default 12345; fixes the evaluation/donor user partition and the held-out items, independent of SEED)
     SUBSAMPLE_USERS / SUBSAMPLE_ITEMS (smoke tests only: keep N random users / N most popular items before splitting)

EXPERIMENTS (all evaluation = full ranking, Recall/NDCG@10/20, same metric code as round 3)
 e1 (controlled leakage, global temporal test set fixed). GT split 80/10/10 by time. Index space = users/items of GT train.
    Warm GT test users are split once (E1_SPLIT_SEED) into evaluation users E (50%) and donor users D (50%).
    Each E user's warm test items are split at random into held-out H (ceil 50%) and own-future F (rest).
    Leak sets: L_D = all warm test-period rows of D users (cross-user leakage); F (same-user leakage, excludes H).
    GT train is shuffled once into a common base B (|GT train| - |L_D| - |F| rows) and a reserve R.
      a = B + R                       (= GT train)                       no leakage
      b = B + L_D + R[:|F|]           cross-user leakage                  same size N
      c = B + L_D + F                 cross-user + same-user leakage     same size N
      f = B + F + R[:|L_D|]           same-user leakage only              same size N
    Test = H of E users in every condition. Ranking mask = condition train + GT val + F in every condition (F is masked
    even when not trained on, so the candidate set is identical across conditions). Validation (early stopping and
    closed-form grids) = warm GT val rows; closed-form values are chosen on condition a and reused for b/c/f.
 e2 (graph-propagation leakage). Same E/D/H/F construction; base = full GT train, L = L_D + F (the leak set of e1-c).
    LightGCN variants: neither (adj=train, BPR positives=train), adj (adj=train+L, positives=train),
    both (adj=train+L, positives=train+L), loss (adj=train, positives=train+L); swept over K in E2_LAYERS
    (K=0 has no propagation, so only neither/loss are run there). EASE/MostPop/ItemKNN get neither/both as references.
 aligned (population-aligned re-evaluation). random / loo split exactly as round 3 (same seed -> same split). Each model
    is scored once and evaluated twice: 'warm' (round-3 metric) and 'aligned' = test rows whose user is a warm GT test
    user and whose item is in GT train (users with no such row dropped).
 gtrefit. Standard GT split. Per model: 'norefit' = round-3 protocol (fit on train, early stop on val, mask train+val);
    'refit' = refit on train+val (closed form: same chosen value; BPR/LightGCN/SimGCL: the best epoch of the norefit run,
    no validation), index space = train+val. Both evaluated on the round-3 warm test rows ('warm') and on test rows warm
    w.r.t. train+val ('warmtv'). Per-user arrays carry raw user ids so the two runs pair by user.
 gt. Standard GT split (or the pre-split STF files of E3: DATASET ends with -stf, column s) + any model incl. the
    time-aware baselines: RecentPop (popularity over the last w fraction of train rows by time),
    EASE-TD / ItemKNN-TD (rows weighted 0.5 ** ((t_max - t) / (h * train time span)), h in HL_GRID; EASE-TD uses the
    lambda chosen for EASE in the same process). Reports warm / full / wu populations like round 3.
"""
import sys, os, time, json, math, numpy as np, pandas as pd, scipy.sparse as sp, torch
import torch.nn.functional as F
import warnings; warnings.filterwarnings('ignore', category=UserWarning)

EXP = sys.argv[1]; SEED = int(sys.argv[2]); MODELS = sys.argv[3].split(',')
CONDS = sys.argv[4].split(',') if len(sys.argv) > 4 and sys.argv[4] else None
rng = np.random.default_rng(SEED); torch.manual_seed(SEED)
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATASET = os.environ.get('DATASET', 'ml-1m'); DATA = os.environ.get('DATA_DIR', 'data/ml-1m')
OUT = os.environ.get('OUT_DIR', 'out_v5'); os.makedirs(OUT, exist_ok=True)
TAG = os.environ.get('TAG', f'{EXP}_{DATASET}_s{SEED}')
EPOCHS = int(os.environ.get('EPOCHS', '400')); PATIENCE = int(os.environ.get('PATIENCE', '5'))
MAX_MIN = float(os.environ.get('MAX_MIN', '0')); BS = int(os.environ.get('BS', '8192'))
LAYERS = int(os.environ.get('LAYERS', '3'))
LR = {'BPR-MF': float(os.environ.get('LR_BPR', '1e-3')), 'LightGCN': float(os.environ.get('LR_LGCN', '1e-3')),
      'SimGCL': float(os.environ.get('LR_SIMGCL', '1e-3'))}
def grid(name, default): return [float(x) for x in os.environ.get(name, default).split(',')]
EASE_GRID = grid('EASE_GRID', '500,1000,2000,5000,10000,20000,50000' if DATASET == 'ml-1m' else '10,25,50,100,200,500,1000,2000,5000')
KNN_GRID = [int(x) for x in grid('KNN_GRID', '50,100,200,500,1000')]
RECENT_GRID = grid('RECENT_GRID', '0.01,0.02,0.05,0.1,0.25,0.5,1')
HL_GRID = grid('HL_GRID', '0.02,0.05,0.1,0.25,0.5,1,1e9')
print('device', DEV, 'exp', EXP, 'dataset', DATASET, 'seed', SEED, 'torch', torch.__version__, flush=True)
Ks = [10, 20]; T00 = time.time()

# ---------------- data ----------------
def load():
    if DATASET == 'ml-1m':
        d = pd.read_csv(f'{DATA}/ratings.dat', sep='::', engine='python', names=['u', 'i', 'r', 't']).drop(columns='r')
    else:
        d = pd.read_csv(f'{DATA}/{DATASET}.csv')
        d = d[['u', 'i', 't', 's']] if 's' in d.columns else d[['u', 'i', 't']]
    su, si = int(os.environ.get('SUBSAMPLE_USERS', '0')), int(os.environ.get('SUBSAMPLE_ITEMS', '0'))
    if su or si:   # smoke tests only
        r0 = np.random.default_rng(0)
        if si: d = d[d.i.isin(d.i.value_counts().index[:si])]
        if su:   # pre-split (STF) files: draw users among training users so that warm test rows remain
            uu = d[d.s == 0].u.unique() if 's' in d.columns else d.u.unique(); d = d[d.u.isin(r0.choice(uu, min(su, len(uu)), replace=False))]
        print('SUBSAMPLED (smoke only)', len(d), flush=True)
    return d.reset_index(drop=True)
df = load()
print('interactions', len(df), 'users', df.u.nunique(), 'items', df.i.nunique(), flush=True)

def split_random(d):
    p = rng.permutation(len(d)); n = len(d); a, b = int(.8 * n), int(.9 * n)
    return d.iloc[p[:a]], d.iloc[p[a:b]], d.iloc[p[b:]]
def split_loo(d):
    d = d.sort_values(['u', 't'], kind='stable'); rk = d.groupby('u').cumcount(ascending=False)
    return d[rk >= 2], d[rk == 1], d[rk == 0]
def split_global(d):
    if 's' in d.columns:   # E3 pre-split (split-then-filter) file
        return d[d.s == 0].drop(columns='s'), d[d.s == 1].drop(columns='s'), d[d.s == 2].drop(columns='s')
    d = d.sort_values('t', kind='stable'); n = len(d); a, b = int(.8 * n), int(.9 * n)
    return d.iloc[:a], d.iloc[a:b], d.iloc[b:]

class Index:
    def __init__(self, tr):
        self.users = np.array(sorted(set(tr.u))); self.items = np.array(sorted(set(tr.i)))
        self.uid = {u: k for k, u in enumerate(self.users)}; self.iid = {i: k for k, i in enumerate(self.items)}
        self.nu, self.ni = len(self.users), len(self.items)
    def warm(self, x): return x[x.u.isin(self.uid) & x.i.isin(self.iid)]
    def csr(self, x):   # binary user-item matrix of the rows of x that fall in this index space
        x = self.warm(x); u = x.u.map(self.uid).values; i = x.i.map(self.iid).values
        M = sp.csr_matrix((np.ones(len(u), np.float32), (u, i)), shape=(self.nu, self.ni)); M.sum_duplicates(); M.data[:] = 1
        return M
    def wcsr(self, x, t_max, span, hl):   # time-decayed weights (EASE-TD / ItemKNN-TD)
        x = self.warm(x); w = np.power(0.5, (t_max - x.t.values.astype(np.float64)) / (hl * span)).astype(np.float32)
        M = sp.csr_matrix((w, (x.u.map(self.uid).values, x.i.map(self.iid).values)), shape=(self.nu, self.ni))
        M.sum_duplicates(); return M

def binar(M): M = M.tocsr().copy(); M.data[:] = 1; return M

# ---------------- evaluation (round-3 metric code; nall optional) ----------------
def evaluate(score_fn, X_mask, T, users, nall=None):
    K = max(Ks); disc = 1 / torch.log2(torch.arange(2, K + 2, device=DEV, dtype=torch.float32))
    if len(users) == 0:
        e = {f'{m}@{k}': np.zeros(0, np.float32) for k in Ks for m in ('Recall', 'NDCG')}
        return {k: float('nan') for k in e}, e, (dict(e) if nall is not None else None)
    keys = [f'{m}@{k}' for k in Ks for m in ('Recall', 'NDCG')]
    res = {k: [] for k in keys}; resw = {k: [] for k in keys}
    for s in range(0, len(users), 1024):
        b = users[s:s + 1024]
        S = score_fn(b); S = S if torch.is_tensor(S) else torch.as_tensor(np.asarray(S, dtype=np.float32), device=DEV)
        S = S.float().clone(); S[torch.as_tensor(X_mask[b].toarray(), device=DEV) > 0] = -float('inf')
        top = torch.topk(S, K, dim=1).indices
        G = torch.as_tensor(T[b].toarray(), device=DEV) > 0
        hit = torch.gather(G, 1, top).float(); ng = G.sum(1).float()
        na = torch.as_tensor(nall[b], device=DEV).float() if nall is not None else None
        for k in Ks:
            h = hit[:, :k]; cs = torch.cumsum(disc[:k], 0); dcg = (h * disc[:k]).sum(1)
            idcg = cs[(ng.clamp(min=1, max=k) - 1).long()]
            res[f'Recall@{k}'].append(torch.where(ng > 0, h.sum(1) / ng.clamp(min=1), torch.nan).cpu().numpy())
            res[f'NDCG@{k}'].append(torch.where(ng > 0, dcg / idcg, torch.nan).cpu().numpy())
            if na is not None:
                idcg_a = cs[(na.clamp(min=1, max=k) - 1).long()]
                resw[f'Recall@{k}'].append((h.sum(1) / na.clamp(min=1)).cpu().numpy())
                resw[f'NDCG@{k}'].append((dcg / idcg_a).cpu().numpy())
    res = {k: np.concatenate(v) for k, v in res.items()}
    resw = {k: np.concatenate(v) for k, v in resw.items()} if nall is not None else None
    return {k: float(np.nanmean(v)) for k, v in res.items()}, res, resw
def rows_users(T): return np.flatnonzero(np.diff(T.indptr))

# ---------------- closed-form / count models (round-3 code; weighted input allowed) ----------------
def m_pop(X):
    pop = torch.as_tensor(np.asarray(X.sum(0)).ravel(), device=DEV, dtype=torch.float32)
    return lambda b: pop.unsqueeze(0).expand(len(b), -1)
def m_itemknn(X, k=100, Xs=None):   # Xs: scoring profile (weighted for -TD); similarity from X (weighted for -TD)
    Xs = X if Xs is None else Xs
    S = torch.as_tensor((X.T @ X).toarray(), device=DEV, dtype=torch.float32)
    norm = torch.sqrt(torch.diag(S)) + 1e-8; S = S / norm[:, None] / norm[None, :]; S.fill_diagonal_(0)
    k = min(k, S.shape[0] - 1); v, idx = torch.topk(S, k, dim=1)
    if DEV.type == 'cpu':
        n = S.shape[0]; del S
        Ssp = sp.csr_matrix((v.numpy().ravel(), idx.numpy().ravel(), np.arange(0, n * k + 1, k)), shape=(n, n))
        return lambda b: torch.from_numpy((Xs[b] @ Ssp).toarray().astype(np.float32))
    S = torch.zeros_like(S).scatter_(1, idx, v)
    return lambda b: torch.as_tensor(Xs[b].toarray(), device=DEV, dtype=torch.float32) @ S
def m_ease(X, lam=500., Xs=None):
    Xs = X if Xs is None else Xs
    G = (X.T @ X).toarray().astype(np.float64); G[np.diag_indices_from(G)] += lam
    P = np.linalg.inv(G); del G
    B = P / (-np.diag(P)); del P; np.fill_diagonal(B, 0); B = B.astype(np.float32)
    if DEV.type == 'cpu': return lambda b: torch.from_numpy(np.asarray(Xs[b] @ B, dtype=np.float32))
    B = torch.as_tensor(B, device=DEV)
    return lambda b: torch.as_tensor(Xs[b].toarray(), device=DEV, dtype=torch.float32) @ B

# ---------------- embedding models (round-3 code; positives and adjacency may differ for e2) ----------------
def norm_adj(X):
    A = sp.bmat([[None, X], [X.T, None]]).tocsr()
    dg = np.asarray(A.sum(1)).ravel(); dinv = np.power(dg + 1e-8, -.5)
    A = (sp.diags(dinv) @ A @ sp.diags(dinv)).tocsr().astype(np.float32)
    return torch.sparse_csr_tensor(torch.as_tensor(A.indptr, dtype=torch.long), torch.as_tensor(A.indices, dtype=torch.long),
                                   torch.as_tensor(A.data), size=A.shape).to(DEV)
def sample_neg(u, ni, POSKEY, g):
    n = torch.randint(0, ni, (len(u),), device=DEV, generator=g)
    for _ in range(3):
        key = u * ni + n; pos = torch.searchsorted(POSKEY, key).clamp(max=len(POSKEY) - 1); m = POSKEY[pos] == key
        if not bool(m.any()): break
        n = torch.where(m, torch.randint(0, ni, (len(u),), device=DEV, generator=g), n)
    return n
def train_emb(Xpos, Va, va_users, Xmask_val, nu, ni, lightgcn, Xadj=None, dim=64, layers=3, lr=1e-3, reg=1e-4, fixed_epochs=None):
    """fixed_epochs=None: early stopping on validation (every 5 epochs, patience PATIENCE, cap MAX_MIN) as round 3.
    fixed_epochs=e: train exactly e epochs without validation (refit), same per-run cap x1.5."""
    g = torch.Generator(device=DEV); g.manual_seed(SEED)
    U = torch.nn.Embedding(nu, dim).to(DEV); I = torch.nn.Embedding(ni, dim).to(DEV)
    torch.nn.init.normal_(U.weight, std=.1); torch.nn.init.normal_(I.weight, std=.1)
    opt = torch.optim.Adam(list(U.parameters()) + list(I.parameters()), lr=lr)
    coo = Xpos.tocoo()
    pu = torch.as_tensor(coo.row, device=DEV, dtype=torch.long); pi = torch.as_tensor(coo.col, device=DEV, dtype=torch.long)
    POSKEY = torch.sort(pu * ni + pi).values
    prop = lightgcn and layers > 0
    if prop: At = norm_adj(Xpos if Xadj is None else Xadj)
    def full():
        if not prop: return U.weight, I.weight
        E = torch.cat([U.weight, I.weight]); acc = E
        for _ in range(layers): E = At @ E; acc = acc + E
        E = acc / (layers + 1); return E[:nu], E[nu:]
    best, best_state, bad, log, t_start = -1, None, 0, [], time.time(); n_pos = len(pu)
    n_ep = fixed_epochs or EPOCHS
    for ep in range(n_ep):
        t0 = time.time(); perm = torch.randperm(n_pos, device=DEV, generator=g); tot = torch.zeros((), device=DEV)
        for s in range(0, n_pos, BS):
            b = perm[s:s + BS]; u = pu[b]; p = pi[b]; n = sample_neg(u, ni, POSKEY, g)
            Eu, Ei = full(); eu, ep_, en = Eu[u], Ei[p], Ei[n]
            loss = -F.logsigmoid((eu * ep_).sum(1) - (eu * en).sum(1)).mean()
            r = (U.weight[u].norm(2).pow(2) + I.weight[p].norm(2).pow(2) + I.weight[n].norm(2).pow(2)) / len(b)
            loss = loss + reg * r
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.detach()
        if fixed_epochs:
            if MAX_MIN and time.time() - t_start > 1.5 * MAX_MIN * 60: print('    refit cap hit at epoch', ep + 1, flush=True); log.append(('cap', ep + 1)); break
            continue
        if (ep + 1) % 5 == 0:
            with torch.no_grad():
                Eu, Ei = full(); Ue, Ie = Eu.detach().clone(), Ei.detach().clone()
                vm, _, _ = evaluate(lambda bb: Ue[torch.as_tensor(bb, device=DEV)] @ Ie.T, Xmask_val, Va, va_users)
            log.append((ep + 1, round(float(tot), 2), round(vm['NDCG@20'], 4), round(time.time() - t0, 2)))
            print('   ', 'LightGCN' if lightgcn else 'BPR-MF', f'K={layers}' if lightgcn else '', log[-1], flush=True)
            if vm['NDCG@20'] > best: best, bad, best_state = vm['NDCG@20'], 0, (Ue, Ie)
            else:
                bad += 1
                if bad >= PATIENCE: break
            if MAX_MIN and time.time() - t_start > MAX_MIN * 60: print('    MAX_MIN cap hit', flush=True); break
    if fixed_epochs or best_state is None:
        with torch.no_grad(): Eu, Ei = full(); best_state = (Eu.detach().clone(), Ei.detach().clone())
    Ue, Ie = best_state
    return (lambda bb: Ue[torch.as_tensor(bb, device=DEV)] @ Ie.T), log
def train_simgcl(Xpos, Va, va_users, Xmask_val, nu, ni, dim=64, n_layers=2, cl_rate=0.5, eps=0.1, tau=0.2, reg=1e-4, lr=1e-3, fixed_epochs=None):
    bs = int(os.environ.get('BS_SIMGCL', '2048')); max_epoch = fixed_epochs or int(os.environ.get('SIMGCL_EPOCHS', '20'))
    g = torch.Generator(device=DEV); g.manual_seed(SEED)
    Ue = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(nu, dim)).to(DEV))
    Ie = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(ni, dim)).to(DEV))
    opt = torch.optim.Adam([Ue, Ie], lr=lr); At = norm_adj(Xpos); coo = Xpos.tocoo()
    pu = torch.as_tensor(coo.row, device=DEV, dtype=torch.long); pi = torch.as_tensor(coo.col, device=DEV, dtype=torch.long)
    POSKEY = torch.sort(pu * ni + pi).values
    def fwd(perturbed=False):
        ego = torch.cat([Ue, Ie], 0); alls = []
        for _ in range(n_layers):
            ego = At @ ego
            if perturbed: ego = ego + torch.sign(ego) * F.normalize(torch.rand(ego.shape, device=DEV, generator=g), dim=-1) * eps
            alls.append(ego)
        E = torch.stack(alls, 1).mean(1); return E[:nu], E[nu:]
    def infonce(v1, v2):
        v1, v2 = F.normalize(v1, dim=1), F.normalize(v2, dim=1)
        return -torch.diag(F.log_softmax((v1 @ v2.T) / tau, dim=1)).mean()
    best, best_state, bad, log, t_start = -1, None, 0, [], time.time(); n_pos = len(pu)
    for ep in range(max_epoch):
        t0 = time.time(); perm = torch.randperm(n_pos, device=DEV, generator=g); tot = torch.zeros((), device=DEV)
        for s in range(0, n_pos, bs):
            b = perm[s:s + bs]; u = pu[b]; p = pi[b]; n = sample_neg(u, ni, POSKEY, g)
            ru, ri = fwd(); eu, ep_, en = ru[u], ri[p], ri[n]
            rec = torch.mean(-torch.log(10e-6 + torch.sigmoid((eu * ep_).sum(1) - (eu * en).sum(1))))
            l2 = reg * (torch.norm(eu, p=2) / eu.shape[0] + torch.norm(ep_, p=2) / ep_.shape[0])
            uu, ii = torch.unique(u), torch.unique(p); u1, i1 = fwd(True); u2, i2 = fwd(True)
            loss = rec + l2 + cl_rate * (infonce(u1[uu], u2[uu]) + infonce(i1[ii], i2[ii]))
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.detach()
        if fixed_epochs: continue
        with torch.no_grad():
            ru, ri = fwd(); Uc, Ic = ru.detach().clone(), ri.detach().clone()
            vm, _, _ = evaluate(lambda bb: Uc[torch.as_tensor(bb, device=DEV)] @ Ic.T, Xmask_val, Va, va_users)
        log.append((ep + 1, round(float(tot), 2), round(vm['NDCG@20'], 4), round(time.time() - t0, 2)))
        print('    SimGCL', log[-1], flush=True)
        if vm['NDCG@20'] > best: best, bad, best_state = vm['NDCG@20'], 0, (Uc, Ic)
        else:
            bad += 1
            if bad >= PATIENCE: break
        if MAX_MIN and time.time() - t_start > MAX_MIN * 60: print('    MAX_MIN cap hit', flush=True); break
    if fixed_epochs or best_state is None:
        with torch.no_grad(): ru, ri = fwd(); best_state = (ru.detach().clone(), ri.detach().clone())
    Uc, Ic = best_state
    return (lambda bb: Uc[torch.as_tensor(bb, device=DEV)] @ Ic.T), log
def best_epoch(log):
    v = [x for x in log if isinstance(x[0], int)]
    return max(v, key=lambda x: x[2])[0] if v else None

# ---------------- model fitting with validation grids ----------------
CHOSEN = {}   # model -> chosen hyper-parameter (closed form, tuned once per process: e1 tunes on condition a)
def fit(name, ctx, Xtr, Va, va_users, Xmask_val, idx, trdf=None, fixed_epochs=None, layers=None, Xadj=None):
    """returns score_fn, info dict. Closed-form grids are evaluated on validation unless CHOSEN already has a value."""
    info = {}
    def pick(cands, build):
        if name in CHOSEN: return build(CHOSEN[name]), {'param': CHOSEN[name], 'tuned_in': 'reused'}
        best = (-1, None, None); vals = {}
        for c in cands:
            sf = build(c); vm, _, _ = evaluate(sf, Xmask_val, Va, va_users); vals[str(c)] = round(vm['NDCG@20'], 5)
            if vm['NDCG@20'] > best[0]: best = (vm['NDCG@20'], c, sf)
            else: del sf
        CHOSEN[name] = best[1]
        at_edge = len(cands) > 1 and best[1] in (cands[0], cands[-1])
        print(f'    grid {name}', vals, '-> chosen', best[1], '(AT GRID EDGE)' if at_edge else '', flush=True)
        return best[2], {'param': best[1], 'val_grid': vals, 'at_grid_edge': bool(at_edge), 'tuned_in': ctx}
    if name == 'MostPop': return m_pop(Xtr), {}
    if name == 'ItemKNN': return pick(KNN_GRID, lambda k: m_itemknn(Xtr, int(k)))
    if name == 'EASE': return pick(EASE_GRID, lambda l: m_ease(Xtr, l))
    if name in ('RecentPop', 'EASE-TD', 'ItemKNN-TD'):
        t = trdf.sort_values('t', kind='stable'); tw = idx.warm(t); t_max = float(tw.t.max()); span = max(float(tw.t.max() - tw.t.min()), 1.)
        if name == 'RecentPop':
            def b(w): return m_pop(idx.csr(tw.iloc[int((1 - w) * len(tw)):]))
            return pick(RECENT_GRID, b)
        lam = CHOSEN.get('EASE', EASE_GRID[0] if len(EASE_GRID) == 1 else None); k = CHOSEN.get('ItemKNN', KNN_GRID[0] if len(KNN_GRID) == 1 else None)
        if name == 'EASE-TD':
            assert lam is not None, 'EASE-TD needs EASE earlier in MODELS (or a one-value EASE_GRID)'
            sf, inf = pick(HL_GRID, lambda h: (lambda Xw: m_ease(Xw, lam))(idx.wcsr(tw, t_max, span, h))); inf['lam'] = lam; return sf, inf
        assert k is not None, 'ItemKNN-TD needs ItemKNN earlier in MODELS (or a one-value KNN_GRID)'
        sf, inf = pick(HL_GRID, lambda h: (lambda Xw: m_itemknn(Xw, int(k)))(idx.wcsr(tw, t_max, span, h))); inf['k'] = k; return sf, inf
    if name == 'SimGCL':
        sf, log = train_simgcl(Xtr, Va, va_users, Xmask_val, idx.nu, idx.ni, lr=LR['SimGCL'], fixed_epochs=fixed_epochs)
        return sf, {'val_log': log, 'best_epoch': best_epoch(log), 'lr': LR['SimGCL']}
    if name in ('BPR-MF', 'LightGCN'):
        L = LAYERS if layers is None else layers
        sf, log = train_emb(Xtr, Va, va_users, Xmask_val, idx.nu, idx.ni, lightgcn=(name == 'LightGCN'), Xadj=Xadj,
                            layers=L, lr=LR[name], fixed_epochs=fixed_epochs)
        return sf, {'val_log': log, 'best_epoch': best_epoch(log), 'lr': LR[name], 'layers': L if name == 'LightGCN' else None}
    raise ValueError(name)

results = {'exp': EXP, 'dataset': DATASET, 'seed': SEED, 'device': str(DEV), 'version': 'v5', 'torch': torch.__version__}
def save(key, model, m, pw, users_raw, extra_arrays=None):
    results.setdefault(key, {})[model] = m
    json.dump(results, open(f'{OUT}/res_{TAG}.json', 'w'), indent=1, default=str)
    arr = {f'{pop}_{k}': v.astype(np.float32) for pop, d in pw.items() if d is not None for k, v in d.items()}
    np.savez_compressed(f'{OUT}/peruser_{TAG}_{key}_{model}.npz', users_raw=np.asarray(users_raw).astype(str), **arr, **(extra_arrays or {}))
    print(' ', key, model, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items() if k not in ('val_log', 'val_grid')}, flush=True)
def mean_of(d): return {k: float(np.nanmean(v)) for k, v in d.items()}
def tsec(t0):
    if DEV.type == 'cuda': torch.cuda.synchronize()
    return round(time.time() - t0, 1)

# ---------------- E1 / E2 construction ----------------
def build_leak(tr, va, te):
    idx = Index(tr); r0 = np.random.default_rng(int(os.environ.get('E1_SPLIT_SEED', '12345')))
    tew = idx.warm(te).drop_duplicates(['u', 'i'])
    tu = np.array(sorted(tew.u.unique())); r0.shuffle(tu); E = set(tu[:len(tu) // 2]); D = set(tu[len(tu) // 2:])
    LD = tew[tew.u.isin(D)]
    te_e = tew[tew.u.isin(E)].copy(); te_e['r'] = r0.random(len(te_e))
    te_e = te_e.sort_values(['u', 'r']); rk = te_e.groupby('u').cumcount(); n = te_e.groupby('u').u.transform('size')
    H = te_e[rk < np.ceil(n / 2)].drop(columns='r'); Fo = te_e[rk >= np.ceil(n / 2)].drop(columns='r')
    trs = tr.iloc[r0.permutation(len(tr))]; mc = len(LD) + len(Fo); B, R = trs.iloc[:len(tr) - mc], trs.iloc[len(tr) - mc:]
    st = dict(N=len(tr), test_users_warm=len(tu), eval_users=len(E), donor_users=len(D), H_rows=len(H), F_rows=len(Fo),
              LD_rows=len(LD), leak_frac_c=round(mc / len(tr), 4), cut_t=int(tr.t.max()))
    return idx, B, R, LD, H, Fo, st

def run_leak():
    tr, va, te = split_global(df); idx, B, R, LD, H, Fo, st = build_leak(tr, va, te)
    print('leak construction', st, flush=True); results['construction'] = st
    vaw = idx.warm(va); Va = idx.csr(vaw); va_users = rows_users(Va); Fm = idx.csr(Fo); Ht = idx.csr(H); ev = rows_users(Ht)
    if EXP == 'e1':
        conds = {'a': pd.concat([B, R]), 'b': pd.concat([B, LD, R.iloc[:len(Fo)]]), 'c': pd.concat([B, LD, Fo]),
                 'f': pd.concat([B, Fo, R.iloc[:len(LD)]])}
        todo = CONDS or ['a', 'b', 'c']
        if 'a' not in todo and any(m in MODELS for m in ('ItemKNN', 'EASE', 'RecentPop', 'EASE-TD', 'ItemKNN-TD')):
            todo = ['a'] + todo; print('  adding condition a (closed-form values are tuned there)', flush=True)
        for c in ['a', 'b', 'c', 'f']:
            if c not in todo: continue
            d = conds[c]; X = idx.csr(d); mask = binar(X + Va + Fm)
            info = dict(rows=int(X.nnz), after_cut_pct=round(100 * float((d.t > st['cut_t']).mean()), 3),
                        zero_deg_items=int((np.asarray(X.sum(0)).ravel() == 0).sum()), zero_deg_users=int((np.asarray(X.sum(1)).ravel() == 0).sum()))
            results.setdefault(f'cond_{c}', {})['info'] = info; print('cond', c, info, flush=True)
            for name in MODELS:
                t0 = time.time(); sf, inf = fit(name, f'e1-{c}', X, Va, va_users, X, idx, trdf=d)
                _, pw, _ = evaluate(sf, mask, Ht, ev)
                m = mean_of(pw); m.update(inf); m['sec'] = tsec(t0); save(f'cond_{c}', name, m, {'warm': pw}, idx.users[ev]); del sf
                if DEV.type == 'cuda': torch.cuda.empty_cache()
    else:   # e2
        L = pd.concat([LD, Fo]); Xb = idx.csr(tr); Xl = binar(Xb + idx.csr(L)); mask = binar(Xb + Va + Fm)
        variants = {'neither': (Xb, Xb), 'adj': (Xb, Xl), 'both': (Xl, Xl), 'loss': (Xl, Xb)}   # (positives, adjacency)
        todo = CONDS or list(variants); Ls = [int(x) for x in os.environ.get('E2_LAYERS', '0,1,2,3,4').split(',')]
        LOSS_LAYERS = [int(x) for x in os.environ.get('E2_LOSS_LAYERS', '0,1,2,3,4').split(',')]   # loss-only variant run at these K
        for name in MODELS:
            if name == 'LightGCN':
                for K in Ls:
                    for v in todo:
                        if K == 0 and v in ('adj', 'both'): continue   # no propagation: adj == neither, both == loss
                        if v == 'loss' and K not in LOSS_LAYERS: continue
                        Xp, Xa = variants[v]; t0 = time.time()
                        sf, inf = fit(name, 'e2', Xp, Va, va_users, Xp, idx, layers=K, Xadj=Xa)
                        _, pw, _ = evaluate(sf, mask, Ht, ev); m = mean_of(pw); m.update(inf); m['sec'] = tsec(t0)
                        save(f'{v}_K{K}', name, m, {'warm': pw}, idx.users[ev]); del sf
            else:   # references without a graph: training data with / without the leak set
                for v in ('neither', 'both'):
                    if v not in todo: continue
                    Xp = variants[v][0]; t0 = time.time()
                    sf, inf = fit(name, f'e2-{v}', Xp, Va, va_users, Xp, idx, trdf=tr if v == 'neither' else pd.concat([tr, L]))
                    _, pw, _ = evaluate(sf, mask, Ht, ev); m = mean_of(pw); m.update(inf); m['sec'] = tsec(t0)
                    save(f'{v}_ref', name, m, {'warm': pw}, idx.users[ev]); del sf

def run_aligned():
    gtr, gva, gte = split_global(df); gidx = Index(gtr); GU = set(gidx.warm(gte).u); GI = set(gtr.i)
    results['gt_reference'] = dict(gt_warm_test_users=len(GU), gt_train_items=len(GI))
    for proto, fn in (('random', split_random), ('loo', split_loo)):
        if CONDS and proto not in CONDS: continue
        tr, va, te = fn(df); idx = Index(tr)
        Va = idx.csr(va); va_users = rows_users(Va); X = idx.csr(tr); Xtv = binar(X + Va)
        tew = idx.warm(te); Te = idx.csr(tew); Ta = idx.csr(tew[tew.u.isin(GU) & tew.i.isin(GI)])
        ev, eva = rows_users(Te), rows_users(Ta)
        info = dict(test_users_warm=len(ev), test_rows_warm=int(Te.nnz), aligned_users=len(eva), aligned_rows=int(Ta.nnz))
        results.setdefault(proto, {})['info'] = info; print(proto, info, flush=True); CHOSEN.clear()
        for name in MODELS:
            t0 = time.time(); sf, inf = fit(name, f'aligned-{proto}', X, Va, va_users, X, idx, trdf=tr)
            _, pw, _ = evaluate(sf, Xtv, Te, ev); _, pa, _ = evaluate(sf, Xtv, Ta, eva)
            m = {f'warm_{k}': v for k, v in mean_of(pw).items()}; m.update({f'aligned_{k}': v for k, v in mean_of(pa).items()})
            m.update(inf); m['sec'] = tsec(t0)
            save(proto, name, m, {'warm': pw}, idx.users[ev]); save(proto + '_aligned', name, m, {'aligned': pa}, idx.users[eva]); del sf

def run_gtrefit():
    tr, va, te = split_global(df); trva = pd.concat([tr, va])
    i0, i1 = Index(tr), Index(trva); tew0 = i0.warm(te).drop_duplicates(['u', 'i']); tew1 = i1.warm(te).drop_duplicates(['u', 'i'])
    Va0 = i0.csr(va); vu0 = rows_users(Va0); X0 = i0.csr(tr); M0 = binar(X0 + Va0); X1 = i1.csr(trva)
    T0w = i0.csr(tew0); T1w = i1.csr(tew0); T1tv = i1.csr(tew1)
    e0, e1w, e1tv = rows_users(T0w), rows_users(T1w), rows_users(T1tv)
    info = dict(warm_rows=int(T0w.nnz), warm_users=len(e0), warmtv_rows=int(T1tv.nnz), warmtv_users=len(e1tv))
    results['info'] = info; print('gtrefit', info, flush=True)
    for name in MODELS:
        t0 = time.time(); sf, inf = fit(name, 'gtrefit', X0, Va0, vu0, X0, i0, trdf=tr)
        _, pw, _ = evaluate(sf, M0, T0w, e0)
        m = mean_of(pw); m.update(inf); m['sec'] = tsec(t0); save('norefit', name, m, {'warm': pw}, i0.users[e0]); del sf
        # refit on train+val: same closed-form value (CHOSEN), embedding models: best epoch of the norefit run
        fe = inf.get('best_epoch'); t0 = time.time()
        if name in ('BPR-MF', 'LightGCN', 'SimGCL') and not fe: print('  no best epoch; skip refit', name, flush=True); continue
        sf, inf2 = fit(name, 'gtrefit-refit', X1, None, None, None, i1, trdf=trva, fixed_epochs=fe)
        _, pw1, _ = evaluate(sf, X1, T1w, e1w); _, pw2, _ = evaluate(sf, X1, T1tv, e1tv)
        m = {f'warm_{k}': v for k, v in mean_of(pw1).items()}; m.update({f'warmtv_{k}': v for k, v in mean_of(pw2).items()})
        m.update(inf2); m['refit_epochs'] = fe; m['sec'] = tsec(t0)
        save('refit', name, m, {'warm': pw1}, i1.users[e1w]); save('refit_warmtv', name, m, {'warmtv': pw2}, i1.users[e1tv]); del sf

def run_gt():
    tr, va, te = split_global(df); idx = Index(tr)
    us, its = set(tr.u), set(tr.i); tep = te[['u', 'i']].drop_duplicates()
    info = dict(train=len(tr), val=len(va), test=len(te), users=idx.nu, items=idx.ni,
                test_rows_all=len(tep), test_users_all=int(tep.u.nunique()),
                test_usercold_rows_pct=round(100 * float((~tep.u.isin(us)).mean()), 2),
                test_itemcold_rows_pct=round(100 * float((tep.u.isin(us) & ~tep.i.isin(its)).mean()), 2))
    Va = idx.csr(va); va_users = rows_users(Va); X = idx.csr(tr); Xtv = binar(X + Va)
    tew = idx.warm(tep); Te = idx.csr(tew); info['test_rows_warm'] = int(Te.nnz)
    na = tep[tep.u.isin(us)].groupby('u').size(); nall = np.zeros(idx.nu, np.int64); nall[na.index.map(idx.uid).values] = na.values
    ev = np.flatnonzero(nall > 0); info['test_users_warm'] = int(len(rows_users(Te))); info['test_users_wu'] = len(ev)
    results['info'] = info; print('gt', info, flush=True)
    for name in MODELS:
        t0 = time.time(); sf, inf = fit(name, 'gt', X, Va, va_users, X, idx, trdf=tr)
        _, pw, pwu = evaluate(sf, Xtv, Te, ev, nall)
        m = mean_of(pw); m.update({f'full_{k}': float(v.sum() / info['test_users_all']) for k, v in pwu.items()})
        m.update({f'wu_{k}': float(v.mean()) for k, v in pwu.items()}); m.update(inf); m['sec'] = tsec(t0)
        save('gt', name, m, {'warm': pw, 'wu': pwu}, idx.users[ev]); del sf

{'e1': run_leak, 'e2': run_leak, 'aligned': run_aligned, 'gtrefit': run_gtrefit, 'gt': run_gt}[EXP]()
results['total_sec'] = round(time.time() - T00, 1); json.dump(results, open(f'{OUT}/res_{TAG}.json', 'w'), indent=1, default=str)
print('DONE', TAG, results['total_sec'], 's', flush=True)
