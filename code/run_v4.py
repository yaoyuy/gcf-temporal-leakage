"""v4 (round 3) = run_v3.py + three additions, nothing else changed for the 5 round-2 models:
(1) Evaluation reports two populations (same top-K lists, same masking as v3):
    - 'warm'  : test interactions whose user AND item occur in TRAIN (this is exactly the v3/r2 metric; r2 numbers are 'warm').
    - 'full'  : ALL test interactions. Cold items stay in each user's ground truth (counted in Recall/IDCG denominators,
                unretrievable); users absent from train cannot be scored by any model and count as 0. 'full' = mean over all
                raw test users. 'wu' (warm users, all their items) is stored per user; full = sum(wu)/n_test_users_all.
    Counts recorded in info: test_users_all, test_users_wu (users in train with >=1 test row), test_users_warm (r2 test users),
    test_rows_all, test_rows_warm (unique (u,i) pairs).
(2) Per-user arrays saved per run as npz: users (internal ids), Recall/NDCG @10/@20 for 'warm' (NaN where no warm item) and 'wu'.
(2b) ItemKNN k (env KNN_K, default 100) and EASE lambda (env EASE_LAM, default 500) are settable; closed-form models also
    report validation NDCG@20 ('val_NDCG@20') so k / lambda can be tuned on validation like the lr grid.
(3) Two more graph CF models:
    - SimGCL: Yu et al., "Are Graph Augmentations Necessary? Simple Graph Contrastive Learning for Recommendation", SIGIR 2022.
      Port of SELFRec (github.com/Coder-Yu/SELFRec, model/graph/SimGCL.py, conf/SimGCL.yaml): n_layer=2, lambda(cl_rate)=0.5,
      eps=0.1, tau=0.2, emb 64, batch 2048, reg 1e-4 (l2_reg_loss on batch user/pos embeddings), xavier init, max.epoch 20,
      per-epoch validation, keep best. lr chosen from a small grid on seed 1 (default 1e-3).
    - UltraGCN: Mao et al., "UltraGCN: Ultra Simplification of Graph Convolutional Networks for Recommendation", CIKM 2021.
      Port of the official main.py (github.com/reczoo/RecZoo/tree/main/matching/gnn/UltraGCN, formerly xue-pai/UltraGCN):
      constraint loss L_C (BCE with omega weights w1..w4, uniform negatives without sifting), item-item loss L_I
      (ii_neighbor_num=10), gamma * ||Theta||^2/2, sum reduction. Per-dataset configs from that repo:
      ml-1m -> config/ultragcn_movielens1m_m1.ini; gowalla -> ultragcn_gowalla_m1.ini; Amazon sets -> ultragcn_amazonbooks_m1.ini.
      Deviation: embedding dim fixed to 64 for all models (ML-1M config uses 128); validation every 5 epochs with patience 5
      and the same per-run minute cap as BPR-MF/LightGCN instead of early_stop_epoch on the test set.
Usage: python run_v4.py SEED EPOCHS PROTOS MODELS   env: DATASET, DATA_DIR, OUT_DIR, PATIENCE, MAX_MIN, LR, BS, TAG
"""
import sys, os, time, json, numpy as np, pandas as pd, scipy.sparse as sp, torch
import torch.nn.functional as F

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 1
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 300
PROTOS = sys.argv[3].split(',') if len(sys.argv) > 3 else ['random', 'loo', 'global_temporal']
MODELS = sys.argv[4].split(',') if len(sys.argv) > 4 else ['MostPop', 'ItemKNN', 'EASE', 'BPR-MF', 'LightGCN']
rng = np.random.default_rng(SEED); torch.manual_seed(SEED)
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATASET = os.environ.get('DATASET', 'ml-1m')
DATA = os.environ.get('DATA_DIR', 'data/ml-1m')
OUT = os.environ.get('OUT_DIR', 'out_v4'); os.makedirs(OUT, exist_ok=True)
PATIENCE = int(os.environ.get('PATIENCE', '5'))
MAX_MIN = float(os.environ.get('MAX_MIN', '0'))  # 0 = no cap
print('device', DEV, 'dataset', DATASET, 'torch', torch.__version__, flush=True)
Ks = [10, 20]

def load(name):
    if name == 'ml-1m':
        d = pd.read_csv(f'{DATA}/ratings.dat', sep='::', engine='python', names=['u', 'i', 'r', 't'])
        return d.drop(columns='r')
    return pd.read_csv(f'{DATA}/{name}.csv', usecols=['u', 'i', 't'])

df = load(DATASET)
print('interactions', len(df), 'users', df.u.nunique(), 'items', df.i.nunique(), flush=True)

def split_random(d):
    p = rng.permutation(len(d)); n = len(d); a, b = int(.8 * n), int(.9 * n)
    return d.iloc[p[:a]], d.iloc[p[a:b]], d.iloc[p[b:]]

def split_loo(d):
    d = d.sort_values(['u', 't'], kind='stable'); rk = d.groupby('u').cumcount(ascending=False)
    return d[rk >= 2], d[rk == 1], d[rk == 0]

def split_global(d):
    d = d.sort_values('t', kind='stable'); n = len(d); a, b = int(.8 * n), int(.9 * n)
    return d.iloc[:a], d.iloc[a:b], d.iloc[b:]

def prep(tr, va, te):
    us, its = set(tr.u), set(tr.i); stats = {}; out = []
    for name, x in (('val', va), ('test', te)):
        keep = x.u.isin(us) & x.i.isin(its)
        stats[f'{name}_cold_rows_pct'] = round(100 * (1 - keep.mean()), 2); out.append(x[keep])
    va_w, te_w = out
    uid = {u: k for k, u in enumerate(sorted(us))}; iid = {i: k for k, i in enumerate(sorted(its))}
    f = lambda x: (x.u.map(uid).values, x.i.map(iid).values)
    # full-population bookkeeping (new in v4)
    te_pairs = te[['u', 'i']].drop_duplicates()
    n_all = te_pairs[te_pairs.u.isin(us)].groupby('u').size()          # unique test items per warm user, incl. cold items
    nall = np.zeros(len(uid), dtype=np.int64); nall[n_all.index.map(uid).values] = n_all.values
    stats.update(test_users_all=int(te_pairs.u.nunique()), test_users_wu=int((nall > 0).sum()),
                 test_rows_all=int(len(te_pairs)), test_rows_warm=int(len(te_w[['u', 'i']].drop_duplicates())))
    return f(tr), f(va_w), f(te_w), len(uid), len(iid), stats, nall

def leak_stats(tr, te):
    first_te = te.groupby('u').t.min(); m = tr.join(first_te.rename('t0'), on='u')
    cut = df.t.quantile(.8)
    return {'train_after_user_first_test_pct': round(float(100 * (m.t > m.t0).mean()), 2),
            'train_after_global_cut_pct': round(float(100 * (tr.t > cut).mean()), 2)}

def csr(u, i, nu, ni):
    return sp.csr_matrix((np.ones(len(u), dtype=np.float32), (u, i)), shape=(nu, ni))

def evaluate(score_fn, X_mask, T, users, nall=None):
    """T: csr of warm ground truth. Metric definitions identical to v2/v3 for the warm population.
    If nall is given (unique test items per user incl. cold items), also returns 'wu' metrics with nall as denominator."""
    K = max(Ks); disc = 1 / torch.log2(torch.arange(2, K + 2, device=DEV, dtype=torch.float32))
    keys = [f'{m}@{k}' for k in Ks for m in ('Recall', 'NDCG')]
    res = {k: [] for k in keys}; resw = {k: [] for k in keys}
    for s in range(0, len(users), 1024):
        b = users[s:s + 1024]
        S = score_fn(b)
        S = S if torch.is_tensor(S) else torch.as_tensor(np.asarray(S, dtype=np.float32), device=DEV)
        S = S.float().clone()
        S[torch.as_tensor(X_mask[b].toarray(), device=DEV) > 0] = -float('inf')
        top = torch.topk(S, K, dim=1).indices
        G = torch.as_tensor(T[b].toarray(), device=DEV) > 0
        hit = torch.gather(G, 1, top).float(); ng = G.sum(1).float()
        na = torch.as_tensor(nall[b], device=DEV).float() if nall is not None else None
        for k in Ks:
            h = hit[:, :k]; cs = torch.cumsum(disc[:k], 0); dcg = (h * disc[:k]).sum(1)
            # warm (r2 definition); users without warm items get NaN (only possible when nall is given)
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

# ---------- models (MostPop/ItemKNN/EASE/BPR-MF/LightGCN unchanged from v3; CPU scoring path added) ----------
def m_pop(X):
    pop = torch.as_tensor(np.asarray(X.sum(0)).ravel(), device=DEV, dtype=torch.float32)
    return lambda b: pop.unsqueeze(0).expand(len(b), -1)

def m_itemknn(X, k=100):
    S = torch.as_tensor((X.T @ X).toarray(), device=DEV)
    norm = torch.sqrt(torch.diag(S)) + 1e-8
    S = S / norm[:, None] / norm[None, :]; S.fill_diagonal_(0)
    v, idx = torch.topk(S, k, dim=1)
    if DEV.type == 'cpu':  # same top-k similarity matrix, sparse product on CPU (identical math)
        n = S.shape[0]; del S
        Ssp = sp.csr_matrix((v.numpy().ravel(), idx.numpy().ravel(), np.arange(0, n * k + 1, k)), shape=(n, n))
        return lambda b: torch.from_numpy((X[b] @ Ssp).toarray().astype(np.float32))
    S = torch.zeros_like(S).scatter_(1, idx, v)
    return lambda b: torch.as_tensor(X[b].toarray(), device=DEV) @ S

def m_ease(X, lam=500.):
    G = (X.T @ X).toarray().astype(np.float64)
    G[np.diag_indices_from(G)] += lam
    P = np.linalg.inv(G); del G
    B = P / (-np.diag(P)); del P; np.fill_diagonal(B, 0)
    B = B.astype(np.float32)
    if DEV.type == 'cpu':
        return lambda b: torch.from_numpy(np.asarray(X[b] @ B, dtype=np.float32))
    B = torch.as_tensor(B, device=DEV)
    return lambda b: torch.as_tensor(X[b].toarray(), device=DEV) @ B

LR = float(os.environ.get('LR', '1e-3')); BS = int(os.environ.get('BS', '4096'))
import warnings; warnings.filterwarnings('ignore', category=UserWarning)

def norm_adj(X):
    A = sp.bmat([[None, X], [X.T, None]]).tocsr()
    dg = np.asarray(A.sum(1)).ravel(); dinv = np.power(dg + 1e-8, -.5)
    A = (sp.diags(dinv) @ A @ sp.diags(dinv)).tocsr().astype(np.float32)
    return torch.sparse_csr_tensor(torch.as_tensor(A.indptr, dtype=torch.long), torch.as_tensor(A.indices, dtype=torch.long),
                                   torch.as_tensor(A.data), size=A.shape).to(DEV)

def sample_neg(u, ni, POSKEY, g):
    n = torch.randint(0, ni, (len(u),), device=DEV, generator=g)
    for _ in range(3):
        key = u * ni + n; pos = torch.searchsorted(POSKEY, key).clamp(max=len(POSKEY) - 1)
        m = POSKEY[pos] == key
        if not bool(m.any()): break
        n = torch.where(m, torch.randint(0, ni, (len(u),), device=DEV, generator=g), n)
    return n

def train_emb(X, Va, va_users, nu, ni, lightgcn, dim=64, layers=3, lr=None, reg=1e-4, bs=None):
    lr = lr or LR; bs = bs or BS
    g = torch.Generator(device=DEV); g.manual_seed(SEED)
    U = torch.nn.Embedding(nu, dim).to(DEV); I = torch.nn.Embedding(ni, dim).to(DEV)
    torch.nn.init.normal_(U.weight, std=.1); torch.nn.init.normal_(I.weight, std=.1)
    opt = torch.optim.Adam(list(U.parameters()) + list(I.parameters()), lr=lr)
    coo = X.tocoo()
    pu = torch.as_tensor(coo.row, device=DEV, dtype=torch.long); pi = torch.as_tensor(coo.col, device=DEV, dtype=torch.long)
    POSKEY = torch.sort(pu * ni + pi).values
    if lightgcn: At = norm_adj(X)
    def full():
        if not lightgcn: return U.weight, I.weight
        E = torch.cat([U.weight, I.weight]); acc = E
        for _ in range(layers): E = At @ E; acc = acc + E
        E = acc / (layers + 1)
        return E[:nu], E[nu:]
    best, best_state, bad, log, t_start = -1, None, 0, [], time.time()
    n_pos = len(pu)
    for ep in range(EPOCHS):
        t0 = time.time(); perm = torch.randperm(n_pos, device=DEV, generator=g); tot = torch.zeros((), device=DEV)
        for s in range(0, n_pos, bs):
            b = perm[s:s + bs]; u = pu[b]; p = pi[b]
            n = sample_neg(u, ni, POSKEY, g)
            Eu, Ei = full()
            eu, ep_, en = Eu[u], Ei[p], Ei[n]
            loss = -torch.nn.functional.logsigmoid((eu * ep_).sum(1) - (eu * en).sum(1)).mean()
            r = (U.weight[u].norm(2).pow(2) + I.weight[p].norm(2).pow(2) + I.weight[n].norm(2).pow(2)) / len(b)
            loss = loss + reg * r
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.detach()
        if (ep + 1) % 5 == 0:
            with torch.no_grad():
                Eu, Ei = full(); Ue, Ie = Eu.detach().clone(), Ei.detach().clone()
                vm, _, _ = evaluate(lambda bb: Ue[torch.as_tensor(bb, device=DEV)] @ Ie.T, X, Va, va_users)
            log.append((ep + 1, round(float(tot), 2), round(vm['NDCG@20'], 4), round(time.time() - t0, 2)))
            print('   ', 'LightGCN' if lightgcn else 'BPR-MF', log[-1], flush=True)
            if vm['NDCG@20'] > best: best, bad, best_state = vm['NDCG@20'], 0, (Ue, Ie)
            else:
                bad += 1
                if bad >= PATIENCE: break
            if MAX_MIN and time.time() - t_start > MAX_MIN * 60: print('    MAX_MIN cap hit', flush=True); break
    Ue, Ie = best_state
    return (lambda bb: Ue[torch.as_tensor(bb, device=DEV)] @ Ie.T), log

# ---------- SimGCL (SELFRec port) ----------
def train_simgcl(X, Va, va_users, nu, ni, dim=64, n_layers=2, cl_rate=0.5, eps=0.1, tau=0.2, reg=1e-4, lr=None, bs=None, max_epoch=None):
    lr = lr or LR; bs = int(os.environ.get('BS_SIMGCL', '2048')) if bs is None else bs
    max_epoch = max_epoch or int(os.environ.get('SIMGCL_EPOCHS', '20'))   # SELFRec conf: max.epoch 20
    g = torch.Generator(device=DEV); g.manual_seed(SEED)
    Ue = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(nu, dim)).to(DEV))
    Ie = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(ni, dim)).to(DEV))
    opt = torch.optim.Adam([Ue, Ie], lr=lr)
    At = norm_adj(X)
    coo = X.tocoo()
    pu = torch.as_tensor(coo.row, device=DEV, dtype=torch.long); pi = torch.as_tensor(coo.col, device=DEV, dtype=torch.long)
    POSKEY = torch.sort(pu * ni + pi).values
    def fwd(perturbed=False):
        ego = torch.cat([Ue, Ie], 0); alls = []
        for _ in range(n_layers):
            ego = At @ ego
            if perturbed:
                noise = torch.rand(ego.shape, device=DEV, generator=g)
                ego = ego + torch.sign(ego) * F.normalize(noise, dim=-1) * eps
            alls.append(ego)
        E = torch.stack(alls, 1).mean(1)
        return E[:nu], E[nu:]
    def infonce(v1, v2):
        v1, v2 = F.normalize(v1, dim=1), F.normalize(v2, dim=1)
        return -torch.diag(F.log_softmax((v1 @ v2.T) / tau, dim=1)).mean()
    best, best_state, bad, log, t_start = -1, None, 0, [], time.time(); n_pos = len(pu)
    for ep in range(max_epoch):
        t0 = time.time(); perm = torch.randperm(n_pos, device=DEV, generator=g); tot = torch.zeros((), device=DEV)
        for s in range(0, n_pos, bs):
            b = perm[s:s + bs]; u = pu[b]; p = pi[b]; n = sample_neg(u, ni, POSKEY, g)
            ru, ri = fwd()
            eu, ep_, en = ru[u], ri[p], ri[n]
            rec = torch.mean(-torch.log(10e-6 + torch.sigmoid((eu * ep_).sum(1) - (eu * en).sum(1))))
            l2 = reg * (torch.norm(eu, p=2) / eu.shape[0] + torch.norm(ep_, p=2) / ep_.shape[0])
            uu, ii = torch.unique(u), torch.unique(p)
            u1, i1 = fwd(True); u2, i2 = fwd(True)
            cl = cl_rate * (infonce(u1[uu], u2[uu]) + infonce(i1[ii], i2[ii]))
            loss = rec + l2 + cl
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.detach()
        with torch.no_grad():   # SELFRec evaluates every epoch
            ru, ri = fwd(); Uc, Ic = ru.detach().clone(), ri.detach().clone()
            vm, _, _ = evaluate(lambda bb: Uc[torch.as_tensor(bb, device=DEV)] @ Ic.T, X, Va, va_users)
        log.append((ep + 1, round(float(tot), 2), round(vm['NDCG@20'], 4), round(time.time() - t0, 2)))
        print('    SimGCL', log[-1], flush=True)
        if vm['NDCG@20'] > best: best, bad, best_state = vm['NDCG@20'], 0, (Uc, Ic)
        else:
            bad += 1
            if bad >= PATIENCE: break
        if MAX_MIN and time.time() - t_start > MAX_MIN * 60: print('    MAX_MIN cap hit', flush=True); break
    Uc, Ic = best_state
    return (lambda bb: Uc[torch.as_tensor(bb, device=DEV)] @ Ic.T), log

# ---------- UltraGCN (official main.py port) ----------
ULTRA_CFG = {  # from reczoo/RecZoo matching/gnn/UltraGCN/config/*.ini
    'ml-1m':   dict(w1=1e-7, w2=1., w3=1e-7, w4=1., neg=200, neg_w=200., gamma=1e-4, lam=1e-3, init=1e-3, bs=1024, ii_k=10),   # movielens1m_m1
    'gowalla': dict(w1=1e-6, w2=1., w3=1e-6, w4=1., neg=1500, neg_w=300., gamma=1e-4, lam=5e-4, init=1e-4, bs=512, ii_k=10),   # gowalla_m1
    'amazon':  dict(w1=1e-8, w2=1., w3=1., w4=1e-8, neg=500, neg_w=500., gamma=1e-4, lam=2.75, init=1e-4, bs=1024, ii_k=10),   # amazonbooks_m1
}
def train_ultragcn(X, Va, va_users, nu, ni, dim=64, lr=None):
    c = ULTRA_CFG[os.environ.get('ULTRA_CFG', 'amazon' if DATASET.startswith('amz') else DATASET)]
    lr = lr or LR; bs = int(os.environ.get('BS_ULTRA', c['bs']))
    g = torch.Generator(device=DEV); g.manual_seed(SEED)
    Ue = torch.nn.Parameter(torch.empty(nu, dim, device=DEV).normal_(0, c['init'], generator=g))
    Ie = torch.nn.Parameter(torch.empty(ni, dim, device=DEV).normal_(0, c['init'], generator=g))
    opt = torch.optim.Adam([Ue, Ie], lr=lr)
    du = np.asarray(X.sum(1)).ravel(); di = np.asarray(X.sum(0)).ravel()
    beta_u = torch.as_tensor(np.sqrt(du + 1) / du, device=DEV, dtype=torch.float32)
    beta_i = torch.as_tensor(1 / np.sqrt(di + 1), device=DEV, dtype=torch.float32)
    # item-item constraint matrix: G = X^T X (diagonal kept, as in official default ii_diagonal_zero=False)
    Gs = (X.T @ X).tocsr(); gD = np.asarray(Gs.sum(0)).ravel()
    gb_u = torch.as_tensor(np.sqrt(gD + 1) / gD, device=DEV, dtype=torch.float32)
    gb_i = torch.as_tensor(1 / np.sqrt(gD + 1), device=DEV, dtype=torch.float32)
    nbr = torch.zeros(ni, c['ii_k'], dtype=torch.long, device=DEV); sim = torch.zeros(ni, c['ii_k'], device=DEV)
    for s in range(0, ni, 2048):
        rows = torch.as_tensor(Gs[s:s + 2048].toarray(), device=DEV)
        w = gb_u[s:s + 2048, None] * gb_i[None, :] * rows
        v, ix = torch.topk(w, c['ii_k'], dim=1); nbr[s:s + 2048] = ix; sim[s:s + 2048] = v
    del Gs
    coo = X.tocoo()
    pu = torch.as_tensor(coo.row, device=DEV, dtype=torch.long); pi = torch.as_tensor(coo.col, device=DEV, dtype=torch.long)
    best, best_state, bad, log, t_start = -1, None, 0, [], time.time(); n_pos = len(pu)
    for ep in range(EPOCHS):
        t0 = time.time(); perm = torch.randperm(n_pos, device=DEV, generator=g); tot = torch.zeros((), device=DEV)
        for s in range(0, n_pos, bs):
            b = perm[s:s + bs]; u = pu[b]; p = pi[b]
            n = torch.randint(0, ni, (len(b), c['neg']), device=DEV, generator=g)   # uniform, sampling_sift_pos=no
            eu = Ue[u]
            pos_s = (eu * Ie[p]).sum(-1); neg_s = (eu.unsqueeze(1) * Ie[n]).sum(-1)
            pw = c['w1'] + c['w2'] * beta_u[u] * beta_i[p]
            nw = c['w3'] + c['w4'] * beta_u[u].unsqueeze(1) * beta_i[n]
            neg_l = F.binary_cross_entropy_with_logits(neg_s, torch.zeros_like(neg_s), weight=nw, reduction='none').mean(-1)
            pos_l = F.binary_cross_entropy_with_logits(pos_s, torch.ones_like(pos_s), weight=pw, reduction='none')
            loss = (pos_l + neg_l * c['neg_w']).sum()
            loss = loss + c['gamma'] * ((Ue ** 2).sum() + (Ie ** 2).sum()) / 2
            li = -sim[p] * (eu.unsqueeze(1) * Ie[nbr[p]]).sum(-1).sigmoid().log()
            loss = loss + c['lam'] * li.sum()
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.detach()
        if (ep + 1) % 5 == 0:
            with torch.no_grad():
                Uc, Ic = Ue.detach().clone(), Ie.detach().clone()
                vm, _, _ = evaluate(lambda bb: Uc[torch.as_tensor(bb, device=DEV)] @ Ic.T, X, Va, va_users)
            log.append((ep + 1, round(float(tot), 2), round(vm['NDCG@20'], 4), round(time.time() - t0, 2)))
            print('    UltraGCN', log[-1], flush=True)
            if vm['NDCG@20'] > best: best, bad, best_state = vm['NDCG@20'], 0, (Uc, Ic)
            else:
                bad += 1
                if bad >= PATIENCE: break
            if MAX_MIN and time.time() - t_start > MAX_MIN * 60: print('    MAX_MIN cap hit', flush=True); break
    Uc, Ic = best_state
    return (lambda bb: Uc[torch.as_tensor(bb, device=DEV)] @ Ic.T), log

# ---------- run ----------
results = {'dataset': DATASET, 'seed': SEED, 'device': str(DEV), 'lr': LR, 'bs': BS, 'version': 'v4'}
tag = os.environ.get('TAG', f'{DATASET}_s{SEED}')
for proto, fn in (('random', split_random), ('loo', split_loo), ('global_temporal', split_global)):
    if proto not in PROTOS: continue
    tr, va, te = fn(df)
    info = leak_stats(tr, te)
    (tu, ti), (vu, vi), (eu, ei), nu, ni, cold, nall = prep(tr, va, te)
    info.update(cold); info.update(train=len(tu), val=len(vu), test=len(eu), users=nu, items=ni)
    X = csr(tu, ti, nu, ni); Xtv = X + csr(vu, vi, nu, ni); Xtv.data[:] = 1
    Va = csr(vu, vi, nu, ni); Va.data[:] = 1; Te = csr(eu, ei, nu, ni); Te.data[:] = 1
    va_users = np.flatnonzero(np.diff(Va.indptr)); te_users = np.flatnonzero(np.diff(Te.indptr))
    info['test_users'] = int(len(te_users)); info['test_users_warm'] = int(len(te_users))
    ev_users = np.flatnonzero(nall > 0)   # superset of te_users
    print(proto, info, flush=True)
    res = {'info': info}
    for name in MODELS:
        if DEV.type == 'cuda': torch.cuda.synchronize()
        t0 = time.time(); log = None
        if name == 'MostPop': sf = m_pop(X)
        elif name == 'ItemKNN': sf = m_itemknn(X, k=int(os.environ.get('KNN_K', '100')))
        elif name == 'EASE': sf = m_ease(X, lam=float(os.environ.get('EASE_LAM', '500')))
        elif name == 'SimGCL': sf, log = train_simgcl(X, Va, va_users, nu, ni)
        elif name == 'UltraGCN': sf, log = train_ultragcn(X, Va, va_users, nu, ni)
        else: sf, log = train_emb(X, Va, va_users, nu, ni, lightgcn=(name == 'LightGCN'))
        if log is None:   # closed-form / non-learned models: validation score for hyper-parameter selection (round 3)
            vm, _, _ = evaluate(sf, X, Va, va_users); vbase = vm['NDCG@20']
        _, pw, pwu = evaluate(sf, Xtv, Te, ev_users, nall)
        m = {k: float(np.nanmean(v)) for k, v in pw.items()}                       # warm (== r2 definition)
        m.update({f'full_{k}': float(v.sum() / info['test_users_all']) for k, v in pwu.items()})   # all test users, cold = 0
        m.update({f'wu_{k}': float(v.mean()) for k, v in pwu.items()})              # warm users, all their items
        m['sec'] = round(time.time() - t0, 1)
        if log: m['val_log'] = log
        else: m['val_NDCG@20'] = vbase; m['params'] = {'KNN_K': os.environ.get('KNN_K', '100'), 'EASE_LAM': os.environ.get('EASE_LAM', '500')}
        res[name] = m
        np.savez_compressed(f'{OUT}/peruser_{tag}_{proto}_{name}.npz', users=ev_users.astype(np.int32),
                            **{f'warm_{k}': v.astype(np.float32) for k, v in pw.items()},
                            **{f'wu_{k}': v.astype(np.float32) for k, v in pwu.items()}, nall=nall[ev_users].astype(np.int32))
        results[proto] = res
        json.dump(results, open(f'{OUT}/res_{tag}.json', 'w'), indent=1)
        print(' ', name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items() if k != 'val_log'}, flush=True)
        del sf
        if DEV.type == 'cuda': torch.cuda.empty_cache()
print('DONE')
