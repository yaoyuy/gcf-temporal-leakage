"""Build all paper figures in two styles from real round-3 results only (round-2 version: make_figs_r2.py).
Style 'nat' : nature-figure skill guidance (7 pt Helvetica-clone, open L-spines, NMI pastel family,
              direct labels, minimal ink, 183 mm wide).
Style 'ai'  : Applied Intelligence / Springer artwork rules (Helvetica-clone 8-9 pt at final size,
              174 mm double / 84 mm single column, boxed axes with light y-grid as in recent OA
              Applied Intelligence recommender papers, colour + hatch/marker redundancy for greyscale).
Data sources: exp/out/r3/summary.csv (means/SD over seeds; warm population unless stated) and
paper/analysis/info_r3.json (split diagnostics written by run_v4.py, seeds 1-5; built by analysis/collect_r3.py).
"""
import os
ROOT = os.environ.get('GCF_ROOT', '.')  # project root containing exp/out/r2..r5 and paper/
import json, sys, os
import numpy as np, pandas as pd, matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import kendalltau

ROOT = ROOT
OUT = f'{ROOT}/paper/figs'
S = pd.read_csv(f'{ROOT}/exp/out/r3/summary.csv')
INFO = json.load(open(f'{ROOT}/paper/analysis/info_r3.json'))
COLMAP = {'ndcg20_mean': 'warm_NDCG@20_mean', 'ndcg20_std': 'warm_NDCG@20_std', 'recall20_mean': 'warm_Recall@20_mean', 'recall20_std': 'warm_Recall@20_std'}
DS = ['ml-1m', 'amz-games', 'amz-music', 'gowalla']
DSN = {'ml-1m': 'MovieLens-1M', 'amz-games': 'Video Games', 'amz-music': 'Musical Instruments', 'gowalla': 'Gowalla'}
DSS = ['ML-1M', 'Games', 'Music', 'Gowalla']
PTK = ['Random', 'LOO', 'Temporal']
PR = ['random', 'loo', 'global_temporal']
PRN = {'random': 'Random', 'loo': 'LOO', 'global_temporal': 'Global temporal'}
M = ['MostPop', 'ItemKNN', 'EASE', 'BPR-MF', 'LightGCN', 'SimGCL']
MM = 25.4

def val(ds, pr, mo, col):
    r = S[(S.dataset == ds) & (S.protocol == pr) & (S.model == mo)]
    return float(r[COLMAP.get(col, col)].iloc[0])

# ---------------- styles ----------------
NAT = dict(nat=True, font=7, label=8, width=183, colors={
    'MostPop': '#D8D8D8', 'ItemKNN': '#B4C0E4', 'EASE': '#7884B4', 'BPR-MF': '#484878', 'LightGCN': '#E4848C', 'SimGCL': '#F2C4A0'},
    hatch={m: '' for m in M}, marker={m: 'o' for m in M})
AI = dict(font=8.5, label=9.5, width=174, colors={
    'MostPop': '#BDBDBD', 'ItemKNN': '#6BAED6', 'EASE': '#08519C', 'BPR-MF': '#FDAE6B', 'LightGCN': '#D94801', 'SimGCL': '#7F2704'},
    hatch={'MostPop': '', 'ItemKNN': '////', 'EASE': '', 'BPR-MF': '\\\\\\\\', 'LightGCN': 'xxx', 'SimGCL': '...'},
    marker={'MostPop': 'o', 'ItemKNN': 's', 'EASE': 'D', 'BPR-MF': 'v', 'LightGCN': '^', 'SimGCL': 'P'})

def setup(st):
    mpl.rcdefaults()
    base = {'font.family': 'sans-serif', 'font.sans-serif': ['Nimbus Sans', 'Arial', 'Helvetica', 'DejaVu Sans'],
            'pdf.fonttype': 42, 'svg.fonttype': 'none', 'font.size': st['font'],
            'axes.labelsize': st['font'], 'xtick.labelsize': st['font'], 'ytick.labelsize': st['font'],
            'legend.fontsize': st['font'], 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
            'lines.linewidth': 1.0, 'hatch.linewidth': 0.5}
    if st.get('nat', False):
        base.update({'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False})
    else:
        base.update({'axes.grid': True, 'axes.grid.axis': 'y', 'grid.color': '#D0D0D0', 'grid.linewidth': 0.5,
                     'axes.axisbelow': True, 'legend.frameon': True, 'legend.edgecolor': '#B0B0B0', 'legend.framealpha': 1})
    mpl.rcParams.update(base)

def plabel(ax, s, st):
    ax.text(-0.02, 1.04, s, transform=ax.transAxes, fontsize=st['label'], fontweight='bold', ha='right', va='bottom')

def save(fig, name):
    fig.savefig(f'{name}.pdf', bbox_inches='tight', pad_inches=0.02)
    fig.savefig(f'{name}.png', dpi=300, bbox_inches='tight', pad_inches=0.02)
    if '/paper/Fig' in name: fig.savefig(f'{name}.eps', bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)

# ---------------- Fig: NDCG@20 per protocol ----------------
def fig_ndcg(st, name):
    setup(st)
    w = st['width'] / MM
    narrow = st['width'] < 150
    fig, axes = plt.subplots(2, 2, figsize=(w, w * 0.62)) if narrow else plt.subplots(1, 4, figsize=(w, w * 0.30))
    axes = np.ravel(axes)
    bw = 0.135
    for k, (ax, ds) in enumerate(zip(axes, DS)):
        for j, mo in enumerate(M):
            x = np.arange(3) + (j - (len(M) - 1) / 2) * bw
            y = [val(ds, p, mo, 'ndcg20_mean') for p in PR]; e = [val(ds, p, mo, 'ndcg20_std') for p in PR]
            ax.bar(x, y, bw, yerr=e, color=st['colors'][mo], hatch=st['hatch'][mo], edgecolor='#333333' if (not st.get('nat', False)) else 'none',
                   linewidth=0.4, error_kw=dict(lw=0.5, capsize=1.2, capthick=0.5), label=mo)
        ax.set_xticks(range(3)); ax.set_xticklabels(PTK)
        if (not st.get('nat', False)):
            ax.set_title(f'({chr(97+k)}) {DSN[ds]}', fontsize=st['font'], pad=3); ax.tick_params(axis='x', labelsize=st['font'])
            ax.set_xlim(-0.5, 2.5)
        else: plabel(ax, chr(97 + k), st); ax.set_title(DSN[ds], fontsize=st['font'], pad=3)
        if k == 0 or (narrow and k == 2): ax.set_ylabel('NDCG@20')
        ax.tick_params(axis='x', length=0)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=6, loc='lower center', bbox_to_anchor=(0.5, 1.0), handlelength=1.4, columnspacing=0.8 if narrow else 1.2)
    fig.tight_layout(w_pad=0.8, h_pad=0.8)
    save(fig, name)

# ---------------- Fig: bump chart of ranks ----------------
def ranks(ds, col='ndcg20_mean'):
    out = {}
    for p in PR:
        v = {m: val(ds, p, m, col) for m in M}
        order = sorted(M, key=lambda m: -v[m])
        for r, m in enumerate(order): out[(p, m)] = r + 1
    return out

def fig_bump(st, name):
    setup(st)
    w = st['width'] / MM
    nat = st.get('nat', False)
    fig, axes = plt.subplots(2, 2, figsize=(w, w * (0.62 if st['width'] < 150 else 0.44)), sharey=True) if nat else plt.subplots(1, 4, figsize=(w, w * 0.27), sharey=True)
    axes = axes.ravel()
    XS = [0, 1.25, 2.5]
    for k, (ax, ds) in enumerate(zip(axes, DS)):
        R = ranks(ds)
        for mo in M:
            y = [R[(p, mo)] for p in PR]
            hero = mo in ('LightGCN', 'EASE')
            if st.get('nat', False):
                c = st['colors'][mo] if hero else '#A8A8A8'
                ax.plot(XS, y, '-o', color=c, lw=1.6 if hero else 0.8, ms=3.2 if hero else 2.4, zorder=3 if hero else 2)
                ax.text(XS[-1] + 0.12, y[-1], mo, va='center', fontsize=st['font'], color='#272727' if hero else '#767676')
            else:
                ls = '-' if hero else '--'
                ax.plot(XS, y, ls, marker=st['marker'][mo], color=st['colors'][mo], lw=1.4 if hero else 0.9,
                        ms=4, mec='#333333', mew=0.4, label=mo, zorder=3 if hero else 2)
        ax.set_xticks(XS); ax.set_xticklabels(PTK)
        ax.set_xlim(-0.3, XS[-1] + 0.3 if (not st.get('nat', False)) else XS[-1] + 0.8); ax.set_ylim(len(M) + 0.5, 0.5); ax.set_yticks(range(1, len(M) + 1))
        if (not st.get('nat', False)):
            ax.set_title(f'({chr(97+k)}) {DSN[ds]}', fontsize=st['font'], pad=3); ax.grid(axis='x', visible=False)
        else:
            if st['width'] < 150:  # final manuscript version: same panel-label style as the other figures
                ax.set_title(f'({chr(97+k)}) {DSN[ds]}', fontsize=st['font'], pad=3)
            else:
                ax.set_title(f'{chr(97+k)}', fontsize=st['label'], fontweight='bold', loc='left', pad=3)
                ax.text(0.5, 1.02, DSN[ds], transform=ax.transAxes, ha='center', va='bottom', fontsize=st['font'])
            ax.spines['left'].set_visible(k % 2 == 0)
            if k % 2: ax.tick_params(axis='y', length=0)
        if k == 0 or (nat and k == 2): ax.set_ylabel('Rank by NDCG@20')
        ax.tick_params(axis='x', length=0)
    if (not st.get('nat', False)):
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, ncol=6, loc='lower center', bbox_to_anchor=(0.5, 1.0), handlelength=2.2, columnspacing=1.2)
    fig.tight_layout(w_pad=2.0 if nat else 0.8, h_pad=1.2)
    save(fig, name)

# ---------------- Fig: Kendall tau heatmap ----------------
def tau_mat(col):
    pairs = [('random', 'loo'), ('random', 'global_temporal'), ('loo', 'global_temporal')]
    T = np.zeros((len(DS), 3))
    for i, ds in enumerate(DS):
        for j, (a, b) in enumerate(pairs):
            T[i, j] = kendalltau([val(ds, a, m, col) for m in M], [val(ds, b, m, col) for m in M])[0]
    return T

def fig_tau(st, name):
    setup(st)
    single = (not st.get('nat', False))
    w = (84 if single else 120) / MM
    fig, axes = plt.subplots(1, 2, figsize=(w, w * (0.62 if single else 0.42)), sharey=True)
    cmap = 'Greys' if (not st.get('nat', False)) else mpl.colors.LinearSegmentedColormap.from_list('p', ['#E4CCD8', '#7884B4', '#484878'])
    for k, (ax, col, t) in enumerate(zip(axes, ['ndcg20_mean', 'recall20_mean'], ['NDCG@20', 'Recall@20'])):
        T = tau_mat(col)
        im = ax.imshow(T, cmap=cmap, vmin=0 if (not st.get('nat', False)) else 0.2, vmax=1, aspect='auto')
        for i in range(T.shape[0]):
            for j in range(3):
                dark = T[i, j] > (0.6 if (not st.get('nat', False)) else 0.7)
                ax.text(j, i, f'{T[i,j]:.2f}', ha='center', va='center', fontsize=st['font'], color='white' if dark else 'black')
        ax.set_xticks(range(3)); ax.set_xticklabels(['R–L', 'R–G', 'L–G'])
        ax.set_yticks(range(len(DS))); ax.set_yticklabels(['ML-1M', 'Games', 'Music', 'Gowalla'])
        ax.grid(False); ax.tick_params(length=0)
        for sp in ax.spines.values(): sp.set_visible((not st.get('nat', False)))
        if (not st.get('nat', False)): ax.set_title(f'({chr(97+k)}) {t}', fontsize=st['font'], pad=3)
        else: plabel(ax, chr(97 + k), st); ax.set_title(t, fontsize=st['font'], pad=3)
    fig.tight_layout(w_pad=0.6)
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.03, aspect=18)
    cb.set_label('Kendall τ', fontsize=st['font']); cb.outline.set_linewidth(0.5); cb.ax.tick_params(labelsize=st['font'] - 0.5, width=0.5)
    save(fig, name)

# ---------------- Fig: leakage and cold start ----------------
def info(ds, pr, key):
    v = INFO[f'{ds}|{pr}']; xs = [v[s][key] for s in sorted(v)]
    return float(np.mean(xs)), float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0

def fig_leak(st, name):
    setup(st)
    w = st['width'] / MM
    narrow = st['width'] < 150
    fig, axes = plt.subplots(1, 3, figsize=(w, w * (0.40 if narrow else 0.29)))
    pcol = {'random': '#7884B4', 'loo': '#B4C0E4', 'global_temporal': '#E4848C'} if st.get('nat', False) else \
           {'random': '#08519C', 'loo': '#9ECAE1', 'global_temporal': '#D94801'}
    phat = {'random': '', 'loo': '////', 'global_temporal': 'xxx'} if (not st.get('nat', False)) else {p: '' for p in PR}
    bw = 0.27; x = np.arange(len(DS))
    panels = [('train_after_user_first_test_pct', 'Leak (%)', 'random'),
              ('train_after_global_cut_pct', 'Future (%)', None),
              ('test_cold_rows_pct', 'Cold (%)', 'global_temporal')]
    for k, (ax, (key, yl, hl)) in enumerate(zip(axes, panels)):
        for j, p in enumerate(PR):
            m = [info(ds, p, key)[0] for ds in DS]; e = [info(ds, p, key)[1] for ds in DS]
            ax.bar(x + (j - 1) * bw, m, bw, yerr=e if p == 'random' else None, color=pcol[p], hatch=phat[p],
                   edgecolor='#333333' if (not st.get('nat', False)) else 'none', linewidth=0.4, label=PRN[p],
                   error_kw=dict(lw=0.5, capsize=1.2, capthick=0.5))
            for xi, mi in zip(x + (j - 1) * bw, m):
                if mi == 0 and st.get('nat', False):
                    ax.text(xi, 1.0, '0', ha='center', va='bottom', fontsize=st['font'] - 0.5, color='#555555')
                elif p == hl:
                    ax.text(xi, mi + 1.5, f'{mi:.1f}', ha='center', va='bottom', fontsize=st['font'] - 0.5)
        ax.set_xticks(x); ax.set_xticklabels(DSS, rotation=35 if narrow else 0, ha='right' if narrow else 'center', rotation_mode='anchor')
        ax.set_ylim(0, 30 if k == 1 else 112); ax.set_ylabel(yl); ax.tick_params(axis='x', length=0)
        if k != 1: ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_xlim(-0.8, len(DS) - 0.15)
        if (not st.get('nat', False)): ax.tick_params(axis='x', labelsize=st['font'])
        if (not st.get('nat', False)): ax.set_title(f'({chr(97+k)})', fontsize=st['font'], pad=3, loc='left')
        else: plabel(ax, chr(97 + k), st)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=3, loc='lower center', bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(w_pad=1.0)
    save(fig, name)

NAT8 = dict(NAT, font=8, label=9, width=131)
AI131 = dict(AI, width=131, font=8.6, label=9.5)

if __name__ == '__main__':
    os.makedirs(f'{OUT}/nat', exist_ok=True); os.makedirs(f'{OUT}/ai', exist_ok=True)
    for st, tag in ((NAT, 'nat'), (AI, 'ai')):
        fig_ndcg(st, f'{OUT}/{tag}/ndcg'); fig_bump(st, f'{OUT}/{tag}/bump')
        fig_tau(st, f'{OUT}/{tag}/tau'); fig_leak(st, f'{OUT}/{tag}/leak')
    # final (winner) versions for main.tex; see FIGURE_CHOICES.md
    # file numbers follow the printed figure order (Springer: name files Fig<number>)
    fig_leak(AI131, f'{ROOT}/paper/Fig1'); fig_ndcg(AI131, f'{ROOT}/paper/Fig2')
    fig_bump(NAT8, f'{ROOT}/paper/Fig3'); fig_tau(AI131, f'{ROOT}/paper/Fig4')
    print('ok')
