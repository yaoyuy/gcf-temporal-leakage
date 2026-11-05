"""Convert raw public files to u,i,t csv (implicit, dedup user-item keeping earliest time).
Amazon Reviews'23 5-core rating_only (McAuley Lab): used as released (already 5-core + dedup).
Gowalla (SNAP loc-gowalla_totalCheckins): dedup user-location keeping first check-in, then iterative K-core."""
import pandas as pd, numpy as np, sys
def kcore(d, k):
    while True:
        uc = d.u.map(d.u.value_counts()); ic = d.i.map(d.i.value_counts())
        keep = (uc >= k) & (ic >= k)
        if keep.all(): return d
        d = d[keep]
def stats(n, d): print(n, len(d), d.u.nunique(), d.i.nunique(), 'density %.5f' % (len(d)/d.u.nunique()/d.i.nunique()), flush=True)
ONLY = sys.argv[2] if len(sys.argv) > 2 else 'all'   # 'gowalla' re-prepares Gowalla only
for c, short in (() if ONLY == 'gowalla' else (('Video_Games','amz-games'), ('Musical_Instruments','amz-music'))):
    d = pd.read_csv(f'raw/{c}.csv.gz').rename(columns={'user_id':'u','parent_asin':'i','timestamp':'t'})[['u','i','t']]
    d = d.sort_values('t', kind='stable').drop_duplicates(['u','i'])
    stats(short, d); d.to_csv(f'{short}.csv', index=False)
K = int(sys.argv[1]) if len(sys.argv) > 1 else 15
g = pd.read_csv('raw/gowalla.txt.gz', sep='\t', names=['u','t','lat','lon','i'], usecols=['u','t','i'])
# FIX 2026-10-25 (round 3): the old line `pd.to_datetime(g.t).astype('int64') // 10**9` assumed ns resolution; with
# pandas >= 2 the parsed column can carry a coarser unit, which left t in units of 1000 s. Cast explicitly to seconds.
g['t'] = pd.to_datetime(g.t, utc=True).astype('datetime64[s, UTC]').astype('int64')
g = g.sort_values('t', kind='stable').drop_duplicates(['u','i'])
stats('gowalla-dedup', g)
g = kcore(g, K); stats(f'gowalla-{K}core', g); g.to_csv('gowalla.csv', index=False)
