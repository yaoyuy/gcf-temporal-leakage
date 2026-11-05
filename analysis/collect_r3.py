"""Round-3 inputs for tables/figures (no hand-typed numbers).
Reads exp/out/r3: res_*.json (split diagnostics 'info' per dataset x protocol x seed), stats.md (Wilcoxon-Holm table),
queue_state.json (Kaggle session seconds per kernel = last log timestamp), *_log.txt (first line: device).
Writes info_r3.json, wilcoxon_r3.json, compute_r3.json."""
import os
ROOT = os.environ.get('GCF_ROOT', '.')  # project root containing exp/out/r2..r5 and paper/
import json, glob, os, re, collections
R = os.path.join(ROOT, 'exp/out/r3')
info = collections.defaultdict(dict)
for p in sorted(glob.glob(R + '/gcf-r3-*/**/*res_*.json', recursive=True)):
    if 'smoke' in p: continue
    d = json.load(open(p)); ds, s = d.get('dataset'), d.get('seed')
    for pr, dd in d.items():
        if isinstance(dd, dict) and 'info' in dd:
            if s in info[(ds, pr)]: assert info[(ds, pr)][s] == dd['info'], p
            info[(ds, pr)][s] = dd['info']
json.dump({f'{a}|{b}': {str(s): v[s] for s in sorted(v)} for (a, b), v in sorted(info.items())}, open('info_r3.json', 'w'), indent=1)
W = []
for line in open(R + '/stats.md'):
    c = [x.strip() for x in line.strip().strip('|').split('|')]
    if len(c) == 10 and c[0] in ('ml-1m', 'amz-games', 'amz-music', 'gowalla') and c[2] in ('warm', 'full'):
        W.append(dict(dataset=c[0], protocol=c[1], pop=c[2], best=c[3], vs=c[4], diff=float(c[5]), n=int(c[6]),
                      p_raw=float(c[7]), p_holm=float(c[8]), sig=c[9] == 'yes'))
json.dump(W, open('wilcoxon_r3.json', 'w'), indent=0)
q = json.load(open(R + '/queue_state.json')); comp = {}
for k, v in q.items():
    first = open(f'{R}/gcf-r3-{k}_log.txt').readline()
    comp[k] = dict(seconds=v['log_seconds'], status=v['final_status'], gpu=first.startswith('cuda True'), first_line=first.strip())
gpu = sum(v['seconds'] for k, v in comp.items() if v['gpu'] and k != 'smoke'); cpu = sum(v['seconds'] for k, v in comp.items() if not v['gpu'] and k != 'smokecpu')
comp['_totals'] = dict(gpu_s_results=gpu, gpu_h_results=round(gpu / 3600, 2), gpu_s_smoke=comp['smoke']['seconds'],
                       gpu_h_incl_smoke=round((gpu + comp['smoke']['seconds']) / 3600, 2), cpu_s_results=cpu, cpu_h_results=round(cpu / 3600, 2))
json.dump(comp, open('compute_r3.json', 'w'), indent=1)
print(comp['_totals']); print(len(W), 'wilcoxon rows'); print({k: sorted(v) for k, v in info.items()})
