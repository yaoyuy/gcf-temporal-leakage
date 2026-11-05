# Post-hoc users-only population alignment of round 3 (no new training)

Round-3 per-user warm NDCG@20 for random / LOO restricted to users that are warm GT test users (in GT train with ≥1 warm GT test row). Items are NOT restricted (needs the aligned re-run). Split reconstruction: same code and seeds as run_v4.py.

## ml-1m (warm GT test users: 1008)

| model | random (all) | random (GT users) | loo (all) | loo (GT users) | GT | seeds (rnd/loo/GT) |
|---|---|---|---|---|---|---|
| MostPop | 0.1339 | 0.2202 | 0.0258 | 0.0300 | 0.2345 | 5/1/1 |
| ItemKNN | 0.2283 | 0.3003 | 0.0463 | 0.0362 | 0.2345 | 5/1/1 |
| EASE | 0.3210 | 0.4042 | 0.0626 | 0.0465 | 0.2550 | 5/1/1 |
| BPR-MF | 0.2565 | 0.3416 | 0.0529 | 0.0459 | 0.2578 | 5/5/5 |
| LightGCN | 0.2852 | 0.3693 | 0.0578 | 0.0508 | 0.2625 | 5/5/5 |
- τ(random, GT): all users 0.400, GT-warm users only 0.400; best EASE → EASE (GT best LightGCN)
- τ(loo, GT): all users 0.400, GT-warm users only 0.600; best EASE → LightGCN (GT best LightGCN)

## gowalla (warm GT test users: 6244)

| model | random (all) | random (GT users) | loo (all) | loo (GT users) | GT | seeds (rnd/loo/GT) |
|---|---|---|---|---|---|---|
| MostPop | 0.0273 | 0.0286 | 0.0192 | 0.0168 | 0.0265 | 5/1/1 |
| ItemKNN | 0.1434 | 0.1372 | 0.0522 | 0.0502 | 0.0586 | 5/1/1 |
| EASE | 0.1786 | 0.1754 | 0.0686 | 0.0659 | 0.0772 | 5/1/1 |
| BPR-MF | 0.1188 | 0.1156 | 0.0540 | 0.0507 | 0.0663 | 5/5/5 |
| LightGCN | 0.1533 | 0.1514 | 0.0647 | 0.0615 | 0.0772 | 5/5/5 |
- τ(random, GT): all users 0.600, GT-warm users only 0.600; best EASE → EASE (GT best LightGCN)
- τ(loo, GT): all users 0.800, GT-warm users only 0.800; best EASE → EASE (GT best LightGCN)

## amz-music (warm GT test users: 10726)

| model | random (all) | random (GT users) | loo (all) | loo (GT users) | GT | seeds (rnd/loo/GT) |
|---|---|---|---|---|---|---|
| MostPop | 0.0221 | 0.0188 | 0.0168 | 0.0105 | 0.0135 | 5/1/1 |
| ItemKNN | 0.0206 | 0.0122 | 0.0109 | 0.0050 | 0.0046 | 5/1/1 |
| EASE | 0.0494 | 0.0339 | 0.0304 | 0.0165 | 0.0149 | 5/1/1 |
| BPR-MF | 0.0135 | 0.0102 | 0.0103 | 0.0053 | 0.0057 | 4/4/3 |
| LightGCN | 0.0436 | 0.0316 | 0.0315 | 0.0166 | 0.0166 | 4/4/3 |
- τ(random, GT): all users 0.600, GT-warm users only 0.600; best EASE → EASE (GT best LightGCN)
- τ(loo, GT): all users 0.800, GT-warm users only 1.000; best LightGCN → LightGCN (GT best LightGCN)

## amz-games (warm GT test users: 8895)

| model | random (all) | random (GT users) | loo (all) | loo (GT users) | GT | seeds (rnd/loo/GT) |
|---|---|---|---|---|---|---|
| MostPop | 0.0186 | 0.0164 | 0.0159 | 0.0070 | 0.0114 | 5/1/1 |
| ItemKNN | 0.0596 | 0.0311 | 0.0259 | 0.0059 | 0.0060 | 5/1/1 |
| EASE | 0.0845 | 0.0535 | 0.0414 | 0.0138 | 0.0113 | 5/1/1 |
| BPR-MF | 0.0384 | 0.0188 | 0.0206 | 0.0057 | 0.0057 | 5/5/5 |
| LightGCN | 0.0637 | 0.0375 | 0.0376 | 0.0144 | 0.0161 | 5/5/4 |
- τ(random, GT): all users 0.200, GT-warm users only 0.200; best EASE → EASE (GT best LightGCN)
- τ(loo, GT): all users 0.200, GT-warm users only 0.800; best EASE → LightGCN (GT best LightGCN)

