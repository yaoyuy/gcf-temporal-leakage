# E1 condition a vs full GT: decomposition (exp/e1a_decompose.py)

| dataset | population | n users | LightGCN | EASE | LightGCN−EASE | Wilcoxon p (raw) |
|---|---|---|---|---|---|---|
| ml-1m | full GT, all warm users | 1008 | 0.2625 | 0.2586 | +0.0038 | 0.424 |
| ml-1m | full GT, E users only | 504 | 0.2651 | 0.2613 | +0.0038 | 0.453 |
| ml-1m | E1-a (E users, H items, F masked) | 504 | 0.1749 | 0.1763 | -0.0014 | 0.85 |
| gowalla | full GT, all warm users | 6244 | 0.0772 | 0.0772 | +0.0000 | 0.58 |
| gowalla | full GT, E users only | 3122 | 0.0774 | 0.0782 | -0.0008 | 0.992 |
| gowalla | E1-a (E users, H items, F masked) | 3122 | 0.0646 | 0.0653 | -0.0008 | 0.705 |
| amz-music | full GT, all warm users | 10726 | 0.0166 | 0.0151 | +0.0016 | 2.62e-05 |
| amz-music | full GT, E users only | 5363 | 0.0155 | 0.0142 | +0.0013 | 0.00718 |
| amz-music | E1-a (E users, H items, F masked) | 5363 | 0.0148 | 0.0126 | +0.0022 | 4.3e-05 |
| amz-games | full GT, all warm users | 8895 | 0.0161 | 0.0113 | +0.0048 | 8.74e-09 |
| amz-games | full GT, E users only | 4447 | 0.0176 | 0.0119 | +0.0056 | 2.22e-06 |
| amz-games | E1-a (E users, H items, F masked) | 4447 | 0.0167 | 0.0111 | +0.0055 | 9.02e-06 |
