# Graph CF under global temporal splits: leakage, cold starts and leaderboard instability

Code and aggregated results for the manuscript "Revisiting Graph Collaborative Filtering under Global Temporal Splits: Leakage, Cold Starts and Leaderboard Instability" (Xiangyu He, Mianyang City College; manuscript under review).

## Contents
- `code/run_v4.py` – splits (random 80/10/10, leave-one-out, global temporal 80/10/10), models (MostPop, ItemKNN, EASE, BPR-MF, LightGCN, SimGCL; UltraGCN implemented but not run), full-ranking evaluation (warm and full populations), per-user arrays.
- `code/driver_v4.py` – hyperparameter grids and seed loop used for the reported runs (Kaggle notebooks, Tesla T4 GPU / CPU sessions).
- `code/summarize_r3.py` – aggregation, Kendall's tau, paired Wilcoxon signed-rank tests with Holm correction.
- `code/run_v5.py` – supplementary experiments (rounds 4–5): controlled leakage (E1), leakage channels (E2), refit, population alignment, split-then-filter and time-aware variants (E3); model code identical to `run_v4.py`.
- `code/driver_v5.py` (round 4) and `code/driver_v6.py` (round 5) – grids, seeds and run modes for those experiments.
- `code/summarize_r4.py`, `code/summarize_r5.py`, `code/e1a_decompose_r4.py`, `code/gt_extgrid_wilcoxon_r4.py` – aggregation and statistics for rounds 4–5.
- `data/prep_datasets.py` – preprocessing of MovieLens-1M, Amazon Reviews'23 (Video Games, Musical Instruments; 5-core) and Gowalla (15-core); `md5_prepared.txt` gives the MD5 of each prepared file.
- `results/r3/` – per-run summary (`summary.csv`) and statistics (`stats.md`).
- `results/r4/` – aggregated results of the controlled leakage (E1), leakage-channel (E2), refit, population-alignment and split-then-filter/time-aware (E3) experiments: `summary_r4.csv`, `stats_r4.md/.json` (from `exp/summarize_r4.py`), `gt_extgrid.*` (GT with the extended EASE/ItemKNN grid), `e1a_decompose.*`, `posthoc_r3_aligned.md`. Aggregates only; no raw data.
- `results/r5/` – round-5 aggregates merged with round 4 (`summary_r45.csv`, `stats_r45.md/.json`, `gt_extgrid.md`, `e1a_decompose.md`); these are the numbers reported in the paper.
- `analysis/`, `figures/` – scripts that produce the tables and figures of the paper.

## Data
All datasets are public and are not redistributed here: MovieLens-1M (GroupLens), Amazon Reviews'23 (McAuley Lab), Gowalla (SNAP). Run `data/prep_datasets.py` on the raw files and check the MD5 hashes.

## Reproduction
```
python code/driver_v4.py <dataset> <data_dir> basetune   # MostPop, ItemKNN k grid, EASE lambda grid
python code/driver_v4.py <dataset> <data_dir> rerun      # BPR-MF, LightGCN
python code/driver_v4.py <dataset> <data_dir> new        # SimGCL
python code/summarize_r3.py
python code/driver_v6.py <dataset> <data_dir> <mode>   # round-5 modes: e1plan, stfx, alignedcpu_ext
python code/summarize_r5.py
```
Aggregation scripts read raw run outputs from `$GCF_ROOT/exp/out/r2..r5` (set `GCF_ROOT`, default `.`).
Requirements: Python 3.13, PyTorch 2.11, NumPy, SciPy, pandas, Matplotlib.

## License
Code: MIT (see `LICENSE`). Datasets keep their original licenses and are not redistributed.
