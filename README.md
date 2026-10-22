# sweet.tv — TV programme recommender (LLM4Rec midterm exam)

Top-5 TV programmes per user for weeks 31-42 of the Kaggle community competition
[sweet.tv — TV program recommender](https://www.kaggle.com/competitions/sweettv-tv-program-recommender) (MAP@5).
Individual work, HSE LLM4Rec course, AVITO group.

**Final submission:** `outputs/submissions/s03_hybrid.csv` — candidate generation + gradient-boosting re-ranker.
Kaggle (late submission, ref 57531540): **public 0.42177, private 0.41517**.

## Results

Validation: history = ISO weeks 11-20, target = weeks 21-30, 1,246 test users with target activity,
candidate pool = labelled programmes airing in weeks 21-30, repeats allowed, popularity fallback, K = 5.
`A5_half` (primary) = user's top-5 programmes by number of airings watched >= 50%; it is the target definition
that reproduces both leaderboard calibration points (see `src/13_gt_grid.py`, `src/14_count_hypothesis.py`).

| Method | A5_half | A5_any | T5sec | Kaggle public / private |
|---|---|---|---|---|
| Popularity (reference, distinct viewers) | 0.2922 | 0.3202 | 0.1771 | 0.28234 / 0.27809 (s01) |
| Personal history (own airing counts) | 0.3891 | 0.4496 | 0.2857 | — |
| Personal history by watched seconds | 0.2953* | 0.3387* | 0.3014* | 0.27882 / 0.24850 (s02) |
| MF: PureSVD, 128 factors | 0.3847 | 0.4465 | 0.2827 | — |
| MF: ALS (implicit), 16 factors | 0.2075 | 0.2394 | 0.1468 | — |
| Content-based (TF-IDF metadata) | 0.2054 | 0.2171 | 0.1392 | — |
| **Hybrid (final)** | **0.4493** | **0.4646** | 0.2612 | **0.42177 / 0.41517 (s03)** |

\* from `outputs/11_gt_sensitivity.csv` / `outputs/14_count_hypothesis.csv` (same protocol, seconds-based ranking).
Paired bootstrap over users (2,000 resamples): hybrid − personal history = +0.062 [0.055, 0.070];
hybrid − popularity = +0.159 [0.148, 0.170] (`outputs/52_bootstrap.csv`).

## Protocol decisions

- **Join:** a view matches a schedule row when `vsetv_id == channel_id` and the intervals overlap by > 0 s.
  Both files use local wall-clock time (both show the +3600 s DST gap on 2020-03-29, `src/02b_dst_check.py`).
  Verified against a brute-force join on 320 views, 0 mismatches (`src/03b_verify_join.py`).
- **Watched:** the log has no views shorter than 61 s. Training signal = number of airings with any overlap
  (`n_airings`); an airing watched >= 50% defines the validation target. Overlap fractions are bimodal
  (38% of user-airings < 10% covered, 17.5% > 90%), see `src/04_threshold_eda.py`.
- **`tv_show_id = 0`:** dropped (organisers: "irrelevant TV shows, drop them"); kept only in the join diagnostics,
  which show that 53% of viewing seconds land on labelled airtime although only 10.8% of schedule rows are labelled.
- **Candidate pool:** labelled programmes that air in the target window (test: weeks 31-42 schedule, used only
  for what airs where and when, plus programme metadata). Repeats are allowed, identically for every method.
- **Cold start:** every test user is in the log; any list shorter than 5 is padded with popularity (0 users needed it
  in the final submission).
- **Top-5:** each method ranks the pool by its own score (count, cosine, MF dot product, re-ranker probability),
  takes the best 5 finite positive scores and pads with popularity; scores are never mixed across methods
  except as features of the re-ranker.
- **Hybrid:** candidates = own history ∪ PureSVD top-50 ∪ content top-20 ∪ popularity top-30 (recall 0.841);
  `HistGradientBoostingClassifier` on 20 features, fitted on non-test users for validation and on all users
  (history 11-20 → labels 21-30) for the submission, applied to history 21-30 → weeks 31-42.

## Reproduce

```sh
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
# accept the competition rules on Kaggle first, token in ~/.kaggle/access_token
.venv/bin/kaggle competitions download sweettv-tv-program-recommender -p data
unzip data/sweettv-tv-program-recommender.zip -d data
sh run_all.sh          # every stage in order, logs in outputs/logs/, ~5 min
```

`run_all.sh` regenerates every table in `outputs/` and the three submission files byte-for-byte
(SHA-256 of `s03_hybrid.csv`: `1cb1da466c7c21bc…`). Note: `src/04_threshold_eda.py` also writes
`data/parquet/airings.parquet`, which all later stages read.

| Stage | Scripts |
|---|---|
| Data, join, checks | `00_prepare`, `01_eda`, `02_time_alignment`, `02b_dst_check`, `03_join`, `03b_verify_join`, `04_threshold_eda`, `05_repeat_stats`, `check_metric` |
| Baselines and LB calibration | `10_baselines`, `11_gt_sensitivity`, `12_baseline_submissions`, `13_gt_grid`, `14_count_hypothesis`, `15_viewcount_variant` |
| Method exploration | `20_als`, `21_mf_variants`, `30_content`, `31_content_ablation` |
| Unified comparison, hybrid | `40_methods_eval`, `50_hybrid_eval`, `51_hybrid_experiments`, `52_robustness` |
| Submission, examples | `60_final_submission`, `70_examples` |

Shared code: `common.py` (windows, pool, ground truth, MAP@5), `methods.py` (popularity, personal history, ALS,
PureSVD, content-based), `hybrid.py` (candidates, features, re-ranker), `submission.py` (writer + format checks).

## Environment

Apple M3 Pro (12 cores, 36 GB), macOS 15.6.1, Python 3.12.14; library versions in `requirements.txt`.
Seeds: `SEED = 42` for SVD, ALS, the re-ranker and the bootstrap; the join check samples views with seed 7.
Full pipeline runtime: 4 min 34 s.

## AI assistance

Code and analysis were produced in a Claude Code session (model Claude Opus 5.5) directed and checked by the author;
the unmodified session log is submitted with the report.
