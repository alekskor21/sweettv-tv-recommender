"""Stage 50: validate the hybrid. History 11-20 -> target 21-30; re-ranker fitted on the
non-test users' candidates (labels = A5_half GT), evaluated on the 1,246 test users."""
import time

import numpy as np
import pandas as pd

from common import P, SPLITS, evaluate, gt_variants, load_airings, load_schedule, test_uids
from hybrid import FEATURES, add_labels, build_candidates, ranker, recs_from_scores

t0 = time.time()
air = load_airings()
sched = load_schedule()
sched_full = pd.read_parquet(P / "schedule.parquet")
sched_full["week"] = sched_full["start_time"].dt.isocalendar().week.astype(int)
n_users = int(air["uid"].max()) + 1
test = test_uids()
train_users = np.setdiff1d(np.arange(n_users), test)
sp = SPLITS["val"]

c, pool, i_pop = build_candidates(air, sched, sched_full, sp["train"], sp["target"], n_users, np.arange(n_users))
c = add_labels(c, air, sp["target"], np.arange(n_users))
print(f"candidates: {len(c):,} rows, {c.groupby('uid').size().median():.0f} per user (median); built in {time.time() - t0:.1f}s")
gts = gt_variants(air, sp["target"], test)
gt_half = gts["A5_half"]
pos_total = sum(len(v) for v in gt_half.values())
covered = c[c["uid"].isin(test) & c["label"]].shape[0]
print(f"candidate recall of A5_half GT items (test users): {covered / pos_total:.3f}")

tr = c[c["uid"].isin(train_users)]
te = c[c["uid"].isin(test)].copy()
t1 = time.time()
model = ranker().fit(tr[FEATURES], tr["label"])
print(f"ranker fitted on {len(tr):,} rows ({tr['uid'].nunique()} non-test users) in {time.time() - t1:.1f}s, "
      f"iterations used: {model.n_iter_}")
te["p"] = model.predict_proba(te[FEATURES])[:, 1]
fallback = list(i_pop.reindex(pool.index).fillna(0).sort_values(ascending=False).index)
res = {"hybrid (GBM re-ranker)": evaluate(recs_from_scores(te, "p", test, fallback), gts),
       "hybrid, s03 legacy (no cold rule)": evaluate(recs_from_scores(te, "p", test, fallback, cold_fallback=False), gts)}
for col in ["h_cnt", "h_expected", "svd"]:
    res[f"single feature: {col}"] = evaluate(recs_from_scores(te, col, test, fallback), gts)
print(pd.DataFrame(res).T.to_string())

# permutation-free importance proxy: drop-one-feature-group effect is costly; report gain-free
# sanity instead: MAP of the hybrid when svd/cb are blanked (shows their contribution)
for blank in [["svd"], ["cb"], ["svd", "cb"], ["ch_aff"], ["h_expected", "h_rate"]]:
    tmp = te.copy()
    tmp[blank] = np.nan if blank[0] in ("svd", "cb") else 0
    tmp["p"] = model.predict_proba(tmp[FEATURES])[:, 1]
    print(f"  with {blank} blanked at prediction: {evaluate(recs_from_scores(tmp, 'p', test, fallback), gts)}")
te[["uid", "tv_show_id", "p", "label"]].to_parquet(P / "val_hybrid_scores.parquet", index=False)
print(f"runtime {time.time() - t0:.1f}s")
