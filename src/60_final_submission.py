"""Stage 60: final hybrid submission.

Re-ranker fit: all users, features from history 11-20, labels (A5_half) from 21-30 - both before the
test window. Prediction: features from history 21-30 (same 10-week length the ranker was trained on),
candidate pool and airing counts from the weeks 31-42 schedule only. No view in weeks 31-42 exists in
the data, so none can be used. Users with no labelled airing in weeks 21-30 get the popularity top-5
(21-30 viewers), the same fallback as every single method.
Usage: 60_final_submission.py <name> [--s03]   (--s03 reproduces the third submission, made before the
uniform cold-start rule: there the re-ranker also ordered cold users' candidates from item features).
"""
import sys
import time

import numpy as np
import pandas as pd

from common import P, load_airings, load_schedule, test_uids
from hybrid import FEATURES, add_labels, build_candidates, ranker, recs_from_scores
from submission import write_submission

name = sys.argv[1] if len(sys.argv) > 1 else "s04_hybrid"
legacy_s03 = "--s03" in sys.argv[2:]
t0 = time.time()
air = load_airings()
sched = load_schedule()
sched_full = pd.read_parquet(P / "schedule.parquet")
sched_full["week"] = sched_full["start_time"].dt.isocalendar().week.astype(int)
n_users = int(air["uid"].max()) + 1
all_users = np.arange(n_users)
test = test_uids()

fit_c, _, _ = build_candidates(air, sched, sched_full, (11, 20), (21, 30), n_users, all_users)
fit_c = add_labels(fit_c, air, (21, 30), all_users)
model = ranker().fit(fit_c[FEATURES], fit_c["label"])
print(f"ranker fitted on {len(fit_c):,} rows, {model.n_iter_} iterations")

c, pool, i_pop = build_candidates(air, sched, sched_full, (21, 30), (31, 42), n_users, test)
c["p"] = model.predict_proba(c[FEATURES])[:, 1]
fallback = list(i_pop.reindex(pool.index).fillna(0).sort_values(ascending=False).index)
recs = recs_from_scores(c, "p", test, fallback, cold_fallback=not legacy_s03)
cold = int((c.groupby("uid")["u_cnt"].first().reindex(test, fill_value=0) == 0).sum())
print(f"test candidates: {len(c):,}; cold test users (no labelled airing in 21-30): {cold} -> "
      f"{'re-ranker order (s03 legacy)' if legacy_s03 else 'popularity top-5'}")
write_submission(recs, name, pool.index)
c[["uid", "tv_show_id", "p"] + FEATURES].to_parquet(P / f"{name}_scores.parquet", index=False)
print(f"runtime {time.time() - t0:.1f}s")
