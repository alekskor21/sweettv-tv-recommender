"""Stage 60: final hybrid submission.

Re-ranker fit: all users, features from history 11-20, labels (A5_half) from 21-30 - both before the
test window. Prediction: features from history 21-30 (same 10-week length the ranker was trained on),
candidate pool and airing counts from the weeks 31-42 schedule only. No view in weeks 31-42 exists in
the data, so none can be used. Fallback for any user with < 5 candidates: popularity (21-30 viewers).
"""
import sys
import time

import numpy as np
import pandas as pd

from common import P, load_airings, load_schedule, test_uids
from hybrid import FEATURES, add_labels, build_candidates, ranker, recs_from_scores
from submission import write_submission

name = sys.argv[1] if len(sys.argv) > 1 else "s03_hybrid"
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
recs = recs_from_scores(c, "p", test, fallback)
short = sum(c.groupby("uid").size().reindex(test, fill_value=0) < 5)
print(f"test candidates: {len(c):,}; users with < 5 candidates (padded by popularity): {short}")
write_submission(recs, name, pool.index)
c[["uid", "tv_show_id", "p"] + FEATURES].to_parquet(P / f"{name}_scores.parquet", index=False)
print(f"runtime {time.time() - t0:.1f}s")
