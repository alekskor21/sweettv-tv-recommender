"""Stage 51: hybrid variants on validation (history 11-20 -> 21-30, ranker on non-test users,
scored on test users). Includes the proper ablation (re-fit without a feature group)."""
import time

import numpy as np
import pandas as pd

from common import P, SPLITS, evaluate, gt_variants, load_airings, load_schedule, test_uids
from hybrid import FEATURES, FEATURES_EXT, add_labels, build_candidates, ranker, recs_from_scores

air = load_airings()
sched = load_schedule()
sched_full = pd.read_parquet(P / "schedule.parquet")
sched_full["week"] = sched_full["start_time"].dt.isocalendar().week.astype(int)
n_users = int(air["uid"].max()) + 1
all_users = np.arange(n_users)
test = test_uids()
train_users = np.setdiff1d(all_users, test)
sp = SPLITS["val"]
gts = gt_variants(air, sp["target"], test)
rows = []


def run(name, c, features, **kw):
    t1 = time.time()
    tr, te = c[c["uid"].isin(train_users)], c[c["uid"].isin(test)].copy()
    model = ranker(features, **kw).fit(tr[features], tr["label"])
    te["p"] = model.predict_proba(te[features])[:, 1]
    fb = list(te.groupby("tv_show_id")["i_pop"].first().sort_values(ascending=False).index)
    rows.append({"variant": name, **evaluate(recs_from_scores(te, "p", test, fb), gts),
                 "iters": model.n_iter_, "sec": round(time.time() - t1, 1)})
    print(rows[-1], flush=True)


c = build_candidates(air, sched, sched_full, sp["train"], sp["target"], n_users, all_users)[0]
c = add_labels(c, air, sp["target"], all_users)
run("final configuration (base features)", c, FEATURES)
run("+ regularity/recency features", c, FEATURES_EXT)
run("+ regularity, 63 leaves", c, FEATURES_EXT, max_leaf_nodes=63)
run("+ regularity, lr 0.1", c, FEATURES_EXT, learning_rate=0.1)

# proper ablation on the extended model: re-fit without each group
groups = {"MF score (svd)": ["svd"], "content score (cb)": ["cb"], "svd + cb": ["svd", "cb"],
          "channel affinity": ["ch_aff"], "own history counts": ["h_cnt", "h_cnt_half", "h_sec", "h_cnt_recent",
                                                                 "h_share", "h_rate", "h_expected", "h_cnt_last2", "h_weeks", "h_gap"],
          "target-window airing counts": ["i_air_target", "i_air_ratio", "h_expected"]}
for gname, cols in groups.items():
    run(f"ablation: without {gname}", c, [f for f in FEATURES_EXT if f not in cols])

# wider candidate set
c2 = build_candidates(air, sched, sched_full, sp["train"], sp["target"], n_users, all_users, n_svd=100, n_cb=50, n_pop=50)[0]
c2 = add_labels(c2, air, sp["target"], all_users)
gt_half = gts["A5_half"]
rec = c2[c2["uid"].isin(test) & c2["label"]].shape[0] / sum(len(v) for v in gt_half.values())
print(f"wider candidates: {len(c2):,} rows, recall {rec:.3f}")
run("+ regularity, wider candidates (svd100/cb50/pop50)", c2, FEATURES_EXT)

pd.DataFrame(rows).to_csv("outputs/51_hybrid_experiments.csv", index=False)
