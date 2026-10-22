"""Stage 40: unified comparison of the individual methods under the final protocol.

Split: fit on ISO weeks 11-20, evaluate on 21-30; users: the 1,246 test users with target activity;
pool: labelled programmes airing in 21-30; repeats allowed; fallback: popularity; K = 5.
Hyper-parameters chosen on A5_half (primary GT); grids are small and reported in full.
"""
import itertools
import time

import pandas as pd

from common import (P, SPLITS, candidate_pool, evaluate, gt_variants, interactions, load_airings,
                    load_schedule, popularity_ranking, test_uids)
from methods import ALS, ContentBased, PersonalHistory, Popularity, PureSVD, top_k

air = load_airings()
sched = load_schedule()
sched_full = pd.read_parquet(P / "schedule.parquet")
sched_full["week"] = sched_full["start_time"].dt.isocalendar().week.astype(int)
uids = test_uids()
n_users = int(air["uid"].max()) + 1
sp = SPLITS["val"]
pool = candidate_pool(sched, sp["target"])
cand = pool.index.values
gts = gt_variants(air, sp["target"], uids)
inter = interactions(air, sp["train"], "any")
pop = popularity_ranking(inter, pool)
rows = []


def run(name, params, model_fn):
    t1 = time.time()
    model = model_fn()
    fit_sec = time.time() - t1
    recs = top_k(model.score(uids, cand), uids, cand, pop)
    rows.append({"method": name, "params": params, **evaluate(recs, gts),
                 "fit_s": round(fit_sec, 2), "total_s": round(time.time() - t1, 2)})
    print(rows[-1], flush=True)


run("popularity", "distinct viewers", lambda: Popularity().fit(inter))
run("personal history", "airing count", lambda: PersonalHistory().fit(inter))
for f, a, m in itertools.product([16, 32, 64], [1.0, 10.0], [1, 3]):
    run("ALS", f"factors={f} alpha={a} min_airings={m}", lambda: ALS(f, 0.1, a, m).fit(inter, n_users))
for f in [16, 32, 64, 128]:
    run("PureSVD", f"factors={f}", lambda: PureSVD(f).fit(inter, n_users))
for m, fields in itertools.product([1, 3, 10], [("cat", "genre", "decade", "dir", "act"),
                                                ("cat", "genre", "decade", "ch", "dir", "act")]):
    run("content-based", f"min_airings={m} fields={'+'.join(fields)}",
        lambda: ContentBased(m, fields).fit(inter, n_users, sched_full, sp["target"][1]))

res = pd.DataFrame(rows)
res.to_csv("outputs/40_methods_eval.csv", index=False)
best = res.sort_values("A5_half", ascending=False).groupby("method").head(1)
print("\nbest configuration per method (selected on A5_half):")
print(best.to_string(index=False))
