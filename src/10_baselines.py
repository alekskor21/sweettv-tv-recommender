"""Stage 10: reference baselines on the validation split (fit 11-20, evaluate 21-30).

- popularity (required reference): distinct viewers in the train window, pool-restricted;
- popularity variants: total seconds, last-4-weeks viewers;
- personal history ("re-watch"): the user's own programmes by watched seconds, pool-restricted,
  padded with popularity. Not one of the four required methods; it measures how far habit goes.
"""
import time

import numpy as np
import pandas as pd

from common import (SPLITS, candidate_pool, fill_to_k, ground_truth, interactions, load_airings,
                    load_schedule, mapk, popularity_ranking, test_uids)

t0 = time.time()
air = load_airings()
sched = load_schedule()
uids = test_uids()
sp = SPLITS["val"]
gt = ground_truth(air, sp["target"], uids)
pool = candidate_pool(sched, sp["target"])
print(f"users with GT: {len(gt)} | pool size: {len(pool)}")

rows = []
for rule in ["any", "300s", "half", "300s_or_half"]:
    inter = interactions(air, sp["train"], rule)
    pop = popularity_ranking(inter, pool)
    rows.append((rule, "popularity: distinct viewers", mapk(gt, {u: pop[:5] for u in gt})))

    pop_sec = inter.groupby("tv_show_id")["overlap_sec"].sum()
    pop_sec = list(pop_sec[pop_sec.index.isin(pool.index)].sort_values(ascending=False).index)
    rows.append((rule, "popularity: total seconds", mapk(gt, {u: pop_sec[:5] for u in gt})))

    last = interactions(air, (sp["train"][1] - 3, sp["train"][1]), rule)
    pop_last = popularity_ranking(last, pool)
    rows.append((rule, "popularity: viewers, last 4 weeks", mapk(gt, {u: pop_last[:5] for u in gt})))

    own = inter[inter["tv_show_id"].isin(pool.index)].sort_values(["uid", "overlap_sec"], ascending=[True, False])
    own_lists = own.groupby("uid")["tv_show_id"].apply(list).to_dict()
    recs = {u: fill_to_k(own_lists.get(u, []), pop) for u in gt}
    rows.append((rule, "personal history + popularity fill", mapk(gt, recs)))

res = pd.DataFrame(rows, columns=["watched rule", "method", "MAP@5"])
print(res.pivot(index="method", columns="watched rule", values="MAP@5").round(4).to_string())
res.to_csv("outputs/10_baselines.csv", index=False)
print(f"runtime {time.time() - t0:.1f}s")
