"""Stage 13: after LB calibration point #1 (popularity: public 0.28234 / private 0.27809),
find which ground-truth definitions reproduce ~0.28 for popularity on validation and what they
predict for personal history, so submission #2 can separate them."""
import pandas as pd

from common import (SPLITS, candidate_pool, fill_to_k, interactions, load_airings, load_schedule,
                    mapk, popularity_ranking, test_uids)

air = load_airings()
sched = load_schedule()
uids = test_uids()
sp = SPLITS["val"]
pool = candidate_pool(sched, sp["target"])
inter = interactions(air, sp["train"], "any")
pop = popularity_ranking(inter, pool)
own = inter[inter["tv_show_id"].isin(pool.index)].sort_values(["uid", "overlap_sec"], ascending=[True, False])
own_lists = own.groupby("uid")["tv_show_id"].apply(list).to_dict()

t = air[(air["week"] >= sp["target"][0]) & (air["week"] <= sp["target"][1]) & air["uid"].isin(uids)].assign(one=1)
rows = []
for weight in ["overlap_sec", "one"]:
    for thr_name, sub in [("any", t), (">=300s", t[t["overlap_sec"] >= 300]), (">=50%", t[t["frac"] >= .5])]:
        g = sub.groupby(["uid", "tv_show_id"], as_index=False)[weight].sum().sort_values(["uid", weight], ascending=[True, False])
        lists = g.groupby("uid")["tv_show_id"].apply(list).to_dict()
        for n in [5, 10, 15, 20, 30, None]:
            gt = {u: (v[:n] if n else v) for u, v in lists.items()}
            p = mapk(gt, {u: pop[:5] for u in gt})
            o = mapk(gt, {u: fill_to_k(own_lists.get(u, []), pop) for u in gt})
            rows.append(("seconds" if weight == "overlap_sec" else "airings", thr_name, n or "all", round(p, 4), round(o, 4)))
res = pd.DataFrame(rows, columns=["rank by", "airing filter", "top-N", "popularity", "personal"])
res.to_csv("outputs/13_gt_grid.csv", index=False)
near = res[(res["popularity"] - 0.28).abs() <= 0.03]
print("definitions with validation popularity within 0.28 +- 0.03:")
print(near.to_string(index=False))
