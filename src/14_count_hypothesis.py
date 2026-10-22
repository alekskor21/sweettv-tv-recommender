"""Stage 14: after LB point #2 (personal history: public 0.27882 / private 0.24850, i.e. ~equal
to popularity), the count-based ground truth (top-5 by number of airings watched) is the only
family that reproduces both LB numbers on validation. Under that family, test how personal
history and popularity should be scored, and how fast personal history decays with time gap."""
import pandas as pd

from common import SPLITS, candidate_pool, fill_to_k, load_airings, load_schedule, mapk, test_uids

air = load_airings()
sched = load_schedule()
uids = test_uids()
sp = SPLITS["val"]
pool = candidate_pool(sched, sp["target"])
air = air.assign(one=1)


def window(weeks, flt=None):
    a = air[(air["week"] >= weeks[0]) & (air["week"] <= weeks[1])]
    if flt == "300s":
        a = a[a["overlap_sec"] >= 300]
    elif flt == "half":
        a = a[a["frac"] >= .5]
    return a


def topk_lists(a, col, users=None, k=None):
    if users is not None:
        a = a[a["uid"].isin(users)]
    g = a.groupby(["uid", "tv_show_id"], as_index=False)[col].sum().sort_values(["uid", col], ascending=[True, False])
    return g.groupby("uid")["tv_show_id"].apply(lambda s: list(s.head(k)) if k else list(s)).to_dict()


gts = {f"A5_{f or 'any'}": topk_lists(window(sp["target"], f), "one", uids, 5) for f in (None, "300s", "half")}


def ev(recs):
    return {n: round(mapk(gt, recs), 4) for n, gt in gts.items()}


rows = []
# popularity variants
for name, (weeks, flt, col, how) in {
    "pop: distinct viewers, 11-20": ((11, 20), None, "uid", "nunique"),
    "pop: airings>=50% count, 11-20": ((11, 20), "half", "one", "sum"),
    "pop: distinct viewers, 17-20": ((17, 20), None, "uid", "nunique"),
    "pop: airings>=50% count, 17-20": ((17, 20), "half", "one", "sum"),
}.items():
    s = getattr(window(weeks, flt).groupby("tv_show_id")[col], how)()
    rank = list(s[s.index.isin(pool.index)].sort_values(ascending=False).index)
    rows.append((name, ev({u: rank[:5] for u in uids})))
pop = rank_default = None
s = window((11, 20)).groupby("tv_show_id")["uid"].nunique()
pop = list(s[s.index.isin(pool.index)].sort_values(ascending=False).index)

# personal history variants: scoring column x filter x history window
for weeks in [(11, 20), (15, 20), (17, 20), (19, 20)]:
    for flt in (None, "300s", "half"):
        for col in ("overlap_sec", "one"):
            h = window(weeks, flt)
            h = h[h["tv_show_id"].isin(pool.index)]
            lists = topk_lists(h, col)
            recs = {u: fill_to_k(lists.get(u, []), pop) for u in uids}
            rows.append((f"personal: weeks {weeks[0]}-{weeks[1]}, filter {flt or 'any'}, by {'seconds' if col == 'overlap_sec' else 'count'}", ev(recs)))

res = pd.DataFrame([{"method": n, **r} for n, r in rows])
res.to_csv("outputs/14_count_hypothesis.csv", index=False)
print(res.to_string(index=False))

# decay: history 11-15 scored against later and later 5-week targets (A5_half definition)
print("\ndecay of personal history (history 11-15, by count >=50%), target = 5-week window:")
h = window((11, 15), "half")
for tw in [(16, 20), (21, 25), (26, 30)]:
    pl = candidate_pool(sched, tw)
    lists = topk_lists(h[h["tv_show_id"].isin(pl.index)], "one")
    gt = topk_lists(window(tw, "half"), "one", uids, 5)
    sp_ = window((11, 15)).groupby("tv_show_id")["uid"].nunique()
    pp = list(sp_[sp_.index.isin(pl.index)].sort_values(ascending=False).index)
    print(f"  target {tw}: personal {mapk(gt, {u: fill_to_k(lists.get(u, []), pp) for u in uids}):.4f}"
          f" | popularity {mapk(gt, {u: pp[:5] for u in uids}):.4f}")
