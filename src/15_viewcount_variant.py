"""Stage 15: one more count definition - number of joined (view x programme) rows, so returning
to the same airing after zapping counts again. Check its fit to both LB points."""
import pandas as pd

from common import P, SPLITS, candidate_pool, fill_to_k, load_airings, load_schedule, mapk, test_uids

j = pd.read_parquet(P / "joined.parquet", columns=["uid", "tv_show_id", "week", "overlap_sec", "prog_sec"])
j = j[j["tv_show_id"] != 0].assign(one=1)
air = load_airings()
sched = load_schedule()
uids = test_uids()
sp = SPLITS["val"]
pool = candidate_pool(sched, sp["target"])


def lists(df, col, users=None, k=None):
    if users is not None:
        df = df[df["uid"].isin(users)]
    g = df.groupby(["uid", "tv_show_id"], as_index=False)[col].sum().sort_values(["uid", col], ascending=[True, False])
    return g.groupby("uid")["tv_show_id"].apply(lambda s: list(s.head(k)) if k else list(s)).to_dict()


jt = j[(j["week"] >= 21) & (j["week"] <= 30)]
gt_v5 = lists(jt, "one", uids, 5)
gt_v5_half = lists(jt[jt["overlap_sec"] >= 0.5 * jt["prog_sec"]], "one", uids, 5)
jh = j[(j["week"] >= 11) & (j["week"] <= 20) & j["tv_show_id"].isin(pool.index)]
ah = air[(air["week"] >= 11) & (air["week"] <= 20) & air["tv_show_id"].isin(pool.index)].assign(one=1)
s = ah.groupby("tv_show_id")["uid"].nunique()
pop = list(s.sort_values(ascending=False).index)
cands = {
    "popularity": {u: pop[:5] for u in uids},
    "personal by seconds (= s02)": {u: fill_to_k(v, pop) for u, v in lists(ah, "overlap_sec").items()},
    "personal by airings count": {u: fill_to_k(v, pop) for u, v in lists(ah, "one").items()},
    "personal by view-rows count": {u: fill_to_k(v, pop) for u, v in lists(jh, "one").items()},
}
for name, gt in [("V5 (view rows)", gt_v5), ("V5_half", gt_v5_half)]:
    print(name, {n: round(mapk(gt, r), 4) for n, r in cands.items()})
