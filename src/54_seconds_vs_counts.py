"""Stage 54: check the mechanism claimed for the s02 failure - does ranking by watched seconds favour
long programmes, and ranking by airing counts favour short daily ones (news)? Validation split."""
import pandas as pd

from common import P, SPLITS, candidate_pool, gt_variants, interactions, load_airings, load_schedule, test_uids

air = load_airings()
sched = load_schedule()
uids = test_uids()
sp = SPLITS["val"]
pool = candidate_pool(sched, sp["target"])
full = pd.read_parquet(P / "schedule.parquet")
lab = full[full["tv_show_id"] != 0]
dur = lab.groupby("tv_show_id")["duration"].median()
cat = lab.groupby("tv_show_id")["tv_show_category"].first()

inter = interactions(air, sp["train"], "any")
inter = inter[inter["uid"].isin(uids) & inter["tv_show_id"].isin(pool.index)]


def describe(name, lists):
    ids = pd.Series([i for v in lists.values() for i in v])
    shares = ids.map(cat).value_counts(normalize=True).round(3).head(4).to_dict()
    print(f"{name:42s} median programme length {ids.map(dur).median() / 60:5.0f} min | categories {shares}")


for col, name in [("overlap_sec", "personal top-5 by seconds (s02 logic)"), ("n_airings", "personal top-5 by airing count")]:
    s = inter.sort_values(["uid", col], ascending=[True, False]).groupby("uid").head(5)
    describe(name, s.groupby("uid")["tv_show_id"].apply(list).to_dict())
gts = gt_variants(air, sp["target"], uids)
describe("target T5sec (top-5 by seconds)", gts["T5sec"])
describe("target A5_half (top-5 by airings >=50%)", gts["A5_half"])
