"""Stage 5: how much of a user's future top programmes is already in their past?
Split: history = airings in ISO weeks 11-20, target = weeks 21-30, test users only.
Target ("ground truth" proxy) = user's top-5 programmes by total watched seconds."""
from pathlib import Path

import numpy as np
import pandas as pd

P = Path(__file__).resolve().parent.parent / "data" / "parquet"
air = pd.read_parquet(P / "airings.parquet")
users = pd.read_parquet(P / "user_index.parquet")["user_id"]
test = set(pd.read_parquet(P / "submission_template.parquet")["user_id"])
test_uid = set(np.flatnonzero(users.isin(test).values))
sched = pd.read_parquet(P / "schedule_dedup.parquet", columns=["tv_show_id", "block"])

hist = air[(air["week"] <= 20)]
targ = air[(air["week"] >= 21) & (air["week"] <= 30) & air["uid"].isin(test_uid)]
ut = targ.groupby(["uid", "tv_show_id"])["overlap_sec"].sum().reset_index()
ut = ut.sort_values(["uid", "overlap_sec"], ascending=[True, False])
gt = ut.groupby("uid").head(5)
print("test users with target activity:", gt["uid"].nunique(), "| GT size median:", gt.groupby("uid").size().median())

uh = hist.groupby(["uid", "tv_show_id"])["overlap_sec"].sum().reset_index()
seen = set(zip(uh["uid"], uh["tv_show_id"]))
gt_seen = np.array([(u, s) in seen for u, s in zip(gt["uid"], gt["tv_show_id"])])
print(f"GT items the user watched (any overlap) in weeks 11-20: {gt_seen.mean():.3f}")

uh = uh.sort_values(["uid", "overlap_sec"], ascending=[True, False])
for k in (5, 10, 20):
    topk = set(zip(*uh.groupby("uid").head(k)[["uid", "tv_show_id"]].values.T))
    print(f"GT items inside the user's own top-{k} of weeks 11-20: "
          f"{np.mean([(u, s) in topk for u, s in zip(gt['uid'], gt['tv_show_id'])]):.3f}")

ids_1120 = set(sched.loc[(sched["block"] == "11-20") & (sched["tv_show_id"] != 0), "tv_show_id"])
print(f"GT items whose id does not air at all in weeks 11-20 schedule: {(~gt['tv_show_id'].isin(ids_1120)).mean():.3f}")

# overlap of top-5 sets between consecutive windows: how stable are tastes?
top_h = uh.groupby("uid").head(5)
top_h = top_h[top_h["uid"].isin(gt["uid"])]
inter = gt.merge(top_h, on=["uid", "tv_show_id"]).groupby("uid").size()
print("per-user |top5(11-20) ∩ top5(21-30)| distribution:",
      inter.reindex(gt["uid"].unique(), fill_value=0).value_counts().sort_index().to_dict())

# global concentration: how many programmes make up the GT
cnt = gt["tv_show_id"].value_counts()
print("distinct programmes in all GT lists:", len(cnt), "| share of GT slots taken by top-20 programmes:",
      round(cnt.head(20).sum() / len(gt), 3))
