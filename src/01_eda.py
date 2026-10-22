"""Stage 1: exploratory checks that shape the join and the validation design."""
from pathlib import Path

import numpy as np
import pandas as pd

P = Path(__file__).resolve().parent.parent / "data" / "parquet"
views = pd.read_parquet(P / "views.parquet")
sched = pd.read_parquet(P / "schedule.parquet")
sub = pd.read_parquet(P / "submission_template.parquet")


def section(title):
    print(f"\n=== {title}")


section("time ranges")
views["week"] = views["start_time"].dt.isocalendar().week.astype(int)
sched["week"] = sched["start_time"].dt.isocalendar().week.astype(int)
print("views:", views["start_time"].min(), "->", views["stop_time"].max(),
      "| ISO weeks", views["week"].min(), "-", views["week"].max())
for b, g in sched.groupby("block"):
    print(f"schedule {b}:", g["start_time"].min(), "->", g["end_time"].max(),
          "| ISO weeks", g["week"].min(), "-", g["week"].max())

section("view_sec sanity")
diff = (views["stop_time"] - views["start_time"]).dt.total_seconds() - views["view_sec"]
print("rows where view_sec != stop-start:", int((diff != 0).sum()))
print("view_sec quantiles:", views["view_sec"].quantile([.1, .25, .5, .75, .9, .99]).to_dict(),
      "max", views["view_sec"].max())
print("share of views < 60 s:", round((views["view_sec"] < 60).mean(), 3),
      "| < 300 s:", round((views["view_sec"] < 300).mean(), 3))

section("channels")
vch = set(views["channel_id"].unique())
sch = set(sched["channel_id"].unique())
print("channels in views:", len(vch), "| in schedule:", len(sch), "| views-channels missing from schedule:", len(vch - sch))
miss_share = views["channel_id"].isin(vch - sch).mean()
print("share of view rows on channels absent from schedule:", round(miss_share, 4))
lab_ch = set(sched.loc[sched["tv_show_id"] != 0, "channel_id"].unique())
print("channels with any labelled programme:", len(lab_ch),
      "| share of view rows on such channels:", round(views["channel_id"].isin(lab_ch).mean(), 4))

section("test users")
test_users = set(sub["user_id"])
log_users = set(views["user_id"])
print("test users:", len(test_users), "| in view log:", len(test_users & log_users),
      "| log users not in test:", len(log_users - test_users))
per_user = views.groupby("user_id").agg(n=("view_sec", "size"), sec=("view_sec", "sum"),
                                        first_week=("week", "min"), last_week=("week", "max"))
tu = per_user.loc[per_user.index.isin(test_users)]
print("test users: views per user quantiles", tu["n"].quantile([.1, .5, .9]).to_dict())
print("test users: hours watched quantiles", (tu["sec"] / 3600).quantile([.1, .5, .9]).round(1).to_dict())
print("test users active in weeks 21-30:", int((tu["last_week"] >= 21).sum()),
      "| active in weeks 11-20:", int((tu["first_week"] <= 20).sum()),
      "| last week = 30:", int((tu["last_week"] == 30).sum()))

section("schedule quality")
s = sched.sort_values(["channel_id", "start_time"])
nxt = s.groupby("channel_id")["start_time"].shift(-1)
gap = (nxt - s["end_time"]).dt.total_seconds()
print("consecutive pairs on a channel: overlap (gap<0)", int((gap < 0).sum()),
      "| gap>0", int((gap > 0).sum()), "| exact", int((gap == 0).sum()))
print("exact duplicate (channel,start) rows:", int(s.duplicated(["channel_id", "start_time"]).sum()))
print("programme duration quantiles (s):", sched["duration"].quantile([.1, .5, .9, .99]).to_dict())

section("programme ids")
lab = sched[sched["tv_show_id"] != 0]
ids = {b: set(g["tv_show_id"]) for b, g in lab.groupby("block")}
print("distinct ids per block:", {b: len(v) for b, v in ids.items()})
train_ids = ids["11-20"] | ids["21-30"]
print("test ids also in train schedule:", len(ids["31-42"] & train_ids),
      "| new in test:", len(ids["31-42"] - train_ids))
print("ids in 21-30 also in 11-20:", len(ids["21-30"] & ids["11-20"]))
attr = lab.groupby("tv_show_id").agg(n_title=("tv_show_title", "nunique"),
                                     n_cat=("tv_show_category", "nunique"),
                                     n_ch=("channel_id", "nunique"))
print("ids with >1 category:", int((attr["n_cat"] > 1).sum()),
      "| ids airing on >1 channel:", int((attr["n_ch"] > 1).sum()),
      "| ids with >1 title string:", int((attr["n_title"] > 1).sum()))
print("categories:", lab["tv_show_category"].value_counts().to_dict())
print("unlabelled categories:", sched.loc[sched["tv_show_id"] == 0, "tv_show_category"].value_counts().head(9).to_dict())
for col in ["tv_show_genre_1", "year_of_production", "director", "actors"]:
    print(f"labelled rows with {col} filled:", round(lab[col].notna().mean(), 3))
