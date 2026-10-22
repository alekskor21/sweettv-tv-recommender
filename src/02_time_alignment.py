"""Stage 2: check that view-log clock and schedule clock agree before joining.

Idea: viewers switch channels around programme boundaries, so if both files use
the same clock, view start times cluster near programme start times at shift 0.
A timezone mismatch would move the peak to a whole-hour shift.
"""
from pathlib import Path

import numpy as np
import pandas as pd

P = Path(__file__).resolve().parent.parent / "data" / "parquet"
views = pd.read_parquet(P / "views.parquet", columns=["channel_id", "start_time", "stop_time", "view_sec"])
sched = pd.read_parquet(P / "schedule.parquet", columns=["channel_id", "start_time"])

# rows where view_sec disagrees with stop-start: look at the size of the disagreement
d = (views["stop_time"] - views["start_time"]).dt.total_seconds() - views["view_sec"]
bad = views.loc[d != 0].assign(diff=d[d != 0])
print("view_sec mismatches:", len(bad), "| diff values:", bad["diff"].value_counts().head(5).to_dict())
print("their start dates:", bad["start_time"].dt.date.value_counts().head(5).to_dict())
print("min view_sec:", views["view_sec"].min())

rng = np.random.default_rng(42)
sample = views.iloc[rng.choice(len(views), 500_000, replace=False)]
starts_by_ch = {c: np.sort(g["start_time"].values.astype("datetime64[s]").astype(np.int64))
                for c, g in sched.groupby("channel_id")}


def near_share(shift_sec, tol):
    hits = total = 0
    for c, g in sample.groupby("channel_id"):
        ps = starts_by_ch.get(c)
        if ps is None:
            continue
        v = g["start_time"].values.astype("datetime64[s]").astype(np.int64) + shift_sec
        i = np.clip(np.searchsorted(ps, v), 1, len(ps) - 1)
        dist = np.minimum(np.abs(v - ps[i - 1]), np.abs(ps[i] - v))
        hits += int((dist <= tol).sum())
        total += len(v)
    return hits / total


print("\nshare of view starts within 60 s of a programme start, by shift applied to view times:")
for h in range(-3, 4):
    print(f"  shift {h:+d} h: {near_share(h * 3600, 60):.4f}")
print("fine shifts around 0 (minutes):")
for m in (-10, -5, -2, -1, 0, 1, 2, 5, 10):
    print(f"  shift {m:+d} min: {near_share(m * 60, 60):.4f}")

# Hourly shifts are not identifiable from round boundaries (:00/:30 repeat every hour).
# Use only "odd" boundaries (minute not a multiple of 5) which do not repeat hourly,
# and test both view starts and view stops against them.
sched_full = pd.read_parquet(P / "schedule.parquet", columns=["channel_id", "start_time", "end_time"])
odd = sched_full[sched_full["start_time"].dt.minute % 5 != 0]
print("\nshare of programme starts on odd minutes:", round(len(odd) / len(sched_full), 3))
odd_by_ch = {c: np.sort(g["start_time"].values.astype("datetime64[s]").astype(np.int64))
             for c, g in odd.groupby("channel_id")}


def near_odd(col, shift_sec, tol=60):
    hits = total = 0
    for c, g in sample.groupby("channel_id"):
        ps = odd_by_ch.get(c)
        if ps is None or len(ps) < 2:
            continue
        v = g[col].values.astype("datetime64[s]").astype(np.int64) + shift_sec
        i = np.clip(np.searchsorted(ps, v), 1, len(ps) - 1)
        dist = np.minimum(np.abs(v - ps[i - 1]), np.abs(ps[i] - v))
        hits += int((dist <= tol).sum())
        total += len(v)
    return hits / total


print("share within 60 s of an odd-minute programme boundary (start / stop of view):")
for h in range(-3, 4):
    print(f"  shift {h:+d} h: {near_odd('start_time', h * 3600):.4f} / {near_odd('stop_time', h * 3600):.4f}")
