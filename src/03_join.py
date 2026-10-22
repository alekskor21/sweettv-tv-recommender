"""Stage 3: join views to schedule (same channel, interval overlap > 0).

Every schedule row is joined, including tv_show_id = 0, so the share of viewing
time that lands on unlabelled airtime can be measured; filtering by a "watched"
threshold happens later. Both files use local wall-clock time (see 02b_dst_check).
Output: one row per (view, programme airing) with the overlap in seconds.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

P = Path(__file__).resolve().parent.parent / "data" / "parquet"
t0 = time.time()

views = pd.read_parquet(P / "views.parquet")
sched = pd.read_parquet(P / "schedule.parquet",
                        columns=["channel_id", "start_time", "end_time", "duration", "tv_show_id", "block"])
sched = sched.drop_duplicates(["channel_id", "start_time"]).sort_values(["channel_id", "start_time"])
sched = sched.reset_index(drop=True)

users = pd.Index(views["user_id"].unique())
views["uid"] = users.get_indexer(views["user_id"]).astype(np.int32)
pd.Series(users, name="user_id").to_frame().to_parquet(P / "user_index.parquet")


def to_s(x):
    return x.values.astype("datetime64[s]").astype(np.int64)


vs_all, ve_all = to_s(views["start_time"]), to_s(views["stop_time"])
ps_all, pe_all = to_s(sched["start_time"]), to_s(sched["end_time"])

out = []
sched_ch = sched["channel_id"].values
bounds = np.flatnonzero(np.r_[True, sched_ch[1:] != sched_ch[:-1], True])
ch_slice = {sched_ch[bounds[k]]: (bounds[k], bounds[k + 1]) for k in range(len(bounds) - 1)}

for ch, vidx in views.groupby("channel_id").indices.items():
    if ch not in ch_slice:
        continue
    a, b = ch_slice[ch]
    ps, pe = ps_all[a:b], pe_all[a:b]
    vs, ve = vs_all[vidx], ve_all[vidx]
    i0 = np.maximum(np.searchsorted(ps, vs, side="right") - 1, 0)
    i1 = np.searchsorted(ps, ve, side="left") - 1
    n = np.maximum(i1 - i0 + 1, 0)
    rep_v = np.repeat(np.arange(len(vidx)), n)
    offs = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    rep_p = i0[rep_v] + offs
    ov = np.minimum(ve[rep_v], pe[rep_p]) - np.maximum(vs[rep_v], ps[rep_p])
    keep = ov > 0
    out.append(pd.DataFrame({
        "view_idx": vidx[rep_v[keep]].astype(np.int32),
        "sched_idx": (a + rep_p[keep]).astype(np.int32),
        "overlap_sec": ov[keep].astype(np.int32),
    }))

j = pd.concat(out, ignore_index=True)
j["uid"] = views["uid"].values[j["view_idx"]]
j["tv_show_id"] = sched["tv_show_id"].values[j["sched_idx"]]
j["prog_sec"] = sched["duration"].values[j["sched_idx"]].astype(np.int32)
j["prog_start"] = sched["start_time"].values[j["sched_idx"]]
j["week"] = pd.DatetimeIndex(j["prog_start"]).isocalendar().week.values.astype(np.int16)
j.to_parquet(P / "joined.parquet", index=False)
sched.to_parquet(P / "schedule_dedup.parquet", index=False)

# --- diagnostics
tot_view = views["view_sec"].sum()
matched = j["overlap_sec"].sum()
lab = j["tv_show_id"] != 0
print(f"joined rows: {len(j):,} | programmes per view (median): {j.groupby('view_idx').size().median():.0f}")
print(f"view seconds matched to any schedule row: {matched / tot_view:.3f}")
print(f"  ...to labelled programmes (tv_show_id != 0): {j.loc[lab, 'overlap_sec'].sum() / tot_view:.3f}")
print(f"views touching at least one labelled programme: {j.loc[lab, 'view_idx'].nunique() / len(views):.3f}")
print(f"done in {time.time() - t0:.1f}s")
