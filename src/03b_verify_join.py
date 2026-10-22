"""Stage 3b: brute-force re-check of the vectorised join on 300 random views."""
from pathlib import Path

import numpy as np
import pandas as pd

P = Path(__file__).resolve().parent.parent / "data" / "parquet"
views = pd.read_parquet(P / "views.parquet")
sched = pd.read_parquet(P / "schedule_dedup.parquet")
j = pd.read_parquet(P / "joined.parquet")

rng = np.random.default_rng(7)
idx = rng.choice(len(views), 300, replace=False)
long_idx = views.index[views["view_sec"] > 20000][:20].values  # include very long views
mismatch = 0
for vi in np.r_[idx, long_idx]:
    v = views.iloc[vi]
    s = sched[sched["channel_id"] == v["channel_id"]]
    ov = (s["end_time"].clip(upper=v["stop_time"]) - s["start_time"].clip(lower=v["start_time"])).dt.total_seconds()
    brute = sorted(zip(s.index[ov > 0], ov[ov > 0].astype(int)))
    fast = sorted(zip(j.loc[j["view_idx"] == vi, "sched_idx"], j.loc[j["view_idx"] == vi, "overlap_sec"]))
    mismatch += brute != fast
print("views checked:", len(idx) + len(long_idx), "| mismatches:", mismatch)
