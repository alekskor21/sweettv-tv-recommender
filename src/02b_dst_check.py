"""Stage 2b: DST fingerprint. Ukraine moved clocks 03:00 -> 04:00 on 2020-03-29
and 04:00 -> 03:00 on 2020-10-25. A file kept in local wall-clock time shows a
+3600 s gap between consecutive programmes that night; a UTC file does not."""
from pathlib import Path

import pandas as pd

P = Path(__file__).resolve().parent.parent / "data" / "parquet"
s = pd.read_parquet(P / "schedule.parquet", columns=["channel_id", "start_time", "end_time", "duration"])
s = s.sort_values(["channel_id", "start_time"])
s["next_start"] = s.groupby("channel_id")["start_time"].shift(-1)
s["gap"] = (s["next_start"] - s["end_time"]).dt.total_seconds()
g = s[s["gap"] != 0].copy()
g["date"] = g["end_time"].dt.date
print("non-zero gaps by date (top):")
print(g.groupby("date")["gap"].agg(["size", "median"]).sort_values("size", ascending=False).head(6))
night = g[(g["end_time"] >= "2020-03-29 02:00") & (g["end_time"] <= "2020-03-29 05:00")]
print("\ngaps ending 2020-03-29 02:00-05:00:", len(night), "| gap values:", night["gap"].value_counts().head(3).to_dict(),
      "| channels:", night["channel_id"].nunique(), "of", s["channel_id"].nunique())
print("their end_time hour:", night["end_time"].dt.hour.value_counts().to_dict())
