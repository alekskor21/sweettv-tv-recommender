"""Stage 0: read raw CSVs once, parse timestamps, save typed parquet copies.

user_id values reach 1.8e19 (> int64 max), so they are kept as strings to avoid
silent precision loss; numeric codes are assigned later where needed.
"""
import time
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
OUT = DATA / "parquet"
OUT.mkdir(exist_ok=True)

t0 = time.time()

views = pd.read_csv(
    DATA / "dataset11-30.csv",
    dtype={"user_id": str, "vsetv_id": "int32", "duraton": "int64"},
)
views["start_time"] = pd.to_datetime(views["start_time"], format="%Y-%m-%d %H:%M:%S")
views["stop_time"] = pd.to_datetime(views["stop_time"], format="%Y-%m-%d %H:%M:%S")
views = views.rename(columns={"vsetv_id": "channel_id", "duraton": "view_sec"})
views.to_parquet(OUT / "views.parquet", index=False)
print(f"views: {len(views):,} rows, {views['user_id'].nunique():,} users")

sched_files = {
    "11-20": "export_arh_11-20-final.csv",
    "21-30": "export_arh_21-30-final.csv",
    "31-42": "export_arh_31-42-final.csv",
}
parts = []
for block, fname in sched_files.items():
    s = pd.read_csv(DATA / fname, dtype={"year_of_production": str, "tv_show_id": "int64"})
    s["start_time"] = pd.to_datetime(s["start_time"], format="%d.%m.%Y %H:%M:%S")
    s["block"] = block
    parts.append(s)
    print(f"schedule {block}: {len(s):,} rows, labelled {int((s['tv_show_id'] != 0).sum()):,}")
sched = pd.concat(parts, ignore_index=True)
sched["end_time"] = sched["start_time"] + pd.to_timedelta(sched["duration"], unit="s")
sched.to_parquet(OUT / "schedule.parquet", index=False)

sub = pd.read_csv(DATA / "submission.csv", dtype=str)
sub.to_parquet(OUT / "submission_template.parquet", index=False)
print(f"submission template: {len(sub):,} users")

print(f"done in {time.time() - t0:.1f}s")
