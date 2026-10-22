"""Write a Kaggle submission in the template's user order and validate it before upload."""
import pandas as pd

from common import OUT, P, K, load_users


def write_submission(recs, name, pool_ids):
    """recs: uid -> list of K programme ids. Writes outputs/submissions/<name>.csv and checks it."""
    users = load_users()
    tmpl = pd.read_parquet(P / "submission_template.parquet")
    uid_of = pd.Index(users).get_indexer(tmpl["user_id"])
    lines = []
    for user_id, uid in zip(tmpl["user_id"], uid_of):
        ids = recs[uid]
        lines.append(f"{user_id},{' '.join(str(int(x)) for x in ids)}")
    path = OUT / "submissions" / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("user_id,tv_show_id\n" + "\n".join(lines) + "\n")
    check_submission(path, pool_ids)
    return path


def check_submission(path, pool_ids):
    sub = pd.read_csv(path, dtype=str)
    tmpl = pd.read_parquet(P / "submission_template.parquet")
    assert list(sub.columns) == ["user_id", "tv_show_id"], sub.columns
    assert len(sub) == len(tmpl) == 1260, len(sub)
    assert (sub["user_id"].values == tmpl["user_id"].values).all(), "user order differs from template"
    ids = sub["tv_show_id"].str.split(" ")
    assert (ids.str.len() == K).all(), "every row needs exactly 5 ids"
    assert ids.apply(lambda x: len(set(x)) == K).all(), "duplicate id inside a row"
    flat = set(int(x) for row in ids for x in row)
    assert 0 not in flat, "tv_show_id 0 must not be recommended"
    assert flat <= set(int(x) for x in pool_ids), "id outside the test-window candidate pool"
    print(f"OK {path.name}: {len(sub)} rows x {K} ids, {len(flat)} distinct ids, all in pool")
