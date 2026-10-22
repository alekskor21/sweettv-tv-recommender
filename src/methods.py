"""Individual recommenders behind one interface: fit(...) then score(uids, cand_ids) -> matrix.

Signal for every method: number of airings a user watched of a programme (`n_airings`), the
quantity the leaderboard appears to rank by (see 14_count_hypothesis). Unknown items score -inf.
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import pandas as pd
import scipy.sparse as sps
from implicit.als import AlternatingLeastSquares
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from common import K, SEED, fill_to_k


def _matrix(inter, n_users, items, values):
    return sps.csr_matrix((values.astype(np.float32), (inter["uid"].values, items.get_indexer(inter["tv_show_id"]))),
                          shape=(n_users, len(items)))


def _expand(scores_known, items, cand_ids):
    """Place scores for known items into a matrix over cand_ids; unknown items get -inf."""
    col = items.get_indexer(cand_ids)
    out = np.full((scores_known.shape[0], len(cand_ids)), -np.inf, dtype=np.float32)
    ok = col >= 0
    out[:, ok] = scores_known[:, col[ok]]
    return out


class Popularity:
    """Distinct viewers in the training window (required reference)."""

    def fit(self, inter):
        self.pop = inter.groupby("tv_show_id")["uid"].nunique()
        return self

    def score(self, uids, cand_ids):
        s = self.pop.reindex(cand_ids).fillna(0).values.astype(np.float32)
        return np.tile(s, (len(uids), 1))


class PersonalHistory:
    """The user's own airing counts (re-watch); not one of the required methods."""

    def fit(self, inter):
        self.inter = inter.set_index(["uid", "tv_show_id"])["n_airings"]
        return self

    def score(self, uids, cand_ids):
        out = np.zeros((len(uids), len(cand_ids)), dtype=np.float32)
        pos = pd.Index(cand_ids)
        sub = self.inter[self.inter.index.get_level_values(0).isin(uids)].reset_index()
        row = pd.Index(uids).get_indexer(sub["uid"])
        col = pos.get_indexer(sub["tv_show_id"])
        ok = col >= 0
        out[row[ok], col[ok]] = sub["n_airings"].values[ok]
        return out


class ALS:
    """Implicit-feedback MF (Hu, Koren, Volinsky 2008): confidence 1 + alpha*log1p(n_airings)
    on pairs with n_airings >= min_airings."""

    def __init__(self, factors=32, reg=0.1, alpha=1.0, min_airings=1, iterations=20):
        self.p = dict(factors=factors, reg=reg, alpha=alpha, min_airings=min_airings, iterations=iterations)

    def fit(self, inter, n_users):
        inter = inter[inter["n_airings"] >= self.p["min_airings"]]
        self.items = pd.Index(np.sort(inter["tv_show_id"].unique()))
        m = _matrix(inter, n_users, self.items, 1 + self.p["alpha"] * np.log1p(inter["n_airings"].values))
        self.model = AlternatingLeastSquares(factors=self.p["factors"], regularization=self.p["reg"],
                                             iterations=self.p["iterations"], random_state=SEED, num_threads=8)
        self.model.fit(m, show_progress=False)
        return self

    def score(self, uids, cand_ids):
        known = self.model.user_factors[uids] @ self.model.item_factors.T
        return _expand(known, self.items, cand_ids)


class PureSVD:
    """MF by truncated SVD of the log1p(n_airings) matrix (Cremonesi, Koren, Turrin 2010)."""

    def __init__(self, factors=32):
        self.factors = factors

    def fit(self, inter, n_users):
        self.items = pd.Index(np.sort(inter["tv_show_id"].unique()))
        x = _matrix(inter, n_users, self.items, np.log1p(inter["n_airings"].values)).astype(np.float64)
        u, s, vt = svds(x, k=self.factors, random_state=SEED)
        self.U, self.V = (u * s).astype(np.float32), vt.T.astype(np.float32)
        return self

    def score(self, uids, cand_ids):
        return _expand(self.U[uids] @ self.V.T, self.items, cand_ids)


def split_names(x, limit=None):
    if not isinstance(x, str):
        return []
    names = [n.strip() for n in x.split(",") if n.strip()]
    return names[:limit] if limit else names


def item_tokens(sched_full, max_week, fields=("cat", "genre", "decade", "ch", "dir", "act")):
    """Metadata tokens per programme id from schedule rows up to ISO week `max_week`."""
    s = sched_full[(sched_full["tv_show_id"] != 0) & (sched_full["week"] <= max_week)]
    s = s.drop_duplicates(["tv_show_id", "channel_id", "tv_show_genre_1", "director", "actors"])
    toks = {}
    for row in s.itertuples():
        t = toks.setdefault(row.tv_show_id, set())
        if "cat" in fields:
            t.add(f"cat={row.tv_show_category}")
        if "genre" in fields:
            t.update(f"genre={g}" for g in (row.tv_show_genre_1, row.tv_show_genre_2, row.tv_show_genre_3) if isinstance(g, str))
        if "dir" in fields:
            t.update(f"dir={d}" for d in split_names(row.director))
        if "act" in fields:
            t.update(f"act={a}" for a in split_names(row.actors, 5))
        if "decade" in fields and isinstance(row.year_of_production, str) and row.year_of_production[:4].isdigit():
            t.add(f"decade={int(row.year_of_production[:4]) // 10 * 10}")
        if "ch" in fields:
            t.add(f"ch={row.channel_id}")
    return toks


class ContentBased:
    """TF-IDF item vectors from schedule metadata; profile = sum of item vectors weighted by
    log1p(n_airings) over pairs with n_airings >= min_airings; score = cosine."""

    def __init__(self, min_airings=1, fields=("cat", "genre", "decade", "ch", "dir", "act")):
        self.min_airings, self.fields = min_airings, fields

    def fit(self, inter, n_users, sched_full, max_week):
        toks = item_tokens(sched_full, max_week, self.fields)
        self.items = pd.Index(sorted(toks))
        self.vec = TfidfVectorizer(analyzer=lambda x: x, min_df=2, sublinear_tf=True)
        self.mat = normalize(self.vec.fit_transform([sorted(toks[i]) for i in self.items]))
        inter = inter[(inter["n_airings"] >= self.min_airings) & inter["tv_show_id"].isin(self.items)]
        ui = _matrix(inter, n_users, self.items, np.log1p(inter["n_airings"].values))
        self.profiles = normalize(ui @ self.mat)
        return self

    def score(self, uids, cand_ids):
        known = (self.profiles[uids] @ self.mat.T).toarray()
        return _expand(known, self.items, cand_ids)


def top_k(scores, uids, cand_ids, fallback, k=K):
    """Turn a score matrix into exactly k ids per user: best finite scores first, then popularity."""
    cand_ids = np.asarray(cand_ids)
    order = np.argsort(-scores, axis=1)[:, :k]
    recs = {}
    for i, u in enumerate(uids):
        ranked = [cand_ids[j] for j in order[i] if np.isfinite(scores[i, j]) and scores[i, j] > 0]
        recs[u] = fill_to_k(ranked, fallback, k)
    return recs
