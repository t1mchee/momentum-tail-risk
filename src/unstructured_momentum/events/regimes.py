"""Let the data say how many reversal regimes there are.

Rather than imposing a taxonomy from the literature, this assembles a characteristic
vector for each detected episode and clusters it. The honest constraint is sample size:
there are on the order of 25-30 distinct modern-era episodes. At that N, *any* clustering
algorithm will return clusters, silhouette will pick a k, and the result will look
convincing while being noise.

So the primary model-selection criterion here is **stability under resampling**, not fit.
`select_k` bootstraps the episode set, re-clusters, and measures how consistently pairs of
episodes land together (adjusted Rand index against the full-sample labels). A k whose
partition survives resampling is worth interpreting; one that does not is an artefact,
and the correct output in that case is "the data do not support a partition", which is a
legitimate finding rather than a failure.

Features are chosen to span the mechanisms that are *a priori* plausible without assuming
how many there are:

* severity            -- depth, vol-standardized depth
* breadth             -- how many constructions saw it
* market context      -- return during, prior two-year market, volatility
* deleveraging        -- cross-factor joint stress, dispersion, correlation
* leg mechanics       -- winner/loser/spread betas at the pre-episode peak

All context features are sampled at ``peak_at``, so every one of them is knowable before
the episode begins. Nothing here is contemporaneous with the drawdown except the
deliberately-labelled ``*_at_trough`` columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from ..factor import comovement, constructions

#: Sampled at peak_at (knowable in advance) unless the name says otherwise.
FEATURES = [
    "worst_depth",
    "worst_vol_std",
    "n_constructions",
    "mkt_during",
    "mkt_prior_2y",
    "trail_vol",
    "joint_z_at_trough",
    "dispersion_at_trough",
    "factor_corr_at_peak",
    "winner_beta_at_peak",
    "loser_beta_at_peak",
]


def episode_features(consensus: pd.DataFrame, stacked: pd.DataFrame) -> pd.DataFrame:
    """Attach deleveraging and leg-mechanics context to consensus events."""
    cm = comovement.panel()
    betas = constructions.leg_betas("vw")

    peaks = (
        stacked.merge(
            consensus.reset_index()[["event_id", "trough_at"]],
            left_on="trough_at",
            right_on="trough_at",
            how="inner",
        )
        .groupby("event_id")["peak_at"]
        .min()
    )

    out = consensus.copy()
    out["peak_at"] = peaks

    def at(idx: pd.Series, frame: pd.DataFrame, col: str) -> np.ndarray:
        return frame[col].reindex(pd.DatetimeIndex(idx), method="ffill").to_numpy()

    out["joint_z_at_trough"] = at(out["trough_at"], cm, "joint_z")
    out["dispersion_at_trough"] = at(out["trough_at"], cm, "factor_dispersion")
    out["factor_corr_at_peak"] = at(out["peak_at"], cm, "factor_avg_corr")
    out["winner_beta_at_peak"] = at(out["peak_at"], betas, "winner_beta")
    out["loser_beta_at_peak"] = at(out["peak_at"], betas, "loser_beta")
    return out


def _matrix(feat: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    X = feat[FEATURES].astype(float)
    X = X.dropna()
    return StandardScaler().fit_transform(X), X


def select_k(
    feat: pd.DataFrame,
    *,
    kmax: int = 6,
    n_boot: int = 200,
    frac: float = 0.8,
    seed: int = 0,
) -> pd.DataFrame:
    """Compare k by silhouette AND by bootstrap stability.

    Stability is the criterion that matters at this sample size. For each k we resample
    ``frac`` of episodes, re-cluster, and compare labels on the shared episodes with the
    adjusted Rand index. Mean ARI near 1 means the partition is real; near 0 means the
    algorithm is drawing lines through noise.
    """
    X, frame = _matrix(feat)
    n = len(X)
    rng = np.random.default_rng(seed)
    rows = []

    for k in range(2, min(kmax, n - 1) + 1):
        base = KMeans(n_clusters=k, n_init=25, random_state=seed).fit_predict(X)
        sil = silhouette_score(X, base) if len(set(base)) > 1 else np.nan

        aris = []
        for _ in range(n_boot):
            idx = rng.choice(n, size=max(int(frac * n), k + 1), replace=False)
            lab = KMeans(n_clusters=k, n_init=10, random_state=int(rng.integers(1e6))).fit_predict(
                X[idx]
            )
            aris.append(adjusted_rand_score(base[idx], lab))

        rows.append(
            {
                "k": k,
                "silhouette": sil,
                "stability_ari_mean": float(np.mean(aris)),
                "stability_ari_p10": float(np.percentile(aris, 10)),
            }
        )
    return pd.DataFrame(rows).set_index("k")


def fit(feat: pd.DataFrame, k: int, *, method: str = "kmeans", seed: int = 0) -> pd.Series:
    """Assign cluster labels. ``method`` is 'kmeans' or 'ward'."""
    X, frame = _matrix(feat)
    model = (
        KMeans(n_clusters=k, n_init=50, random_state=seed)
        if method == "kmeans"
        else AgglomerativeClustering(n_clusters=k, linkage="ward")
    )
    return pd.Series(model.fit_predict(X), index=frame.index, name="cluster")


def agreement(feat: pd.DataFrame, k: int) -> float:
    """Do k-means and Ward agree? Cross-algorithm agreement is a second stability check."""
    return adjusted_rand_score(fit(feat, k, method="kmeans"), fit(feat, k, method="ward"))


def profile(feat: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Cluster means on the raw (unstandardized) features, for interpretation."""
    df = feat.loc[labels.index, FEATURES].copy()
    df["cluster"] = labels
    prof = df.groupby("cluster").mean()
    prof["n"] = df.groupby("cluster").size()
    return prof
