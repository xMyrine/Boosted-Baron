"""Small helpers shared by the training notebooks. Deliberately standalone: it needs pandas, numpy,
scikit-learn and matplotlib, NOT the `oracle` package, so the whole `training/` folder works on Google Colab.

Everything that could leak lives in the dataset bundle (built by `oracle dataset`). This file only does
evaluation plumbing: loading, time-based splits, metrics, calibration plots, saving models.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

LABEL = "y_blue_win"


# ---------------------------------------------------------------- loading

@dataclass
class Bundle:
    matches: pd.DataFrame     # one row per game, sorted by start time; index = game id
    seats: pd.DataFrame       # one row per player per game (for jokes now, performance models later)
    manifest: dict
    path: Path

    @property
    def features(self) -> list[str]:
        return list(self.manifest["columns"]["match_features"])

    def feature_group(self, *prefixes: str) -> list[str]:
        """e.g. feature_group('d_mean_') or feature_group('d_mean_', 'lane_')."""
        return [c for c in self.features if c.startswith(prefixes)]

    def describe(self) -> str:
        m = self.manifest
        return (f"{m['dataset_id']}  built {m['built_at']}\n"
                f"rows: {m['rows']['matches']} matches, {m['rows']['seats']} seats "
                f"(skipped: {m['skipped']})\n"
                f"time: {m['time_range']}\npatches: {', '.join(m['patches'])}\n"
                f"features: {len(self.features)} match-level, {len(m['columns']['seat_features'])} per seat")


def find_bundle(data_dir: str | Path) -> Path:
    """The most recently built bundle under data_dir (a folder containing manifest.json)."""
    data_dir = Path(data_dir)
    candidates = [p.parent for p in data_dir.glob("*/manifest.json")]
    if (data_dir / "manifest.json").exists():
        candidates.append(data_dir)
    if not candidates:
        raise FileNotFoundError(f"no dataset bundle under {data_dir}. Run `uv run oracle dataset` first.")
    return max(candidates, key=lambda p: json.loads((p / "manifest.json").read_text())["built_at"])


def load(path: str | Path) -> Bundle:
    path = Path(path)
    if not (path / "manifest.json").exists():
        path = find_bundle(path)
    return Bundle(pd.read_parquet(path / "matches.parquet").sort_values("as_of_ms"),
                  pd.read_parquet(path / "seats.parquet"),
                  json.loads((path / "manifest.json").read_text()), path)


# ---------------------------------------------------------------- splits

@dataclass
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def __repr__(self) -> str:
        def span(d: pd.DataFrame) -> str:
            if d.empty:
                return "empty"
            a, b = (pd.to_datetime(d["as_of_ms"].agg(f), unit="ms").strftime("%Y-%m-%d") for f in ("min", "max"))
            return f"{len(d):>5} rows  {a} .. {b}  blue win {d[LABEL].mean():.3f}"
        return f"train {span(self.train)}\nval   {span(self.val)}\ntest  {span(self.test)}"


def time_split(df: pd.DataFrame, val: float = 0.15, test: float = 0.15, embargo_h: float = 1.0) -> Split:
    """Train < val < test by game start time. Why not random: player form drifts and the same players reappear,
    so a random split lets the model peek at the future. The embargo drops rows starting within `embargo_h`
    hours after the previous part ends (a game still running across the boundary can't link them)."""
    df = df.sort_values("as_of_ms")
    n = len(df)
    i_val, i_test = int(n * (1 - val - test)), int(n * (1 - test))
    train, v, t = df.iloc[:i_val], df.iloc[i_val:i_test], df.iloc[i_test:]
    gap = embargo_h * 3_600_000
    if len(train):
        v = v[v["as_of_ms"] >= train["end_ms"].max() + gap]
    if len(v):
        t = t[t["as_of_ms"] >= v["end_ms"].max() + gap]
    return Split(train, v, t)


def rolling_origin(df: pd.DataFrame, n_folds: int = 4, min_train: float = 0.4) -> list[tuple[np.ndarray, np.ndarray]]:
    """Forward-chaining CV folds for sklearn's `cv=` (train on the past, validate on the next block).
    Use on TRAIN only: the val/test parts stay untouched for honest final numbers."""
    n = len(df)
    start = int(n * min_train)
    edges = np.linspace(start, n, n_folds + 1).astype(int)
    return [(np.arange(0, edges[i]), np.arange(edges[i], edges[i + 1])) for i in range(n_folds)
            if edges[i + 1] > edges[i]]


# ---------------------------------------------------------------- metrics

def evaluate(y: np.ndarray | pd.Series, p: np.ndarray, name: str = "model") -> dict:
    """Probability metrics first. Log loss and Brier reward HONEST probabilities; accuracy only checks the side
    of 50 %. Constant-prior log loss is ~0.693: beating it by a few thousandths is a real result here."""
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    y = np.asarray(y)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    both = len(np.unique(y)) == 2
    return {"model": name, "n": len(y), "log_loss": log_loss(y, p, labels=[0, 1]),
            "brier": brier_score_loss(y, p), "auc": roc_auc_score(y, p) if both else np.nan,
            "accuracy": float(((p > 0.5) == y).mean())}


def compare(rows: list[dict]) -> pd.DataFrame:
    out = pd.DataFrame(rows).set_index("model")
    base = out["log_loss"].iloc[0]
    out["vs_first_logloss"] = out["log_loss"] - base   # negative = better than the first row (the baseline)
    return out.round(4)


def bootstrap_ci(y, p, metric=None, groups=None, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95 % interval of a metric by resampling whole GROUPS (e.g. weeks): rows share players and days, so
    resampling single rows would pretend we have more independent evidence than we do."""
    from sklearn.metrics import log_loss

    metric = metric or (lambda yy, pp: log_loss(yy, np.clip(pp, 1e-6, 1 - 1e-6), labels=[0, 1]))
    y, p = np.asarray(y), np.asarray(p)
    groups = np.arange(len(y)) if groups is None else np.asarray(groups)
    uniq = np.unique(groups)
    idx_by_group = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        pick = np.concatenate([idx_by_group[g] for g in rng.choice(uniq, size=len(uniq), replace=True)])
        stats.append(metric(y[pick], p[pick]))
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def week_of(df: pd.DataFrame) -> np.ndarray:
    return (df["as_of_ms"] // (7 * 86_400_000)).to_numpy()


# ---------------------------------------------------------------- plots

def reliability_plot(y, probs: dict[str, np.ndarray], bins: int = 10, ax=None):
    """Calibration curve: in games we called '70 %', did blue win ~70 % of the time? The diagonal is perfect."""
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    ax = ax or plt.subplots(figsize=(5, 5))[1]
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1, label="perfect")
    for name, p in probs.items():
        frac, mean = calibration_curve(np.asarray(y), np.asarray(p), n_bins=bins, strategy="quantile")
        ax.plot(mean, frac, marker="o", label=name)
    ax.set_xlabel("predicted P(blue wins)")
    ax.set_ylabel("observed blue win rate")
    ax.legend()
    return ax


# ---------------------------------------------------------------- saving

def save_model(directory: str | Path, model, bundle: Bundle, features: list[str], metrics: list[dict],
               notes: str = "") -> Path:
    """joblib model + a JSON model card: which dataset/feature code/params trained it, and how well it did.
    The bot refuses a model whose feature_code_version/params don't match its own feature code, and warns when the
    scikit-learn version recorded here differs from the one it runs: a pickle only loads reliably on the same version."""
    import joblib
    import sklearn

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, directory / "model.joblib")
    m = bundle.manifest
    card = {"saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "dataset_id": m["dataset_id"],
            "feature_code_version": m["feature_code_version"], "params": m["params"],
            "data_time_range": m["time_range"], "patches": m["patches"], "features": features,
            "metrics": metrics, "notes": notes, "sklearn_version": sklearn.__version__}
    (directory / "model_card.json").write_text(json.dumps(card, indent=2, default=float))
    return directory
