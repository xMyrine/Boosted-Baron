"""Helpers for the bot tests: a tiny trained model saved the way `trainlib.save_model` saves one."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from oracle.features import FEATURE_CODE_VERSION, FeatureParams

ROOT = Path(__file__).resolve().parent.parent
FEATURES = ["d_mean_wr_shrunk", "d_mean_kda", "d_mean_champ_meta_wr", "lane_TOP_kda"]


def training_frame(n: int = 80, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """Synthetic blue-minus-red features where `d_mean_kda` and `d_mean_wr_shrunk` carry the signal.
    `lane_TOP_kda` has holes, so the imputer grows a missing-indicator column for it."""
    rng = np.random.default_rng(seed)
    x = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
    x.loc[::7, "lane_TOP_kda"] = np.nan
    logit = 1.2 * x["d_mean_kda"] + 0.6 * x["d_mean_wr_shrunk"]
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return x, y


def linear_model():
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x, y = training_frame()
    return make_pipeline(SimpleImputer(add_indicator=True), StandardScaler(), LogisticRegression(C=1.0)).fit(x, y)


def boosted_model():
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline

    x, y = training_frame()
    return make_pipeline(SimpleImputer(add_indicator=True), GradientBoostingClassifier(n_estimators=10)).fit(x, y)


def save_model(directory: Path, model, *, card: dict | None = None) -> Path:
    """Write model.joblib and a model card the way trainlib.save_model does."""
    import joblib
    import sklearn

    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, directory / "model.joblib")
    base = {"saved_at": "2026-11-05T12:30:25", "dataset_id": "win_v1@test", "feature_code_version": FEATURE_CODE_VERSION,
            "params": json.loads(FeatureParams().to_json()), "features": FEATURES, "metrics": [], "notes": "",
            "sklearn_version": sklearn.__version__}
    (directory / "model_card.json").write_text(json.dumps({**base, **(card or {})}))
    return directory


def import_trainlib():
    sys.path.insert(0, str(ROOT / "training"))
    try:
        import trainlib
    finally:
        sys.path.remove(str(ROOT / "training"))
    return trainlib
