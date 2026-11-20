"""Load a trained model and its card, refuse a mismatch, predict P(blue wins), and explain it.

scikit-learn and joblib are imported here only, and only inside `Predictor.load`, so the rest of the app (the crawler,
the dataset builder, the renderer) runs without the `bot` dependency group.

Explanation is exact for the logistic-regression pipeline: every step but the last is applied, and each column's
contribution to the log-odds is `coef * transformed value`. The contributions plus the intercept sum to the model's
log-odds, so a driver is a share of the number, not a story told afterwards. A non-linear model gets no drivers.

Security: joblib files are pickles and run code on load. Only load models you trained yourself.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from oracle.bot.errors import ModelError
from oracle.features import FEATURE_CODE_VERSION, Features, FeatureParams

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ModelCard:
    saved_at: str
    dataset_id: str
    feature_code_version: int
    params: dict[str, Any]
    features: tuple[str, ...]
    sklearn_version: str | None

    @classmethod
    def parse(cls, raw: Mapping[str, Any], where: Path) -> ModelCard:
        try:
            return cls(
                saved_at=str(raw["saved_at"]), dataset_id=str(raw["dataset_id"]),
                feature_code_version=int(raw["feature_code_version"]), params=dict(raw["params"]),
                features=tuple(raw["features"]), sklearn_version=raw.get("sklearn_version"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelError(f"{where}/model_card.json is not a model card ({type(exc).__name__}: {exc}). "
                             "Save models with trainlib.save_model.") from exc

    def mismatches(self, params: FeatureParams) -> list[str]:
        """Reasons this model must not serve the running feature code (empty = compatible)."""
        problems = []
        if self.feature_code_version != FEATURE_CODE_VERSION:
            problems.append(f"feature code version: model {self.feature_code_version}, running {FEATURE_CODE_VERSION}")
        running = json.loads(params.to_json())
        for key in sorted(running.keys() | self.params.keys()):
            if self.params.get(key) != running.get(key):
                problems.append(f"param {key}: model {self.params.get(key)!r}, running {running.get(key)!r}")
        return problems


@dataclass(frozen=True, slots=True)
class Contribution:
    column: str     # a model input column, e.g. 'd_mean_wr_shrunk' or 'missingindicator_lane_TOP_kda'
    value: float    # what the model saw after imputing and scaling
    logit: float    # coef * value: log-odds that BLUE wins


@dataclass(frozen=True, slots=True)
class Prediction:
    p_blue: float
    intercept: float                         # log-odds of blue at an all-average game: the side advantage
    contributions: tuple[Contribution, ...]  # empty when the model cannot explain itself


@dataclass(frozen=True)
class Predictor:
    model: Any
    card: ModelCard
    model_id: str
    warnings: tuple[str, ...]

    @classmethod
    def load(cls, directory: Path, params: FeatureParams | None = None) -> Predictor:
        import joblib
        import sklearn

        card_path, model_path = directory / "model_card.json", directory / "model.joblib"
        if not card_path.is_file() or not model_path.is_file():
            raise ModelError(f"no model in {directory} (needs model.joblib and model_card.json). "
                             "Train one in training/ and save it with trainlib.save_model.")
        card = ModelCard.parse(json.loads(card_path.read_text()), directory)
        problems = card.mismatches(params or FeatureParams())
        if problems:
            raise ModelError(f"{directory.name} was trained on different features and would give wrong numbers. "
                             f"Rebuild the dataset and retrain. Differences: {'; '.join(problems)}")
        warnings = []
        if card.sklearn_version is None:
            warnings.append(f"model card has no sklearn_version (running scikit-learn {sklearn.__version__}); "
                            "retrain with the current trainlib to record it")
        elif card.sklearn_version != sklearn.__version__:
            warnings.append(f"model was saved with scikit-learn {card.sklearn_version}, running {sklearn.__version__}: "
                            "predictions may differ or loading may fail")
        for warning in warnings:
            log.warning(warning)
        return cls(joblib.load(model_path), card, f"{directory.name} {card.dataset_id} {card.saved_at}", tuple(warnings))

    def predict(self, features: Features) -> Prediction:
        """P(blue wins) for the single game in `features`, with the exact log-odds contributions if available."""
        missing = [c for c in self.card.features if c not in features.matches.columns]
        if missing:
            raise ModelError(f"the feature code no longer produces {missing[:3]}: the model needs retraining")
        x = features.matches[list(self.card.features)]
        if len(x) != 1:
            raise ValueError(f"predict takes one game, got {len(x)}")
        classes = list(self.model.classes_)
        p_blue = float(self.model.predict_proba(x)[0, classes.index(1)])
        explained = _explain(self.model, x)
        if explained is None:
            return Prediction(p_blue, 0.0, ())
        return Prediction(p_blue, *explained)


def _explain(model: Any, x: Any) -> tuple[float, tuple[Contribution, ...]] | None:
    """(intercept, contributions) for a linear final step, else None. Never raises for a model that is not linear."""
    steps = getattr(model, "steps", None)
    final = steps[-1][1] if steps else model
    coef, intercept = getattr(final, "coef_", None), getattr(final, "intercept_", None)
    if coef is None or intercept is None or np.ndim(coef) != 2 or coef.shape[0] != 1:
        return None
    try:
        if steps and len(steps) > 1:
            before = model[:-1]
            transformed = np.asarray(before.transform(x), dtype=float)
            names = list(before.get_feature_names_out(list(x.columns)))
        else:
            transformed, names = np.asarray(x, dtype=float), list(x.columns)
    except (AttributeError, ValueError):
        return None
    if transformed.shape != (1, coef.shape[1]) or len(names) != coef.shape[1]:
        return None
    logits = coef[0] * transformed[0]
    return float(intercept[0]), tuple(Contribution(n, float(v), float(c)) for n, v, c in zip(names, transformed[0], logits, strict=True))
