"""Leak-free, model-ready features. One code path for training (batch) and the bot (one live game).

Private: `_sources` (all SQL). Everything a feature may know at game start enters through `Pregame` or the
seat table with the same columns; outcomes and true roles exist only as labels in `Dataset`.
"""

from oracle.features._sources import LeakageError
from oracle.features.dataset import Dataset, Features, build_features, features_for_live
from oracle.features.params import FEATURE_CODE_VERSION, FeatureParams
from oracle.features.pregame import Pregame, Seat, Side
from oracle.features.roles import ROLES
from oracle.features.seat import FEATURE_DOCS

__all__ = [
    "FEATURE_CODE_VERSION", "FEATURE_DOCS", "ROLES", "Dataset", "FeatureParams", "Features", "LeakageError",
    "Pregame", "Seat", "Side", "build_features", "features_for_live",
]
