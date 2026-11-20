"""Feature hyperparameters. They are part of the dataset version: change one and the version id changes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

# Bump when the LOGIC of any feature changes (not for docstrings). Recorded in every dataset manifest and
# model card, so a model can refuse features built by different code.
FEATURE_CODE_VERSION = 1


@dataclass(frozen=True, slots=True)
class FeatureParams:
    queue_id: int = 420                 # ranked solo/duo only
    history_n: int = 20                 # prior games per player; MUST equal the crawler's DatasetSpec.history_n
    recent_n: int = 5                   # short form window ("last 5 games")
    session_hours: float = 12.0         # "today": games that ended within this many hours before T
    streak_cap: int = 10
    # Beta-Binomial shrinkage = add kappa pseudo-games at the prior mean. kappa ~ 0.25 / sd^2 - 1 for a prior
    # mean of 0.5 and a between-group sd of true win rates: players ~5 % -> ~100, champions ~2.9 % -> ~300.
    kappa_player: float = 100.0
    kappa_player_champ: float = 5.0     # player's win rate ON this champion, shrunk toward their overall rate
    kappa_meta: float = 300.0
    champ_window_days: float = 30.0     # champion meta win rate: games that ended in [T - 30 d, T)
    role_window_days: float = 90.0      # champion role shares for role inference
    role_prior_strength: float = 3.0    # pseudo-games the champion's role prior is worth (see roles.py)
    # 0 = strict (only snapshots taken before the game ended: clean). > 0 also accepts snapshots taken up to this
    # many days AFTER the game. That leaks (the snapshot contains the game's LP/mastery change) and exists only
    # to MEASURE the leak in a notebook. Never train a bot model with it.
    snapshot_lag_days: float = 0.0

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @property
    def id(self) -> str:
        blob = f"{FEATURE_CODE_VERSION}|{self.to_json()}".encode()
        return hashlib.sha256(blob).hexdigest()[:10]
