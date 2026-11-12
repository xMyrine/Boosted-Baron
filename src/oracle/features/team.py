"""Seat features -> one row per match, for the win model.

Every column is BLUE minus RED, so swapping sides flips the sign of every column (the model is antisymmetric).
A logistic regression's intercept then measures the real blue-side advantage, and nothing else can.

  d_mean_<f>         team mean difference, for every numeric seat feature
  d_min_<f>/d_max_<f> weakest / strongest player difference, for a curated set ("your team has the first-timer")
  lane_<ROLE>_<f>    the two players in the same inferred role: blue minus red (lane matchups)
  ctx_<f>            symmetric game context (same value either way round), e.g. how high-elo the game is

NaN-aware: a team mean skips players without a value; a difference is NaN if either side has none.
LightGBM uses NaN natively; for logistic regression the training pipeline imputes it (with an indicator).
"""

from __future__ import annotations

import pandas as pd

from oracle.features.roles import ROLES
from oracle.features.seat import CATEGORICAL

MIN_MAX = ["n_games", "wr_shrunk", "kda", "deaths_pm", "champ_games", "champ_meta_wr", "role_share",
           "rank_score", "mastery_points_log"]
LANE = ["wr_shrunk", "kda", "cs_pm", "dmg_pm", "gold_pm", "champ_games", "champ_meta_wr", "off_role",
        "rank_score", "mastery_points_log"]
CONTEXT = ["rank_score", "n_games"]


def match_rows(seats: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """`seats` (game, seat, side, ...) and `feats` (indexed by (game, seat)) -> one row per game, index = game."""
    f = feats.join(seats.set_index(["game", "seat"])[["side"]])
    numeric = [c for c in feats.columns if c not in CATEGORICAL]
    blue, red = f[f["side"] == 100], f[f["side"] == 200]
    bg, rg = blue.groupby(level="game"), red.groupby(level="game")

    parts = [
        (bg[numeric].mean() - rg[numeric].mean()).add_prefix("d_mean_"),
        (bg[MIN_MAX].min() - rg[MIN_MAX].min()).add_prefix("d_min_"),
        (bg[MIN_MAX].max() - rg[MIN_MAX].max()).add_prefix("d_max_"),
        f.groupby(level="game")[CONTEXT].mean().add_prefix("ctx_"),
    ]
    lanes = []
    for role in ROLES:
        b = blue[blue["role"] == role].droplevel("seat")[LANE]
        r = red[red["role"] == role].droplevel("seat")[LANE]
        lanes.append((b - r).add_prefix(f"lane_{role}_"))
    parts.extend(lanes)
    out = pd.concat(parts, axis=1)
    games = pd.Index(seats["game"].drop_duplicates(), name="game")
    return out.reindex(games).astype(float)
