"""Role inference: who plays which lane, from information available at game start.

Why: spectator-v5 (what the bot sees) has no roles, so the true `team_position` must never be a feature.
We infer it the same way in training and live, and keep the true role only to MEASURE the inference.

Model (a tiny naive-Bayes-style score, then an assignment):
    score[player, role] = log( (games player played in role in their history + a * P(role | champion))
                               / (history games + a) )
  P(role | champion) is the as-of share of that champion's recent picks in each role ("Lee Sin is a jungler").
  `a` (role_prior_strength) is how many pseudo-games the champion prior is worth versus the player's own history.
Then each team gets the permutation of the 5 roles with the highest total score (5! = 120 options, brute force:
vectorised over all teams). This guarantees one player per role, like the real game.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from oracle.features._sources import DAY_MS, AsOfCounter

ROLES = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")
_PERMS = np.array(list(itertools.permutations(range(5))))   # (120, 5): perm[k][seat_in_team] = role index


def role_index(roles: pd.Series) -> np.ndarray:
    """'TOP'.. -> 0..4; anything else ('' on remakes, unknown) -> -1."""
    lookup = {r: i for i, r in enumerate(ROLES)}
    return roles.map(lookup).fillna(-1).astype(int).to_numpy()


class ChampionMeta:
    """As-of champion statistics from every stored game: win rate and role shares in [T - window, T)."""

    def __init__(self, events: pd.DataFrame) -> None:
        champ = events["champion_id"].to_numpy()
        end = events["end_ms"].to_numpy()
        win = events["win"].to_numpy()
        self._champ = AsOfCounter(champ, end, win)
        ridx = role_index(events["role"])
        ok = ridx >= 0
        self._role = AsOfCounter(champ[ok] * 8 + ridx[ok], end[ok], win[ok])

    def win_rate(self, champion_id: np.ndarray, t_ms: np.ndarray, window_days: float) -> tuple[np.ndarray, np.ndarray]:
        """(games, wins) of the champion in games that ended in [T - window, T)."""
        return self._champ.window(champion_id, t_ms, int(window_days * DAY_MS))

    def role_shares(self, champion_id: np.ndarray, t_ms: np.ndarray, window_days: float) -> np.ndarray:
        """(n, 5) Laplace-smoothed P(role | champion) as of T."""
        counts = np.stack([self._role.window(champion_id * 8 + r, t_ms, int(window_days * DAY_MS))[0]
                           for r in range(5)], axis=1)
        return (counts + 1.0) / (counts.sum(axis=1, keepdims=True) + 5.0)


def role_history_counts(seats_index: pd.MultiIndex, history: pd.DataFrame) -> np.ndarray:
    """(n_seats, 5) how often each seat's player played each role in their usable history."""
    h = history.assign(r=role_index(history["team_position"]))
    h = h[h["r"] >= 0]
    counts = pd.crosstab([h["game"], h["seat"]], h["r"]).reindex(columns=range(5), fill_value=0)
    return counts.reindex(seats_index, fill_value=0).to_numpy(dtype=float)


def infer_roles(seats: pd.DataFrame, hist_counts: np.ndarray, champ_shares: np.ndarray, *,
                prior_strength: float) -> np.ndarray:
    """Role index 0..4 per seat. `seats` must be sorted by (game, seat) with exactly 10 seats per game."""
    n_hist = hist_counts.sum(axis=1, keepdims=True)
    score = np.log((hist_counts + prior_strength * champ_shares) / (n_hist + prior_strength))   # (n, 5)
    teams = score.reshape(-1, 5, 5)                                     # (n_teams, seat_in_team, role)
    totals = teams[:, np.arange(5), _PERMS].sum(axis=2)                 # (n_teams, 120)
    best = _PERMS[totals.argmax(axis=1)]                                # (n_teams, 5)
    return best.reshape(-1)
