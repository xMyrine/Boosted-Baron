"""Per-seat features: one row per (game, seat), computed from that player's past only.

THE place features live. Each family is one small vectorised function `(ctx) -> DataFrame on ctx.index`.
To add a feature: add a column in a family (or a new family to FAMILIES) and a line to FEATURE_DOCS.
`compute()` checks that every family returns exactly the seat index, so pandas can't silently misalign rows.

History H = the seat's usable prior games (ended before T, not remakes, at most history_n), most recent first.
Rates are POOLED (sum of kills / sum of minutes), not averages of per-game rates: long games weigh more,
and one 15-minute stomp can't dominate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from oracle.features._sources import DAY_MS
from oracle.features.params import FeatureParams
from oracle.features.roles import ROLES, ChampionMeta, infer_roles, role_history_counts

KEY = ["game", "seat"]

# name -> (definition, why it might predict). Shown in the notebooks; every numeric feature must be listed.
FEATURE_DOCS: dict[str, tuple[str, str]] = {
    # form
    "n_games": ("usable history games (0..history_n)", "new / returning / smurf accounts; data quality"),
    "wr_shrunk": ("(W + k/2) / (N + k), k = kappa_player", "form matchmaking hasn't absorbed yet (weak: expect noise)"),
    "wr_recent": ("raw win share in the last recent_n games", "short-term form"),
    "streak": ("signed run from the most recent game, +W / -L, capped", "momentum or tilt; joke material"),
    "session_games": ("games that ended within session_hours before T", "long sessions: fatigue / tilt"),
    "session_losses": ("losses among those", "tilt queue"),
    "hours_since_last": ("log1p(hours since the last usable game ended)", "rust after a break"),
    # performance (pooled over H)
    "kda": ("ln((K + A + 1) / (D + 1))", "skill beyond W/L"),
    "kills_pm": ("kills per minute", "aggression / carry potential"),
    "deaths_pm": ("deaths per minute", "feeding"),
    "assists_pm": ("assists per minute", "teamplay"),
    "deaths_max": ("most deaths in one history game", "'has fed recently'"),
    "dmg_pm": ("damage to champions per minute", "carry potential"),
    "dmg_taken_pm": ("damage taken per minute", "frontline / or just getting caught"),
    "gold_pm": ("gold per minute", "economy"),
    "cs_pm": ("creep score per minute", "mechanics proxy"),
    "vision_pm": ("vision score per minute", "map awareness (ward score)"),
    "kp": ("(K + A) / team kills", "involvement"),
    "dmg_share": ("damage / team damage", "how much the team relies on them"),
    # champion
    "champ_games": ("history games on the champion picked now", "first-timers lose"),
    "champ_wr_shrunk": ("win rate on this champion, shrunk toward their overall rate", "comfort pick"),
    "days_since_champ": ("days since they last played it (NaN if not in history)", "rusty on the pick"),
    "champ_meta_wr": ("champion win rate, all stored games ended in [T-30d, T), shrunk (kappa_meta)", "meta strength"),
    "champ_meta_games": ("log1p(those games)", "popularity / blind-pick safety"),
    # role
    "role_share": ("share of history played in the inferred role", "on-role vs off-role"),
    "off_role": ("1 if inferred role != main role (most played in history)", "autofill / off-role"),
    "role_conf": ("champion's as-of share of picks in the inferred role", "weird picks for the lane"),
    # snapshots (mostly NaN until the live watcher has collected clean ones)
    "rank_score": ("tier*4 + division + LP/100 from the latest clean league snapshot", "skill"),
    "ranked_games": ("season ranked games (wins + losses) at that snapshot", "experience"),
    "ranked_wr_shrunk": ("season ranked win rate, shrunk (kappa_player)", "season form"),
    "mastery_points_log": ("log1p(champion mastery points) on the pick, latest clean snapshot", "champion experience"),
    "mastery_level": ("champion mastery level on the pick", "champion experience"),
}

CATEGORICAL = ["role", "main_role"]   # string columns, kept for jokes, lane matching and analysis


@dataclass(frozen=True)
class SeatContext:
    seats: pd.DataFrame        # indexed (game, seat): side, puuid, champion_id, as_of_ms, snapshot_until_ms
    history: pd.DataFrame      # see _sources.history; has a `same_champ` column added
    snaps: pd.DataFrame        # see _sources.snapshots, indexed (game, seat)
    meta: ChampionMeta
    p: FeatureParams

    @property
    def index(self) -> pd.MultiIndex:
        return self.seats.index

    def per_seat(self, s: pd.Series | pd.DataFrame, fill: float = np.nan) -> pd.Series | pd.DataFrame:
        return s.reindex(self.index, fill_value=fill)


def _form(ctx: SeatContext) -> pd.DataFrame:
    h, p = ctx.history, ctx.p
    g = h.groupby(KEY)
    n = ctx.per_seat(g.size(), 0.0)
    w = ctx.per_seat(g["win"].sum(), 0.0)
    recent = h[h["games_ago"] < p.recent_n].groupby(KEY)["win"].mean()

    first = g["win"].transform("first")
    broken = (h["win"] != first).astype(int).groupby([h["game"], h["seat"]]).cummax()
    run = (broken == 0).groupby([h["game"], h["seat"]]).sum()
    sign = g["win"].first().map({1: 1, 0: -1})
    streak = (run * sign).clip(-p.streak_cap, p.streak_cap)

    age_h = (h["as_of_ms"] - h["prior_end_ms"]) / 3_600_000
    in_session = h[age_h <= p.session_hours]
    return pd.DataFrame({
        "n_games": n,
        "wr_shrunk": (w + p.kappa_player / 2) / (n + p.kappa_player),
        "wr_recent": ctx.per_seat(recent),
        "streak": ctx.per_seat(streak, 0.0),
        "session_games": ctx.per_seat(in_session.groupby(KEY).size(), 0.0),
        "session_losses": ctx.per_seat((1 - in_session["win"]).groupby([in_session["game"], in_session["seat"]]).sum(), 0.0),
        "hours_since_last": ctx.per_seat(np.log1p(age_h.groupby([h["game"], h["seat"]]).min())),
    })


def _performance(ctx: SeatContext) -> pd.DataFrame:
    s = ctx.history.groupby(KEY)[["kills", "deaths", "assists", "damage_to_champs", "damage_taken", "gold_earned",
                                  "cs", "vision_score", "minutes", "team_kills", "team_damage"]].sum()
    s = ctx.per_seat(s)   # NaN for players without history
    mins = s["minutes"].where(s["minutes"] > 0)
    return pd.DataFrame({
        "kda": np.log((s["kills"] + s["assists"] + 1) / (s["deaths"] + 1)),
        "kills_pm": s["kills"] / mins,
        "deaths_pm": s["deaths"] / mins,
        "assists_pm": s["assists"] / mins,
        "deaths_max": ctx.per_seat(ctx.history.groupby(KEY)["deaths"].max()),
        "dmg_pm": s["damage_to_champs"] / mins,
        "dmg_taken_pm": s["damage_taken"] / mins,
        "gold_pm": s["gold_earned"] / mins,
        "cs_pm": s["cs"] / mins,
        "vision_pm": s["vision_score"] / mins,
        "kp": (s["kills"] + s["assists"]) / s["team_kills"].where(s["team_kills"] > 0),
        "dmg_share": s["damage_to_champs"] / s["team_damage"].where(s["team_damage"] > 0),
    })


def _champion(ctx: SeatContext, overall_wr: pd.Series) -> pd.DataFrame:
    h, p = ctx.history, ctx.p
    on = h[h["same_champ"]]
    g = on.groupby(KEY)
    nc = ctx.per_seat(g.size(), 0.0)
    wc = ctx.per_seat(g["win"].sum(), 0.0)
    last_end = ctx.per_seat(g["prior_end_ms"].max())
    t = ctx.seats["as_of_ms"].to_numpy()
    games, wins = ctx.meta.win_rate(ctx.seats["champion_id"].to_numpy(), t, p.champ_window_days)
    return pd.DataFrame({
        "champ_games": nc,
        "champ_wr_shrunk": (wc + p.kappa_player_champ * overall_wr) / (nc + p.kappa_player_champ),
        "days_since_champ": (ctx.seats["as_of_ms"] - last_end) / DAY_MS,
        "champ_meta_wr": (wins + p.kappa_meta / 2) / (games + p.kappa_meta),
        "champ_meta_games": np.log1p(games),
    }, index=ctx.index)


def _role(ctx: SeatContext) -> pd.DataFrame:
    counts = role_history_counts(ctx.index, ctx.history)
    t = ctx.seats["as_of_ms"].to_numpy()
    shares = ctx.meta.role_shares(ctx.seats["champion_id"].to_numpy(), t, ctx.p.role_window_days)
    role = infer_roles(ctx.seats, counts, shares, prior_strength=ctx.p.role_prior_strength)
    n = counts.sum(axis=1)
    has = n > 0
    rows = np.arange(len(role))
    main = np.where(has, counts.argmax(axis=1), -1)
    return pd.DataFrame({
        "role": np.array(ROLES, dtype=object)[role],
        "main_role": np.where(has, np.array(ROLES, dtype=object)[np.maximum(main, 0)], None),
        "role_share": np.where(has, counts[rows, role] / np.maximum(n, 1), np.nan),
        "off_role": np.where(has, (main != role).astype(float), np.nan),
        "role_conf": shares[rows, role],
    }, index=ctx.index)


def _snapshots(ctx: SeatContext) -> pd.DataFrame:
    s, p = ctx.snaps, ctx.p
    return pd.DataFrame({
        "rank_score": s["rank_score"],
        "ranked_games": s["ranked_games"],
        "ranked_wr_shrunk": (s["ranked_wins"] + p.kappa_player / 2) / (s["ranked_games"] + p.kappa_player),
        "mastery_points_log": np.log1p(s["mastery_points"]),
        "mastery_level": s["mastery_level"],
    }, index=ctx.index)


def compute(seats: pd.DataFrame, history: pd.DataFrame, snaps: pd.DataFrame, meta: ChampionMeta,
            p: FeatureParams) -> pd.DataFrame:
    """All seat features. `seats` sorted by (game, seat), 10 per game. Returns one row per seat, same order."""
    seats = seats.set_index(KEY)
    picks = seats[["champion_id"]].rename(columns={"champion_id": "pick"}).reset_index()
    history = history.merge(picks, on=KEY, how="inner")
    history["same_champ"] = history["champion_id"] == history["pick"]
    history = history.sort_values(KEY + ["games_ago"]).reset_index(drop=True)   # families rely on recent-first
    ctx = SeatContext(seats, history, snaps.reindex(seats.index), meta, p)

    form = _form(ctx)
    families: list[Callable[[], pd.DataFrame]] = [
        lambda: form,
        lambda: _performance(ctx),
        lambda: _champion(ctx, form["wr_shrunk"]),
        lambda: _role(ctx),
        lambda: _snapshots(ctx),
    ]
    parts = []
    for fam in families:
        part = fam()
        if not part.index.equals(ctx.index):
            raise AssertionError(f"feature family returned a misaligned index: {list(part.columns)}")
        parts.append(part)
    out = pd.concat(parts, axis=1)
    numeric = [c for c in out.columns if c not in CATEGORICAL]
    missing_docs = set(numeric) - FEATURE_DOCS.keys()
    if missing_docs:
        raise AssertionError(f"undocumented features: {sorted(missing_docs)}")
    out[numeric] = out[numeric].astype(float).replace([np.inf, -np.inf], np.nan)
    return out
