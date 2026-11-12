"""Private: every SQL query feature building runs. Nothing outside this module talks to SQLite.

All loaders take the seat table (one row per game x seat) and run ONE query for all seats at once, so the
same code serves 50,000 training seats and the bot's 10 live seats. The seats are put in a TEMP table, which
works on a read-only connection and never blocks the crawler.

Leakage rules enforced here (and re-checked by `check_history`):
  * history: a past game is usable for a seat at T only if it ENDED before T (start + duration < T);
  * everything is restricted to what was fetched by the knowledge cut-off K (reproducible datasets);
  * snapshots: only those taken before the seat's `snapshot_until_ms` (training: game end; live: now).
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from typing import Iterator

import numpy as np
import pandas as pd

MINUTE_MS = 60_000
DAY_MS = 86_400_000

# Tier ladder for a single numeric rank. Apex tiers have no divisions: LP carries the order there.
TIERS = ("IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND", "MASTER", "GRANDMASTER", "CHALLENGER")
DIVISIONS = {"IV": 0, "III": 1, "II": 2, "I": 3}


class LeakageError(AssertionError):
    """A loaded row violates the 'only the past' rule. Always a bug: never catch it."""


@contextmanager
def _seat_table(con: sqlite3.Connection, seats: pd.DataFrame) -> Iterator[str]:
    con.execute("DROP TABLE IF EXISTS temp._seat")
    con.execute("CREATE TEMP TABLE _seat (game TEXT, seat INTEGER, puuid TEXT, as_of_ms INTEGER, "
                "snapshot_until_ms INTEGER, PRIMARY KEY (game, seat))")
    try:
        con.executemany("INSERT INTO temp._seat VALUES (?, ?, ?, ?, ?)",
                        seats[["game", "seat", "puuid", "as_of_ms", "snapshot_until_ms"]].itertuples(index=False, name=None))
        yield "temp._seat"
    finally:
        con.execute("DROP TABLE IF EXISTS temp._seat")


# ---------------------------------------------------------------- training anchors (the ONLY label source)

def anchors(con: sqlite3.Connection, *, queue_id: int, knowledge_ms: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Finished anchors and their 10 seats WITH outcomes. Returns (games, seats).

    games: game, depth, as_of_ms, end_ms, duration_min, patch, y_blue_win
    seats: SEAT_COLUMNS + labels (true_role, y_win, y_kills, ...). `snapshot_until_ms` = game end: rank and mastery
           only change once a game is over, so a snapshot taken before the end is clean.
    """
    games = pd.read_sql_query("""
        SELECT a.match_id AS game, a.depth, m.game_start_ms AS as_of_ms,
               m.game_start_ms + m.game_duration_s * 1000 AS end_ms,
               m.game_duration_s / 60.0 AS duration_min, m.patch,
               CAST(m.winning_team_id = 100 AS INTEGER) AS y_blue_win
        FROM anchor a
        JOIN match m ON m.match_id = a.match_id
        JOIN raw_match r ON r.match_id = a.match_id
        WHERE m.queue_id = :q AND m.is_remake = 0 AND m.winning_team_id IS NOT NULL AND r.fetched_at_ms <= :k
        ORDER BY m.game_start_ms, a.match_id""", con, params={"q": queue_id, "k": knowledge_ms})
    seats = pd.read_sql_query("""
        SELECT p.match_id AS game, p.team_id AS side, p.participant_id, p.puuid, p.champion_id,
               m.game_start_ms AS as_of_ms, m.game_start_ms + m.game_duration_s * 1000 AS snapshot_until_ms,
               p.team_position AS true_role, p.win AS y_win, p.kills AS y_kills, p.deaths AS y_deaths,
               p.assists AS y_assists, p.damage_to_champs AS y_damage, p.damage_taken AS y_damage_taken,
               p.gold_earned AS y_gold, p.cs AS y_cs, p.vision_score AS y_vision, p.champ_level AS y_champ_level,
               m.game_duration_s / 60.0 AS y_minutes
        FROM anchor a
        JOIN match m ON m.match_id = a.match_id
        JOIN raw_match r ON r.match_id = a.match_id
        JOIN participant p ON p.match_id = a.match_id
        WHERE m.queue_id = :q AND m.is_remake = 0 AND m.winning_team_id IS NOT NULL AND r.fetched_at_ms <= :k""",
        con, params={"q": queue_id, "k": knowledge_ms})
    # Seat 0..9 = blue then red, by participant id within a side (the same normalisation Pregame applies).
    seats = seats.sort_values(["game", "side", "participant_id"]).reset_index(drop=True)
    seats["seat"] = seats.groupby("game").cumcount()
    return games, seats.drop(columns="participant_id")


def tracked_puuids(con: sqlite3.Connection) -> set[str]:
    return {p for (p,) in con.execute("SELECT puuid FROM tracked_player")}


# ---------------------------------------------------------------- per-seat history

def coverage(con: sqlite3.Connection, seats: pd.DataFrame, *, queue_id: int, history_n: int,
             knowledge_ms: int) -> pd.DataFrame:
    """Per seat: is its history listing present (>= history_n requested) and fully fetched by K?"""
    with _seat_table(con, seats) as t:
        cov = pd.read_sql_query(f"""
            SELECT s.game, s.seat,
                   h.n_requested IS NOT NULL AND h.n_requested >= :n AS listed,
                   (SELECT count(*) FROM history_listing_item li
                     WHERE li.puuid = s.puuid AND li.queue_id = :q AND li.before_ms = s.as_of_ms) AS n_listed,
                   (SELECT count(*) FROM history_listing_item li
                     JOIN raw_match r ON r.match_id = li.match_id AND r.fetched_at_ms <= :k
                     WHERE li.puuid = s.puuid AND li.queue_id = :q AND li.before_ms = s.as_of_ms) AS n_fetched
            FROM {t} s
            LEFT JOIN history_listing h
                   ON h.puuid = s.puuid AND h.queue_id = :q AND h.before_ms = s.as_of_ms AND h.fetched_at_ms <= :k""",
            con, params={"q": queue_id, "n": history_n, "k": knowledge_ms})
    cov["listed"] = cov["listed"].astype(bool)
    cov["complete"] = cov["listed"] & (cov["n_listed"] == cov["n_fetched"])
    return cov.set_index(["game", "seat"])


def history(con: sqlite3.Connection, seats: pd.DataFrame, *, queue_id: int, history_n: int,
            knowledge_ms: int) -> pd.DataFrame:
    """Each seat's usable prior games, most recent first: ended before T, not remakes, at most history_n.

    Columns: game, seat, games_ago (0 = most recent usable), prior_end_ms, minutes, champion_id, team_position,
    win, kills, deaths, assists, damage_to_champs, damage_taken, gold_earned, cs, vision_score, team_kills,
    team_damage. `team_position` is the role in THAT past game: legitimate history, not the current role.
    """
    with _seat_table(con, seats) as t:
        h = pd.read_sql_query(f"""
            SELECT s.game, s.seat, s.as_of_ms, pm.match_id AS prior_match_id, pm.game_start_ms AS prior_start_ms,
                   pm.game_start_ms + pm.game_duration_s * 1000 AS prior_end_ms,
                   pm.game_duration_s / 60.0 AS minutes, pm.is_remake,
                   pp.champion_id, pp.team_position, pp.win, pp.kills, pp.deaths, pp.assists,
                   pp.damage_to_champs, pp.damage_taken, pp.gold_earned, pp.cs, pp.vision_score,
                   (SELECT sum(t.kills) FROM participant t
                     WHERE t.match_id = pm.match_id AND t.team_id = pp.team_id) AS team_kills,
                   (SELECT sum(t.damage_to_champs) FROM participant t
                     WHERE t.match_id = pm.match_id AND t.team_id = pp.team_id) AS team_damage
            FROM {t} s
            JOIN history_listing h
              ON h.puuid = s.puuid AND h.queue_id = :q AND h.before_ms = s.as_of_ms AND h.fetched_at_ms <= :k
            JOIN history_listing_item li
              ON li.puuid = h.puuid AND li.queue_id = h.queue_id AND li.before_ms = h.before_ms
            JOIN raw_match r ON r.match_id = li.match_id AND r.fetched_at_ms <= :k
            JOIN match pm ON pm.match_id = li.match_id
            JOIN participant pp ON pp.match_id = pm.match_id AND pp.puuid = s.puuid""",
            con, params={"q": queue_id, "k": knowledge_ms})
    # Fix dtypes: an empty result (e.g. a live seat without a listing yet) comes back as object columns.
    ints = ["seat", "as_of_ms", "prior_start_ms", "prior_end_ms", "is_remake", "champion_id", "win"]
    floats = ["minutes", "kills", "deaths", "assists", "damage_to_champs", "damage_taken", "gold_earned", "cs",
              "vision_score", "team_kills", "team_damage"]
    h = h.astype({**{c: "int64" for c in ints}, **{c: "float64" for c in floats}})
    h = h[(h["is_remake"] == 0) & (h["prior_end_ms"] < h["as_of_ms"]) & (h["prior_match_id"] != h["game"])]
    h = h.drop_duplicates(["game", "seat", "prior_match_id"])
    h = h.sort_values(["game", "seat", "prior_start_ms"], ascending=[True, True, False])
    h["games_ago"] = h.groupby(["game", "seat"]).cumcount()
    h = h[h["games_ago"] < history_n].reset_index(drop=True)
    check_history(h)
    return h.drop(columns=["is_remake", "prior_start_ms"])


def check_history(h: pd.DataFrame) -> None:
    """Boundary check: the 'only the past' rule must hold for every loaded row."""
    if (h["prior_end_ms"] >= h["as_of_ms"]).any():
        raise LeakageError("history contains a game that had not ended at T")
    if (h["prior_match_id"] == h["game"]).any():
        raise LeakageError("history contains the game being predicted")
    if h.duplicated(["game", "seat", "prior_match_id"]).any():
        raise LeakageError("duplicate history rows")


# ---------------------------------------------------------------- population (champion) statistics, as of T

def champion_events(con: sqlite3.Connection, *, queue_id: int, knowledge_ms: int) -> pd.DataFrame:
    """Every stored game's picks with their end time: champion_id, role, win, end_ms. Used ONLY through
    as-of windows ([T - window, T)), so a row never sees its own or later games."""
    return pd.read_sql_query("""
        SELECT p.champion_id, p.team_position AS role, p.win, m.game_start_ms + m.game_duration_s * 1000 AS end_ms
        FROM match m
        JOIN raw_match r ON r.match_id = m.match_id AND r.fetched_at_ms <= :k
        JOIN participant p ON p.match_id = m.match_id
        WHERE m.queue_id = :q AND m.is_remake = 0""", con, params={"q": queue_id, "k": knowledge_ms})


class AsOfCounter:
    """Counts (games, wins) per key in the window [T - window_ms, T) with one sort + binary search.

    Java analogy: a sorted array per key with Collections.binarySearch for the window bounds, done for all
    keys at once by packing (key, time) into one int64.
    """

    _SHIFT = np.int64(1) << np.int64(42)   # end times (~1.8e12 ms) fit below 2**42

    def __init__(self, keys: np.ndarray, end_ms: np.ndarray, wins: np.ndarray) -> None:
        packed = keys.astype(np.int64) * self._SHIFT + end_ms.astype(np.int64)
        order = np.argsort(packed, kind="stable")
        self._packed = packed[order]
        self._cum_wins = np.concatenate([[0], np.cumsum(wins.astype(np.int64)[order])])

    def window(self, keys: np.ndarray, t_ms: np.ndarray, window_ms: int) -> tuple[np.ndarray, np.ndarray]:
        k = keys.astype(np.int64) * self._SHIFT
        hi = np.searchsorted(self._packed, k + t_ms.astype(np.int64), side="left")         # end < T
        lo = np.searchsorted(self._packed, k + (t_ms.astype(np.int64) - window_ms), side="left")
        return (hi - lo).astype(float), (self._cum_wins[hi] - self._cum_wins[lo]).astype(float)


# ---------------------------------------------------------------- snapshots (rank, mastery)

def snapshots(con: sqlite3.Connection, seats: pd.DataFrame, *, knowledge_ms: int) -> pd.DataFrame:
    """Per seat, the latest league and mastery snapshot taken before `snapshot_until_ms`.

    Columns (NaN when there is no such snapshot): rank_score, ranked_games, ranked_wins, mastery_points,
    mastery_level, rank_snapshot_ms, mastery_snapshot_ms.
    """
    with _seat_table(con, seats) as t:
        raw = pd.read_sql_query(f"""
            SELECT kind, puuid, taken_at_ms, payload FROM snapshot
            WHERE taken_at_ms <= :k AND puuid IN (SELECT puuid FROM {t})""", con, params={"k": knowledge_ms})
    key = seats[["game", "seat", "puuid", "champion_id", "snapshot_until_ms"]]
    out = key.set_index(["game", "seat"])[[]].copy()

    league_rows = []
    for r in raw[raw["kind"] == "league"].itertuples(index=False):
        solo = next((e for e in json.loads(r.payload) if e.get("queueType") == "RANKED_SOLO_5x5"), None)
        score = wins = losses = np.nan
        if solo is not None and solo.get("tier") in TIERS:
            tier = TIERS.index(solo["tier"])
            div = DIVISIONS.get(solo.get("rank", "I"), 3) if tier < TIERS.index("MASTER") else 0
            score = tier * 4 + div + solo.get("leaguePoints", 0) / 100.0
            wins, losses = solo.get("wins", np.nan), solo.get("losses", np.nan)
        league_rows.append((r.puuid, r.taken_at_ms, score, wins, losses))
    league = pd.DataFrame(league_rows, columns=["puuid", "rank_snapshot_ms", "rank_score", "ranked_wins", "ranked_losses"])
    out = out.join(_latest_before(key, league, "rank_snapshot_ms"))
    out["ranked_games"] = out["ranked_wins"] + out["ranked_losses"]
    out = out.drop(columns="ranked_losses")

    mastery_rows = []
    for r in raw[raw["kind"] == "mastery"].itertuples(index=False):
        for e in json.loads(r.payload):
            mastery_rows.append((r.puuid, r.taken_at_ms, int(e["championId"]), e.get("championPoints", np.nan),
                                 e.get("championLevel", np.nan)))
    mastery = pd.DataFrame(mastery_rows, columns=["puuid", "mastery_snapshot_ms", "champion_id", "mastery_points",
                                                  "mastery_level"])
    # A champion missing from a mastery snapshot means 0 points on it (never played), not "unknown".
    taken = mastery[["puuid", "mastery_snapshot_ms"]].drop_duplicates()
    latest = _latest_before(key, taken, "mastery_snapshot_ms")
    latest = latest.join(key.set_index(["game", "seat"])[["puuid", "champion_id"]])
    latest = latest.reset_index().merge(mastery, how="left", on=["puuid", "mastery_snapshot_ms", "champion_id"])
    has_snap = latest["mastery_snapshot_ms"].notna()
    latest.loc[has_snap, ["mastery_points", "mastery_level"]] = (
        latest.loc[has_snap, ["mastery_points", "mastery_level"]].fillna(0.0))
    out = out.join(latest.set_index(["game", "seat"])[["mastery_snapshot_ms", "mastery_points", "mastery_level"]])
    return out.astype(float)


def _latest_before(key: pd.DataFrame, snaps: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """For each seat: the row of `snaps` (same puuid) with the largest time_col < snapshot_until_ms."""
    cols = [c for c in snaps.columns if c != "puuid"]
    if snaps.empty:
        return pd.DataFrame(np.nan, index=key.set_index(["game", "seat"]).index, columns=cols)
    m = key[["game", "seat", "puuid", "snapshot_until_ms"]].merge(snaps, on="puuid", how="inner")
    m = m[m[time_col] < m["snapshot_until_ms"]].sort_values(time_col)
    m = m.groupby(["game", "seat"]).tail(1).set_index(["game", "seat"])[cols]
    return m.reindex(key.set_index(["game", "seat"]).index)
