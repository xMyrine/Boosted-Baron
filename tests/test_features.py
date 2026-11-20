"""Feature building: leakage, train/serve parity, symmetry, and worked examples on a random synthetic world."""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

from oracle.features import Dataset, FeatureParams, LeakageError, Pregame, build_features
from oracle.features import _sources
from oracle.features.pregame import SEAT_COLUMNS
from featureworld import HOUR, add_snapshots, random_world

P = FeatureParams()
FAR = 10**15   # knowledge cut-off "everything"


@pytest.fixture
def world(tmp_path):
    con = random_world(tmp_path / "w.sqlite3")
    yield con, tmp_path / "w.sqlite3"
    con.close()


def anchor_seats(con) -> pd.DataFrame:
    _, seats = _sources.anchors(con, queue_id=420, knowledge_ms=FAR)
    return seats


def one_game(seats: pd.DataFrame, game: str) -> pd.DataFrame:
    return seats[seats["game"] == game][SEAT_COLUMNS]


def assert_same(a: pd.DataFrame, b: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(a.sort_index(axis=1), b.sort_index(axis=1), check_exact=False, rtol=1e-12)


# ---------------------------------------------------------------- dataset shape

def test_dataset_has_one_row_per_complete_anchor_and_ten_seats(world):
    con, _ = world
    ds = Dataset.build(con, knowledge_ms=FAR)
    assert len(ds.matches) > 20
    assert ds.matches["as_of_ms"].is_monotonic_increasing
    assert (ds.seats.groupby("game").size() == 10).all()
    assert set(ds.matches["y_blue_win"]) == {0, 1}
    feats = ds.manifest["columns"]["match_features"]
    assert not any(c.startswith("y_") or c in ("true_role",) for c in feats)
    assert ds.seats["role"].isin(["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]).all()


def test_bundle_round_trips(world, tmp_path):
    con, _ = world
    ds = Dataset.build(con, knowledge_ms=FAR)
    back = Dataset.load(ds.save(tmp_path / "bundle"))
    assert_same(back.matches, ds.matches)
    assert back.manifest["dataset_id"] == ds.manifest["dataset_id"]


# ---------------------------------------------------------------- leakage

def test_future_invariance(world, tmp_path):
    """Scramble every stat/outcome of games not finished at T (incl. the game itself) and delete later games:
    the features at T must not change by a single bit."""
    con, path = world
    seats = anchor_seats(con)
    game = seats["game"].unique()[len(seats["game"].unique()) // 2]
    target = one_game(seats, game)
    t = int(target["as_of_ms"].iloc[0])
    before = build_features(con, target, P, knowledge_ms=FAR)

    c2 = sqlite3.connect(tmp_path / "scrambled.sqlite3", isolation_level=None)
    con.backup(c2)   # a plain file copy would miss pages still in the WAL
    changed = c2.execute("""UPDATE participant SET win = 1 - win, kills = kills + 7, deaths = 0, cs = cs * 3,
                            damage_to_champs = 1, vision_score = 99, team_position = 'UTILITY'
                            WHERE match_id IN (SELECT match_id FROM match
                                               WHERE game_start_ms + game_duration_s * 1000 >= ?)""", (t,)).rowcount
    c2.execute("UPDATE match SET winning_team_id = 300 - winning_team_id WHERE game_start_ms + game_duration_s * 1000 >= ?", (t,))
    later = [m for (m,) in c2.execute("SELECT match_id FROM match WHERE game_start_ms > ?", (t,))]
    c2.executemany("DELETE FROM participant WHERE match_id = ?", [(m,) for m in later])
    c2.executemany("DELETE FROM match WHERE match_id = ?", [(m,) for m in later])
    assert changed > 10 and later, "the test must actually change the future"

    after = build_features(c2, target, P, knowledge_ms=FAR)
    assert_same(after.seats, before.seats)
    assert_same(after.matches, before.matches)


def test_listed_game_that_ended_after_T_is_dropped(world):
    con, _ = world
    seats = anchor_seats(con)
    target = one_game(seats, seats["game"].iloc[-1])
    row = target.iloc[0]
    # A game that started 10 minutes before T but ends 20 minutes after it (Riot lists by start time).
    con.execute("INSERT INTO raw_match VALUES ('EUW1_LEAK', 'ok', x'00', 0, 1)")
    con.execute("INSERT INTO match VALUES ('EUW1_LEAK', 'EUW1', 420, ?, 1800, '16.9', '16.9', 100, 0)",
                (int(row.as_of_ms) - 600_000,))
    con.execute("""INSERT INTO participant VALUES ('EUW1_LEAK', 1, ?, 'n', 't', 100, 1, 'TOP', 1, 99, 0, 99,
                   1, 1, 1, 1, 1, 1, ?, 420)""", (row.puuid, int(row.as_of_ms) - 600_000))
    con.execute("INSERT INTO history_listing_item VALUES (?, 420, ?, 999, 'EUW1_LEAK')", (row.puuid, int(row.as_of_ms)))
    h = _sources.history(con, target, queue_id=420, history_n=20, knowledge_ms=FAR)
    assert "EUW1_LEAK" not in set(h["prior_match_id"])


def test_check_history_rejects_future_rows():
    h = pd.DataFrame({"game": ["g"], "seat": [0], "as_of_ms": [100], "prior_end_ms": [100], "prior_match_id": ["x"]})
    with pytest.raises(LeakageError):
        _sources.check_history(h)


def test_knowledge_cutoff_hides_later_fetches(world):
    con, _ = world
    seats = anchor_seats(con)
    target = one_game(seats, seats["game"].iloc[-1])
    early = build_features(con, target, P, knowledge_ms=0)
    assert not early.coverage["complete"].any()


def test_snapshots_after_game_end_are_ignored_unless_leaky_lag(world):
    con, _ = world
    seats = anchor_seats(con)
    target = one_game(seats, seats["game"].iloc[-1]).copy()
    p0 = target.iloc[0]
    end = int(seats[seats["game"] == p0.game]["snapshot_until_ms"].iloc[0])
    add_snapshots(con, p0.puuid, end + HOUR, tier="DIAMOND", division="I", lp=0)
    clean = _sources.snapshots(con, target, knowledge_ms=FAR)
    assert np.isnan(clean["rank_score"].iloc[0])
    target["snapshot_until_ms"] += 2 * HOUR   # what snapshot_lag_days > 0 does
    leaky = _sources.snapshots(con, target, knowledge_ms=FAR)
    assert leaky["rank_score"].iloc[0] == 6 * 4 + 3


# ---------------------------------------------------------------- parity and symmetry

def test_batch_invariance(world):
    con, _ = world
    seats = anchor_seats(con)
    full = build_features(con, seats[SEAT_COLUMNS], P, knowledge_ms=FAR)
    game = seats["game"].iloc[0]
    single = build_features(con, one_game(seats, game), P, knowledge_ms=FAR)
    assert_same(single.matches, full.matches.loc[[game]])
    assert_same(single.seats, full.seats.loc[[game]])


def test_spectator_payload_gives_the_same_row_as_training(world):
    con, _ = world
    seats = anchor_seats(con)
    game = seats["game"].iloc[-1]
    target = one_game(seats, game)
    platform, game_id = game.split("_")
    payload = {"gameId": game_id, "platformId": platform, "gameQueueConfigId": 420,
               "gameStartTime": int(target["as_of_ms"].iloc[0]),
               "participants": [{"puuid": r.puuid, "teamId": r.side, "championId": r.champion_id}
                                for r in target.sample(frac=1, random_state=1).sort_values("seat").itertuples()]}
    live = Pregame.from_spectator(payload, now_ms=0).to_frame()
    assert live[["game", "seat", "side", "puuid", "champion_id", "as_of_ms"]].equals(
        target[["game", "seat", "side", "puuid", "champion_id", "as_of_ms"]].reset_index(drop=True))
    a = build_features(con, live, P, knowledge_ms=FAR)
    b = build_features(con, target, P, knowledge_ms=FAR)
    assert_same(a.matches, b.matches)


def test_swapping_sides_flips_every_difference(world):
    con, _ = world
    seats = anchor_seats(con)
    target = one_game(seats, seats["game"].iloc[-3])
    swapped = target.copy()
    swapped["side"] = 300 - swapped["side"]
    swapped["seat"] = (swapped["seat"] + 5) % 10
    a = build_features(con, target, P, knowledge_ms=FAR).matches
    b = build_features(con, swapped, P, knowledge_ms=FAR).matches
    diff = [c for c in a.columns if not c.startswith("ctx_")]
    ctx = [c for c in a.columns if c.startswith("ctx_")]
    assert_same(b[diff], -a[diff])
    assert_same(b[ctx], a[ctx])


# ---------------------------------------------------------------- worked examples

def test_streak_form_and_empty_history():
    from oracle.features.roles import ChampionMeta
    from oracle.features.seat import compute

    seats = pd.DataFrame({"game": ["g"] * 10, "seat": range(10), "side": [100] * 5 + [200] * 5,
                          "puuid": [f"p{i}" for i in range(10)], "champion_id": range(1, 11),
                          "as_of_ms": [10 * HOUR] * 10, "snapshot_until_ms": [11 * HOUR] * 10})
    wins = [1, 1, 0, 1]   # most recent first: W W L W -> streak +2
    hist = pd.DataFrame({
        "game": "g", "seat": 0, "as_of_ms": 10 * HOUR, "prior_match_id": [f"m{i}" for i in range(4)],
        "games_ago": range(4), "prior_end_ms": [9 * HOUR, 8 * HOUR, 7 * HOUR, -3 * HOUR], "minutes": 30.0,
        "champion_id": [1, 1, 2, 3], "team_position": "TOP", "win": wins, "kills": 3, "deaths": [1, 2, 9, 0],
        "assists": 4, "damage_to_champs": 20_000, "damage_taken": 15_000, "gold_earned": 10_000, "cs": 210,
        "vision_score": 15, "team_kills": 20, "team_damage": 80_000,
    })
    events = pd.DataFrame({"champion_id": [1], "role": ["TOP"], "win": [1], "end_ms": [HOUR]})
    snaps = pd.DataFrame(np.nan, index=seats.set_index(["game", "seat"]).index,
                         columns=["rank_score", "ranked_games", "ranked_wins", "mastery_points", "mastery_level"])
    f = compute(seats, hist, snaps, ChampionMeta(events), P)

    s0, s1 = f.iloc[0], f.iloc[1]
    assert s0["streak"] == 2
    assert s0["n_games"] == 4 and s0["champ_games"] == 2
    assert s0["wr_shrunk"] == pytest.approx((3 + 50) / (4 + 100))
    assert s0["session_games"] == 3          # ended within 12 h of T
    assert s0["deaths_max"] == 9
    assert s0["cs_pm"] == pytest.approx(210 * 4 / 120)
    assert s0["kp"] == pytest.approx((12 + 16) / 80)
    assert s0["days_since_champ"] == pytest.approx(1 / 24)
    assert s0["role_share"] == 1.0 and s0["off_role"] == 0.0 and f["role"].iloc[0] == "TOP"
    # no history: neutral form, unknown performance
    assert s1["n_games"] == 0 and s1["wr_shrunk"] == 0.5 and s1["streak"] == 0
    assert np.isnan(s1["kda"]) and np.isnan(s1["off_role"])


def test_mastery_missing_champion_means_zero_points(world):
    con, _ = world
    seats = anchor_seats(con)
    target = one_game(seats, seats["game"].iloc[-1])
    p0 = target.iloc[0]
    add_snapshots(con, p0.puuid, int(p0.as_of_ms) - HOUR, mastery={int(p0.champion_id) + 1000: 5_000})
    snap = _sources.snapshots(con, target, knowledge_ms=FAR)
    assert snap["mastery_points"].iloc[0] == 0.0
    assert snap["rank_score"].iloc[0] == 3 * 4 + 2 + 0.5
    assert np.isnan(snap["mastery_points"].iloc[1])   # no snapshot at all: unknown, not zero
