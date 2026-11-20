"""A synthetic, randomly generated mirror for feature tests: random winners, durations, stats and roles.

Written straight into SQLite through the real schema (no crawl), shaped like a finished crawl: every anchor's
10 players have a listing of their last `history_n` games that STARTED before the anchor (Riot's semantics).
"""

from __future__ import annotations

import json
import random
import sqlite3
from pathlib import Path

from oracle.mirror import Mirror

ROLES = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")
T0 = 1_780_000_000_000
HOUR = 3_600_000


def random_world(path: Path, *, seed: int = 0, n_players: int = 30, n_matches: int = 150, n_anchors: int = 40,
                 history_n: int = 20) -> sqlite3.Connection:
    rng = random.Random(seed)
    Mirror.open(path).close()
    con = sqlite3.connect(path, isolation_level=None)
    players = [f"puuid-{i:03d}" for i in range(n_players)]
    main_role = {p: ROLES[i % 5] for i, p in enumerate(players)}
    pool = {p: [rng.randrange(1, 60) for _ in range(3)] for p in players}

    busy_until = {p: 0 for p in players}
    matches: list[tuple[str, int, list[str]]] = []
    t = T0
    con.execute("BEGIN")
    for m in range(n_matches):
        t += rng.randrange(1, 4) * HOUR
        free = [p for p in players if busy_until[p] < t]
        ten = rng.sample(free, 10)
        duration = rng.randrange(15, 45) * 60
        mid = f"EUW1_{m:05d}"
        winner = rng.choice((100, 200))
        con.execute("INSERT INTO raw_match VALUES (?, 'ok', ?, ?, 1)", (mid, b"{}", t + duration * 1000 + HOUR))
        con.execute("INSERT INTO match VALUES (?, 'EUW1', 420, ?, ?, '16.1.1', ?, ?, 0)",
                    (mid, t, duration, f"16.{1 + m // 50}", winner))
        for team_id, side in ((100, ten[:5]), (200, ten[5:])):
            roles = list(ROLES)
            rng.shuffle(roles)
            # mostly main roles: sort so a player gets their main role when it's free
            assigned = _assign(side, roles, main_role, rng)
            for j, p in enumerate(side):
                pid = j + 1 if team_id == 100 else j + 6
                con.execute("""INSERT INTO participant VALUES (?, ?, ?, 'n', 't', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                               ?, ?, ?, 420)""",
                            (mid, pid, p, team_id, rng.choice(pool[p]), assigned[p], int(team_id == winner),
                             rng.randrange(0, 15), rng.randrange(0, 12), rng.randrange(0, 20),
                             rng.randrange(5_000, 40_000), rng.randrange(5_000, 40_000),
                             rng.randrange(6_000, 18_000), rng.randrange(20, 300), rng.randrange(5, 80),
                             rng.randrange(8, 18), t))
                busy_until[p] = t + duration * 1000
        matches.append((mid, t, ten))

    con.execute("INSERT INTO tracked_player VALUES (?, 'n', 't', 0)", (players[0],))
    for seq, (mid, start, ten) in enumerate(matches[-n_anchors:]):
        con.execute("INSERT INTO anchor (match_id, depth, source, added_at_ms) VALUES (?, 0, 'tracked', 0)", (mid,))
        for p in ten:
            add_listing(con, p, start, history_n)
    con.execute("COMMIT")
    return con


def add_listing(con: sqlite3.Connection, puuid: str, before_ms: int, n: int) -> None:
    prior = con.execute("""SELECT p.match_id FROM participant p WHERE p.puuid = ? AND p.game_start_ms < ?
                           ORDER BY p.game_start_ms DESC LIMIT ?""", (puuid, before_ms, n)).fetchall()
    con.execute("INSERT OR IGNORE INTO history_listing VALUES (?, 420, ?, ?, ?)", (puuid, before_ms, n, before_ms))
    for ord_, (mid,) in enumerate(prior):
        con.execute("INSERT OR IGNORE INTO history_listing_item VALUES (?, 420, ?, ?, ?)", (puuid, before_ms, ord_, mid))


def add_snapshots(con: sqlite3.Connection, puuid: str, taken_at_ms: int, *, tier: str = "GOLD", division: str = "II",
                  lp: int = 50, mastery: dict[int, int] | None = None) -> None:
    league = [{"queueType": "RANKED_SOLO_5x5", "tier": tier, "rank": division, "leaguePoints": lp,
               "wins": 30, "losses": 20}]
    con.execute("INSERT INTO snapshot VALUES ('league', ?, ?, ?)", (puuid, taken_at_ms, json.dumps(league)))
    m = [{"championId": c, "championPoints": pts, "championLevel": 5} for c, pts in (mastery or {}).items()]
    con.execute("INSERT INTO snapshot VALUES ('mastery', ?, ?, ?)", (puuid, taken_at_ms, json.dumps(m)))


def _assign(side: list[str], roles: list[str], main_role: dict[str, str], rng: random.Random) -> dict[str, str]:
    out: dict[str, str] = {}
    free = list(roles)
    for p in side:
        if main_role[p] in free and rng.random() < 0.8:
            out[p] = main_role[p]
            free.remove(main_role[p])
    for p in side:
        if p not in out:
            out[p] = free.pop()
    return out
