"""SQL for the `prophecy` table, a state machine per live game: pending -> posted | failed. Private to `mirror`.

Every transition is a single UPDATE guarded by `status = 'pending'`, so a repeated or late call changes nothing.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from oracle.ids import Puuid

ERROR_CHARS = 500


@dataclass(frozen=True, slots=True)
class ProphecyClaim:
    """A prophecy that is not finished: claimed, maybe computed, not yet posted."""

    platform_id: str
    game_id: int
    subject: Puuid
    side: int
    payload: str                  # the spectator game as JSON text
    claimed_at_ms: int
    result_json: str | None       # the stored Prophecy, once it was computed

    @property
    def game_key(self) -> str:
        return f"{self.platform_id.upper()}_{self.game_id}"


@dataclass(frozen=True, slots=True)
class ProphecyCounts:
    pending: int
    posted: int
    failed: int
    with_number: int = 0     # prophecies that carry a probability (not clouded)


@dataclass(frozen=True, slots=True)
class ScoredProphecy:
    """A stored number next to what happened."""

    game_key: str
    p_blue: float
    blue_won: bool
    complete: bool


def claim(con: sqlite3.Connection, platform_id: str, game_id: int, subject: Puuid, side: int, payload: str,
          now_ms: int) -> ProphecyClaim | None:
    """Insert the pending row. The claim if this call created it, None if the game was already claimed."""
    cur = con.execute(
        """INSERT OR IGNORE INTO prophecy (platform_id, game_id, status, subject_puuid, subject_side, payload,
                                           claimed_at_ms)
           VALUES (?, ?, 'pending', ?, ?, ?, ?)""",
        (platform_id, game_id, subject, side, payload, now_ms))
    return ProphecyClaim(platform_id, game_id, subject, side, payload, now_ms, None) if cur.rowcount == 1 else None


def record(con: sqlite3.Connection, platform_id: str, game_id: int, *, model_id: str, p_blue: float | None,
           complete: bool | None, result_json: str) -> bool:
    cur = con.execute(
        """UPDATE prophecy SET model_id = ?, p_blue = ?, complete = ?, prophecy_json = ?
           WHERE platform_id = ? AND game_id = ? AND status = 'pending'""",
        (model_id, p_blue, None if complete is None else int(complete), result_json, platform_id, game_id))
    return cur.rowcount == 1


def finish(con: sqlite3.Connection, platform_id: str, game_id: int, *, message_id: str, now_ms: int) -> bool:
    cur = con.execute(
        """UPDATE prophecy SET status = 'posted', finished_at_ms = ?, discord_message_id = ?
           WHERE platform_id = ? AND game_id = ? AND status = 'pending'""",
        (now_ms, message_id, platform_id, game_id))
    return cur.rowcount == 1


def fail(con: sqlite3.Connection, platform_id: str, game_id: int, *, error: str, now_ms: int) -> bool:
    cur = con.execute(
        """UPDATE prophecy SET status = 'failed', finished_at_ms = ?, error = ?
           WHERE platform_id = ? AND game_id = ? AND status = 'pending'""",
        (now_ms, error[:ERROR_CHARS] or "unknown error", platform_id, game_id))
    return cur.rowcount == 1


def pending(con: sqlite3.Connection) -> list[ProphecyClaim]:
    return [ProphecyClaim(p, g, Puuid(s), side, payload, claimed, result) for p, g, s, side, payload, claimed, result
            in con.execute("""SELECT platform_id, game_id, subject_puuid, subject_side, payload, claimed_at_ms,
                                     prophecy_json
                              FROM prophecy WHERE status = 'pending' ORDER BY claimed_at_ms""")]


def counts(con: sqlite3.Connection) -> ProphecyCounts:
    by_status = dict(con.execute("SELECT status, count(*) FROM prophecy GROUP BY status"))
    numbers = con.execute("SELECT count(p_blue) FROM prophecy").fetchone()[0]
    return ProphecyCounts(by_status.get("pending", 0), by_status.get("posted", 0), by_status.get("failed", 0), numbers)


def scored(con: sqlite3.Connection) -> list[ScoredProphecy]:
    """Prophecies with a number whose game has finished (no remakes)."""
    return [ScoredProphecy(key, p_blue, bool(blue_won), bool(complete)) for key, p_blue, blue_won, complete in con.execute(
        """SELECT m.match_id, p.p_blue, m.winning_team_id = 100, p.complete
           FROM prophecy p JOIN match m ON m.match_id = upper(p.platform_id) || '_' || p.game_id
           WHERE p.p_blue IS NOT NULL AND m.winning_team_id IS NOT NULL AND m.is_remake = 0
           ORDER BY m.game_start_ms""")]
