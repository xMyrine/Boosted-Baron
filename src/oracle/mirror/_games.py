"""Read-only questions about stored games, for replaying a finished game as if it were live. Private to `mirror`.

They take a connection (not a `Mirror`), so a replay can run on `Mirror.read_only` without touching the database.
"""

from __future__ import annotations

import sqlite3
import zlib
from dataclasses import dataclass

from oracle.mirror import _frontier
from oracle.mirror._project import PayloadError, champion_names as names_in


@dataclass(frozen=True, slots=True)
class StoredPlayer:
    participant_id: int
    puuid: str
    team_id: int
    champion_id: int


@dataclass(frozen=True, slots=True)
class StoredGame:
    match_id: str
    platform_id: str
    queue_id: int
    game_start_ms: int
    winning_team_id: int | None   # None = no result recorded
    players: tuple[StoredPlayer, ...]

    def spectator_payload(self) -> dict[str, object]:
        """The game as spectator-v5 would have shown it at its start: no roles, no outcome."""
        return {
            "gameId": int(self.match_id.rsplit("_", 1)[1]), "platformId": self.platform_id,
            "gameQueueConfigId": self.queue_id, "gameStartTime": self.game_start_ms,
            "participants": [{"puuid": p.puuid, "teamId": p.team_id, "championId": p.champion_id}
                             for p in self.players],
        }


def stored_game(con: sqlite3.Connection, match_id: str) -> StoredGame | None:
    head = con.execute(
        "SELECT platform_id, queue_id, game_start_ms, winning_team_id FROM match WHERE match_id = ?", (match_id,)
    ).fetchone()
    if head is None:
        return None
    players = tuple(StoredPlayer(*row) for row in con.execute(
        """SELECT participant_id, puuid, team_id, champion_id FROM participant
           WHERE match_id = ? ORDER BY participant_id""", (match_id,)))
    return StoredGame(match_id, *head, players)


def newest_complete_anchor(con: sqlite3.Connection, *, queue_id: int, history_n: int) -> str | None:
    """The newest finished anchor whose ten players all have a full history stored: a game worth replaying."""
    row = con.execute(
        f"""SELECT a.match_id FROM anchor a JOIN match m ON m.match_id = a.match_id
            WHERE m.queue_id = :q AND m.is_remake = 0 AND m.winning_team_id IS NOT NULL
              AND NOT {_frontier.INCOMPLETE_ANCHOR}
            ORDER BY m.game_start_ms DESC LIMIT 1""", {"q": queue_id, "n": history_n}).fetchone()
    return None if row is None else row[0]


def champion_names(con: sqlite3.Connection) -> dict[int, str]:
    """Champion id -> Data Dragon name for every champion in the stored games. One payload per champion is read."""
    names: dict[int, str] = {}
    for (match_id,) in con.execute("SELECT min(match_id) FROM participant GROUP BY champion_id"):
        row = con.execute("SELECT payload FROM raw_match WHERE match_id = ? AND outcome = 'ok'", (match_id,)).fetchone()
        if row is None:
            continue
        try:
            names.update(names_in(zlib.decompress(row[0])))
        except (PayloadError, zlib.error):
            continue
    return names


def tracked_names(con: sqlite3.Connection) -> dict[str, str]:
    """PUUID -> game name of every tracked player."""
    return dict(con.execute("SELECT puuid, game_name FROM tracked_player ORDER BY added_at_ms"))
