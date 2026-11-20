"""What is knowable when a game starts: the ONE input type of feature building.

Training turns each finished anchor into this shape; the bot (later) builds it from a spectator-v5 payload.
It deliberately has no outcome, no true role and no patch, so no feature can read them (the spectator API
does not provide them either). Role is *inferred* from history + champion, identically in both paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import pandas as pd

SEAT_COLUMNS = ["game", "seat", "side", "puuid", "champion_id", "as_of_ms", "snapshot_until_ms"]


class Side(IntEnum):
    BLUE = 100
    RED = 200


@dataclass(frozen=True, slots=True)
class Seat:
    puuid: str
    side: Side
    champion_id: int


@dataclass(frozen=True, slots=True)
class Pregame:
    """One game at its start. `as_of_ms` (T) is the cut-off: only games that ENDED before T may be used.

    `seats` are normalised: 5 blue then 5 red, input order kept within a side. Seat index 0..9 is that order.
    """

    key: str
    as_of_ms: int
    queue_id: int
    seats: tuple[Seat, ...]

    def __post_init__(self) -> None:
        if len(self.seats) != 10:
            raise ValueError(f"{self.key}: expected 10 seats, got {len(self.seats)}")
        sides = [s.side for s in self.seats]
        if sides != [Side.BLUE] * 5 + [Side.RED] * 5:
            raise ValueError(f"{self.key}: seats must be 5 blue then 5 red, got {sides}")
        if len({s.puuid for s in self.seats}) != 10:
            raise ValueError(f"{self.key}: duplicate puuid")

    @classmethod
    def from_spectator(cls, game: dict[str, Any], *, now_ms: int) -> Pregame:
        """spectator-v5 active-game payload -> Pregame. `gameStartTime` is 0 while loading: fall back to now."""
        parts = game["participants"]
        seats = [Seat(str(p["puuid"]), Side(int(p["teamId"])), int(p["championId"])) for p in parts]
        seats.sort(key=lambda s: s.side)   # stable: keeps spectator order within a side
        start = int(game.get("gameStartTime") or 0)
        key = f"{str(game['platformId']).upper()}_{game['gameId']}"
        return cls(key, start if start > 0 else now_ms, int(game.get("gameQueueConfigId") or 0), tuple(seats))

    def to_frame(self) -> pd.DataFrame:
        """The seat table feature building consumes (same columns training builds from SQL).

        Live, every snapshot already taken is pre-game, so the snapshot deadline is "now-ish" (T + 1 h).
        """
        return pd.DataFrame(
            [(self.key, i, int(s.side), s.puuid, s.champion_id, self.as_of_ms, self.as_of_ms + 3_600_000)
             for i, s in enumerate(self.seats)],
            columns=SEAT_COLUMNS,
        )
