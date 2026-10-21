"""Identifier and time types shared by every module, plus their boundary parsers.

All timestamps are epoch MILLISECONDS (`EpochMs`), matching Riot's game times and the `_ms` columns.
The only seconds conversion lives in `riot.gateway.RiotApi.match_ids`.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import NewType

Puuid = NewType("Puuid", str)
MatchId = NewType("MatchId", str)
EpochMs = NewType("EpochMs", int)

_MATCH_ID = re.compile(r"^[A-Z0-9]+_\d+$")


def now_ms() -> EpochMs:
    return EpochMs(time.time_ns() // 1_000_000)


def parse_match_id(raw: str) -> MatchId:
    if not _MATCH_ID.match(raw):
        raise ValueError(f"not a match id: {raw!r}")
    return MatchId(raw)


@dataclass(frozen=True, slots=True)
class RiotId:
    """A player's public name, `gameName#tagLine`. Can change over time; the PUUID is the stable key."""

    game_name: str
    tag_line: str

    @classmethod
    def parse(cls, text: str) -> RiotId:
        name, sep, tag = text.strip().rpartition("#")
        if not sep or not name or not tag:
            raise ValueError(f"expected 'Name#TAG', got {text!r}")
        return cls(name, tag)

    def __str__(self) -> str:
        return f"{self.game_name}#{self.tag_line}"
