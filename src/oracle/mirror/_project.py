"""Pure functions from Riot match JSON to table rows. The only code that knows Riot's match field names.

Bump PARSER_VERSION whenever the output changes; `oracle reproject` then rebuilds older rows from raw JSON.
"""

from __future__ import annotations

import json
from dataclasses import astuple, dataclass

PARSER_VERSION = 1


class PayloadError(ValueError):
    """The JSON doesn't have the shape we expect (Riot changed something, or a non-standard game)."""


@dataclass(frozen=True, slots=True)
class MatchRow:
    match_id: str
    platform_id: str
    queue_id: int
    game_start_ms: int
    game_duration_s: int
    game_version: str
    patch: str
    winning_team_id: int | None
    is_remake: int

    def values(self) -> tuple[object, ...]:
        return astuple(self)


@dataclass(frozen=True, slots=True)
class ParticipantRow:
    match_id: str
    participant_id: int
    puuid: str
    riot_name: str | None
    riot_tag: str | None
    team_id: int
    champion_id: int
    team_position: str
    win: int
    kills: int
    deaths: int
    assists: int
    damage_to_champs: int
    damage_taken: int
    gold_earned: int
    cs: int
    vision_score: int
    champ_level: int
    game_start_ms: int
    queue_id: int

    def values(self) -> tuple[object, ...]:
        return astuple(self)


def patch_of(game_version: str) -> str:
    """'14.20.618.1234' -> '14.20'."""
    parts = game_version.split(".")
    if len(parts) < 2:
        raise PayloadError(f"unexpected gameVersion {game_version!r}")
    return f"{parts[0]}.{parts[1]}"


def match_rows(body: bytes) -> tuple[MatchRow, list[ParticipantRow]]:
    try:
        doc = json.loads(body)
        info = doc["info"]
        match_id = doc["metadata"]["matchId"]
        queue_id = int(info["queueId"])
        start_ms = int(info["gameStartTimestamp"])
        players = info["participants"]
        if len(players) != 10:
            raise PayloadError(f"{match_id}: expected 10 participants, got {len(players)}")
        winners = [t["teamId"] for t in info.get("teams", []) if t.get("win")]
        match = MatchRow(
            match_id=match_id,
            platform_id=info["platformId"],
            queue_id=queue_id,
            game_start_ms=start_ms,
            game_duration_s=int(info["gameDuration"]),
            game_version=info["gameVersion"],
            patch=patch_of(info["gameVersion"]),
            winning_team_id=winners[0] if len(winners) == 1 else None,
            is_remake=int(any(p.get("gameEndedInEarlySurrender", False) for p in players)),
        )
        participants = [
            ParticipantRow(
                match_id=match_id,
                participant_id=int(p["participantId"]),
                puuid=p["puuid"],
                riot_name=p.get("riotIdGameName"),
                riot_tag=p.get("riotIdTagline"),
                team_id=int(p["teamId"]),
                champion_id=int(p["championId"]),
                team_position=p.get("teamPosition") or "",
                win=int(bool(p["win"])),
                kills=int(p["kills"]),
                deaths=int(p["deaths"]),
                assists=int(p["assists"]),
                damage_to_champs=int(p["totalDamageDealtToChampions"]),
                damage_taken=int(p["totalDamageTaken"]),
                gold_earned=int(p["goldEarned"]),
                cs=int(p["totalMinionsKilled"]) + int(p["neutralMinionsKilled"]),
                vision_score=int(p["visionScore"]),
                champ_level=int(p["champLevel"]),
                game_start_ms=start_ms,
                queue_id=queue_id,
            )
            for p in players
        ]
    except PayloadError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise PayloadError(f"cannot project match: {exc!r}") from exc
    return match, participants
