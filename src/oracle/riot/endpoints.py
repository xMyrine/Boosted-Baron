"""The single list of Riot API methods this project calls."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Host(Enum):
    """Which routing value serves an endpoint. Each routing value has its own rate-limit budget."""

    PLATFORM = "platform"   # e.g. euw1
    REGIONAL = "regional"   # e.g. europe
    ACCOUNT = "account"     # regional, except sea -> asia (see config._ACCOUNT_REGIONAL)


@dataclass(frozen=True, slots=True)
class EndpointSpec:
    path: str
    host: Host
    not_found_is_data: bool   # True: 404 is an answer (match gone, not in game); False: 404 is our bug


class Endpoint(Enum):
    ACCOUNT_BY_RIOT_ID = EndpointSpec("/riot/account/v1/accounts/by-riot-id/{game_name}/{tag_line}", Host.ACCOUNT, True)
    MATCH_IDS = EndpointSpec("/lol/match/v5/matches/by-puuid/{puuid}/ids", Host.REGIONAL, False)
    MATCH = EndpointSpec("/lol/match/v5/matches/{match_id}", Host.REGIONAL, True)
    TIMELINE = EndpointSpec("/lol/match/v5/matches/{match_id}/timeline", Host.REGIONAL, True)
    LEAGUE_BY_PUUID = EndpointSpec("/lol/league/v4/entries/by-puuid/{puuid}", Host.PLATFORM, False)
    MASTERY_BY_PUUID = EndpointSpec("/lol/champion-mastery/v4/champion-masteries/by-puuid/{puuid}", Host.PLATFORM, False)
    ACTIVE_GAME = EndpointSpec("/lol/spectator/v5/active-games/by-summoner/{puuid}", Host.PLATFORM, True)
