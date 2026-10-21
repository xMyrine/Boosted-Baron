"""An in-memory fake of the Riot API (same interface as oracle.riot.RiotApi) for offline tests."""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from dataclasses import dataclass, field

from oracle.ids import EpochMs, MatchId, Puuid, RiotId
from oracle.riot import GONE, Doc, Gone, TransientError


def match_json(match_id: str, start_ms: int, puuids: list[str], *, queue: int = 420, remake: bool = False) -> bytes:
    players = [
        {
            "participantId": i + 1, "puuid": puuid, "riotIdGameName": f"p{i}", "riotIdTagline": "EUW",
            "teamId": 100 if i < 5 else 200, "championId": 10 + i, "teamPosition": "TOP", "win": i < 5,
            "kills": i, "deaths": 1, "assists": 2, "totalDamageDealtToChampions": 1000 * i, "totalDamageTaken": 500,
            "goldEarned": 9000, "totalMinionsKilled": 150, "neutralMinionsKilled": 10, "visionScore": 20,
            "champLevel": 15, "gameEndedInEarlySurrender": remake,
        }
        for i, puuid in enumerate(puuids)
    ]
    return json.dumps({
        "metadata": {"matchId": match_id},
        "info": {
            "platformId": "EUW1", "queueId": queue, "gameStartTimestamp": start_ms, "gameDuration": 1800,
            "gameVersion": "14.20.618.1234", "participants": players,
            "teams": [{"teamId": 100, "win": True}, {"teamId": 200, "win": False}],
        },
    }).encode()


@dataclass
class World:
    """Matches as Riot would know them, plus Riot ID -> PUUID."""

    matches: dict[str, tuple[int, list[str], int]] = field(default_factory=dict)   # id -> (start_ms, puuids, queue)
    accounts: dict[str, str] = field(default_factory=dict)

    def add(self, match_id: str, start_ms: int, puuids: list[str], queue: int = 420) -> None:
        assert len(puuids) == 10
        self.matches[match_id] = (start_ms, puuids, queue)


class FakeApi:
    def __init__(self, world: World) -> None:
        self.world = world
        self.calls: Counter[str] = Counter()
        self.transient_failures: Counter[str] = Counter()   # match_id -> how many times to fail first
        self.block: asyncio.Event | None = None              # if set, match() waits on it (simulate in-flight)

    async def account(self, riot_id: RiotId) -> Puuid | None:
        self.calls["account"] += 1
        puuid = self.world.accounts.get(str(riot_id))
        return Puuid(puuid) if puuid else None

    async def match_ids(self, puuid: Puuid, *, queue: int, start: int = 0, count: int = 100,
                        start_ms: EpochMs | None = None, end_ms: EpochMs | None = None) -> list[MatchId]:
        self.calls["match_ids"] += 1
        games = [(s, mid) for mid, (s, ps, q) in self.world.matches.items() if puuid in ps and q == queue]
        if start_ms is not None:
            games = [(s, m) for s, m in games if s // 1000 >= start_ms // 1000]
        if end_ms is not None:   # Riot filters in whole seconds
            games = [(s, m) for s, m in games if s // 1000 < end_ms // 1000]
        games.sort(reverse=True)
        return [MatchId(m) for _, m in games[start:start + count]]

    async def match(self, match_id: MatchId) -> Doc | Gone:
        self.calls["match"] += 1
        if self.block is not None:
            await self.block.wait()
        if self.transient_failures[match_id] > 0:
            self.transient_failures[match_id] -= 1
            raise TransientError("http 503")
        if match_id not in self.world.matches:
            return GONE
        start, puuids, queue = self.world.matches[match_id]
        return Doc(match_json(match_id, start, puuids, queue=queue))

    async def timeline(self, match_id: MatchId) -> Doc | Gone:
        self.calls["timeline"] += 1
        return Doc(b'{"info": {"frames": []}}')

    async def league_entries(self, puuid: Puuid) -> Doc:
        self.calls["league"] += 1
        return Doc(b'[{"queueType": "RANKED_SOLO_5x5", "tier": "GOLD", "rank": "II", "leaguePoints": 40,'
                   b' "wins": 10, "losses": 9}]')

    async def mastery(self, puuid: Puuid) -> Doc:
        self.calls["mastery"] += 1
        return Doc(b"[]")

    async def active_game(self, puuid: Puuid) -> Doc | None:
        return None


class FakeGateway:
    """Just enough of RiotGateway for crawl(): a background handle and a call counter."""

    def __init__(self, api: FakeApi) -> None:
        self.api = api

    @property
    def calls_sent(self) -> int:
        return sum(self.api.calls.values())

    def background(self) -> FakeApi:
        return self.api

    def drain_stats(self) -> list:
        return []
