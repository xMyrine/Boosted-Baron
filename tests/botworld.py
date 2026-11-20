"""A live game for the service tests: ten players with stored-by-Riot history, a recording announcer, a tiny model.

The world has the owner and nine others. Each has three earlier ranked games, so preparing the live game takes
10 listing calls and 30 match calls (every earlier game is a different match).
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path

from botkit import linear_model, save_model
from fakes import FakeApi, World

from oracle.bot.champions import ChampionNames
from oracle.bot.errors import AnnounceError
from oracle.bot.predictor import Predictor
from oracle.bot.prophecy import Message
from oracle.bot.service import ProphecyService
from oracle.ids import RiotId, now_ms
from oracle.mirror import DatasetSpec, LiveCapture, Mirror
from oracle.riot import AuthError

DAY = 86_400_000
OWNER = RiotId("Owner", "EUW")
LOBBY = ["owner", *[f"live{i}" for i in range(1, 10)]]
SPEC = DatasetSpec()


def live_world() -> World:
    world = World(accounts={str(OWNER): "owner"})
    n = 0
    now = now_ms()
    for player in LOBBY:
        for k in range(3):
            n += 1
            world.add(f"EUN1_{1000 + n}", now - (k + 1) * DAY, [player, *[f"x{n}_{j}" for j in range(9)]])
    return world


@dataclass
class RecordingAnnouncer:
    sent: list[Message] = field(default_factory=list)
    fail_next: int = 0
    crash_next: bool = False

    async def announce(self, message: Message) -> str:
        if self.crash_next:
            raise asyncio.CancelledError
        if self.fail_next:
            self.fail_next -= 1
            raise AnnounceError("Discord answered 403 Missing Access")
        self.sent.append(message)
        return f"msg-{len(self.sent)}"


class RefusingApi(FakeApi):
    """Riot has rejected the key (401)."""

    async def match_ids(self, *args, **kwargs):
        raise AuthError("Riot rejected the API key (401)")


NAMES = ChampionNames({10 + i: n for i, n in enumerate(
    ["Kayle", "MasterYi", "Alistar", "Annie", "Ashe", "Brand", "Braum", "Caitlyn", "Corki", "Darius"])})


def predictor(tmp_path: Path) -> Predictor:
    return Predictor.load(save_model(tmp_path / "model", linear_model()))


def build(tmp_path: Path, api: FakeApi | None = None, *, announcer: RecordingAnnouncer | None = None, **kw):
    """(mirror, api, announcer, service) over an empty database; the owner is tracked."""
    api = api or FakeApi(live_world())
    mirror = Mirror.open(tmp_path / "oracle.sqlite3")
    asyncio.run(mirror.track(api, OWNER))
    announcer = announcer or RecordingAnnouncer()
    service = ProphecyService(mirror=mirror, api=api, spectator=api, predictor=predictor(tmp_path), announcer=announcer,
                              db_path=tmp_path / "oracle.sqlite3", names=NAMES, **kw)
    return mirror, api, announcer, service


async def capture(mirror: Mirror, api: FakeApi, game_id: int = 4242, *, queue: int = 420) -> LiveCapture:
    api.start_live_game(game_id, LOBBY, queue=queue)
    found = await mirror.capture_live(api, "owner", SPEC)
    assert found is not None
    return found


async def serve(service: ProphecyService) -> None:
    """Run the consumer until everything queued is processed, then stop it."""
    task = asyncio.create_task(service.run())
    await service.idle()
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


def stored(mirror: Mirror, columns: str = "status, discord_message_id") -> list[tuple]:
    return mirror._con.execute(f"SELECT {columns} FROM prophecy ORDER BY game_id").fetchall()


def payload(game_id: int = 4242, *, puuids: list[str] = LOBBY, queue: int = 420) -> str:
    return json.dumps({"gameId": game_id, "platformId": "EUN1", "gameQueueConfigId": queue, "gameStartTime": 0,
                       "participants": [{"puuid": p, "championId": 10 + i, "teamId": 100 if i < 5 else 200}
                                        for i, p in enumerate(puuids)]})
