"""The only code that speaks HTTP to Riot or sees the API key.

    gateway = RiotGateway.open(settings)        # one per process: takes data/riot.lock
    crawler_api = gateway.background()          # crawler: low priority, waits as long as needed
    bot_api = gateway.interactive(deadline_s=8) # bot: served first, fails fast with DeadlineExceeded

Priority is a property of the handle, not a call argument, so the crawler cannot take the bot's priority.
Rate limits, retries, 429s and routing are handled here; callers only see domain values and `riot.errors`.
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import logging
import os
import random
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from urllib.parse import quote

import httpx

from oracle.config import Settings
from oracle.ids import EpochMs, MatchId, Puuid, RiotId, parse_match_id
from oracle.riot._limiter import Lane, Priority
from oracle.riot.endpoints import Endpoint, Host
from oracle.riot.errors import (
    AuthError,
    BadRequestError,
    DeadlineExceeded,
    ForbiddenError,
    KeyInUse,
    TransientError,
)

log = logging.getLogger(__name__)

INTERACTIVE_RESERVE = 0.25   # share of each window kept free for the bot while it runs
MAX_THROTTLES = 3            # 429s tolerated for one request before giving up on it (no endless loops)
MAX_RETRIES = {Priority.INTERACTIVE: 1, Priority.BACKGROUND: 3}   # 5xx / network retries


@dataclass(frozen=True, slots=True)
class Doc:
    """A verbatim response body. Stored as-is so it can be re-parsed later without re-fetching."""

    body: bytes

    def json(self) -> Any:
        return json.loads(self.body)


class Gone(Enum):
    """Riot answered 404 for a resource that may legitimately not exist (e.g. a match past retention)."""

    GONE = "gone"


GONE = Gone.GONE


# ------------------------------------------------------------------ response classification (pure)

@dataclass(frozen=True, slots=True)
class Ok:
    pass


@dataclass(frozen=True, slots=True)
class NotFound:
    pass


@dataclass(frozen=True, slots=True)
class Throttled:
    scope: str | None          # X-Rate-Limit-Type: application | method | service | None
    retry_after_s: float | None


@dataclass(frozen=True, slots=True)
class Transient:
    reason: str


@dataclass(frozen=True, slots=True)
class Fatal:
    status: int


Outcome = Ok | NotFound | Throttled | Transient | Fatal


def classify(status: int | None, headers: Mapping[str, str]) -> Outcome:
    """Map an HTTP result to what the gateway should do. `status=None` means a network error/timeout."""
    if status is None:
        return Transient("network")
    if 200 <= status < 300:
        return Ok()
    if status == 404:
        return NotFound()
    if status == 429:
        retry_after: float | None
        try:
            retry_after = float(headers.get("Retry-After", ""))
        except ValueError:
            retry_after = None
        return Throttled(headers.get("X-Rate-Limit-Type"), retry_after)
    if status >= 500:
        return Transient(f"http {status}")
    return Fatal(status)


def _stat_outcome(outcome: Outcome) -> str:
    match outcome:
        case Ok():
            return "ok"
        case NotFound():
            return "not_found"
        case Throttled():
            return "throttled"
        case Transient(reason=reason):
            return "server_error" if reason.startswith("http") else "network"
        case Fatal():
            return "client_error"


# ------------------------------------------------------------------ gateway

CallStat = tuple[str, str, str, str, str]   # (day, routing, endpoint, lane, outcome)


class RiotGateway:
    """Owns the HTTP client, one rate-limit Lane per routing value, and the per-machine key lock."""

    def __init__(
        self,
        settings: Settings,
        client: httpx.AsyncClient,
        lock_fd: int | None,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        lane_options: Mapping[str, Any] | None = None,
    ) -> None:
        self.calls_sent = 0
        self._settings = settings
        self._client = client
        self._lock_fd = lock_fd
        self._clock = clock
        self._sleep = sleep
        self._lane_options = dict(lane_options or {})
        self._lanes: dict[str, Lane] = {}
        self._interactive_present = False
        self._auth_failed = False
        self._disabled: set[Endpoint] = set()
        self._inflight: dict[tuple[str, tuple[tuple[str, Any], ...]], asyncio.Future[bytes | None]] = {}
        self._stats: Counter[CallStat] = Counter()
        self._rng = random.Random()

    @classmethod
    def open(cls, settings: Settings, *, transport: httpx.AsyncBaseTransport | None = None) -> RiotGateway:
        """Take the key lock (one process per machine owns the rate budget) and build the client."""
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        fd = os.open(settings.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            raise KeyInUse(f"another process holds {settings.lock_path} (is the crawler or bot running?)") from None
        client = httpx.AsyncClient(
            headers={"X-Riot-Token": settings.api_key.reveal()},
            timeout=httpx.Timeout(10.0),
            transport=transport,
        )
        return cls(settings, client, fd)

    async def __aenter__(self) -> RiotGateway:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        for task in list(self._inflight.values()):
            task.cancel()
        await self._client.aclose()
        if self._lock_fd is not None:
            os.close(self._lock_fd)   # releases the flock
            self._lock_fd = None

    def interactive(self, *, deadline_s: float = 8.0) -> RiotApi:
        """Handle for a waiting human. Creating one turns on the background reserve for this process."""
        self._interactive_present = True
        return RiotApi(self, Priority.INTERACTIVE, deadline_s)

    def background(self) -> RiotApi:
        return RiotApi(self, Priority.BACKGROUND, None)

    @property
    def violations(self) -> int:
        return sum(lane.violations for lane in self._lanes.values())

    def drain_stats(self) -> list[tuple[CallStat, int]]:
        """Call counts since the last drain, for Mirror.record_calls."""
        items = list(self._stats.items())
        self._stats.clear()
        return items

    # ---------------------------------------------------------------- internals

    def _host_name(self, host: Host) -> str:
        match host:
            case Host.PLATFORM:
                return self._settings.platform
            case Host.REGIONAL:
                return self._settings.regional
            case Host.ACCOUNT:
                return self._settings.account_regional

    def _lane(self, host_name: str) -> Lane:
        lane = self._lanes.get(host_name)
        if lane is None:
            lane = self._lanes[host_name] = Lane(
                host_name, reserve=self._reserve, clock=self._clock, **self._lane_options
            )
        return lane

    def _reserve(self) -> float:
        return INTERACTIVE_RESERVE if self._interactive_present else 0.0

    async def request(
        self,
        endpoint: Endpoint,
        priority: Priority,
        deadline_s: float | None,
        query: Mapping[str, Any] | None = None,
        **path_params: str,
    ) -> bytes | None:
        """GET one endpoint. Returns the body, or None for a 404 the endpoint treats as data.

        Identical concurrent requests share one HTTP call (single-flight): e.g. two players listing the
        same match, or the bot and the crawler wanting the same match at once.
        """
        spec = endpoint.value
        host = self._host_name(spec.host)
        path = spec.path.format(**{k: quote(v, safe="") for k, v in path_params.items()})
        url = f"https://{host}.api.riotgames.com{path}"
        key = (url, tuple(sorted((query or {}).items())))
        deadline = None if deadline_s is None else self._clock() + deadline_s

        task = self._inflight.get(key)
        if task is None:
            task = asyncio.ensure_future(self._send(endpoint, host, url, query, priority, deadline))
            self._inflight[key] = task
            task.add_done_callback(lambda t: self._forget(key, t))
        if deadline is None:
            return await asyncio.shield(task)
        try:
            return await asyncio.wait_for(asyncio.shield(task), max(0.0, deadline - self._clock()))
        except TimeoutError:
            raise DeadlineExceeded(None) from None

    def _forget(self, key: tuple[str, tuple[tuple[str, Any], ...]], task: asyncio.Future[bytes | None]) -> None:
        if self._inflight.get(key) is task:
            del self._inflight[key]
        if not task.cancelled():
            task.exception()   # mark retrieved: a caller that timed out must not cause "never retrieved" noise

    async def _send(
        self,
        endpoint: Endpoint,
        host: str,
        url: str,
        query: Mapping[str, Any] | None,
        priority: Priority,
        deadline: float | None,
    ) -> bytes | None:
        lane = self._lane(host)
        method = endpoint.name
        failures = throttles = 0
        while True:
            if self._auth_failed:
                raise AuthError("Riot rejected the API key (401). Check RIOT_API_KEY in .env.")
            if endpoint in self._disabled:
                raise ForbiddenError(f"{endpoint.name} is forbidden for this key (403)")

            await lane.acquire(priority, method, deadline)
            self.calls_sent += 1
            try:
                resp = await self._client.get(url, params=query)
            except httpx.TransportError as exc:      # includes timeouts
                lane.release(method, None)
                outcome: Outcome = Transient(type(exc).__name__)
                body = b""
            except BaseException:                    # cancelled mid-flight: give the slot back
                lane.release(method, None)
                raise
            else:
                lane.release(method, resp.headers)
                outcome = classify(resp.status_code, resp.headers)
                body = resp.content
            self._stats[(datetime.now(UTC).strftime("%Y-%m-%d"), host, method,
                         priority.name.lower(), _stat_outcome(outcome))] += 1

            match outcome:
                case Ok():
                    return body
                case NotFound():
                    if endpoint.value.not_found_is_data:
                        return None
                    raise BadRequestError(f"{endpoint.name}: unexpected 404")
                case Throttled(scope=scope, retry_after_s=retry_after):
                    throttles += 1
                    if throttles > MAX_THROTTLES:
                        raise TransientError(f"{endpoint.name}: still throttled after {MAX_THROTTLES} tries")
                    lane.throttled(method, scope, retry_after, throttles)
                case Transient(reason=reason):
                    failures += 1
                    if failures > MAX_RETRIES[priority]:
                        raise TransientError(f"{endpoint.name}: {reason}")
                    delay = min(30.0, 2.0 ** failures) * self._rng.uniform(0.5, 1.0)
                    if deadline is not None and self._clock() + delay > deadline:
                        raise TransientError(f"{endpoint.name}: {reason}")
                    log.info("%s: %s, retry %d in %.1fs", endpoint.name, reason, failures, delay)
                    await self._sleep(delay)
                case Fatal(status=401):
                    self._auth_failed = True
                    raise AuthError("Riot rejected the API key (401). Check RIOT_API_KEY in .env.")
                case Fatal(status=403):
                    self._disabled.add(endpoint)
                    log.error("%s returned 403: disabled for this process", endpoint.name)
                    raise ForbiddenError(f"{endpoint.name} is forbidden for this key (403)")
                case Fatal(status=status):
                    raise BadRequestError(f"{endpoint.name}: HTTP {status}")


class RiotApi:
    """A handle with a fixed priority. One method per Riot endpoint; returns domain values."""

    def __init__(self, gateway: RiotGateway, priority: Priority, deadline_s: float | None) -> None:
        self._gateway = gateway
        self._priority = priority
        self._deadline_s = deadline_s

    async def _get(self, endpoint: Endpoint, query: Mapping[str, Any] | None = None, **path: str) -> bytes | None:
        return await self._gateway.request(endpoint, self._priority, self._deadline_s, query, **path)

    async def account(self, riot_id: RiotId) -> Puuid | None:
        """Riot ID -> PUUID, or None if no such account."""
        body = await self._get(Endpoint.ACCOUNT_BY_RIOT_ID, game_name=riot_id.game_name, tag_line=riot_id.tag_line)
        return None if body is None else Puuid(json.loads(body)["puuid"])

    async def match_ids(
        self,
        puuid: Puuid,
        *,
        queue: int,
        start: int = 0,
        count: int = 100,
        start_ms: EpochMs | None = None,
        end_ms: EpochMs | None = None,
    ) -> list[MatchId]:
        """Newest first. Riot's startTime/endTime filters are epoch SECONDS: the one conversion point."""
        query: dict[str, Any] = {"queue": queue, "start": start, "count": count}
        if start_ms is not None:
            query["startTime"] = start_ms // 1000
        if end_ms is not None:
            query["endTime"] = end_ms // 1000
        body = await self._get(Endpoint.MATCH_IDS, query, puuid=puuid)
        assert body is not None
        return [parse_match_id(raw) for raw in json.loads(body)]

    async def match(self, match_id: MatchId) -> Doc | Gone:
        body = await self._get(Endpoint.MATCH, match_id=match_id)
        return GONE if body is None else Doc(body)

    async def timeline(self, match_id: MatchId) -> Doc | Gone:
        body = await self._get(Endpoint.TIMELINE, match_id=match_id)
        return GONE if body is None else Doc(body)

    async def league_entries(self, puuid: Puuid) -> Doc:
        body = await self._get(Endpoint.LEAGUE_BY_PUUID, puuid=puuid)
        assert body is not None
        return Doc(body)

    async def mastery(self, puuid: Puuid) -> Doc:
        body = await self._get(Endpoint.MASTERY_BY_PUUID, puuid=puuid)
        assert body is not None
        return Doc(body)

    async def active_game(self, puuid: Puuid) -> Doc | None:
        """The live game the player is in, or None."""
        body = await self._get(Endpoint.ACTIVE_GAME, puuid=puuid)
        return None if body is None else Doc(body)
