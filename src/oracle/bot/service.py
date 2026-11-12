"""`ProphecyService`: one live game in, one stored and announced prophecy out. Knows no Discord and no HTTP.

    submit(capture)      called by the crawler's live watcher (or `check_now`): claims the game, queues it
    run()                the consumer: prepare -> features and model -> record -> announce, one game at a time

The lifecycle of a game is the `prophecy` table's state machine (pending -> posted | failed). The claim is the
INSERT of the pending row, so submitting a game twice does nothing. A crash leaves a pending row, and `recover()`
settles every one at the next start:

    pending row, claimed within RESUME_WINDOW_MS   resume it: announce the stored result, or compute it again
                                                   from the stored spectator payload (what was fetched is kept)
    pending row, older                             mark it failed: the game is over, a late omen is noise
    archived live game, recent, no row at all      claim and queue it: found while the bot was not running

Announcing is at-least-once: a crash between Discord accepting the message and the database noting it posts the
same prophecy a second time on restart. A missing prophecy would be worse than a repeated one.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from oracle.bot.announce import Announcer
from oracle.bot.champions import ChampionNames
from oracle.bot.forecast import compute, foresee
from oracle.bot.predictor import Predictor
from oracle.bot.prophecy import Clouded, Foreseen, Prophecy, Reason, render
from oracle.features import FeatureParams, Pregame, Side
from oracle.ids import EpochMs, Puuid, now_ms
from oracle.mirror import LiveCapture, Mirror, ProphecyClaim
from oracle.riot import AuthError, ForbiddenError, RiotApi

log = logging.getLogger(__name__)

PREPARE_DEADLINE_S = 300.0                 # reading ten histories; a cold game needs ~210 calls at ~100 per 2 min
RESUME_WINDOW_MS = 20 * 60_000             # a pending row older than this is a finished game: do not post late
UNKNOWN_NAME = "Summoner"


@dataclass(frozen=True, slots=True)
class Summoned:
    game_key: str
    subject_name: str


@dataclass(frozen=True, slots=True)
class AlreadyProphesied:
    game_key: str


@dataclass(frozen=True, slots=True)
class NobodyRanked:
    pass


CheckResult = Summoned | AlreadyProphesied | NobodyRanked


class ProphecyService:
    def __init__(
        self,
        *,
        mirror: Mirror,
        api: RiotApi,
        spectator: RiotApi,
        predictor: Predictor,
        announcer: Announcer,
        db_path: Path,
        names: ChampionNames,
        params: FeatureParams | None = None,
        prepare_deadline_s: float = PREPARE_DEADLINE_S,
        resume_window_ms: int = RESUME_WINDOW_MS,
        clock: Callable[[], int] = now_ms,
    ) -> None:
        self._mirror = mirror
        self._api = api                  # long deadline: reading histories may wait for the rate budget
        self._spectator = spectator      # short deadline: a person is waiting for /predict
        self._predictor = predictor
        self._announcer = announcer
        self._db_path = db_path
        self._names = names
        self._params = params or FeatureParams()
        self._deadline_s = prepare_deadline_s
        self._resume_ms = resume_window_ms
        self._clock = clock
        self._queue: asyncio.Queue[ProphecyClaim] = asyncio.Queue()
        self._queued: set[str] = set()   # game keys queued or in progress: enqueueing twice does nothing

    # ---------------------------------------------------------------- entry points

    def submit(self, capture: LiveCapture) -> bool:
        """Claim a live game and queue it. False if it is not a game we prophesy about or someone claimed it."""
        if capture.queue_id != self._params.queue_id:
            log.debug("game %s_%s is queue %s: no prophecy", capture.platform_id, capture.game_id, capture.queue_id)
            return False
        side = next((int(p["teamId"]) for p in json.loads(capture.payload)["participants"]
                     if p.get("puuid") == capture.subject), None)
        if side is None:
            log.warning("game %s_%s: the player who found it is not among its participants", capture.platform_id,
                        capture.game_id)
            return False
        claim = self._mirror.claim_prophecy(capture.platform_id, capture.game_id, capture.subject, side, capture.payload)
        if claim is None:
            return False
        self._enqueue(claim)
        return True

    async def check_now(self) -> CheckResult:
        """Look for a ranked game among the tracked players right now (one spectator call each, until one is found)."""
        for riot_id, puuid in self._mirror.tracked():
            doc = await self._spectator.active_game(puuid)
            if doc is None:
                continue
            game = doc.json()
            if game.get("gameQueueConfigId") != self._params.queue_id:
                continue
            capture = LiveCapture(str(game["platformId"]), int(game["gameId"]), self._params.queue_id, 0, puuid,
                                  doc.body.decode())
            key = f"{capture.platform_id.upper()}_{capture.game_id}"
            return Summoned(key, riot_id.game_name) if self.submit(capture) else AlreadyProphesied(key)
        return NobodyRanked()

    def recover(self) -> None:
        """Settle what an earlier run left behind (see the module docstring). Call before `run`; calling it twice
        changes nothing."""
        for capture in self._mirror.unprophesied_live_games(self._clock() - self._resume_ms):
            log.info("prophesying %s_%s, archived while the bot was not running", capture.platform_id, capture.game_id)
            self.submit(capture)
        for claim in self._mirror.pending_prophecies():
            age_ms = self._clock() - claim.claimed_at_ms
            if age_ms <= self._resume_ms:
                log.info("resuming prophecy %s (claimed %.0f min ago)", claim.game_key, age_ms / 60_000)
                self._enqueue(claim)
            else:
                log.info("abandoning prophecy %s (claimed %.0f min ago)", claim.game_key, age_ms / 60_000)
                self._mirror.fail_prophecy(claim.platform_id, claim.game_id,
                                           error=f"abandoned: found {age_ms / 60_000:.0f} min ago, before a restart")

    async def run(self) -> None:
        """Process queued games one at a time, forever. A failure ends one game, never the loop."""
        while True:
            claim = await self._queue.get()
            try:
                await self._process(claim)
            except Exception as exc:
                log.exception("prophecy for %s failed", claim.game_key)
                self._mirror.fail_prophecy(claim.platform_id, claim.game_id, error=f"{type(exc).__name__}: {exc}")
            finally:
                self._queued.discard(claim.game_key)
                self._queue.task_done()

    async def idle(self) -> None:
        """Return when every queued game has been processed."""
        await self._queue.join()

    # ---------------------------------------------------------------- one game

    def _enqueue(self, claim: ProphecyClaim) -> None:
        if claim.game_key not in self._queued:
            self._queued.add(claim.game_key)
            self._queue.put_nowait(claim)

    async def _process(self, claim: ProphecyClaim) -> None:
        if claim.result_json is not None:
            prophecy = Prophecy.from_json(claim.result_json)   # computed before a crash: only announcing is left
        else:
            prophecy = await self._prophesy(claim)
            outcome = prophecy.outcome
            foreseen = outcome if isinstance(outcome, Foreseen) else None
            self._mirror.record_prophecy(
                claim.platform_id, claim.game_id, model_id=prophecy.model_id,
                p_blue=foreseen.p_blue if foreseen else None, complete=foreseen.complete if foreseen else None,
                result_json=prophecy.to_json())
        try:
            message_id = await self._announcer.announce(render(prophecy))
        except Exception as exc:
            log.warning("could not announce %s: %s", claim.game_key, exc)
            self._mirror.fail_prophecy(claim.platform_id, claim.game_id, error=str(exc))
            return
        self._mirror.finish_prophecy(claim.platform_id, claim.game_id, message_id=message_id)
        log.info("prophecy for %s posted", claim.game_key)

    async def _prophesy(self, claim: ProphecyClaim) -> Prophecy:
        started = time.monotonic()
        pregame = Pregame.from_spectator(json.loads(claim.payload), now_ms=claim.claimed_at_ms)
        subject_name = {str(puuid): riot_id.game_name for riot_id, puuid in self._mirror.tracked()}.get(
            claim.subject, UNKNOWN_NAME)

        def clouded(reason: Reason) -> Prophecy:
            return Prophecy(pregame.key, claim.subject, subject_name, Side(claim.side), self._predictor.model_id,
                            time.monotonic() - started, Clouded(reason))

        report = await self._mirror.prepare_live(
            self._api, [Puuid(s.puuid) for s in pregame.seats], EpochMs(pregame.as_of_ms),
            queue_id=self._params.queue_id, history_n=self._params.history_n, deadline_s=self._deadline_s)
        if isinstance(report.fatal, (AuthError, ForbiddenError)):
            return clouded(Reason.RIOT_REFUSED)
        if report.fatal is not None:
            raise report.fatal
        if report.timed_out or report.missed:
            log.warning("%s: histories incomplete (%d missed, timed out: %s)", pregame.key, report.missed,
                        report.timed_out)
        try:
            features, prediction = await asyncio.to_thread(
                compute, self._db_path, pregame, self._predictor, self._params)
        except Exception:
            log.exception("%s: features or model failed", pregame.key)
            return clouded(Reason.PREDICTION_FAILED)
        return foresee(pregame, claim.subject, subject_name, features, prediction, self._names,
                       model_id=self._predictor.model_id, prepared_in_s=time.monotonic() - started)
