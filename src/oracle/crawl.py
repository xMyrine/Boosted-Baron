"""The crawl loop: ask the mirror what is missing, fill it, repeat. Only schedules; knows no SQL or HTTP.

One worker loop per routing lane (regional: matches, platform: rank/mastery) so the two rate budgets are
used in parallel. Each loop keeps a few fills in flight; the gateway's limiter decides when they are sent.
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from oracle.ids import now_ms
from oracle.mirror import DatasetSpec, FillResult, Gap, KeyMismatch, LiveCapture, Mirror
from oracle.riot import AuthError, BadRequestError, ForbiddenError, Host, RiotError, RiotGateway

log = logging.getLogger(__name__)

FATAL = (AuthError, ForbiddenError, BadRequestError, KeyMismatch)


@dataclass(frozen=True, slots=True)
class CrawlBudget:
    max_calls: int | None = None             # this run
    max_calls_per_day: int | None = 50_000   # across runs (UTC day), from api_call_stat
    forever: bool = False                    # keep polling for new games instead of stopping when done
    concurrency: Mapping[Host, int] = field(default_factory=lambda: {Host.REGIONAL: 3, Host.PLATFORM: 2})
    idle_poll_s: float = 60.0
    watch_interval_s: float = 60.0           # --forever only: how often tracked players are checked for a live game


@dataclass
class CrawlReport:
    run_id: int = 0
    calls: int = 0
    results: Counter[str] = field(default_factory=Counter)   # e.g. 'match filled', 'listing deferred'
    promoted: int = 0
    stop_reason: str | None = None
    error: str | None = None

    def summary(self) -> str:
        done = ", ".join(f"{n} {what}" for what, n in sorted(self.results.items())) or "nothing"
        text = f"run #{self.run_id}: {self.calls} calls; {done}; {self.promoted} anchors promoted; stop: {self.stop_reason}"
        return text + (f"\nerror: {self.error}" if self.error else "")


async def crawl(
    mirror: Mirror,
    gateway: RiotGateway,
    spec: DatasetSpec,
    budget: CrawlBudget,
    *,
    stop: asyncio.Event | None = None,
    on_live: Callable[[LiveCapture], None] | None = None,
) -> CrawlReport:
    """Fill gaps until none are left (or forever), the budget is spent, or `stop` is set.

    `on_live` is called (synchronously, so it must not block) for every newly captured live game of a tracked
    player; the Discord bot uses it to start a prophecy. An exception in the hook is logged, never fatal.
    Safe to cancel at any point (Ctrl-C): an interrupted fill writes nothing, and the next run resumes.
    """
    api = gateway.background()
    stop = stop or asyncio.Event()
    report = CrawlReport(run_id=mirror.run_started(spec))
    calls_at_start = gateway.calls_sent
    calls_before_today = mirror.calls_today()
    regional_done = asyncio.Event()

    def spent() -> str | None:
        used = gateway.calls_sent - calls_at_start
        if budget.max_calls is not None and used >= budget.max_calls:
            return "budget"
        if budget.max_calls_per_day is not None and calls_before_today + used >= budget.max_calls_per_day:
            return "daily_cap"
        return None

    async def idle(seconds: float, *wake_on: asyncio.Event) -> None:
        waiters = [asyncio.create_task(event.wait()) for event in (stop, *wake_on)]
        await asyncio.wait(waiters, timeout=seconds, return_when=asyncio.FIRST_COMPLETED)
        for waiter in waiters:
            waiter.cancel()

    async def lane_loop(host: Host) -> None:
        pending: dict[asyncio.Task[FillResult], Gap] = {}
        concurrency = budget.concurrency[host]
        try:
            while not stop.is_set():
                if reason := spent():
                    report.stop_reason = reason
                    stop.set()
                    break
                if len(pending) < concurrency:
                    busy = {gap.key for gap in pending.values()}
                    fresh = [g for g in mirror.gaps(spec, host, now_ms(), limit=concurrency * 2) if g.key not in busy]
                    if not fresh and host is Host.REGIONAL:
                        report.promoted += (promoted := mirror.promote(spec, now_ms()))
                        if promoted:
                            continue
                    for gap in fresh[: concurrency - len(pending)]:
                        pending[asyncio.create_task(mirror.fill(gap, api))] = gap
                    if not pending:
                        if host is Host.PLATFORM and not regional_done.is_set():
                            await idle(5.0, regional_done)   # new anchors may still bring snapshot work
                        elif budget.forever:
                            await idle(budget.idle_poll_s)
                        else:
                            break
                        continue
                done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    gap = pending.pop(task)
                    report.results[f"{gap.kind} {task.result().value}"] += 1   # re-raises fatal errors
                mirror.record_calls(gateway.drain_stats())
        except FATAL as exc:
            report.stop_reason = "error"
            report.error = f"{type(exc).__name__}: {exc}"
            stop.set()
        finally:
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            if host is Host.REGIONAL:
                regional_done.set()

    async def watch_loop() -> None:
        """Clean pre-game data: when a tracked player is in a new live game, snapshot all 10 players' rank and
        mastery right away (rank only changes after a game ends, so these are leak-free training features)."""
        while not stop.is_set():
            for _, puuid in mirror.tracked():
                try:
                    capture = await mirror.capture_live(api, puuid, spec)
                except AuthError as exc:
                    report.stop_reason, report.error = "error", f"AuthError: {exc}"
                    stop.set()
                    return
                except ForbiddenError as exc:
                    log.error("live watcher disabled: %s", exc)
                    return
                except RiotError as exc:
                    log.warning("live check failed, retrying next round: %s", exc)
                    continue
                if capture is not None:
                    report.results["live game captured"] += 1
                    log.info("live game %s_%s captured: %d pre-game snapshots",
                             capture.platform_id, capture.game_id, capture.snapshots)
                    if on_live is not None:
                        try:
                            on_live(capture)
                        except Exception:
                            log.exception("on_live hook failed for %s_%s", capture.platform_id, capture.game_id)
            mirror.record_calls(gateway.drain_stats())
            await idle(budget.watch_interval_s)

    try:
        try:
            await mirror.verify_key(api)
        except FATAL as exc:
            report.stop_reason = "error"
            report.error = f"{type(exc).__name__}: {exc}"
            return report
        loops = [lane_loop(Host.REGIONAL), lane_loop(Host.PLATFORM)]
        if budget.forever:
            loops.append(watch_loop())
        await asyncio.gather(*loops)
        report.stop_reason = report.stop_reason or ("stopped" if stop.is_set() else "drained")
        return report
    finally:
        report.calls = gateway.calls_sent - calls_at_start
        mirror.record_calls(gateway.drain_stats())
        mirror.run_ended(report.run_id, report.calls, report.stop_reason or "interrupted")
