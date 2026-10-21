"""Client-side rate limiting for one routing value (a "lane"). Private to the `riot` package.

Riot enforces, per routing value, an *application* limit (all our calls) and a *method* limit per endpoint,
each as several windows (e.g. 20 calls / 1 s and 100 calls / 120 s). We mirror that locally:

    Lane (one per routing value)
      ├─ app Bucket       windows learned from X-App-Rate-Limit
      └─ method Buckets   one per endpoint, from X-Method-Rate-Limit
           each Bucket keeps a sliding log of send timestamps

Why a sliding log: it is stricter than Riot's fixed windows, so we can never exceed them.
Why "learn from headers": the real limits depend on the key; response headers are the truth.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import math
import random
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import IntEnum

from oracle.riot.errors import DeadlineExceeded

log = logging.getLogger(__name__)

DEFAULT_APP_LIMITS = "20:1,100:120"   # used until the first response teaches us the real ones
SAFETY = 0.95                         # use at most 95 % of any advertised limit
PAD_S = 1.0                           # widen each window by network latency
MIN_RESERVE = 2                       # interactive reserve is at least 2 calls per window
BREAKER_VIOLATIONS = 3                # application 429s within BREAKER_WINDOW_S ...
BREAKER_WINDOW_S = 600.0
BREAKER_OPEN_S = 300.0                # ... pause background traffic this long
DERATE_FLOOR = 0.1
DERATE_RECOVERY_PER_S = 0.05 / 60     # background pace recovers +5 % per minute after a 429


class Priority(IntEnum):
    INTERACTIVE = 0   # a human is waiting (bot)
    BACKGROUND = 1    # nobody is waiting (crawler)


@dataclass(frozen=True, slots=True)
class Window:
    limit: int
    seconds: float


def parse_limits(spec: str) -> tuple[Window, ...]:
    """'20:1,100:120' -> (Window(20, 1), Window(100, 120))."""
    windows = []
    for part in spec.split(","):
        limit, seconds = part.strip().split(":")
        windows.append(Window(int(limit), float(seconds)))
    return tuple(windows)


def parse_counts(spec: str) -> dict[float, int]:
    """'3:1,41:120' -> {1.0: 3, 120.0: 41} (window seconds -> calls Riot has counted)."""
    counts = {}
    for part in spec.split(","):
        count, seconds = part.strip().split(":")
        counts[float(seconds)] = int(count)
    return counts


class Bucket:
    """One budget (application or one method) as a sliding log of send times."""

    def __init__(self, windows: tuple[Window, ...], *, safety: float = SAFETY, pad_s: float = PAD_S) -> None:
        self.windows = windows
        self.blocked_until = 0.0
        self._sent: deque[float] = deque()
        self._safety = safety
        self._pad_s = pad_s

    def usable(self, window: Window, reserve: float) -> int:
        """How many calls in `window` we allow ourselves; `reserve` is the fraction held back for interactive."""
        cap = math.floor(window.limit * self._safety)
        if reserve > 0:
            cap -= max(MIN_RESERVE, math.ceil(window.limit * reserve))
        return max(1, cap)

    def wait_s(self, now: float, reserve: float = 0.0) -> float:
        """Seconds until one more send fits in every window (0 = now)."""
        wait = max(0.0, self.blocked_until - now)
        self._trim(now)
        for window in self.windows:
            cap = self.usable(window, reserve)
            inside = self._inside(window.seconds, now)
            if len(inside) >= cap:
                # The (len - cap + 1) oldest sends must leave the window before one more fits.
                wait = max(wait, inside[len(inside) - cap] + window.seconds + self._pad_s - now)
        return wait

    def record(self, now: float) -> None:
        self._sent.append(now)

    def freeze(self, until: float) -> None:
        self.blocked_until = max(self.blocked_until, until)

    def observe(self, limits: str | None, counts: str | None, now: float) -> bool:
        """Learn from response headers. Returns True if `limits` was present.

        Counts only ever raise our local log ("server is truth"): if Riot has seen more calls than we
        recorded (a crashed previous process, a stray script), add phantom sends at `now`.
        """
        if limits:
            learned = parse_limits(limits)
            if learned != self.windows:
                log.info("rate limits learned: %s", limits)
                self.windows = learned
        if counts:
            for seconds, server_count in parse_counts(counts).items():
                local = len(self._inside(seconds, now))
                if server_count > local:
                    self._sent.extend([now] * (server_count - local))
        return bool(limits)

    def _inside(self, seconds: float, now: float) -> list[float]:
        horizon = now - (seconds + self._pad_s)
        return [t for t in self._sent if t > horizon]

    def _trim(self, now: float) -> None:
        longest = max((w.seconds for w in self.windows), default=0.0)
        horizon = now - (longest + self._pad_s)
        while self._sent and self._sent[0] <= horizon:
            self._sent.popleft()


@dataclass(eq=False, slots=True)
class _Waiter:
    priority: Priority
    seq: int
    method: str


class Lane:
    """All requests to one routing value. Grants send permits in priority order without exceeding any bucket.

    Policy, all hidden from callers:
    - INTERACTIVE waiters are served before BACKGROUND ones (FIFO within a priority).
    - BACKGROUND may use only `limit - reserve` of each window while `reserve()` > 0 (the bot is running),
      and is paced evenly across the long window instead of bursting.
    - A cold lane (no limit headers seen yet) allows one request in flight, so the first response can
      correct our counts before we send more.
    - Application 429s halve the background pace; three within 10 min open a 5 min background breaker.
    """

    def __init__(
        self,
        name: str,
        *,
        reserve: Callable[[], float],
        clock: Callable[[], float] = time.monotonic,
        rng: random.Random | None = None,
        safety: float = SAFETY,
        pad_s: float = PAD_S,
    ) -> None:
        self.name = name
        self.violations = 0
        self.app = Bucket(parse_limits(DEFAULT_APP_LIMITS), safety=safety, pad_s=pad_s)
        self._methods: dict[str, Bucket] = {}
        self._reserve = reserve
        self._clock = clock
        self._rng = rng or random.Random()
        self._safety = safety
        self._pad_s = pad_s
        self._waiters: list[_Waiter] = []
        self._seq = itertools.count()
        self._changed = asyncio.Event()
        self._warm = False
        self._in_flight = 0
        self._last_background = -math.inf
        self._derate = 1.0
        self._derate_at = clock()
        self._recent_violations: deque[float] = deque()
        self._breaker_until = 0.0

    # ---------------------------------------------------------------- permits

    async def acquire(self, priority: Priority, method: str, deadline: float | None = None) -> None:
        """Return when the caller may send now; the send is recorded before returning.

        `deadline` (monotonic clock) makes the call fail fast with DeadlineExceeded instead of waiting.
        """
        me = _Waiter(priority, next(self._seq), method)
        self._waiters.append(me)
        try:
            while True:
                now = self._clock()
                if self._grantable(me, now):
                    self._grant(me, now)
                    return
                wait: float | None = self.wait_s(priority, method, now)
                if wait == 0.0:
                    wait = None            # eligible, but queued behind someone: wait for a change
                if deadline is not None:
                    remaining = deadline - now
                    if remaining <= 0 or (wait is not None and math.isfinite(wait) and wait > remaining):
                        raise DeadlineExceeded(wait if wait is not None and math.isfinite(wait) else None)
                    wait = remaining if wait is None or not math.isfinite(wait) else wait
                elif wait is not None and not math.isfinite(wait):
                    wait = None
                changed = self._changed
                try:
                    await asyncio.wait_for(changed.wait(), timeout=wait)
                except TimeoutError:
                    pass
        finally:
            if me in self._waiters:
                self._waiters.remove(me)
                self._notify()

    def release(self, method: str, headers: Mapping[str, str] | None) -> None:
        """A response (or network error, headers=None) arrived for a granted request."""
        now = self._clock()
        self._in_flight -= 1
        if headers is not None:
            if self.app.observe(headers.get("X-App-Rate-Limit"), headers.get("X-App-Rate-Limit-Count"), now):
                self._warm = True
            self._method(method).observe(
                headers.get("X-Method-Rate-Limit"), headers.get("X-Method-Rate-Limit-Count"), now
            )
        self._notify()

    def throttled(self, method: str, scope: str | None, retry_after_s: float | None, attempt: int) -> None:
        """Riot answered 429. `scope` is X-Rate-Limit-Type: application | method | service (or missing)."""
        now = self._clock()
        jitter = self._rng.uniform(0.25, 1.0)
        if scope == "application":
            self.app.freeze(now + (retry_after_s or 1.0) + jitter)
            self.violations += 1
            self._derate = max(DERATE_FLOOR, self._current_derate(now) * 0.5)
            self._derate_at = now
            self._recent_violations.append(now)
            while self._recent_violations and self._recent_violations[0] < now - BREAKER_WINDOW_S:
                self._recent_violations.popleft()
            if len(self._recent_violations) >= BREAKER_VIOLATIONS:
                self._breaker_until = now + BREAKER_OPEN_S
                log.error("%s: %d application 429s in 10 min; pausing background calls for 5 min",
                          self.name, len(self._recent_violations))
            log.warning("%s: application 429 (our budget model was wrong: other usage of this key?)", self.name)
        elif scope == "method":
            self._method(method).freeze(now + (retry_after_s or 1.0) + jitter)
            self.violations += 1
            log.warning("%s: method 429 on %s", self.name, method)
        else:
            # Riot's backend is overloaded; not our budget. Retry-After is often missing.
            backoff = retry_after_s if retry_after_s else min(64.0, 2.0 ** attempt) * self._rng.uniform(0.5, 1.0)
            self._method(method).freeze(now + backoff)
            log.info("%s: service 429 on %s, backing off %.1fs", self.name, method, backoff)
        self._notify()

    # ---------------------------------------------------------------- policy

    def wait_s(self, priority: Priority, method: str, now: float) -> float:
        """Seconds until a request of this priority/method could be sent, ignoring other waiters.
        math.inf = unknown until an in-flight response arrives (cold lane)."""
        if not self._warm and self._in_flight > 0:
            return math.inf
        reserve = self._reserve() if priority is Priority.BACKGROUND else 0.0
        wait = max(self.app.wait_s(now, reserve), self._method(method).wait_s(now, reserve))
        if priority is Priority.BACKGROUND:
            wait = max(wait, self._breaker_until - now, self._last_background + self._pace_s(now, reserve) - now)
        return max(0.0, wait)

    def _grantable(self, me: _Waiter, now: float) -> bool:
        if self.wait_s(me.priority, me.method, now) > 0:
            return False
        for other in sorted(self._waiters, key=lambda w: (w.priority, w.seq)):
            if other is me:
                return True
            if other.priority < me.priority:
                return False           # an interactive waiter is queued: background waits
            if self.wait_s(other.priority, other.method, now) == 0:
                return False           # an earlier waiter of my priority can go first
        return True

    def _grant(self, me: _Waiter, now: float) -> None:
        self.app.record(now)
        self._method(me.method).record(now)
        self._in_flight += 1
        if me.priority is Priority.BACKGROUND:
            self._last_background = now
        self._waiters.remove(me)
        self._notify()

    def _pace_s(self, now: float, reserve: float) -> float:
        """Even spacing for background sends: the long window's usable budget spread over its length."""
        if not self.app.windows:
            return 0.0
        longest = max(self.app.windows, key=lambda w: w.seconds)
        return longest.seconds / self.app.usable(longest, reserve) / self._current_derate(now)

    def _current_derate(self, now: float) -> float:
        return min(1.0, self._derate + (now - self._derate_at) * DERATE_RECOVERY_PER_S)

    def _method(self, method: str) -> Bucket:
        bucket = self._methods.get(method)
        if bucket is None:
            bucket = self._methods[method] = Bucket((), safety=self._safety, pad_s=self._pad_s)
        return bucket

    def _notify(self) -> None:
        self._changed.set()
        self._changed = asyncio.Event()
