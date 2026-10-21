import asyncio
import math

import pytest

from oracle.riot import DeadlineExceeded
from oracle.riot._limiter import Bucket, Lane, Priority, Window, parse_counts, parse_limits


def test_parse_headers():
    assert parse_limits("20:1,100:120") == (Window(20, 1), Window(100, 120))
    assert parse_counts("3:1,41:120") == {1.0: 3, 120.0: 41}


def test_bucket_waits_until_oldest_send_leaves_the_window():
    bucket = Bucket((Window(3, 10),), safety=1.0, pad_s=0.0)
    for t in (0.0, 1.0, 2.0):
        assert bucket.wait_s(t) == 0
        bucket.record(t)
    assert bucket.wait_s(3.0) == pytest.approx(7.0)    # the send at t=0 expires at t=10
    assert bucket.wait_s(10.0) == 0


def test_safety_margin_and_reserve():
    bucket = Bucket((Window(100, 120),))
    assert bucket.usable(Window(100, 120), 0.0) == 95
    assert bucket.usable(Window(100, 120), 0.25) == 70
    assert bucket.usable(Window(20, 1), 0.25) == 14
    assert bucket.usable(Window(1, 1), 0.25) == 1        # never zero


def test_server_counts_only_raise_local_counts():
    bucket = Bucket((Window(3, 10),), safety=1.0, pad_s=0.0)
    bucket.record(0.0)
    bucket.observe("3:10", "3:10", now=1.0)              # Riot saw 3 calls (e.g. a crashed previous run)
    assert bucket.wait_s(1.0) > 0
    bucket.observe("3:10", "1:10", now=1.0)              # a lower server count never removes entries
    assert bucket.wait_s(1.0) > 0


def test_cold_lane_allows_one_request_in_flight():
    lane = Lane("europe", reserve=lambda: 0.0, pad_s=0.0)
    lane._in_flight = 1
    assert lane.wait_s(Priority.INTERACTIVE, "MATCH", now=0.0) == math.inf


def test_reserve_holds_back_background_but_not_interactive():
    lane = Lane("europe", reserve=lambda: 0.5, safety=1.0, pad_s=0.0)
    lane._warm = True
    lane.app.windows = (Window(4, 10),)
    lane.app.record(0.0)
    lane.app.record(0.0)
    assert lane.app.wait_s(1.0, reserve=0.5) > 0          # background may use 4 - 2 = 2
    assert lane.wait_s(Priority.INTERACTIVE, "MATCH", now=1.0) == 0


def test_application_429_freezes_lane_and_breaker_opens_after_three():
    clock = [100.0]
    lane = Lane("europe", reserve=lambda: 0.0, clock=lambda: clock[0], pad_s=0.0)
    lane._warm = True
    for _ in range(3):
        lane.throttled("MATCH", "application", 2.0, attempt=1)
    assert lane.wait_s(Priority.INTERACTIVE, "MATCH", now=101.0) > 0     # frozen for Retry-After
    clock[0] = 110.0
    assert lane.wait_s(Priority.INTERACTIVE, "MATCH", now=110.0) == 0
    assert lane.wait_s(Priority.BACKGROUND, "MATCH", now=110.0) > 200    # breaker: background paused 5 min


def test_method_429_freezes_only_that_method():
    lane = Lane("europe", reserve=lambda: 0.0, clock=lambda: 0.0, pad_s=0.0)
    lane._warm = True
    lane.throttled("MATCH", "method", 5.0, attempt=1)
    assert lane.wait_s(Priority.INTERACTIVE, "MATCH", now=0.0) >= 5.0
    assert lane.wait_s(Priority.INTERACTIVE, "MATCH_IDS", now=0.0) == 0


async def _warm_lane(limits: str, reserve: float = 0.0) -> Lane:
    lane = Lane("europe", reserve=lambda: reserve, safety=1.0, pad_s=0.0)
    await lane.acquire(Priority.INTERACTIVE, "MATCH")
    lane.release("MATCH", {"X-App-Rate-Limit": limits, "X-App-Rate-Limit-Count": "1:" + limits.split(":")[1]})
    return lane


def test_interactive_jumps_ahead_of_earlier_background():
    async def scenario() -> list[str]:
        lane = await _warm_lane("1:0.3")
        order: list[str] = []

        async def take(name: str, priority: Priority) -> None:
            await lane.acquire(priority, "MATCH")
            order.append(name)
            lane.release("MATCH", None)

        background = asyncio.create_task(take("background", Priority.BACKGROUND))
        await asyncio.sleep(0.01)
        interactive = asyncio.create_task(take("interactive", Priority.INTERACTIVE))
        await asyncio.gather(background, interactive)
        return order

    assert asyncio.run(scenario()) == ["interactive", "background"]


def test_interactive_deadline_fails_fast():
    async def scenario() -> None:
        lane = await _warm_lane("1:5")
        await lane.acquire(Priority.INTERACTIVE, "MATCH", deadline=lane._clock() + 0.05)

    with pytest.raises(DeadlineExceeded):
        asyncio.run(scenario())
