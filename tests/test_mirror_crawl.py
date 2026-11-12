import asyncio

import pytest
from fakes import FakeApi, FakeGateway, World

from oracle.crawl import CrawlBudget, crawl
from oracle.ids import MatchId, Puuid, RiotId, now_ms
from oracle.mirror import DatasetSpec, FetchMatch, FillResult, Mirror
from oracle.mirror._project import PayloadError, match_rows, patch_of
from oracle.riot import Host

DAY = 86_400_000
OWNER = RiotId("Owner", "EUW")
SPEC = DatasetSpec(history_n=2, max_depth=0, snapshot_kinds=("league",))


def small_world() -> World:
    """The owner played 3 ranked games; every other player has 2 earlier games of their own."""
    world = World(accounts={str(OWNER): "owner"})
    now = now_ms()
    n = 0
    for g in range(3):
        start = now - (g + 1) * DAY
        others = [f"o{g}_{i}" for i in range(9)]
        world.add(f"EUW1_{100 + g}", start, ["owner", *others])
        for player in others:
            for k in range(2):
                n += 1
                world.add(f"EUW1_{1000 + n}", start - (k + 1) * 3_600_000, [player, *[f"x{n}_{j}" for j in range(9)]])
    return world


def run_crawl(mirror: Mirror, api: FakeApi, spec: DatasetSpec = SPEC, **budget):
    return asyncio.run(crawl(mirror, FakeGateway(api), spec, CrawlBudget(**budget)))


@pytest.fixture
def mirror(tmp_path):
    with Mirror.open(tmp_path / "oracle.sqlite3") as m:
        yield m


def test_full_crawl_builds_complete_leak_free_anchors(mirror):
    api = FakeApi(small_world())
    asyncio.run(mirror.track(api, OWNER))
    report = run_crawl(mirror, api)

    assert report.stop_reason == "drained" and report.error is None
    status = mirror.status(SPEC)
    assert status.anchors == {0: (3, 3)}
    assert status.gaps == {}
    rows = mirror._con.execute("SELECT anchor_start_ms, prior_start_ms FROM anchor_history").fetchall()
    assert len(rows) == 3 * 9 * 2 + (2 + 1 + 0)   # 9 others x 2 games + the owner's own earlier games (capped at n=2)
    assert all(prior < anchor for anchor, prior in rows)


def test_rerun_after_completion_makes_no_new_match_calls(mirror):
    api = FakeApi(small_world())
    asyncio.run(mirror.track(api, OWNER))
    run_crawl(mirror, api)
    before = dict(api.calls)
    run_crawl(mirror, api)
    assert api.calls["match"] == before["match"]


def test_fill_is_idempotent(mirror):
    api = FakeApi(small_world())
    gap = FetchMatch(MatchId("EUW1_100"))
    assert asyncio.run(mirror.fill(gap, api)) is FillResult.FILLED
    assert asyncio.run(mirror.fill(gap, api)) is FillResult.FILLED
    assert mirror._con.execute("SELECT count(*) FROM raw_match").fetchone()[0] == 1
    assert mirror._con.execute("SELECT count(*) FROM participant").fetchone()[0] == 10


def test_interrupted_fill_writes_nothing_and_the_gap_remains(mirror):
    api = FakeApi(small_world())
    asyncio.run(mirror.track(api, OWNER))
    run_crawl(mirror, api, max_calls=2)                     # verify key + first sync page
    gap = mirror.gaps(SPEC, Host.REGIONAL, now_ms(), 1)[0]
    assert isinstance(gap, FetchMatch)

    async def kill_mid_call():
        api.block = asyncio.Event()                         # the HTTP call never returns ...
        task = asyncio.create_task(mirror.fill(gap, api))
        await asyncio.sleep(0.01)
        task.cancel()                                       # ... and the process is "killed"
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(kill_mid_call())
    assert mirror._con.execute("SELECT count(*) FROM raw_match").fetchone()[0] == 0
    assert mirror.gaps(SPEC, Host.REGIONAL, now_ms(), 1)[0] == gap


def test_transient_failure_defers_the_gap_with_backoff(mirror):
    api = FakeApi(small_world())
    api.transient_failures["EUW1_100"] = 1
    asyncio.run(mirror.track(api, OWNER))
    gap = FetchMatch(MatchId("EUW1_100"))
    assert asyncio.run(mirror.fill(gap, api)) is FillResult.DEFERRED
    run_crawl(mirror, api, max_calls=2)                     # key check + sync create the anchor
    assert gap not in mirror.gaps(SPEC, Host.REGIONAL, now_ms(), 100)
    assert mirror.retry_failed() == 1
    assert gap in mirror.gaps(SPEC, Host.REGIONAL, now_ms(), 100)


def test_sync_pages_are_checkpointed(mirror):
    world = World(accounts={str(OWNER): "owner"})
    now = now_ms()
    for i in range(150):
        world.add(f"EUW1_{i}", now - (i + 1) * 3_600_000, ["owner", *[f"q{i}_{j}" for j in range(9)]])
    api = FakeApi(world)
    asyncio.run(mirror.track(api, OWNER))
    sync = mirror.gaps(SPEC, Host.REGIONAL, now_ms(), 1)[0]

    asyncio.run(mirror.fill(sync, api))
    cursor = mirror._con.execute("SELECT page_offset, next_due_ms FROM sync_cursor").fetchone()
    assert cursor == (100, 0)                               # mid-cycle: resumes at page 2
    asyncio.run(mirror.fill(sync, api))
    offset, next_due = mirror._con.execute("SELECT page_offset, next_due_ms FROM sync_cursor").fetchone()
    assert offset == 0 and next_due > now                   # cycle complete: due again after the poll interval
    assert mirror._con.execute("SELECT count(*) FROM anchor").fetchone()[0] == 150


def test_promotion_adds_depth_one_anchors_within_wip_limit(mirror):
    api = FakeApi(small_world())
    asyncio.run(mirror.track(api, OWNER))
    spec = DatasetSpec(history_n=2, max_depth=1, wip_anchors=5, max_anchors=10, snapshot_kinds=())
    report = run_crawl(mirror, api, spec)
    assert report.promoted == 7                             # max_anchors 10 - 3 owner games
    depth1 = mirror.status(spec).anchors[1]
    assert depth1 == (7, 7)


def test_key_from_another_app_is_refused(mirror):
    api = FakeApi(small_world())
    asyncio.run(mirror.track(api, OWNER))
    api.world.accounts[str(OWNER)] = "different-encrypted-puuid"
    report = run_crawl(mirror, api)
    assert report.stop_reason == "error" and "KeyMismatch" in report.error


def test_projection():
    from fakes import match_json
    match, players = match_rows(match_json("EUW1_1", 1_700_000_000_000, [f"p{i}" for i in range(10)]))
    assert match.patch == "14.20" and match.winning_team_id == 100 and match.is_remake == 0
    assert players[0].cs == 160 and players[9].team_id == 200
    assert patch_of("15.1.2.3") == "15.1"
    with pytest.raises(PayloadError):
        match_rows(match_json("EUW1_1", 0, [f"p{i}" for i in range(9)] + ["p9"])[:-5])


def test_live_game_is_captured_once_with_pre_game_snapshots_for_all_ten(mirror):
    api = FakeApi(small_world())
    asyncio.run(mirror.track(api, OWNER))
    lobby = ["owner", *[f"lobby{i}" for i in range(9)]]
    api.start_live_game(4242, lobby)
    spec = DatasetSpec()                                     # league + mastery

    capture = asyncio.run(mirror.capture_live(api, Puuid("owner"), spec))
    assert capture is not None and capture.game_id == 4242 and capture.snapshots == 20
    kinds = mirror._con.execute("SELECT kind, count(DISTINCT puuid) FROM snapshot GROUP BY kind").fetchall()
    assert dict(kinds) == {"league": 10, "mastery": 10}

    calls = sum(api.calls.values())
    assert asyncio.run(mirror.capture_live(api, Puuid("owner"), spec)) is None    # same game: only 1 spectator call
    assert sum(api.calls.values()) == calls + 1


def test_live_game_in_another_queue_is_archived_without_snapshots(mirror):
    api = FakeApi(small_world())
    api.start_live_game(7, ["owner", *[f"aram{i}" for i in range(9)]], queue=450)
    capture = asyncio.run(mirror.capture_live(api, Puuid("owner"), DatasetSpec()))
    assert capture is not None and capture.snapshots == 0
    assert mirror._con.execute("SELECT count(*) FROM snapshot").fetchone()[0] == 0
    assert mirror._con.execute("SELECT count(*) FROM live_game").fetchone()[0] == 1


def test_not_in_game_costs_one_call_and_writes_nothing(mirror):
    api = FakeApi(small_world())
    assert asyncio.run(mirror.capture_live(api, Puuid("owner"), DatasetSpec())) is None
    assert api.calls["active_game"] == 1
    assert mirror._con.execute("SELECT count(*) FROM live_game").fetchone()[0] == 0


def test_forever_crawl_watches_tracked_players_for_live_games(mirror):
    api = FakeApi(small_world())
    asyncio.run(mirror.track(api, OWNER))
    api.start_live_game(99, ["owner", *[f"lobby{i}" for i in range(9)]])

    async def run_briefly():
        stop = asyncio.Event()
        task = asyncio.create_task(crawl(mirror, FakeGateway(api), SPEC,
                                         CrawlBudget(forever=True, watch_interval_s=0.05), stop=stop))
        await asyncio.sleep(0.5)
        stop.set()
        return await task

    report = asyncio.run(run_briefly())
    assert report.results["live game captured"] == 1
    assert mirror._con.execute("SELECT count(*) FROM live_game").fetchone()[0] == 1
