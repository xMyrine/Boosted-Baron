"""The automatic path end to end on a synthetic world: crawl hook -> claim -> prepare -> predict -> announce."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace

import pytest

pytest.importorskip("sklearn")
pytest.importorskip("joblib")

from botworld import (  # noqa: E402
    LOBBY,
    NAMES,
    SPEC,
    RecordingAnnouncer,
    RefusingApi,
    build,
    capture,
    live_world,
    payload,
    serve,
    stored,
)
from fakes import FakeApi, FakeGateway  # noqa: E402

from oracle.bot.prophecy import Foreseen, Prophecy, SeatView, render  # noqa: E402
from oracle.bot.service import AlreadyProphesied, NobodyRanked, Summoned  # noqa: E402
from oracle.crawl import CrawlBudget, crawl  # noqa: E402
from oracle.features import Side  # noqa: E402
from oracle.ids import now_ms  # noqa: E402


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------- the happy path

def test_one_live_game_gives_exactly_one_message(tmp_path):
    mirror, api, announcer, service = build(tmp_path)

    async def scenario():
        assert service.submit(await capture(mirror, api)) is True
        await serve(service)

    run(scenario())
    [message] = announcer.sent
    assert message.title.startswith("Prophecy for Owner: ")
    assert "10/10 histories read" in message.footer
    assert stored(mirror) == [("posted", "msg-1")]
    assert (api.calls["match_ids"], api.calls["match"]) == (10, 30)        # 10 listings, 3 earlier games each


def test_the_stored_row_holds_the_number_the_message_shows(tmp_path):
    mirror, api, announcer, service = build(tmp_path)
    run(_one_game(mirror, api, service))
    p_blue, complete, model_id, blob = mirror._con.execute(
        "SELECT p_blue, complete, model_id, prophecy_json FROM prophecy").fetchone()
    prophecy = Prophecy.from_json(blob)
    assert (prophecy.outcome.p_blue, complete) == (p_blue, 1)
    assert model_id.startswith("model win_v1@test ")
    assert render(prophecy) == announcer.sent[0]                          # stored JSON and message agree


async def _one_game(mirror, api, service, game_id=4242):
    service.submit(await capture(mirror, api, game_id))
    await serve(service)


def test_preparing_a_game_whose_history_is_stored_makes_no_calls(tmp_path):
    mirror, api, announcer, service = build(tmp_path)
    run(_one_game(mirror, api, service, 1))
    calls = dict(api.calls)
    claim = mirror.claim_prophecy("EUN1", 2, "owner", 100, payload(2))
    report = run(mirror.prepare_live(api, LOBBY, claim.claimed_at_ms, queue_id=420, history_n=20, deadline_s=5))
    assert (report.listed, report.fetched, report.missed) == (10, 0, 0)      # new listing time, same matches
    assert api.calls["match"] == calls["match"]


# ---------------------------------------------------------------- idempotency

def test_the_same_game_delivered_twice_is_still_one_message(tmp_path):
    mirror, api, announcer, service = build(tmp_path)

    async def scenario():
        found = await capture(mirror, api)
        assert service.submit(found) is True
        assert service.submit(found) is False
        await serve(service)
        assert service.submit(found) is False                               # even after it was posted

    run(scenario())
    assert len(announcer.sent) == 1 and stored(mirror) == [("posted", "msg-1")]


def test_a_second_service_on_the_same_database_cannot_claim_the_game_again(tmp_path):
    mirror, api, announcer, service = build(tmp_path)
    other = RecordingAnnouncer()
    twin = type(service)(mirror=mirror, api=api, spectator=api, predictor=service._predictor, announcer=other,
                         db_path=tmp_path / "oracle.sqlite3", names=NAMES)

    async def scenario():
        found = await capture(mirror, api)
        assert service.submit(found) and not twin.submit(found)
        await serve(service)
        await serve(twin)

    run(scenario())
    assert len(announcer.sent) == 1 and other.sent == []


def test_only_ranked_solo_games_get_a_prophecy(tmp_path):
    mirror, api, announcer, service = build(tmp_path)

    async def scenario():
        assert service.submit(await capture(mirror, api, 7, queue=450)) is False
        await serve(service)

    run(scenario())
    assert announcer.sent == [] and stored(mirror) == []
    assert api.calls["match"] == 0                                           # nothing was prepared either


# ---------------------------------------------------------------- degraded modes

def test_a_blown_deadline_still_predicts_and_says_so(tmp_path):
    mirror, api, announcer, service = build(tmp_path, prepare_deadline_s=0.3)
    api.block = asyncio.Event()                                              # match fetches never return

    run(_one_game(mirror, api, service))
    [message] = announcer.sent
    assert "0 of 10 histories" in "\n".join(message.lines)
    assert "0/10 histories read" in message.footer and message.colour == 0xF1C40F
    assert mirror._con.execute("SELECT complete, status FROM prophecy").fetchone() == (0, "posted")


def test_one_unreadable_history_flags_the_message_but_keeps_the_rest(tmp_path):
    mirror, api, announcer, service = build(tmp_path)
    for match_id, (_, players, _) in api.world.matches.items():
        if players[0] == "live5":
            api.transient_failures[match_id] = 99                           # live5's earlier games keep failing
    run(_one_game(mirror, api, service))
    [message] = announcer.sent
    assert "9 of 10 histories" in "\n".join(message.lines)
    assert "9/10 histories read" in message.footer


def test_a_refused_key_is_announced_as_a_clouded_vision(tmp_path):
    mirror, api, announcer, service = build(tmp_path, RefusingApi(live_world()))
    run(_one_game(mirror, api, service))
    [message] = announcer.sent
    assert message.title == "The vision is clouded for Owner"
    assert "RIOT_API_KEY" in message.lines[0]
    assert mirror._con.execute("SELECT p_blue, status FROM prophecy").fetchone() == (None, "posted")


def test_a_model_that_breaks_is_announced_as_a_clouded_vision_and_the_service_goes_on(tmp_path):
    mirror, api, announcer, service = build(tmp_path)
    service._predictor = replace(service._predictor, card=replace(service._predictor.card, features=("gone",)))
    run(_one_game(mirror, api, service))
    [message] = announcer.sent
    assert "Something broke" in message.lines[0]
    assert mirror._con.execute("SELECT p_blue, complete, status FROM prophecy").fetchone() == (None, None, "posted")


def test_a_failing_announcer_marks_the_row_failed_and_the_next_game_still_gets_through(tmp_path):
    mirror, api, announcer, service = build(tmp_path)
    announcer.fail_next = 1

    async def scenario():
        service.submit(await capture(mirror, api, 1))
        service.submit(await capture(mirror, api, 2))
        await serve(service)

    run(scenario())
    assert len(announcer.sent) == 1
    rows = mirror._con.execute("SELECT game_id, status, error, p_blue IS NOT NULL FROM prophecy ORDER BY game_id").fetchall()
    assert rows == [(1, "failed", "Discord answered 403 Missing Access", 1), (2, "posted", None, 1)]


def test_a_game_that_blows_up_marks_its_row_failed_and_the_loop_survives(tmp_path):
    mirror, api, announcer, service = build(tmp_path)

    async def scenario():
        lobby = LOBBY[:9] + ["owner"]                                        # a duplicate seat: Pregame refuses it
        service.submit(await capture(mirror, api, 1))
        mirror.claim_prophecy("EUN1", 2, "owner", 100, payload(2, puuids=lobby))
        service.recover()
        service.submit(await capture(mirror, api, 3))
        await serve(service)

    run(scenario())
    statuses = {gid: (status, error) for gid, status, error in
                mirror._con.execute("SELECT game_id, status, error FROM prophecy")}
    assert statuses[1][0] == "posted" and statuses[3][0] == "posted"
    assert statuses[2][0] == "failed" and "duplicate puuid" in statuses[2][1]
    assert len(announcer.sent) == 2


# ---------------------------------------------------------------- restart: a pending row converges

def test_a_pending_row_without_a_result_is_resumed_after_a_restart(tmp_path):
    mirror, api, announcer, service = build(tmp_path)
    mirror.claim_prophecy("EUN1", 4242, "owner", 100, payload())           # the crash came right after the claim
    service.recover()
    service.recover()                                                        # recovering twice changes nothing
    run(serve(service))
    assert len(announcer.sent) == 1 and stored(mirror) == [("posted", "msg-1")]


def test_a_pending_row_with_a_stored_result_is_announced_again_without_recomputing(tmp_path):
    mirror, api, announcer, service = build(tmp_path)
    seats = tuple(SeatView(Side.BLUE if i < 5 else Side.RED, "TOP", NAMES[10 + i], i == 0, 20, 0, 3, "TOP", False, True)
                  for i in range(10))
    stored_prophecy = Prophecy("EUN1_4242", "owner", "Owner", Side.BLUE, "model x y", 9.0, Foreseen(0.57, (), seats))
    mirror.claim_prophecy("EUN1", 4242, "owner", 100, payload())
    mirror.record_prophecy("EUN1", 4242, model_id="model x y", p_blue=0.57, complete=True,
                           result_json=stored_prophecy.to_json())          # the crash came before the announce
    service.recover()
    run(serve(service))
    assert announcer.sent == [render(stored_prophecy)]
    assert sum(api.calls.values()) == 1                                      # only the tracking call: nothing re-fetched
    assert stored(mirror) == [("posted", "msg-1")]


def test_a_pending_row_from_a_finished_game_is_abandoned_not_posted_late(tmp_path):
    mirror, api, announcer, service = build(tmp_path, clock=lambda: now_ms() + 3 * 3_600_000)
    mirror.claim_prophecy("EUN1", 4242, "owner", 100, payload())
    service.recover()
    run(serve(service))
    assert announcer.sent == []
    status, error = mirror._con.execute("SELECT status, error FROM prophecy").fetchone()
    assert status == "failed" and error.startswith("abandoned: found 180 min ago")


def test_a_game_archived_by_the_watcher_but_never_claimed_is_prophesied_at_the_next_start(tmp_path):
    mirror, api, announcer, service = build(tmp_path)
    run(capture(mirror, api))                                                # the crash came before the claim
    assert stored(mirror) == []
    service.recover()
    service.recover()
    run(serve(service))
    assert len(announcer.sent) == 1 and stored(mirror) == [("posted", "msg-1")]


def test_an_old_archived_game_or_one_in_another_queue_is_not_prophesied_at_start(tmp_path):
    mirror, api, announcer, service = build(tmp_path)
    run(capture(mirror, api, 1, queue=450))
    run(capture(mirror, api, 2))
    late = type(service)(mirror=mirror, api=api, spectator=api, predictor=service._predictor, announcer=announcer,
                         db_path=tmp_path / "oracle.sqlite3", names=NAMES, clock=lambda: now_ms() + 3 * 3_600_000)
    late.recover()
    service.recover()
    run(serve(late))
    run(serve(service))
    assert len(announcer.sent) == 1 and [row[0] for row in stored(mirror, "game_id")] == [2]


def test_a_posted_row_is_left_alone_by_the_next_start(tmp_path):
    mirror, api, announcer, service = build(tmp_path)
    run(_one_game(mirror, api, service))
    restarted = type(service)(mirror=mirror, api=api, spectator=api, predictor=service._predictor, announcer=announcer,
                              db_path=tmp_path / "oracle.sqlite3", names=NAMES)
    restarted.recover()
    run(serve(restarted))
    assert len(announcer.sent) == 1


def test_a_crash_while_announcing_leaves_a_row_that_the_next_start_posts(tmp_path):
    mirror, api, announcer, service = build(tmp_path)
    announcer.crash_next = True                                              # the process dies inside announce()
    run(_one_game_allow_cancel(mirror, api, service))
    assert stored(mirror) == [("pending", None)]
    assert mirror._con.execute("SELECT prophecy_json IS NOT NULL FROM prophecy").fetchone() == (1,)

    announcer.crash_next = False
    service.recover()
    run(serve(service))
    assert len(announcer.sent) == 1 and stored(mirror) == [("posted", "msg-1")]


async def _one_game_allow_cancel(mirror, api, service):
    service.submit(await capture(mirror, api))
    task = asyncio.create_task(service.run())
    await asyncio.wait({task}, timeout=10)                                   # the cancel inside announce() ends it
    assert task.cancelled()


# ---------------------------------------------------------------- the crawler hook and /predict

def test_the_forever_crawl_hands_new_games_to_the_service(tmp_path):
    mirror, api, announcer, service = build(tmp_path)
    api.start_live_game(4242, LOBBY)

    async def scenario():
        stop = asyncio.Event()
        crawling = asyncio.create_task(crawl(mirror, FakeGateway(api), SPEC,
                                             CrawlBudget(forever=True, watch_interval_s=0.05), stop=stop,
                                             on_live=service.submit))
        serving = asyncio.create_task(service.run())
        for _ in range(100):
            if announcer.sent:
                break
            await asyncio.sleep(0.05)
        stop.set()
        serving.cancel()
        await asyncio.gather(crawling, serving, return_exceptions=True)

    run(scenario())
    assert len(announcer.sent) == 1                                          # polled many times, one prophecy


def test_a_hook_that_raises_does_not_stop_the_crawl(tmp_path):
    mirror, api, announcer, service = build(tmp_path)
    api.start_live_game(4242, LOBBY)
    seen = []

    def broken(found):
        seen.append(found.game_id)
        raise RuntimeError("bug in the hook")

    async def scenario():
        stop = asyncio.Event()
        task = asyncio.create_task(crawl(mirror, FakeGateway(api), SPEC, CrawlBudget(forever=True, watch_interval_s=0.05),
                                         stop=stop, on_live=broken))
        await asyncio.sleep(0.3)
        stop.set()
        return await task

    report = run(scenario())
    assert seen == [4242] and report.error is None and report.results["live game captured"] == 1


def test_check_now_summons_once_and_then_reports_the_game_as_known(tmp_path):
    mirror, api, announcer, service = build(tmp_path)

    async def scenario():
        assert await service.check_now() == NobodyRanked()
        api.start_live_game(4242, LOBBY)
        assert await service.check_now() == Summoned("EUN1_4242", "Owner")
        assert await service.check_now() == AlreadyProphesied("EUN1_4242")
        await serve(service)

    run(scenario())
    assert len(announcer.sent) == 1


def test_check_now_ignores_games_in_other_queues(tmp_path):
    mirror, api, announcer, service = build(tmp_path)
    api.start_live_game(5, LOBBY, queue=450)
    assert run(service.check_now()) == NobodyRanked()
    assert stored(mirror) == []


def test_a_game_found_by_check_now_is_not_prophesied_again_by_the_watcher(tmp_path):
    mirror, api, announcer, service = build(tmp_path)

    async def scenario():
        api.start_live_game(4242, LOBBY)
        await service.check_now()
        service.submit(await mirror.capture_live(api, "owner", SPEC))        # the watcher's later sighting
        await serve(service)

    run(scenario())
    assert len(announcer.sent) == 1


# ---------------------------------------------------------------- parity with training

def test_the_service_gives_the_number_the_training_pipeline_gives_for_the_same_game(tmp_path):
    from botkit import FEATURES
    from featureworld import random_world

    from oracle.features import Dataset

    path = tmp_path / "oracle.sqlite3"
    con = random_world(path)
    ds = Dataset.build(con)
    game = ds.matches.index[len(ds.matches) // 2]
    seats = ds.seats[ds.seats["game"] == game].sort_values("seat")
    start = int(ds.matches.loc[game, "as_of_ms"])
    payload_json = json.dumps({
        "gameId": int(game.split("_")[1]), "platformId": game.split("_")[0], "gameQueueConfigId": 420,
        "gameStartTime": start,
        "participants": [{"puuid": r.puuid, "teamId": int(r.side), "championId": int(r.champion_id)}
                         for r in seats.sample(frac=1, random_state=3).itertuples()]})
    con.close()

    mirror, api, announcer, service = build(tmp_path, api=FakeApi(_empty_world()))
    subject = seats.iloc[7].puuid                                            # a red player
    mirror.claim_prophecy(game.split("_")[0], int(game.split("_")[1]), subject, 200, payload_json)
    service.recover()
    run(serve(service))

    row = mirror._con.execute("SELECT p_blue, complete, subject_side FROM prophecy").fetchone()
    expected = service._predictor.model.predict_proba(ds.matches.loc[[game], FEATURES])[0, 1]
    assert row[0] == pytest.approx(expected, abs=1e-12) and row[1] == 1
    [message] = announcer.sent
    assert f"**{round(100 * (1 - expected))}%**" in message.lines[0]         # a red subject sees 1 - P(blue)


def _empty_world():
    from fakes import World
    return World(accounts={"Owner#EUW": "owner"})
