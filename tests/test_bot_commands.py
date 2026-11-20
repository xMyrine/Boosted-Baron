"""`/predict`: what it answers in each situation, without a Discord connection."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from oracle.bot.commands import NO_ANSWER, WRONG_CHANNEL, answer, handle_predict
from oracle.bot.service import AlreadyProphesied, NobodyRanked, Summoned
from oracle.riot import DeadlineExceeded

CHANNEL = 42


class FakeService:
    def __init__(self, result=None, error=None):
        self.result, self.error, self.calls = result, error, 0

    async def check_now(self):
        self.calls += 1
        if self.error:
            raise self.error
        return self.result


class FakeInteraction:
    def __init__(self, channel_id=CHANNEL):
        self.channel_id = channel_id
        self.log: list[tuple] = []
        self.response = SimpleNamespace(
            send_message=self._record("send_message"), defer=self._record("defer"))
        self.followup = SimpleNamespace(send=self._record("followup"))

    def _record(self, name):
        async def call(*args, **kwargs):
            self.log.append((name, args, kwargs))
        return call


def ask(service, interaction=None):
    interaction = interaction or FakeInteraction()
    asyncio.run(handle_predict(interaction, service, CHANNEL))
    return interaction.log


def test_a_game_found_is_summoned_and_answered_privately():
    log = ask(FakeService(Summoned("EUN1_7", "Fox")))
    assert log[0][0] == "defer" and log[0][2] == {"ephemeral": True, "thinking": True}
    assert log[1] == ("followup", ("The oracle is reading the histories for Fox's game (EUN1_7). "
                                   "The prophecy will appear in the channel.",), {"ephemeral": True})


def test_a_known_game_is_reported_as_already_prophesied():
    assert ask(FakeService(AlreadyProphesied("EUN1_7")))[1][1] == ("The oracle has already spoken about EUN1_7.",)


def test_nobody_in_a_ranked_game_is_said_plainly():
    assert ask(FakeService(NobodyRanked()))[1][1] == ("Nobody tracked is in a ranked solo/duo game right now.",)


def test_another_channel_costs_no_riot_calls():
    service = FakeService(NobodyRanked())
    log = ask(service, FakeInteraction(channel_id=999))
    assert log == [("send_message", (WRONG_CHANNEL,), {"ephemeral": True})] and service.calls == 0


def test_a_busy_riot_budget_is_reported_not_raised():
    log = ask(FakeService(error=DeadlineExceeded(30)))
    assert log[1][1] == (NO_ANSWER.format(why="rate budget busy (eta 30s)"),)


def test_answer_is_exhaustive_over_the_results():
    assert {answer(r).split()[0] for r in (Summoned("k", "n"), AlreadyProphesied("k"), NobodyRanked())} == {
        "The", "Nobody"}
