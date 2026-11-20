"""The Discord adapter against a fake client: embed shape, retries, and the two errors that must not retry."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

discord = pytest.importorskip("discord")
aiohttp = pytest.importorskip("aiohttp")

import asyncio  # noqa: E402

from oracle.bot.announce import ConsoleAnnouncer, DiscordAnnouncer, to_embed  # noqa: E402
from oracle.bot.errors import AnnounceError  # noqa: E402
from oracle.bot.prophecy import Field, Message  # noqa: E402

MESSAGE = Message("Prophecy for Fox: Mid Ahri", ("**62%** for Fox.", "A second line."),
                  (Field("Blue side (roles inferred)", "Top Gnar"),), "win_lr - 10/10 histories read", 0x2ECC71)


def http_error(cls, status: int):
    return cls(SimpleNamespace(status=status, reason="x"), "boom")


class FakeChannel:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.embeds = []

    async def send(self, *, embed):
        self.embeds.append(embed)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(id=outcome)


class FakeClient:
    def __init__(self, channel):
        self.channel, self.asked = channel, []

    def get_partial_messageable(self, channel_id):
        self.asked.append(channel_id)
        return self.channel


def announce(outcomes, attempts=4):
    channel = FakeChannel(outcomes)
    slept: list[float] = []

    async def sleep(seconds):
        slept.append(seconds)

    announcer = DiscordAnnouncer(FakeClient(channel), 555, attempts=attempts, sleep=sleep)
    try:
        result = asyncio.run(announcer.announce(MESSAGE))
    except AnnounceError as exc:
        result = exc
    return result, channel, slept


def test_the_embed_carries_every_part_of_the_message():
    embed = to_embed(MESSAGE)
    assert embed.title == "Prophecy for Fox: Mid Ahri"
    assert embed.description == "**62%** for Fox.\nA second line."
    assert (embed.fields[0].name, embed.fields[0].value) == ("Blue side (roles inferred)", "Top Gnar")
    assert embed.footer.text == "win_lr - 10/10 histories read" and embed.colour.value == 0x2ECC71


def test_a_successful_send_returns_the_message_id_as_text():
    result, channel, slept = announce([987654321])
    assert result == "987654321" and len(channel.embeds) == 1 and slept == []


def test_server_errors_are_retried_with_doubling_backoff():
    result, channel, slept = announce([http_error(discord.DiscordServerError, 503),
                                       http_error(discord.DiscordServerError, 502), 42])
    assert result == "42" and slept == [1.0, 2.0] and len(channel.embeds) == 3


def test_network_errors_are_retried_too():
    result, _, slept = announce([aiohttp.ClientConnectionError("reset"), 7])
    assert result == "7" and slept == [1.0]


def test_giving_up_after_the_last_attempt_names_the_cause():
    result, channel, slept = announce([http_error(discord.DiscordServerError, 500)] * 3, attempts=3)
    assert isinstance(result, AnnounceError) and "failed after 3 tries" in str(result)
    assert slept == [1.0, 2.0] and len(channel.embeds) == 3


def test_missing_access_fails_at_once_and_says_how_to_fix_it():
    result, channel, slept = announce([http_error(discord.Forbidden, 403)])
    assert isinstance(result, AnnounceError) and "403 Missing Access" in str(result)
    assert "View Channel, Send Messages and Embed Links" in str(result)
    assert len(channel.embeds) == 1 and slept == []


def test_an_unknown_channel_points_at_the_setting():
    result, _, _ = announce([http_error(discord.NotFound, 404)])
    assert isinstance(result, AnnounceError) and "DISCORD_CHANNEL_ID" in str(result) and "555" in str(result)


def test_the_console_announcer_prints_plain_text(capsys):
    assert asyncio.run(ConsoleAnnouncer().announce(MESSAGE)) == "console"
    out = capsys.readouterr().out
    assert out.startswith("Prophecy for Fox: Mid Ahri\n\n62% for Fox.") and "**" not in out


def test_the_client_registers_predict_for_the_channels_guild_without_any_intent():
    from oracle.bot.commands import make_client

    seen = {}

    class StubService:
        async def check_now(self):
            from oracle.bot.service import NobodyRanked
            return NobodyRanked()

    client = make_client(555, lambda: StubService())
    assert client.intents.value == 0                                         # no privileged (or any) intent

    async def fetch_channel(channel_id):
        seen["channel"] = channel_id
        return SimpleNamespace(guild=discord.Object(id=777))

    async def sync(*, guild):
        seen["synced"] = guild.id

    client.fetch_channel, client.tree.sync = fetch_channel, sync

    async def scenario():
        await client.setup_hook()
        command = client.tree.get_command("predict", guild=discord.Object(id=777))
        replies = []

        async def defer(**kw):
            replies.append(("defer", kw))

        async def send(text, **kw):
            replies.append((text, kw))

        interaction = SimpleNamespace(channel_id=555, response=SimpleNamespace(defer=defer),
                                      followup=SimpleNamespace(send=send))
        await command.callback(interaction)
        return command, replies

    command, replies = asyncio.run(scenario())
    assert seen == {"channel": 555, "synced": 777} and command.name == "predict"
    assert replies[-1] == ("Nobody tracked is in a ranked solo/duo game right now.", {"ephemeral": True})


def test_a_failure_to_register_predict_does_not_stop_the_client_starting():
    from oracle.bot.commands import make_client

    client = make_client(555, lambda: None)

    async def fetch_channel(channel_id):
        raise http_error(discord.Forbidden, 403)

    client.fetch_channel = fetch_channel
    asyncio.run(client.setup_hook())                                         # logs an error, does not raise


class LoginClient:
    """Stands in for discord.Client in the REST-only path of `oracle prophesy --send`."""

    def __init__(self, fail_login: bool = False):
        self.fail_login, self.token, self.closed = fail_login, None, False
        self.channel = FakeChannel([31337])

    async def login(self, token):
        if self.fail_login:
            raise discord.LoginFailure("Improper token has been passed.")
        self.token = token

    def get_partial_messageable(self, channel_id):
        return self.channel

    async def close(self):
        self.closed = True


def settings(token="aaa.bbb.ccc"):
    from oracle.config import DiscordSettings, DiscordToken
    return DiscordSettings(DiscordToken(token), 555)


def test_send_once_logs_in_with_the_token_posts_and_always_logs_out(monkeypatch):
    from oracle.bot import announce

    client = LoginClient()
    monkeypatch.setattr(announce, "make_client", lambda: client)
    assert asyncio.run(announce.send_once(settings("aaa.bbb.ccc"), MESSAGE)) == "31337"
    assert client.token == "aaa.bbb.ccc" and client.closed and len(client.channel.embeds) == 1


def test_send_once_reports_a_rejected_token_without_echoing_it(monkeypatch):
    from oracle.bot import announce

    client = LoginClient(fail_login=True)
    monkeypatch.setattr(announce, "make_client", lambda: client)
    with pytest.raises(AnnounceError, match="rejected DISCORD_BOT_TOKEN") as caught:
        asyncio.run(announce.send_once(settings("secret.token.value"), MESSAGE))
    assert "secret.token.value" not in str(caught.value) and client.closed


def test_a_bad_token_stops_the_bot_with_a_message_instead_of_a_traceback():
    from oracle.bot.app import _discord
    from oracle.config import Settings, ApiKey

    class Client:
        async def start(self, token):
            raise discord.LoginFailure("Improper token has been passed.")

    bot_settings = Settings(ApiKey("RGAPI-x"), "eun1", None, discord=settings())
    with pytest.raises(AnnounceError, match="rejected DISCORD_BOT_TOKEN"):
        asyncio.run(_discord(Client(), bot_settings))
