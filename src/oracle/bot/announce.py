"""Delivering a `Message`: the port (`Announcer`) and its adapters.

    ConsoleAnnouncer   prints the text (dry runs)
    DiscordAnnouncer   posts an embed to one channel through discord.py

discord.py is imported lazily, so this module loads without the `bot` dependency group. The bot needs no
privileged intents: it never reads messages, and `get_partial_messageable` sends without caching channels.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Protocol

from oracle.bot.errors import AnnounceError
from oracle.bot.prophecy import Message
from oracle.config import DiscordSettings

if TYPE_CHECKING:
    import discord

log = logging.getLogger(__name__)

SEND_ATTEMPTS = 4
FIRST_BACKOFF_S = 1.0


class Announcer(Protocol):
    async def announce(self, message: Message) -> str:
        """Deliver the message and return its id on the platform."""
        ...


class ConsoleAnnouncer:
    async def announce(self, message: Message) -> str:
        print(message.as_text())
        return "console"


def make_client() -> discord.Client:
    import discord

    return discord.Client(intents=discord.Intents.none())


def to_embed(message: Message) -> discord.Embed:
    import discord

    embed = discord.Embed(title=message.title, description="\n".join(message.lines),
                          colour=discord.Colour(message.colour))
    for field in message.fields:
        embed.add_field(name=field.name, value=field.value, inline=field.inline)
    embed.set_footer(text=message.footer)
    return embed


class DiscordAnnouncer:
    def __init__(self, client: discord.Client, channel_id: int, *, attempts: int = SEND_ATTEMPTS,
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep) -> None:
        self._client = client
        self._channel_id = channel_id
        self._attempts = attempts
        self._sleep = sleep

    async def announce(self, message: Message) -> str:
        """Send, retrying network errors, 5xx and 429 with doubling backoff. 403 and 404 fail at once."""
        import discord

        channel = self._client.get_partial_messageable(self._channel_id)
        embed = to_embed(message)
        delay = FIRST_BACKOFF_S
        for attempt in range(1, self._attempts + 1):
            try:
                sent = await channel.send(embed=embed)
                return str(sent.id)
            except discord.Forbidden as exc:
                raise AnnounceError(
                    "Discord answered 403 Missing Access: the bot is not in the server or cannot post in the "
                    "channel. Re-invite it and give it View Channel, Send Messages and Embed Links there."
                ) from exc
            except discord.NotFound as exc:
                raise AnnounceError(f"Discord has no channel {self._channel_id}: check DISCORD_CHANNEL_ID.") from exc
            except Exception as exc:
                if not _retryable(exc) or attempt == self._attempts:
                    raise AnnounceError(f"Discord send failed after {attempt} tries: {exc}") from exc
                log.warning("Discord send failed (%s), retry %d in %.0fs", exc, attempt, delay)
                await self._sleep(delay)
                delay *= 2
        raise AssertionError("unreachable")


def _retryable(exc: Exception) -> bool:
    import aiohttp
    import discord

    if isinstance(exc, discord.HTTPException):
        return exc.status >= 500 or exc.status == 429
    return isinstance(exc, (aiohttp.ClientError, TimeoutError, OSError))


async def send_once(settings: DiscordSettings, message: Message) -> str:
    """Log in over REST only (no gateway connection), post one message, log out. For `oracle prophesy --send`."""
    import discord

    client = make_client()
    try:
        await client.login(settings.token.reveal())
    except discord.LoginFailure as exc:
        await client.close()
        raise AnnounceError("Discord rejected DISCORD_BOT_TOKEN (login failed). Reset the token in the developer "
                            "portal and update .env.") from exc
    try:
        return await DiscordAnnouncer(client, settings.channel_id).announce(message)
    finally:
        await client.close()
