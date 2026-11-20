"""The `/predict` slash command: ask the oracle about a ranked game that is running right now.

The command needs no privileged intent. It is registered for one guild (the one that owns the configured channel),
so it shows up at once instead of after Discord's global cache delay, and it answers only in that channel.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from oracle.bot.service import AlreadyProphesied, CheckResult, NobodyRanked, ProphecyService, Summoned
from oracle.riot import RiotError

if TYPE_CHECKING:
    import discord

log = logging.getLogger(__name__)

WRONG_CHANNEL = "Ask the oracle in its own channel."
NO_ANSWER = "Riot did not answer in time ({why}). Try again in a minute."


def answer(result: CheckResult) -> str:
    match result:
        case Summoned(game_key=key, subject_name=name):
            return f"The oracle is reading the histories for {name}'s game ({key}). The prophecy will appear in the channel."
        case AlreadyProphesied(game_key=key):
            return f"The oracle has already spoken about {key}."
        case NobodyRanked():
            return "Nobody tracked is in a ranked solo/duo game right now."


async def handle_predict(interaction: Any, service: ProphecyService, channel_id: int) -> None:
    """The command body. `interaction` is a discord.Interaction; only `channel_id`, `response` and `followup` are used."""
    if interaction.channel_id != channel_id:
        await interaction.response.send_message(WRONG_CHANNEL, ephemeral=True)
        return
    await interaction.response.defer(ephemeral=True, thinking=True)
    try:
        text = answer(await service.check_now())
    except RiotError as exc:
        text = NO_ANSWER.format(why=exc)
    await interaction.followup.send(text, ephemeral=True)


def make_client(channel_id: int, service: Callable[[], ProphecyService]) -> discord.Client:
    """A gateway client that registers `/predict` for the channel's guild when it starts.

    `service` is looked up when the command runs, because the service needs this client's announcer first."""
    import discord
    from discord import app_commands

    class OracleClient(discord.Client):
        def __init__(self) -> None:
            super().__init__(intents=discord.Intents.none())
            self.tree = app_commands.CommandTree(self)

        async def setup_hook(self) -> None:
            try:
                channel = await self.fetch_channel(channel_id)
                guild = getattr(channel, "guild", None)
                if guild is None:
                    raise ValueError("DISCORD_CHANNEL_ID is not a server channel")
                scope = discord.Object(id=guild.id)

                async def predict(interaction):   # unannotated: discord.py resolves annotation strings here, where discord is imported lazily
                    await handle_predict(interaction, service(), channel_id)

                self.tree.add_command(app_commands.Command(
                    name="predict", description="Ask the oracle about a ranked game running right now",
                    callback=predict), guild=scope)
                await self.tree.sync(guild=scope)
            except (discord.DiscordException, ValueError) as exc:
                log.error("could not register /predict (%s); automatic prophecies still work", exc)

    return OracleClient()
