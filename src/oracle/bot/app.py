"""`oracle bot`: one process that crawls forever, watches for live games, and posts a prophecy for each.

    crawl(forever, on_live=service.submit)   the live watcher claims a new ranked game for the service
    service.run()                            prepare -> predict -> record -> announce, one game at a time
    client.start(token)                      the Discord connection (absent with --dry-run)

All three share one event loop and one `RiotGateway`, so the process holds the key lock: it replaces
`oracle crawl --forever` and cannot run next to another crawler. If any of the three stops, the others are
stopped too, so a dead Riot key or Discord login never leaves a bot that looks alive and posts nothing.
"""

from __future__ import annotations

import asyncio
import logging

from oracle.bot.announce import Announcer, ConsoleAnnouncer, DiscordAnnouncer
from oracle.bot.champions import ChampionNames
from oracle.bot.errors import AnnounceError
from oracle.bot.predictor import Predictor
from oracle.bot.service import PREPARE_DEADLINE_S, ProphecyService
from oracle.config import Settings
from oracle.crawl import CrawlBudget, CrawlReport, crawl
from oracle.features import FeatureParams
from oracle.mirror import DatasetSpec, Mirror, champion_names
from oracle.riot import RiotGateway

log = logging.getLogger(__name__)

SPECTATOR_DEADLINE_S = 8.0     # /predict: a person is waiting
SHUTDOWN_GRACE_S = 5.0


async def run_bot(settings: Settings, *, dry_run: bool) -> int:
    params = FeatureParams()
    predictor = Predictor.load(settings.model_dir, params)
    logging.getLogger("discord.http").setLevel(logging.WARNING)

    async with RiotGateway.open(settings) as gateway:
        with Mirror.open(settings.db_path) as mirror:
            con = Mirror.read_only(settings.db_path)
            try:
                names = ChampionNames(champion_names(con))
            finally:
                con.close()
            stop = asyncio.Event()
            client = None
            announcer: Announcer = ConsoleAnnouncer()
            if not dry_run:
                from oracle.bot.commands import make_client

                assert settings.discord is not None   # Settings.from_env(require_discord=True) guarantees it
                client = make_client(settings.discord.channel_id, lambda: service)
                announcer = DiscordAnnouncer(client, settings.discord.channel_id)
            service = ProphecyService(
                mirror=mirror, api=gateway.interactive(deadline_s=PREPARE_DEADLINE_S),
                spectator=gateway.interactive(deadline_s=SPECTATOR_DEADLINE_S), predictor=predictor,
                announcer=announcer, db_path=settings.db_path, names=names, params=params)
            service.recover()
            tasks: dict[str, asyncio.Task] = {
                "crawl": asyncio.create_task(crawl(mirror, gateway, DatasetSpec(history_n=params.history_n),
                                                   CrawlBudget(forever=True), stop=stop, on_live=service.submit)),
                "service": asyncio.create_task(service.run()),
            }
            if client is not None:
                tasks["discord"] = asyncio.create_task(_discord(client, settings))
            log.info("the oracle is watching%s (model %s)", " (dry run: printing, not posting)" if dry_run else "",
                     predictor.model_id)
            try:
                done, _ = await asyncio.wait(tasks.values(), return_when=asyncio.FIRST_COMPLETED)
                return _outcome(tasks, done)
            finally:
                await _shut_down(tasks, stop, client)


async def _discord(client, settings: Settings) -> None:
    import discord

    assert settings.discord is not None
    try:
        await client.start(settings.discord.token.reveal())
    except discord.LoginFailure as exc:
        raise AnnounceError("Discord rejected DISCORD_BOT_TOKEN (login failed). Reset the token in the developer "
                            "portal and update .env.") from exc


def _outcome(tasks: dict[str, asyncio.Task], done: set[asyncio.Task]) -> int:
    """Why the bot is stopping, and the exit code."""
    for name, task in tasks.items():
        if task not in done:
            continue
        if task.cancelled():
            log.error("%s stopped unexpectedly", name)
            return 1
        exc = task.exception()
        if exc is not None:
            raise exc
        if name == "crawl":
            report: CrawlReport = task.result()
            log.info(report.summary())
            return 1 if report.error else 0
        log.error("%s stopped unexpectedly", name)
    return 1


async def _shut_down(tasks: dict[str, asyncio.Task], stop: asyncio.Event, client) -> None:
    """Stop Discord and the service at once, and give the crawl a moment to finish its bookkeeping."""
    stop.set()
    for name, task in tasks.items():
        if name != "crawl":
            task.cancel()
    if client is not None:
        await client.close()
    crawling = tasks["crawl"]
    if not crawling.done():
        try:
            await asyncio.wait_for(asyncio.shield(crawling), SHUTDOWN_GRACE_S)
        except (TimeoutError, asyncio.CancelledError):
            crawling.cancel()
    await asyncio.gather(*tasks.values(), return_exceptions=True)
