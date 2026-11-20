"""Command line: `uv run --env-file .env oracle <command>`. Composition root: builds Settings, gateway, mirror."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from oracle.bot.errors import BotError
from oracle.config import ConfigError, Settings
from oracle.crawl import CrawlBudget, crawl
from oracle.ids import RiotId
from oracle.mirror import DatasetSpec, Mirror
from oracle.riot import KeyInUse, RiotError, RiotGateway

log = logging.getLogger("oracle")


async def _track(settings: Settings, riot_id: RiotId) -> None:
    async with RiotGateway.open(settings) as gateway:
        with Mirror.open(settings.db_path) as mirror:
            puuid = await mirror.track(gateway.background(), riot_id)
    print(f"tracking {riot_id} ({puuid[:8]}…). Run `oracle crawl` to fetch their games.")


async def _crawl(settings: Settings, args: argparse.Namespace) -> int:
    budget = CrawlBudget(max_calls=args.max_calls, forever=args.forever)
    async with RiotGateway.open(settings) as gateway:
        with Mirror.open(settings.db_path) as mirror:
            report = await crawl(mirror, gateway, DatasetSpec(), budget)
    print(report.summary())
    return 1 if report.error else 0


def _dataset(settings: Settings, out: Path | None) -> None:
    from oracle.features import Dataset   # pandas import is slow: only for this command

    con = Mirror.read_only(settings.db_path)
    try:
        ds = Dataset.build(con)
    finally:
        con.close()
    m = ds.manifest
    target = out or settings.data_dir.parent / "training" / "data" / m["dataset_id"].replace("@", "-")
    ds.save(target)
    print(f"{m['dataset_id']}: {m['rows']['matches']} matches, {m['rows']['seats']} seats "
          f"({m['skipped']['incomplete_history']} anchors skipped: history not complete yet)")
    print(f"patches {m['patches'][0] if m['patches'] else '-'}..{m['patches'][-1] if m['patches'] else '-'}; "
          f"saved to {target}")


def _prophesy(settings: Settings, args: argparse.Namespace) -> int:
    from oracle.bot.replay import replay   # pandas and scikit-learn load only for the bot commands

    result = replay(settings, args.match_id, args.seat)
    print(result.message.as_text())
    if args.send:
        from oracle.bot.announce import send_once

        assert settings.discord is not None   # Settings.from_env(require_discord=True) guarantees it
        message_id = asyncio.run(send_once(settings.discord, result.message))
        print(f"\nsent to channel {settings.discord.channel_id} (message {message_id})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="oracle", description="Riot data pipeline for Oracle of the Rift.")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("track", help="start tracking a player (1 API call)").add_argument("riot_id", help="'Name#TAG'")
    sub.add_parser("untrack", help="stop tracking a player (keeps their data)").add_argument("riot_id")
    sub.add_parser("tracked", help="list tracked players")
    crawl_p = sub.add_parser("crawl", help="fetch missing data; Ctrl-C any time, rerun to resume")
    crawl_p.add_argument("--max-calls", type=int, default=None, help="stop after this many API calls")
    crawl_p.add_argument("--forever", action="store_true", help="keep polling for new games")
    sub.add_parser("status", help="progress, gaps and failures (no API calls)")
    sub.add_parser("reproject", help="re-parse stored raw JSON after a parser change (no API calls)")
    sub.add_parser("retry-failed", help="clear failure backoffs so failed fetches are retried")
    ds_p = sub.add_parser("dataset", help="build the training bundle (parquet + manifest) for the notebooks (no API calls)")
    ds_p.add_argument("--out", type=Path, default=None, help="output folder (default: training/data/<dataset id>)")
    bot_p = sub.add_parser("bot", help="run the Discord bot: crawl forever and prophesy every ranked game of a tracked player")
    bot_p.add_argument("--dry-run", action="store_true", help="print prophecies instead of posting them (no Discord needed)")
    sub.add_parser("prophecies", help="score the stored prophecies against finished games (no API calls)")
    pro_p = sub.add_parser("prophesy", help="replay a stored game as if it were live and show the real result (no API calls)")
    pro_p.add_argument("match_id", nargs="?", default=None, help="e.g. EUN1_4017659773 (default: newest with complete history)")
    pro_p.add_argument("--seat", type=int, default=None, help="participant 1-10 who plays the subject (default: first tracked player)")
    pro_p.add_argument("--send", action="store_true", help="also post the message to the Discord channel")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)   # it logs every request URL at INFO

    needs_key = args.command in ("track", "crawl", "bot")
    try:
        settings = Settings.from_env(
            require_key=needs_key,
            require_model=args.command in ("prophesy", "bot"),
            require_discord=(args.command == "prophesy" and args.send) or (args.command == "bot" and not args.dry_run),
        )
        match args.command:
            case "track":
                asyncio.run(_track(settings, RiotId.parse(args.riot_id)))
            case "crawl":
                return asyncio.run(_crawl(settings, args))
            case "untrack":
                with Mirror.open(settings.db_path) as mirror:
                    found = mirror.untrack(RiotId.parse(args.riot_id))
                print("untracked" if found else f"{args.riot_id} is not tracked")
            case "tracked":
                with Mirror.open(settings.db_path) as mirror:
                    for riot_id, puuid in mirror.tracked():
                        print(f"{riot_id}  ({puuid[:8]}…)")
            case "status":
                with Mirror.open(settings.db_path) as mirror:
                    print(mirror.status(DatasetSpec()).summary())
            case "reproject":
                with Mirror.open(settings.db_path) as mirror:
                    ok, failed = mirror.reproject()
                print(f"reprojected {ok} matches, {failed} failed")
            case "dataset":
                _dataset(settings, args.out)
            case "retry-failed":
                with Mirror.open(settings.db_path) as mirror:
                    print(f"cleared {mirror.retry_failed()} failure rows")
            case "prophesy":
                return _prophesy(settings, args)
            case "bot":
                from oracle.bot.app import run_bot

                return asyncio.run(run_bot(settings, dry_run=args.dry_run))
            case "prophecies":
                from oracle.bot.scoring import report

                with Mirror.open(settings.db_path) as mirror:
                    print(report(mirror.prophecy_counts(), mirror.scored_prophecies()))
    except KeyboardInterrupt:
        print("\nstopped. Nothing is lost: rerun the same command to resume.")
        return 130
    except (BotError, ConfigError, KeyInUse, LookupError, ValueError, RiotError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
