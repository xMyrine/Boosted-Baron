"""Command line: `uv run --env-file .env oracle <command>`. Composition root: builds Settings, gateway, mirror."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

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
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)   # it logs every request URL at INFO

    needs_key = args.command in ("track", "crawl")
    try:
        settings = Settings.from_env(require_key=needs_key)
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
            case "retry-failed":
                with Mirror.open(settings.db_path) as mirror:
                    print(f"cleared {mirror.retry_failed()} failure rows")
    except KeyboardInterrupt:
        print("\nstopped. Nothing is lost: rerun the same command to resume.")
        return 130
    except (ConfigError, KeyInUse, LookupError, ValueError, RiotError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
