"""Mirror: the only code that touches SQLite. A local, append-mostly copy of what Riot told us.

    with Mirror.open(path) as mirror:
        await mirror.track(api, RiotId.parse("Name#TAG"))
        for gap in mirror.gaps(spec, Host.REGIONAL, now_ms(), limit=10):
            await mirror.fill(gap, api)          # one API call, then ONE transaction

Checkpointing: `fill` awaits the API first and then writes everything in a single transaction that
contains no `await`. A crash or Ctrl-C therefore leaves either the whole result or nothing, and since
gaps are derived from the data, a restart simply recomputes what is still missing.
"""

from __future__ import annotations

import importlib.resources
import logging
import re
import sqlite3
import zlib
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

from oracle.ids import EpochMs, MatchId, Puuid, RiotId, now_ms
from oracle.mirror import _frontier
from oracle.mirror._project import PARSER_VERSION, PayloadError, match_rows
from oracle.mirror.gaps import (
    DatasetSpec,
    FetchMatch,
    FetchTimeline,
    Gap,
    ListHistory,
    SnapshotKind,
    SyncTracked,
    TakeSnapshot,
)
from oracle.riot import GONE, BadRequestError, Doc, Gone, Host, RiotApi, TransientError

log = logging.getLogger(__name__)

SCHEMA_SQL = importlib.resources.files("oracle.mirror").joinpath("schema.sql").read_text()
SCHEMA_VERSION = int(re.search(r"PRAGMA user_version = (\d+);", SCHEMA_SQL).group(1))  # type: ignore[union-attr]

SYNC_PAGE = 100                   # match-v5 ids: max count per call
SETTLE_MS = 3 * 3_600_000         # re-list this far back: a game running during a sync shows up later
FAILURE_BASE_MS = 10 * 60_000     # retry backoff: 10 min * 2^attempts ...
FAILURE_CAP_MS = 24 * 3_600_000   # ... capped at 24 h
PARK_AFTER = 6                    # attempts; parked gaps wait for `oracle retry-failed`


class KeyMismatch(Exception):
    """The API key belongs to a different Riot app than the one that produced the stored PUUIDs."""


class FillResult(Enum):
    FILLED = "filled"
    GONE = "gone"          # 404 recorded as a fact: the gap is closed for good
    DEFERRED = "deferred"  # transient failure recorded in fetch_failure: the gap returns after its backoff


@dataclass(frozen=True, slots=True)
class MirrorStatus:
    tracked: list[str]
    anchors: dict[int, tuple[int, int]]       # depth -> (total, complete)
    gaps: dict[str, int]                      # gap kind -> currently known gaps
    pending_listings: int                     # listings not fetched yet: each will reveal up to n matches
    history_n: int
    failures_deferred: int
    failures_parked: int
    unprojected: int
    matches: int
    calls_today: int
    last_run: str | None

    def summary(self) -> str:
        known = sum(n for kind, n in self.gaps.items() if kind != "snapshot")
        upper = known + self.pending_listings * self.history_n
        lines = [
            f"tracked    {', '.join(self.tracked) or '(none: run `oracle track \"Name#TAG\"`)'}",
            "anchors    " + ("  ".join(f"depth{d} {c}/{t} complete" for d, (t, c) in sorted(self.anchors.items()))
                             or "none yet"),
            f"matches    {self.matches} stored, {self.unprojected} unprojected",
            "gaps       " + ("  ".join(f"{k} {n}" for k, n in sorted(self.gaps.items())) or "none"),
            f"estimate   {known}..{upper} regional calls left for current anchors (~{upper / 2700:.1f} h at most)",
            f"failures   {self.failures_deferred} deferred, {self.failures_parked} parked",
            f"calls      {self.calls_today} today",
        ]
        if self.last_run:
            lines.append(f"last run   {self.last_run}")
        return "\n".join(lines)


class Mirror:
    def __init__(self, con: sqlite3.Connection) -> None:
        self._con = con

    # ---------------------------------------------------------------- lifecycle

    @classmethod
    def open(cls, path: Path) -> Mirror:
        """Open (creating if needed) the database. Applies schema.sql to an empty file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(path, isolation_level=None)   # transactions are explicit (see _tx)
        con.execute("PRAGMA foreign_keys = ON")
        con.execute("PRAGMA busy_timeout = 30000")
        con.execute("PRAGMA synchronous = NORMAL")
        version = con.execute("PRAGMA user_version").fetchone()[0]
        if version == 0:
            con.executescript(SCHEMA_SQL)
            con.execute("PRAGMA journal_mode = WAL")
            log.info("created %s at schema v%d", path, SCHEMA_VERSION)
        elif version != SCHEMA_VERSION:
            con.close()
            raise RuntimeError(f"{path} is at schema v{version}, this code expects v{SCHEMA_VERSION}")
        return cls(con)

    @staticmethod
    def read_only(path: Path) -> sqlite3.Connection:
        """A read-only connection for notebooks (no lock, never blocks the crawler)."""
        return sqlite3.connect(f"file:{path}?mode=ro", uri=True)

    def close(self) -> None:
        self._con.close()

    def __enter__(self) -> Mirror:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        """One write transaction. Never `await` inside: that is what makes every fill atomic."""
        self._con.execute("BEGIN IMMEDIATE")
        try:
            yield self._con
        except BaseException:
            self._con.execute("ROLLBACK")
            raise
        self._con.execute("COMMIT")

    # ---------------------------------------------------------------- tracked players

    async def track(self, api: RiotApi, riot_id: RiotId) -> Puuid:
        """Start tracking a player: their ranked games become depth-0 anchors (training rows)."""
        puuid = await api.account(riot_id)
        if puuid is None:
            raise LookupError(f"no Riot account named {riot_id}")
        with self._tx() as con:
            con.execute(
                """INSERT INTO tracked_player (puuid, game_name, tag_line, added_at_ms) VALUES (?, ?, ?, ?)
                   ON CONFLICT (puuid) DO UPDATE SET game_name = excluded.game_name, tag_line = excluded.tag_line""",
                (puuid, riot_id.game_name, riot_id.tag_line, now_ms()),
            )
        return puuid

    def untrack(self, riot_id: RiotId) -> bool:
        """Stop syncing a player. Already-collected anchors and matches are kept."""
        with self._tx() as con:
            row = con.execute(
                "SELECT puuid FROM tracked_player WHERE lower(game_name) = lower(?) AND lower(tag_line) = lower(?)",
                (riot_id.game_name, riot_id.tag_line),
            ).fetchone()
            if row is None:
                return False
            con.execute("DELETE FROM sync_cursor WHERE puuid = ?", row)
            con.execute("DELETE FROM tracked_player WHERE puuid = ?", row)
        return True

    def tracked(self) -> list[tuple[RiotId, Puuid]]:
        return [
            (RiotId(name, tag), Puuid(puuid))
            for puuid, name, tag in self._con.execute(
                "SELECT puuid, game_name, tag_line FROM tracked_player ORDER BY added_at_ms"
            )
        ]

    async def verify_key(self, api: RiotApi) -> None:
        """PUUIDs are encrypted per Riot API app. Re-resolve one tracked Riot ID (1 call): a different PUUID
        means this key belongs to another app and every stored PUUID would be wrong for API calls."""
        players = self.tracked()
        if not players:
            return
        riot_id, stored = players[0]
        current = await api.account(riot_id)
        if current is None:
            log.warning("could not re-resolve %s (renamed?); skipping the key check", riot_id)
        elif current != stored:
            raise KeyMismatch(
                f"{riot_id} resolves to a different PUUID with this API key: the key belongs to another Riot app. "
                "Use the original key, or start a fresh database."
            )

    # ---------------------------------------------------------------- work

    def gaps(self, spec: DatasetSpec, host: Host, now: EpochMs, limit: int) -> list[Gap]:
        """The most valuable missing pieces for one routing lane, best first."""
        if host is Host.PLATFORM:
            found = _frontier.platform(self._con, spec, now, limit)
        elif host is Host.REGIONAL:
            found = _frontier.regional(self._con, spec, now, limit)
        else:
            return []
        found.sort(key=lambda pair: pair[0])
        seen: set[tuple[str, str, str]] = set()
        out: list[Gap] = []
        for _, gap in found:
            if gap.key not in seen:
                seen.add(gap.key)
                out.append(gap)
        return out[:limit]

    async def fill(self, gap: Gap, api: RiotApi) -> FillResult:
        """Make the one API call this gap needs and record the answer atomically."""
        try:
            match gap:
                case SyncTracked():
                    return await self._fill_sync(gap, api)
                case FetchMatch():
                    return await self._fill_match(gap, api)
                case ListHistory():
                    return await self._fill_listing(gap, api)
                case FetchTimeline():
                    return await self._fill_timeline(gap, api)
                case TakeSnapshot():
                    return await self._fill_snapshot(gap, api)
        except TransientError as exc:
            self._record_failure(gap, str(exc), park=False)
            return FillResult.DEFERRED
        except BadRequestError as exc:
            self._record_failure(gap, str(exc), park=True)
            raise

    def promote(self, spec: DatasetSpec, now: EpochMs) -> int:
        """Turn fetched history games of shallower anchors into new anchors, within the WIP and size limits."""
        total = self._con.execute("SELECT count(*) FROM anchor").fetchone()[0]
        room = min(spec.max_anchors - total, spec.wip_anchors - _frontier.count_incomplete_promoted(self._con, spec))
        if room <= 0 or spec.max_depth <= 0:
            return 0
        candidates = _frontier.promotion_candidates(self._con, spec, now, room)
        with self._tx() as con:
            con.executemany(
                "INSERT OR IGNORE INTO anchor (match_id, depth, source, added_at_ms) VALUES (?, ?, 'promoted', ?)",
                [(match_id, depth, now) for match_id, depth in candidates],
            )
        return len(candidates)

    async def _fill_sync(self, gap: SyncTracked, api: RiotApi) -> FillResult:
        now = now_ms()
        row = self._con.execute(
            "SELECT since_ms, page_offset, cycle_started_ms FROM sync_cursor WHERE puuid = ? AND queue_id = ?",
            (gap.puuid, gap.queue_id),
        ).fetchone()
        since, offset, cycle_started = row if row else (gap.lookback_start_ms, 0, None)
        ids = await api.match_ids(gap.puuid, queue=gap.queue_id, start=offset, count=SYNC_PAGE,
                                  start_ms=EpochMs(since))
        started = cycle_started if offset > 0 and cycle_started is not None else now
        with self._tx() as con:
            con.executemany(
                """INSERT INTO anchor (match_id, depth, source, added_at_ms) VALUES (?, 0, 'tracked', ?)
                   ON CONFLICT (match_id) DO UPDATE SET depth = 0, source = 'tracked'""",
                [(match_id, now) for match_id in ids],
            )
            if len(ids) < SYNC_PAGE:   # cycle complete: next cycle re-lists from a little before this one began
                cursor = (started - SETTLE_MS, 0, None, now + gap.poll_ms)
            else:                      # more pages: checkpoint the position
                cursor = (since, offset + len(ids), started, 0)
            con.execute(
                """INSERT INTO sync_cursor (puuid, queue_id, since_ms, page_offset, cycle_started_ms, next_due_ms)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT (puuid, queue_id) DO UPDATE SET since_ms = excluded.since_ms,
                       page_offset = excluded.page_offset, cycle_started_ms = excluded.cycle_started_ms,
                       next_due_ms = excluded.next_due_ms""",
                (gap.puuid, gap.queue_id, *cursor),
            )
            self._clear_failure(con, gap)
        return FillResult.FILLED

    async def _fill_match(self, gap: FetchMatch, api: RiotApi) -> FillResult:
        doc = await api.match(gap.match_id)
        with self._tx() as con:
            inserted = self._insert_raw(con, "raw_match", gap.match_id, doc)
            if inserted and isinstance(doc, Doc):
                self._project(con, gap.match_id, doc.body)
            self._clear_failure(con, gap)
        return FillResult.GONE if doc is GONE else FillResult.FILLED

    async def _fill_listing(self, gap: ListHistory, api: RiotApi) -> FillResult:
        ids = await api.match_ids(gap.puuid, queue=gap.queue_id, count=gap.n, end_ms=gap.before_ms)
        key = (gap.puuid, gap.queue_id, gap.before_ms)
        with self._tx() as con:
            con.execute("DELETE FROM history_listing WHERE puuid = ? AND queue_id = ? AND before_ms = ?", key)
            con.execute(
                """INSERT INTO history_listing (puuid, queue_id, before_ms, n_requested, fetched_at_ms)
                   VALUES (?, ?, ?, ?, ?)""",
                (*key, gap.n, now_ms()),
            )
            con.executemany(
                """INSERT INTO history_listing_item (puuid, queue_id, before_ms, ord, match_id)
                   VALUES (?, ?, ?, ?, ?)""",
                [(*key, ord_, match_id) for ord_, match_id in enumerate(ids)],
            )
            self._clear_failure(con, gap)
        return FillResult.FILLED

    async def _fill_timeline(self, gap: FetchTimeline, api: RiotApi) -> FillResult:
        doc = await api.timeline(gap.match_id)
        with self._tx() as con:
            self._insert_raw(con, "raw_timeline", gap.match_id, doc)
            self._clear_failure(con, gap)
        return FillResult.GONE if doc is GONE else FillResult.FILLED

    async def _fill_snapshot(self, gap: TakeSnapshot, api: RiotApi) -> FillResult:
        match gap.snapshot:
            case SnapshotKind.LEAGUE:
                doc = await api.league_entries(gap.puuid)
            case SnapshotKind.MASTERY:
                doc = await api.mastery(gap.puuid)
        with self._tx() as con:
            con.execute(
                "INSERT OR IGNORE INTO snapshot (kind, puuid, taken_at_ms, payload) VALUES (?, ?, ?, ?)",
                (gap.snapshot.value, gap.puuid, now_ms(), doc.body.decode()),
            )
            self._clear_failure(con, gap)
        return FillResult.FILLED

    # ---------------------------------------------------------------- write helpers (inside a transaction)

    @staticmethod
    def _insert_raw(con: sqlite3.Connection, table: str, match_id: MatchId, doc: Doc | Gone) -> bool:
        """Insert-only. Returns False if another fill already stored this id (idempotent replay)."""
        payload = zlib.compress(doc.body) if isinstance(doc, Doc) else None
        cur = con.execute(
            f"INSERT OR IGNORE INTO {table} (match_id, outcome, payload, fetched_at_ms) VALUES (?, ?, ?, ?)",
            (match_id, "ok" if payload is not None else "gone", payload, now_ms()),
        )
        return cur.rowcount == 1

    @staticmethod
    def _project(con: sqlite3.Connection, match_id: MatchId, body: bytes) -> bool:
        """(Re)build match + participant rows from raw JSON. A bad payload keeps the raw row unprojected."""
        try:
            match, participants = match_rows(body)
        except PayloadError as exc:
            log.warning("%s stored raw but not projected: %s", match_id, exc)
            return False
        con.execute("SAVEPOINT project")
        try:
            con.execute("DELETE FROM participant WHERE match_id = ?", (match_id,))
            con.execute("DELETE FROM match WHERE match_id = ?", (match_id,))
            con.execute("INSERT INTO match VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", match.values())
            con.executemany(
                "INSERT INTO participant VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [p.values() for p in participants],
            )
            con.execute("UPDATE raw_match SET parsed_version = ? WHERE match_id = ?", (PARSER_VERSION, match_id))
        except sqlite3.IntegrityError as exc:
            con.execute("ROLLBACK TO project")
            log.warning("%s stored raw but not projected: %s", match_id, exc)
            return False
        finally:
            con.execute("RELEASE project")
        return True

    @staticmethod
    def _clear_failure(con: sqlite3.Connection, gap: Gap) -> None:
        con.execute("DELETE FROM fetch_failure WHERE kind = ? AND k1 = ? AND k2 = ?", gap.key)

    def _record_failure(self, gap: Gap, error: str, *, park: bool) -> None:
        now = now_ms()
        with self._tx() as con:
            row = con.execute(
                "SELECT attempts FROM fetch_failure WHERE kind = ? AND k1 = ? AND k2 = ?", gap.key
            ).fetchone()
            attempts = (row[0] if row else 0) + 1
            retry_at = now + min(FAILURE_BASE_MS * 2 ** attempts, FAILURE_CAP_MS)
            parked = int(park or attempts >= PARK_AFTER)
            con.execute(
                """INSERT INTO fetch_failure (kind, k1, k2, attempts, next_retry_at_ms, parked, last_error)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT (kind, k1, k2) DO UPDATE SET attempts = excluded.attempts,
                       next_retry_at_ms = excluded.next_retry_at_ms, parked = excluded.parked,
                       last_error = excluded.last_error""",
                (*gap.key, attempts, retry_at, parked, error[:500]),
            )
        log.warning("%s %s failed (attempt %d%s): %s", gap.kind, gap.key[1], attempts,
                    ", parked" if parked else "", error)

    # ---------------------------------------------------------------- bookkeeping

    def record_calls(self, stats: Iterable[tuple[tuple[str, str, str, str, str], int]]) -> None:
        rows = [(*key, n) for key, n in stats]
        if not rows:
            return
        with self._tx() as con:
            con.executemany(
                """INSERT INTO api_call_stat (day, routing, endpoint, lane, outcome, calls) VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT (day, routing, endpoint, lane, outcome) DO UPDATE SET calls = calls + excluded.calls""",
                rows,
            )

    def calls_today(self) -> int:
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        return self._con.execute("SELECT coalesce(sum(calls), 0) FROM api_call_stat WHERE day = ?",
                                 (today,)).fetchone()[0]

    def run_started(self, spec: DatasetSpec) -> int:
        with self._tx() as con:
            cur = con.execute("INSERT INTO crawl_run (started_ms, spec_json) VALUES (?, ?)", (now_ms(), spec.to_json()))
        return int(cur.lastrowid or 0)

    def run_ended(self, run_id: int, calls: int, stop_reason: str) -> None:
        with self._tx() as con:
            con.execute("UPDATE crawl_run SET ended_ms = ?, calls = ?, stop_reason = ? WHERE run_id = ?",
                        (now_ms(), calls, stop_reason, run_id))

    def retry_failed(self) -> int:
        """Forget all failure rows so deferred and parked gaps are tried again."""
        with self._tx() as con:
            return con.execute("DELETE FROM fetch_failure").rowcount

    def reproject(self) -> tuple[int, int]:
        """Rebuild match/participant from raw JSON for rows parsed by an older (or no) parser. 0 API calls."""
        ok = failed = 0
        after = ""   # keyset pagination: failed rows stay unprojected, so they must not be re-selected forever
        while True:
            batch = self._con.execute(
                """SELECT match_id, payload FROM raw_match
                   WHERE outcome = 'ok' AND (parsed_version IS NULL OR parsed_version < ?)
                     AND match_id > ? ORDER BY match_id LIMIT 500""",
                (PARSER_VERSION, after),
            ).fetchall()
            if not batch:
                return ok, failed
            with self._tx() as con:
                for match_id, payload in batch:
                    if self._project(con, MatchId(match_id), zlib.decompress(payload)):
                        ok += 1
                    else:
                        failed += 1
            after = batch[-1][0]

    def status(self, spec: DatasetSpec) -> MirrorStatus:
        con, now = self._con, now_ms()
        anchors: dict[int, tuple[int, int]] = {}
        for depth, total, incomplete in con.execute(
            f"SELECT a.depth, count(*), sum({_frontier.INCOMPLETE_ANCHOR}) FROM anchor a GROUP BY a.depth",
            {"n": spec.history_n},
        ):
            anchors[depth] = (total, total - (incomplete or 0))
        gaps: dict[str, int] = {}
        for _, gap in [*_frontier.regional(con, spec, now, 10**9), *_frontier.platform(con, spec, now, 10**9)]:
            gaps[gap.kind] = gaps.get(gap.kind, 0) + 1
        deferred, parked = con.execute(
            "SELECT coalesce(sum(parked = 0), 0), coalesce(sum(parked = 1), 0) FROM fetch_failure"
        ).fetchone()
        last = con.execute(
            "SELECT run_id, calls, stop_reason, started_ms FROM crawl_run ORDER BY run_id DESC LIMIT 1"
        ).fetchone()
        return MirrorStatus(
            tracked=[str(riot_id) for riot_id, _ in self.tracked()],
            anchors=anchors,
            gaps=gaps,
            pending_listings=gaps.get("listing", 0),
            history_n=spec.history_n,
            failures_deferred=deferred,
            failures_parked=parked,
            unprojected=con.execute(
                "SELECT count(*) FROM raw_match WHERE outcome = 'ok' AND parsed_version IS NULL").fetchone()[0],
            matches=con.execute("SELECT count(*) FROM raw_match").fetchone()[0],
            calls_today=self.calls_today(),
            last_run=(f"#{last[0]}: {last[1]} calls, stop: {last[2] or 'running or killed'}" if last else None),
        )
