"""SQL that derives the work list (gaps) from the database. Private to `mirror`.

Ordering = value per call. On the regional lane:
  band 0  tracked players' syncs (new anchors)
  band 1  anchors in (depth, seq) order; per anchor: its match (step 0), its 10 listings (1), its history matches (2)
  band 2  timelines
Depth-first per anchor means an interrupted crawl leaves the most COMPLETE training rows for the calls spent.
On the platform lane: rank/mastery snapshots for players of the most recent anchors first.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from oracle.ids import EpochMs, MatchId, Puuid
from oracle.mirror.gaps import (
    DAY_MS,
    DatasetSpec,
    FetchMatch,
    FetchTimeline,
    Gap,
    ListHistory,
    SnapshotKind,
    SyncTracked,
    TakeSnapshot,
)

SortKey = tuple[int, int, int, int]


def _not_blocked(kind: str, k1: str, k2: str) -> str:
    """SQL predicate: no failure row is holding this gap back (parked, or waiting for its retry time)."""
    return (
        f"NOT EXISTS (SELECT 1 FROM fetch_failure f WHERE f.kind = '{kind}' AND f.k1 = {k1} AND f.k2 = {k2}"
        " AND (f.parked = 1 OR f.next_retry_at_ms > :now))"
    )


# Correlated predicate on `a` (an anchor row): true while the anchor still lacks something.
INCOMPLETE_ANCHOR = """(
    NOT EXISTS (SELECT 1 FROM raw_match r WHERE r.match_id = a.match_id)
    OR EXISTS (SELECT 1 FROM match m JOIN participant p ON p.match_id = m.match_id
               LEFT JOIN history_listing h
                      ON h.puuid = p.puuid AND h.queue_id = m.queue_id AND h.before_ms = m.game_start_ms
               WHERE m.match_id = a.match_id AND m.is_remake = 0 AND (h.puuid IS NULL OR h.n_requested < :n))
    OR EXISTS (SELECT 1 FROM match m JOIN participant p ON p.match_id = m.match_id
               JOIN history_listing_item li
                      ON li.puuid = p.puuid AND li.queue_id = m.queue_id AND li.before_ms = m.game_start_ms
               WHERE m.match_id = a.match_id
                 AND NOT EXISTS (SELECT 1 FROM raw_match r WHERE r.match_id = li.match_id))
)"""


def regional(con: sqlite3.Connection, spec: DatasetSpec, now: EpochMs, limit: int) -> list[tuple[SortKey, Gap]]:
    p: dict[str, Any] = {"now": now, "q": spec.queue_id, "n": spec.history_n, "limit": limit,
                         "tl_depth": spec.timelines_max_depth}
    out: list[tuple[SortKey, Gap]] = []

    for (puuid,) in con.execute(f"""
        SELECT t.puuid FROM tracked_player t
        LEFT JOIN sync_cursor c ON c.puuid = t.puuid AND c.queue_id = :q
        WHERE (c.puuid IS NULL OR c.next_due_ms <= :now)
          AND {_not_blocked('sync', 't.puuid', 'CAST(:q AS TEXT)')}
        ORDER BY t.added_at_ms LIMIT :limit""", p):
        out.append(((0, 0, 0, 0), SyncTracked(Puuid(puuid), spec.queue_id, spec.lookback_start_ms(now),
                                              spec.sync_poll_s * 1000)))

    for match_id, depth, seq in con.execute(f"""
        SELECT a.match_id, a.depth, a.seq FROM anchor a
        WHERE NOT EXISTS (SELECT 1 FROM raw_match r WHERE r.match_id = a.match_id)
          AND {_not_blocked('match', 'a.match_id', "''")}
        ORDER BY a.depth, a.seq LIMIT :limit""", p):
        out.append(((1, depth, seq, 0), FetchMatch(MatchId(match_id))))

    for depth, seq, puuid, queue_id, start_ms in con.execute(f"""
        SELECT a.depth, a.seq, pa.puuid, m.queue_id, m.game_start_ms FROM anchor a
        JOIN match m ON m.match_id = a.match_id
        JOIN participant pa ON pa.match_id = a.match_id
        LEFT JOIN history_listing h
               ON h.puuid = pa.puuid AND h.queue_id = m.queue_id AND h.before_ms = m.game_start_ms
        WHERE m.is_remake = 0 AND m.queue_id = :q
          AND (h.puuid IS NULL OR h.n_requested < :n)
          AND {_not_blocked('listing', 'pa.puuid', "m.queue_id || ':' || m.game_start_ms")}
        ORDER BY a.depth, a.seq, pa.participant_id LIMIT :limit""", p):
        out.append(((1, depth, seq, 1), ListHistory(Puuid(puuid), queue_id, EpochMs(start_ms), spec.history_n)))

    for match_id, depth, seq in con.execute(f"""
        SELECT li.match_id, MIN(a.depth) AS d, MIN(a.seq) AS s FROM anchor a
        JOIN match m ON m.match_id = a.match_id
        JOIN participant pa ON pa.match_id = a.match_id
        JOIN history_listing_item li
          ON li.puuid = pa.puuid AND li.queue_id = m.queue_id AND li.before_ms = m.game_start_ms
        WHERE NOT EXISTS (SELECT 1 FROM raw_match r WHERE r.match_id = li.match_id)
          AND {_not_blocked('match', 'li.match_id', "''")}
        GROUP BY li.match_id ORDER BY d, s LIMIT :limit""", p):
        out.append(((1, depth, seq, 2), FetchMatch(MatchId(match_id))))

    for match_id, depth, seq in con.execute(f"""
        SELECT a.match_id, a.depth, a.seq FROM anchor a
        JOIN match m ON m.match_id = a.match_id
        WHERE a.depth <= :tl_depth AND m.is_remake = 0
          AND NOT EXISTS (SELECT 1 FROM raw_timeline t WHERE t.match_id = a.match_id)
          AND {_not_blocked('timeline', 'a.match_id', "''")}
        ORDER BY a.depth, a.seq LIMIT :limit""", p):
        out.append(((2, depth, seq, 0), FetchTimeline(MatchId(match_id))))

    return out


def platform(con: sqlite3.Connection, spec: DatasetSpec, now: EpochMs, limit: int) -> list[tuple[SortKey, Gap]]:
    """One snapshot per (kind, player) taken after that player's most recent anchor in the window."""
    out: list[tuple[SortKey, Gap]] = []
    since = now - spec.snapshot_window_days * DAY_MS
    for i, kind in enumerate(spec.snapshot_kinds):
        p = {"now": now, "since": since, "kind": kind, "limit": limit}
        for puuid, last_ms in con.execute(f"""
            WITH recent AS (
                SELECT pa.puuid, MAX(m.game_start_ms) AS last_ms FROM anchor a
                JOIN match m ON m.match_id = a.match_id
                JOIN participant pa ON pa.match_id = a.match_id
                WHERE m.game_start_ms >= :since
                GROUP BY pa.puuid)
            SELECT r.puuid, r.last_ms FROM recent r
            WHERE NOT EXISTS (SELECT 1 FROM snapshot s
                              WHERE s.kind = :kind AND s.puuid = r.puuid AND s.taken_at_ms >= r.last_ms)
              AND {_not_blocked('snapshot', 'r.puuid', ':kind')}
            ORDER BY r.last_ms DESC LIMIT :limit""", p):
            out.append(((0, -last_ms, i, 0), TakeSnapshot(SnapshotKind(kind), Puuid(puuid))))
    return out


def count_incomplete_promoted(con: sqlite3.Connection, spec: DatasetSpec) -> int:
    return con.execute(
        f"SELECT count(*) FROM anchor a WHERE a.depth > 0 AND {INCOMPLETE_ANCHOR}", {"n": spec.history_n}
    ).fetchone()[0]


def promotion_candidates(
    con: sqlite3.Connection, spec: DatasetSpec, now: EpochMs, limit: int
) -> list[tuple[MatchId, int]]:
    """Fetched history games of shallower anchors that are not anchors yet, in the order they'd be worked."""
    rows = con.execute("""
        SELECT li.match_id, MIN(a.depth) + 1 FROM anchor a
        JOIN match m ON m.match_id = a.match_id
        JOIN participant pa ON pa.match_id = a.match_id
        JOIN history_listing_item li
          ON li.puuid = pa.puuid AND li.queue_id = m.queue_id AND li.before_ms = m.game_start_ms
        JOIN match hm ON hm.match_id = li.match_id
        WHERE a.depth < :max_depth AND hm.is_remake = 0 AND hm.queue_id = :q AND hm.game_start_ms >= :since
          AND NOT EXISTS (SELECT 1 FROM anchor x WHERE x.match_id = li.match_id)
        GROUP BY li.match_id
        ORDER BY MIN(a.depth), MIN(a.seq), MIN(li.ord)
        LIMIT :limit""",
        {"max_depth": spec.max_depth, "q": spec.queue_id, "since": spec.lookback_start_ms(now), "limit": limit},
    ).fetchall()
    return [(MatchId(match_id), depth) for match_id, depth in rows]
