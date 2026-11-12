"""What the dataset should contain (`DatasetSpec`) and the kinds of missing pieces (`Gap`).

A gap is one API call's worth of work. Gaps are never stored: Mirror.gaps() derives them from the
database every time, so there is no queue that can disagree with the data.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import ClassVar

from oracle.ids import EpochMs, MatchId, Puuid
from oracle.riot import Host

DAY_MS = 86_400_000


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    queue_id: int = 420                  # ranked solo/duo
    history_n: int = 20                  # prior games per player per anchor. MUST match what the bot uses live.
    lookback_days: int = 120             # tracked players' games older than this are not synced
    max_depth: int = 1                   # 0 = only tracked players' games become training rows
    max_anchors: int = 5000              # promotion stops here
    wip_anchors: int = 20                # at most this many incomplete promoted anchors at once
    timelines_max_depth: int = 0         # fetch timelines for anchors up to this depth; -1 = none
    snapshot_window_days: int = 3        # rank/mastery snapshots only for anchors this recent
    snapshot_kinds: tuple[str, ...] = ("league", "mastery")
    sync_poll_s: int = 900               # how often a tracked player's match list is re-checked

    def lookback_start_ms(self, now: EpochMs) -> EpochMs:
        return EpochMs(now - self.lookback_days * DAY_MS)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


class SnapshotKind(StrEnum):
    LEAGUE = "league"
    MASTERY = "mastery"


GapKey = tuple[str, str, str]   # (kind, k1, k2) = fetch_failure's primary key


@dataclass(frozen=True, slots=True)
class SyncTracked:
    """Fetch the next page of a tracked player's match-id list."""

    kind: ClassVar[str] = "sync"
    host: ClassVar[Host] = Host.REGIONAL
    puuid: Puuid
    queue_id: int
    lookback_start_ms: EpochMs   # first sync only: how far back to list
    poll_ms: int                 # after a complete cycle, re-check this much later

    @property
    def key(self) -> GapKey:
        return (self.kind, self.puuid, str(self.queue_id))


@dataclass(frozen=True, slots=True)
class FetchMatch:
    kind: ClassVar[str] = "match"
    host: ClassVar[Host] = Host.REGIONAL
    match_id: MatchId

    @property
    def key(self) -> GapKey:
        return (self.kind, self.match_id, "")


@dataclass(frozen=True, slots=True)
class ListHistory:
    """'The n most recent queue matches of puuid that started before before_ms.'"""

    kind: ClassVar[str] = "listing"
    host: ClassVar[Host] = Host.REGIONAL
    puuid: Puuid
    queue_id: int
    before_ms: EpochMs
    n: int

    @property
    def key(self) -> GapKey:   # n is not part of the key: a longer listing replaces a shorter one
        return (self.kind, self.puuid, f"{self.queue_id}:{self.before_ms}")


@dataclass(frozen=True, slots=True)
class FetchTimeline:
    kind: ClassVar[str] = "timeline"
    host: ClassVar[Host] = Host.REGIONAL
    match_id: MatchId

    @property
    def key(self) -> GapKey:
        return (self.kind, self.match_id, "")


@dataclass(frozen=True, slots=True)
class TakeSnapshot:
    kind: ClassVar[str] = "snapshot"
    host: ClassVar[Host] = Host.PLATFORM
    snapshot: SnapshotKind
    puuid: Puuid

    @property
    def key(self) -> GapKey:
        return (self.kind, self.puuid, self.snapshot.value)


Gap = SyncTracked | FetchMatch | ListHistory | FetchTimeline | TakeSnapshot
