"""Build model-ready tables from the mirror, and save/load them as a portable bundle (parquet + manifest).

The bundle is what the training notebooks (locally or on Google Colab) read: they need pandas, not SQLite,
the crawler or this package. Two tables, one build:

  matches.parquet  one row per complete anchor: meta + `y_blue_win` + d_*/lane_*/ctx_* features (win model)
  seats.parquet    one row per player per anchor: seat features + labels y_kills, y_deaths, ... (performance
                   models later). Same features, so the win and performance models never disagree on inputs.

Both training and the bot go through `build_features`, so features are computed by exactly one code path.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from oracle.features import _sources
from oracle.features.params import FEATURE_CODE_VERSION, FeatureParams
from oracle.features.pregame import SEAT_COLUMNS, Pregame
from oracle.features.roles import ChampionMeta
from oracle.features.seat import CATEGORICAL, FEATURE_DOCS, compute
from oracle.features.team import match_rows

MATCH_META = ["as_of_ms", "end_ms", "duration_min", "patch", "depth", "has_tracked"]
SEAT_META = ["side", "puuid", "champion_id", "is_tracked", "true_role", "as_of_ms"]


@dataclass(frozen=True)
class Features:
    """Output of feature building: same type for training batches and one live game."""

    seats: pd.DataFrame       # indexed (game, seat): numeric seat features + role/main_role
    matches: pd.DataFrame     # indexed game: d_* / lane_* / ctx_* columns
    coverage: pd.DataFrame    # indexed (game, seat): listed, n_listed, n_fetched, complete

    def complete_games(self) -> pd.Index:
        ok = self.coverage["complete"].groupby(level="game").all()
        return ok.index[ok]


def build_features(con: sqlite3.Connection, seats: pd.DataFrame, p: FeatureParams, *,
                   knowledge_ms: int, meta: ChampionMeta | None = None) -> Features:
    """THE feature code path. `seats`: SEAT_COLUMNS, 10 rows per game (seat 0-4 blue, 5-9 red)."""
    seats = seats[SEAT_COLUMNS].sort_values(["game", "seat"]).reset_index(drop=True)
    counts = seats.groupby("game").size()
    if (counts != 10).any():
        raise ValueError(f"games without exactly 10 seats: {list(counts.index[counts != 10])[:5]}")
    if meta is None:
        meta = ChampionMeta(_sources.champion_events(con, queue_id=p.queue_id, knowledge_ms=knowledge_ms))
    hist = _sources.history(con, seats, queue_id=p.queue_id, history_n=p.history_n, knowledge_ms=knowledge_ms)
    snaps = _sources.snapshots(con, seats, knowledge_ms=knowledge_ms)
    cov = _sources.coverage(con, seats, queue_id=p.queue_id, history_n=p.history_n, knowledge_ms=knowledge_ms)
    seat_feats = compute(seats, hist, snaps, meta, p)
    return Features(seat_feats, match_rows(seats, seat_feats), cov)


def features_for_live(con: sqlite3.Connection, pregame: Pregame, p: FeatureParams | None = None) -> Features:
    """The bot's entry point (later): one game, same code as training. Needs the 10 listings fetched first."""
    return build_features(con, pregame.to_frame(), p or FeatureParams(), knowledge_ms=int(time.time() * 1000))


@dataclass
class Dataset:
    matches: pd.DataFrame
    seats: pd.DataFrame
    manifest: dict = field(default_factory=dict)

    # ------------------------------------------------------------ build

    @classmethod
    def build(cls, con: sqlite3.Connection, p: FeatureParams | None = None, *,
              knowledge_ms: int | None = None) -> Dataset:
        """Complete anchors only (all 10 histories listed and fetched), sorted by start time."""
        p = p or FeatureParams()
        k = knowledge_ms if knowledge_ms is not None else int(time.time() * 1000)
        games, seat_rows = _sources.anchors(con, queue_id=p.queue_id, knowledge_ms=k)
        seat_rows = seat_rows.copy()
        if p.snapshot_lag_days > 0:   # leaky on purpose: see FeatureParams.snapshot_lag_days
            seat_rows["snapshot_until_ms"] += int(p.snapshot_lag_days * _sources.DAY_MS)
        tracked = _sources.tracked_puuids(con)
        seat_rows["is_tracked"] = seat_rows["puuid"].isin(tracked).astype(int)

        feats = build_features(con, seat_rows, p, knowledge_ms=k)
        complete = feats.complete_games()
        skipped = {"incomplete_history": int(len(games) - len(complete))}

        g = games.set_index("game").loc[lambda d: d.index.isin(complete)]
        g["has_tracked"] = seat_rows.groupby("game")["is_tracked"].max().reindex(g.index).astype(int)
        matches = g[MATCH_META + ["y_blue_win"]].join(feats.matches).sort_values("as_of_ms")

        labels = [c for c in seat_rows.columns if c.startswith("y_")]
        s = seat_rows.set_index(["game", "seat"])
        seats = s.loc[s.index.get_level_values("game").isin(complete), SEAT_META + labels].join(feats.seats)
        seats = seats.join(feats.coverage[["n_listed"]])
        seats = seats.loc[matches.index].reset_index()   # same game order as matches

        cov = feats.coverage
        manifest = {
            "dataset_id": f"win_v{FEATURE_CODE_VERSION}@{p.id}",
            "built_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
            "knowledge_cutoff_ms": k,
            "feature_code_version": FEATURE_CODE_VERSION,
            "params": json.loads(p.to_json()),
            "git_sha": _git_sha(),
            "rows": {"matches": len(matches), "seats": len(seats), "anchors_seen": len(games)},
            "skipped": skipped,
            "seats_incomplete": int((~cov["complete"]).sum()),
            "time_range": [_iso(matches["as_of_ms"].min()), _iso(matches["as_of_ms"].max())] if len(matches) else None,
            "patches": sorted(matches["patch"].unique().tolist(), key=_patch_key),
            "columns": {
                "label": "y_blue_win",
                "match_meta": MATCH_META,
                "match_features": [c for c in matches.columns if c not in MATCH_META + ["y_blue_win"]],
                "seat_meta": SEAT_META + ["n_listed"],
                "seat_labels": labels,
                "seat_features": [c for c in feats.seats.columns if c not in CATEGORICAL],
                "seat_categorical": CATEGORICAL,
            },
            "feature_docs": {k_: list(v) for k_, v in FEATURE_DOCS.items()},
        }
        return cls(matches, seats, manifest)

    # ------------------------------------------------------------ portable bundle

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        self.matches.to_parquet(directory / "matches.parquet")
        self.seats.to_parquet(directory / "seats.parquet", index=False)
        (directory / "manifest.json").write_text(json.dumps(self.manifest, indent=2))
        return directory

    @classmethod
    def load(cls, directory: Path) -> Dataset:
        return cls(pd.read_parquet(directory / "matches.parquet"), pd.read_parquet(directory / "seats.parquet"),
                   json.loads((directory / "manifest.json").read_text()))


def _git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              check=True, cwd=Path(__file__).parent).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _iso(ms: float) -> str:
    return pd.Timestamp(int(ms), unit="ms", tz="UTC").isoformat(timespec="minutes")


def _patch_key(patch: str) -> tuple[int, ...]:
    return tuple(int(x) for x in patch.split(".") if x.isdigit())
