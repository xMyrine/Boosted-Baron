"""In-place upgrades of an existing database to the schema in `schema.sql`.

A fresh database gets `schema.sql` whole. An older one runs the migrations above its version, in order. Each
migration is additive (new tables, never a rewrite of facts), runs in one transaction together with its version
bump, and is safe to run twice: another process can finish the same step while we wait for the write lock.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Migration:
    version: int                   # the schema version the database has after this step
    statements: tuple[str, ...]    # one complete SQL statement each


V2_PROPHECY = Migration(2, ("""
CREATE TABLE IF NOT EXISTS prophecy (
    platform_id TEXT NOT NULL,
    game_id INTEGER NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'posted', 'failed')),
    subject_puuid TEXT NOT NULL,
    subject_side INTEGER NOT NULL CHECK (subject_side IN (100, 200)),
    payload TEXT NOT NULL CHECK (json_valid(payload)),
    claimed_at_ms INTEGER NOT NULL,
    finished_at_ms INTEGER,
    model_id TEXT,
    p_blue REAL CHECK (p_blue IS NULL OR p_blue BETWEEN 0 AND 1),
    complete INTEGER CHECK (complete IN (0, 1)),
    prophecy_json TEXT CHECK (prophecy_json IS NULL OR json_valid(prophecy_json)),
    discord_message_id TEXT,
    error TEXT,
    PRIMARY KEY (platform_id, game_id),
    CHECK ((p_blue IS NULL) = (complete IS NULL)),
    CHECK (CASE status
        WHEN 'pending' THEN finished_at_ms IS NULL AND discord_message_id IS NULL
        WHEN 'posted' THEN finished_at_ms IS NOT NULL AND discord_message_id IS NOT NULL AND prophecy_json IS NOT NULL
        ELSE finished_at_ms IS NOT NULL AND error IS NOT NULL END)
) STRICT""",))

MIGRATIONS = (V2_PROPHECY,)   # ordered by version; each is the next integer after the one before it


def upgrade(con: sqlite3.Connection, from_version: int, to_version: int) -> None:
    """Apply every migration in (from_version, to_version]. `con` must be in autocommit mode (isolation_level=None)."""
    for migration in MIGRATIONS:
        if not from_version < migration.version <= to_version:
            continue
        con.execute("BEGIN IMMEDIATE")
        try:
            current = con.execute("PRAGMA user_version").fetchone()[0]
            if current < migration.version:
                for statement in migration.statements:
                    con.execute(statement)
                con.execute(f"PRAGMA user_version = {migration.version}")
                log.info("migrated the database to schema v%d", migration.version)
        except BaseException:
            con.execute("ROLLBACK")
            raise
        con.execute("COMMIT")
