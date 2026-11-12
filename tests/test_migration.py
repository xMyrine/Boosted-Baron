"""Upgrading an existing database in place: rows survive, the result equals a fresh install, and a rerun is a no-op."""

from __future__ import annotations

import re
import sqlite3

import pytest
from featureworld import random_world

from oracle.mirror import Mirror, migrations
from oracle.mirror.migrations import Migration

TABLES = ("tracked_player", "anchor", "match", "participant", "raw_match", "history_listing", "history_listing_item")


def counts(path) -> dict[str, int]:
    con = sqlite3.connect(path)
    try:
        return {t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in TABLES}
    finally:
        con.close()


def version(path) -> int:
    con = sqlite3.connect(path)
    try:
        return con.execute("PRAGMA user_version").fetchone()[0]
    finally:
        con.close()


def objects(path) -> dict[str, str]:
    """Every table, index and view with comments and whitespace removed, so two builds compare by meaning."""
    con = sqlite3.connect(path)
    try:
        rows = con.execute("SELECT name, sql FROM sqlite_master WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%'")
        return {name: " ".join(re.sub(r"--[^\n]*", "", sql).split()) for name, sql in rows}
    finally:
        con.close()


@pytest.fixture
def old_db(tmp_path):
    """A populated database exactly as the previous release left it: schema v1, no prophecy table."""
    path = tmp_path / "old.sqlite3"
    random_world(path).close()
    con = sqlite3.connect(path)
    con.execute("DROP TABLE prophecy")
    con.execute("PRAGMA user_version = 1")
    con.close()
    return path


def test_an_old_database_gains_the_table_and_keeps_every_row(old_db):
    before = counts(old_db)
    assert before["match"] > 100 and version(old_db) == 1

    Mirror.open(old_db).close()

    assert version(old_db) == 2
    assert counts(old_db) == before
    assert "prophecy" in objects(old_db)


def test_opening_again_changes_nothing(old_db):
    Mirror.open(old_db).close()
    first = (version(old_db), objects(old_db), counts(old_db))
    Mirror.open(old_db).close()
    assert (version(old_db), objects(old_db), counts(old_db)) == first


def test_the_upgraded_schema_equals_a_fresh_install(old_db, tmp_path):
    Mirror.open(tmp_path / "fresh.sqlite3").close()
    Mirror.open(old_db).close()
    assert objects(old_db) == objects(tmp_path / "fresh.sqlite3")


def test_a_step_that_fails_leaves_the_database_at_its_old_version(old_db, monkeypatch):
    broken = Migration(2, ("CREATE TABLE prophecy (x INTEGER) STRICT", "THIS IS NOT SQL"))
    monkeypatch.setattr(migrations, "MIGRATIONS", (broken,))
    with pytest.raises(sqlite3.OperationalError):
        Mirror.open(old_db)
    assert version(old_db) == 1
    assert "prophecy" not in objects(old_db)


def test_a_step_another_process_already_applied_is_skipped(old_db):
    Mirror.open(old_db).close()
    con = sqlite3.connect(old_db, isolation_level=None)
    con.execute("INSERT INTO prophecy (platform_id, game_id, status, subject_puuid, subject_side, payload, "
                "claimed_at_ms) VALUES ('EUN1', 1, 'pending', 'p', 100, '{}', 0)")
    migrations.upgrade(con, 1, 2)      # what a second opener does after waiting for the write lock
    assert con.execute("SELECT count(*) FROM prophecy").fetchone()[0] == 1
    assert con.execute("PRAGMA user_version").fetchone()[0] == 2


def test_a_database_from_newer_code_is_refused(tmp_path):
    path = tmp_path / "future.sqlite3"
    Mirror.open(path).close()
    con = sqlite3.connect(path)
    con.execute("PRAGMA user_version = 99")
    con.close()
    with pytest.raises(RuntimeError, match="schema v99, newer than this code"):
        Mirror.open(path)


def test_read_only_connections_work_on_a_database_that_is_not_upgraded_yet(old_db):
    con = Mirror.read_only(old_db)
    try:
        assert con.execute("SELECT count(*) FROM match").fetchone()[0] > 100
    finally:
        con.close()
    assert version(old_db) == 1
