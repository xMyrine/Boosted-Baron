"""`oracle prophesy`, `oracle prophecies` and `oracle bot` as the owner runs them, on a synthetic database."""

from __future__ import annotations

import asyncio
import hashlib
import sqlite3

import pytest

pytest.importorskip("sklearn")
pytest.importorskip("joblib")

from botkit import linear_model, save_model  # noqa: E402
from featureworld import random_world  # noqa: E402

from oracle.bot.app import run_bot  # noqa: E402
from oracle.cli import main  # noqa: E402
from oracle.config import ApiKey, Settings  # noqa: E402
from oracle.mirror import Mirror  # noqa: E402
from oracle.riot import KeyInUse, RiotGateway  # noqa: E402


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A working directory with a database full of finished games and a saved model. No .env above it."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    random_world(tmp_path / "data" / "oracle.sqlite3").close()
    save_model(tmp_path / "model", linear_model())
    for key in ("DISCORD_BOT_TOKEN", "DISCORD_CHANNEL_ID"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("RIOT_PLATFORM", "eun1")
    monkeypatch.setenv("RIOT_API_KEY", "RGAPI-test")
    monkeypatch.setenv("ORACLE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("ORACLE_MODEL_DIR", str(tmp_path / "model"))
    return tmp_path


def digest(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_prophesy_replays_the_newest_complete_game_and_shows_the_real_result(home, capsys):
    db = home / "data" / "oracle.sqlite3"
    before = digest(db)
    assert main(["prophesy"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("Prophecy for Summoner: ")
    assert "Real result: " in out and "The oracle said " in out
    assert "Blue side (roles inferred):" in out and "Red side (roles inferred):" in out
    assert "histories read" in out.splitlines()[-1]
    assert digest(db) == before                                              # a replay never writes


def test_prophesy_takes_a_match_id_and_a_seat_for_the_subject(home, capsys):
    con = sqlite3.connect(home / "data" / "oracle.sqlite3")
    game = con.execute("SELECT match_id FROM anchor ORDER BY seq LIMIT 1 OFFSET 20").fetchone()[0]
    con.close()
    assert main(["prophesy", game, "--seat", "8"]) == 0
    out = capsys.readouterr().out
    assert f"Replay of {game}" in out


def test_prophesy_with_an_unknown_game_or_seat_fails_with_a_message(home, capsys):
    assert main(["prophesy", "EUN1_1"]) == 1
    assert "EUN1_1 is not stored" in capsys.readouterr().err
    assert main(["prophesy", "--seat", "11"]) == 1
    assert "--seat must be between 1 and 10" in capsys.readouterr().err


def test_prophesy_send_without_discord_settings_fails_before_any_network_call(home, capsys):
    assert main(["prophesy", "--send"]) == 1
    assert "DISCORD_BOT_TOKEN is missing" in capsys.readouterr().err


def test_prophesy_with_a_model_from_other_feature_code_says_what_to_retrain(home, capsys):
    save_model(home / "model", linear_model(), card={"feature_code_version": 0})
    assert main(["prophesy"]) == 1
    assert "Rebuild the dataset and retrain" in capsys.readouterr().err


def test_prophecies_reports_the_stored_numbers_against_finished_games(home, capsys):
    con = sqlite3.connect(home / "data" / "oracle.sqlite3", isolation_level=None)
    con.execute("INSERT INTO raw_match VALUES ('EUN1_5555', 'ok', x'00', 0, 1)")
    con.execute("INSERT INTO match VALUES ('EUN1_5555', 'EUN1', 420, 1000, 1800, '16.1.1', '16.1', 100, 0)")
    con.close()
    with Mirror.open(home / "data" / "oracle.sqlite3") as mirror:
        mirror.claim_prophecy("EUN1", 5555, "p", 100, "{}")
        mirror.record_prophecy("EUN1", 5555, model_id="m", p_blue=0.75, complete=True, result_json="{}")
        mirror.finish_prophecy("EUN1", 5555, message_id="1")
    assert main(["prophecies"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "prophecies  0 pending, 1 posted, 0 failed"
    assert lines[1] == "scored      1 finished games, 0 still waiting for the crawler to fetch them"
    assert lines[3] == "oracle         0.2877  0.0625"                       # -ln(0.75), (1 - 0.75)^2


# ---------------------------------------------------------------- `oracle bot`

def bot_settings(home) -> Settings:
    return Settings(ApiKey("RGAPI-test"), "eun1", home / "bot-data", model_dir=home / "model")


def test_the_bot_stops_cleanly_on_ctrl_c_and_frees_the_key_lock(home):
    settings = bot_settings(home)

    async def scenario():
        task = asyncio.create_task(run_bot(settings, dry_run=True))
        await asyncio.sleep(0.5)                                             # crawl, service and watcher are running
        task.cancel()                                                        # what Ctrl-C does to the main task
        await asyncio.gather(task, return_exceptions=True)
        assert task.cancelled()
        async with RiotGateway.open(settings):                               # the lock is free again
            pass

    asyncio.run(scenario())
    con = sqlite3.connect(settings.db_path)
    run = con.execute("SELECT ended_ms IS NOT NULL, stop_reason FROM crawl_run").fetchone()
    assert run == (1, "stopped")                                             # the crawl recorded a clean end


def test_the_bot_cannot_start_next_to_another_crawler(home):
    settings = bot_settings(home)

    async def scenario():
        async with RiotGateway.open(settings):
            with pytest.raises(KeyInUse):
                await run_bot(settings, dry_run=True)

    asyncio.run(scenario())


def test_the_bot_refuses_a_model_that_does_not_match_before_touching_riot(home):
    save_model(home / "model", linear_model(), card={"feature_code_version": 7})
    from oracle.bot.errors import ModelError

    with pytest.raises(ModelError):
        asyncio.run(run_bot(bot_settings(home), dry_run=True))
    assert not (home / "bot-data" / "riot.lock").exists()                    # no gateway was opened
