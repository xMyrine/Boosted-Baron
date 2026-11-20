"""The prophecy table as a state machine: claim once, record, then post or fail, never back."""

from __future__ import annotations

import sqlite3

import pytest
from featureworld import random_world

from oracle.mirror import Mirror, ProphecyCounts


@pytest.fixture
def mirror(tmp_path):
    random_world(tmp_path / "w.sqlite3").close()
    with Mirror.open(tmp_path / "w.sqlite3") as m:
        yield m


def claim(mirror: Mirror, game_id: int = 7, platform: str = "EUW1"):
    return mirror.claim_prophecy(platform, game_id, "puuid-000", 100, '{"game": 1}')


def test_only_the_first_claim_of_a_game_wins(mirror):
    first = claim(mirror)
    assert (first.game_key, first.subject, first.side, first.result_json) == ("EUW1_7", "puuid-000", 100, None)
    assert claim(mirror) is None
    assert claim(mirror, platform="euw1") is None            # the platform id is normalised
    assert mirror.prophecy_counts() == ProphecyCounts(pending=1, posted=0, failed=0)


def test_a_posted_prophecy_is_final(mirror):
    claim(mirror)
    assert mirror.record_prophecy("EUW1", 7, model_id="m", p_blue=0.6, complete=True, result_json='{"a": 1}')
    assert mirror.finish_prophecy("EUW1", 7, message_id="123") is True
    assert mirror.finish_prophecy("EUW1", 7, message_id="456") is False
    assert mirror.fail_prophecy("EUW1", 7, error="late") is False
    assert mirror.record_prophecy("EUW1", 7, model_id="m", p_blue=0.1, complete=False, result_json="{}") is False
    assert mirror.pending_prophecies() == []
    assert mirror.prophecy_counts() == ProphecyCounts(0, 1, 0, with_number=1)


def test_a_failed_prophecy_is_final_and_keeps_its_error(mirror):
    claim(mirror)
    assert mirror.fail_prophecy("EUW1", 7, error="Discord answered 403") is True
    assert mirror.finish_prophecy("EUW1", 7, message_id="1") is False
    error = mirror._con.execute("SELECT error FROM prophecy").fetchone()[0]
    assert error == "Discord answered 403"


def test_pending_prophecies_come_back_with_their_stored_result(mirror):
    claim(mirror, 1)
    claim(mirror, 2)
    mirror.record_prophecy("EUW1", 2, model_id="m", p_blue=0.55, complete=False, result_json='{"p": 0.55}')
    first, second = mirror.pending_prophecies()
    assert (first.game_key, first.result_json, first.subject, first.side) == ("EUW1_1", None, "puuid-000", 100)
    assert (second.game_key, second.result_json) == ("EUW1_2", '{"p": 0.55}')
    assert first.payload == '{"game": 1}'


def test_the_table_rejects_states_the_machine_cannot_reach(mirror):
    claim(mirror)
    with pytest.raises(sqlite3.IntegrityError):               # posted needs a message id
        mirror._con.execute("UPDATE prophecy SET status = 'posted', finished_at_ms = 1")
    with pytest.raises(sqlite3.IntegrityError):               # a number without its completeness flag
        mirror._con.execute("UPDATE prophecy SET p_blue = 0.5")
    with pytest.raises(sqlite3.IntegrityError):               # pending must not look finished
        mirror._con.execute("UPDATE prophecy SET finished_at_ms = 1")


def test_scoring_joins_a_stored_number_with_the_finished_match(mirror):
    con = mirror._con
    con.execute("INSERT INTO raw_match VALUES ('EUW1_5555', 'ok', x'00', 0, 1)")
    con.execute("INSERT INTO match VALUES ('EUW1_5555', 'EUW1', 420, 1000, 1800, '16.1.1', '16.1', 200, 0)")
    claim(mirror, 5555)
    mirror.record_prophecy("EUW1", 5555, model_id="m", p_blue=0.7, complete=True, result_json="{}")
    claim(mirror, 999_999)                                    # a game we never stored: waits for its result
    mirror.record_prophecy("EUW1", 999_999, model_id="m", p_blue=0.4, complete=False, result_json="{}")

    [scored] = mirror.scored_prophecies()
    assert (scored.game_key, scored.p_blue, scored.blue_won, scored.complete) == ("EUW1_5555", 0.7, False, True)
