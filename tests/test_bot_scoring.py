"""`oracle prophecies`: log loss and Brier against the coin flip, computed by hand."""

from __future__ import annotations

import math

import pytest

from oracle.bot.scoring import report, score
from oracle.mirror import ProphecyCounts, ScoredProphecy


def row(p: float, blue_won: bool, complete: bool = True) -> ScoredProphecy:
    return ScoredProphecy("EUN1_1", p, blue_won, complete)


def test_log_loss_and_brier_of_two_games():
    result = score([row(0.7, True), row(0.4, False)])
    assert result.n == 2
    assert result.log_loss == pytest.approx(-(math.log(0.7) + math.log(0.6)) / 2)     # 0.43375
    assert result.brier == pytest.approx((0.3**2 + 0.4**2) / 2)                       # 0.125


def test_an_always_fifty_percent_oracle_scores_exactly_the_coin_flip():
    result = score([row(0.5, True), row(0.5, False)])
    assert result.log_loss == pytest.approx(math.log(2)) and result.brier == pytest.approx(0.25)


def test_a_certain_wrong_answer_is_finite():
    assert math.isfinite(score([row(0.0, True)]).log_loss)


def test_nothing_to_score_is_none():
    assert score([]) is None


def test_the_report_lists_counts_scores_and_the_baseline():
    text = report(ProphecyCounts(pending=1, posted=2, failed=1, with_number=4),
                  [row(0.7, True), row(0.4, False, complete=False)])
    assert text.splitlines() == [
        "prophecies  1 pending, 2 posted, 1 failed",
        "scored      2 finished games, 2 still waiting for the crawler to fetch them",
        "             log loss   brier",
        "oracle         0.4338  0.1250",
        "10/10 reads    0.3567  0.0900   n=1",
        "coin flip      0.6931  0.2500",
        "n is 2: differences under about 0.02 are noise below a few hundred games.",
    ]


def test_the_report_explains_an_empty_table():
    assert report(ProphecyCounts(0, 0, 0), []).splitlines()[1] == "scored      none yet (no prophecy has been made)"
    assert "1 waiting" in report(ProphecyCounts(1, 0, 0, with_number=1), []).splitlines()[1]
