"""Turning model output into a prophecy seen from the subject's side."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from oracle.bot.champions import ChampionNames, readable
from oracle.bot.forecast import foresee
from oracle.bot.phrasebook import Family
from oracle.bot.predictor import Contribution, Prediction
from oracle.bot.prophecy import Foreseen
from oracle.features import Features, Pregame, Seat, Side

KEY = "EUN1_55"
NAMES = ChampionNames({10: "Kayle", 11: "MasterYi", 12: "TwistedFate", 13: "MonkeyKing"})


def pregame() -> Pregame:
    seats = tuple(Seat(f"p{i}", Side.BLUE if i < 5 else Side.RED, 10 + i % 4) for i in range(10))
    return Pregame(KEY, 1_000, 420, seats)


def features(complete: bool = True) -> Features:
    index = pd.MultiIndex.from_product([[KEY], range(10)], names=["game", "seat"])
    seats = pd.DataFrame({"role": ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"] * 2, "main_role": ["TOP"] * 10,
                          "n_games": 20.0, "streak": [-4.0] + [0.0] * 9, "champ_games": 0.0,
                          "off_role": [1.0, 0.0, np.nan] + [0.0] * 7}, index=index)
    coverage = pd.DataFrame({"complete": [complete] * 10}, index=index)
    return Features(seats, pd.DataFrame(), coverage)


# +2.0 log-odds for blue from form, -0.5 from KDA, +0.1 from the side advantage.
PREDICTION = Prediction(p_blue=1 / (1 + math.exp(-1.6)), intercept=0.1, contributions=(
    Contribution("d_mean_wr_shrunk", 1.0, 2.0), Contribution("d_mean_kda", -1.0, 0.5 * -1)))


def make(subject: str):
    return foresee(pregame(), subject, "Fox", features(), PREDICTION, NAMES, model_id="win_lr x y", prepared_in_s=5.0)


def test_a_blue_subject_keeps_the_signs_of_the_model():
    prophecy = make("p2")
    assert prophecy.subject_side is Side.BLUE
    assert prophecy.p_subject_win == pytest.approx(PREDICTION.p_blue)
    form = next(d for d in prophecy.outcome.drivers if d.column == "d_mean_wr_shrunk")
    kda = next(d for d in prophecy.outcome.drivers if d.column == "d_mean_kda")
    assert form.family is Family.FORM and form.label == "team average win rate"
    assert form.points > 0 > kda.points


def test_a_red_subject_gets_the_mirror_image():
    red = make("p7")
    assert red.subject_side is Side.RED
    assert red.p_subject_win == pytest.approx(1 - PREDICTION.p_blue)
    form = next(d for d in red.outcome.drivers if d.column == "d_mean_wr_shrunk")
    kda = next(d for d in red.outcome.drivers if d.column == "d_mean_kda")
    assert form.points < 0 < kda.points                  # what helps blue hurts red
    side = next(d for d in red.outcome.drivers if d.column == "intercept")
    assert side.points < 0 and side.family is Family.SIDE   # the blue-side advantage is a handicap for red


def test_driver_points_are_the_change_in_win_probability_without_that_driver():
    logit = 1.6
    form = next(d for d in make("p2").outcome.drivers if d.column == "d_mean_wr_shrunk")
    expected = 100 * (1 / (1 + math.exp(-logit)) - 1 / (1 + math.exp(-(logit - 2.0))))
    assert form.points == pytest.approx(expected)


def test_at_most_three_drivers_per_direction_strongest_first():
    many = Prediction(0.6, 0.0, tuple(Contribution(f"d_mean_kda{i}", 1.0, 0.1 * (i - 4)) for i in range(9)))
    drivers = foresee(pregame(), "p2", "Fox", features(), many, NAMES, model_id="m x y", prepared_in_s=1).outcome.drivers
    assert [d.column for d in drivers if d.points > 0] == ["d_mean_kda8", "d_mean_kda7", "d_mean_kda6"]
    assert [d.column for d in drivers if d.points < 0] == ["d_mean_kda0", "d_mean_kda1", "d_mean_kda2"]
    assert [abs(d.points) for d in drivers] == sorted((abs(d.points) for d in drivers), reverse=True)


def test_seat_views_carry_what_the_jokes_need():
    outcome = make("p2").outcome
    assert isinstance(outcome, Foreseen)
    top, subject = outcome.seats[0], outcome.seats[2]
    assert (top.role, top.champion, top.streak, top.off_role, top.champ_games) == ("TOP", "Kayle", -4, True, 0)
    assert subject.is_subject and subject.off_role is None   # no history to judge
    assert sum(s.is_subject for s in outcome.seats) == 1


def test_incomplete_coverage_shows_up_as_missing_histories():
    outcome = foresee(pregame(), "p2", "Fox", features(complete=False), PREDICTION, NAMES, model_id="m x y",
                      prepared_in_s=1).outcome
    assert outcome.histories_missing == 10 and not outcome.complete


def test_a_subject_outside_the_game_is_an_error():
    with pytest.raises(ValueError, match="not one of the ten players"):
        make("nobody")


def test_champion_names_are_readable_and_unknown_ids_get_a_neutral_label():
    assert [NAMES[i] for i in (11, 12, 13)] == ["Master Yi", "Twisted Fate", "Wukong"]
    assert NAMES[999] == "Champion #999"
    assert readable("JarvanIV") == "Jarvan IV" and readable("KSante") == "K'Sante"
