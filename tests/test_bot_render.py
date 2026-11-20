"""Rendering a prophecy: the numbers, the sides, the jokes, and the storage format. No Discord, no scikit-learn."""

from __future__ import annotations

from dataclasses import replace

import pytest

from oracle.bot.phrasebook import FEATURE_LABELS, Family, describe_column
from oracle.bot.prophecy import Clouded, Driver, Foreseen, Prophecy, Reason, SeatView, render
from oracle.features import FEATURE_DOCS, Side

ROLES = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")
BLUE = ("Gnar", "Lee Sin", "Ahri", "Jinx", "Lulu")
RED = ("Renekton", "Viego", "Syndra", "Ezreal", "Nautilus")


def seat(side: Side, i: int, **kw) -> SeatView:
    base = dict(side=side, role=ROLES[i], champion=(BLUE if side is Side.BLUE else RED)[i], is_subject=False,
                n_games=20, streak=0, champ_games=6, main_role=ROLES[i], off_role=False, history_complete=True)
    return SeatView(**{**base, **kw})


def table(subject_side: Side = Side.BLUE, **by_seat: dict) -> tuple[SeatView, ...]:
    """Ten calm seats. The subject plays mid for `subject_side`; `by_seat` overrides e.g. blue_1={'streak': -5}."""
    seats = []
    for side in (Side.BLUE, Side.RED):
        for i in range(5):
            kw = dict(by_seat.get(f"{side.name.lower()}_{i}", {}))
            if side is subject_side and i == 2:
                kw["is_subject"] = True
            seats.append(seat(side, i, **kw))
    return tuple(seats)


def prophecy(p_blue: float = 0.62, side: Side = Side.BLUE, drivers: tuple[Driver, ...] = (), seats=None,
             key: str = "EUN1_100") -> Prophecy:
    seats = seats if seats is not None else table(side)
    return Prophecy(key, "puuid-fox", "Fox", side, "win_lr win_v1@abc 2026-11-05T12:30:25", 41.6,
                    Foreseen(p_blue, drivers, seats))


FORM_UP = Driver("d_mean_wr_shrunk", Family.FORM, "team average win rate", 2.4)
KDA_DOWN = Driver("d_mean_kda", Family.PERFORMANCE, "team average KDA", -1.3)


def test_blue_subject_sees_the_probability_of_blue():
    message = render(prophecy(0.62))
    assert message.title == "Prophecy for Fox: Mid Ahri"
    assert "**62%**" in message.lines[0] and "Fox" in message.lines[0]
    assert message.footer == "win_lr · 10/10 histories read · 42s · EUN1_100"


def test_red_subject_sees_the_flipped_probability():
    message = render(prophecy(0.30, Side.RED))
    assert "**70%**" in message.lines[0]
    assert "30%" not in message.lines[0]
    assert render(prophecy(0.30, Side.BLUE)).lines[0].count("**30%**") == 1


def test_bands_pick_different_colours():
    assert render(prophecy(0.70)).colour == 0x2ECC71        # dominant
    assert render(prophecy(0.50)).colour == 0x95A5A6        # coin flip
    assert render(prophecy(0.43)).colour == 0xE67E22        # underdog
    assert render(prophecy(0.30)).colour == 0xE74C3C        # doomed


def test_strongest_driver_and_its_opposition_are_named_with_their_points():
    message = render(prophecy(0.58, drivers=(FORM_UP, KDA_DOWN)))
    assert "the team average win rate adds 2.4 points for Fox" in message.lines[1]
    assert "the team average KDA costs 1.3 points" in message.lines[2]


def test_a_driver_that_hurts_leads_with_the_hurting_wording():
    message = render(prophecy(0.45, drivers=(Driver("d_mean_kda", Family.PERFORMANCE, "team average KDA", -3.1),)))
    assert "the team average KDA costs Fox 3.1 points" in message.lines[1]


def test_faint_drivers_admit_the_oracle_is_guessing():
    message = render(prophecy(0.51, drivers=(Driver("intercept", Family.SIDE, "the side of the map", 0.2),)))
    assert message.lines[1] == "No single signal moves the odds by more than 0.5 points. The oracle is mostly guessing."


def test_a_model_without_drivers_renders_without_driver_lines():
    message = render(prophecy(0.58, drivers=()))
    assert len(message.lines) == 1


def test_players_are_named_by_role_and_champion_never_by_riot_name():
    seats = table(blue_1={"streak": -5}, red_3={"off_role": True, "main_role": "UTILITY"},
                  blue_2={"champ_games": 0})
    message = render(prophecy(0.5, seats=seats))
    body = "\n".join(message.lines)
    assert "Fox's jungler on Lee Sin has lost 5 games in a row" in body
    assert "Fox on Ahri has not played this champion in the last 20 games" in body
    assert "puuid" not in body and "puuid" not in message.as_text()


def test_at_most_two_jokes_strongest_first_and_the_subject_counts_extra():
    seats = table(blue_2={"streak": -6}, red_0={"streak": 7}, red_1={"off_role": True, "main_role": "TOP"},
                  red_3={"champ_games": 0})
    lines = render(prophecy(0.5, seats=seats)).lines[1:]
    assert len(lines) == 2
    assert lines[0].startswith("Fox on Ahri walks in on a 6-game losing streak")
    assert lines[1].startswith("The enemy top laner on Renekton has won 7 in a row")


def test_enemy_streak_is_described_as_the_enemy_role():
    message = render(prophecy(0.5, seats=table(red_0={"streak": 6})))
    assert "The enemy top laner on Renekton has won 6 in a row" in "\n".join(message.lines)


def test_a_red_subject_has_red_allies_and_blue_enemies():
    seats = table(Side.RED, red_1={"streak": -3}, blue_1={"streak": -3})
    body = "\n".join(render(prophecy(0.5, Side.RED, seats=seats)).lines)
    assert "Fox's jungler on Viego" in body            # the red jungler is an ally of a red subject
    assert "Lee Sin" not in body                       # one streak kind per line, the ally is the stronger fact


def test_incomplete_histories_are_flagged_in_text_footer_and_colour():
    seats = tuple(replace(s, history_complete=i >= 3) for i, s in enumerate(table()))
    message = render(prophecy(0.62, seats=seats))
    assert "only 7 of 10 histories" in "\n".join(message.lines)
    assert "7/10 histories read" in message.footer
    assert message.colour == 0xF1C40F


def test_clouded_messages_carry_no_number():
    refused = Prophecy("EUN1_7", "puuid-fox", "Fox", Side.BLUE, "win_lr x y", 3.0, Clouded(Reason.RIOT_REFUSED))
    message = render(refused)
    assert message.title == "The vision is clouded for Fox"
    assert "RIOT_API_KEY" in message.lines[0] and "%" not in message.lines[0]
    assert refused.p_subject_win is None


def test_the_same_game_always_reads_the_same_and_games_differ():
    first = [render(prophecy(0.62, key=f"EUN1_{n}")).lines[0] for n in range(40)]
    assert first == [render(prophecy(0.62, key=f"EUN1_{n}")).lines[0] for n in range(40)]
    assert len(set(first)) == 2    # both wordings of the band appear across games


def test_as_text_drops_bold_markers_and_lists_both_sides():
    text = render(prophecy(0.62)).as_text()
    assert "**" not in text
    assert "Blue side (roles inferred):\n  Top Gnar\n  Jungle Lee Sin\n  Mid Ahri" in text
    assert "Red side (roles inferred):\n  Top Renekton" in text


def test_subject_is_bold_in_the_side_field():
    blue = render(prophecy(0.62)).fields[0]
    assert "Mid **Ahri**" in blue.value


@pytest.mark.parametrize("outcome", ["foreseen", "clouded"])
def test_stored_json_round_trips(outcome):
    original = prophecy(0.62, drivers=(FORM_UP, KDA_DOWN)) if outcome == "foreseen" else Prophecy(
        "EUN1_7", "puuid-fox", "Fox", Side.RED, "win_lr x y", 3.0, Clouded(Reason.PREDICTION_FAILED))
    assert Prophecy.from_json(original.to_json()) == original


def test_every_seat_feature_has_a_label_so_new_features_cannot_slip_in_silently():
    assert set(FEATURE_DOCS) == set(FEATURE_LABELS)


def test_columns_are_described_by_scope_and_missing_indicators_read_as_missing_data():
    assert describe_column("d_mean_wr_shrunk") == (Family.FORM, "team average win rate")
    assert describe_column("d_min_deaths_pm") == (Family.PERFORMANCE, "lowest deaths per minute on each side")
    assert describe_column("lane_MIDDLE_kda") == (Family.PERFORMANCE, "mid lane matchup (KDA)")
    assert describe_column("ctx_rank_score") == (Family.RANK, "all-player average rank")
    assert describe_column("missingindicator_lane_JUNGLE_off_role") == (
        Family.ROLE, "missing data for the jungle lane matchup (off-role rate)")
    assert describe_column("intercept") == (Family.SIDE, "the side of the map")
    assert describe_column("d_mean_brand_new_thing") == (Family.OTHER, "d_mean_brand_new_thing")
