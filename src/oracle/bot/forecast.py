"""One live game in, one `Prophecy` out: the feature, model and assembly steps shared by the service and the replay.

`compute` is blocking (SQLite, pandas, scikit-learn): call it through `asyncio.to_thread`. It opens its own
read-only connection, because a sqlite3 connection belongs to the thread that made it.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from oracle.bot.champions import ChampionNames
from oracle.bot.phrasebook import describe_column
from oracle.bot.predictor import Prediction, Predictor
from oracle.bot.prophecy import Driver, Foreseen, Prophecy, SeatView
from oracle.features import Features, FeatureParams, Pregame, Side, features_for_live
from oracle.mirror import Mirror

DRIVERS_PER_DIRECTION = 3


def compute(db_path: Path, pregame: Pregame, predictor: Predictor, params: FeatureParams) -> tuple[Features, Prediction]:
    con = Mirror.read_only(db_path)
    try:
        features = features_for_live(con, pregame, params)
    finally:
        con.close()
    return features, predictor.predict(features)


def foresee(
    pregame: Pregame,
    subject: str,
    subject_name: str,
    features: Features,
    prediction: Prediction,
    names: ChampionNames,
    *,
    model_id: str,
    prepared_in_s: float,
) -> Prophecy:
    """Assemble the numbers from the model into a `Prophecy` seen from the subject's side."""
    subject_seat = next((s for s in pregame.seats if s.puuid == subject), None)
    if subject_seat is None:
        raise ValueError(f"{pregame.key}: subject is not one of the ten players")
    sign = 1.0 if subject_seat.side is Side.BLUE else -1.0
    seats = tuple(_seat_view(pregame, i, features, names, subject) for i in range(10))
    outcome = Foreseen(prediction.p_blue, _drivers(prediction, sign), seats)
    return Prophecy(pregame.key, subject, subject_name, subject_seat.side, model_id, prepared_in_s, outcome)


def _drivers(prediction: Prediction, sign: float) -> tuple[Driver, ...]:
    """Contributions seen from the subject's side: a feature that helps blue hurts a red subject."""
    if not prediction.contributions:
        return ()
    parts = [("intercept", prediction.intercept), *((c.column, c.logit) for c in prediction.contributions)]
    total = sign * sum(logit for _, logit in parts)
    drivers = []
    for column, logit in parts:
        family, label = describe_column(column)
        drivers.append(Driver(column, family, label, 100.0 * (_sigmoid(total) - _sigmoid(total - sign * logit))))
    helps = sorted((d for d in drivers if d.points > 0), key=lambda d: -d.points)[:DRIVERS_PER_DIRECTION]
    hurts = sorted((d for d in drivers if d.points < 0), key=lambda d: d.points)[:DRIVERS_PER_DIRECTION]
    return tuple(sorted(helps + hurts, key=lambda d: -abs(d.points)))


def _seat_view(pregame: Pregame, i: int, features: Features, names: ChampionNames, subject: str) -> SeatView:
    seat = pregame.seats[i]
    row = features.seats.loc[(pregame.key, i)]
    complete = bool(features.coverage.loc[(pregame.key, i), "complete"])
    return SeatView(
        side=seat.side,
        role=_text(row["role"]),
        champion=names[seat.champion_id],
        is_subject=seat.puuid == subject,
        n_games=int(row["n_games"]),
        streak=int(row["streak"]),
        champ_games=int(row["champ_games"]),
        main_role=_text(row["main_role"]),
        off_role=None if pd.isna(row["off_role"]) else bool(row["off_role"]),
        history_complete=complete,
    )


def _text(value: object) -> str | None:
    return None if value is None or (isinstance(value, float) and math.isnan(value)) else str(value)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))
