"""Loading, refusing and explaining a model. Skipped when the `bot` dependency group is not installed."""

from __future__ import annotations

import math

import pandas as pd
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("joblib")

from botkit import FEATURES, boosted_model, import_trainlib, linear_model, save_model, training_frame  # noqa: E402

from oracle.bot.predictor import ModelError, Predictor  # noqa: E402
from oracle.features import Features  # noqa: E402


def one_game(**values: float) -> Features:
    row = {f: 0.0 for f in FEATURES} | values
    return Features(pd.DataFrame(), pd.DataFrame([row], index=pd.Index(["EUN1_1"], name="game")), pd.DataFrame())


@pytest.fixture
def linear(tmp_path):
    return Predictor.load(save_model(tmp_path / "win_lr", linear_model()))


def test_probability_is_the_pipelines_probability(linear):
    x = pd.DataFrame([{f: 0.5 for f in FEATURES}])
    expected = linear.model.predict_proba(x)[0, 1]
    assert linear.predict(one_game(**{f: 0.5 for f in FEATURES})).p_blue == pytest.approx(expected)


def test_contributions_plus_intercept_equal_the_logit(linear):
    game = one_game(d_mean_kda=1.5, d_mean_wr_shrunk=-0.7, lane_TOP_kda=float("nan"))
    prediction = linear.predict(game)
    logit = float(linear.model.decision_function(game.matches[FEATURES])[0])
    assert prediction.intercept + sum(c.logit for c in prediction.contributions) == pytest.approx(logit, abs=1e-9)
    assert 1 / (1 + math.exp(-logit)) == pytest.approx(prediction.p_blue, abs=1e-9)


def test_the_signal_feature_is_the_strongest_driver_and_helps_blue_when_positive(linear):
    prediction = linear.predict(one_game(d_mean_kda=2.0))
    top = max(prediction.contributions, key=lambda c: abs(c.logit))
    assert top.column == "d_mean_kda" and top.logit > 0


def test_missing_indicator_columns_are_named_after_their_feature(linear):
    columns = [c.column for c in linear.predict(one_game()).contributions]
    assert columns == FEATURES + ["missingindicator_lane_TOP_kda"]


def test_a_missing_value_is_imputed_not_an_error(linear):
    prediction = linear.predict(one_game(lane_TOP_kda=float("nan")))
    indicator = next(c for c in prediction.contributions if c.column == "missingindicator_lane_TOP_kda")
    assert indicator.value > 0     # the indicator fires, in standardised units


def test_a_non_linear_model_still_predicts_but_has_no_drivers(tmp_path):
    predictor = Predictor.load(save_model(tmp_path / "win_gb", boosted_model()))
    prediction = predictor.predict(one_game(d_mean_kda=2.0))
    assert 0.0 < prediction.p_blue < 1.0
    assert prediction.contributions == ()


def test_a_model_from_other_feature_code_is_refused(tmp_path):
    directory = save_model(tmp_path / "old", linear_model(), card={"feature_code_version": 0})
    with pytest.raises(ModelError, match=r"feature code version: model 0, running 1"):
        Predictor.load(directory)


def test_a_model_trained_with_other_params_is_refused(tmp_path):
    import json

    from oracle.features import FeatureParams

    params = json.loads(FeatureParams().to_json()) | {"history_n": 10}
    directory = save_model(tmp_path / "short", linear_model(), card={"params": params})
    with pytest.raises(ModelError, match=r"param history_n: model 10, running 20"):
        Predictor.load(directory)


def test_a_missing_model_says_what_to_do(tmp_path):
    with pytest.raises(ModelError, match="Train one in training/"):
        Predictor.load(tmp_path / "nothing")


def test_a_model_that_needs_a_column_the_feature_code_dropped_is_refused(linear):
    full = one_game()
    game = Features(full.seats, full.matches.drop(columns="d_mean_kda"), full.coverage)
    with pytest.raises(ModelError, match="d_mean_kda"):
        linear.predict(game)


def test_sklearn_version_in_the_card_is_checked_but_only_warns(tmp_path):
    import sklearn
    assert Predictor.load(save_model(tmp_path / "ok", linear_model())).warnings == ()
    missing = Predictor.load(save_model(tmp_path / "none", linear_model(), card={"sklearn_version": None}))
    assert "no sklearn_version" in missing.warnings[0]
    older = Predictor.load(save_model(tmp_path / "old", linear_model(), card={"sklearn_version": "0.1.0"}))
    assert older.warnings == (f"model was saved with scikit-learn 0.1.0, running {sklearn.__version__}: "
                              "predictions may differ or loading may fail",)


def test_trainlib_records_the_sklearn_version_in_the_card(tmp_path):
    import json
    import sklearn
    tl = import_trainlib()
    x, y = training_frame()
    manifest = {"dataset_id": "win_v1@t", "feature_code_version": 1, "params": {}, "time_range": None, "patches": []}
    bundle = tl.Bundle(pd.DataFrame(), pd.DataFrame(), manifest, tmp_path)
    tl.save_model(tmp_path / "m", linear_model(), bundle, FEATURES, [])
    card = json.loads((tmp_path / "m" / "model_card.json").read_text())
    assert card["sklearn_version"] == sklearn.__version__
