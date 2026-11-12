# training/: model training, outside the app

The app (`src/oracle`) crawls data and computes leak-free features. This folder trains models on an exported
**bundle** of those features. It doesn't import the app, so it runs anywhere pandas runs: locally or on Google Colab.

```
uv run oracle dataset            # app side: SQLite -> training/data/win_v1-<id>/ (no API calls, ~seconds)
        │
        ▼
training/
  trainlib.py                    # loading, time splits, metrics, calibration plot, save_model (no app imports)
  01_explore.ipynb               # what's in the data, missing values, role-inference accuracy, univariate signal
  02_win_baseline.ipynb          # constant prior -> logistic regression (+ CV, calibration, coefficients, explanations)
  03_win_boosting.ipynb          # random forest vs LightGBM, early stopping, Platt calibration, SHAP, ablation
  data/<bundle>/                 # git-ignored: matches.parquet, seats.parquet, manifest.json
  artifacts/<model>/             # git-ignored: model.joblib + model_card.json
```

## The bundle

| file | one row per | contents |
|---|---|---|
| `matches.parquet` | game | `y_blue_win`; `d_mean_*` / `d_min_*` / `d_max_*` (blue − red team aggregates), `lane_<ROLE>_*` (same-role matchups), `ctx_*` (symmetric context); meta (`as_of_ms`, `patch`, `depth`, `has_tracked`) |
| `seats.parquet` | player in a game | 32 per-player features, inferred `role`, `is_tracked`, and labels `y_kills`, `y_deaths`, `y_assists`, `y_damage`, `y_cs`, `y_vision`, … ready for the performance models |
| `manifest.json` | — | dataset id (feature code version + params hash), knowledge cut-off, patches, skip counts, column groups, and a definition of every feature |

Every feature uses only games that **ended before the predicted game started**. The tests in `tests/test_features.py`
check this, so the notebooks don't have to worry about leakage. Splits must still be **by time**: use `tl.time_split`.

## Run locally

```bash
uv run oracle dataset                                  # rebuild whenever the crawl has added games
uv run --group training jupyter lab training/
```

**macOS + LightGBM:** LightGBM needs the OpenMP runtime, so run `brew install libomp`. This needs the Apple-Silicon
Homebrew in `/opt/homebrew`; an Intel Homebrew in `/usr/local` installs an x86 build that LightGBM can't load.
Notebooks 01 and 02 don't need it.

## Run on Google Colab

1. Build a bundle locally: `uv run oracle dataset`.
2. In Google Drive, create `MyDrive/oracle-training/` and upload `trainlib.py`, the three notebooks and the `data/`
   folder into it.
3. Open a notebook from Drive with Colab and run all cells. The first cell mounts Drive, installs
   `lightgbm`/`shap` if needed, and loads the newest bundle in `data/`.

There's no need for a GPU. The data is a few thousand rows, and every model here trains in seconds on a CPU. Colab
is just a convenient place to run notebooks. Keep the Drive folder private: bundles contain PUUIDs (no names).

## Reading the results honestly

* Pre-game win prediction is hard. The constant baseline's log loss is about 0.693, and beating it by 0.005–0.02
  on validation is a real result. **Validation AUC above ~0.65 means: hunt for a leak.**
* With fewer than ~1,000 games, validation has very few rows, and differences of ±0.02 log loss are noise.
  Check `tl.bootstrap_ci`. On random data, the notebooks' "best model" still "wins" by luck.
* Keep the test block sealed (`EVALUATE_TEST = False`) until you're done choosing.
* Rank and mastery columns stay empty until `oracle crawl --forever` has captured them **at game start**. The
  notebooks drop all-empty columns automatically, so retrain once coverage grows.
