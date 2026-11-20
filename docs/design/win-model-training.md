# Win model: data preparation and training setup

Status: **implemented** (`src/oracle/features/`, `oracle dataset`, `training/`). It builds on
`feature-builder.md` and records where it deviates from that proposal.

## Problem

The goal is to predict P(blue wins) at game start, and later each player's K/D/A, damage and so on, with the numbers
eventually fed to a small LLM for a Discord post. That requires:

1. **Leak-free features** for every training game, computed by the **same code** the bot will run on one live game.
2. A **convenient training loop outside the app**, in notebooks that also run on Google Colab.
3. Output shaped so the **performance models** and the **LLM prompt** can reuse it without new data work.

## Usage (caller's view)

```bash
uv run oracle dataset                       # -> training/data/win_v1-<id>/{matches,seats}.parquet + manifest.json
uv run --group training jupyter lab training/   # or upload training/ to Drive and open in Colab
```

```python
import trainlib as tl                       # standalone: pandas + sklearn only
b = tl.load("data")                         # newest bundle
split = tl.time_split(b.matches, val=0.15, test=0.15, embargo_h=1)
X = b.feature_group("d_mean_", "lane_")    # or b.features for all 102
model.fit(split.train[X], split.train["y_blue_win"])
tl.compare([tl.evaluate(split.val["y_blue_win"], model.predict_proba(split.val[X])[:, 1], "LR")])
tl.save_model("artifacts/win_lr", model, b, X, metrics)   # model.joblib + model_card.json (dataset id, params)
```

```python
# The bot, later: one game, same code path as training.
from oracle.features import Pregame, features_for_live
f = features_for_live(ro_con, Pregame.from_spectator(payload, now_ms=now))
p_blue = model.predict_proba(f.matches[card["features"]])[0, 1]
# f.seats has every player's values (streak, champ_games, off_role, ...) -> LLM prompt / jokes
```

## Shape

### Load-bearing decisions

1. **Two halves joined by a file.** `src/oracle/features` turns SQLite into a portable parquet **bundle**.
   `training/` consumes only the bundle. Colab needs no repo, no SQLite and no crawler. All leakage logic lives in
   the app half, where it's tested; the notebooks can't reintroduce it, apart from splitting badly, which
   `tl.time_split` handles.
2. **One input shape.** A seat table (`game, seat, side, puuid, champion_id, as_of_ms, snapshot_until_ms`) goes
   through `build_features`. Training builds it from anchors, and the bot builds it via `Pregame.from_spectator`.
   Outcomes and true roles exist only as `y_*` / `true_role` label columns, attached after features are built.
3. **Seat features first, team rows derived.** Every feature is computed per player (32 columns). The win model's
   row is derived from those: blue − red team mean, min and max, same-role lane differences, and symmetric context.
   So the performance model (one row per player) and the LLM prompt use exactly the same per-player numbers.
4. **Features have no fit.** Population statistics (champion meta win rate, champion role shares) are recomputed
   for each row from games that ended in `[T − window, T)`, using a sorted-array `AsOfCounter`. Anything learned
   from training rows (imputation, scaling, the model) lives in the sklearn pipeline, fitted on train only.
5. **Roles are inferred, never read.** The spectator has no roles. `roles.py` scores each player × role from their
   history and the champion's as-of role shares, then picks the best one-role-per-player assignment per team (all
   120 permutations, vectorised). The true role is only used to measure accuracy: 80% on the first 36 real games.

### Module map

```
src/oracle/features/
  pregame.py   Pregame / Seat / Side, from_spectator, to_frame (the bot boundary)
  _sources.py  private: ALL SQL. anchors (only label source), history (end < T, knowledge cut-off), coverage,
               champion events + AsOfCounter, snapshots (latest before snapshot_until_ms), LeakageError
  roles.py     ChampionMeta (as-of champion win rate / role shares), infer_roles
  seat.py      THE feature catalogue: families _form/_performance/_champion/_role/_snapshots + FEATURE_DOCS
  team.py      seat features -> match row (d_mean/d_min/d_max/lane/ctx)
  params.py    FeatureParams (kappas, windows, snapshot policy) + FEATURE_CODE_VERSION -> dataset id
  dataset.py   build_features (the one code path), features_for_live, Dataset.build/save/load
training/
  trainlib.py  load, time_split (+embargo), rolling_origin CV, evaluate, bootstrap_ci by week,
               reliability_plot, save_model (+model card)
  01_explore / 02_win_baseline / 03_win_boosting .ipynb
```

### Features (32 per seat; definitions are in `seat.FEATURE_DOCS` and every manifest)

| Family | Features |
|---|---|
| Form | `n_games`, `wr_shrunk` (κ = 100), `wr_recent` (last 5), `streak` (±10), `session_games`, `session_losses` (12 h), `hours_since_last` |
| Performance (pooled per minute) | `kda`, `kills_pm`, `deaths_pm`, `assists_pm`, `deaths_max`, `dmg_pm`, `dmg_taken_pm`, `gold_pm`, `cs_pm`, `vision_pm`, `kp`, `dmg_share` |
| Champion | `champ_games`, `champ_wr_shrunk` (toward the player's own rate), `days_since_champ`, `champ_meta_wr` (30 d, κ = 300), `champ_meta_games` |
| Role | `role_share`, `off_role` (inferred role ≠ main role), `role_conf` |
| Snapshots | `rank_score`, `ranked_games`, `ranked_wr_shrunk`, `mastery_points_log`, `mastery_level` |

**Rank and mastery: honest status.** Riot only returns *current* rank and mastery, so a value is clean for a game
only if it was captured before that game ended. Right now 0% of training rows have one (18 snapshots exist, all
taken after the games). The features exist with the strict rule. They're NaN until the live watcher
(`oracle crawl --forever`) has captured enough games at game start, and the notebooks drop all-empty columns
automatically. `FeatureParams.snapshot_lag_days > 0` deliberately accepts later snapshots, as a way to *measure*
the leak in a notebook. A bot model must never be trained with it.

## Synthesis decision

Two whole-shape candidates were compared. The multi-model arena tooling isn't installed in this environment, so
both were sketched in-session.

* **A: package + bundle (chosen).** Features live in the app package, with one code path for training and the bot.
  A parquet bundle is the hand-off, and the notebooks are standalone.
* **B: self-contained notebooks.** The notebooks query SQLite with pandas and build features themselves, to be
  ported to the package later. It's quicker to start and everything is visible in one place. But it creates a
  second feature implementation the bot would need, which is exactly the train/serve skew the project rules warn
  about. It also needs the SQLite file on Colab, and leakage tests would have to target notebooks.
* **C: Colab clones the repo and builds from an uploaded SQLite.** One code path like A, but Colab would need repo
  access (the repo is private) and the whole DB uploaded. A gets the same parity with a 1 MB upload.

Screened against the red flags: in A, a contributor editing a notebook can't introduce leakage, and adding a
feature means one family function plus one `FEATURE_DOCS` line (an undocumented feature fails `compute`). The
public surface is `Dataset.build/save/load`, `build_features`, `features_for_live` and `Pregame`.

### Deviations from `feature-builder.md`

| Proposal | Now | Why |
|---|---|---|
| `@seat_feature` registry with declared sources, derived prefetch | Plain family functions in `seat.py` | The bot doesn't exist yet. Prefetch is the only consumer of declared sources, so add it with the bot (subtract first) |
| Golden-vector feature-set id | `FEATURE_CODE_VERSION` (manual bump) + params hash | Much simpler. The model card stores both, and the risk is forgetting a bump: covered by review |
| 8 MVP features, diff-of-means only | 32 seat features → 102 match columns (mean/min/max/lane/ctx) | The owner asked for "the more the merrier". LR uses a subset or regularisation; LightGBM handles all |
| No role (v2) | Inferred role, `off_role`, lane matchups | The owner asked for it. Inference is identical in both paths, so no skew |
| Rank/mastery in v2 | Present now, strict-clean, mostly NaN | Columns are ready. Coverage grows with `--forever` |
| sklearn imported by `dataset.py` | The app needs only pandas + pyarrow; sklearn/LightGBM/SHAP are in the `training` group | The bot will need sklearn to unpickle a model: add it to runtime then |

## Tradeoffs accepted

* **Champion stats are recomputed for every row** rather than fitted once. That costs more CPU per build (still under
  2 s now), and in exchange there's no fit discipline at call sites and no stale priors.
* **Strict 10/10 completeness.** A game is dropped unless all 10 histories are listed and fetched. Today that keeps
  36 of 60 fetched anchors, and the share grows as the crawl finishes.
* **Many correlated columns.** Fine for boosting, and LR needs regularisation (C tuned by forward-chaining CV).
  Notebook 03's ablation shows which groups earn their place.

## Verification

* `tests/test_features.py` (12 tests on a random synthetic world): future invariance (scramble and delete everything
  not finished at T → features bit-identical), a listed game ending after T is dropped, knowledge cut-off,
  clean vs leaky snapshots, batch invariance, spectator-payload parity, side-swap antisymmetry, worked examples
  (streak, shrinkage, pooled rates, missing mastery = 0 vs no snapshot = NaN).
* All three notebooks were executed end-to-end on a 300-game synthetic bundle and on the real 36-game bundle.
  LightGBM couldn't load on this Mac (no `libomp`), so notebook 03 was run with a test stand-in for LightGBM;
  the LightGBM cells themselves are unverified locally.

## Next

1. Keep crawling. Rebuild with `oracle dataset` and re-run notebooks 02 and 03 at ~1,000 and ~5,000 games.
2. Run `oracle crawl --forever` while playing, so rank and mastery get clean coverage.
3. Performance models on `seats.parquet`: Poisson / negative-binomial for K/D/A, quantile LightGBM for damage/CS.
4. LLM step: the prompt input is the prediction, the top SHAP/LR contributions and the per-player rows from
   `Features.seats`.
