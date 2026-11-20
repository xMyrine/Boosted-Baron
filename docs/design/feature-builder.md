# Feature builder: design (proposal, not implemented)

Status: **superseded in part by `win-model-training.md` (implemented)**, which lists the deviations. Kept for the reasoning behind the leakage rules.
Arena sketches (two independent designs plus a judge) are kept outside the repo. This document is the synthesis.

## Problem

The mirror stores leak-free *raw* facts. What's missing is the step that turns them into model-ready rows. Two consumers need those rows, and they must get **identical** features:

| | Training / evaluation (notebook) | The bot (later) |
|---|---|---|
| Games | ~5,000 finished anchors, in batch | 1 live game, at game start |
| The 10 seats come from | `match`/`participant`, which also hold the **outcome** and the true **role** | spectator-v5: puuid, team, champion. **No role, no outcome, no patch** |
| History | already crawled | fetched in seconds, under a tight rate budget |
| Label | `winning_team_id` | unknown |

Four failure modes typical of this kind of project shape the design:

1. **Train/serve skew.** Two code paths drift apart, or training uses a field the bot never has. Role is the trap: `participant.team_position` sits right there in training data.
2. **Leakage through fitted statistics.** A champion's win rate, or a shrinkage prior, computed over *all* matches contains the future and the row's own label.
3. **Unreproducible datasets.** The mirror keeps growing and backfills the past, so "the same query next week" returns different rows.
4. **Correlated samples.** The same players appear across many anchors, so naive evaluation looks better than reality.

## Usage (caller's view)

**Notebook: train and evaluate the baseline**
```python
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from oracle.features import WIN_MVP, Dataset, save_model
from oracle.mirror import Mirror

con = Mirror.read_only(settings.db_path)
ds = Dataset.build(con, WIN_MVP)            # complete anchors only, sorted by start, data cut-off pinned to "now"
print(ds.version)                           # win_mvp@3f9a1c20e1 | known<=2026-10-21 09:12Z | 4,812 rows | patches 25.12..25.20
split = ds.time_split(val=0.15, test=0.15, embargo_h=1)      # train < val < test by start time; test stays sealed

pipe = make_pipeline(SimpleImputer(add_indicator=True), StandardScaler(), LogisticRegression())
pipe.fit(split.train.X, split.train.y)      # every LEARNED parameter is fit here, on train only
p = pipe.predict_proba(split.val.X)[:, 1]
print(log_loss(split.val.y, p), "vs constant", log_loss(split.val.y, [split.train.y.mean()] * len(p)))
save_model("data/models/win_lr.joblib", pipe, ds.card(split, metrics=...))
```

**Bot: one game, the same code path**
```python
pipe, card = load_model(path, WIN_MVP)                       # refuses if WIN_MVP's id changed since training
pregame = Pregame.from_live(live_game)                       # boundary: validated, no role / outcome / patch
await WIN_MVP.prefetch(mirror, riot.interactive(deadline_s=8), pregame)   # exactly what WIN_MVP needs
feats = WIN_MVP.build(ro_con, [pregame])                     # 1 row, same columns as training
if feats.complete()[pregame.key]:
    p_blue = pipe.predict_proba(feats.X)[0, 1]               # feats.seats holds per-player values for the jokes
```

**Adding a feature: one function in `catalog.py`**
```python
@seat_feature(needs={Source.HISTORY}, team_agg="max")
def worst_deaths(ctx: SeatContext) -> pd.Series:
    """Most deaths in one of the last n games. Why: a 'feeder' signal (and joke material)."""
    return ctx.per_seat(ctx.history.groupby(SEAT)["deaths"].max(), fill=np.nan)

WIN_V2 = WIN_MVP.extend("win_v2", worst_deaths)   # columns, sources, bot prefetch and id are all derived
```

## Shape

### Load-bearing decisions

1. **One input type, `Pregame`: what's knowable when the game starts.** It holds `key`, `as_of_ms` (T), `queue_id` and 10 × `Seat(puuid, side, champion_id)`. It has **no outcome, no role, no patch**, so no feature can read them. Training turns each anchor into a `Pregame` (the label goes only into `Dataset.y`), and the bot builds one with `Pregame.from_live`. Both call `FeatureSet.build(con, pregames)`. This is encode-lessons-in-structure: role isn't "not used", it's *impossible to use*.
2. **Features have no `fit`, and models have no SQL.** A past game may be used for a row at T only if it **ended before T** (`start + duration < T`) and was fetched by the cut-off K.
   - That includes population statistics. A champion's meta win rate is recomputed for each row from games that ended in `[T − 30 d, T)`. This is CatBoost's "ordered target statistics" with time as the order, or, in Java terms, event sourcing (replay the log up to T).
   - Everything learned *from training rows* (imputation, scaling, encoders, embeddings, the model itself) lives in the sklearn `Pipeline`, fit on the train split.
3. **A feature is one vectorised function, registered in `catalog.py`** (owner's choice). It receives a `SeatContext` holding DataFrames for all seats in the build: `seats` (index `SEAT = (game, seat)`), `history` (one row per usable prior game) and `champ_window`. It returns a float Series on exactly `seats.index`, which the engine checks, so pandas can't silently align it. The helpers `per_seat`, `hist_count` and `hist_sum` cover the common idioms, and every feature ships with worked-example tests.
   - The decorator declares which **sources** it reads and how a team aggregates it.
   - Everything else is derived: column names, which sources get loaded, what the bot prefetches, and the feature-set id.
   - Reading an undeclared source raises an error, because the bot wouldn't have prefetched it.
4. **Data access is vectorised and private.** `_sources.py` runs one SQL query per source for all seats at once (thousands in training, 10 in the bot). It counts as-of champion windows with one sort plus `searchsorted`, and it also returns **coverage**.

### Data structures

| Type | What | Invariants (where) |
|---|---|---|
| `Pregame(key, as_of_ms, queue_id, seats)` | A game at its start | 10 seats, 5 v 5, unique puuids, blue first, ordered by seat within a side (`__post_init__`) |
| `Seat(puuid, side, champion_id)` | One slot | `side` is a `Side` enum (BLUE = 100, RED = 200) |
| `SeatContext` | What a feature sees: `seats`, `history` (columns `game, seat, games_ago, prior_end_ms, minutes, champion_id, same_champion, team_position` (of *that* past game: legitimate), `win, kills, deaths, assists, damage_to_champs, gold_earned, cs, vision_score`), `champ_window` (`games, wins`), `params` | Each source is reachable only if declared (`UndeclaredSource`). History rows are checked at load: `prior_end_ms < T`, prior ≠ own game, no duplicates (`LeakageError`) |
| `SeatFeature(name, fn, needs, team_agg)` | One feature | Unique names in a set. A docstring is required (definition, why, leakage) |
| `FeatureSet(name, features, params)` | A versioned selection | `columns`, `sources` and `id` are derived properties |
| `Features(X, seats, coverage)` | Output of `build`, the same type for the bot and training | `X.index` = input keys in input order, float64 columns = `fs.columns` |
| `Dataset(X, y, meta, version)` | Training rows | Sorted by T, one row per match, `y` is the only label |
| `DatasetVersion` / `ModelCard` | id, K, `history_n`, parser version, patch range, digest of keys, skip counts by reason, git sha | `load_model` refuses a mismatched feature-set id |

### Flow

```
anchors SQL (the ONLY query reading winning_team_id) ─┬─► y, meta
                                                      └─► [Pregame] ─┐
spectator ─► Pregame.from_live ─► prefetch (bot only) ───────────────┤
                                                                     ▼
                         FeatureSet.build(con, pregames, knowledge_ms=K)
   _sources.load(needed sources) ── HISTORY: 1 query (+ coverage) ── CHAMP_WINDOW: 1 query + as-of index
   for each feature: fn(ctx restricted to its needs) → Series on seats.index   (checked: exact index, float, no ±inf)
   team aggregate (f.team_agg) → blue − red → X  (MATCH grain)
                                                                     ▼
                  Features(X, seats, coverage) ─► Dataset (complete rows + y + version)  |  bot: predict
```

**Grain.**
- `MATCH` (the MVP): `<feature>_diff` = blue team aggregate − red team aggregate. The model is then antisymmetric, and the **logistic-regression intercept is the blue-side advantage** (no side-swap augmentation, which would erase that real effect).
- `FOCAL_SEAT` (later, for performance models): one row per (game, focal player). It reuses the same seat features.

**Joke attribution comes free.** With team-mean columns, logistic regression's logit splits *exactly* per player (`coef/scale × value/5`). So "your jungler is the problem" works in v1 without SHAP. The per-seat values come from `Features.seats`.

### Module map

```
src/oracle/features/
  __init__.py     public: Pregame, Seat, Side, FeatureSet, Features, Dataset, Split, WIN_MVP, save_model, load_model
  pregame.py      Pregame, Seat, Side, from_live (the one adapter from the bot's LiveGame)
  engine.py       SeatContext, seat_feature, SeatFeature, FeatureParams, FeatureSet.build/prefetch, Features
  catalog.py      THE place features live: MVP functions + named sets (WIN_MVP)
  _sources.py     private: Source enum, SQL, loaders, as-of champion index, coverage, prefetch per source, anchor rows
  dataset.py      Dataset.build, time_split (+embargo), rolling_origin CV, DatasetVersion
  artifact.py     ModelCard, save_model, load_model (the parity guard at bot start-up)
tests/test_features.py
```

Tracing a game crosses three files: `engine.py` → `_sources.py` → `catalog.py`.

### MVP feature set `win_mvp` (8 seat features → 8 `_diff` columns, plus the intercept)

"History" means the seat's listed games (`games_ago < history_n`, **history_n = 20**) that are fetched, not remakes, and **ended before T**. A feature may use a shorter **window** of its own via `@seat_feature(window=10)` (e.g. form over the last 10, champion experience over all 20). The window is part of the feature-set id.

| Feature | Team agg | Definition | Why it should predict | Leakage notes |
|---|---|---|---|---|
| `form_wr` | mean | `(W + κ/2)/(N + κ)` | Results matchmaking hasn't absorbed yet (climbing or falling) | **Expect weak signal**: even 20 games is mostly noise, and matchmaking pushes everyone toward 50 % |
| `streak` | mean | Signed run from the most recent game, capped at ±5 | Momentum or tilt; great joke material | — |
| `session_games` | mean | History games that ended in `[T − 12 h, T)` | Fatigue or tilt sessions (the "today" in CLAUDE.md) | Uses prior games' end times only |
| `n_games` | min | Usable history games (0..n) | New, returning or smurf accounts. `min` = the least-known player | Also a data-quality signal |
| `champ_exp` | min | History games on the champion picked now | A first-time pick; `min` = the team's first-timer | The pick is known at game start in both paths |
| `kda` | mean | `ln((ΣK + ΣA + 1)/(ΣD + 1))`, pooled. None if there's no history | Skill beyond W/L | Role effects mostly cancel in team means (one of each role per side) |
| `cs_per_min` | mean | `ΣCS / Σminutes`, pooled | A mechanics proxy | Same role caveat |
| `champ_meta_wr` | mean | `(Wc + κc/2)/(Nc + κc)` over mirror games of that champion that **ended in `[T − 30 d, T)`** | Champion strength in the current meta; the rolling window tracks patches | **A population statistic done as-of.** Games still running at T are excluded. No global fit |

**Shrinkage.** Beta-Binomial shrinkage means adding κ pseudo-games at the prior mean. For a prior mean of 0.5 and a between-groups SD σ, `κ ≈ 0.25/σ² − 1`.
- Champion win rates spread by about 2.5–3 %, so **κc ≈ 300**.
- Player win rates spread by about 4–5 %, so **κ ≈ 100** for form. With a 10-game window, form then barely moves off 50 %, which is the honest answer.

The κ values are hyperparameters in `FeatureParams` (part of the id). An empirical-Bayes notebook on the training period can refine them later. Note: when every seat has the same N, shrinkage is an affine map that logistic regression undoes anyway. It only matters where N varies (new accounts, rare champions).

**Deliberately not in the MVP:**
- **Role.** The spectator doesn't provide it. Later it can be inferred from history positions and Smite, *identically* in both paths.
- **Rank and mastery (planned for v2).** They're post-game or missing for almost every backfilled row, and imputing defaults would teach the model a training/serving gap (live, rank is always present). Clean data is being collected from now on: the live watcher in `oracle crawl --forever` snapshots all 10 players at game start. A snapshot is clean for a match iff `taken_at_ms < game end`. v2 adds `Source.RANK` / `Source.MASTERY` using the latest clean snapshot (missing → imputer + indicator) and trains on the period with clean coverage. A rank-like as-of rating (Elo/Glicko/TrueSkill) covers every row in the meantime.
- **Patch as a feature.** The spectator doesn't provide it, and the bot always plays on an unseen patch.
- **Champion one-hots or embeddings.** These are model-side encoders.
- **EWMA form and Elo/Glicko.** They come later as as-of sources, using the same rule.

**Expectation.** Pre-game win prediction is hard. Beating the constant-prior log loss (≈ 0.693) by a few thousandths on a time split is a real result. A validation AUC above ~0.65 means hunting for a leak before celebrating.

### Coverage and completeness (both paths)

The HISTORY query uses LEFT JOINs, so for each game it also returns:
- `seats_listed`: seats with a listing at `(puuid, 420, T)` and `n_requested ≥ history_n`;
- `history_pending`: listed but not fetched;
- `history_used`;
- `history_future`: listed but ended at or after T. **This should be 0.** If it isn't, it measures how Riot's `endTime` behaves, which answers an open question from the import design with data.

`complete()` means all 10 seats are listed and nothing is pending. Training keeps only complete rows and counts the rest by reason. The bot uses the same test to decide whether to prophesy.

### Reproducibility

- **Data cut-off.** `Dataset.build(con, fs, knowledge_ms=K)` sees only the anchors, listings and matches fetched by K, so a crawl running in the background doesn't change the dataset. `DatasetVersion` records K, the feature-set id, `history_n` (read from the last `crawl_run.spec_json`, not hard-coded), the parser version range, the patch range, a digest of the keys and skip counts.
- **Known limit.** If `history_n` grows, listings are re-fetched with a newer `fetched_at_ms`, so an old K can no longer see them. Old cuts are for rebuilding recent experiments, not for archives.
- **Feature-set id.** `name@sha256(params, [(name, team_agg, needs)], golden vectors)`. A **golden-vector** test computes stored rows from a fixed synthetic world. Changing a feature's logic changes the golden output, and therefore the id. Editing a docstring doesn't.

### Evaluation helpers

- `time_split(val, test, embargo_h=1)` splits strictly forward in time, one row per match, with a gap so no train game is still running when validation starts.
- `rolling_origin(n_folds)` provides CV folds for sklearn's `cv=` **on the train+val part only**. Test stays sealed.
- Report metrics on depth-0 rows (your games, the bot's real distribution) next to all rows. Use a **block bootstrap by week** for intervals, because rows share players.

### How "no leakage" and "train/serve parity" are tested

All tests run on a synthetic `random_world(seed)` crawled into a temporary mirror. `fakes.match_json` gets random winners and durations; today blue always wins, which would make outcome tests pass trivially.

1. **Future invariance.** For a sample of anchors at T: in a copy of the database, scramble every outcome and stat column of games not finished at T (including the anchor itself) and delete everything that started later. Features must be **bit-identical**. The test also asserts that *some* later row did change, so it can't pass vacuously.
2. **Batch invariance.** `build([pg])` equals `build(all)` restricted to that game. This catches anything that peeks across rows in a batch.
3. **Spectator parity.** Build a spectator-shaped payload from an anchor and check that `from_live(payload) == pregame_of(anchor)`, with equal X rows. In the bot phase: delete the listings, run `prefetch` against `FakeApi` (10 listing calls, 0 match calls), and the row must be equal.
4. **Antisymmetry.** Swapping sides flips every `_diff` column's sign exactly.
5. **Loader oracle.** For every (anchor, seat), the loader's history equals `anchor_history` rows that pass the end < T rule.
6. **Declared sources and the import firewall.** An undeclared read raises. Only `_sources.py` imports `sqlite3`. Only `dataset.py`/`artifact.py` import sklearn.
7. **Per-feature worked examples.** For instance, `WWL…` → streak +2; N = 0 → form 0.5; no history → kda None.

### Dependencies (need your OK)

| Package | Where | Why |
|---|---|---|
| `pandas` | runtime | The DataFrame sklearn, SHAP, statsmodels and every tutorial use. Used for `X`, `Dataset` and `read_sql`. The bot needs it too |
| `scikit-learn` | runtime | Pipeline, logistic regression, metrics, calibration. The bot unpickles the Pipeline. It brings `numpy` and `joblib` |
| `jupyterlab`, `matplotlib` | `notebook` dependency group | Exploration and calibration plots |
| `polars`, `pyarrow` | no | 500k rows is small, and nothing is materialised |
| `lightgbm`, `shap` | later | Ask when we get there |

## Synthesis decision

Two runners (Opus, Sonnet) sketched independently, and a Sonnet judge scored them. **Both converged** on:
- one pre-game input type with no role or outcome;
- features as registered functions with derived columns;
- no fit step, with population statistics computed as-of game *end* < T;
- blue-minus-red rows with the intercept as the side advantage;
- no materialised feature store;
- a data cut-off for reproducibility;
- future-invariance and parity tests.

**Base: candidate 1** ("one Pregame in"). The judge scored it 25.5 vs 23. It's stronger on train/serve parity (the bot's prefetch is derived from declared sources, plus batch-invariance and prefetch tests), on coverage reporting, and on the model-card guard. Candidate 2 assumed a patch value the spectator doesn't provide, and its bot hand-assembled what to fetch.

**Grafted from candidate 2:**
- **Feature functions are pure per-seat Python** over plain objects (`SeatView`/`PastGame`), not pandas Series. They're easier to read and test for a Java developer, and unit-testable without a database. Data loading stays vectorised. *This is the main deviation from the judge's pure pick, and it's question 1 below.*
- The future-invariance test with scrambled stats, and the time-split embargo plus rolling-origin CV.
- Boundary checks on loaded history (`LeakageError`), skip-reason counts, the loader-vs-`anchor_history` oracle, the antisymmetry test, the import firewall, and golden vectors (replacing candidate 1's "hash the whole package" id).

**Fixed, based on the judge's review:**
- κ values derived from the stated formula (κ ≈ 100 for form, ≈ 300 for champions); both candidates had used values that were too small.
- `history_n` read from the crawl spec instead of a constant.
- A known limit stated for old cut-offs.
- The test set stays out of CV.
- Block bootstrap by time instead of "by match".
- Seat order normalised for exact parity.

**Rejected:**
- Candidate 2's ±3-day post-game rank snapshots (they leak at promotion boundaries).
- Its all-time champion ledger (stale across patches).
- Its symmetric `__mean` columns (no linear signal in a diff model).
- Auto-joining new features to the baseline set.

## Tradeoffs accepted

- **A second "before T" query.** We accept it next to `anchor_history`, because live games have no anchor row and this one uses the stricter end < T rule. A test ties the two together.
- **Champion statistics recomputed for every row.** We accept this instead of fitting them once, in exchange for no fit discipline at call sites and no stale priors in the bot.
- **Vectorised pandas idioms** in features (owner's choice). We accept them in exchange for speed and the tutorial ecosystem. The exact-index check, the helpers and worked-example tests per feature keep them reviewable.
- **Dropping incomplete anchors.** We accept it in exchange for training on exactly what a fully prefetched bot sees.
- **Diff-of-means only in the MVP.** We accept losing within-team spread (`min` covers the weakest player) in exchange for 8 readable coefficients.

## Alternatives considered

- **Features as SQL views over `anchor_history`.** Every tweak would be a schema bump that `Mirror.open` rejects, live games have no anchor row, and streaks and shrinkage are awkward in SQL.
- **sklearn-style `fit/transform` for priors and encoders.** Leakage safety would depend on every caller fitting on the right split, in-sample target statistics overfit, and the bot's champion rates would go stale after one patch.
- **Vectorised pandas features (`ctx -> Series`).** This is candidate 1's original style. It's faster at large scale and closer to tutorials, but the idioms (`groupby`, `cummax`, index alignment) are harder to read and review. It's question 1 below.
- **A materialised feature store** (a table or parquet). It's a second copy of derived data that can drift, the bot can't use it, and a build takes seconds anyway.

## Open questions (need the owner)

1. ~~Feature style~~. **Decided: vectorised pandas.**
2. ~~Dependencies~~. **Decided: yes**, pandas and scikit-learn at runtime, plus a `notebook` group.
3. ~~Completeness~~. **Decided: strict 10/10.**
4. **Champion window.** 30 days by end time, or "current + previous patch"?
5. **Headline metric.** Depth-0 rows (your games), with all rows as the secondary?
6. **Risk: spectator start time.** It can differ from match-v5's `gameStartTimestamp`, or be 0 while loading. `from_live` falls back to `now`, and `prefetch` uses `pregame.as_of_ms` for the listings so the key matches.
7. **Risk: denser champion windows in training.** Training windows include games backfilled because of later anchors, so they're denser than what the bot sees live. The bot should log `Nc` per seat so we can compare.

## Next implementation step

1. `pregame.py`.
2. `_sources` HISTORY with coverage, and the anchor rows.
3. The engine, with `form_wr`/`streak`/`n_games`.
4. The future-invariance, batch-invariance and loader-oracle tests on `random_world`.
5. The remaining 5 features, `Dataset`/`time_split`, and a first notebook baseline against the constant-prior log loss.
