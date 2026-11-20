# Discord bot: design (implemented)

Status: **implemented** (`src/oracle/bot/`, `oracle bot`, `oracle prophesy`, `oracle prophecies`). Operations guide:
[`docs/wiki/Discord-Bot.md`](../wiki/Discord-Bot.md). The owner's decisions and every deviation from this proposal are
in "Implementation notes" at the end. Builds on `riot-import.md` (gateway, mirror, live watcher) and
`win-model-training.md` (`features_for_live`, model artifacts).

## Problem

A trained win model exists as a file (`training/artifacts/<name>/model.joblib` + `model_card.json`). We want: the
owner starts a ranked game → a prophecy appears in a Discord channel. Four constraints decide the shape:

1. **One process owns the Riot key.** `RiotGateway.open` takes `data/riot.lock`; a second process gets `KeyInUse`.
   The bot and the crawler therefore cannot be two programs. This was already anticipated in `riot-import.md`.
2. **A cold game is expensive.** `features_for_live` needs the history of all 10 players: 10 listing calls plus up
   to 10 × `history_n` (20) match fetches, about 210 calls on the regional host, against roughly 100 calls per
   2 minutes. A game with 9 strangers takes several minutes to prepare, not seconds.
3. **Discord needs a responsive event loop.** A bot connection sends a heartbeat every ~40 s. Feature building is
   synchronous pandas + SQLite, so it must not run on the loop (the same rule as "don't block the Netty thread").
4. **The numbers are the source of truth.** The text is rendered from a structured result, so templates now and an
   LLM later are interchangeable, and the result can be stored and scored after the game.

## Shape

```
                         one process: `oracle bot`   (one asyncio loop, holds riot.lock)
 ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
 │  crawl(forever=True)                    ProphecyService                       Announcer        │
 │  ┌────────────────┐  LiveCapture   ┌───────────────────────────┐  Prophecy  ┌──────────────┐  │
 │  │ watch_loop     │ ─────────────▶ │ 1 prepare  (Riot, async)  │ ─────────▶ │ render embed │──┼──▶ Discord
 │  │ spectator poll │  asyncio.Queue │ 2 features (thread)       │            │ send         │  │    channel
 │  │ lane loops     │                │ 3 predict  (thread)       │            └──────────────┘  │
 │  └───────┬────────┘                │ 4 explain                 │                   ▲          │
 │          │ background priority     │ 5 record                  │   /predict (later)│          │
 │          ▼                         └───────┬───────────────────┘ ──────────────────┘          │
 │     RiotGateway  ◀── interactive priority ─┘          │ read-only connection                  │
 │          │                                            ▼                                       │
 └──────────┼──────────────────────────────────── oracle.sqlite3 ────────────────────────────────┘
            ▼                                    model.joblib + model_card.json (loaded at start)
        Riot API
```

Three pieces, each replaceable on its own:

| Piece | Responsibility | Knows about |
|---|---|---|
| **Detector** (exists: `crawl.watch_loop`) | Notice a new live game of a tracked player, archive the payload, snapshot rank/mastery | Riot, mirror |
| **`ProphecyService`** (new) | Turn one live game into a `Prophecy`: numbers only, no text, no Discord | mirror, features, model |
| **`Announcer`** (new, a port) | Turn a `Prophecy` into a message and deliver it | Discord only |

In Spring terms: the detector publishes an application event, the service is the `@Service`, and the announcer is an
outbound adapter behind an interface. The queue between detector and service means a slow prediction never delays
the next spectator poll, and `/predict` later simply puts the same kind of job on the same path.

### The pipeline for one game

1. **Detect.** `watch_loop` already polls spectator-v5 every 60 s and calls `Mirror.capture_live`. Change: `crawl`
   accepts an `on_live` callback and passes it the capture. Only `queue_id == 420` continues; other queues get no
   prophecy (the model has never seen them).
2. **Prepare.** New `Mirror.prepare_live(api, pregame)`: fetch the 10 listings and the missing history matches,
   using `gateway.interactive()` so it is served before crawl work and uses the reserved headroom. It reuses the
   existing fill code, so everything fetched is also normal mirror data. It runs under a deadline (proposed: 5 min).
3. **Features.** `features_for_live(ro_con, Pregame.from_spectator(payload, now_ms))` in `asyncio.to_thread`, with
   its own read-only connection. Same code path as training, so no train/serve skew.
4. **Predict.** `model.predict_proba(f.matches[card["features"]])[0, 1]` gives P(blue). The owner's number is that
   value, or 1 − it when the owner is on red. The model is loaded once at start, and loading **fails** if the card's
   `feature_code_version` or `params` differ from the running code (the check `save_model` was written for).
5. **Explain.** For the LR model: coefficient × standardised value per feature, top 3 for and against. This is
   exact for a linear model, so SHAP is only needed once a tree model wins.
6. **Record, then send.** Insert the `Prophecy` into a `prophecy` table keyed by `(platform_id, game_id)`, then
   post. The key makes a restart idempotent (no double post), and the row is the input for the scoring below.

### The hand-off type

```python
@dataclass(frozen=True, slots=True)
class Prophecy:
    game_key: str                 # "EUW1_7512345678"
    owner_side: Side
    p_owner_win: float | None     # None = could not predict (see degraded modes)
    drivers: tuple[Driver, ...]   # (feature, seat or team, value, contribution), strongest first
    seats: tuple[SeatView, ...]   # per player: champion, inferred role, streak, champ_games, off_role, ...
    complete: bool                # all 10 histories were available
    model: str                    # model card id: dataset id + saved_at
    prepared_in_s: float
```

`render(prophecy) -> Message` is a pure function (templates keyed on the numbers: probability band, the top driver,
streaks). An LLM renderer later has the same signature, and tests need no Discord.

### Degraded modes (decided up front, because they will happen on the first evening)

| Situation | Behaviour |
|---|---|
| Deadline hit, histories incomplete | Predict anyway and post with `complete=False` ("the vision is clouded"). **Caveat:** the model was trained only on games with 10/10 complete histories, so this is outside its training distribution. Alternative: post no number at all |
| Riot error / `DeadlineExceeded` | Post a short failure line, record it, keep running |
| Discord send fails | Retry with backoff a few times, then log. The `prophecy` row stays, marked unsent |
| Model card mismatch | The bot refuses to start, with a message saying which model to retrain |
| Bot restarted mid-game | `live_game` already has the game, so the detector stays silent. On start, the service checks for a live game without a `prophecy` row |

### What this adds for learning: live scoring

Every prophecy is stored **before** the outcome exists. When the crawler later fetches the finished match, a
`oracle prophecies` report joins the two and prints log loss / Brier against the constant baseline. This is the
only truly out-of-sample evaluation the project will have, and it cannot leak by construction. It matters here
because the current models were validated on 6 games: the notebook metrics are noise, and the bot's own track
record will be the honest measure.

## Options for the Discord side

The `Announcer` port keeps this choice cheap to change.

| | A. `discord.py`, in-process (recommended) | B. Webhook | C. Raw REST with a bot token |
|---|---|---|---|
| How | Gateway WebSocket + REST, library handles both | One HTTPS POST to a channel webhook URL, via the `httpx` we already have | `POST /channels/{id}/messages` via `httpx` |
| New dependency | `discord.py` (pulls `aiohttp`) | none | none |
| Secret | bot token + channel id | webhook URL | bot token + channel id |
| `/predict`, buttons, reactions | yes | no (send only) | no, unless we also run an inbound HTTP endpoint |
| Loop integration | `async with client: await client.start(token)` inside our own loop, next to `crawl` | trivial | trivial |
| Teaches | how a bot actually works (gateway, intents, interactions) | almost nothing | Discord's REST API |

Recommendation: **A**, because `/predict` is in the plan and it needs the gateway. No privileged intents are
needed (we never read message content). B is the fallback if you want the pipeline proven end to end before adding
a dependency: it is about 20 lines behind the same port, and swapping to A later touches nothing else.

## Module map (proposed)

```
src/oracle/bot/
  prophecy.py    Prophecy / Driver / SeatView, render() + templates      (pure, no I/O)
  predictor.py   load model + card, compatibility check, predict, explain (sklearn imported here only)
  service.py     ProphecyService: prepare -> features -> predict -> record
  announce.py    Announcer protocol + the Discord adapter
  app.py         wiring: gateway, mirror, crawl task, queue, service, announcer; graceful shutdown
cli.py           + `oracle bot`, + `oracle prophecies`
mirror           + prepare_live(); crawl() + on_live; schema + prophecy table
config.py        + DISCORD_BOT_TOKEN (masked like ApiKey), DISCORD_CHANNEL_ID, ORACLE_MODEL_DIR
```

Dependencies: `scikit-learn` and `joblib` move from the `training` group to a `bot` group (needed to unpickle), plus
`discord.py` for option A. `oracle crawl` and `oracle dataset` stay as light as today.

## Tradeoffs accepted

* **One process, one failure domain.** A crash in the bot stops the crawler too. Accepted: the key lock forces it,
  and both are resumable.
* **Minutes, not seconds, for a cold game.** The prophecy lands during the early game, not in champion select.
  Spectator-v5 only shows a game once it has started loading, so "before the game" was never available anyway.
* **joblib is version-sensitive.** A model pickled with one scikit-learn version may not load with another (Colab
  vs local). Mitigation: record the sklearn version in the model card and check it at load.
* **Pickle executes code on load.** Fine for a file we trained ourselves; never load a model from elsewhere.

## Verification plan

* `render` and `predictor.explain`: plain unit tests on hand-built `Prophecy` values.
* `ProphecyService`: the existing synthetic world (`tests/featureworld.py`) + fake gateway (`tests/fakes.py`) and a
  recording fake `Announcer`: one game → exactly one message, a restart → still one, a deadline → `complete=False`.
* Parity: service output for a finished synthetic game equals the row `Dataset.build` produces for it.
* Manual: `oracle bot --dry-run` prints the rendered prophecy for the newest archived `live_game` instead of sending.

## Decisions needed

1. Discord side: **A** `discord.py` (recommended), B webhook, or C raw REST.
2. Incomplete histories at the deadline: post a flagged number (recommended), or post no number.
3. Schema change: add the `prophecy` table (needed for idempotency and live scoring).
4. Dependencies: a `bot` group with `scikit-learn`, `joblib` and (for A) `discord.py`.
5. Which model the bot loads first: `win_lr` (recommended: simplest, exact explanations) or `win_best`.
6. Where it runs: the PC you play on is enough (outbound connections only), since it is on whenever you play.

## Build order (each step small and reviewable)

1. `prophecy.py` + `predictor.py` + `oracle bot --dry-run` on an archived `live_game` (no Discord, no new Riot calls).
2. `announce.py`: send that dry-run prophecy to the channel.
3. `prepare_live` + `on_live` + `service.py`: the automatic path.
4. `prophecy` table + `oracle prophecies` scoring report.
5. `/predict` slash command.

## Implementation notes

**Decisions taken** (they resolve "Decisions needed"): `discord.py` in-process (option A); an incomplete history at the
deadline still gets a flagged prediction; the `prophecy` table exists; dependencies are in a `bot` group; the default
model is `training/artifacts/win_lr`; the bot runs on the owner's PC.

**Deviations from the proposal, and why.**

* **The subject is not "the owner".** Eight players are tracked, so the subject of a prophecy is the tracked player whose
  spectator check found the game, and there is one prophecy per game. `Prophecy.owner_side` became `subject_side`, and
  the envelope gained `subject` (PUUID) and `subject_name` (the tracked player's game name, used in the title).
* **`Prophecy` is an envelope around a union.** `outcome` is `Foreseen | Clouded`, not a nullable number plus flags.
  `p_subject_win` and `complete` are derived (from `p_blue` and the subject's side, and from per-seat history flags), not
  stored, so they cannot disagree with their sources. `Clouded` has two reasons: `PREDICTION_FAILED` and `RIOT_REFUSED`.
* **Degraded modes changed in one row.** The proposal posted a failure line for any Riot error or deadline. Now a blown
  deadline or a failed fetch still predicts (flagged), and only a 401 or 403 while preparing gives a clouded message.
* **Restart rules are defined.** A `pending` row recent enough (20 minutes) is resumed, an older one becomes `failed`
  (`abandoned`), and a recently archived `live_game` with no row is claimed at start. The proposal only said the service
  "checks for a live game without a prophecy row". Posting is at-least-once. The `prophecy` row also stores the
  spectator payload (not in the proposal), so a resume needs no Riot call to start.
* **Extra modules.** `phrasebook.py` (wording tables), `forecast.py` (features, model, and assembly, shared by the service
  and the replay), `errors.py`, `champions.py`, `commands.py` (`/predict`), `replay.py`, and `scoring.py`. The proposal
  listed five modules.
* **`oracle bot --dry-run` is not the manual check.** No `live_game` payload was archived yet, so the offline check is
  `oracle prophesy [MATCH_ID]`: it replays a finished game through the live pipeline and prints the real result next to
  the prediction. `oracle bot --dry-run` runs the whole bot but prints instead of posting.
* **`prepare_live` takes PUUIDs and a time, not a `Pregame`.** The mirror must not import the feature package (and
  pandas). Preparation reuses `Mirror.fill`, so a second call for the same game fetches only what is missing.
* **`/predict` does its own spectator check.** It does not call `capture_live`, which would race the watcher (both would
  snapshot the same ten players). The claim makes the two paths idempotent: whichever sees the game first prophesies it,
  and the watcher still archives and snapshots it later. A game summoned by `/predict` before the watcher saw it has no
  pre-game rank snapshots at prediction time.
* **Two interactive handles.** One with a 300 s deadline for preparation, one with 8 s for `/predict`.
* **Explanations count the side of the map.** The intercept of the logistic regression is a driver ("the side of the
  map"). A driver's "points" are the change in the subject's win probability with that contribution removed. The top
  three for and the top three against are kept.
* **Champion names come from the stored match JSON** (`championName` is a Data Dragon id such as `TwistedFate`, made
  readable by a small table of exceptions). There is no Data Dragon call. An id not in the database reads `Champion #N`.
* **Migrations.** `Mirror.open` upgrades an older database in place (`mirror/migrations.py`, ordered and additive, one
  transaction per step) instead of raising on any version mismatch. A database newer than the code is still refused.
  `schema.sql` stays the one fresh-install schema, and `tests/test_migration.py` fails if a migrated database differs
  from a fresh one.
* **Config.** `DISCORD_BOT_TOKEN` and `DISCORD_CHANNEL_ID` are parsed only by commands that need them (masked like
  `ApiKey`). `ORACLE_MODEL_DIR` defaults to `./training/artifacts/win_lr`, relative to `.env`.
* **Model card.** `trainlib.save_model` now writes `sklearn_version`. The bot warns when it is missing or differs, and
  fails when `feature_code_version` or `params` differ.
* **`db/init.sh`** now says when an existing database is behind the current schema (it still never modifies a database).

**Not built.** The performance models, an LLM renderer (the `render(prophecy) -> Message` signature is the seam), and a
Data Dragon download.

**Not verified against the real services.** A real Discord login and post, `/predict` registration against Discord,
and a real live game. They are exercised with fakes only. The "Check the first run" section of `docs/wiki/Discord-Bot.md`
lists how to check each.
