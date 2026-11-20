# Discord bot: running the oracle

## Overview

**Use `oracle bot` instead of `oracle crawl --forever`.** The bot runs the same forever-crawl inside its own process and adds the prophecies on top. Both hold the Riot key lock (`data/riot.lock`), so you can run one or the other, never both. Stop a running crawler before you start the bot.

While the bot runs, it does this for every ranked solo/duo game (queue 420) of a tracked player:

1. The live watcher notices the game. It checks each tracked player every 60 s, snapshots rank and mastery for all ten players, and archives the spectator data in `live_game`.
2. The bot claims the game by inserting a `pending` row in the `prophecy` table. A game never gets a second prophecy, even when two tracked players are in it or the bot restarts.
3. It reads the history of all ten players: 10 listings plus every listed match that is not stored yet. This runs at interactive priority, under a deadline of 300 s.
4. It builds the features with the same code as training, runs the model, and computes how much each feature moved the number.
5. It stores the prophecy as JSON in the `prophecy` row, then posts an embed to your channel.

The **subject** of a prophecy is the tracked player whose spectator check found the game. Everyone else appears by role and champion. Spectator-v5 shows a game only after the loading screen starts, so the prophecy arrives a few minutes into the game, not in champion select.

The bot runs on your own PC and makes outbound connections only: Discord's gateway, the Discord REST API, and the Riot API.

## Set up Discord (once)

1. Open the [Discord developer portal](https://discord.com/developers/applications) and select **New Application**.
2. On the **Bot** page, select **Reset Token** and copy the token. Leave all three **Privileged Gateway Intents** off. The bot never reads messages.
3. Build the invite URL. Replace `APPLICATION_ID` with the **Application ID** from the **General Information** page:

   ```
   https://discord.com/oauth2/authorize?client_id=APPLICATION_ID&scope=bot%20applications.commands&permissions=19456
   ```

   The scopes are `bot` and `applications.commands`. The permissions value 19456 is View Channel (1024) plus Send Messages (2048) plus Embed Links (16384), and nothing else.
4. Open the URL, pick your server, and authorize.
5. In Discord, turn on **Developer Mode** (User Settings, Advanced). Right-click the channel the bot should post in and select **Copy Channel ID**.
6. Put the token and the channel ID in `.env`. Never commit `.env`.

If the channel limits who can see it, give the bot's role View Channel, Send Messages, and Embed Links there too.

## Configure and install

```bash
uv sync --group bot               # discord.py, scikit-learn, joblib
```

| Variable | Needed by | Meaning |
|---|---|---|
| `DISCORD_BOT_TOKEN` | `oracle bot`, `oracle prophesy --send` | Bot token. Three parts separated by dots. The program masks it in every log line and traceback. |
| `DISCORD_CHANNEL_ID` | `oracle bot`, `oracle prophesy --send` | Numeric ID of the channel to post in. |
| `ORACLE_MODEL_DIR` | `oracle bot`, `oracle prophesy` | Folder with `model.joblib` and `model_card.json`. Default `./training/artifacts/win_lr`, relative to the `.env` file. |
| `RIOT_API_KEY` | `oracle bot` | As for the crawler. `oracle prophesy` and `oracle prophecies` need no key. |

Every command reads only the settings it needs. `oracle crawl` never fails over a missing Discord token.

## Try it without being in a game

`oracle prophesy` takes a finished game from the database, rebuilds what the spectator would have shown at its start, and runs it through the exact pipeline of a live game. It prints the message and, because the game is over, the real result. It makes no API calls and writes nothing to the database.

```bash
oracle prophesy                          # newest stored game with a complete history
oracle prophesy EUN1_4017659773          # a specific stored game
oracle prophesy EUN1_4016838293 --seat 7 # make participant 7 the subject (1-10; blue is 1-5, red is 6-10)
oracle prophesy --send                   # also post it to the Discord channel
```

By default the subject is the first tracked player in the game. If nobody tracked played, it is participant 1.

`--send` logs in over Discord's REST API only (no gateway connection), posts one message, and logs out. It tests the token, the channel ID, and the permissions without needing a game or the bot. A replay posted this way ends with a `Replay of ...` line, so nobody mistakes it for a live prophecy.

## Command reference

| Command | Riot calls | What it does |
|---|---|---|
| `oracle bot` | see below | Runs the forever-crawl, the live watcher, the prophecy service, and the Discord connection. Ctrl-C closes Discord, stops the crawl cleanly, and releases `data/riot.lock`. |
| `oracle bot --dry-run` | same as `oracle bot` | Does everything except Discord: it prints each prophecy instead of posting it, and needs no token or channel. It still calls Riot. |
| `oracle prophesy [MATCH_ID] [--seat N] [--send]` | 0 | The replay above. |
| `oracle prophecies` | 0 | Scores stored prophecies against finished games. See "Read oracle prophecies". |
| `/predict` in Discord | 1 spectator call per tracked player, until one is in a game | Runs a spectator check now instead of waiting up to 60 s. |

What `oracle bot` costs in Riot calls:

| When | Calls | Host |
|---|---|---|
| At start | 1 (the key check) | `europe` |
| Every 60 s | 1 spectator call per tracked player (8 players: 8 calls per minute) | platform |
| A new ranked game | 20 snapshot calls (league and mastery for ten players) | platform |
| Preparing that game | 10 listings, plus up to 200 matches (10 players times 20 games). Players already in the database cost almost nothing | `europe` |
| `/predict` | up to 1 spectator call per tracked player | platform |

A cold game is expensive because the `europe` budget is about 100 calls per 2 minutes. Ten strangers need about 210 calls, so the deadline of 300 s can expire. The bot then posts what it has, flagged (see "When something goes wrong"). The calls still land in the database, so the crawler and the next game with those players benefit.

While the bot runs, the crawler may use only about 75 % of each rate window. The rest is reserved for the bot's calls.

`/predict` answers only you (an ephemeral reply) and only in the configured channel. The reply is one of:

- "The oracle is reading the histories for NAME's game (KEY). The prophecy will appear in the channel."
- "The oracle has already spoken about KEY."
- "Nobody tracked is in a ranked solo/duo game right now."

The command is registered for the server that owns the channel, at startup, so it appears immediately. It needs the `applications.commands` scope from the invite URL.

## Check the first run

Three checks cover the parts that tests cannot reach, in this order:

1. Run `oracle prophesy --send`. A message appears in the channel. That proves the token, the channel ID, and the three permissions.
2. Start `oracle bot --dry-run` shortly before or during one of your own ranked games. The log shows `live game ... captured` within a minute of the loading screen, and the prophecy prints a few minutes later. That proves the Riot half end to end, with no Discord involved.
3. Start `oracle bot`, wait for the log line `the oracle is watching`, and type `/predict` in the channel. While you are in a ranked game the reply says the oracle is reading the histories, and the prophecy follows. Outside a game it says nobody tracked is in a ranked game.

## What a message looks like

This is real output of `oracle prophesy EUN1_4017659773` on the project database. In Discord it is an embed with the same text, a colour that follows the odds, and the two sides as columns.

```
Prophecy for PLAYER: Top Rengar

A coin flip dressed as destiny: 52% for PLAYER.
The champion picks please the oracle: the mid lane matchup (games on the picked champion) adds 6.7 points.
Against that, the top lane matchup (off-role rate) costs 5.0 points.
PLAYER on Rengar walks in on a 4-game losing streak. The oracle does not say tilt. The numbers do.
PLAYER's bot laner on LeBlanc is off-role: their history says mid laner. Brave.
Replay of EUN1_4017659773, a finished game. Real result: blue won, so PLAYER's side won. The oracle said 52%.

Blue side (roles inferred):
  Top Rengar
  Jungle Ezreal
  Mid Fizz
  Bot LeBlanc
  Support Tahm Kench

Red side (roles inferred):
  Top Zoe
  Jungle Graves
  Mid Kennen
  Bot Varus
  Support Seraphine

win_lr · 10/10 histories read · 1s · EUN1_4017659773
```

Every number in the text comes from the stored prophecy:

- The percentage is the model's P(blue wins), flipped to the subject's side when the subject is red.
- The driver lines name the strongest feature for and against the subject. The "points" are the change in the subject's win probability when that feature's contribution to the log-odds is removed. For the logistic-regression model the contributions are exact: they add up to the model's log-odds.
- The jokes appear only when a fact holds: a streak of at least 3 losses or 4 wins, a player off their main role, a champion they have not played in their history. At most two appear, subject first.
- The footer shows the model folder name, how many of the ten histories were read, how long preparation took, and the game key.
- The `Replay of` line exists only in replays.

The wording of a game is fixed by its key: the same game always reads the same. Roles are inferred from history and champion, not read from Riot, and are right about 80 % of the time.

## When something goes wrong

| Situation | What the bot does |
|---|---|
| The deadline expires, or some matches fail, so fewer than ten histories are complete | Predicts anyway and says so: an extra line ("only 7 of 10 histories could be read"), an amber embed, and `7/10 histories read` in the footer. The row records `complete = 0`. The model trained only on games with all ten histories, so treat the number as a rumour. |
| Riot answers 401 or 403 while preparing | Posts "The vision is clouded" with the reason and no number. The crawler also stops on a 401, so the bot exits with the error. |
| Feature building or the model raises | Posts "The vision is clouded" ("Something broke ..."), logs the traceback, and keeps running. |
| Discord refuses or fails to deliver | Retries network errors, 5xx, and 429 up to 4 times with doubling waits (1 s, 2 s, 4 s). A 403 or 404 fails at once. Then the row becomes `failed` with the error text, and the bot moves on. A failed row is not retried later. |
| The model card does not match the feature code | The bot refuses to start and names the differences. |
| The game is not queue 420 | Nothing. No row, no message. |
| Two tracked players are in the same game | One prophecy. The first one found is the subject. |
| The bot is stopped or crashes mid-game | See below. |

**Restart rules.** The `prophecy` row has three states: `pending`, then `posted` or `failed`. `posted` and `failed` are final. At every start the bot settles what the last run left:

- A `pending` row found less than 20 minutes ago is resumed. If it already holds a computed prophecy, the bot only posts it. If not, it computes it again from the stored spectator data and fetches only what is still missing.
- A `pending` row older than 20 minutes becomes `failed` with the error `abandoned`. The game is over, and a late omen is noise.
- A game the watcher archived in the last 20 minutes with no `prophecy` row at all (the crash came before the claim, or you started the bot after the game began) gets a prophecy now.

Posting is at-least-once. If the bot dies after Discord accepted the message but before the database noted it, the next start posts the same prophecy again. A repeated message is better than a missing one.

## Swap the model after retraining

1. Rebuild the dataset and retrain (`training/README.md`). Save with `tl.save_model("artifacts/<name>", ...)`.
2. Point `ORACLE_MODEL_DIR` at the folder, or overwrite the old folder.
3. Restart the bot. The model loads once at start.

The bot checks the model card against the running code:

- It **refuses** a model whose `feature_code_version` or `params` differ from the running feature code. The error lists each difference, for example `param history_n: model 10, running 20`. Retrain on a dataset built by the current code.
- It **warns** when the card has no `sklearn_version` (cards saved before this feature) or when it differs from the installed scikit-learn. A pickle can load wrongly or fail across versions. Retrain with the current `trainlib` to record the version.

Only a linear model (logistic regression, alone or as the last step of a pipeline) gets driver lines. Any other model still predicts, and the message simply has no driver lines. Joblib files are pickles and run code when loaded, so load only models you trained yourself.

## Read `oracle prophecies`

```
prophecies  0 pending, 12 posted, 1 failed
scored      9 finished games, 3 still waiting for the crawler to fetch them
             log loss   brier
oracle         0.6712  0.2398
10/10 reads    0.6650  0.2371   n=7
coin flip      0.6931  0.2500
n is 9: differences under about 0.02 are noise below a few hundred games.
```

The numbers above are illustrative and show only the format. Run the command for yours.

- Every prophecy is stored before its game ends, so this comparison cannot leak. It is the only fully out-of-sample evaluation the project has.
- A prophecy is scored when its finished match is in the database. The crawler lists a tracked player's games every 15 minutes, so a game can take that long, plus the fetch, to appear. The `still waiting` count is prophecies with a number whose match is not stored yet.
- Log loss punishes confident wrong answers. Brier is the mean squared error of the probability. An oracle that always says 50 % scores 0.6931 and 0.25, so lower than those is better.
- The `10/10 reads` line appears when some prophecies had incomplete histories. Compare it with the overall line to see whether the flagged ones hurt.
- Below a few hundred games the numbers are mostly noise.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `error: another process holds .../riot.lock (is the crawler or bot running?)` | A crawler, a `track`, or another bot owns the key. Stop it. `oracle bot` replaces `oracle crawl --forever`, so never run both. |
| `error: win_lr was trained on different features ... Differences: ...` | The feature code or its parameters changed since the model was trained. Rebuild the dataset with `oracle dataset` and retrain. |
| `error: ORACLE_MODEL_DIR (...) has no model.joblib` | The folder does not hold a saved model. Train one, or fix the path. A relative path is relative to `.env`. |
| `warning: model card has no sklearn_version` | Harmless. Retrain with the current `trainlib` to record the version. |
| LightGBM fails to load (`libomp`) | Only LightGBM models need the OpenMP runtime (`brew install libomp`). The default `win_lr` is scikit-learn only and does not. |
| `Discord answered 403 Missing Access` | The bot is not in the server, or it cannot see or post in the channel. Re-invite it with the URL above and check the channel's permission overrides for View Channel, Send Messages, and Embed Links. |
| Log line `PyNaCl is not installed, voice will NOT be supported` | Harmless. The bot never uses voice. |
| `Discord rejected DISCORD_BOT_TOKEN (login failed)` | The token is wrong or was reset. Copy a new one from the Bot page into `.env`. |
| `DISCORD_CHANNEL_ID must be the numeric id` | You copied the channel name or the server ID. Use **Copy Channel ID** with Developer Mode on. |
| A game started and nothing was posted | Check in this order. Is the game ranked solo/duo (queue 420)? Normal, flex, and ARAM get no prophecy. Is the player tracked (`oracle tracked`)? Did a minute pass since the loading screen? Spectator-v5 only lists a game once it has started. Is the row `failed`? `oracle prophecies` shows the count, and the bot's log shows the error. |
| `/predict` does not appear | The invite lacked the `applications.commands` scope, or the bot could not read the channel at startup. The log says `could not register /predict`. Re-invite and restart. The automatic prophecies work without it. |
| `/predict` answers "Riot did not answer in time" | The `europe` or platform budget is busy. Wait a minute and try again. |
| The message says `only N of 10 histories` | Some players' histories did not finish in time. A game with several new players takes minutes. The data stays in the database, so the same players are fast next time. |
| A prophecy never appears in `oracle prophecies` scores | The crawler has not fetched the finished match yet. Keep the bot running. |

## Where things live

| Path | Role |
|---|---|
| `src/oracle/bot/prophecy.py` | `Prophecy` (a `Foreseen` or `Clouded` outcome), its JSON, `Message`, and `render` |
| `src/oracle/bot/phrasebook.py` | The oracle's voice as tables: openings, driver lines, jokes, feature labels |
| `src/oracle/bot/predictor.py` | Loads the model and card, checks them, predicts, and computes the exact contributions |
| `src/oracle/bot/forecast.py` | Features, model, and assembly of one game, shared by the service and the replay |
| `src/oracle/bot/service.py` | `ProphecyService`: claim, prepare, predict, record, announce, and the restart rules |
| `src/oracle/bot/announce.py` | The `Announcer` port, the console adapter, and the Discord adapter |
| `src/oracle/bot/commands.py` | The `/predict` command |
| `src/oracle/bot/app.py` | Wiring and shutdown for `oracle bot` |
| `src/oracle/bot/replay.py` | `oracle prophesy` |
| `src/oracle/bot/scoring.py` | `oracle prophecies` |
| `src/oracle/mirror/migrations.py` | In-place upgrade of an older database (schema v1 to v2 adds `prophecy`) |
| `docs/design/discord-bot.md` | Design rationale and the list of deviations |
| `tests/test_service.py`, `tests/test_bot_*.py` | The tests. Run `uv run --group bot pytest -q`. Without the `bot` group, the tests that need scikit-learn or discord.py skip. |
