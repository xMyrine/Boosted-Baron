-- Oracle of the Rift: mirror schema v2 (SQLite >= 3.37 for STRICT, JSON1 for views).
-- The single source of the schema. Applied to an empty database (PRAGMA user_version = 0) by
-- `docker compose run --rm db-init` (db/init.sh) and by Mirror.open(). One transaction:
-- the database ends up either fully at v2 or untouched.
-- An existing database is upgraded in place by oracle/mirror/migrations.py (ordered, additive). Whoever changes
-- this file adds the migration for the change; tests/test_bot_migration.py fails when the two disagree.
--
-- journal_mode = WAL is persistent in the file (set once by db/init.sh); notebooks read while the crawler writes.
-- Per-connection PRAGMAs (set in Mirror.open):
--   synchronous  = NORMAL    a committed txn survives an app crash; an OS crash may drop the last few.
--                            Fine: work is derived, lost rows simply reappear as gaps.
--   foreign_keys = ON, busy_timeout = 30000
--
-- Units: every timestamp is INTEGER epoch MILLISECONDS, column suffix _ms. The only seconds conversion
-- happens inside RiotApi.match_ids (Riot's startTime/endTime are seconds).
--
-- Three families; the family tells you who writes a table and when:
--   FACTS       What Riot told us, and when. Insert-only (a listing may be replaced by a longer one).
--   PROJECTION  Pure function of FACTS (mirror/_project.py), written in the same txn. `oracle reproject` rebuilds.
--   DECISIONS   What we chose to collect, plus retry / accounting bookkeeping.
--
-- There is deliberately NO queue / pending / in_progress table. Work = DatasetSpec - FACTS, computed by
-- Mirror.gaps(). A crash leaves nothing to clean up.

BEGIN;


-- ============================================================ DECISIONS

-- Seeds: players whose own games are always anchors (the owner, maybe friends).
-- PUUIDs are encrypted per Riot API app. Before crawling, Mirror re-resolves one tracked Riot ID (1 call):
-- a different PUUID means the key belongs to another app, and the crawl refuses to start.
CREATE TABLE tracked_player (
    puuid        TEXT PRIMARY KEY,
    game_name    TEXT NOT NULL,              -- also used for the PUUID check above
    tag_line     TEXT NOT NULL,
    added_at_ms  INTEGER NOT NULL
) STRICT;

-- Incremental sync of a tracked player's match-id list. Backfill and every later poll are ONE state machine:
--   cycle start : since_ms = lookback horizon (first time) or previous cycle_started_ms - SETTLE (3 h)
--   each page   : insert discovered anchors + page_offset += 100      -- ONE txn per page (resumable)
--   cycle end   : short page -> since_ms := cycle_started_ms - SETTLE; page_offset := 0; next_due_ms := now + poll
-- WHY the 3 h overlap: a game still running during a sync appears later with a start time in the past.
-- Offsets can shift if a new game finishes mid-walk: that only yields a duplicate id (no-op), never a gap.
CREATE TABLE sync_cursor (
    puuid             TEXT NOT NULL REFERENCES tracked_player(puuid),
    queue_id          INTEGER NOT NULL,
    since_ms          INTEGER NOT NULL,
    page_offset       INTEGER NOT NULL DEFAULT 0,
    cycle_started_ms  INTEGER,
    next_due_ms       INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (puuid, queue_id)
) STRICT;

-- Anchors: matches that will be training rows. Each needs pre-match history for all 10 participants.
-- seq = completion order: the frontier works anchors in (depth, seq), so an interrupted crawl leaves the
-- maximum number of COMPLETE rows. An anchor row may exist before its match is fetched.
-- Persisted (not derived) so the anchor set doesn't shift under a half-finished crawl.
CREATE TABLE anchor (
    seq          INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id     TEXT NOT NULL UNIQUE,
    depth        INTEGER NOT NULL CHECK (depth >= 0),   -- 0 = tracked player's game; d+1 = promoted from depth d
    source       TEXT NOT NULL CHECK (source IN ('tracked', 'promoted', 'manual')),
    added_at_ms  INTEGER NOT NULL
) STRICT;

-- Transient failures only (5xx after retries, 429 retry cap, unparseable body). Sparse: a row exists only
-- while an item is failing; deleted in the same txn as the successful fill. 404 is NOT a failure (it's a fact).
-- Key columns mirror Gap.key so frontier SQL anti-joins on columns, never on formatted strings:
--   sync (puuid, queue)  match|timeline (match_id, '')  listing (puuid, queue:before_ms)  snapshot (puuid, kind)
CREATE TABLE fetch_failure (
    kind              TEXT NOT NULL CHECK (kind IN ('sync', 'match', 'timeline', 'listing', 'snapshot')),
    k1                TEXT NOT NULL,
    k2                TEXT NOT NULL DEFAULT '',
    attempts          INTEGER NOT NULL CHECK (attempts >= 1),
    next_retry_at_ms  INTEGER NOT NULL,      -- now + min(10 min * 2^attempts, 24 h)
    parked            INTEGER NOT NULL DEFAULT 0 CHECK (parked IN (0, 1)),   -- attempts >= 6; cleared by `oracle retry-failed`
    last_error        TEXT NOT NULL,         -- never contains the API key
    PRIMARY KEY (kind, k1, k2)
) STRICT;

-- One row per `oracle crawl`, for "dataset as of run N with spec S".
CREATE TABLE crawl_run (
    run_id        INTEGER PRIMARY KEY,
    started_ms    INTEGER NOT NULL,
    ended_ms      INTEGER,                   -- NULL = running, or the process died (harmless)
    spec_json     TEXT NOT NULL,
    calls         INTEGER NOT NULL DEFAULT 0,
    stop_reason   TEXT                       -- drained|budget|daily_cap|error|stopped|interrupted
) STRICT;

-- Call accounting for visibility ("how many calls did we spend, on what") and the cross-run daily cap.
-- Best-effort; rate-limit SAFETY never depends on it (the limiter trusts response headers).
CREATE TABLE api_call_stat (
    day        TEXT NOT NULL,                -- UTC 'YYYY-MM-DD'
    routing    TEXT NOT NULL,                -- 'europe' | 'euw1'
    endpoint   TEXT NOT NULL,                -- Endpoint enum name, e.g. 'MATCH'
    lane       TEXT NOT NULL CHECK (lane IN ('interactive', 'background')),
    outcome    TEXT NOT NULL CHECK (outcome IN ('ok', 'not_found', 'throttled', 'server_error', 'client_error', 'network')),
    calls      INTEGER NOT NULL,
    PRIMARY KEY (day, routing, endpoint, lane, outcome)
) STRICT;


-- One row per live game the bot prophesied about; the row is a small state machine:
--   pending -> posted | failed        (posted and failed are final)
-- The INSERT of the pending row is the idempotency claim: the second attempt for the same game inserts nothing and
-- does nothing. `payload` is the spectator game, so a restart can resume without asking Riot again.
-- A pending row left by a crash converges on the next start (service.RESUME_WINDOW_MS): resumed if it is recent,
-- else marked failed. `prophecy_json` is the stored Prophecy (bot/prophecy.py); it is the source of truth that
-- `oracle prophecies` scores against the finished match, so it is written BEFORE the outcome exists.
CREATE TABLE prophecy (
    platform_id         TEXT NOT NULL,                 -- 'EUN1'; the match id is platform_id || '_' || game_id
    game_id             INTEGER NOT NULL,
    status              TEXT NOT NULL CHECK (status IN ('pending', 'posted', 'failed')),
    subject_puuid       TEXT NOT NULL,                 -- the tracked player whose spectator check found the game
    subject_side        INTEGER NOT NULL CHECK (subject_side IN (100, 200)),
    payload             TEXT NOT NULL CHECK (json_valid(payload)),
    claimed_at_ms       INTEGER NOT NULL,
    finished_at_ms      INTEGER,
    model_id            TEXT,
    p_blue              REAL CHECK (p_blue IS NULL OR p_blue BETWEEN 0 AND 1),   -- NULL: the vision was clouded
    complete            INTEGER CHECK (complete IN (0, 1)),                       -- all ten histories were read
    prophecy_json       TEXT CHECK (prophecy_json IS NULL OR json_valid(prophecy_json)),
    discord_message_id  TEXT,
    error               TEXT,
    PRIMARY KEY (platform_id, game_id),
    CHECK ((p_blue IS NULL) = (complete IS NULL)),
    CHECK (CASE status
        WHEN 'pending' THEN finished_at_ms IS NULL AND discord_message_id IS NULL
        WHEN 'posted' THEN finished_at_ms IS NOT NULL AND discord_message_id IS NOT NULL AND prophecy_json IS NOT NULL
        ELSE finished_at_ms IS NOT NULL AND error IS NOT NULL END)
) STRICT;


-- ============================================================ FACTS

-- One row per match id we ever asked for. Presence = "never ask again".
CREATE TABLE raw_match (
    match_id        TEXT PRIMARY KEY,
    outcome         TEXT NOT NULL CHECK (outcome IN ('ok', 'gone')),  -- gone = 404
    payload         BLOB,                    -- zlib(verbatim JSON bytes), ~5 KB
    fetched_at_ms   INTEGER NOT NULL,
    parsed_version  INTEGER,                 -- _project.PARSER_VERSION that produced match/participant; NULL = unprojected
    CHECK ((outcome = 'ok') = (payload IS NOT NULL)),
    CHECK (parsed_version IS NULL OR outcome = 'ok')
) STRICT;

CREATE TABLE raw_timeline (                  -- ~100 KB compressed: fetched only for anchors with depth <= timelines_max_depth
    match_id       TEXT PRIMARY KEY,
    outcome        TEXT NOT NULL CHECK (outcome IN ('ok', 'gone')),
    payload        BLOB,
    fetched_at_ms  INTEGER NOT NULL,
    CHECK ((outcome = 'ok') = (payload IS NOT NULL))
) STRICT;

-- "As listed at fetched_at_ms, the n most recent queue_id matches of puuid that started before before_ms."
-- Keyed by a point in time, not by an anchor: the bot's live game uses the same table and code path.
-- Replaced as a whole (delete items, insert) when history_n grows. Never partially written.
CREATE TABLE history_listing (
    puuid          TEXT NOT NULL,
    queue_id       INTEGER NOT NULL,
    before_ms      INTEGER NOT NULL,
    n_requested    INTEGER NOT NULL CHECK (n_requested > 0),
    fetched_at_ms  INTEGER NOT NULL,
    PRIMARY KEY (puuid, queue_id, before_ms)
) STRICT;

CREATE TABLE history_listing_item (
    puuid      TEXT NOT NULL,
    queue_id   INTEGER NOT NULL,
    before_ms  INTEGER NOT NULL,
    ord        INTEGER NOT NULL CHECK (ord >= 0),     -- 0 = most recent prior game
    match_id   TEXT NOT NULL,
    PRIMARY KEY (puuid, queue_id, before_ms, ord),
    FOREIGN KEY (puuid, queue_id, before_ms) REFERENCES history_listing ON DELETE CASCADE
) STRICT;
CREATE INDEX history_listing_item_by_match ON history_listing_item(match_id);

-- league-v4 / champion-mastery-v4 return CURRENT state only: each call is an un-refetchable snapshot.
-- Unranked = '[]': still a row, so we don't ask again.
-- FEATURE RULE (leakage): a snapshot is CLEAN for a match iff it was taken before the match ENDED
-- (taken_at_ms < game_start_ms + game_duration_s * 1000): rank and mastery only change once a game is over.
-- Use the latest clean snapshot. The live watcher (`oracle crawl --forever`) takes clean ones for all 10 players
-- at game start. A later snapshot already contains that game's LP/points change: approximate and leaky.
CREATE TABLE snapshot (
    kind         TEXT NOT NULL CHECK (kind IN ('league', 'mastery')),
    puuid        TEXT NOT NULL,
    taken_at_ms  INTEGER NOT NULL,
    payload      TEXT NOT NULL CHECK (json_valid(payload)),
    PRIMARY KEY (kind, puuid, taken_at_ms)
) STRICT;

-- spectator-v5 data exists only while the game runs. Archive every sighting (bans, champ picks).
CREATE TABLE live_game (
    platform_id  TEXT NOT NULL,
    game_id      INTEGER NOT NULL,           -- final match id = upper(platform_id) || '_' || game_id
    taken_at_ms  INTEGER NOT NULL,
    payload      TEXT NOT NULL CHECK (json_valid(payload)),
    PRIMARY KEY (platform_id, game_id, taken_at_ms)
) STRICT;


-- ============================================================ PROJECTION (rebuildable from raw_match)

-- Exists iff raw_match.parsed_version IS NOT NULL. Written only by _project via Mirror (fill and reproject).
CREATE TABLE match (
    match_id         TEXT PRIMARY KEY REFERENCES raw_match(match_id),
    platform_id      TEXT NOT NULL,
    queue_id         INTEGER NOT NULL,
    game_start_ms    INTEGER NOT NULL,       -- info.gameStartTimestamp
    game_duration_s  INTEGER NOT NULL,
    game_version     TEXT NOT NULL,          -- '14.20.618.1234'
    patch            TEXT NOT NULL,          -- '14.20'
    winning_team_id  INTEGER CHECK (winning_team_id IN (100, 200)),
    is_remake        INTEGER NOT NULL CHECK (is_remake IN (0, 1))
) STRICT;
CREATE INDEX match_by_start ON match(game_start_ms);
CREATE INDEX match_by_patch ON match(patch);

-- 10 rows per match. queue_id/game_start_ms copied from match by the same function, so
-- "a player's games before T" is one index range scan.
-- Subset of Riot's ~100 fields: need more? add column + bump PARSER_VERSION + `oracle reproject`.
CREATE TABLE participant (
    match_id          TEXT NOT NULL REFERENCES match(match_id) ON DELETE CASCADE,
    participant_id    INTEGER NOT NULL CHECK (participant_id BETWEEN 1 AND 10),
    puuid             TEXT NOT NULL,
    riot_name         TEXT,                  -- name at match time (names change)
    riot_tag          TEXT,
    team_id           INTEGER NOT NULL CHECK (team_id IN (100, 200)),
    champion_id       INTEGER NOT NULL,
    team_position     TEXT NOT NULL,         -- TOP|JUNGLE|MIDDLE|BOTTOM|UTILITY|'' (remakes)
    win               INTEGER NOT NULL CHECK (win IN (0, 1)),
    kills             INTEGER NOT NULL,
    deaths            INTEGER NOT NULL,
    assists           INTEGER NOT NULL,
    damage_to_champs  INTEGER NOT NULL,
    damage_taken      INTEGER NOT NULL,
    gold_earned       INTEGER NOT NULL,
    cs                INTEGER NOT NULL,      -- lane + neutral minions
    vision_score      INTEGER NOT NULL,
    champ_level       INTEGER NOT NULL,
    game_start_ms     INTEGER NOT NULL,
    queue_id          INTEGER NOT NULL,
    PRIMARY KEY (match_id, participant_id),
    UNIQUE (match_id, puuid)
) STRICT;
CREATE INDEX participant_by_player_time ON participant(puuid, queue_id, game_start_ms);


-- ============================================================ VIEWS (derived; feature code reads these)

CREATE VIEW anchor_info AS
SELECT a.seq, a.match_id, a.depth, a.source,
       m.queue_id, m.game_start_ms, m.patch,
       m.game_start_ms AS before_ms            -- the listing key for this anchor
FROM anchor a
JOIN match m ON m.match_id = a.match_id;

-- Leak-free history: per anchor, per participant, their listed prior games with stats.
-- Feature code reads history ONLY from here. The WHERE re-checks time order, so the guarantee holds
-- regardless of whether Riot's endTime filters on creation, start or end.
CREATE VIEW anchor_history AS
SELECT ai.match_id       AS anchor_match_id,
       ai.depth          AS anchor_depth,
       ai.game_start_ms  AS anchor_start_ms,
       ai.patch          AS anchor_patch,
       ap.puuid,
       ap.team_id        AS anchor_team_id,
       ap.team_position  AS anchor_position,
       ap.champion_id    AS anchor_champion_id,
       li.ord            AS games_ago,
       pm.match_id       AS prior_match_id,
       pm.game_start_ms  AS prior_start_ms,
       pm.patch          AS prior_patch,
       pm.is_remake      AS prior_is_remake,
       pp.champion_id, pp.team_position, pp.win, pp.kills, pp.deaths, pp.assists,
       pp.damage_to_champs, pp.gold_earned, pp.cs, pp.vision_score
FROM anchor_info ai
JOIN participant ap          ON ap.match_id = ai.match_id
JOIN history_listing_item li ON li.puuid = ap.puuid AND li.queue_id = ai.queue_id AND li.before_ms = ai.before_ms
JOIN match pm                ON pm.match_id = li.match_id
JOIN participant pp          ON pp.match_id = pm.match_id AND pp.puuid = ap.puuid
WHERE pm.game_start_ms < ai.game_start_ms;

-- Ranked entries per snapshot (see FEATURE RULE on `snapshot`).
CREATE VIEW league_entry AS
SELECT s.puuid, s.taken_at_ms,
       json_extract(e.value, '$.queueType')    AS queue_type,
       json_extract(e.value, '$.tier')         AS tier,
       json_extract(e.value, '$.rank')         AS division,
       json_extract(e.value, '$.leaguePoints') AS lp,
       json_extract(e.value, '$.wins')         AS wins,
       json_extract(e.value, '$.losses')       AS losses
FROM snapshot s, json_each(s.payload) e
WHERE s.kind = 'league';

PRAGMA user_version = 2;

COMMIT;
