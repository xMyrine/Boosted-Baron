# Riot import pipeline: design (proposal, not implemented)

Status: **implemented (import pipeline); bot methods pending.** Schema: [`src/oracle/mirror/schema.sql`](../../src/oracle/mirror/schema.sql). Any `oracle` command creates the database on first use; `docker compose run --rm db-init` does the same without Python. See "Implementation notes" at the end.

## Problem

We want to pull League data from the Riot API into SQLite, for leak-free modelling, using a personal key read from `.env`. Three constraints decide the shape:

1. **Frequency.** The key gets about 100 requests per 2 minutes, with a separate budget for each routing value: `europe` serves account-v1 and match-v5, and `euw1` serves league, mastery and spectator. Repeated 429s risk a blacklist. The real limits must come from response headers. Finished matches are immutable, so we fetch each one exactly once.
2. **Reactivity.** A crawl runs for hours or days. Later, the Discord bot will need about 10–100 calls within seconds of a game starting, using the same key. Giving the bot priority is not enough on its own: with a 120 s window, a crawl that has already spent the window can still make the bot wait up to 2 minutes. The bot also needs **reserved headroom**, and it must **fail fast** instead of hanging.
3. **Checkpoint.** A crawl gets killed (Ctrl-C, laptop sleep, crash). A restart must lose nothing, duplicate nothing, need no cleanup, and fetch only what is new.

There is also a data-science constraint from CLAUDE.md. Every training row needs the history of all 10 players *before* that match started. That makes the fan-out large (≈ 1 + 10 + 10·n calls per row), and history has to be "as of" a timestamp, not "as of now".

## Usage (caller's view)

```bash
# .env (git-ignored; .env.example has the names only)
RIOT_API_KEY=RGAPI-...
RIOT_PLATFORM=euw1                 # the regional host (europe) is derived from this
ORACLE_DATA_DIR=data

uv run --env-file .env oracle track "Name#TAG"          # Riot ID -> PUUID, start tracking (1 call)
uv run oracle tracked                                   # list tracked players; `oracle untrack "Name#TAG"` stops syncing
uv run --env-file .env oracle crawl --max-calls 3000    # ~1 h. Ctrl-C any time; rerun to resume
uv run --env-file .env oracle crawl --forever           # keep syncing new games (poll every 15 min)
uv run oracle status                                    # no network: progress, failures, calls today
uv run oracle reproject                                 # re-parse stored raw JSON after a parser change (0 calls)
```

**CLI crawl**
```python
settings = Settings.from_env()
spec = DatasetSpec.default()
async with RiotGateway.open(settings) as riot:     # takes data/riot.lock: one key owner per machine
    with Mirror.open(settings.db_path) as mirror:
        report = await crawl(mirror, riot, spec, CrawlBudget(max_calls=3000))
print(report.summary())
```

**Bot (later). The crawler runs inside the bot process.**
```python
riot = RiotGateway.open(settings)
mirror = Mirror.open(settings.db_path)
asyncio.create_task(crawl(mirror, riot, spec, CrawlBudget(forever=True)))
api = riot.interactive(deadline_s=8)              # served first, reserve kept free for it

game = await mirror.live_game(api, owner_puuid)   # None = not in game; archived either way
try:
    hist = await asyncio.gather(*(mirror.history(api, p.puuid, before_ms=game.start_ms, n=spec.history_n)
                                  for p in game.participants))
except DeadlineExceeded:
    await channel.send("The Oracle's vision is clouded… (rate budget busy)")
```

**Notebook (read-only, no lock, no network)**
```python
con = Mirror.read_only(Path("data/oracle.sqlite3"))
hist = pd.read_sql("SELECT * FROM anchor_history", con)   # leak-free by construction
```

## Shape

### The core idea: the data is the checkpoint

SQLite holds **facts** (what Riot said, and when) and **decisions** (who we track, which matches are training rows). It does **not** hold a job queue. The work still to do is *computed*: `Mirror.gaps(spec)` = what `DatasetSpec` says should exist, minus what is stored, ordered by value. The crawler is a reconcile loop that asks for gaps, fills each one, and repeats.

> Java analogy: this is a Kubernetes controller (level-triggered: "make actual = desired"), not a Quartz/Spring-Batch job store (edge-triggered tasks with leases).

What this gives us:
- **Crash-safe.** `fill(gap)` makes the HTTP call first. It then writes one SQLite transaction with no `await` inside: raw payload, parsed rows, failure row cleared, cursor advanced. A kill either committed everything or nothing. There are no "in progress" rows, so restart needs no recovery code.
- **Idempotent.** Every write is keyed by a natural key (match id; `(puuid, queue, before_ms)`; `(kind, puuid, taken_at_ms)`). Running a fill twice rewrites the same facts.
- **Incremental.** A fetched match is never a gap again. New work comes only from tracked players playing new games, or from new anchors.
- **Policy changes are free.** If `history_n` goes from 10 to 15, every short listing becomes a gap again. There is no queue to migrate.
- **Cost of a crash.** At most the calls that were in flight are repeated (concurrency is 3 on `europe` and 2 on `euw1`). No duplicate rows.

### Data model: anchor-centric (`src/oracle/mirror/schema.sql`)

| Family | Tables | Rule |
|---|---|---|
| Facts | `raw_match`, `raw_timeline`, `history_listing` + `_item`, `snapshot`, `live_game` | Insert-only and the source of truth. A 404 is a fact (`outcome='gone'`) and is never retried. |
| Projection | `match`, `participant` | Pure function of `raw_match`, written in the same transaction. `oracle reproject` rebuilds them; `parsed_version` makes the rebuild selective. |
| Decisions | `tracked_player`, `sync_cursor`, `anchor`, `fetch_failure`, `crawl_run`, `api_call_stat` | What we chose to collect, plus retry and accounting bookkeeping. |

Terms:
- **Anchor**: a match that becomes a training row.
- **History listing**: the fact "the n most recent queue-420 match ids of player P that started before time T". It comes from `match_ids?endTime=T&count=n`. Each anchor needs 10 listings at T = its start, plus the matches those listings name.
- Listings are keyed by *time*, not by anchor. The bot's live prediction ("history before now") therefore uses the same table and code path as training. **Train/serve consistency is structural.**
- The **`anchor_history` view** joins everything and re-checks `prior.start < anchor.start`. Feature code reads history only from this view, so leakage is prevented by the structure, not by discipline.

Where anchors come from:
1. **Depth 0: tracked players' own games** (the owner, maybe friends). An incremental sync uses `sync_cursor`. Both the backfill and later polls run the same page cycle, and the cursor is committed **per page**, so a crash mid-backfill resumes at the right page. The next cycle starts 3 h before the previous one started. A game that was still running during the last sync shows up later with a start time in the past, so the 3 h overlap catches it. Duplicates are no-ops.
2. **Depth 1+: promotion.** History games of depth-d anchors that are already fetched get promoted to depth d+1. This is bounded by `max_depth`, `max_anchors`, and a WIP limit on incomplete promoted anchors. It turns the 9 co-players' games we fetch anyway into thousands of extra training rows (owner games alone are only a few hundred rows, too few for gradient boosting).

**Frontier order on `europe`:**
1. Tracked syncs.
2. Anchors in `(depth, seq)` order: for each, the anchor match, then its 10 listings, then its history matches.
3. Timelines, last, and only for depth 0.

This is depth-first: one anchor is finished before the next one starts. **Wherever the crawl stops, you hold the maximum number of *complete* rows for the calls spent.**

**On `euw1`:** league and mastery snapshots for participants of recently played anchors, newest first. League-v4 knows only "now", so a snapshot is an approximation. The feature rule is: use the latest snapshot taken at or before the match start. If there isn't one, use the nearest later snapshot and add a `lag` feature, because a later snapshot already includes that game's LP change.

### Fetching: one gateway, two priority handles

```
RiotGateway   (one per process; flock on data/riot.lock)
  └─ Lane     one per routing value: europe, euw1
       ├─ app Bucket      windows learned from X-App-Rate-Limit
       └─ method Buckets  one per endpoint, from X-Method-Rate-Limit
            └─ Window     sliding log of send timestamps
```

- **Header-driven.** Limits start at the defaults (`20:1,100:120`) and are replaced by the headers from the first response. The `*-Count` headers correct the local log *upward* only. This also absorbs calls from a previous process that died, or from a stray script.
- **Conservative.**
  - A sliding log is stricter than Riot's fixed windows.
  - The limiter uses only 95 % of each limit and pads each window by 1 s to allow for latency.
  - A "cold" lane allows 1 request in flight until its first response arrives.
- **Even pacing for background calls.** Background calls are spaced evenly across the long window (about one call per 1.3 s) instead of bursting 100 calls and then going silent for 2 minutes. Bursting would wreck reactivity.
- **Priority belongs to the handle, not to a call argument.**
  - `crawl()` builds its own `riot.background()`, and the bot holds `riot.interactive(deadline_s)`.
  - The crawler cannot accidentally take the bot's priority.
  - Interactive callers are served first and have their own in-flight slots.
- **Reserve.** While an interactive handle exists (that is, the bot is running), background may use only `limit − max(2, 25 %)` of each window. A CLI-only crawl uses the full budget.
- **Deadlines.** If an interactive call can't get a permit within its deadline, it raises `DeadlineExceeded(eta)` immediately, without sending anything.
- **Single-flight.** Two callers asking for the same match at the same time (bot and crawler, or two players who share a game) share one in-flight request.

**Response handling** is one pure function, `classify(status, headers) -> Outcome`:

| Response | Action |
|---|---|
| 200 | Return `Doc` (verbatim bytes). |
| 404 | It's a fact: the match is `GONE`, or spectator returns `None` (not in game). |
| 429 `application` | Freeze the whole lane for Retry-After + jitter. Halve the background pace (AIMD: it recovers +5 % per minute). Count the violation. Re-queue the request with the same place in line. **3 in 10 min opens a breaker:** background is paused for 5 min. |
| 429 `method` | Freeze only that method's bucket. |
| 429 `service` / no Retry-After | Exponential backoff with jitter (1 → 64 s) on that method. |
| any 429, 4th time for one request | Give up on the request: `TransientError` → `fetch_failure`. 429s can't loop forever. |
| 5xx / timeout | Retry 3× in background, or 1× interactive (within the deadline). Then `fetch_failure` with backoff of 10 min·2^k, capped at 24 h, and parked after 6 attempts. |
| 401 | Trip the gateway: every call fails with `AuthError`, and the crawl stops. |
| 403 | Disable *that endpoint* (e.g. a product not enabled for the key) and report it loudly. |
| other 4xx | `BadRequestError`. The gap is parked and the crawl stops loudly, because it's our bug. |

> Java analogies: lanes are a bulkhead, the breaker is Resilience4j's CircuitBreaker, and the sliding log is a stricter cousin of Resilience4j's RateLimiter.

### Module map

```
src/oracle/
  config.py          Settings.from_env, ApiKey (masked repr), PLATFORM_TO_REGIONAL (the one routing table)
  ids.py             Puuid, MatchId, RiotId, EpochMs; boundary parsers
  riot/              the ONLY code that speaks HTTP or sees the key        [httpx]
    __init__.py      RiotGateway, RiotApi, Doc, GONE, errors
    endpoints.py     Endpoint enum: the single list of Riot methods (path, routing, "404 means")
    gateway.py       RiotGateway (lock, lanes, retry loop, single-flight), RiotApi, classify()
    _limiter.py      Window, Bucket, Lane (priority dispatcher, reserve, pacing, AIMD, breaker)
  mirror/            the ONLY code that touches SQLite
    __init__.py      Mirror, DatasetSpec, Gap kinds, LiveGame, PastGame
    mirror.py        open/migrate, track, gaps, fill, promote, history, live_game, status, reproject
    _frontier.py     one SQL query per gap kind + the priority order
    _project.py      pure JSON -> rows (the only code that knows Riot's field names)
    schema.sql       the one schema source (compose mounts this same file)
  crawl.py           crawl(): per-lane workers, budget, promotion cadence, stop rules
  cli.py             track | plan | crawl | status | reproject
```

Following a match from frontier to disk crosses three files: `crawl.py` → `mirror/mirror.py` → `riot/gateway.py`. Underscore modules are private, and an import-linter rule fails the build if anything outside their package imports them.

### Key signatures (bodies `NotImplementedError` until sign-off)

```python
# riot/gateway.py
class RiotGateway:
    @classmethod
    def open(cls, settings: Settings) -> "RiotGateway": ...   # raises KeyInUse if the lock is held
    def interactive(self, *, deadline_s: float = 8.0) -> "RiotApi": ...
    def background(self) -> "RiotApi": ...
    async def aclose(self) -> None: ...

class RiotApi:                       # a handle: the lane is fixed at creation
    async def account(self, riot_id: RiotId) -> Puuid | None: ...
    async def match_ids(self, puuid: Puuid, *, queue: int, start: int = 0, count: int = 100,
                        start_ms: EpochMs | None = None, end_ms: EpochMs | None = None) -> list[MatchId]: ...
    async def match(self, match_id: MatchId) -> Doc | Gone: ...
    async def timeline(self, match_id: MatchId) -> Doc | Gone: ...
    async def league_entries(self, puuid: Puuid) -> Doc: ...
    async def mastery(self, puuid: Puuid) -> Doc: ...
    async def active_game(self, puuid: Puuid) -> Doc | None: ...

def classify(status: int | None, headers: Mapping[str, str]) -> Outcome: ...   # pure, unit-tested

# mirror/mirror.py
@dataclass(frozen=True)
class DatasetSpec:
    queue_id: int = 420
    history_n: int = 10              # MUST equal what the bot uses live
    lookback_days: int = 120         # anchors older than this are ignored
    max_depth: int = 1
    max_anchors: int = 5000
    wip_anchors: int = 20
    timelines_max_depth: int = 0
    snapshot_window_days: int = 3

Gap = SyncTracked | FetchMatch | ListHistory | FetchTimeline | TakeSnapshot   # each has .key and .routing

class Mirror:
    @classmethod
    def open(cls, path: Path) -> "Mirror": ...   # applies schema.sql if user_version = 0
    @staticmethod
    def read_only(path: Path) -> sqlite3.Connection: ...
    async def track(self, api: RiotApi, riot_id: RiotId) -> Puuid: ...
    def gaps(self, spec: DatasetSpec, routing: Routing, now_ms: EpochMs, limit: int) -> list[Gap]: ...
    async def fill(self, gap: Gap, api: RiotApi) -> FillResult: ...  # FILLED | GONE | DEFERRED
    def promote(self, spec: DatasetSpec) -> int: ...
    async def live_game(self, api: RiotApi, puuid: Puuid) -> LiveGame | None: ...
    async def history(self, api: RiotApi, puuid: Puuid, *, before_ms: EpochMs, n: int) -> list[PastGame]: ...
    def status(self, spec: DatasetSpec) -> MirrorStatus: ...
    def reproject(self) -> int: ...

# crawl.py
async def crawl(mirror: Mirror, riot: RiotGateway, spec: DatasetSpec, budget: CrawlBudget) -> CrawlReport: ...
```

### Sequence: kill -9 mid-crawl

1. On `europe`, fill A is inside `BEGIN…COMMIT`, fill B is waiting on HTTP, and fill C is queued in the lane. The process dies.
2. WAL recovery: A is either fully committed or absent. A partial row cannot exist.
3. Restart. The OS released the lock. The lanes start cold (defaults, 1 request in flight).
4. `gaps()` recomputes from the data. B and C are still gaps (and so is A, if it didn't commit). There is nothing to reset.
5. The first response carries `X-App-Rate-Limit-Count: 2:1,88:120`. The lane pads its log up to 88, which covers the dead process's calls, so the crawl doesn't trigger a 429.
6. Cost: at most 3 repeated calls on `europe` and 2 on `euw1`.

### Sequence: a 429

1. A notebook script uses the key outside the gateway. Locally the count is 60/95, but Riot's count is 100. A background `match` call gets 429 `application`, `Retry-After: 37`.
2. The lane freezes `europe` for 37 s plus jitter, halves the background pace, logs WARN ("external usage?"), and re-queues the request with the same seq.
3. The bot's spectator poll runs on `euw1` and is unaffected. A bot call on `europe` with an 8 s deadline gets `DeadlineExceeded(eta≈37)` immediately and answers in character.
4. When the freeze ends, interactive waiters go first, and background resumes at half pace. A third application-429 within 10 minutes opens the 5-minute breaker.

### Budget arithmetic

- **Throughput:** about 2,700 calls/h on `europe` with a CLI-only crawl, or about 2,100/h with the bot running.
- **Per anchor at `history_n = 10`:** 1 + 10 + up to 100 calls, in practice about 80–100 because histories overlap. That's about 25–35 complete rows/hour.
- **Owner's last 300 ranked games:** about 25–30k calls, roughly 10–12 h.
- **5,000 anchors:** about 2 weeks of `--forever`. `max_anchors` makes the bound explicit, and `oracle status` prints the estimate before you spend anything.
- **Live prediction:** 1 spectator call + 10 listings + up to 100 matches. Players already in the mirror cost about 1 call. When the crawler is idle this takes a few seconds; with a crawl running, up to about a minute in the worst case. → see open question 4.

## Synthesis decision

Two sketches were made independently, one by Opus and one by Sonnet, and a cross-judge (Sonnet) scored them. Both **converged** on the core shape: no job-queue table, one transaction per item as the checkpoint, raw JSON kept separate from the projection, an flock so one process owns the key, priority fixed by the handle, header-driven limits, and depth-bounded `endTime` history. The judge and I both picked **candidate 1 ("gaps, not queues")** as the base. It scored better on frequency, reactivity and leak-safety, because history listings are keyed by time and reused by the bot, and because of the `anchor_history` re-check and promotion to depth-1 anchors.

**Grafted from candidate 2:**
- Even background pacing, AIMD derating, the 429 breaker, and interactive in-flight slots.
- A PUUID-scoping guard (PUUIDs are encrypted per API app). Simplified on review: no stored key fingerprint. Before a crawl, re-resolve one tracked Riot ID; a different PUUID means a different app, and the crawl refuses.
- `parsed_version` for a selective reproject.
- A per-page `sync_cursor` (`page_offset`) instead of one big sync transaction.
- `api_call_stat` plus a `max_calls_per_day` cap across runs.
- A no-network cost estimate (folded into `oracle status`).
- STRICT tables, and **all times in milliseconds**, converted to seconds only at the `match_ids` call.
- Failure rows keyed by `(kind, k1, k2)` columns, so SQL anti-joins never build strings.
- The rank-snapshot rule "latest snapshot at or before the match start, else the nearest later one with a lag feature".

**Fixed, based on the judge's review:**
- Removed candidate 1's 10 s "quiet period", which idled the crawl about a third of the time.
- A 403 disables only its endpoint instead of tripping the whole gateway.
- Added single-flight dedup.
- Capped 429 retries per request.
- `history_n` now defaults to 10 to match CLAUDE.md's "last ~10 games".

**Rejected:**
- Candidate 2's `raw_doc.status='wanted'` stubs. They store work state a second time, next to the facts.
- Candidate 2's always-on 15 % reserve, which wastes budget when the bot isn't running.
- Candidate 2's depth-0-only targets, which give too few training rows.

## Tradeoffs accepted

- We accept running frontier SQL on every refill (tens of ms) in exchange for having no queue state that can drift from the data.
- We accept that a crash repeats up to the in-flight calls in exchange for no leases and no recovery code.
- We accept one listing call per (anchor, player), even for fully synced players, in exchange for one uniform, leak-free history path shared by training and the bot.
- We accept that one process owns the key (the bot hosts the crawler, so stop CLI crawls before starting the bot) in exchange for exact priority without IPC.
- We accept about 5 % of throughput lost to safety margins in exchange for essentially zero 429s.
- We accept storing raw compressed JSON plus a projection (≈5 KB per match, ≈100 KB per timeline) in exchange for re-parsing without re-crawling.
- We accept that rank snapshots are approximations, available only for recent anchors, in exchange for being honest about leakage. Running `--forever` builds rank coverage over time.
- We accept synchronous `sqlite3` on the event loop (each call takes milliseconds and never spans an `await`) in exchange for one dependency fewer.

## Alternatives considered

- **A durable job queue** (a `task` table with status, leases and acks). It's familiar from Spring Batch, but "done" would exist twice (the task row and the data). It needs lease sweeps after crashes, and queued tasks must be rewritten when the policy changes. It only pays off with several worker processes, which one shared rate budget rules out.
- **A player-centric snowball** (BFS over players, full history each). It has simple per-player checkpoints, but it spends most calls on history no row needs, and "last n before T" would need gap-free coverage for every player.
- **A wrapper library** (Pulsefire, Riot-Watcher). Neither is known to offer priority lanes with a reserve (Riot-Watcher is synchronous). Layering our priority on top of their limiter would put two limiters in one path. `gateway.py`'s HTTP call is the seam if we switch later. Teaching value also favours writing the limiter ourselves.
- **A separate limiter service** (Redis or a SQLite token table) so the bot and crawler can be separate processes. It adds IPC on every call and another moving part; `Lane` is the seam if we ever need it.

## Open questions (need the owner)

1. **Riot ID.** Region is decided: EUW (`RIOT_PLATFORM=euw1` in `.env`, regional `europe` derived). Which Riot ID(s) to track: just you, or friends too?
2. **`history_n`.** 10 by default. It drives crawl cost (~10·n calls per row), live cost, and how rich the form features are. It must be the same in training and in the bot. Is 5 (cheaper, faster live) or 20 (richer) better for you?
3. **Dataset bounds.** `max_depth = 1`, `max_anchors = 5000`, 120-day lookback (or "last K patches")?
4. **Live latency.** With a crawl running, a cold live prediction can take up to about a minute. Is that OK, or should the bot pause the crawler while you're in a game? (Easy: `crawl()` takes a pause event.)
5. **Timelines.** Only for your own games (default), or none for now?
6. **Dependencies.** Is `httpx` OK as the only runtime dependency, with `uv run --env-file .env` instead of `python-dotenv`? Dev-only: `pytest`.
7. **Process model.** Will the bot host the crawler (assumed)? If not, we need the limiter-service alternative.

**To verify empirically** (first spike): the personal key's real limits; whether `endTime` filters on game creation, start or end (the view re-checks either way); whether service 429s carry `Retry-After`; whether spectator `gameStartTime` is 0 during loading.

**Risk.** Frontier SQL at about 1M participant rows. The indexes cover the joins, but we should measure. If needed, we'll materialise `anchor.complete_at_ms`, owned by `Mirror`.

## Next implementation step

Implement `riot/_limiter.py` and `classify()` as pure logic with fake-clock unit tests (windows, header reconcile, reserve, pacing, freezes, breaker, deadlines), because a limiter bug is what gets a key blacklisted. Then write a 20-line spike that fetches one match through the limiter and prints the personal key's real rate-limit headers.

## Implementation notes (deviations from the sketch)

- **Built:** `config`, `ids`, `riot/` (endpoints, errors, `_limiter`, gateway), `mirror/` (gaps, `_frontier`, `_project`, mirror), `crawl`, `cli`, and 28 offline tests (`uv run pytest`).
- **Deferred to the bot phase:** `Mirror.live_game` and `Mirror.history`, the read-through helpers the bot will call. The gateway's `interactive()` handle and `RiotApi.active_game` already exist.
- **`oracle plan` merged into `oracle status`.** Both were no-network reports, so one command shows progress, gaps, a cost estimate and failures.
- **`RiotGateway.open` is synchronous.** Use it as `async with RiotGateway.open(settings) as gateway:`.
- **`SyncTracked` gaps carry their own lookback start and poll interval**, so `Mirror` holds no spec state.
- **Schema moved into the package** (`src/oracle/mirror/schema.sql`), so the installed code can always find it. `compose.yaml` mounts the same file.

## Adding players to track

- **Now:** `oracle track "Name#TAG"` (1 call: Riot ID → PUUID), `oracle untrack "Name#TAG"`, `oracle tracked`.
- **Later:** a Discord `/track Name#TAG` slash command. It will call the same `Mirror.track(api, riot_id)` with the bot's interactive handle, so there is one code path for tracking. Restrict it to the server owner or a role: every tracked player adds roughly 100 calls per game they play.
