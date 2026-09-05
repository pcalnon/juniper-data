# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **A rescue ladder for shares outstanding — 37 unresolvable tickers become 1.** The 2026-09-04
  census found 37 of 503 S&P 500 constituents returning nothing from either shares concept, so their
  `total_shares` and `market_cap` were zero-filled. **It was never missing data; it was the wrong
  endpoint.** SEC's `companyconcept` and `companyfacts` disagree for the same CIK, taxonomy and tag:
  for `KO`, `companyconcept` returns 636 bytes and **0 facts** while `companyfacts` returns **71**,
  with a current count of 4.30 billion shares.

  When `companyconcept` yields nothing, the generator now walks `companyfacts`:

  | Rung | Concept | Meaning | Rescues |
  |---|---|---|---:|
  | 1–2 | `dei:EntityCommonStockSharesOutstanding`, `us-gaap:CommonStockSharesOutstanding` | point-in-time | **18** |
  | 3 | `us-gaap:WeightedAverageNumberOfSharesOutstandingBasic` | **period average — not the same quantity** | **10** |
  | — | none | genuinely absent (`STZ`) | 1 |

  `companyfacts` costs ~1.15 s and ~5 MB against `companyconcept`'s ~0.20 s and ~600 B, so it stays a
  **fallback**: paying it per symbol would cut the 14-symbol cap to ~9.

  **Rung 3 is recorded, not hidden.** A market cap built on a period-average share count is a
  different quantity from one built on point-in-time shares, so those symbols are annotated
  `degraded` and logged — never silently mixed in with the rest.

- **`DatasetMeta.data_quality` — the permanent record of what is wrong with what is present.**
  Sibling of `truncation` (which says how much is *missing*), carrying `degraded`, `unrescued`,
  `rows_affected` and the `policy` applied. `None` when the dataset is clean, so presence alone
  answers the question. Travels the same reserved-channel route, popped before checksum and NPZ
  persist.

### Changed

- **A dataset whose fundamentals cannot be resolved is now REFUSED by default.** Previously
  `fundamentals_fill="zero"` turned an unresolvable share count into `total_shares = 0.0` and
  `market_cap = 0.0` — a value no listed company can have — and shipped it silently. The caller now
  takes an explicit path:

  | Gate | `incomplete_rows` | Outcome |
  |---|---|---|
  | unset *(default)* | — | **422**, naming the affected symbols, the row count, and both remedies |
  | set | `accept` *(default)* | rows kept and filled, dataset annotated |
  | set | `drop` | those symbols excluded, **dataset still annotated with what went** |

  The gate is the existing `allow_truncation` — request parameter,
  `JUNIPER_DATA_EQUITIES_ALLOW_TRUNCATION`, or the matching `.env` entry. `incomplete_rows` adds the
  accept-vs-drop refinement an interactive consumer needs, via
  `JUNIPER_DATA_EQUITIES_INCOMPLETE_ROWS`. Dropping every symbol fails rather than succeeding with an
  empty dataset.

- **Six `equities` feature columns that cost no extra request** — the matrix goes 10 → 16. Every one
  was already being downloaded and thrown away:

  | Column | Where it was already coming from |
  |---|---|
  | `adj_close` | parsed out of the Yahoo response, then dropped at the feature step |
  | `dividend`, `split_ratio` | arrive on the **same** call once `actions=True` is set |
  | `days_since_week52_high`, `days_since_week52_low` | fall out of the rolling window already computed |
  | `days_since_report` | the `filed` date already inside the SEC shares payload |

  **Order is part of the contract**: the existing ten keep their positions, so a consumer indexing by
  position is unaffected; the six are appended.

  The three underlying **dates** ship as their own row-aligned YYYYMMDD arrays —
  `week52_high_date_*`, `week52_low_date_*`, `report_date_*` — rather than as feature columns,
  because a raw date in a float32 matrix is a number whose magnitude means nothing. "Days since" is
  the form a model can use. Verified live against AAPL 2013–2021: 34 dividends, and both real splits
  (7:1 on 2014-06-09, 4:1 on 2020-08-31).

### Fixed

- **Look-ahead leak: shares outstanding were visible before they were filed.** The SEC series was
  aligned on the **period end** and forward-filled, so Apple's quarter ending 2021-03-27 — not filed
  until 2021-04-29 — reached every trade date in those five weeks, and `market_cap`, derived from it,
  inherited the leak. A live run surfaced the new `days_since_report` at **−19 days**; a negative
  filing age is the leak stating itself out loud.

  Alignment is now on the **filing date**. A point with no `filed` is **dropped rather than guessed**:
  when a figure became public is exactly what is unknown for it, and substituting the period end
  reinstates the leak. Same class as the normalisation leak fixed in juniper-data#314, and the same
  rule from decision 7 of the ecosystem partition design — no quantity that was not knowable at a
  row's date may reach that row.

- **A present-but-empty SEC concept suppressed the shares fallback.** SEC answers `200` with
  `{"units": {"shares": {}}}` for some filers — `KO` and `ABT` among them. The guard was
  `if payload and payload.get("units")`, and that dict is **truthy**, so the loop accepted the empty
  concept and **broke before trying the `us-gaap` fallback**. `BIIB` has 42 usable facts there and
  was getting none of them: `total_shares` and `market_cap` silently became `0.0` under the default
  `fundamentals_fill="zero"` — a value no listed company can have, and one nothing downstream
  distinguishes from a measurement. The guard now counts facts instead of testing truthiness.

- **A ticker with no shares data at all now says so.** Roughly 4–6% of the bundled S&P 500 universe
  reports no shares concept to SEC under either tag (`KO`, `ABT`, `ABNB`, `AMT`, `BG` confirmed).
  That is not a bug to fix upstream, but it was completely silent: the generator warned only when the
  fetch *raised*, never when it returned nothing. It now logs a warning naming the ticker and the
  `fundamentals_fill` policy that will paper over the gap.

- **`equities` is bounded to 14 symbols, and the silent slice is gone** (defect register
  `APD-DATA-018`, second half — the row is now closed). `EQUITIES_DEFAULT_MAX_SYMBOLS` was `None`,
  meaning **all 503 bundled S&P 500 constituents**, which measurement puts at **18–34 minutes**
  against a 30 s request budget. Worse, `generators/equities/generator.py` cut the universe with a
  bare `ordered[: params.max_symbols]` — it **truncated silently**, recorded nothing, and returned a
  dataset indistinguishable from a complete one.

  **The cap is in SYMBOLS, not bytes, and that is the load-bearing choice.** Measurement on
  2026-09-04 (`util/ad-hoc/2026-09-04_measure_equities_payloads.py`) found **163× the payload costs
  1.16× the time**, because ingest cost is per *request*: ~1.85 s per Yahoo call plus 1–2 SEC XBRL
  calls. One symbol over 26 years is 210 KB and ~2 s; the Russell 3000 over **one day** is 92 KB and
  1.7–3.2 h. A byte cap would admit the expensive request and reject the cheap one — it is not
  merely the wrong unit, it runs *counter* to the cost.

  **14 = 30 s ÷ 2.1 s per symbol.** Same contract as `csv_import`: an oversized universe is
  **refused with 422** until the caller opts in via `allow_truncation`,
  `JUNIPER_DATA_EQUITIES_ALLOW_TRUNCATION`, or the matching `.env` entry; a request may only *lower*
  the cap (`min(requested, deployment)`), and `max_symbols=None` means "no request-side limit", not
  "unbounded". The surviving prefix is deterministic, so a truncated dataset is reproducible.

  **`equities_seq` inherits all of it**, because it reuses the same universe resolution — including
  the annotation, since inheriting a bound while dropping the record of it is the worse half to skip.

- **The truncation descriptor is now one shape for every generator.** `DatasetMeta.truncation`
  carries `reason`, `unit` (`bytes` or `symbols`), `cap`, `requested`, `imported` and
  `records_imported`, replacing the byte-specific `cap_bytes` / `bytes_total` / `bytes_read` of the
  entry below. A consumer can now answer "is this partial, and by how much" without knowing which
  generator produced it. Unreleased-to-unreleased change; no published artifact carries the old keys.

- **`csv_import` sources are bounded, and an over-cap import must be authorised** (defect register
  `APD-DATA-018`). Generation runs inside the request, so a source large enough to outlive the
  client's socket timeout produces a request that cannot succeed however long the caller waits. The
  remedy is to bound the input rather than move generation to an async job.

  **The cap is 128 MiB, and it is measured rather than picked.**
  `util/ad-hoc/2026-09-04_measure_csv_import_throughput.py` timed the whole `generate()` path — the
  parse *and* the per-cell float conversion, which is a large share of the cost — at a median
  **14.4 MB/s** (15.3 MB/s at 3.5 MB, degrading to 13.6 MB/s at 35 MB). 128 MiB is therefore ~8.9 s
  of parsing, inside the ~30 s client budget with room for split, checksum and NPZ persist. Above
  that size the binding constraint stops being time and becomes memory: `_load_csv` materialises one
  Python dict per row before any array exists.

  **An oversized source is refused, not silently truncated.** `POST /v1/datasets` answers **422**
  naming the source size, the cap, and the remedy. 422 is deliberate — it is already on this API's
  surface, so no new status code is introduced.

  **Truncation is opt-in, through three surfaces.** `allow_truncation` as a request parameter, the
  `JUNIPER_DATA_CSV_IMPORT_ALLOW_TRUNCATION` environment variable, or the matching `.env` entry. The
  cap itself is `max_bytes` / `JUNIPER_DATA_CSV_IMPORT_MAX_BYTES`.

  **The bound cannot be defeated by the party it bounds.** A request may only *lower* the cap — the
  effective value is `min(requested, deployment)` — so `max_bytes: 10000000000` cannot skip it, and a
  generated client that serialises schema defaults cannot silently raise a lower operator ceiling.
  `allow_truncation` is a logical OR for the same reason in the same direction: a client cannot opt
  *out* of an operator's choice. **`stat` is only a cheap pre-check, never the bound** — the read
  itself is capped and re-checks what it actually got, so a FIFO (which reports `st_size == 0`) or a
  file that grows between the stat and the open cannot slip past. `JUNIPER_DATA_CSV_IMPORT_MAX_BYTES`
  is constrained `> 0`, because Python reads `read(n)` with `n < 0` as "read everything" — a mistyped
  environment variable would otherwise invert the cap into its opposite.

  **An authorised truncation is recorded permanently.** `DatasetMeta.truncation` carries
  `truncated`, `reason`, `bytes_read`, `bytes_total`, `cap_bytes` and `records_imported`, and is
  persisted with the artifact — so a trainer loading the NPZ months later still learns the data is a
  prefix of its source. It is metadata, not a transient warning. `None` means complete, so the
  field's presence alone answers the question.

  **The cut lands on a record boundary.** A byte cap slices mid-record essentially always, so the
  reader trims to the last newline, drops a partial multi-byte character, and discards a final row
  left short — a newline *inside a quoted field* is legal CSV, so the newline trim alone is not
  sufficient. A truncated JSON array decodes as many complete elements as fit rather than failing;
  JSONL drops only an unparseable **final** line, so a corrupt line mid-file still raises instead of
  importing as a short dataset.

### Fixed

- **The Postgres store carried five hand-maintained copies of `DatasetMeta`'s field list, and every
  one had drifted.** The DDL, `_meta_to_row`, `_row_to_meta`, the upsert and the update each
  transcribed the same 30 fields independently. All five are now **derived from the model**, so a
  new field appears in the schema, the INSERT, the UPDATE and the read without anyone remembering
  to add it.

  Three defects fell out of that drift, all latent because the store is not yet wired into the
  serving path:

  - `n_classes` and `class_distribution` were declared `NOT NULL` long after the model made both
    nullable for regression and sequence artifacts (WS-1 / #168) — **the first regression dataset
    written would have failed the INSERT.**
  - Seven fields were silently dropped on every round trip: `task_type`, `sequence`, `lookback`,
    `time_unit`, `dt_scaling`, `target_scaling` and `truncation`. (The issue named six; `truncation`
    was missed there too.)
  - `_row_to_meta` raised `TypeError` on a NULL `class_distribution`, because its
    `isinstance(..., dict) else json.loads(...)` fallback did not admit `None`.

  `_row_to_meta` now also tolerates a **column that is absent or NULL** where the model supplies a
  default, so a code-ahead-of-schema deploy loads rather than raising `KeyError`. Added columns are
  emitted as `ADD COLUMN IF NOT EXISTS`, and any that is `NOT NULL` carries a `DEFAULT` — without
  one, an `ADD COLUMN` against a populated table fails.

  **No test asserted `_row_to_meta(_meta_to_row(m)) == m`**, which is why five copies could drift
  apart unnoticed. That round trip is now pinned for classification, regression and sequence
  artifacts, alongside schema and statement coverage of every model field.

- **`juniper_data.generators.csv_import` could not be imported on its own.** A circular import
  raised `ImportError: cannot import name 'VERSION'`:

  ```
  csv_import/__init__ -> csv_import.generator  (imports juniper_data.api.settings)
                      -> juniper_data.api/__init__  (eagerly imported .app)
                      -> api.app -> api.routes.generators
                      -> juniper_data.generators.csv_import   # still initialising
  ```

  Importing a submodule initialises its parent package first, so the ordinary leaf import
  `from juniper_data.api.settings import get_settings` dragged the entire FastAPI app and every
  route into the cycle.

  Fixed by exposing `create_app` **lazily** from `juniper_data/api/__init__.py` via a module
  `__getattr__` (PEP 562). `Settings` and `get_settings` stay eager — they are leaves. Nothing
  that merely wants settings pulls the routes in any more, so this closes the whole class rather
  than only the `csv_import` instance. `from juniper_data.api import create_app` is unchanged.

  The defect was **masked in service use** — app startup imports the routes long before anything
  touches `csv_import` — and surfaced only when the subpackage was imported *first*, which is what
  a test, script or external consumer does. It broke pytest collection of
  `test_csv_import_generator.py` in isolation.

  A new `tests/unit/test_no_import_cycles.py` asserts every one of the 16 registered generator
  subpackages imports standalone, and that importing `api.settings` does not load the route
  modules. All of it runs in **subprocesses**: a cycle is a property of a cold interpreter, and an
  in-process assertion would pass with the defect fully present.

- **The feature normaliser was fit on the full set, including chronologically-later test rows.**
  `equities`, `equities_seq` and `csv_import` computed min/max over every row and then applied
  those statistics to the training features — look-ahead leakage, with the training data scaled by
  a maximum that exists only in its future. The fit now comes from the **training rows only**,
  which is decision 7 of the ecosystem partition design.

  `csv_import` needed a structural change rather than a one-line one: it normalised inside
  `_load_and_preprocess`, which runs *before* the split exists, so a train-only fit was not merely
  wrong there but impossible. It now splits first and fits on `X_train`.

  **Deliberate consequence:** `X_test` and `X_full` are no longer bounded by `[0, 1]`. Only the
  fitted partition is. Rows outside the training range legitimately fall outside it, and that
  excursion is real out-of-sample movement — scaling it away is precisely what the leak was doing.

  Two existing tests asserted the `[0, 1]` bound on `X_full` and so were pinning the leak in place
  without naming it; both now assert the bound on `X_train`, and a new discriminating test asserts
  that `X_test` **escapes** the bound — an assertion a full-matrix fit cannot satisfy.

  This is opt-in: `normalize_features` defaults to `False`, and the one persisted equities artifact
  carries `false`.

- **Nine generators were not reproducible at their documented defaults.** `checkerboard`,
  `circles`, `gaussian`, `moon`, `xor`, `csv_import`, `mnist`, `arc_agi` and `equities` declared
  `seed: int | None = Field(default=None)`, so two identical calls at the default configuration
  produced different data — and `shuffle_and_split` re-drew the partition boundaries from OS
  entropy as well. `spiral` was the only 2-D generator with a concrete default; its value (42) is
  now the shared one.

  The default resolves from `JUNIPER_DATA_DEFAULT_GENERATOR_SEED` at import time, falling back to
  the compiled-in `DEFAULT_GENERATOR_SEED`, so a deployment can pin it without editing code.
  Import-time resolution rather than a pydantic `default_factory` keeps the value visible as a
  concrete `default` in `model_json_schema()`, which published clients read. A malformed, negative
  or empty override falls back rather than raising — a configuration error must not make the
  package unimportable.

  **Passing `seed=None` explicitly is unchanged**: it still opts into a fresh draw per call, and
  still receives the BUG-JD-04 cache nonce in `generate_dataset_id` so a seedless request cannot
  collide with a stale seedless artifact. Only the *default* moved.

  `mackey_glass`'s `init_noise_std` gets the same treatment via
  `JUNIPER_DATA_DEFAULT_MACKEY_GLASS_INIT_NOISE_STD`. That knob decides whether that generator's
  `seed` has any effect at all — the seed is consumed only inside `if init_noise_std > 0`, so at
  the default every seed produces byte-identical output. Documented and intentional; exposing the
  knob is what lets a deployment make the generator seed-sensitive without editing code.

  Note `equities` is a special case: its seed is *"unused for the temporal split; retained for API
  parity"*, so defaulting it there is cosmetic. Its real non-reproducibility source is untouched by
  this change — `end_date` defaults to the wall clock, so the same params yield different data on
  different days.

- **`arc_agi` returned an empty dataset, silently, in over ten minutes.** Its Hub source
  `fchollet/arc-agi` does not exist — and was never verified to: it was hardcoded when the
  generator was introduced and never changed, and every test patches the Hub loader, so no test
  ever resolved it live. The fallback `multimodal-reasoning-lab/ARC-AGI` is a
  multimodal reasoning-trace dataset — columns `Question` / `Text Reasoning Trace` / `Final Answer`
  plus 46 image columns, with no `train` or `test`. The parse used `item.get("train", [])`, so all
  2000 rows produced empty tasks and `generate()` returned `X_full` with shape `(0, 900)`: a valid,
  entirely empty dataset, after decoding 17,232 images that were then discarded.

  Nothing rejected it. juniper-data has no empty-dataset check in its API or core layers, so the
  artifact would have been persisted, content-addressed, and served to a trainer as real data.

  **Fixed in three parts.** The source moves to `lordspline/arc-agi` split `training`, which
  carries the canonical ARC schema — note the split is `training`, not `train`, so the old request
  would have failed here too; the pair is now pinned by a test. The unusable fallback is
  **removed**: one that cannot satisfy the parse is worse than none, because it converts a loud
  outage into silent bad data. And two guards were added — a schema check that reads the declared
  columns *before* parsing any row, so a mismatch names its cause, and a non-empty backstop in
  `generate()` that catches the class regardless of why the arrays came back empty.

  **Measured: `> 600 s` → 1.30 s cold (0.67 s warm), and `(0, 900)` → `(1717, 900)`.** Both figures
  in the pair are cold runs; an earlier note compared a warm pre-fix run against a cold post-fix
  one. The old runtime was **not** "almost entirely image decoding": a ~1.09 GB parquet download
  was a substantial share of it, and only 17,232 of the 92,000 image cells were populated — 2000
  rows × 46 columns is the cell count, not a count of images decoded.

  **A checked-in test asserted the defect as correct** and is inverted here, with the reason
  recorded at the test: `test_generate_local_missing_subdirs` asserted that an empty `local_path`
  yields a zero-sample dataset and called that handling the case "gracefully". A caller who typos
  the path now gets an error instead of an empty dataset.


- **`GET /{dataset_id}/artifact` streams the NPZ instead of materialising it** (defect-register
  `APD-DATA-016`). The route returned a `StreamingResponse` wrapping
  `io.BytesIO(get_artifact_bytes(...))`, which bounds the **socket buffer**, not process memory:
  the whole artifact existed in RAM before the response object did, once per concurrent download,
  while the endpoint's name invited the opposite assumption.

  New `DatasetStore.open_artifact_stream(dataset_id, chunk_size=...)` yields the artifact in
  chunks. **It is deliberately not abstract**: the base implementation falls back to
  `get_artifact_bytes`, so all seven existing backends keep working untouched and adding a new one
  does not get harder. `LocalFSDatasetStore` overrides it with a real chunked file read, so the
  memory bound is a property of the *store* rather than of the route.

  **Absence is decided eagerly, before the generator is returned.** A generator body does not run
  until first iteration, so an existence check placed inside one would defer the 404 until after
  the route had already committed to a 200 and sent headers — it would answer *200 with an empty
  body* for a missing dataset. A test pins this by asserting the call returns `None` rather than a
  generator; the mutation that moves the check inside fails exactly that arm.

  Not a wire change: the delivered bytes are identical, verified against the materialised read.
  Chunk size is `ARTIFACT_STREAM_CHUNK_SIZE` (1 MiB).

  **Out of scope, deliberately:** the `ETag` / conditional-GET half of the same primer passage
  belongs to `APD-DATA-017`, which is part of the owner-routed REST group and is not touched here.
  The `application/octet-stream` half was already fixed — `BINARY_MEDIA_TYPE` is `application/zip`
  and owns every call site.

### Changed

- **`juniper-service-core` ceiling raised to `<0.8.0`** so 0.7.0 can be adopted. 0.7.0 re-files
  juniper-ml#1332 under `Added` — it introduces `WorkerCoordinator.release_worker_tasks`, which
  reclaims a worker's in-flight tasks on a clean `/ws/workers` disconnect or a mid-result abort
  instead of waiting out `task_reassignment_timeout`. juniper-data does not import
  `juniper_service_core.workers` or `.websocket`, so this is a ceiling raise for adoptability, not
  a behaviour change here. The lockfile still pins `==0.6.0` and is refreshed separately once 0.7.0
  publishes — constraint-mode `Lockfile Freshness` asks only whether the lock still *satisfies*
  pyproject, so it stays green while stale and nothing prompts the adoption.

## [0.12.0] - 2026-08-30

### Added

- **`GET /v1/generators` carries the `install_hint` it is documented to carry (`#277`, plan W-4).**
  `GeneratorInfo` declared five fields and no hint, while juniper-ml's experiment driver refused an
  unavailable generator by telling the operator to "see `GET /v1/generators` for the install hint".
  The endpoint now returns it, so the driver's message resolves to something real instead of
  pointing at a field that did not exist.
- **The rate-limit window is operator-configurable (`APD-DATA-033`, `#297`).** `RateLimiter` has
  always accepted `window_seconds` and uses it in three places, but `create_app` passed only
  `requests_per_minute` and `enabled`, so the window was the one knob of three an operator could
  not set. Adds `Settings.rate_limit_window_seconds`, overridable as
  `JUNIPER_DATA_RATE_LIMIT_WINDOW_SECONDS`, defaulted from the same
  `DEFAULT_RATE_LIMIT_WINDOW_SECONDS` object the constructor uses rather than a second copy.
  Worth restating what this was **not**: that constant was never unwired — it is load-bearing at
  all three sites. The gap was only the absent operator-facing setting.
- **Total ordering and keyset pagination on `/v1/datasets/filter` (`APD-DATA-011`,
  `APD-DATA-012`, `#283`).** `filter_datasets` sorted on `created_at` alone; `list.sort` is stable,
  so rows sharing a timestamp came back in whatever order enumeration produced — and
  `LocalFSDatasetStore` enumerates with `Path.glob`, which specifies no ordering. Measured: the
  same six datasets fed in two insertion orders produced exactly reversed pages. Now sorted by
  `created_at` descending with `dataset_id` ascending as the tie-break, reproducible across calls,
  processes, and both store implementations.

### Security

- **The OpenAPI document is served behind the API key, and declares the scheme
  (`APD-DATA-005` + `APD-DATA-024` — one defect wearing two faces).** `openapi_url` was `None`
  whenever any key was configured, so a **secured deployment served no schema at all**; and
  `api_key_header` was instantiated but referenced nowhere, so what it did serve declared no
  `securitySchemes` — an SDK generated from it never sent `X-API-Key`. Neither was fixable alone:
  adding the scheme to a document nobody can fetch changes nothing.
  **The trap worth knowing about:** `EXEMPT_PATHS` already listed `/docs`, `/openapi.json` and
  `/redoc`, and `SecurityMiddleware._is_exempt()` is a bare `path in EXEMPT_PATHS` evaluated
  *regardless of whether a key was supplied*. So simply re-enabling `openapi_url` would **not**
  have put the document behind the key — it would have served it to **everyone**, while looking
  exactly like the intended fix. All three paths were therefore removed from the exempt set, and
  the document is now authenticated like any other route.
  The interactive explorers (`/docs`, `/redoc`) remain unmounted when keys are configured: they
  are browser pages that fetch `/openapi.json` by XHR with no `X-API-Key` header, so serving them
  behind the key could only 401. Local development, where no keys are set, is unchanged.
  The scheme is declared **per protected router rather than app-wide**, so the document stays
  honest — the health probes are genuinely exempt and are not documented as needing a key.
  `auto_error=False` keeps enforcement in `SecurityMiddleware`; the declaration only describes it.

- **CR-024 request-body cap is no longer bypassable (`APD-DATA-002`).** `RequestBodyLimitMiddleware`
  checked only `if content_length is not None and int(content_length) > max_bytes`, so a chunked
  request that sent no `Content-Length` made the first conjunct false and streamed past the 10 MiB
  cap entirely — an unauthenticated memory-exhaustion vector wherever the service runs in the open
  bare/dev profile. `POST`/`PUT`/`PATCH` are now always stream-read against a cumulative cap and
  aborted with 413 mid-stream, with `Content-Length` demoted to an early-reject hint; an
  under-declared length no longer buys passage either. The fully-read body is cached on
  `request._body` so downstream handlers still parse it normally. This restores parity with the
  implementations already shipping in `juniper-cascor` and `juniper-service-core`, where the same
  fix landed and never propagated here.

### Changed

- **Every route declares an explicit `operation_id` (`APD-DATA-023`).** FastAPI derived the
  `operationId` of all 21 operations from the handler name, the full path and the method
  (`create_dataset_v1_datasets_post`), so a handler rename, a router move or a prefix change
  silently renamed every generated-SDK method. Each decorator now carries
  `operation_id="<handler name>"` (`create_dataset`, `download_artifact`, `readiness_probe`, …),
  decoupling the published id from the Python name. This is the one-time change of every published
  `operationId` from the generated shape to the explicit one; no generated SDK exists in the
  ecosystem (`juniper-data-client` is hand-written), so no consumer is affected.
  `tests/unit/test_operation_ids.py` pins the ids at the call site (AST), on the registered routes
  and as a frozen contract in the published document — renaming a handler must not rename its id.

- **Both binary routes declare `application/zip` (`APD-DATA-025`).** `GET /v1/datasets/{id}/artifact`
  answered `application/octet-stream` while `POST /v1/datasets/batch-export` answered
  `application/zip` — two unrelated inline literals. Both payloads are ZIP containers (an NPZ
  artifact is numpy's zip of `.npy` members; the export is a zip of NPZ files), and
  `application/octet-stream` is only the RFC 9110 §8.3 fallback a recipient may assume when no
  `Content-Type` is present at all, so the artifact route now says `application/zip` too. The one
  spelling lives in `api/constants.py` as `BINARY_MEDIA_TYPE`; both routes derive from it, and the
  `Content-Disposition` filenames (`<id>.npz` / `datasets.zip`) are unchanged. **Wire-visible**
  only in the artifact response's `Content-Type` header; `juniper-data-client` returns
  `response.content` without reading it. `tests/unit/test_binary_media_types.py` pins the
  published value, every `media_type=` call site (AST) and the wire — including that the bytes
  really are a zip.

### Fixed

- **A malformed `Content-Length` is a 400, not a 500 (`APD-DATA-036`).** The unguarded
  `int(content_length)` raised `ValueError` inside a `BaseHTTPMiddleware.dispatch`, which propagates
  *outside* `ExceptionMiddleware` — so the app's own `ValueError` handler never saw it and a bad
  client header surfaced as a server fault. It now returns an explicit 400
  `{"detail": "Invalid Content-Length header"}`, matching both sibling implementations.

- **Serialisation faults are reported as `500`, not `400` (`APD-DATA-034`).** The blanket
  `@app.exception_handler(ValueError)` returned `400`, and `pydantic_core.PydanticSerializationError`
  subclasses `ValueError` — so when the app failed to serialise its *own* response, the caller was
  told they had sent a bad request. The defect was invisible to 5xx alerting, misattributed to the
  client, and stripped of its diagnostic by the generic `"Invalid request parameters"` message.
  Unlike `juniper-cascor`, juniper-data has no `coerce_native_scalars` helper pre-empting the common
  numpy-scalar case, so every such fault landed here. `PydanticSerializationError` is now classified
  as the 500 it is and logged at exception level so the traceback survives; a plain `ValueError` is
  unchanged and still returns `400`.

- **A `GET` can no longer silently undo a tag edit (`APD-DATA-006`).** `record_access` — which fires
  on every metadata read and every artifact download — rewrites the *whole* `DatasetMeta` document
  under `_version_lock`, while `PATCH /v1/datasets/{id}/tags` did its own read-modify-write across
  two `asyncio.to_thread` hops taking no lock at all, so the lock protected nothing against it. A
  plain `GET` could interleave between those hops and write back a pre-edit snapshot, discarding the
  tag change with no error anywhere — a silent lost write triggered by a *safe* method. New
  `DatasetStore.update_tags` does the whole read-modify-write inside that same lock in one thread
  hop, and the route delegates to it. The per-process caveat is unchanged (BUG-JD-05): this
  serialises threads within one process, not across a multi-process deployment.

## [0.11.0] - 2026-07-28

### Added

- **SEC-F01 boot-time auth-posture self-check (HO-2 class)**: the API lifespan now calls `enforce_auth_posture(settings.api_keys, require_auth=False, service_name="juniper-data")` from the new `juniper-service-core>=0.5.0,<0.6.0` dependency (`[api]` extra). An empty/blank `JUNIPER_DATA_API_KEYS` secret now produces a loud "running OPEN" WARNING at startup instead of silently serving unauthenticated behind a healthy health check. Escape hatch: `JUNIPER_SKIP_AUTH_POSTURE_CHECK=1`. A `JUNIPER_DATA_REQUIRE_AUTH` fail-closed flag is proposed as the owner-approved follow-up.
- **SEC-F01 fail-closed flag `JUNIPER_DATA_REQUIRE_AUTH`** (`settings.require_auth`, default `false`): when `true`, the boot-time auth-posture check REFUSES startup (CRITICAL + `AuthPostureError`) if no real API key is configured, instead of only warning — the intended posture wherever secrets are provisioned (the composed juniper-deploy stack). The lifespan now passes `require_auth=settings.require_auth`.

## [0.10.0] - 2026-07-17

### Added

- **`[mnist]` extra ships real MNIST / Fashion-MNIST generation (plan unit D2, juniper-ml
  training-runtime-defects plan §4 I-5).** New optional dependency group `mnist = ["datasets[vision]>=4.0.0"]`
  (Hugging Face `datasets` + Pillow for image decode); included in `[all]`. The `requirements.lock`
  compile surface gains `--extra mnist` (lockfile-update workflow, CI freshness gate, and all
  documented copies of the command), so the service Docker image now ships the chain and MNIST
  generation works in containers out of the box. The image pins `HF_HOME=/app/data/hf-cache` so a
  mounted data volume can persist (or pre-seed, for offline deployments) the Hub download cache.
  Added a real-generation integration test (tiny `n_samples`, slow-marked) that asserts the NPZ
  contract shapes — `(n, 784)` float32 X, `(n, 10)` one-hot y — and skips cleanly when `datasets`
  is not installed or the Hub is unreachable with no cache.

- **Generator availability surfaced (D1 / I-5 of the training-runtime defects plan).** Every entry in
  `GET /v1/generators` and every `GET /v1/generators/{name}/schema` response now carries an additive
  `available: bool` reporting whether the generator's optional dependencies are importable in the
  running deployment. Generator classes may declare an `is_available()` hook: `mnist` reports the HF
  `datasets` package, `equities` / `equities_seq` report the `equities` extra; generators without the
  hook (the numpy-only synthetics, and `arc_agi` whose HF need is parameter-conditional with a
  local-file fallback) default to available.

### Fixed

- **MNIST generator now loads the canonical namespaced Hub repositories** (`ylecun/mnist`,
  `zalando-datasets/fashion_mnist`). Bare canonical names (`load_dataset("mnist")`) are rejected by
  the huggingface-hub 1.x URI layer used by `datasets>=5` ("Repository id must be
  'namespace/name'"), so the stable `MnistParams.dataset` values (`"mnist"` / `"fashion_mnist"` —
  the public parameter surface is unchanged) now map internally to the namespaced repos, which load
  identically on older `datasets` versions. Generator `VERSION` bumped 1.0.0 → 1.0.1.

- **`POST /v1/datasets` no longer masks a missing optional dependency as a generic 500.** A generator
  raising `ImportError` (e.g. `mnist` without the HF `datasets` package — observed live as 71 identical
  masked 500s) now returns **`501 Not Implemented`** with the generator's actionable install hint in
  `detail` (`pip install datasets`). The unknown-generator and invalid-params 400 contracts and the
  generic 500 handler for genuinely unexpected errors are unchanged. See juniper-ml
  `notes/JUNIPER_2026-07-11_JUNIPER-CANOPY_TRAINING-RUNTIME-DEFECTS-PLAN.md` §4 I-5.

### Changed

- **CI: per-file coverage is now a blocking gate (ecosystem per-file rollout C-5).** The unit-tests
  job's advisory per-module step — `scripts/check_module_coverage.py`, which only *warned* on modules
  below 85% — is replaced by the shared `juniper-coverage-gap-map --enforce` gate from
  `juniper-ci-tools>=0.6.0,<0.7.0`. CI now **fails** when any source file's statement coverage is
  below 90% or any packaged sub-module's pooled (statement-weighted) coverage is below 95%. The gate
  computes statement % itself from `reports/coverage.json`, so the `branch = true` coverage config
  does not change the gate basis. See juniper-ml
  `notes/JUNIPER_ECOSYSTEM_PER_FILE_COVERAGE_ROLLOUT_SCOPING_2026-06-30.md`. `check_module_coverage.py`
  is retained (still wired into the pre-push hook and `util/run_coverage.bash`).

### Tests

- **Lifted `unit and not slow` lane coverage of `juniper_data` from 86.8% to 98.3%** to satisfy the
  new gate, with no production-code changes. Existing but unmarked genuine unit tests were brought
  into the lane (`test_checkerboard/circles/csv_import/gaussian/moon_generator.py`,
  `test_cached_store.py` — the `@pytest.mark.unit` marker was missing so CI's `-m "unit"` selection
  silently excluded them), and new deterministic (no-sleep) tests were added for the API security
  module, the batch dataset endpoints, the sequence-windowing validation branches, the equities /
  windowed-equities generators' fetch/cache/conditioning helpers (SEC + yfinance mocked, offline),
  and the package `__init__` arc-agi import-fallback branch.

## [0.9.0] - 2026-06-22

### Added

- **New `delay_product` synthetic time-series generator (DP-3 capacity instrument).** An
  irregularly-sampled sinusoid superposition (the same non-uniform Δt sampling as `irregular_sine`)
  whose regression target is the **bilinear product of two delayed in-window values**,
  `y = x(t−τ₁)·x(t−τ₂)`, with `lag1` / `lag2` step-delays kept strictly inside the lookback. The
  product is a quadratic form in the (linear) LMU memory state, so a **linear readout provably cannot
  fit it** (r² bounded below 1) while a **non-linear (random-Fourier-feature) readout can** — making
  it the capacity-demonstrating dataset that exposes a clear nonlinear ≫ linear r² gap, complementing
  the near-linear synthetics where the linear readout is already at its ceiling. Emits the standard
  additive 3-D NPZ contract (`{X, y, dt, target_dt, observed_mask}_{train,test,full}`,
  `task_type="regression"`, `time_unit="steps"`) and reuses the leakage-safe `window_timed_series`
  windowing (the target reads only the emitted window contents; `y_full == concat(train, test)`).
  Registered as `delay_product`; numpy-only, no extra. See juniper-ml
  `notes/JUNIPER_RECURRENCE_DP3_READOUT_SPECTRUM_DESIGN_2026-06-20.md` §8a.

## [0.8.0] - 2026-06-19

### Added

- **Configurable equities regression target (`regression_target`).** `EquitiesParams` (inherited by
  `EquitiesSeqParams`) gains a `regression_target: "next_close" | "return" | "log_return"` option
  controlling the `y_reg_*` array representation:
  - `next_close` (default) — the raw next-day close price, **byte-identical** to prior output;
  - `return` — the simple next-day return `next_close / close - 1`;
  - `log_return` — the log next-day return `ln(next_close / close)`.

  The raw close is non-stationary (it trends with the price level), which a bounded-memory recurrent
  regressor extrapolates poorly; the return variants are stationary — the standard conditioning for
  forecasting on trending price data. Both the flat `equities` and the windowed `equities_seq`
  generators honor it via a shared `EquitiesGenerator._regression_target` helper. The one-hot
  `direction` target, the feature matrix, and every other array are unchanged, and the default keeps
  all existing artifacts byte-identical. Motivated by the juniper-recurrence Δt-LMU `equities_seq`
  finding (raw-close target → r² ≈ −50; see juniper-ml
  `notes/JUNIPER_RECURRENCE_EVALUATION_FINDINGS_2026-06-18.md`).

## [0.7.1] - 2026-06-19

### Fixed

- **Ship the equities S&P 500 constituents CSV inside the wheel.** 0.7.0 packaged only `*.py`, so
  `sp500_constituents.csv` was absent from the built wheel and the `equities` / `equities_seq`
  generators raised `FileNotFoundError` from a pip install of `juniper-data[equities]==0.7.0`. Adds
  a `[tool.setuptools.package-data]` entry (`juniper_data.generators.equities = ["*.csv"]`) so the
  constituents list ships, plus a CI build-step assertion that the CSV is present in the built wheel.
  No API change.

## [0.7.0] - 2026-06-19

### Added

- **Advisory `dt` / `target` scaling meta channel** (juniper-data#179 §A; Δt
  note §6.5): a generator may now report *how* its per-step `dt` and regression
  target should be standardized, via a reserved `"scaling"` key in the
  `generate()` return dict that the dataset route pops into two new optional
  `DatasetMeta` fields — `dt_scaling` and `target_scaling` (JSON-safe
  descriptors, e.g. `{"method": "standardize", "mean": …, "std": …, "min": …,
  "max": …}` or `{"method": "identity"}`; `target_scaling` is keyed by
  target-array name). The scaling is **advisory** — the NPZ keeps RAW arrays
  (every contract invariant, e.g. `dt[:, 0] == 0`, stays intact), so a consumer
  standardizes at ingestion and denormalizes for metrics using the persisted
  stats; nothing in the NPZ is transformed. New `core/scaling.py`
  (`standardize_descriptor` + the exact-inverse `standardize` /
  `inverse_standardize`, with a std≈0 guard for constant data) and
  `core/meta.py::pop_scaling_meta`. The four synthetic generators gain a
  `scaling: "identity" | "standardize"` parameter (default `identity`; the
  `standardize` descriptors are fit on the **train split** only, no test
  leakage). These stats are not derivable from the final arrays, so this is the
  non-derivable generator→meta path deferred from WS-1's lean scope; it closes
  the WS-1 §B denorm round-trip acceptance check. Adds `core/scaling.py` unit
  tests, a `pop_scaling_meta` channel test, a parametrized synthetic emission +
  denorm round-trip suite, and e2e route assertions.

- **Irregular-Δt synthetic generator** (`irregular_sine`): a fourth numpy-only
  regression generator (juniper-data#179 §A) that samples a continuous-time
  sinusoid superposition at **non-uniform** times — each inter-sample gap is
  `sample_dt · U[1 − jitter, 1 + jitter]` — so the windowed artifact carries a
  **genuinely non-uniform** per-step `dt` and a variable `target_dt`. It is the
  synthetic, offline, known-answer counterpart to `equities_seq`'s calendar-gap
  irregularity, exercising the irregular-Δt contract independently of real market
  data (the signal stays closed form at each irregular sample time, so it is
  deterministic given the seed). Backed by a new `window_timed_series(values,
  times, …)` helper (the irregular-Δt sibling of `window_regular_series`, deriving
  per-step `dt` from explicit sample times); `window_regular_series` is unchanged.
  New unit tests (non-uniform-`dt` contract, closed-form known answer at irregular
  times, `jitter` control, determinism), a `window_timed_series` Hypothesis
  property test, and an added `irregular_sine` end-to-end route +
  contract-validator case.

- **Synthetic time-series regression generators** (`multi_sine`,
  `mackey_glass`, `ar_p`): three numpy-only, deterministic, offline
  generators that emit the additive 3-D sequence NPZ contract (WS-1 /
  juniper-data#168) as `task_type="regression"` — the recurse CLI
  "hello-world" datasets ([OQ-5], juniper-data#179 §A). Each samples a
  process at a regular Δt and windows it into `(W, L, 1)` sequences with a
  per-step `dt`, a fixed `target_dt` forecast horizon, an all-ones
  `observed_mask`, and the regression target carried directly in `y_*`
  (no one-hot, so `compute_shape_meta` leaves `n_classes` /
  `class_distribution` None). `multi_sine` is a superposition of K sinusoids
  (closed-form known answer when noise-free); `mackey_glass` integrates the
  canonical chaotic delay-differential equation (β=0.2, γ=0.1, n=10, τ=17)
  by a discrete Euler scheme; `ar_p` is an autoregressive process with
  Gaussian innovations (default stable AR(2)). A new `window_regular_series`
  helper (the regular-Δt sibling of `window_one_ticker`) and a shared
  `SyntheticSequenceParams` base back all three. Unlike `equities` these
  need **no optional extra** (pure numpy) — the zero-dependency smoke
  datasets for the 3-D contract. Registered in the dataset route with
  `time_unit="steps"`; closes the WS-1 §B "pure-regression generator
  traverses the route end-to-end" acceptance check. Adds per-generator unit
  tests (determinism + known-answer + contract), a `window_regular_series`
  Hypothesis property test, and an end-to-end route + client-contract
  integration test.

- **Build provenance on `/v1/health` + `/v1/health/ready`.** The service now
  reports the source `git_sha` and ISO-8601 `build_date` baked into its image
  at build time. New `GIT_SHA` / `BUILD_DATE` / `APP_VERSION` Dockerfile
  build-args become OCI labels (`org.opencontainers.image.revision` /
  `.created` / `.version` — the hardcoded `version="0.6.0"` label is now driven
  by `APP_VERSION`) plus `JUNIPER_DATA_GIT_SHA` / `_BUILD_DATE` env vars; a new
  `juniper_data.provenance` accessor reads them back (both `null` outside a
  provenance-stamped image — local dev / a bare `docker build`). The values are
  also passed into `set_build_info(...)` (Prometheus `juniper_data_build` Info
  metric) and the shared `ReadinessResponse`. Foundation for the ecosystem
  stale-image-detection effort — see juniper-ml
  [`notes/BUILD_PROVENANCE_DESIGN_2026-06-14.md`](https://github.com/pcalnon/juniper-ml/blob/main/notes/JUNIPER_2026-06-14_JUNIPER-ECOSYSTEM_BUILD-PROVENANCE-DESIGN.md).
  Requires `juniper-observability>=0.4.0`.

- **Equities S&P 500 time-series generator** (`equities`): new dataset
  generator producing daily per-(ticker, day) records for S&P 500
  constituents from 2000 to the present. Sources daily OHLCV from Yahoo
  Finance (`yfinance`) and shares-outstanding history from SEC EDGAR XBRL
  (`dei:EntityCommonStockSharesOutstanding`, ~2009→present, forward-filled
  to daily); derives 52-week high/low (rolling 252-session), market
  capitalization (`close × shares`), and a configurable-purchase-date cost
  basis. Emits the canonical NPZ contract with **dual targets**: a one-hot
  next-day direction label (`y_*`, keeps `n_classes == 2`) plus an
  auxiliary next-day-close regression target carried in extra `y_reg_*`
  arrays, with compact row-aligned identifier arrays (`ticker_code_*` +
  `ticker_vocab`, `date_*` as YYYYMMDD ints) and a temporal (date-ordered)
  per-ticker train/test split. The universe and ticker → (name, CIK) map
  ship as a bundled `sp500_constituents.csv` snapshot (503 names). New
  optional extra `juniper-data[equities]` (`yfinance` + `pandas`); SEC
  access uses stdlib `urllib` with retry/backoff. 21 unit tests in
  `test_equities_generator.py` (network mocked). Pre-2009 fundamentals are
  unavailable and represented per a `fundamentals_fill` strategy
  (`zero` / `nan` / `drop`); the dataset is survivorship-biased by current
  constituents and non-deterministic for live ranges (documented).

### Changed

- **`.dockerignore` now also excludes nested `**/*.egg-info/` + `**/*.dist-info/`** (defense-in-depth). The existing `*.egg-info/` matched only the build-context root. juniper-data is **not** vulnerable to the build-provenance version-shadow — it runs `python -m juniper_data` off the installed package (no `/app/src` on `PYTHONPATH`) and its egg-info is at the repo root (already excluded) — so this is preventive consistency with the canopy [#362](https://github.com/pcalnon/juniper-canopy/pull/362) / cascor fixes, where a stale `src/*.egg-info` COPYed onto an import path shadowed `importlib.metadata`'s version.

- **`juniper_data.__version__` now derives from `importlib.metadata`** instead
  of a hardcoded literal (OQ-1 of the build-provenance effort), so it can no
  longer drift from `pyproject.toml`'s `[project].version`. Falls back to the
  literal only in a bare source checkout where the distribution is not
  installed.

- **SEC-16 / POC remediation §2.2**: `MetricsAuthMiddleware.trusted_ips`
  now accepts CIDR ranges (`"172.18.0.0/16"`) in addition to bare IP
  literals. IPv6 zone-ids are stripped (`fe80::1%eth0` → `fe80::1`) and
  IPv4-mapped IPv6 client addresses (`::ffff:172.18.0.5`) are unwrapped to
  IPv4 before the membership check, so a Docker container appearing as
  the v6-mapped form matches an IPv4 CIDR allowlist entry. Implementation
  switches from `frozenset[str]` literal-equality to
  `ipaddress.ip_network` membership; `Settings.metrics_trusted_ips` gains
  a fail-loud `field_validator` so unparseable entries (typos like
  `"172.18.0.0/164"`) raise at app startup instead of silently producing
  an empty allowlist that 403s every scrape. New regression class
  `TestMetricsAuthMiddlewareCIDR` in `test_phase1d_security.py` (8 tests)
  pins: CIDR v4 allow + miss, mixed CIDR + literal, CIDR v6 allow,
  IPv4-mapped IPv6 vs IPv4 CIDR (the docker-bridge regression), IPv6
  zone-id strip, invalid-CIDR raises at `Settings()`, and the default
  `["127.0.0.1", "::1"]` still works. Existing tests that spoofed the
  TestClient hostname `"testclient"` were updated to
  `client=("127.0.0.1", 12345)` so they pass under fail-loud validation.
  Closes the second half of
  [`notes/poc/POC_REMEDIATION_PLAN_2026-05-27.md`](https://github.com/pcalnon/juniper-deploy/blob/main/notes/poc/POC_REMEDIATION_PLAN_2026-05-27.md)
  Wave-1 (§2.1 shipped via #155).

### Added

- **`util/test_agents_md_version_drift.py`** -- portable port of juniper-ml's lint test pinning `AGENTS.md`'s `**Version**:` header to `pyproject.toml`'s `[project].version`. Catches the failure class where a `pyproject.toml` bump leaves the agent-facing contract stale. Bundled with a one-line `AGENTS.md` bump 0.5.0 → 0.6.0 to clear the pre-existing drift this lint surfaces. Wired into the CI tests job next to the existing `test_workflow_script_paths.py` lint.

- **METRICS-MON R3.7 (soak complete)**: macOS leg of the unit-tests CI matrix flipped from `experimental: true` → `experimental: false`, making the `macos-latest` (Python 3.12) leg **required**. Failures on macOS now block the job. The `continue-on-error: ${{ matrix.experimental == true }}` job-level guard is preserved as a future-proof escape hatch for future experimental matrix entries; with `experimental: false` it evaluates to `false`. Soak window 2026-05-01 → 2026-05-15 confirmed clean (per user direction). Closes the post-soak follow-up of the R3.7 fan-out.

- **METRICS-MON R3.7 / seed-(R1.3 design)**: macOS leg added to the unit-tests CI matrix. `.github/workflows/ci.yml::unit-tests` now runs on `${{ matrix.os }}` with a single new `macos-latest` (Apple Silicon / ARM) entry pinned to Python 3.12; Linux legs (Python 3.12 + 3.13 + 3.14) are unchanged. The macOS leg starts in **`continue-on-error: true`** mode for a 2-week soak (2026-04-30 → 2026-05-14) so platform-divergence failures (POSIX-only assumptions in dataset filesystem code, etc.) surface in CI without blocking PRs while environment-specific issues are identified. After the soak, flip the include block's `experimental` flag to `false` to make the macOS leg required. Closes the juniper-data leg of [METRICS_MONITORING_R3_ENTRY_PLAN_2026-04-30.md](https://github.com/pcalnon/juniper-ml/blob/main/notes/code-review/METRICS_MONITORING_R3_ENTRY_PLAN_2026-04-30.md) §3 Q1.

### Changed

- **METRICS-MON R2.1.2 / seed-06**: migrated to the shared `juniper-observability` package. `juniper-data[api]` now depends on `juniper-observability>=0.1.0a0`. The cross-cutting machinery (`DependencyStatus`, `ReadinessResponse`, `JuniperJsonFormatter`, `RequestIdMiddleware`, `PrometheusMiddleware`, `configure_logging`, `configure_sentry`, `get_prometheus_app`, `set_build_info`, `request_id_var`, the `UNMATCHED_ENDPOINT_LABEL` constant, and the `_strip_sensitive_headers` SEC-10 hook) now lives in the shared library and is re-exported from `juniper_data.api.observability` and `juniper_data.api.models.health` for backwards compatibility — every existing import path continues to work unchanged. `MetricsAuthMiddleware` (SEC-16, juniper-data-specific) and the dataset-generation Prometheus metrics (`record_dataset_generation`, `set_datasets_cached`) stay in this repo. New wire-compat snapshot test (`tests/unit/test_r2_1_2_wire_compat.py`) pins the externally-observable shape of `/v1/health/ready`. See [`notes/code-review/METRICS_MONITORING_R2.1_SHARED_OBSERVABILITY_DESIGN_2026-04-28.md`](https://github.com/pcalnon/juniper-ml/blob/main/notes/code-review/METRICS_MONITORING_R2.1_SHARED_OBSERVABILITY_DESIGN_2026-04-28.md) in juniper-ml. Companion PRs migrate juniper-cascor and juniper-canopy in dependency order.

### Changed (potentially breaking)

- **METRICS-MON R1.2 / seed-02 / seed-03**: `/v1/health/ready` now returns **HTTP 503** (not 200) when a required dependency is unhealthy, with body `status="not_ready"`. `/v1/health/live` runs an in-process liveness tick (storage directory check) within a 250 ms budget and returns **HTTP 503** on tick failure or budget exceedance. Both endpoints emit a new `X-Juniper-Readiness` / liveness body fields (`tick`, `duration_ms`) so probe diagnostics surface in orchestrator logs without body parsing. Operators relying on `kubectl` readiness probes pointing at `/v1/health/ready` will now see traffic withdrawn from pods whose storage path is unreachable. See [`notes/code-review/METRICS_MONITORING_R1.2_PROBE_DESIGN_2026-04-27.md`](https://github.com/pcalnon/juniper-ml/blob/main/notes/code-review/METRICS_MONITORING_R1.2_PROBE_DESIGN_2026-04-27.md) in juniper-ml for the cross-repo contract; companion PRs land in juniper-cascor, juniper-canopy, and juniper-deploy. `/v1/health` (the legacy combined endpoint) is unchanged.

### Security

- **SEC-01 / JD-SEC-02**: `APIKeyAuth.validate` now compares the presented API key against each configured key with `hmac.compare_digest`, eliminating the timing side-channel that Python's `in` set-membership comparison exposed. The loop deliberately does not short-circuit on first match.
- **JD-SEC-01**: `LocalFSDatasetStore` validates every `dataset_id` through an allowlist regex (`[A-Za-z0-9][A-Za-z0-9._\-]{0,127}`) and then checks that the resolved path stays within the configured base directory before any filesystem access. `ValueError` is raised for any traversal attempt and mapped to HTTP 400 by the existing exception handler; `DatasetStore.batch_delete` catches it so a single bad ID does not abort the batch.
- **JD-SEC-03 / SEC-02**: `RateLimiter._counters` replaced with `cachetools.TTLCache(maxsize=10_000, ttl=window_seconds)`, giving automatic entry eviction and a hard memory ceiling so rotating source IPs cannot exhaust memory. A one-shot warning is logged when the cache crosses 80% capacity. `cachetools>=5.3.0` added to base dependencies.
- **SEC-04** (Phase 1D, 2026-04-24): `POST /v1/datasets` now runs `generator_class.generate(params)` via `asyncio.to_thread`, moving the potentially CPU-bound generator call off the event-loop thread so concurrent HTTP requests are not blocked while a dataset is being synthesized.
- **SEC-10** (Phase 1D): `configure_sentry` now registers a `before_send` hook (`_strip_sensitive_headers`) that redacts `X-API-Key`, `Authorization`, and `Cookie` on every outbound event. `send_default_pii` still honors the operator-facing `JUNIPER_DATA_SENTRY_SEND_PII` setting (default `False`), but the filter acts as defense-in-depth so API keys never reach Sentry regardless of that flag.
- **SEC-16** (Phase 1D): `/metrics` mount is wrapped by a new `MetricsAuthMiddleware` ASGI shim that restricts access to a configurable trusted-IP allowlist (new setting `metrics_trusted_ips`, default `["127.0.0.1", "::1"]`). Unauthorized scrapers receive a plain-text 403 without hitting the Prometheus ASGI sub-app.

### Added

- Hardcoded-values refactor (Wave 1): three new layer-scoped constants modules — `juniper_data/api/constants.py` (32 symbols: header names, status code defaults, body/rate-limit limits, error message templates, exempt paths), `juniper_data/storage/constants.py` (16 symbols: filenames, metadata keys, table/column names), `juniper_data/core/constants.py` (11 symbols: encoding strings, magic numbers, fixed metadata keys). Per-generator `Field(default=...)` defaults moved from inline literals into named module constants in each generator's `params.py`.
- **XREPO-01b / DC-02**: new `MoonGenerator` (two interleaving half-moons) registered under the `"moon"` key in `GENERATOR_REGISTRY`, closing the client/server gap where `juniper_data_client.constants.GENERATOR_MOON` previously pointed at a generator that did not exist. Includes `MoonParams` (Pydantic), per-field defaults in `generators/moon/defaults.py`, and a full unit suite in `tests/unit/test_moon_generator.py` mirroring the circles generator's tests.

### Changed

- Hardcoded-values refactor (Wave 2 + Wave 3): replaced ~115 inline literals across 17 files in the api, storage, and core layers, plus all 7 generator parameter modules. Application code now imports from the new constants modules instead of embedding literals.
- HTTP status codes now use `starlette.status` constants (e.g., `HTTP_404_NOT_FOUND`) instead of integer literals across `routes/`, `middleware.py`, and `security.py`.
- Encoding literals (`'utf-8'`) consolidated into `core/constants.py` and reused across artifact, secrets, and storage modules.
- AGENTS.md "Code Style Conventions → Constants" section updated to document the layer-scoped constants module pattern and the `starlette.status` rule.

### Fixed

- **CONC-12 / BUG-JD-11** (Track 3 Phase 3D, 2026-04-27): `DatasetStore.record_access` (`juniper_data/storage/base.py`) now wraps the entire `get_meta` → in-memory increment → `update_meta` sequence inside `with self._version_lock:`, so two concurrent requests racing on the same dataset can no longer both read the same count, both increment locally, and both write back the same new value. The class-level `_version_lock` already exists for `save_versioned`; this reuses it. Per-process locking only — multi-process deployments still accept best-effort counting per the BUG-JD-05 caveat. Verified by `juniper_data/tests/unit/test_record_access_concurrency.py::TestRecordAccessAtomicity` (4 tests, including a deepcopy + sleep-widened concurrency test that fails on the pre-fix code with `1 == 16` lost-updates and a tracing-lock test that asserts the lock is held across both `get_meta` and `update_meta` calls).

### Notes

- Pydantic `Field` defaults remain literal-equivalent — Wave 5 verified that `SpiralParams`, `XorParams`, `CirclesParams`, `GaussianParams`, and `CheckerboardParams` produce SHA-256-identical `X_full` / `y_full` / split arrays at `seed=42` between this branch and `origin/main`.
- All existing tests pass without modification; pre-commit (19 hooks: ruff lint+format, mypy, bandit, yamllint, shellcheck) is clean.
- No public API changes; HTTP request/response shapes, settings prefix, and storage formats are unchanged.

## [0.6.0] - 2026-04-08

**Summary**: Dataset versioning, batch operations, systemd integration, PostgreSQL reliability fixes, and security hardening for CSV imports.

### Added: [0.6.0]

- **Dataset versioning support (CAN-DEF-005 Phase 1)**: Logical dataset names with auto-incrementing version numbers, `GET /v1/datasets/versions`, `GET /v1/datasets/latest`
- **Batch operations (CAN-DEF-006)**: `POST /v1/datasets/batch-create`, `PATCH /v1/datasets/batch-tags`, `POST /v1/datasets/batch-export`
- **Docker secrets support** via `get_secret()` utility (`juniper_data/core/secrets.py`)
- **Systemd service unit and management CLI** for juniper-data (`feature/systemd-phase2`)
- **CSV import path traversal protection** (`JUNIPER_DATA_IMPORT_DIR` setting)
- **Documentation link checker `--cross-repo skip` mode** for CI pipelines

### Fixed: [0.6.0]

- **Version synchronized** across `__init__.py`, `pyproject.toml`, and `Dockerfile` to 0.6.0
- **PostgreSQL metadata/artifact split-brain** on save failure — transaction now rolls back both atomically
- **PostgreSQL temp artifact race conditions** on concurrent saves — deterministic cleanup added
- **Advisory lock namespace collision** between dataset ID and version allocation — namespaced locks
- **PostgreSQL dataset versioning metadata persistence** — fields now correctly persisted across store operations
- **Dataset version allocation atomicity** — prevents race condition on concurrent version creation
- **Generic `n_classes` fallback** replacing spiral-specific `params.n_spirals` reference that would crash for non-spiral generators with empty training sets

### Changed: [0.6.0]

- **Updated GitHub Actions**: actions/checkout v4.2.2 -> v6.0.2, actions/setup-python v5.6.0 -> v6.2.0, actions/upload-artifact v4.6.0 -> v7.0.0, actions/cache v5.0.3 -> v5.0.4, codecov/codecov-action v5.5.2 -> v6.0.0, github/codeql-action v3.28.0 -> v4.35.1
- **AGENTS.md comprehensive audit and update** to reflect v0.5.0 conventions, conda environment prerequisites
- **Lockfile update workflow** improved for proper CI generation
- **Documentation suite updates**: developer cheatsheet, documentation overview, broken link fixes, version compatibility in REFERENCE.md

### Security: [0.6.0]

- CSV import generator now validates file paths against configurable `JUNIPER_DATA_IMPORT_DIR` directory to prevent path traversal attacks

### Technical Notes: [0.6.0]

- **SemVer impact**: MINOR — New versioning and batch endpoints, systemd integration, security hardening; backward compatible
- **PostgreSQL fixes**: 5 reliability improvements covering split-brain, race conditions, lock collisions, and metadata persistence

---

## [0.5.0] - 2026-03-03

**Summary**: Comprehensive security hardening — security headers middleware, request body limits, error response sanitization, restrictive CORS defaults, rate limiting enabled by default, /metrics authentication, conditional API docs, and scheduled security scanning.

### Security: [0.5.0]

- Added `SecurityHeadersMiddleware` — X-Content-Type-Options, X-Frame-Options, Referrer-Policy, Permissions-Policy, conditional HSTS
- Added `RequestBodyLimitMiddleware` with configurable max body size (default 10MB)
- Sanitized `ValueError` handler to return generic error messages; internal details logged at DEBUG level
- Changed CORS origins default from `["*"]` to `[]` (no origins allowed by default)
- Changed rate limiting default from disabled to enabled
- Added API key requirement for `/metrics` endpoint (removed from exempt paths)
- Added conditional API docs — `/docs`, `/redoc`, `/openapi.json` disabled when API keys are configured

### Added: [0.5.0]

- `.github/workflows/security-scan.yml` — Weekly scheduled security scanning (Bandit, pip-audit)

### Changed: [0.5.0]

- Updated test fixtures for new security defaults (rate limiting, CORS)

### Technical Notes: [0.5.0]

- **SemVer impact**: MINOR — New middleware, changed security defaults (non-breaking: configurable via env vars)
- **Test count**: 766 passed, 0 failed
- **Part of**: Cross-ecosystem security audit (7 repos, 24 findings)

---

## [0.4.2] - 2026-02-17

**Summary**: CI/CD workflow triggers expanded to cover all JuniperData branches, and `.gitignore` updated to properly exclude `__pycache__` directories.

### Changed: [0.4.2]

- **CI-003: CI workflow branch triggers** — Added `subproject.juniper_data.**` pattern to `.github/workflows/ci.yml` push triggers so CI runs automatically on all JuniperData feature and release branches
- **`.gitignore` cleanup** — Uncommented `__pycache__` exclusion patterns; removed tracked `__pycache__` files from repository

### Technical Notes: [0.4.2]

- **SemVer impact**: PATCH -- CI/CD and gitignore configuration only; no API or code changes
- **Test count**: 699 tests (658 service + 41 client), all passing

---

## [0.4.1] - 2026-02-17

**Summary**: Bug fixes for MNIST generator tests, Bandit security scan compliance, and arc-agi optional dependency. First official JuniperData release with all 699 tests passing and CI/CD pipeline fully green.

### Fixed: [0.4.1]

- **MNIST-001: 12 Failing MNIST Generator Tests**
  - Generator's `_load_and_preprocess` called `ds.with_format("numpy")` for bulk column access, but test mocks didn't configure `with_format()`, returning a generic `MagicMock` that produced empty arrays on `np.array()`
  - Fixed by adding `formatted_ds` mocks returning proper numpy data in `_make_mock_hf_dataset()` and `test_generate_image_without_convert()`
  - Added missing `n_samples` support via `ds.select(range(params.n_samples))` in the generator

- **SEC-007: Bandit B615 nosec Placement**
  - Moved `# nosec B615` from the comment line above `hf_load_dataset()` to inline on the call itself
  - Bandit only honors `# nosec` directives on the same line as the flagged code

- **DEP-001: arc-agi Optional Dependency**
  - Made `arc-agi` an optional dependency to prevent `ImportError` when importing `juniper_data` in environments without `arc-agi>=0.9.0` installed (e.g., juniper-cascor)

### Technical Notes: [0.4.1]

- **SemVer impact**: PATCH -- Bug fixes only; no API changes
- **Test count**: 699 tests (658 service + 41 client), all passing
- **CI/CD**: All jobs green across Python 3.12, 3.13, 3.14

---

## [0.4.0] - 2026-02-17

**Summary**: Integration Infrastructure & Extended Data Sources - Docker containerization, health probes, E2E testing, shared client package, dataset lifecycle management, 8 dataset generators, 7 storage backends, comprehensive CI/CD pipeline with security scanning, and full API documentation for ecosystem integration.

### Added: [0.4.0]

- **DATA-012: Shared JuniperData Client Package** (`juniper_data_client/`)
  - Standalone pip-installable package consolidating client code from juniper-cascor and JuniperCanopy
  - **Package files**: `client.py`, `exceptions.py`, `__init__.py`, `py.typed`, `pyproject.toml`, `README.md`
  - **JuniperDataClient class** with all API methods:
    - Health: `health_check()`, `is_ready()`, `wait_for_ready()`
    - Generators: `list_generators()`, `get_generator_schema()`
    - Datasets: `create_dataset()`, `create_spiral_dataset()`, `list_datasets()`, `get_dataset_metadata()`, `delete_dataset()`
    - Artifacts: `download_artifact_npz()`, `download_artifact_bytes()`, `get_preview()`
  - **Custom exceptions**: `JuniperDataClientError`, `JuniperDataConnectionError`, `JuniperDataTimeoutError`, `JuniperDataNotFoundError`, `JuniperDataValidationError`
  - **Enhanced features** over original implementations:
    - Automatic retry logic with configurable backoff (429, 5xx errors)
    - Connection pooling via `requests.Session` with `HTTPAdapter`
    - Context manager support (`with JuniperDataClient() as client:`)
    - Full type hints with mypy strict mode and `py.typed` marker
  - **Dependencies**: numpy>=1.24.0, requests>=2.28.0, urllib3>=2.0.0

- **DATA-013: Client Test Coverage** (`juniper_data_client/tests/`)
  - 35 comprehensive unit tests using `responses` library for HTTP mocking
  - 96% code coverage (no live service required)
  - Test classes: `TestUrlNormalization`, `TestClientConfiguration`, `TestHealthEndpoints`, `TestGeneratorEndpoints`, `TestDatasetCreation`, `TestDatasetRetrieval`, `TestArtifactDownload`, `TestPreview`, `TestDatasetDeletion`, `TestErrorHandling`

- **DATA-014: XOR Generator** (`juniper_data/generators/xor/`)
  - New classification dataset generator for XOR problem
  - `XorParams`: `n_points_per_quadrant`, `x_range`, `y_range`, `margin`, `noise`, `seed`
  - `XorGenerator`: Stateless generator following `SpiralGenerator` pattern
  - 4 quadrants around origin with opposite classes in diagonal quadrants
  - Registered in `GENERATOR_REGISTRY` alongside `spiral`
  - 18 unit tests with full coverage

- **Gaussian Blobs Generator** (`juniper_data/generators/gaussian/`)
  - Mixture-of-Gaussians classification dataset generator
  - `GaussianParams`: `n_blobs`, `n_points_per_blob`, `centers`, `std`, `seed`
  - Configurable cluster centers and covariance
  - Registered in `GENERATOR_REGISTRY`

- **Concentric Circles Generator** (`juniper_data/generators/circles/`)
  - Binary classification with inner and outer circle classes
  - `CirclesParams`: `n_points`, `inner_radius`, `outer_radius`, `noise`, `seed`
  - Registered in `GENERATOR_REGISTRY`

- **Checkerboard Generator** (`juniper_data/generators/checkerboard/`)
  - 2D grid pattern with alternating class squares
  - `CheckerboardParams`: `n_points`, `grid_size`, `noise`, `seed`
  - Registered in `GENERATOR_REGISTRY`

- **CSV Import Generator** (`juniper_data/generators/csv_import/`)
  - Import datasets from CSV or JSON files
  - `CsvImportParams`: configurable feature and label columns
  - Registered in `GENERATOR_REGISTRY`

- **MNIST Generator** (`juniper_data/generators/mnist/`)
  - MNIST and Fashion-MNIST dataset generator
  - Downloads and prepares standard handwritten digit or fashion item classification datasets
  - `MnistParams`: `variant`, `n_samples`, `seed`
  - Registered in `GENERATOR_REGISTRY`

- **ARC-AGI Generator** (`juniper_data/generators/arc_agi/`)
  - ARC-AGI (Abstraction and Reasoning Corpus) dataset generator
  - Visual reasoning tasks from the ARC benchmark
  - `ArcAgiParams`: task configuration parameters
  - Registered in `GENERATOR_REGISTRY`
  - Requires `arc-agi>=0.9.0` dependency

- **Extended Storage Backends** (`juniper_data/storage/`)
  - `CachedDatasetStore` (`cached.py`) — Caching layer wrapping any `DatasetStore` with error logging
  - `HuggingFaceDatasetStore` (`hf_store.py`) — HuggingFace Hub storage backend
  - `KaggleDatasetStore` (`kaggle_store.py`) — Kaggle Datasets storage backend
  - `PostgresDatasetStore` (`postgres_store.py`) — PostgreSQL storage backend
  - `RedisDatasetStore` (`redis_store.py`) — Redis storage backend

- **Environment Variable Management**
  - `.env` file support via `python-dotenv` / `load_dotenv()`
  - Refactored environment variable retrieval in `__init__.py` for clarity and consistency
  - Added `.env` to `.gitignore` for secret protection

- **Gitleaks Secret Scanning**
  - `.gitleaks.toml` configuration with allowlist for historical commits
  - Integrated into CI pipeline via `gitleaks-action`

- **Extended Test Suite** (15 new test files)
  - Generator tests: `test_gaussian_generator.py`, `test_circles_generator.py`, `test_checkerboard_generator.py`, `test_csv_import_generator.py`, `test_mnist_generator.py`, `test_arc_agi_generator.py`
  - Storage tests: `test_cached_store.py`, `test_hf_store.py`, `test_kaggle_store.py`, `test_postgres_store.py`, `test_redis_store.py`
  - Infrastructure tests: `test_init.py`, `test_middleware.py`, `test_security.py`
  - Integration: `test_security_integration.py`

- **DATA-016: Dataset Lifecycle Management**
  - Enhanced `DatasetMeta` model with lifecycle fields:
    - `tags: List[str]` - Dataset tagging/labeling
    - `ttl_seconds: Optional[int]` - Time-to-live configuration
    - `expires_at: Optional[datetime]` - Computed expiration time
    - `last_accessed_at: Optional[datetime]` - Access tracking
    - `access_count: int` - Usage counter
  - Enhanced `DatasetStore` with lifecycle methods:
    - `update_meta()`, `list_all_metadata()`, `record_access()`
    - `is_expired()`, `delete_expired()`, `filter_datasets()`, `batch_delete()`, `get_stats()`
  - New API endpoints:
    - `GET /v1/datasets/filter` - Filter datasets by generator, tags, dates, sample count
    - `GET /v1/datasets/stats` - Aggregate statistics
    - `POST /v1/datasets/batch-delete` - Bulk delete operation
    - `POST /v1/datasets/cleanup-expired` - Remove expired datasets
    - `PATCH /v1/datasets/{id}/tags` - Add/remove tags
  - 44 new tests (27 unit + 17 integration)

- **Integration Development Plan** (`notes/INTEGRATION_DEVELOPMENT_PLAN.md`)
  - Compiled 20 outstanding work items from 4 documentation files and source code analysis
  - 6 HIGH priority items now COMPLETE (mypy fixes, unused imports, Dockerfile, health checks, E2E tests)
  - 5 MEDIUM priority items remaining (API docs, parameter validation, client consolidation)
  - 6 LOW priority items (generators, storage, lifecycle, auth)
  - 3 DEFERRED items (IPC, GPU, profiling)
  - 10 cross-project references (juniper-cascor: 5, JuniperCanopy: 5)

- **DATA-006: Dockerfile for JuniperData Service**
  - Multi-stage build (builder + runtime) using `python:3.12-slim`
  - Installs with `pip install .[api]` for minimal dependencies
  - Non-root `juniper` user (UID 1000) for security
  - Exposes port 8100 with environment variable configuration
  - `.dockerignore` to exclude tests, docs, notes, and development files

- **DATA-007: Health Check Probes for Container Orchestration**
  - `HEALTHCHECK` instruction in Dockerfile (30s interval, 10s timeout, 5s start period, 3 retries)
  - `GET /v1/health/live` - Liveness probe (returns `{"status": "alive"}`)
  - `GET /v1/health/ready` - Readiness probe (returns `{"status": "ready", "version": "..."}`)
  - Original `/v1/health` endpoint preserved for backward compatibility

- **DATA-008: End-to-End Integration Tests**
  - `juniper_data/tests/integration/test_e2e_workflow.py` with 14 comprehensive E2E tests
  - **TestE2EModernAlgorithm**: create/download/verify flow, determinism, seed variation
  - **TestE2ELegacyCascorAlgorithm**: legacy algorithm flow, legacy vs modern comparison
  - **TestE2EDataContract**: NPZ keys, feature dimensions, one-hot labels, split ratios, metadata
  - **TestE2EErrorHandling**: invalid generator, invalid params, nonexistent dataset, delete verification
  - All tests marked with `@pytest.mark.integration` and `@pytest.mark.slow`

- **DATA-009: API Versioning Strategy Documentation**
  - Created `docs/api/JUNIPER_DATA_API.md` with comprehensive API reference
  - Documents versioning policy following SemVer principles
  - Specifies backward compatibility guarantees and deprecation policy

- **DATA-010: NPZ Artifact Schema Documentation**
  - Added dedicated "NPZ Artifact Schema" section in API documentation
  - Documents all 6 required array keys with shapes and dtypes
  - Includes Python and PyTorch loading examples

- **DATA-011: Parameter Validation Parity with Consumers**
  - Added parameter aliases using Pydantic `AliasChoices` in `SpiralParams`
  - `n_points` accepted as alias for `n_points_per_spiral`
  - `noise_level` accepted as alias for `noise`
  - Added 5 new unit tests verifying alias behavior

### Changed: [0.4.0]

- **GENERATOR_REGISTRY** expanded from 2 to 8 registered generators (spiral, xor, gaussian, circles, checkerboard, csv_import, mnist, arc_agi)
- **pyproject.toml** restructured — Python 3.11-3.14 support, new dependencies (`arc-agi>=0.9.0`, `python-dotenv>=1.0.0`), updated tool configuration
- **.pre-commit-config.yaml** restructured with updated hook versions
- **.gitignore** expanded and reorganized for JuniperData structure (added `.env`, extended exclusions)
- **.flake8** extracted to standalone configuration file
- **conf/ directory** cleaned up — removed ~20 obsolete juniper-cascor-era configuration files, reorganized remaining
- **CLAUDE.md** updated with Integration Context section
  - Added integration points documentation (port, feature flag, data contract, consumers)
  - Added key documentation reference table

### Fixed: [0.4.0]

- **DATA-001: mypy Type Errors in Test Files** (20 errors → 0)
  - Added type narrowing assertions (`assert x is not None`) in test_storage.py and test_storage_workflow.py
  - Added `# type: ignore[arg-type]` with explanation for negative test in test_spiral_generator.py
  - Used `getattr()` pattern for dynamic route/middleware attribute access in test_api_app.py

- **DATA-002: flake8 Unused Imports in datasets.py**
  - Removed unused `Any` and `Dict` imports from `typing` module

- **DATA-003: flake8 Issues in generate_golden_datasets.py**
  - Added `# noqa: E402` comments for late imports after `sys.path` manipulation
  - Converted f-strings without placeholders to regular strings (5 instances)

- **CI-001: pip-audit failing on local package**
  - Fixed grep pattern in CI pipeline to handle modern pip's underscore-normalized package names (`juniper_data` vs `juniper-data`)
  - Previous pattern `grep -iv "^juniper-data=="` missed `juniper_data==` format, causing `pip-audit --strict` to fail

- **CI-002: Bandit security scan format**
  - Resolved SARIF format iteration — installed `bandit[sarif]` dependency for proper SARIF report generation
  - Added `--exit-zero` for SARIF generation with separate blocking check for medium+ severity

- **14 CodeQL code scanning alerts resolved**
  - Assert statements with side-effects (alerts #6-11) — refactored assertions
  - Empty except blocks (alerts #5, #12) — added proper exception handling
  - Unused imports (alert #3) — removed unused import
  - `return`/`yield` outside function (alert #50) — fixed scope
  - Module imported with both `import` and `from import` (alert #51) — cleaned up imports
  - Commented-out code (alerts #52, #53) — removed dead code
  - Variable defined multiple times (alert #70) — removed redundant assignment

### Technical Notes: [0.4.0]

- **SemVer impact**: MINOR - New generators, storage backends, Docker infrastructure, API endpoints, parameter aliases, client package, lifecycle management; backward compatible
- **Source analysis findings**: 0 mypy errors, 14 CodeQL alerts resolved, 16 flake8 issues (B008 intentional FastAPI patterns + minor)
- **Generators**: 8 registered in `GENERATOR_REGISTRY` (spiral, xor, gaussian, circles, checkerboard, csv_import, mnist, arc_agi)
- **Storage backends**: 7 implementations (memory, local_fs, cached, hf_store, kaggle_store, postgres_store, redis_store)
- **JuniperData test count**: 658 tests (589 unit + 69 integration), all passing
- **juniper-data-client test count**: 41 tests (standalone package, 96% coverage)
- **Total test count**: 699 tests (658 service + 41 client), all passing
- **Unit test files**: 25 test files covering all generators, storage backends, API, middleware, security, and infrastructure
- **Integration test files**: 5 test files (API, E2E, lifecycle, storage workflows, security)
- **New package**: `juniper_data_client/` - standalone pip-installable client library
- **CI/CD**: Full pipeline with pre-commit (Python 3.12-3.14), unit tests (80% coverage gate), integration tests, security scanning (Gitleaks, Bandit SARIF, pip-audit, CodeQL), build verification

---

## [0.3.0] - 2026-02-04

**Summary**: Comprehensive Test Suite and CI/CD Enhancement - Security hardening, static analysis expansion, infrastructure improvements.

### Security: [0.3.0]

- **SEC-001: Bandit Security Scanning Now Blocking**
  - Replaced `|| true` with `--exit-zero` for SARIF generation
  - Added blocking check for medium+ severity findings

- **SEC-002: pip-audit Now Strict Mode**
  - Changed from warning-only to `--strict` flag to fail on vulnerabilities

- **SEC-003: Dependabot Configuration**
  - New `.github/dependabot.yml` for automated dependency updates
  - Configured for both pip and GitHub Actions ecosystems
  - Weekly schedule with grouped updates

- **SEC-004: GitHub Actions Pinned to SHA**
  - All GitHub Actions now pinned to specific commit SHAs for supply chain security
  - Includes: checkout, setup-python, cache, upload-artifact, codecov, codeql, gitleaks

### Added: [0.3.0]

- **CodeQL Analysis Workflow** (`.github/workflows/codeql.yml`)
  - Weekly semantic code analysis for security vulnerabilities
  - Runs on push to main/develop and on PRs

- **Codecov Integration**
  - Coverage reports now uploaded to Codecov for trend tracking
  - Added `codecov-action` step in unit-tests job

- **Slow Test Job**
  - New `slow-tests` job for tests marked with `@pytest.mark.slow`
  - Runs weekly and on manual trigger

- **Pre-commit Hooks**
  - Added `pyupgrade` hook for Python syntax modernization (py311+)
  - Added `shellcheck` hook for shell script linting

### Changed: [0.3.0]

- **Static Analysis Now Covers Tests**
  - Flake8 now lints test code (with relaxed SIM117 rules)
  - MyPy now type-checks test code (with `--allow-untyped-defs`)
  - Removed E722 and F401 from global Flake8 ignores

- **Pytest Warnings Configuration**
  - Removed global `-p no:warnings` suppression
  - Added targeted `filterwarnings` for expected dependency warnings
  - Removed `--continue-on-collection-errors` flag

- **MyPy Configuration** (`pyproject.toml`)
  - Tests now included in type checking
  - Added relaxed overrides for test modules
  - Removed test exclusion pattern

### Fixed: [0.3.0]

- **TST-001: Silent ImportError Test Pass**
  - Refactored `test_main.py` to use `pytest.skip()` instead of silent `pass`

- **CFG-003: MyPy Type Errors in Production Code**
  - Added type ignore comments for numpy stubs in `core/artifacts.py`, `storage/memory.py`, `storage/local_fs.py`
  - Fixed `Optional` type annotations in `api/routes/datasets.py` and `api/app.py`

- **Unused Imports** (7 fixed)
  - `tests/fixtures/generate_golden_datasets.py`: removed `os`
  - `tests/integration/test_storage_workflow.py`: removed `Dict`
  - `tests/unit/test_api_app.py`: removed `AsyncMock`
  - `tests/unit/test_api_routes.py`: removed `Dict`, `generators`, `io`
  - `tests/unit/test_main.py`: removed `MagicMock`

### Technical Notes: [0.3.0]

- **SemVer impact**: MINOR – Significant CI/CD infrastructure changes; no API changes
- **Test count**: 207 tests (unchanged, all passing)
- **Coverage**: 100% maintained
- **Documentation**: See `notes/TEST_SUITE_CICD_ENHANCEMENT_DEVELOPMENT_PLAN.md` for full implementation details

---

## [0.2.2] - 2026-02-02

**Summary**: Fixed code coverage configuration and achieved 100% test coverage across all source files.

### Fixed: [0.2.2]

- **Code Coverage Configuration** (`pyproject.toml`, `ci.yml`)
  - Changed from path-based `source` to package-based `source_pkgs = ["juniper_data"]`
  - Simplified CI coverage flags from three path-based `--cov` flags to single `--cov=juniper_data`
  - Coverage now correctly measures at 100% (was 0% due to path mismatch)

### Added: [0.2.2]

- **Comprehensive Unit Tests**
  - `test_main.py` - Tests for `__main__.py` entry point (argument parsing, uvicorn launch)
  - `test_api_app.py` - Tests for FastAPI app factory, lifespan, exception handlers
  - `test_api_routes.py` - Tests for datasets/generators/health route edge cases
  - `test_api_settings.py` - Tests for Settings class and get_settings function
  - Added tests for `get_schema()` in spiral generator
  - Added tests for split edge case when rounding exceeds sample count
  - Added tests for LocalFS storage edge cases (JSON serializer, partial file deletion)
  - Added tests verifying abstract base class behavior

### Technical Notes: [0.2.2]

- **SemVer impact**: PATCH – Configuration fix and test additions only; no API changes
- **Test count**: 207 tests (up from 141)
- **Coverage**: 100% across all 23 source files
- **Root cause**: Nested `juniper_data/juniper_data/` structure caused path-based coverage targets to not exist
- **Documentation**: See `notes/CODE_COVERAGE_FIX.md` for detailed analysis

---

## [0.2.1] - 2026-02-01

**Summary**: CI/CD parity achieved across juniper-cascor, JuniperData, and JuniperCanopy with standardized settings.

### Changed: [0.2.1]

- **CI/CD Configuration Parity**
  - `.pre-commit-config.yaml` (v0.1.1)
    - Line length: 512 for black, isort, flake8
    - Added yamllint hook (v1.35.1, relaxed config)
    - Enabled mypy in CI (fully active)
  - `.github/workflows/ci.yml` (v0.1.1)
    - Coverage threshold: 80% (up from 50%)
    - Added build job with package verification
    - Standardized artifact paths: reports/junit/, reports/htmlcov/, reports/coverage.xml
  - `pyproject.toml` (v0.1.1)
    - Line length: 512 for black/isort
    - Coverage fail_under: 80%

### Technical Notes: [0.2.1]

- **SemVer impact**: PATCH – Configuration changes only; no API changes
- **CI Parity**: All 3 Juniper applications now use identical CI/CD settings

---

## [0.2.0] - 2026-01-31

**Summary**: Added legacy parity mode for spiral generator to achieve statistical compatibility with juniper-cascor's SpiralProblem implementation.

### Added: [0.2.0]

- **Legacy Parity Mode** (`generators/spiral/`)
  - New `algorithm` parameter: `"modern"` (default) or `"legacy_cascor"`
  - New `radius` parameter: Maximum radius/distance (default: 10.0)
  - New `origin` parameter: Center point offset as `(x, y)` tuple (default: `(0.0, 0.0)`)

- **Legacy Cascor Algorithm** (`algorithm="legacy_cascor"`)
  - Sqrt-uniform radial sampling: `sqrt(random) * radius`
  - Distance-as-angle formula: `angle = direction * (distance + offset)`
  - Uniform noise in `[0, noise)` (not zero-centered)
  - Matches statistical properties of original juniper-cascor SpiralProblem

- **New Unit Tests** (8 tests for legacy mode)
  - `test_legacy_mode_generates_correct_shapes`
  - `test_legacy_mode_deterministic_with_seed`
  - `test_legacy_mode_different_from_modern`
  - `test_legacy_mode_uniform_noise_range`
  - `test_legacy_mode_radii_distribution`
  - `test_origin_offset_works`
  - `test_radius_parameter_controls_spread`
  - `test_algorithm_param_validation`

### Technical Notes: [0.2.0]

- **SemVer impact**: MINOR – New features added; backward compatible
- **Test count**: 84 tests passing (up from 76)
- Default behavior unchanged (`algorithm="modern"`)

### Usage: [0.2.0]

```python
# Modern mode (default, same as before)
params = SpiralParams(n_spirals=2, n_points_per_spiral=100)

# Legacy Cascor mode (for parity with juniper-cascor)
params = SpiralParams(
    n_spirals=2,
    n_points_per_spiral=100,
    algorithm="legacy_cascor",
    radius=10.0,
    origin=(0.0, 0.0),
)
```

---

## [0.1.2] - 2026-01-31

**Summary**: Added Conda environment configuration for JuniperData development.

### Added: [0.1.2]

- **Conda Environment** (`conf/conda_environment.yaml`)
  - Python >=3.11 with numpy, pytest, dev tools via conda-forge
  - Editable package installation with `pip install -e .[all]`
  - Full test suite validation (76 tests passing)

### Technical Notes: [0.1.2]

- **SemVer impact**: PATCH – Environment configuration only; no API changes
- **Environment name**: JuniperData
- **Test count**: 76 tests passing in new environment

---

## [0.1.1] - 2026-01-30

**Summary**: Added comprehensive CI/CD pipeline with pre-commit hooks, GitHub Actions workflow, and security scanning. Renamed source directory from `src/` to `juniper_data/` for proper package discovery.

### Added: [0.1.1]

- **CI/CD Pipeline** (`.github/workflows/ci.yml`)
  - Pre-commit job across Python 3.11-3.14 matrix
  - Unit tests with 50% coverage gate
  - Integration tests for PRs and main/develop
  - Security scanning: Gitleaks, Bandit SARIF, pip-audit
  - Quality gate aggregator with proper failure handling
  - pip-based (no conda required)

- **Pre-commit Configuration** (`.pre-commit-config.yaml`)
  - General file checks (YAML, TOML, JSON, merge conflicts)
  - Python formatting: Black (line-length=120)
  - Import sorting: isort (black profile)
  - Linting: Flake8 with bugbear, comprehensions, simplify
  - Type checking: MyPy
  - Security: Bandit SAST scanning

- **Enhanced pyproject.toml**
  - New `test` optional dependency group
  - Bandit configuration section
  - Updated pytest paths for new structure
  - Added dev tools: flake8, bandit, pip-audit, pre-commit

### Changed: [0.1.1]

- **Directory Structure**: Renamed `src/` to `juniper_data/` for proper package discovery
- **Test paths**: Updated from `tests/` to `juniper_data/tests/`

### Technical Notes: [0.1.1]

- **SemVer impact**: PATCH – CI/CD infrastructure only; no API changes
- **Pre-commit status**: All 16 hooks pass
- **Test count**: 76 tests passing

---

## [0.1.0] - 2026-01-29

**Summary**: Initial release of JuniperData - a standalone dataset generation and management service extracted from juniper-cascor as part of the Juniper ecosystem refactoring initiative.

### Added: [0.1.0]

- **Core Generator Module** (`juniper_data/generators/spiral/`)
  - `SpiralParams` - Pydantic model with comprehensive validation
  - `SpiralGenerator` - Pure NumPy N-spiral dataset generator
  - `defaults.py` - Default constants migrated from Cascor
  - Static methods: `generate()`, `_generate_raw()`, `_create_one_hot_labels()`
  - Deterministic reproducibility via `np.random.default_rng(seed)`

- **Core Utilities** (`juniper_data/core/`)
  - `split.py` - Dataset shuffle and split utilities
    - `shuffle_data()` - Shuffle X, y together
    - `split_data()` - Partition into train/test
    - `shuffle_and_split()` - Combined high-level function
  - `dataset_id.py` - Deterministic hash-based dataset ID generation
  - `models.py` - Pydantic models for API contracts
    - `DatasetMeta` - Dataset metadata schema
    - `CreateDatasetRequest/Response` - API request/response models
    - `GeneratorInfo`, `PreviewData` - Additional schemas
  - `artifacts.py` - NPZ artifact handling and checksums

- **Storage Layer** (`juniper_data/storage/`)
  - `DatasetStore` - Abstract base class for storage backends
  - `InMemoryDatasetStore` - In-memory storage for testing
  - `LocalFSDatasetStore` - File-based storage with JSON metadata + NPZ artifacts

- **REST API** (`juniper_data/api/`)
  - FastAPI-based service on port 8100
  - **Health**: `GET /v1/health`
  - **Generators**:
    - `GET /v1/generators` - List available generators
    - `GET /v1/generators/{name}/schema` - Get parameter schema
  - **Datasets**:
    - `POST /v1/datasets` - Create/generate dataset (with caching)
    - `GET /v1/datasets` - List all datasets
    - `GET /v1/datasets/{id}` - Get dataset metadata
    - `GET /v1/datasets/{id}/artifact` - Download NPZ file
    - `GET /v1/datasets/{id}/preview` - Preview samples as JSON
    - `DELETE /v1/datasets/{id}` - Delete dataset
  - Pydantic-settings configuration with `JUNIPER_DATA_` env prefix
  - CORS middleware support

- **Test Suite** (76 tests)
  - 60 unit tests covering generators, split, dataset_id
  - 16 integration tests covering all API endpoints
  - Golden dataset fixtures for parity testing

- **Project Infrastructure**
  - `pyproject.toml` with dependencies and tool configuration
  - `AGENTS.md` with build/test commands
  - `README.md` with installation and usage guide
  - CLI entry point: `python -m juniper_data`

### Technical Notes: [0.1.0]

- **Design Principle**: Pure NumPy core (no PyTorch dependency)
- **Artifact-First API**: Returns NPZ files instead of large JSON payloads
- **Deterministic IDs**: SHA-256 hash of generator + version + params
- **Python Version**: >=3.11
- **Key Dependencies**: numpy>=1.24.0, pydantic>=2.0.0, fastapi>=0.100.0

### Migration Notes: [0.1.0]

- This release corresponds to Phases 0-2 of the juniper-cascor refactoring plan
- Cascor integration (Phase 3) completed in juniper-cascor 0.6.0
- Canopy integration (Phase 4) pending

---

## Version History

| Version | Date       | Description                                             |
| ------- | ---------- | ------------------------------------------------------- |
| 0.7.1   | 2026-06-19 | Fix: equities constituents CSV now ships in the wheel    |
| 0.7.0   | 2026-06-19 | Synthetic dt-sequence generators + scaling meta channel  |
| 0.6.0   | 2026-04-08 | Versioning, batch ops, systemd, PostgreSQL fixes        |
| 0.5.0   | 2026-03-03 | Security hardening                                      |
| 0.4.2   | 2026-02-17 | CI branch triggers, gitignore cleanup                   |
| 0.4.1   | 2026-02-17 | Bug fixes: MNIST tests, Bandit scan, arc-agi dependency |
| 0.4.0   | 2026-02-17 | Integration infrastructure & extended data sources      |
| 0.3.0   | 2026-02-04 | Test suite & CI/CD enhancement                          |
| 0.2.2   | 2026-02-02 | Code coverage configuration fix                         |
| 0.2.1   | 2026-02-01 | CI/CD parity across Juniper                             |
| 0.2.0   | 2026-01-31 | Legacy parity mode for spiral                           |
| 0.1.2   | 2026-01-31 | Conda environment setup                                 |
| 0.1.1   | 2026-01-30 | CI/CD Pipeline & Pre-commit                             |
| 0.1.0   | 2026-01-29 | Initial release (Phases 0-2)                            |

---

## Related Changes

### juniper-cascor 0.6.0 (2026-01-30)

Phase 3 Cascor integration completed:

- Added `JuniperDataClient` for API communication
- Added `SpiralDataProvider` for torch tensor conversion
- Feature flag `JUNIPER_DATA_URL` enables JuniperData mode in SpiralProblem
