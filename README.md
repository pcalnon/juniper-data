# juniper-data

[![PyPI](https://img.shields.io/pypi/v/juniper-data)](https://pypi.org/project/juniper-data/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)

**A FastAPI service that generates, versions, and serves ML datasets as NPZ artifacts.**

`juniper-data` turns a catalogue of dataset generators into a REST service: you ask it for a dataset
by name and parameters, and it returns a versioned, NPZ-formatted train/test/full split. The
catalogue spans synthetic classification problems (two-spiral, concentric circles, XOR, Gaussian
mixtures, moons, checkerboard), image sets (MNIST), the ARC-AGI visual-reasoning families, a CSV/JSON
import path, and a family of **time-series and irregularly-sampled sequence** generators
(autoregressive, Mackey-Glass, multi-sine, delay-product, equities, and the irregular-Δt `equities_seq`
contract). A named-version registry, tag filtering, batch creation, and per-dataset preview round out
the surface. Call `GET /v1/generators` for the live catalogue.

It is the foundational data layer of the platform: the dataset identifiers it returns are the
substrate `juniper-cascor` trains on and `juniper-canopy` visualises.

> **Part of the Juniper platform.** juniper-data is the dataset-generation service of
> [Juniper](https://github.com/pcalnon/juniper-ml) — a multi-package ML research platform built around
> constructive (Cascade-Correlation) and recurrent neural networks. It runs standalone; the rest of the
> platform consumes it over HTTP (see [`juniper-data-client`](https://github.com/pcalnon/juniper-data-client)).

## Install

```bash
pip install juniper-data            # from PyPI
```

For development from a clone (the optional extras are `api`, `arc-agi`, `equities`, `mnist`,
`observability`, `test`, `dev`, `all`):

```bash
git clone https://github.com/pcalnon/juniper-data.git && cd juniper-data
pip install -e ".[all]"
```

### MNIST / Fashion-MNIST (optional extra)

The `mnist` generator loads the real MNIST / Fashion-MNIST datasets from the Hugging Face Hub and
needs the (heavy) `datasets` chain, shipped behind an explicit extra — it is never part of the base
install:

```bash
pip install "juniper-data[mnist]"
```

- **First call downloads from the Hub** into the Hugging Face cache (`HF_HOME`, default
  `~/.cache/huggingface`; the Docker image pins it to `/app/data/hf-cache` so a mounted data volume
  persists it). Later calls are served from the cache.
- **Offline deployments** must seed that cache ahead of time (run one generation for each dataset
  while online, or copy a populated `HF_HOME` in); with `HF_HUB_OFFLINE=1` the generator then works
  entirely from the cache.
- **Without the extra installed**, the generator is unavailable: the registry reports
  `available: false` and `POST /v1/datasets` returns `501` with the install hint instead of a
  masked 500.
- The service Docker image ships the extra (it is compiled into `requirements.lock`), so MNIST
  generation works in containers out of the box.

## Run

```bash
uvicorn --factory juniper_data.api.app:get_app --reload    # binds 127.0.0.1:8100
curl http://localhost:8100/v1/health/ready
curl http://localhost:8100/v1/generators                   # the live generator catalogue
```

Create a dataset over the REST API:

```bash
curl -sX POST localhost:8100/v1/datasets \
  -H 'Content-Type: application/json' \
  -d '{"generator": "spiral", "name": "demo", "params": {"n_spirals": 2, "noise": 0.1}}'
```

Or generate one in-process, without the service:

```python
from juniper_data.generators import SpiralGenerator, SpiralParams

dataset = SpiralGenerator.generate(SpiralParams(n_spirals=2, n_points_per_spiral=100, noise=0.1))
# dataset: dict of float32 arrays — X_train, y_train, X_test, y_test, X_full, y_full
```

## Data contract

Datasets are NPZ archives with the keys `X_train`, `y_train`, `X_test`, `y_test`, `X_full`, `y_full`,
all `float32`. This is the contract every Juniper consumer reads.

## Configuration

Settings load from the `JUNIPER_DATA_` environment namespace (`juniper_data/api/settings.py`) and honor
the Docker `_FILE` secret convention. The most common knobs (full surface in
[`docs/REFERENCE.md`](docs/REFERENCE.md)):

| Variable | Default | Purpose |
|---|---|---|
| `JUNIPER_DATA_HOST` / `JUNIPER_DATA_PORT` | `127.0.0.1` / `8100` | Bind address / port (`0.0.0.0` under Docker). |
| `JUNIPER_DATA_STORAGE_PATH` | `./data/datasets` | Where persisted dataset artifacts live. |
| `JUNIPER_DATA_API_KEYS` | _(unset)_ | CSV / JSON-array of `X-API-Key` values; auth is disabled when unset. |
| `JUNIPER_DATA_LOG_LEVEL` / `_LOG_FORMAT` | `INFO` / `text` | Verbosity / `text` or `json`. |
| `JUNIPER_DATA_METRICS_ENABLED` | `false` | Expose `/metrics` for Prometheus (IP-gated). |

## Docker

```bash
docker build -t juniper-data:latest .
docker run --rm -p 8100:8100 -e JUNIPER_DATA_HOST=0.0.0.0 juniper-data:latest
```

Multi-stage build (Python 3.14-slim); health is probed at `/v1/health/ready`. For the full stack, see
[`juniper-deploy`](https://github.com/pcalnon/juniper-deploy).

## Status

**Live** on PyPI. The current version is shown by the badge above; see [`CHANGELOG.md`](CHANGELOG.md).
Consumed by `juniper-cascor` and `juniper-canopy` via `JUNIPER_DATA_URL`, and by
[`juniper-data-client`](https://github.com/pcalnon/juniper-data-client) programmatically.

## Documentation

- [`docs/QUICK_START.md`](docs/QUICK_START.md) — get running in five minutes
- [`docs/USER_MANUAL.md`](docs/USER_MANUAL.md) — comprehensive usage guide
- [`docs/api/JUNIPER_DATA_API.md`](docs/api/JUNIPER_DATA_API.md) — full REST reference (filtering, batch, tagging, versioning)
- [`docs/REFERENCE.md`](docs/REFERENCE.md) — configuration and environment-variable reference
- [`docs/DOCUMENTATION_OVERVIEW.md`](docs/DOCUMENTATION_OVERVIEW.md) — index of all juniper-data docs

## License

MIT — see [LICENSE](./LICENSE).
