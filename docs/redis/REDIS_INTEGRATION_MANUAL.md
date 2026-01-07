# Redis Integration Manual

## Complete guide for implementing Redis integration in Juniper Canopy

**Version:** 0.1.0 - PLANNED FEATURE  
**Status:** ⚠️ NOT YET IMPLEMENTED  
**Last Updated:** November 5, 2025

---

## ⚠️ Important Notice

**This feature is currently PLANNED but NOT YET IMPLEMENTED.**

This manual provides implementation guidance for developers who want to add Redis support to Juniper Canopy.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Implementation Plan](#implementation-plan)
- [Cache Manager](#cache-manager)
- [API Integration](#api-integration)
- [Pub/Sub Integration](#pubsub-integration)
- [Testing](#testing)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)

---

## Overview

### Purpose

Redis integration will provide:

1. **Metrics Caching** - Cache expensive metric computations
2. **Session Management** - Multi-user session support
3. **Real-Time Streaming** - Pub/Sub for training updates
4. **History Persistence** - Long-term training history storage
5. **Performance** - Reduce database/computation load

### Current State

**What exists:**

- Configuration in `conf/app_config.yaml`
- Environment variable support
- Fallback to in-memory caching

**What's missing:**

- Redis client initialization
- Cache manager implementation
- API endpoint integration
- Pub/Sub handlers
- Tests

---

## Architecture

### Component Overview

```bash
┌─────────────────────────────────────────────────────────────┐
│                     Frontend Dashboard                      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   FastAPI Backend                           │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              REST API Endpoints                        │ │
│  │  /api/metrics, /api/topology, /api/history             │ │
│  └───────────────────┬────────────────────────────────────┘ │
│                      │                                      │
│                      ▼                                      │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              Cache Manager                             │ │
│  │  - get_cached()                                        │ │
│  │  - set_cached()                                        │ │
│  │  - invalidate()                                        │ │
│  │  - publish()                                           │ │
│  │  - subscribe()                                         │ │
│  └───────────────────┬────────────────────────────────────┘ │
└──────────────────────┼──────────────────────────────────────┘
                       │
                       ▼
        ┌───────────────────────────────┐
        │      Redis Server             │
        │  ┌──────────────────────────┐ │
        │  │  Key-Value Store         │ │
        │  │  - cascor:metrics:*      │ │
        │  │  - cascor:topology:*     │ │
        │  │  - cascor:session:*      │ │
        │  └──────────────────────────┘ │
        │  ┌──────────────────────────┐ │
        │  │  Pub/Sub Channels        │ │
        │  │  - cascor:training       │ │
        │  │  - cascor:control        │ │
        │  └──────────────────────────┘ │
        └───────────────────────────────┘
```

### Data Flow

**1. Cache Read:**

```bash
Request → API Endpoint → Cache Manager → Redis GET → Response
                              ↓ (cache miss)
                        Backend Compute → Redis SET → Response
```

**2. Cache Write:**

```bash
Training Update → Cache Manager → Redis SET + PUBLISH → Subscribers
```

**3. Pub/Sub Flow:**

```bash
Training Event → PUBLISH(channel, data) → SUBSCRIBE(channel) → Clients
```

---

## Implementation Plan

### Phase 1: Basic Caching (P1)

**Goal:** Replace in-memory cache with Redis

**Tasks:**

1. **Create cache manager module** (`src/backend/cache_manager.py`)
2. **Implement Redis client initialization**
3. **Add cache operations** (get, set, delete, clear)
4. **Integrate with `/api/metrics` endpoint**
5. **Add tests** (unit + integration)

**Estimated effort:** 4-6 hours

---

### Phase 2: Advanced Caching (P2)

**Goal:** Cache all API endpoints

**Tasks:**

1. **Integrate with `/api/topology` endpoint**
2. **Integrate with `/api/history` endpoint**
3. **Add cache invalidation logic**
4. **Implement TTL strategies**
5. **Add cache monitoring**

**Estimated effort:** 3-4 hours

---

### Phase 3: Pub/Sub Support (P3)

**Goal:** Real-time event streaming

**Tasks:**

1. **Implement pub/sub manager**
2. **Add training event publishing**
3. **Add control command subscription**
4. **Integrate with WebSocket manager**
5. **Add pub/sub tests**

**Estimated effort:** 6-8 hours

---

### Phase 4: Session Management (P4)

**Goal:** Multi-user support

**Tasks:**

1. **Implement session storage**
2. **Add user authentication caching**
3. **Session cleanup/expiration**
4. **Add session tests**

**Estimated effort:** 4-6 hours

---

## Cache Manager

### File Structure

```bash
src/backend/
├── __init__.py
├── cache_manager.py         # NEW - Main cache manager
├── redis_client.py          # NEW - Redis connection
├── cache_strategies.py      # NEW - TTL strategies
└── tests/
    ├── test_cache_manager.py
    └── test_redis_client.py
```

### cache_manager.py

**Implementation skeleton:**

```python
#!/usr/bin/env python
#####################################################################
# Project:       Juniper
# Prototype:     Monitoring and Diagnostic Frontend for CasCor
# File Name:     cache_manager.py
# Author:        Paul Calnon
# Version:       0.1.0
# Date:          2025-11-05
# Description:   Redis cache manager for metrics and data caching
#####################################################################

import json
import logging
from typing import Any, Optional, Dict, List
from datetime import timedelta

import redis
from redis.connection import ConnectionPool

from config_manager import get_config


class CacheManager:
    """
    Redis cache manager for Juniper Canopy.

    Provides:
    - Metrics caching with TTL
    - Topology caching
    - History caching
    - Cache invalidation
    - Pub/Sub support (Phase 3)
    """

    def __init__(self, redis_url: Optional[str] = None):
        """
        Initialize Redis cache manager.

        Args:
            redis_url: Redis connection URL (default: from config)
        """
        self.logger = logging.getLogger(__name__)
        self.config = get_config()

        # Get Redis URL from config or parameter
        self.redis_url = redis_url or self.config.get(
            "backend.cache.redis_url",
            "redis://localhost:6379/0"
        )

        # Get cache settings
        self.ttl_seconds = self.config.get("backend.cache.ttl_seconds", 3600)
        self.max_memory_mb = self.config.get("backend.cache.max_memory_mb", 100)

        # Initialize connection pool
        self.pool = ConnectionPool.from_url(
            self.redis_url,
            max_connections=50,
            decode_responses=True
        )

        # Initialize Redis client
        self.redis_client = redis.Redis(connection_pool=self.pool)

        # Verify connection
        self._verify_connection()

        self.logger.info(f"Redis cache initialized: {self.redis_url}")

    def _verify_connection(self):
        """Verify Redis connection."""
        try:
            self.redis_client.ping()
            self.logger.info("Redis connection verified")
        except redis.ConnectionError as e:
            self.logger.error(f"Redis connection failed: {e}")
            raise

    # ==================== Cache Operations ====================

    def get_cached(self, key: str) -> Optional[Dict]:
        """
        Get cached data.

        Args:
            key: Cache key

        Returns:
            Cached data or None if not found
        """
        try:
            data = self.redis_client.get(key)
            if data is None:
                self.logger.debug(f"Cache miss: {key}")
                return None

            self.logger.debug(f"Cache hit: {key}")
            return json.loads(data)
        except Exception as e:
            self.logger.error(f"Cache get error: {e}", exc_info=True)
            return None

    def set_cached(self, key: str, data: Dict, ttl: Optional[int] = None):
        """
        Set cached data with TTL.

        Args:
            key: Cache key
            data: Data to cache
            ttl: Time-to-live in seconds (default: from config)
        """
        try:
            ttl = ttl or self.ttl_seconds
            serialized = json.dumps(data)
            self.redis_client.setex(key, ttl, serialized)
            self.logger.debug(f"Cached: {key} (TTL: {ttl}s)")
        except Exception as e:
            self.logger.error(f"Cache set error: {e}", exc_info=True)

    def invalidate(self, pattern: str):
        """
        Invalidate cache keys matching pattern.

        Args:
            pattern: Key pattern (e.g., "cascor:metrics:*")
        """
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                self.redis_client.delete(*keys)
                self.logger.info(f"Invalidated {len(keys)} keys: {pattern}")
        except Exception as e:
            self.logger.error(f"Cache invalidation error: {e}", exc_info=True)

    def clear_all(self):
        """Clear all cached data."""
        try:
            self.redis_client.flushdb()
            self.logger.warning("All cache cleared")
        except Exception as e:
            self.logger.error(f"Cache clear error: {e}", exc_info=True)

    # ==================== Metrics Caching ====================

    def cache_metrics(self, metrics: Dict):
        """
        Cache latest metrics.

        Args:
            metrics: Metrics dictionary
        """
        key = "cascor:metrics:latest"
        self.set_cached(key, metrics, ttl=1)  # 1-second TTL for real-time

    def get_cached_metrics(self) -> Optional[Dict]:
        """Get cached metrics."""
        return self.get_cached("cascor:metrics:latest")

    # ==================== Topology Caching ====================

    def cache_topology(self, topology: Dict):
        """Cache network topology."""
        key = "cascor:topology:latest"
        self.set_cached(key, topology, ttl=5)  # 5-second TTL

    def get_cached_topology(self) -> Optional[Dict]:
        """Get cached topology."""
        return self.get_cached("cascor:topology:latest")

    # ==================== History Caching ====================

    def cache_history(self, history: List[Dict]):
        """Cache training history."""
        key = "cascor:history:latest"
        self.set_cached(key, history, ttl=60)  # 60-second TTL

    def get_cached_history(self) -> Optional[List[Dict]]:
        """Get cached history."""
        return self.get_cached("cascor:history:latest")

    # ==================== Pub/Sub (Phase 3) ====================

    def publish(self, channel: str, message: Dict):
        """
        Publish message to channel.

        Args:
            channel: Channel name
            message: Message data
        """
        try:
            serialized = json.dumps(message)
            self.redis_client.publish(channel, serialized)
            self.logger.debug(f"Published to {channel}: {message.get('type')}")
        except Exception as e:
            self.logger.error(f"Publish error: {e}", exc_info=True)

    def subscribe(self, channels: List[str], callback):
        """
        Subscribe to channels.

        Args:
            channels: List of channel names
            callback: Callback function(channel, message)
        """
        try:
            pubsub = self.redis_client.pubsub()
            pubsub.subscribe(**{ch: callback for ch in channels})
            self.logger.info(f"Subscribed to: {channels}")

            # Start listening thread
            thread = pubsub.run_in_thread(sleep_time=0.01, daemon=True)
            return thread
        except Exception as e:
            self.logger.error(f"Subscribe error: {e}", exc_info=True)
            return None

    # ==================== Cleanup ====================

    def close(self):
        """Close Redis connection."""
        try:
            self.redis_client.close()
            self.pool.disconnect()
            self.logger.info("Redis connection closed")
        except Exception as e:
            self.logger.error(f"Close error: {e}", exc_info=True)

    def __del__(self):
        """Destructor: ensure cleanup."""
        try:
            self.close()
        except:
            pass


# ==================== Factory ====================

_cache_manager_instance = None


def get_cache_manager() -> CacheManager:
    """Get singleton cache manager instance."""
    global _cache_manager_instance
    if _cache_manager_instance is None:
        _cache_manager_instance = CacheManager()
    return _cache_manager_instance
```

---

## API Integration

### Integrate with Metrics Endpoint

**In `src/main.py`, modify `/api/metrics` endpoint:**

```python
from backend.cache_manager import get_cache_manager

@app.get("/api/metrics")
async def get_metrics():
    """
    Get current training metrics (with Redis caching).

    Returns:
        Current metrics dictionary
    """
    # Try cache first
    cache_manager = get_cache_manager()
    cached = cache_manager.get_cached_metrics()
    if cached is not None:
        return cached

    # Cache miss - compute metrics
    if demo_mode and demo_mode.is_running:
        metrics = demo_mode.get_current_state()
    elif cascor_integration:
        metrics = cascor_integration.get_current_metrics()
    else:
        metrics = {"error": "No data source available"}

    # Cache for next request
    cache_manager.cache_metrics(metrics)

    return metrics
```

### Integrate with Topology Endpoint

```python
@app.get("/api/network/topology")
async def get_topology():
    """
    Get network topology (with Redis caching).

    Returns:
        Network topology dictionary
    """
    # Try cache first
    cache_manager = get_cache_manager()
    cached = cache_manager.get_cached_topology()
    if cached is not None:
        return cached

    # Cache miss - extract topology
    if demo_mode and demo_mode.is_running:
        topology = demo_mode.get_topology()
    elif cascor_integration:
        topology = cascor_integration.get_network_topology()
    else:
        topology = {"error": "No data source available"}

    # Cache for next request
    cache_manager.cache_topology(topology)

    return topology
```

### Cache Invalidation

**Invalidate cache on training events:**

```python
# In demo_mode.py or cascor_integration.py

def on_epoch_end(self, epoch: int):
    """Called at end of each epoch."""
    # ... existing code ...

    # Invalidate metrics cache
    cache_manager = get_cache_manager()
    cache_manager.invalidate("cascor:metrics:*")

    # Cache new metrics
    metrics = self.get_current_state()
    cache_manager.cache_metrics(metrics)
```

---

## Pub/Sub Integration

### Publishing Training Events

**In `src/demo_mode.py`:**

```python
from backend.cache_manager import get_cache_manager

class DemoMode:
    def __init__(self):
        # ... existing init ...
        self.cache_manager = get_cache_manager()

    def _training_loop(self):
        """Demo training loop with pub/sub."""
        while not self._stop.is_set():
            # ... compute metrics ...

            # Publish training update
            self.cache_manager.publish(
                "cascor:training",
                {
                    "type": "metrics_update",
                    "epoch": self.current_epoch,
                    "loss": loss,
                    "accuracy": accuracy,
                    "timestamp": datetime.now().isoformat()
                }
            )

            # ... rest of loop ...
```

### Subscribing to Control Commands

**In `src/main.py`:**

```python
# Subscribe to control channel at startup
cache_manager = get_cache_manager()

def handle_control_message(channel, message):
    """Handle control commands from Redis."""
    data = json.loads(message['data'])
    command = data.get('command')

    if command == 'pause':
        demo_mode.pause()
    elif command == 'resume':
        demo_mode.resume()
    elif command == 'reset':
        demo_mode.reset()

# Subscribe
cache_manager.subscribe(['cascor:control'], handle_control_message)
```

---

## Testing

### Unit Tests

**`src/tests/unit/test_cache_manager.py`:**

```python
import pytest
from backend.cache_manager import CacheManager

@pytest.fixture
def cache_manager():
    """Create cache manager for testing."""
    manager = CacheManager(redis_url="redis://localhost:6379/15")
    yield manager
    manager.clear_all()
    manager.close()


def test_cache_set_get(cache_manager):
    """Test basic cache set/get."""
    data = {"epoch": 5, "loss": 0.2}
    cache_manager.set_cached("test:key", data, ttl=60)

    cached = cache_manager.get_cached("test:key")
    assert cached == data


def test_cache_miss(cache_manager):
    """Test cache miss returns None."""
    cached = cache_manager.get_cached("nonexistent:key")
    assert cached is None


def test_cache_invalidation(cache_manager):
    """Test cache invalidation."""
    cache_manager.set_cached("test:key1", {"a": 1})
    cache_manager.set_cached("test:key2", {"b": 2})

    cache_manager.invalidate("test:*")

    assert cache_manager.get_cached("test:key1") is None
    assert cache_manager.get_cached("test:key2") is None


def test_metrics_caching(cache_manager):
    """Test metrics caching."""
    metrics = {"epoch": 10, "loss": 0.1}
    cache_manager.cache_metrics(metrics)

    cached = cache_manager.get_cached_metrics()
    assert cached == metrics
```

### Integration Tests

**`src/tests/integration/test_redis_integration.py`:**

```python
import pytest
from fastapi.testclient import TestClient
from main import app

@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


def test_metrics_endpoint_caching(client):
    """Test metrics endpoint uses Redis cache."""
    # First request (cache miss)
    response1 = client.get("/api/metrics")
    assert response1.status_code == 200

    # Second request (cache hit)
    response2 = client.get("/api/metrics")
    assert response2.status_code == 200

    # Should be identical
    assert response1.json() == response2.json()


def test_cache_invalidation_on_update(client):
    """Test cache invalidation on training update."""
    # Get initial metrics
    response1 = client.get("/api/metrics")
    epoch1 = response1.json()["epoch"]

    # Trigger training update (would invalidate cache)
    # ... trigger update ...

    # Get updated metrics
    response2 = client.get("/api/metrics")
    epoch2 = response2.json()["epoch"]

    # Epoch should have changed
    assert epoch2 > epoch1
```

---

## Deployment

### Docker Compose

**Add Redis to `docker-compose.yml` (when created):**

```yaml
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    command: redis-server --appendonly yes
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5

  juniper_canopy:
    build: .
    ports:
      - "8050:8050"
    environment:
      - CASCOR_CACHE_REDIS_URL=redis://redis:6379/0
    depends_on:
      redis:
        condition: service_healthy

volumes:
  redis-data:
```

### Production Configuration

**`conf/production_config.yaml`:**

```yaml
backend:
  cache:
    enabled: true
    type: redis
    redis_url: redis://redis.production.com:6379/0
    ttl_seconds: 3600
    max_memory_mb: 512

    # Production settings
    connection_pool:
      max_connections: 100
      timeout: 10
      retry_on_timeout: true
      socket_keepalive: true
      socket_keepalive_options: {1: 1, 2: 2, 3: 3}

    # Security
    ssl: true
    password: "${REDIS_PASSWORD}"
```

---

## Troubleshooting

### Enable Debug Logging

```python
# In cache_manager.py
logging.basicConfig(level=logging.DEBUG)

# Or in app_config.yaml
logging:
  categories:
    backend:
      console_level: DEBUG
      file_level: DEBUG
```

### Monitor Redis

```bash
# Monitor all commands
redis-cli monitor

# Check memory usage
redis-cli info memory

# Check connected clients
redis-cli client list

# Check key count
redis-cli dbsize
```

### Connection Issues

```bash
# Test connection
redis-cli -h localhost -p 6379 ping

# Check Redis logs
tail -f /var/log/redis/redis-server.log

# Restart Redis
sudo systemctl restart redis-server
```

---

## Next Steps

1. **Implement Phase 1:** Basic caching (metrics endpoint)
2. **Test thoroughly:** Unit + integration tests
3. **Add monitoring:** Track cache hit/miss rates
4. **Implement Phase 2:** Additional endpoint caching
5. **Implement Phase 3:** Pub/Sub support
6. **Document:** Update CHANGELOG.md and AGENTS.md

---

## Additional Resources

- **[REDIS_INTEGRATION_QUICK_START.md](REDIS_INTEGRATION_QUICK_START.md)** - Quick setup guide
- **[REDIS_INTEGRATION_REFERENCE.md](REDIS_INTEGRATION_REFERENCE.md)** - Technical reference
- **[Redis Documentation](https://redis.io/documentation)** - Official docs
- **[redis-py Documentation](https://redis-py.readthedocs.io/)** - Python client

---

**Last Updated:** November 5, 2025  
**Version:** 0.1.0  
**Status:** ⚠️ IMPLEMENTATION GUIDE

**Ready to implement? Start with Phase 1 and create `src/backend/cache_manager.py`!**
