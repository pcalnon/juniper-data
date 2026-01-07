# Redis Integration Reference

## Technical reference for Redis integration in Juniper Canopy

**Version:** 0.1.0 - PLANNED FEATURE  
**Status:** ⚠️ NOT YET IMPLEMENTED  
**Last Updated:** November 5, 2025

---

## ⚠️ Important Notice

**This feature is currently PLANNED but NOT YET IMPLEMENTED.**

This reference documents the planned Redis integration architecture and API for future implementation.

---

## Table of Contents

- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Cache Keys](#cache-keys)
- [TTL Strategies](#ttl-strategies)
- [Pub/Sub Channels](#pubsub-channels)
- [Error Handling](#error-handling)
- [Performance](#performance)
- [Security](#security)

---

## Configuration

### Configuration Parameters

**Location:** `conf/app_config.yaml`

```yaml
backend:
  cache:
    # Enable/disable Redis caching
    enabled: boolean (default: true)

    # Cache type: 'redis' or 'memory'
    type: string (default: 'redis')

    # Redis connection URL
    redis_url: string (default: 'redis://localhost:6379/0')
    # Format: redis://[username:password@]host:port/database

    # Default TTL for cached data (seconds)
    ttl_seconds: integer (default: 3600)

    # Maximum memory usage (MB)
    max_memory_mb: integer (default: 100)

    # Connection pool settings
    connection_pool:
      max_connections: integer (default: 50)
      timeout: integer (default: 5)
      retry_on_timeout: boolean (default: true)
      socket_keepalive: boolean (default: true)

    # Key prefixes for organization
    key_prefixes:
      metrics: string (default: 'cascor:metrics:')
      topology: string (default: 'cascor:topology:')
      session: string (default: 'cascor:session:')
      history: string (default: 'cascor:history:')
```

### Environment Variables

**Override configuration via environment:**

| Variable                     | Type   | Default                    | Description                      |
| ---------------------------- | ------ | -------------------------- | -------------------------------- |
| `CASCOR_CACHE_ENABLED`       | bool   | `true`                     | Enable Redis caching             |
| `CASCOR_CACHE_TYPE`          | string | `redis`                    | Cache type ('redis' or 'memory') |
| `CASCOR_CACHE_REDIS_URL`     | string | `redis://localhost:6379/0` | Redis connection URL             |
| `CASCOR_CACHE_TTL_SECONDS`   | int    | `3600`                     | Default TTL (seconds)            |
| `CASCOR_CACHE_MAX_MEMORY_MB` | int    | `100`                      | Max memory (MB)                  |
| `REDIS_PASSWORD`             | string | -                          | Redis password (for production)  |

**Example:**

```bash
export CASCOR_CACHE_ENABLED=1
export CASCOR_CACHE_TYPE=redis
export CASCOR_CACHE_REDIS_URL=redis://localhost:6379/0
export CASCOR_CACHE_TTL_SECONDS=3600
```

---

## API Reference

### CacheManager Class

**Module:** `src/backend/cache_manager.py` (when implemented)

#### Constructor

```python
CacheManager(redis_url: Optional[str] = None)
```

**Parameters:**

- `redis_url` (str, optional): Redis connection URL. Defaults to config value.

**Raises:**

- `redis.ConnectionError`: If connection to Redis fails

**Example:**

```python
from backend.cache_manager import CacheManager

# Use config URL
cache = CacheManager()

# Or specify URL
cache = CacheManager(redis_url="redis://localhost:6379/1")
```

---

#### get_cached

```python
get_cached(key: str) -> Optional[Dict]
```

Get cached data by key.

**Parameters:**

- `key` (str): Cache key

**Returns:**

- `Dict`: Cached data if found
- `None`: If key not found or expired

**Example:**

```python
data = cache.get_cached("cascor:metrics:latest")
if data:
    print(f"Epoch: {data['epoch']}")
```

---

#### set_cached

```python
set_cached(key: str, data: Dict, ttl: Optional[int] = None)
```

Set cached data with TTL.

**Parameters:**

- `key` (str): Cache key
- `data` (Dict): Data to cache (must be JSON-serializable)
- `ttl` (int, optional): Time-to-live in seconds. Defaults to config value.

**Example:**

```python
metrics = {"epoch": 5, "loss": 0.2}
cache.set_cached("cascor:metrics:latest", metrics, ttl=60)
```

---

#### invalidate

```python
invalidate(pattern: str)
```

Invalidate cache keys matching pattern.

**Parameters:**

- `pattern` (str): Key pattern (supports Redis glob patterns)

**Patterns:**

- `*`: Match any characters
- `?`: Match single character
- `[abc]`: Match one character from set

**Example:**

```python
# Invalidate all metrics
cache.invalidate("cascor:metrics:*")

# Invalidate specific epoch
cache.invalidate("cascor:metrics:epoch_5")
```

---

#### cache_metrics

```python
cache_metrics(metrics: Dict)
```

Cache latest training metrics with 1-second TTL.

**Parameters:**

- `metrics` (Dict): Metrics dictionary

**Example:**

```python
metrics = {
    "epoch": 10,
    "train_loss": 0.15,
    "train_accuracy": 0.95,
    "timestamp": "2025-11-05T12:00:00"
}
cache.cache_metrics(metrics)
```

---

#### get_cached_metrics

```python
get_cached_metrics() -> Optional[Dict]
```

Get cached metrics.

**Returns:**

- `Dict`: Latest metrics if cached
- `None`: If not cached or expired

**Example:**

```python
metrics = cache.get_cached_metrics()
if metrics:
    print(f"Current epoch: {metrics['epoch']}")
```

---

#### cache_topology

```python
cache_topology(topology: Dict)
```

Cache network topology with 5-second TTL.

**Parameters:**

- `topology` (Dict): Network topology dictionary

**Example:**

```python
topology = {
    "input_size": 2,
    "output_size": 1,
    "hidden_units": [...]
}
cache.cache_topology(topology)
```

---

#### get_cached_topology

```python
get_cached_topology() -> Optional[Dict]
```

Get cached network topology.

**Returns:**

- `Dict`: Network topology if cached
- `None`: If not cached or expired

---

#### publish

```python
publish(channel: str, message: Dict)
```

Publish message to Redis Pub/Sub channel.

**Parameters:**

- `channel` (str): Channel name
- `message` (Dict): Message data (JSON-serializable)

**Example:**

```python
cache.publish("cascor:training", {
    "type": "metrics_update",
    "epoch": 5,
    "loss": 0.2
})
```

---

#### subscribe

```python
subscribe(channels: List[str], callback: Callable) -> Thread
```

Subscribe to Redis Pub/Sub channels.

**Parameters:**

- `channels` (List[str]): List of channel names
- `callback` (Callable): Callback function(channel, message)

**Returns:**

- `Thread`: Listener thread (daemon)

**Example:**

```python
def on_message(channel, message):
    data = json.loads(message['data'])
    print(f"Received on {channel}: {data}")

thread = cache.subscribe(["cascor:training"], on_message)
```

---

#### close

```python
close()
```

Close Redis connection and cleanup resources.

**Example:**

```python
cache.close()
```

---

### Factory Function

#### get_cache_manager

```python
get_cache_manager() -> CacheManager
```

Get singleton CacheManager instance.

**Returns:**

- `CacheManager`: Singleton instance

**Example:**

```python
from backend.cache_manager import get_cache_manager

cache = get_cache_manager()
metrics = cache.get_cached_metrics()
```

---

## Cache Keys

### Key Naming Convention

**Format:** `<prefix>:<category>:<identifier>`

**Examples:**

```bash
cascor:metrics:latest
cascor:metrics:epoch_5
cascor:topology:latest
cascor:topology:epoch_10
cascor:history:full
cascor:session:user_123
```

### Key Prefixes

| Prefix             | Purpose          | TTL | Example                  |
| ------------------ | ---------------- | --- | ------------------------ |
| `cascor:metrics:`  | Training metrics | 1s  | `cascor:metrics:latest`  |
| `cascor:topology:` | Network topology | 5s  | `cascor:topology:latest` |
| `cascor:history:`  | Training history | 60s | `cascor:history:full`    |
| `cascor:session:`  | User sessions    | 24h | `cascor:session:abc123`  |
| `cascor:control:`  | Control commands | 5s  | `cascor:control:latest`  |

### Key Patterns

**Glob patterns for invalidation:**

```python
# All metrics
"cascor:metrics:*"

# Specific epoch metrics
"cascor:metrics:epoch_*"

# All topology
"cascor:topology:*"

# All data
"cascor:*"
```

---

## TTL Strategies

### Default TTL Values

| Data Type | TTL | Reason                  |
| --------- | --- | ----------------------- |
| Metrics   | 1s  | Real-time updates       |
| Topology  | 5s  | Changes less frequently |
| History   | 60s | Expensive to compute    |
| Sessions  | 24h | Long-lived user data    |
| Control   | 5s  | Command responses       |

### Custom TTL

**Override default TTL:**

```python
# Short TTL for real-time data
cache.set_cached("cascor:metrics:latest", data, ttl=1)

# Long TTL for static data
cache.set_cached("cascor:config:app", data, ttl=3600)

# No expiration (persistent)
cache.set_cached("cascor:static:info", data, ttl=0)
```

### TTL Best Practices

1. **Real-time data:** 1-5 seconds
2. **Computed data:** 30-60 seconds
3. **Static data:** 1-24 hours
4. **Session data:** 24 hours - 7 days
5. **Temporary data:** 5-30 seconds

---

## Pub/Sub Channels

### Channel Names

| Channel           | Purpose          | Message Type       |
| ----------------- | ---------------- | ------------------ |
| `cascor:training` | Training events  | Metrics updates    |
| `cascor:control`  | Control commands | Pause/resume/reset |
| `cascor:topology` | Topology changes | Network structure  |
| `cascor:events`   | General events   | System events      |

### Message Format

**Standard message structure:**

```json
{
  "type": "string",
  "timestamp": "ISO 8601 timestamp",
  "data": {
    // Type-specific payload
  }
}
```

### Message Types

**Training channel (`cascor:training`):**

```json
{
  "type": "metrics_update",
  "timestamp": "2025-11-05T12:00:00",
  "data": {
    "epoch": 5,
    "train_loss": 0.2,
    "train_accuracy": 0.95
  }
}
```

**Control channel (`cascor:control`):**

```json
{
  "type": "command",
  "timestamp": "2025-11-05T12:00:00",
  "data": {
    "command": "pause",
    "params": {}
  }
}
```

**Topology channel (`cascor:topology`):**

```json
{
  "type": "topology_update",
  "timestamp": "2025-11-05T12:00:00",
  "data": {
    "hidden_units": 5,
    "connections": 25
  }
}
```

---

## Error Handling

### Exception Types

**redis-py exceptions:**

| Exception             | Cause                | Recovery                             |
| --------------------- | -------------------- | ------------------------------------ |
| `ConnectionError`     | Redis not reachable  | Retry connection, fallback to memory |
| `TimeoutError`        | Operation timeout    | Retry with longer timeout            |
| `AuthenticationError` | Invalid password     | Check credentials                    |
| `ResponseError`       | Redis error response | Check command syntax                 |
| `DataError`           | Invalid data type    | Validate data before caching         |

### Error Handling Pattern

```python
from redis.exceptions import ConnectionError, TimeoutError

try:
    cache.set_cached("key", data)
except ConnectionError as e:
    logger.error(f"Redis connection failed: {e}")
    # Fallback to in-memory cache
    memory_cache[key] = data
except TimeoutError as e:
    logger.warning(f"Redis timeout: {e}")
    # Retry with longer timeout
    cache.set_cached("key", data, ttl=10)
except Exception as e:
    logger.error(f"Unexpected error: {e}", exc_info=True)
    # Fail gracefully
```

### Fallback Strategy

**Graceful degradation when Redis unavailable:**

```python
class CacheManager:
    def __init__(self):
        self.redis_available = False
        self.memory_cache = {}

        try:
            self.redis_client = redis.Redis(...)
            self.redis_client.ping()
            self.redis_available = True
        except:
            logger.warning("Redis unavailable, using memory cache")

    def get_cached(self, key):
        if self.redis_available:
            try:
                return self.redis_client.get(key)
            except:
                logger.warning("Redis error, falling back to memory")
                self.redis_available = False

        # Fallback to memory
        return self.memory_cache.get(key)
```

---

## Performance

### Benchmarks (Expected)

**Operation latencies:**

| Operation | Latency | Throughput    |
| --------- | ------- | ------------- |
| GET       | <1ms    | 100,000 ops/s |
| SET       | <1ms    | 80,000 ops/s  |
| PUBLISH   | <1ms    | 50,000 msgs/s |
| SUBSCRIBE | <5ms    | N/A           |

**Network overhead:**

- Local Redis: ~0.1-0.5ms
- Remote Redis (same datacenter): ~1-2ms
- Remote Redis (cross-region): ~10-50ms

### Optimization Strategies

**1. Connection Pooling:**

```python
pool = ConnectionPool(
    max_connections=50,
    socket_keepalive=True
)
```

**2. Pipelining:**

```python
pipe = redis_client.pipeline()
pipe.set("key1", "value1")
pipe.set("key2", "value2")
pipe.execute()
```

**3. Batch Operations:**

```python
# Use MSET for multiple keys
redis_client.mset({
    "cascor:metrics:epoch_1": data1,
    "cascor:metrics:epoch_2": data2
})
```

**4. Compression:**

```python
import gzip
import json

def set_compressed(key, data, ttl):
    serialized = json.dumps(data).encode('utf-8')
    compressed = gzip.compress(serialized)
    redis_client.setex(key, ttl, compressed)
```

### Memory Management

**Monitor memory usage:**

```bash
# Check memory
redis-cli info memory

# Set max memory
redis-cli config set maxmemory 256mb

# Set eviction policy
redis-cli config set maxmemory-policy allkeys-lru
```

**Eviction policies:**

- `noeviction`: Return errors when memory limit reached
- `allkeys-lru`: Evict least recently used keys
- `volatile-lru`: Evict LRU keys with TTL
- `allkeys-random`: Evict random keys
- `volatile-ttl`: Evict keys with shortest TTL

**Recommended:** `allkeys-lru` for cache workloads

---

## Security

### Authentication

**Password authentication:**

```bash
# Set password in redis.conf
requirepass yourpassword

# Or via command
redis-cli config set requirepass yourpassword
```

**Connect with password:**

```python
redis_url = "redis://:yourpassword@localhost:6379/0"
cache = CacheManager(redis_url=redis_url)
```

### TLS/SSL

**Enable TLS:**

```python
import redis

pool = redis.ConnectionPool(
    host='redis.example.com',
    port=6380,
    password='yourpassword',
    ssl=True,
    ssl_cert_reqs='required',
    ssl_ca_certs='/path/to/ca.crt'
)

redis_client = redis.Redis(connection_pool=pool)
```

### Access Control Lists (Redis 6+)

**Create user with limited permissions:**

```bash
# Create read-only user
ACL SETUSER readonly on >password ~cascor:* +get

# Create write user
ACL SETUSER writer on >password ~cascor:* +get +set +del
```

### Network Security

**Bind to specific interface:**

```conf
# redis.conf
bind 127.0.0.1  # Localhost only
# bind 0.0.0.0  # All interfaces (insecure)
```

**Firewall rules:**

```bash
# Allow only specific IPs
sudo ufw allow from 192.168.1.0/24 to any port 6379
```

---

## Monitoring

### Health Checks

```python
def health_check():
    """Check Redis health."""
    try:
        cache = get_cache_manager()
        cache.redis_client.ping()
        return {"redis": "healthy"}
    except Exception as e:
        return {"redis": "unhealthy", "error": str(e)}
```

### Metrics

**Track cache performance:**

```python
import time

class CacheManager:
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.errors = 0

    def get_cached(self, key):
        start = time.time()
        try:
            data = self.redis_client.get(key)
            latency = time.time() - start

            if data:
                self.hits += 1
                logger.debug(f"Cache hit: {key} ({latency*1000:.2f}ms)")
            else:
                self.misses += 1
                logger.debug(f"Cache miss: {key}")

            return data
        except Exception as e:
            self.errors += 1
            logger.error(f"Cache error: {e}")
            return None

    def get_stats(self):
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "errors": self.errors,
            "hit_rate": hit_rate
        }
```

---

## Additional Resources

- **[REDIS_INTEGRATION_QUICK_START.md](REDIS_INTEGRATION_QUICK_START.md)** - Quick setup guide
- **[REDIS_INTEGRATION_MANUAL.md](REDIS_INTEGRATION_MANUAL.md)** - Complete implementation guide
- **[Redis Documentation](https://redis.io/documentation)** - Official Redis docs
- **[redis-py API Reference](https://redis-py.readthedocs.io/en/stable/)** - Python client API

---

**Last Updated:** November 5, 2025  
**Version:** 0.1.0  
**Status:** ⚠️ TECHNICAL REFERENCE

**Ready to implement? See [REDIS_INTEGRATION_MANUAL.md](REDIS_INTEGRATION_MANUAL.md) for implementation guide!**
