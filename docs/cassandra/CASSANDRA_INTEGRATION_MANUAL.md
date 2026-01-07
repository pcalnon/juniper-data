# Cassandra Integration Manual

## Complete guide for implementing Cassandra integration in Juniper Canopy

**Version:** 0.1.0 - PLANNED FEATURE  
**Status:** ⚠️ NOT YET IMPLEMENTED  
**Last Updated:** November 5, 2025

---

## ⚠️ Important Notice

**This feature is currently PLANNED but NOT YET IMPLEMENTED.**

This manual provides implementation guidance for developers who want to add Cassandra support for time-series metrics storage.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Schema Design](#schema-design)
- [Implementation Plan](#implementation-plan)
- [Cassandra Manager](#cassandra-manager)
- [Data Models](#data-models)
- [API Integration](#api-integration)
- [Testing](#testing)
- [Deployment](#deployment)
- [Performance Tuning](#performance-tuning)

---

## Overview

### Purpose

Cassandra integration will provide:

1. **Time-Series Storage** - Store millions of training data points
2. **Scalability** - Horizontal scaling for growing data volumes
3. **Historical Analysis** - Query training history efficiently
4. **High Availability** - Distributed architecture with no SPOF
5. **Long-Term Persistence** - Durable storage for all training runs

### Current State

**What exists:**

- Nothing - Cassandra is not mentioned in current configuration

**What's missing:**

- Configuration in `app_config.yaml`
- Cassandra manager implementation
- Schema and data models
- API endpoints for historical queries
- Tests

### Comparison with Redis

| Aspect             | Redis                 | Cassandra                  |
| ------------------ | --------------------- | -------------------------- |
| **Purpose**        | Real-time caching     | Long-term storage          |
| **Data Model**     | Key-value             | Wide-column                |
| **Query**          | Simple GET/SET        | CQL (SQL-like)             |
| **Persistence**    | Optional (AOF/RDB)    | Always persistent          |
| **Scalability**    | Vertical              | Horizontal                 |
| **Consistency**    | Strong                | Tunable                    |
| **TTL**            | Per-key               | Per-row                    |
| **Use in Juniper** | Current metrics cache | Historical metrics storage |

**Recommended:** Use both - Redis for real-time, Cassandra for long-term.

---

## Architecture

### Component Overview

```bash
┌──────────────────────────────────────────────────────────────┐
│                  Frontend Dashboard                          │
└─────────────────────┬────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────────┐
│                 FastAPI Backend                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │           REST API Endpoints                           │  │
│  │  /api/runs, /api/metrics/range, /api/export            │  │
│  └─────────────┬──────────────────────────────────────────┘  │
│                │                                             │
│    ┌───────────┴──────────┐                                  │
│    ▼                      ▼                                  │
│  ┌─────────────┐   ┌────────────────┐                        │
│  │ Redis Cache │   │ CassandraManager│                       │
│  │ (Real-time) │   │  (Historical)   │                       │
│  └─────────────┘   └────────┬────────┘                       │
└─────────────────────────────┼────────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │    Cassandra Cluster          │
              │  ┌──────────────────────────┐ │
              │  │  Keyspace: cascor        │ │
              │  │                          │ │
              │  │  Tables:                 │ │
              │  │  - training_metrics      │ │
              │  │  - network_topology      │ │
              │  │  - training_runs         │ │
              │  │  - dataset_info          │ │
              │  └──────────────────────────┘ │
              │                               │
              │  Nodes: Node1, Node2, Node3   │
              │  Replication Factor: 3        │
              └───────────────────────────────┘
```

### Data Flow

**1. Training Metrics Storage:**

```bash
Training Update → CassandraManager → Batch Buffer → Flush → Cassandra
                                          ↓ (every 100 metrics or 5s)
                                    INSERT INTO training_metrics
```

**2. Historical Query:**

```bash
API Request → CassandraManager → CQL Query → Cassandra → Results → Response
```

**3. Dual Storage (Redis + Cassandra):**

```bash
Training Update → Redis (cache, TTL=60s)
               └→ Cassandra (persist forever)

Dashboard Request → Redis (try cache first)
                 └→ Cassandra (fallback for historical)
```

---

## Schema Design

### Keyspace

```sql
CREATE KEYSPACE IF NOT EXISTS cascor
  WITH replication = {
    'class': 'NetworkTopologyStrategy',
    'datacenter1': 3,
    'datacenter2': 2
  }
  AND durable_writes = true;

USE cascor;
```

**Replication strategy:**

- **Development:** SimpleStrategy with RF=1
- **Production:** NetworkTopologyStrategy with RF=3

---

### Table: training_metrics

**Time-series table for training metrics:**

```sql
CREATE TABLE training_metrics (
    run_id UUID,                  -- Training run identifier
    timestamp TIMESTAMP,          -- Metric timestamp
    epoch INT,                    -- Training epoch
    step INT,                     -- Training step
    train_loss DOUBLE,            -- Training loss
    train_accuracy DOUBLE,        -- Training accuracy
    val_loss DOUBLE,              -- Validation loss
    val_accuracy DOUBLE,          -- Validation accuracy
    learning_rate DOUBLE,         -- Learning rate
    hidden_units INT,             -- Number of hidden units
    PRIMARY KEY ((run_id), timestamp)
) WITH CLUSTERING ORDER BY (timestamp DESC)
  AND compaction = {
    'class': 'TimeWindowCompactionStrategy',
    'compaction_window_size': 1,
    'compaction_window_unit': 'DAYS'
  }
  AND default_time_to_live = 2592000;  -- 30 days TTL
```

**Partition key:** `run_id` - All metrics for a run in same partition  
**Clustering key:** `timestamp` - Ordered by time (descending)  
**TTL:** 30 days (configurable)

---

### Table: network_topology

**Network structure snapshots:**

```sql
CREATE TABLE network_topology (
    run_id UUID,
    timestamp TIMESTAMP,
    epoch INT,
    input_size INT,
    output_size INT,
    hidden_units INT,
    total_parameters INT,
    topology_json TEXT,           -- Full topology as JSON
    weights_summary TEXT,         -- Weight statistics
    PRIMARY KEY ((run_id), timestamp)
) WITH CLUSTERING ORDER BY (timestamp DESC)
  AND default_time_to_live = 7776000;  -- 90 days TTL
```

---

### Table: training_runs

**Training run metadata:**

```sql
CREATE TABLE training_runs (
    run_id UUID PRIMARY KEY,
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    duration_seconds INT,
    status TEXT,                  -- 'running', 'completed', 'failed'
    dataset_name TEXT,
    dataset_size INT,
    config_json TEXT,             -- Full configuration
    final_epoch INT,
    final_train_loss DOUBLE,
    final_train_accuracy DOUBLE,
    final_val_loss DOUBLE,
    final_val_accuracy DOUBLE,
    best_val_accuracy DOUBLE,
    best_epoch INT,
    total_hidden_units INT,
    total_parameters INT,
    notes TEXT,
    tags SET<TEXT>,               -- User-defined tags
    created_by TEXT,              -- User identifier
    PRIMARY KEY (run_id)
);

-- Secondary index for querying by dataset
CREATE INDEX ON training_runs (dataset_name);

-- Secondary index for status
CREATE INDEX ON training_runs (status);
```

---

### Table: dataset_info

**Dataset metadata:**

```sql
CREATE TABLE dataset_info (
    dataset_id UUID PRIMARY KEY,
    name TEXT,
    description TEXT,
    num_samples INT,
    num_features INT,
    num_classes INT,
    class_distribution MAP<TEXT, INT>,
    created_at TIMESTAMP,
    file_path TEXT,
    checksum TEXT
);
```

---

## Implementation Plan

### Phase 1: Basic Storage (P2)

**Goal:** Store metrics and runs in Cassandra

**Tasks:**

1. **Add Cassandra configuration** to `app_config.yaml`
2. **Create cassandra_manager.py**
3. **Implement connection management**
4. **Implement batch writes** for metrics
5. **Add basic tests**

**Estimated effort:** 6-8 hours

---

### Phase 2: Query API (P3)

**Goal:** Add API endpoints for historical queries

**Tasks:**

1. **Add `/api/runs` endpoint** - List all runs
2. **Add `/api/runs/<id>` endpoint** - Get run details
3. **Add `/api/metrics/range` endpoint** - Query time range
4. **Add pagination support**
5. **Add integration tests**

**Estimated effort:** 4-6 hours

---

### Phase 3: Advanced Features (P4)

**Goal:** Analytics and export

**Tasks:**

1. **Add aggregation queries** (min/max/avg metrics)
2. **Add export endpoints** (CSV, JSON)
3. **Add comparison API** (compare multiple runs)
4. **Add dashboard widgets** for historical analysis
5. **Performance optimization**

**Estimated effort:** 8-10 hours

---

## Cassandra Manager

### File Structure

```bash
src/backend/
├── __init__.py
├── cassandra_manager.py     # NEW - Main Cassandra manager
├── cassandra_models.py      # NEW - Data models
├── cassandra_queries.py     # NEW - CQL queries
└── tests/
    ├── test_cassandra_manager.py
    └── test_cassandra_queries.py
```

### cassandra_manager.py

**Implementation skeleton:**

```python
#####################################################################################################################################################################################################
# Project:       Juniper
# Sub-Project:   JuniperCanopy
# Application:   juniper_canopy
# Purpose:       Monitoring and Diagnostic Frontend for Cascade Correlation Neural Network
#
# Author:        Paul Calnon
# Version:       0.1.0
# File Name:     cassandra_manager.py
# File Path:     <Project>/<Sub-Project>/<Application>/src/backend/
#
# Created Date:  2025-11-05
# Last Modified: <date last changed>
#
# License:       MIT License
# Copyright:     Copyright (c) 2024,2025,2026 Paul Calnon
#
# Description:
# Description:
#     Cassandra manager for time-series metrics storage
#
#####################################################################################################################################################################################################
# Notes:
#     <Additional information about the script>
#
#####################################################################################################################################################################################################
# References:
#     <External information sources or documentation relevant to the script>
#
#####################################################################################################################################################################################################
# TODO :
#     <List of pending tasks or improvements for the script>
#
#####################################################################################################################################################################################################
# COMPLETED:
#     <List of completed tasks or features for the script>
#
#####################################################################################################################################################################################################
import logging
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any
from collections import deque

from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from cassandra.query import SimpleStatement, BatchStatement, ConsistencyLevel

from config_manager import get_config


class CassandraManager:
    """
    Cassandra manager for CasCor time-series storage.

    Provides:
    - Connection management
    - Batch write operations
    - Historical queries
    - Run metadata management
    """

    def __init__(self, contact_points: Optional[List[str]] = None):
        """
        Initialize Cassandra manager.

        Args:
            contact_points: List of Cassandra node addresses
        """
        self.logger = logging.getLogger(__name__)
        self.config = get_config()

        # Get configuration
        self.contact_points = contact_points or self.config.get(
            "backend.cassandra.contact_points",
            ["localhost"]
        )
        self.port = self.config.get("backend.cassandra.port", 9042)
        self.keyspace = self.config.get("backend.cassandra.keyspace", "cascor")

        # Authentication (if enabled)
        auth_enabled = self.config.get("backend.cassandra.auth_provider.enabled", False)
        if auth_enabled:
            username = self.config.get("backend.cassandra.auth_provider.username")
            password = self.config.get("backend.cassandra.auth_provider.password")
            auth_provider = PlainTextAuthProvider(username, password)
        else:
            auth_provider = None

        # Create cluster
        self.cluster = Cluster(
            contact_points=self.contact_points,
            port=self.port,
            auth_provider=auth_provider
        )

        # Connect to cluster
        self.session = self.cluster.connect()
        self.session.set_keyspace(self.keyspace)

        # Batch settings
        self.batch_size = self.config.get("backend.cassandra.batch_size", 100)
        self.flush_interval = self.config.get("backend.cassandra.flush_interval", 5)

        # Batch buffer
        self.metrics_buffer = deque(maxlen=1000)
        self.last_flush = datetime.now()

        # Prepared statements
        self._prepare_statements()

        self.logger.info(f"Cassandra connection established: {self.contact_points}")

    def _prepare_statements(self):
        """Prepare frequently used statements."""
        # Insert metrics
        self.insert_metric_stmt = self.session.prepare(
            """
            INSERT INTO training_metrics (
                run_id, timestamp, epoch, step,
                train_loss, train_accuracy, val_loss, val_accuracy,
                learning_rate, hidden_units
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            USING TTL ?
            """
        )

        # Insert run metadata
        self.insert_run_stmt = self.session.prepare(
            """
            INSERT INTO training_runs (
                run_id, start_time, status, dataset_name, config_json
            ) VALUES (?, ?, ?, ?, ?)
            """
        )

        # Update run on completion
        self.update_run_stmt = self.session.prepare(
            """
            UPDATE training_runs SET
                end_time = ?,
                duration_seconds = ?,
                status = ?,
                final_epoch = ?,
                final_train_loss = ?,
                final_train_accuracy = ?,
                final_val_loss = ?,
                final_val_accuracy = ?
            WHERE run_id = ?
            """
        )

    # ==================== Metrics Storage ====================

    def store_metric(self, run_id: uuid.UUID, metric: Dict[str, Any]):
        """
        Store training metric (buffered).

        Args:
            run_id: Training run UUID
            metric: Metric dictionary
        """
        self.metrics_buffer.append((run_id, metric))

        # Flush if buffer full or interval exceeded
        if len(self.metrics_buffer) >= self.batch_size:
            self.flush_metrics()
        elif (datetime.now() - self.last_flush).total_seconds() > self.flush_interval:
            self.flush_metrics()

    def flush_metrics(self):
        """Flush metrics buffer to Cassandra."""
        if not self.metrics_buffer:
            return

        try:
            batch = BatchStatement(consistency_level=ConsistencyLevel.ONE)

            for run_id, metric in self.metrics_buffer:
                batch.add(
                    self.insert_metric_stmt,
                    (
                        run_id,
                        metric.get('timestamp', datetime.now()),
                        metric.get('epoch', 0),
                        metric.get('step', 0),
                        metric.get('train_loss'),
                        metric.get('train_accuracy'),
                        metric.get('val_loss'),
                        metric.get('val_accuracy'),
                        metric.get('learning_rate'),
                        metric.get('hidden_units'),
                        2592000  # TTL: 30 days
                    )
                )

            self.session.execute(batch)
            count = len(self.metrics_buffer)
            self.metrics_buffer.clear()
            self.last_flush = datetime.now()

            self.logger.debug(f"Flushed {count} metrics to Cassandra")

        except Exception as e:
            self.logger.error(f"Failed to flush metrics: {e}", exc_info=True)

    # ==================== Run Management ====================

    def create_run(
        self,
        run_id: Optional[uuid.UUID] = None,
        dataset_name: str = "unknown",
        config: Optional[Dict] = None
    ) -> uuid.UUID:
        """
        Create new training run.

        Args:
            run_id: Run UUID (auto-generated if None)
            dataset_name: Dataset name
            config: Training configuration

        Returns:
            Run UUID
        """
        run_id = run_id or uuid.uuid4()

        try:
            import json
            config_json = json.dumps(config) if config else "{}"

            self.session.execute(
                self.insert_run_stmt,
                (run_id, datetime.now(), 'running', dataset_name, config_json)
            )

            self.logger.info(f"Created training run: {run_id}")
            return run_id

        except Exception as e:
            self.logger.error(f"Failed to create run: {e}", exc_info=True)
            raise

    def complete_run(
        self,
        run_id: uuid.UUID,
        final_metrics: Dict[str, Any],
        status: str = 'completed'
    ):
        """
        Mark run as completed and update final metrics.

        Args:
            run_id: Run UUID
            final_metrics: Final training metrics
            status: Run status ('completed' or 'failed')
        """
        try:
            # Get run start time
            query = "SELECT start_time FROM training_runs WHERE run_id = ?"
            row = self.session.execute(query, (run_id,)).one()

            if row:
                start_time = row.start_time
                duration = (datetime.now() - start_time).total_seconds()
            else:
                duration = 0

            # Update run
            self.session.execute(
                self.update_run_stmt,
                (
                    datetime.now(),
                    int(duration),
                    status,
                    final_metrics.get('epoch', 0),
                    final_metrics.get('train_loss'),
                    final_metrics.get('train_accuracy'),
                    final_metrics.get('val_loss'),
                    final_metrics.get('val_accuracy'),
                    run_id
                )
            )

            self.logger.info(f"Completed run: {run_id} ({status})")

        except Exception as e:
            self.logger.error(f"Failed to complete run: {e}", exc_info=True)

    # ==================== Queries ====================

    def get_run(self, run_id: uuid.UUID) -> Optional[Dict]:
        """Get training run metadata."""
        try:
            query = "SELECT * FROM training_runs WHERE run_id = ?"
            row = self.session.execute(query, (run_id,)).one()

            if row:
                return self._row_to_dict(row)
            return None

        except Exception as e:
            self.logger.error(f"Failed to get run: {e}", exc_info=True)
            return None

    def list_runs(
        self,
        limit: int = 100,
        status: Optional[str] = None
    ) -> List[Dict]:
        """
        List training runs.

        Args:
            limit: Maximum number of runs to return
            status: Filter by status ('running', 'completed', 'failed')

        Returns:
            List of run dictionaries
        """
        try:
            if status:
                query = f"SELECT * FROM training_runs WHERE status = ? LIMIT {limit}"
                rows = self.session.execute(query, (status,))
            else:
                query = f"SELECT * FROM training_runs LIMIT {limit}"
                rows = self.session.execute(query)

            return [self._row_to_dict(row) for row in rows]

        except Exception as e:
            self.logger.error(f"Failed to list runs: {e}", exc_info=True)
            return []

    def query_metrics(
        self,
        run_id: uuid.UUID,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[Dict]:
        """
        Query metrics for time range.

        Args:
            run_id: Run UUID
            start_time: Start of time range
            end_time: End of time range
            limit: Maximum results

        Returns:
            List of metric dictionaries
        """
        try:
            if start_time and end_time:
                query = """
                    SELECT * FROM training_metrics
                    WHERE run_id = ? AND timestamp >= ? AND timestamp <= ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """
                rows = self.session.execute(query, (run_id, start_time, end_time, limit))
            else:
                query = """
                    SELECT * FROM training_metrics
                    WHERE run_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """
                rows = self.session.execute(query, (run_id, limit))

            return [self._row_to_dict(row) for row in rows]

        except Exception as e:
            self.logger.error(f"Failed to query metrics: {e}", exc_info=True)
            return []

    # ==================== Utilities ====================

    def _row_to_dict(self, row) -> Dict:
        """Convert Cassandra row to dictionary."""
        return {key: getattr(row, key) for key in row._fields}

    def close(self):
        """Close Cassandra connection."""
        try:
            self.flush_metrics()  # Flush remaining
            self.cluster.shutdown()
            self.logger.info("Cassandra connection closed")
        except Exception as e:
            self.logger.error(f"Error closing connection: {e}")

    def __del__(self):
        """Destructor: ensure cleanup."""
        try:
            self.close()
        except:
            pass


# ==================== Factory ====================

_cassandra_manager_instance = None


def get_cassandra_manager() -> CassandraManager:
    """Get singleton Cassandra manager instance."""
    global _cassandra_manager_instance
    if _cassandra_manager_instance is None:
        _cassandra_manager_instance = CassandraManager()
    return _cassandra_manager_instance
```

---

## Data Models

### cassandra_models.py

```python
#!/usr/bin/env python
#####################################################################
# Project:       Juniper
# File Name:     cassandra_models.py
# Description:   Data models for Cassandra storage
#####################################################################

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Dict, Any, List
import uuid


@dataclass
class TrainingMetric:
    """Training metric data point."""
    run_id: uuid.UUID
    timestamp: datetime
    epoch: int
    step: int
    train_loss: Optional[float] = None
    train_accuracy: Optional[float] = None
    val_loss: Optional[float] = None
    val_accuracy: Optional[float] = None
    learning_rate: Optional[float] = None
    hidden_units: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class TrainingRun:
    """Training run metadata."""
    run_id: uuid.UUID
    start_time: datetime
    status: str  # 'running', 'completed', 'failed'
    dataset_name: str
    config_json: str
    end_time: Optional[datetime] = None
    duration_seconds: Optional[int] = None
    final_epoch: Optional[int] = None
    final_train_loss: Optional[float] = None
    final_train_accuracy: Optional[float] = None
    final_val_loss: Optional[float] = None
    final_val_accuracy: Optional[float] = None
    best_val_accuracy: Optional[float] = None
    best_epoch: Optional[int] = None
    total_hidden_units: Optional[int] = None
    total_parameters: Optional[int] = None
    notes: Optional[str] = None
    tags: Optional[List[str]] = None
    created_by: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class NetworkTopologySnapshot:
    """Network topology snapshot."""
    run_id: uuid.UUID
    timestamp: datetime
    epoch: int
    input_size: int
    output_size: int
    hidden_units: int
    total_parameters: int
    topology_json: str
    weights_summary: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
```

---

## API Integration

### Add Endpoints to main.py

```python
from backend.cassandra_manager import get_cassandra_manager

# ==================== Historical Query Endpoints ====================

@app.get("/api/runs")
async def list_training_runs(limit: int = 100, status: Optional[str] = None):
    """
    List training runs.

    Args:
        limit: Maximum number of runs
        status: Filter by status

    Returns:
        List of training runs
    """
    cassandra = get_cassandra_manager()
    runs = cassandra.list_runs(limit=limit, status=status)
    return {"runs": runs, "count": len(runs)}


@app.get("/api/runs/{run_id}")
async def get_training_run(run_id: str):
    """
    Get training run details.

    Args:
        run_id: Run UUID

    Returns:
        Run metadata and metrics
    """
    try:
        run_uuid = uuid.UUID(run_id)
        cassandra = get_cassandra_manager()

        # Get run metadata
        run = cassandra.get_run(run_uuid)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")

        # Get metrics
        metrics = cassandra.query_metrics(run_uuid, limit=1000)

        return {
            "run": run,
            "metrics": metrics,
            "metric_count": len(metrics)
        }

    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid UUID")


@app.get("/api/metrics/range")
async def query_metrics_range(
    run_id: str,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    limit: int = 1000
):
    """
    Query metrics for time range.

    Args:
        run_id: Run UUID
        start_time: ISO timestamp (optional)
        end_time: ISO timestamp (optional)
        limit: Maximum results

    Returns:
        List of metrics
    """
    try:
        run_uuid = uuid.UUID(run_id)

        # Parse timestamps
        start_dt = datetime.fromisoformat(start_time) if start_time else None
        end_dt = datetime.fromisoformat(end_time) if end_time else None

        cassandra = get_cassandra_manager()
        metrics = cassandra.query_metrics(run_uuid, start_dt, end_dt, limit)

        return {"metrics": metrics, "count": len(metrics)}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
```

### Integrate with Demo Mode

```python
# In src/demo_mode.py

from backend.cassandra_manager import get_cassandra_manager
import uuid

class DemoMode:
    def __init__(self):
        # ... existing init ...

        # Cassandra integration
        self.cassandra_enabled = config.get("backend.cassandra.enabled", False)
        if self.cassandra_enabled:
            self.cassandra = get_cassandra_manager()
            self.run_id = self.cassandra.create_run(
                dataset_name="Demo Spiral Dataset",
                config={"mode": "demo", "max_epochs": 100}
            )
        else:
            self.cassandra = None
            self.run_id = None

    def _training_loop(self):
        """Training loop with Cassandra storage."""
        while not self._stop.is_set():
            # ... compute metrics ...

            # Store to Cassandra
            if self.cassandra:
                metric = {
                    'timestamp': datetime.now(),
                    'epoch': self.current_epoch,
                    'step': self.current_step,
                    'train_loss': loss,
                    'train_accuracy': accuracy,
                    'learning_rate': self.learning_rate,
                    'hidden_units': self.hidden_units
                }
                self.cassandra.store_metric(self.run_id, metric)

            # ... rest of loop ...

    def stop(self):
        """Stop demo mode and complete run."""
        self._stop.set()

        # Complete Cassandra run
        if self.cassandra and self.run_id:
            final_metrics = {
                'epoch': self.current_epoch,
                'train_loss': self.current_state.get('train_loss'),
                'train_accuracy': self.current_state.get('train_accuracy')
            }
            self.cassandra.complete_run(self.run_id, final_metrics)
```

---

## Testing

### Unit Tests

```python
# src/tests/unit/test_cassandra_manager.py

import pytest
import uuid
from datetime import datetime
from backend.cassandra_manager import CassandraManager

@pytest.fixture
def cassandra_manager():
    """Create Cassandra manager for testing."""
    manager = CassandraManager(contact_points=["localhost"])
    yield manager
    manager.close()


def test_create_run(cassandra_manager):
    """Test creating training run."""
    run_id = cassandra_manager.create_run(
        dataset_name="test_dataset",
        config={"learning_rate": 0.01}
    )

    assert isinstance(run_id, uuid.UUID)

    # Verify run was created
    run = cassandra_manager.get_run(run_id)
    assert run is not None
    assert run['dataset_name'] == "test_dataset"
    assert run['status'] == 'running'


def test_store_metric(cassandra_manager):
    """Test storing metric."""
    run_id = cassandra_manager.create_run(dataset_name="test")

    metric = {
        'epoch': 1,
        'step': 100,
        'train_loss': 0.5,
        'train_accuracy': 0.8
    }

    cassandra_manager.store_metric(run_id, metric)
    cassandra_manager.flush_metrics()

    # Query metrics
    metrics = cassandra_manager.query_metrics(run_id, limit=10)
    assert len(metrics) > 0


def test_query_time_range(cassandra_manager):
    """Test querying metrics by time range."""
    run_id = cassandra_manager.create_run(dataset_name="test")

    # Store metrics at different times
    for i in range(5):
        metric = {'epoch': i, 'train_loss': 0.5 - i*0.1}
        cassandra_manager.store_metric(run_id, metric)

    cassandra_manager.flush_metrics()

    # Query range
    now = datetime.now()
    start = now.replace(hour=0, minute=0, second=0)
    end = now

    metrics = cassandra_manager.query_metrics(run_id, start, end)
    assert len(metrics) == 5
```

---

## Deployment

### Docker Compose

```yaml
version: '3.8'

services:
  cassandra:
    image: cassandra:4.1
    container_name: cascor-cassandra
    ports:
      - "9042:9042"
    environment:
      - CASSANDRA_CLUSTER_NAME=CasCor Cluster
      - CASSANDRA_DC=datacenter1
      - CASSANDRA_ENDPOINT_SNITCH=GossipingPropertyFileSnitch
    volumes:
      - cassandra-data:/var/lib/cassandra
    healthcheck:
      test: ["CMD", "cqlsh", "-e", "describe keyspaces"]
      interval: 10s
      timeout: 5s
      retries: 5

  juniper_canopy:
    build: .
    ports:
      - "8050:8050"
    environment:
      - CASCOR_CASSANDRA_ENABLED=1
      - CASCOR_CASSANDRA_CONTACT_POINTS=cassandra
    depends_on:
      cassandra:
        condition: service_healthy

volumes:
  cassandra-data:
```

---

## Performance Tuning

### Batch Configuration

```yaml
cassandra:
  batch_size: 100        # Metrics per batch
  flush_interval: 5      # Seconds between flushes
```

**Tuning:**

- Larger batches → Better throughput, higher latency
- Smaller batches → Lower latency, more overhead

### Consistency Levels

```python
# Write consistency
ConsistencyLevel.ONE      # Fastest, least durable
ConsistencyLevel.QUORUM   # Balanced
ConsistencyLevel.ALL      # Slowest, most durable

# Read consistency
ConsistencyLevel.ONE      # Fastest, may be stale
ConsistencyLevel.QUORUM   # Balanced
```

### Compaction Strategy

```sql
-- Time-series data: TimeWindowCompactionStrategy
ALTER TABLE training_metrics
WITH compaction = {
  'class': 'TimeWindowCompactionStrategy',
  'compaction_window_size': 1,
  'compaction_window_unit': 'DAYS'
};
```

---

## Additional Resources

- **[CASSANDRA_INTEGRATION_QUICK_START.md](CASSANDRA_INTEGRATION_QUICK_START.md)** - Quick setup
- **[CASSANDRA_INTEGRATION_REFERENCE.md](CASSANDRA_INTEGRATION_REFERENCE.md)** - Technical reference
- **[Cassandra Documentation](https://cassandra.apache.org/doc/)** - Official docs
- **[DataStax Python Driver](https://docs.datastax.com/en/developer/python-driver/)** - Python client

---

**Last Updated:** November 5, 2025  
**Version:** 0.1.0  
**Status:** ⚠️ IMPLEMENTATION GUIDE

**Ready to implement? Start with Phase 1 and create `src/backend/cassandra_manager.py`!**
