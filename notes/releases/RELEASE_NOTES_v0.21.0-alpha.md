# Juniper Canopy v0.21.0-alpha Release Notes

**Release Date:** 2026-01-09  
**Version:** 0.21.0-alpha  
**Codename:** Phase 3 Wave 1 & 2 - HDF5 Snapshot Capabilities

## Overview

This release completes **Phase 3 Wave 1 and Wave 2** of the Juniper Canopy development roadmap, delivering comprehensive HDF5 snapshot management capabilities and visualization enhancements. Users can now create, restore, and track the history of training state snapshots, as well as view network topology in 3D and save/load custom metric layouts.

> **Status:** ALPHA – Feature-complete for Wave 1 & 2. Wave 3 (Redis/Cassandra integrations) not yet started.

---

## Phase 3 Features Summary

| ID   | Feature                     | Status   | Version |
| ---- | --------------------------- | -------- | ------- |
| P3-1 | Create New Snapshot         | ✅ Done  | 0.19.0  |
| P3-2 | Restore from Snapshot       | ✅ Done  | 0.20.0  |
| P3-3 | Snapshot History            | ✅ Done  | 0.20.0  |
| P3-4 | Metrics Save/Load Buttons   | ✅ Done  | 0.20.0  |
| P3-5 | 3D Interactive Topology     | ✅ Done  | 0.20.0  |
| P3-6 | Redis Integration Tab       | Planned  | -       |
| P3-7 | Cassandra Integration Tab   | Planned  | -       |

---

## What's New

### HDF5 Snapshot Management (P3-1, P3-2, P3-3)

#### Create New Snapshot (P3-1)

- New "Create Snapshot" section in HDF5 Snapshots panel
- Name input field (optional, auto-generates timestamp-based name if empty)
- Description input field (optional)
- "📸 Create Snapshot" button with success/error feedback
- Demo mode creates session-persistent mock snapshots
- Real mode creates HDF5 files via h5py or `cascor_integration`
- New API endpoint: `POST /api/v1/snapshots` (returns 201 Created)

#### Restore from Existing Snapshot (P3-2)

- "🔄 Restore" button added to each snapshot row in table
- Confirmation modal with warning about training state requirements
- Validates training is paused/stopped before restore (409 Conflict if running)
- Demo mode simulates restore by resetting training state
- Real mode loads from HDF5 file via h5py or `cascor_integration`
- Broadcasts state change via WebSocket after restore
- New API endpoint: `POST /api/v1/snapshots/{snapshot_id}/restore`

#### Snapshot History (P3-3)

- New "📜 Snapshot History" collapsible section in HDF5 panel
- Toggle button with arrow indicator (▼/▲)
- Action types with icons and color coding:
  - 📸 CREATE (green)
  - 🔄 RESTORE (yellow)
  - 🗑️ DELETE (red)
- Displays snapshot ID, timestamp, and message for each entry
- History sorted by recency (newest first)
- New API endpoint: `GET /api/v1/snapshots/history`

### Visualization Enhancements (P3-4, P3-5)

#### Metrics Save/Load Buttons (P3-4)

- Save button to persist current training metrics layout
- Load dropdown to select and apply saved layouts
- Delete button to remove saved layouts
- Layouts persist to `conf/layouts/` directory
- Success/error messages displayed for all operations

#### 3D Interactive Topology View (P3-5)

- 2D/3D view mode toggle radio buttons in Network Topology panel
- Layer-based z-axis positioning (Input → Hidden → Output)
- Color-coded nodes by layer (Green/Teal/Red)
- Weight-based edge coloring (Blue negative, Red positive)
- Interactive camera rotation, zoom, and pan
- Dark/light theme support
- Default view remains 2D (existing functionality preserved)

---

## Bug Fixes

### UnboundLocalError in open_restore_modal callback (v0.21.0)

**Problem:** `json` import was inside `contextlib.suppress` block but referenced in the `with` statement, causing `UnboundLocalError`.

**Solution:** Moved `import json` before the `with contextlib.suppress(...)` statement.

**Files:** `src/frontend/components/hdf5_snapshots_panel.py` (lines 893-896)

### Missing contextlib import (v0.21.0)

**Problem:** `contextlib.suppress` was used but `contextlib` was not imported.

**Solution:** Added `import contextlib` to module imports.

---

## Improvements

### Callback Testing Pattern

Introduced a new pattern for testing Dash callbacks by exposing them via `_cb_*` attributes. This enables direct unit testing without requiring a running Dash server. Pattern applied to:

- `HDF5SnapshotsPanel` (8 callbacks exposed)
- `AboutPanel` (2 callbacks exposed)

### Test Coverage

| Component               | Before | After | Change |
| ----------------------- | ------ | ----- | ------ |
| hdf5_snapshots_panel.py | 54%    | 95%   | +41%   |
| about_panel.py          | 73%    | 100%  | +27%   |
| **Overall**             | 93%    | 93%   | ±0%    |

### Test Suite

- **2413 tests** passing (up from 2270)
- **39 skipped** (requires external dependencies)
- **0 failures**
- **102 new tests** added across all versions:
  - P3-1: 23 tests (13 unit + 10 integration)
  - P3-2: 18 tests (9 unit + 9 integration)
  - P3-3: 16 tests (6 unit + 10 integration)
  - Callbacks: 45 tests (39 HDF5 + 6 About)

---

## API Changes

### New Endpoints

| Method | Endpoint                                  | Description           |
| ------ | ----------------------------------------- | --------------------- |
| `POST` | `/api/v1/snapshots`                       | Create new snapshot   |
| `POST` | `/api/v1/snapshots/{snapshot_id}/restore` | Restore from snapshot |
| `GET`  | `/api/v1/snapshots/history`               | Get activity history  |

### Response Codes

**POST /api/v1/snapshots:**

- `201 Created` - Snapshot created successfully
- `500 Internal Server Error` - Creation failed

**POST /api/v1/snapshots/{id}/restore:**

- `200 OK` - Restored successfully
- `404 Not Found` - Snapshot not found
- `409 Conflict` - Training is running (must pause/stop first)
- `500 Internal Server Error` - Restore failed

**GET /api/v1/snapshots/history:**

- `200 OK` - Returns history array (default limit: 50 entries)

---

## Upgrade Notes

This is a backward-compatible release. No migration steps required. All new endpoints are additive.

```bash
# Update and verify
git pull origin main
./demo

# Run test suite
cd src && pytest tests/ -v
```

---

## Known Issues

None. All Phase 3 Wave 1 and Wave 2 features have been implemented and verified.

---

## What's Next

Phase 3 Wave 3 development will focus on:

- **P3-6: Redis Integration Tab** - Redis cluster monitoring and usage statistics
- **P3-7: Cassandra Integration Tab** - Cassandra cluster monitoring and schema management

### Coverage Goals

- `main.py` currently at 79%, target 95%

---

## Contributors

- Paul Calnon

---

## Version History

| Version      | Date       | Description                                 |
| ------------ | ---------- | ------------------------------------------- |
| 0.19.0       | 2026-01-08 | P3-1: Create New Snapshot                   |
| 0.20.0       | 2026-01-09 | P3-2, P3-3, P3-4, P3-5: Wave 1 & 2 Complete |
| 0.21.0-alpha | 2026-01-09 | Bug fixes, callback testing, coverage +68%  |

---

## Links

- [Full Changelog](../CHANGELOG.md)
- [Development Roadmap](../DEVELOPMENT_ROADMAP.md)
- [Phase 3 Documentation](../docs/phase3/README.md)
- [Pull Request Details](PR_DESCRIPTION_PHASE3-WAVE-1_2026-01-09.md)
- [Previous Release: v0.18.0](RELEASE_NOTES_v0.18.0-alpha.md)
