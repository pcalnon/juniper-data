# Juniper Canopy v0.15.0-alpha Release Notes

**Release Date:** 2026-01-07  
**Version:** 0.15.0-alpha  
**Codename:** Phase 0 - Core UX Stabilization

## Overview

This release completes Phase 0 of the Juniper Canopy development roadmap, delivering a fully stable and responsive dashboard experience. All 11 Core UX issues have been resolved, making the training visualization interface production-ready.

## Phase 0 Fixes Summary

| ID    | Issue                           | Status   |
| ----- | ------------------------------- | -------- |
| P0-1  | Training Controls Button State  | ✅ Fixed |
| P0-2  | Meta-Parameters Apply Button    | ✅ Fixed |
| P0-3  | Top Status Bar Updates          | ✅ Fixed |
| P0-4  | Graph Range Persistence         | ✅ Fixed |
| P0-5  | Pan/Lasso Tool Fix              | ✅ Fixed |
| P0-6  | Interaction Persistence         | ✅ Fixed |
| P0-7  | Dark Mode Info Bar              | ✅ Fixed |
| P0-8  | Status Bar Completion States    | ✅ Fixed |
| P0-9  | Legend Display and Positioning  | ✅ Fixed |
| P0-10 | Configuration Test Architecture | ✅ Fixed |
| P0-12 | Learning Rate Float Tolerance   | ✅ Fixed |

## What's New

### Reliable Training Controls

- **P0-1**: Start/Stop/Pause/Resume buttons return to normal state after click (2s timeout)
- **P0-2**: Apply button correctly persists all meta-parameters (`max_hidden_units`, `max_epochs`, `learning_rate`)
- **P0-12**: Learning rate change detection uses float tolerance instead of equality

### Accurate Status Display

- **P0-3**: Unified status bar displays FSM-based status/phase with state-specific colors:
  - Status: Running (green), Paused (orange), Stopped (gray)
  - Phase: Output Training (blue), Candidate Pool (cyan), Idle (gray)
- **P0-8**: Added `COMPLETED` and `FAILED` terminal states to TrainingStatus enum
  - Dashboard displays Completed (cyan) and Failed (red) states

### Responsive Graph Interactions

- **P0-4**: View-state `dcc.Store` preserves user zoom/pan ranges across interval updates
- **P0-5**: Pan/Lasso tools work correctly with `dragmode="pan"` default
- **P0-6**: Zoom, pan, and dragmode persist across network topology updates

### Polished Visuals

- **P0-7**: Theme-aware stats bar with proper dark mode background/text colors
- **P0-9**: Legend at bottom-left with semi-transparent theme-aware backgrounds

### Test Infrastructure

- **P0-10**: Configuration tests use bounds checks instead of equality for YAML values

## Improvements

### Test Coverage

| Component             | Before | After    |
| --------------------- | ------ | -------- |
| metrics_panel.py      | 67%    | 98%      |
| dashboard_manager.py  | 68%    | 93%      |
| network_visualizer.py | 71%    | 99%      |
| decision_boundary.py  | 84%    | 100%     |
| dataset_plotter.py    | 87%    | 99%      |
| main.py               | 79%    | 89%      |
| **Overall**           | ~75%   | **93%+** |

### Test Suite

- **2129 tests** passing (up from ~2097)
- **37 skipped** (requires external dependencies)
- **0 failures**

## Upgrade Notes

This is a backward-compatible release. No migration steps required.

```bash
# Update and verify
git pull origin main
./demo

# Run test suite
cd src && pytest tests/ -v
```

## Known Issues

None. All Phase 0 issues have been resolved.

## What's Next

Phase 1 development will focus on:

- Enhanced network topology visualization
- Training metrics export functionality
- Performance optimizations for large networks

## Contributors

- Paul Calnon

## Version History

| Version      | Date       | Description                                              |
| ------------ | ---------- | -------------------------------------------------------- |
| 0.14.2-alpha | 2026-01-05 | P0-3: Top Status Bar Updates                             |
| 0.14.3-alpha | 2026-01-06 | P0-2, P0-4: Apply Button & Graph Range Persistence       |
| 0.14.4-alpha | 2026-01-06 | P0-10: Configuration Test Architecture + Coverage        |
| 0.15.0-alpha | 2026-01-07 | P0-1, P0-5, P0-6, P0-7, P0-8, P0-9, P0-12: Final Phase 0 |

## Links

- [Full Changelog](../CHANGELOG.md)
- [Development Roadmap](../DEVELOPMENT_ROADMAP.md)
- [Pull Request Details](PR_PHASE0_UX_STABILIZATION_2026-01-07.md)
- [Previous Release: v0.14.1-alpha](RELEASE_NOTES_v0.14.1-alpha.md)
