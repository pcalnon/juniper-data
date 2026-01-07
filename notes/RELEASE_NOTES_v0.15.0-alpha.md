# Juniper Canopy v0.15.0 Release Notes

**Release Date:** 2026-01-07  
**Codename:** Phase 0 - Core UX Stabilization

## Overview

This release completes Phase 0 of the Juniper Canopy development roadmap, delivering a fully stable and responsive dashboard experience. All 11 Core UX issues have been resolved, making the training visualization interface production-ready.

## What's New

### Reliable Training Controls

- **Start/Stop/Pause/Resume buttons** now provide proper visual feedback and return to normal state after activation
- **Apply button** correctly persists all meta-parameters (learning rate, max hidden units, max epochs)
- Parameter changes are detected accurately with proper float tolerance

### Accurate Status Display

- **Unified status bar** shows real-time training status and phase with color-coded indicators:
  - Running (green), Paused (orange), Stopped (gray)
  - Output Training (blue), Candidate Pool (cyan), Idle (gray)
  - Completed (cyan), Failed (red)
- Training completion and failure states are now properly tracked and displayed

### Responsive Graph Interactions

- **Zoom and pan persist** across data updates - no more jarring resets every second
- **Pan and Lasso tools** work correctly with proper default selection
- **Tool selection persists** - switching between pan/lasso/zoom stays active

### Polished Visuals

- **Dark mode support** with theme-aware backgrounds and text colors
- **Legend positioning** at bottom-left with semi-transparent backgrounds for better visibility

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

## Links

- [Full Changelog](../CHANGELOG.md)
- [Development Roadmap](../DEVELOPMENT_ROADMAP.md)
- [Pull Request Details](PR_PHASE0_UX_STABILIZATION_2026-01-07.md)
