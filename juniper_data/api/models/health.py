"""Health check response models — re-exported from ``juniper-observability``.

METRICS-MON R2.1.2 / seed-06: the previous in-repo definitions of
:class:`DependencyStatus` and :class:`ReadinessResponse` have been
promoted into the shared :mod:`juniper_observability` package so all
three Juniper servers consume one source of truth. This module is
preserved as a thin re-export shim for backwards compatibility — any
existing code that imports ``from juniper_data.api.models.health
import DependencyStatus, ReadinessResponse`` continues to work
unchanged.

New code should prefer ``from juniper_observability import …`` to
make the dependency on the shared lib explicit.

See: notes/code-review/METRICS_MONITORING_R2.1_SHARED_OBSERVABILITY_DESIGN_2026-04-28.md
in juniper-ml.
"""

from juniper_observability import DependencyStatus, ReadinessResponse

__all__ = ["DependencyStatus", "ReadinessResponse"]
