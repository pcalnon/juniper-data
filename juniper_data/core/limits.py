"""Input bounds for generators whose work scales with a caller-supplied source.

Mirrors ``juniper_data/core/scaling.py``: that module owns the reserved
``"scaling"`` channel key a generator uses to hand the route metadata that is
not derivable from the final arrays. This module owns the equivalent
``"truncation"`` channel, plus the refusal raised when an input exceeds its cap
and the caller has not opted in to a partial import.

Why a bound exists at all
-------------------------
``APD-DATA-018`` (defect register): generation runs inside the request, so an
input large enough to outlive the client's socket timeout produces a request
that cannot succeed no matter how long the caller waits. The remedy chosen by
the owner (2026-09-04) is to **bound the inputs** rather than move generation
to an async job.

Two generators, two units, one contract
---------------------------------------
The unit is not a style choice; it is whichever quantity the cost actually
tracks, and the two generators differ:

* **csv_import** bounds **bytes**. Its input is a file, and bytes are what an
  operator can enforce without parsing it.
* **equities** bounds **symbols**. Its input is an API fan-out, and measurement
  (2026-09-04) showed cost is per *request*, not per byte -- 163x the payload
  cost 1.16x the time. A byte cap there would be actively misleading: one symbol
  over 26 years is 210 KB and ~2 s, while the Russell 3000 over *one day* is
  92 KB and 1.7-3.2 h. The smaller payload is the far more expensive request, so
  a byte threshold would admit exactly the wrong one.

Everything else is shared, because the failure mode is shared.

Why truncation must be loud
---------------------------
Truncation is the failure mode these bounds were warned about: a silently-partial
dataset is indistinguishable from a complete one to everything downstream, and
juniper-data has no partial-data check in its API or core layers -- the same gap
that let ``arc_agi`` persist a zero-sample dataset (data#318). So the owner's
decision pairs truncation with two non-optional obligations:

1. **The caller must opt in.** An input over the cap is REFUSED until the caller
   says, explicitly, that a partial import is acceptable -- request parameter,
   environment variable, or the matching ``.env`` entry. An interactive consumer
   (juniper-canopy) turns that into a checkbox; a command-line consumer must
   pass it.
2. **The annotation is permanent.** When a truncated dataset IS produced, the
   descriptor below is written into ``DatasetMeta`` and persisted with the
   artifact, so a reader who never saw the HTTP response still learns the data
   is partial. It is not a transient warning.
"""

from __future__ import annotations

from typing import Any

# Reserved key a generator MAY place in its ``generate()`` return dict to hand
# the route a truncation descriptor. Popped before checksumming and NPZ persist
# so the stored arrays stay array-only -- exactly as ``SCALING_META_KEY`` is.
TRUNCATION_META_KEY = "truncation"

UNIT_BYTES = "bytes"
UNIT_SYMBOLS = "symbols"

REASON_BYTE_CAP = "source_exceeded_byte_cap"
REASON_SYMBOL_CAP = "universe_exceeded_symbol_cap"

# The per-generator caps live HERE rather than in each generator's
# ``defaults.py`` because ``api/settings.py`` needs them as its deployment
# defaults, and settings cannot import a generator package without a cycle
# (importing any csv_import submodule runs ``__init__.py`` -> ``generator.py``
# -> ``api.settings``). This module imports nothing from ``api`` or
# ``generators``, so every side can depend on it. Each generator's
# ``defaults.py`` re-exports its own, so the constants stay discoverable where
# the rest of that generator's defaults live.

# 128 MiB, chosen from measurement rather than picked round:
# ``util/ad-hoc/2026-09-04_measure_csv_import_throughput.py`` timed the whole
# ``generate()`` path -- parse AND the per-cell float conversion, which is a
# large share of the cost -- at a median **14.4 MB/s** (15.3 MB/s at 3.5 MB
# degrading to 13.6 MB/s at 35 MB). 128 MiB is therefore ~8.9 s of parsing,
# inside the ~30 s client budget with room for split, checksum and NPZ persist.
#
# The binding constraint above this size is memory, not time: the CSV parser
# materialises one Python dict per row before any array exists, so 128 MiB of
# 20-feature rows is ~700k dicts and several GB of peak objects. Raising this
# without also making the loader streaming trades a timeout for an OOM.
CSV_IMPORT_DEFAULT_MAX_BYTES: int = 128 * 1024 * 1024

# 14 symbols, likewise measured:
# ``util/ad-hoc/2026-09-04_measure_equities_payloads.py`` put the per-symbol cost
# at ~2.1 s (one Yahoo chart request plus 1-2 SEC XBRL calls), so 30 s / 2.1 s
# = 14.1. A second measurement two days earlier gave ~4.0 s/symbol, which would
# imply ~7; the owner chose the optimistic end on 2026-09-04. Both estimates
# agree the previous default -- ``None``, meaning all 503 bundled S&P 500
# constituents -- was 36x to 67x over budget.
EQUITIES_DEFAULT_MAX_SYMBOLS: int = 14

# Truncation is opt-in, never a default, for either generator.
CSV_IMPORT_DEFAULT_ALLOW_TRUNCATION: bool = False
EQUITIES_DEFAULT_ALLOW_TRUNCATION: bool = False


def _describe(unit: str, quantity: int) -> str:
    """Render a quantity in its unit, for a message a caller has to act on."""
    if unit == UNIT_BYTES:
        return f"{quantity / (1024 * 1024):.1f} MB"
    return f"{quantity:,} {unit}"


class InputTooLargeError(ValueError):
    """An input exceeds its cap and truncation was not allowed.

    Subclasses ``ValueError`` deliberately. The route maps this type to **422**,
    but if a future call path forgets to, the app-level ``ValueError`` handler
    answers **400** rather than letting it surface as a generic 500 -- a wrong
    status code in the 4xx family beats reporting a caller error as a server
    fault.

    Attributes:
        unit: ``bytes`` or ``symbols`` -- the quantity being bounded.
        cap: the cap that was exceeded, in ``unit``.
        actual: what the input actually measured, in ``unit``.
    """

    def __init__(self, *, source: str, unit: str, cap: int, actual: int, opt_in_env: str) -> None:
        self.unit = unit
        self.cap = cap
        self.actual = actual
        super().__init__(f"{source} is {_describe(unit, actual)}, over the {_describe(unit, cap)} cap. Re-submit with allow_truncation=true (or set {opt_in_env}=true) to import the first {_describe(unit, cap)}. The resulting dataset will be permanently annotated as truncated.")


def build_truncation_meta(
    *,
    reason: str,
    unit: str,
    cap: int,
    requested: int,
    imported: int,
    records_imported: int,
) -> dict[str, Any]:
    """Build the permanent truncation descriptor stored on ``DatasetMeta``.

    One shape for both generators, so a consumer can answer "is this partial,
    and by how much" without knowing which generator produced it. ``unit`` says
    how to read ``cap`` / ``requested`` / ``imported``; ``records_imported`` is
    always dataset rows, whatever the unit, because that is the question a
    trainer actually asks.

    Args:
        reason: ``REASON_BYTE_CAP`` or ``REASON_SYMBOL_CAP``.
        unit: ``UNIT_BYTES`` or ``UNIT_SYMBOLS``.
        cap: the cap in force for this request.
        requested: how much the input offered, in ``unit``.
        imported: how much survived the cap, in ``unit``.
        records_imported: rows in the resulting dataset.

    Returns:
        The descriptor stored on ``DatasetMeta.truncation``.
    """
    return {
        "truncated": True,
        "reason": reason,
        "unit": unit,
        "cap": cap,
        "requested": requested,
        "imported": imported,
        "records_imported": records_imported,
    }
