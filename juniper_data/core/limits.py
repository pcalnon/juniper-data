"""Input bounds for generators whose work scales with a caller-supplied source.

Mirrors ``juniper_data/core/scaling.py``: that module owns the reserved
``"scaling"`` channel key a generator uses to hand the route metadata that is
not derivable from the final arrays. This module owns the equivalent
``"truncation"`` channel, plus the refusal raised when a source exceeds its cap
and the caller has not opted in to a partial import.

Why a bound exists at all
-------------------------
``APD-DATA-018`` (defect register): generation runs inside the request, so a
source large enough to outlive the client's socket timeout produces a request
that cannot succeed no matter how long the caller waits. The remedy chosen by
the owner (2026-09-04) is to **bound the inputs** rather than move generation
to an async job.

Why truncation must be loud
---------------------------
Truncation is the failure mode this bound was warned about: a silently-partial
dataset is indistinguishable from a complete one to everything downstream, and
juniper-data has no empty-or-partial check in its API or core layers -- the same
gap that let ``arc_agi`` persist a zero-sample dataset (data#318). So the owner's
decision pairs truncation with two non-optional obligations:

1. **The caller must opt in.** A source over the cap is REFUSED until the caller
   says, explicitly, that a partial import is acceptable -- request parameter,
   ``JUNIPER_DATA_CSV_IMPORT_ALLOW_TRUNCATION`` environment variable, or the
   matching ``.env`` entry. An interactive consumer (juniper-canopy) turns that
   into a checkbox; a command-line consumer must pass it.
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

# The csv_import bound lives HERE rather than in
# ``generators/csv_import/defaults.py`` because ``api/settings.py`` needs it too,
# and importing the generator package from settings is circular: importing any
# csv_import submodule executes ``csv_import/__init__.py``, which imports
# ``generator.py``, which imports ``api.settings``. This module imports nothing
# from ``api`` or ``generators``, so both sides can depend on it. ``defaults.py``
# re-exports these names so the generator's own constants stay discoverable
# where every other csv_import default lives.
#
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

# Truncation is opt-in, never a default.
CSV_IMPORT_DEFAULT_ALLOW_TRUNCATION: bool = False


class InputTooLargeError(ValueError):
    """A generator's source exceeds its byte cap and truncation was not allowed.

    Subclasses ``ValueError`` deliberately. The route maps this type to **422**,
    but if a future call path forgets to, the app-level ``ValueError`` handler
    answers **400** rather than letting it surface as a generic 500 -- a wrong
    status code in the 4xx family beats reporting a caller error as a server
    fault.

    Attributes:
        bytes_total: the source's full size in bytes.
        cap_bytes: the cap that was exceeded.
    """

    def __init__(self, *, source: str, bytes_total: int, cap_bytes: int) -> None:
        self.bytes_total = bytes_total
        self.cap_bytes = cap_bytes
        super().__init__(
            f"Source {source!r} is {bytes_total / (1024 * 1024):.1f} MB, over the "
            f"{cap_bytes / (1024 * 1024):.0f} MB cap. Re-submit with allow_truncation=true "
            f"(or set JUNIPER_DATA_CSV_IMPORT_ALLOW_TRUNCATION=true) to import the first "
            f"{cap_bytes / (1024 * 1024):.0f} MB. The resulting dataset will be permanently "
            f"annotated as truncated."
        )


def build_truncation_meta(
    *,
    bytes_read: int,
    bytes_total: int,
    cap_bytes: int,
    records_imported: int,
) -> dict[str, Any]:
    """Build the permanent truncation descriptor.

    Args:
        bytes_read: bytes actually consumed after trimming to a record boundary.
        bytes_total: the source's full size.
        cap_bytes: the cap in force for this request.
        records_imported: rows/objects that survived the trim.

    Returns:
        The descriptor stored on ``DatasetMeta.truncation``.
    """
    return {
        "truncated": True,
        "reason": "source_exceeded_byte_cap",
        "bytes_read": bytes_read,
        "bytes_total": bytes_total,
        "cap_bytes": cap_bytes,
        "records_imported": records_imported,
    }
