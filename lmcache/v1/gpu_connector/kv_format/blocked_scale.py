# SPDX-License-Identifier: Apache-2.0
"""Runtime geometry for blocked-scale KV pages."""

# Standard
from typing import Final

# DSV4 indexer and DSV4.1 main-KV page geometries. Both use the same
# pointer topology; these widths describe the physical row contract.
_BLOCKED_SCALE_GEOMETRIES: Final[dict[int, tuple[int, int]]] = {
    132: (128, 4),
    584: (576, 8),
}


def resolve_blocked_scale_geometry(
    row_bytes: int,
) -> tuple[int, int]:
    """Resolve and validate the value/scale widths of a blocked-scale row.

    Args:
        row_bytes: Logical token-row width in bytes.

    Returns:
        A ``(value_bytes, scale_bytes)`` pair.

    Raises:
        ValueError: If the row width is not a supported variant.
    """
    try:
        return _BLOCKED_SCALE_GEOMETRIES[row_bytes]
    except KeyError as exc:
        raise ValueError(
            "unsupported blocked-scale row width: "
            f"{row_bytes} bytes; expected one of "
            f"{sorted(_BLOCKED_SCALE_GEOMETRIES)}"
        ) from exc
