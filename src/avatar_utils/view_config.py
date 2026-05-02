"""Canonical orbit cameras around Y-up vertical axis.

Azimuth 0° is **front** (+Z); angles increase toward **right** (+X), matching the
previous ``front`` / ``right`` / ``back`` / ``left`` directions at 0° / 90° / 180° / 270°.
"""

from __future__ import annotations

import math
from typing import List

VIEW_AZIMUTH_STEP_DEG = 15
VIEW_ORDER: List[str] = [str(d) for d in range(0, 360, VIEW_AZIMUTH_STEP_DEG)]
NUM_CANONICAL_VIEWS = len(VIEW_ORDER)

# Four views fed to the model (same layout as legacy front / right / back / left).
MODEL_INPUT_VIEW_ORDER: List[str] = ["0", "90", "180", "270"]


def azimuth_direction(deg: float) -> tuple[float, float, float]:
    """Unit vector from origin toward the camera (orbit in XZ, Y up)."""
    rad = math.radians(deg)
    return (math.sin(rad), 0.0, math.cos(rad))


def viewpoint_dict_from_azimuth() -> dict[str, tuple[float, float, float]]:
    """Maps each view label (degrees as string) to a direction toward the camera."""
    return {name: azimuth_direction(float(name)) for name in VIEW_ORDER}


def reconstruction_view_names_from_config(mode: str) -> List[str]:
    """Resolve inference reconstruction PNG set: full 24-view orbit vs four cardinals."""
    m = str(mode).strip().lower()
    if m in ("full_orbit", "orbit", "all", "24"):
        return list(VIEW_ORDER)
    if m in ("cardinal", "model", "four", "4"):
        return list(MODEL_INPUT_VIEW_ORDER)
    raise ValueError(
        f"inference.reconstruction_render_mode must be 'full_orbit' or 'cardinal', got {mode!r}"
    )
