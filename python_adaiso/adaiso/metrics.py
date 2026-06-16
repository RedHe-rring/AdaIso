from __future__ import annotations

import numpy as np

from .geometry import triangle_quality
from .mesh import Mesh


def quality_summary(mesh: Mesh) -> dict[str, float]:
    q = triangle_quality(mesh)
    if len(q) == 0:
        return {"qavg": float("nan"), "qmin": float("nan")}
    return {"qavg": float(np.mean(q)), "qmin": float(np.min(q))}


def mesh_summary(mesh: Mesh) -> dict[str, float | int]:
    out: dict[str, float | int] = {
        "vertices": mesh.vertex_count,
        "faces": mesh.face_count,
    }
    out.update(quality_summary(mesh))
    return out
