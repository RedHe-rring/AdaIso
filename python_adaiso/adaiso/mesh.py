from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass
class Mesh:
    vertices: np.ndarray
    faces: np.ndarray

    def copy(self) -> "Mesh":
        return Mesh(self.vertices.copy(), self.faces.copy())

    @property
    def vertex_count(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def face_count(self) -> int:
        return int(self.faces.shape[0])

    def validate(self) -> None:
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError("vertices must have shape (n, 3)")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError("faces must have shape (m, 3)")
        if self.faces.size:
            if self.faces.min() < 0 or self.faces.max() >= self.vertex_count:
                raise ValueError("faces contain out-of-range vertex indices")


def load_mesh(path: str | Path) -> Mesh:
    """Load a triangular mesh.

    Uses trimesh when available so OBJ/PLY and scenes are handled robustly.
    Falls back to a small OBJ parser if trimesh is unavailable.
    """

    path = Path(path)
    try:
        import trimesh  # type: ignore

        obj = trimesh.load_mesh(str(path), process=False)
        if isinstance(obj, trimesh.Scene):
            geometries = [g for g in obj.geometry.values() if len(g.vertices) and len(g.faces)]
            if not geometries:
                raise ValueError(f"scene has no triangular geometry: {path}")
            obj = trimesh.util.concatenate(geometries)
        vertices = np.asarray(obj.vertices, dtype=np.float64)
        faces = np.asarray(obj.faces, dtype=np.int64)
        if faces.shape[1] != 3:
            raise ValueError("input mesh must be triangulated")
        mesh = Mesh(vertices, faces)
        mesh.validate()
        return mesh
    except ImportError:
        if path.suffix.lower() != ".obj":
            raise RuntimeError("trimesh is required for non-OBJ input")
        return _load_obj(path)


def _load_obj(path: Path) -> Mesh:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                idx: list[int] = []
                for token in line.split()[1:]:
                    idx.append(int(token.split("/")[0]) - 1)
                if len(idx) == 3:
                    faces.append(idx)
                elif len(idx) > 3:
                    first = idx[0]
                    for i in range(1, len(idx) - 1):
                        faces.append([first, idx[i], idx[i + 1]])
    mesh = Mesh(np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64))
    mesh.validate()
    return mesh


def write_obj(path: str | Path, mesh: Mesh) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for v in mesh.vertices:
            f.write(f"v {v[0]:.10g} {v[1]:.10g} {v[2]:.10g}\n")
        for tri in mesh.faces:
            a, b, c = (int(x) + 1 for x in tri)
            f.write(f"f {a} {b} {c}\n")


def compact_vertices(vertices: np.ndarray, faces: np.ndarray) -> Mesh:
    """Drop unused vertices and remap face indices."""

    if len(faces) == 0:
        return Mesh(np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int64))
    used = np.unique(faces.reshape(-1))
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    return Mesh(vertices[used].copy(), remap[faces].astype(np.int64, copy=False))


def remove_degenerate_faces(mesh: Mesh, area_eps: float = 1e-14) -> Mesh:
    """Remove duplicate-index, tiny-area, and repeated unoriented faces."""

    vertices = mesh.vertices
    faces = mesh.faces
    if len(faces) == 0:
        return mesh.copy()

    a = vertices[faces[:, 0]]
    b = vertices[faces[:, 1]]
    c = vertices[faces[:, 2]]
    areas2 = np.linalg.norm(np.cross(b - a, c - a), axis=1)
    distinct = (
        (faces[:, 0] != faces[:, 1])
        & (faces[:, 1] != faces[:, 2])
        & (faces[:, 2] != faces[:, 0])
    )
    keep = distinct & (areas2 > area_eps)
    faces = faces[keep]

    seen: set[tuple[int, int, int]] = set()
    unique_faces: list[np.ndarray] = []
    for tri in faces:
        key = tuple(sorted(int(x) for x in tri))
        if key in seen:
            continue
        seen.add(key)
        unique_faces.append(tri)

    if not unique_faces:
        return Mesh(vertices.copy(), np.zeros((0, 3), dtype=np.int64))
    return compact_vertices(vertices, np.asarray(unique_faces, dtype=np.int64))


def iter_unique_edges(faces: np.ndarray) -> Iterable[tuple[int, int]]:
    seen: set[tuple[int, int]] = set()
    for a, b, c in faces:
        for i, j in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            if i > j:
                i, j = j, i
            if (i, j) not in seen:
                seen.add((i, j))
                yield i, j
