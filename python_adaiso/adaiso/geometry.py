from __future__ import annotations

from collections import defaultdict

import numpy as np

from .mesh import Mesh, iter_unique_edges


def normalize(v: np.ndarray, eps: float = 1e-14) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, norm, out=np.zeros_like(v, dtype=np.float64), where=norm > eps)


def face_normals_and_areas(mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    v = mesh.vertices
    f = mesh.faces
    if len(f) == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros(0, dtype=np.float64)
    cross = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    norm = np.linalg.norm(cross, axis=1)
    normals = np.divide(cross, norm[:, None], out=np.zeros_like(cross), where=norm[:, None] > 1e-14)
    return normals, 0.5 * norm


def vertex_normals(mesh: Mesh) -> np.ndarray:
    normals, areas = face_normals_and_areas(mesh)
    out = np.zeros_like(mesh.vertices, dtype=np.float64)
    for fi, tri in enumerate(mesh.faces):
        weighted = normals[fi] * areas[fi]
        out[tri[0]] += weighted
        out[tri[1]] += weighted
        out[tri[2]] += weighted
    return normalize(out)


def edge_faces(faces: np.ndarray) -> dict[tuple[int, int], list[int]]:
    e2f: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi, (a, b, c) in enumerate(faces):
        for i, j in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            if i > j:
                i, j = j, i
            e2f[(i, j)].append(fi)
    return dict(e2f)


def vertex_neighbors(faces: np.ndarray, n_vertices: int) -> list[set[int]]:
    neigh = [set() for _ in range(n_vertices)]
    for a, b, c in faces:
        a = int(a)
        b = int(b)
        c = int(c)
        neigh[a].update((b, c))
        neigh[b].update((a, c))
        neigh[c].update((a, b))
    return neigh


def vertex_faces(faces: np.ndarray, n_vertices: int) -> list[list[int]]:
    vf = [[] for _ in range(n_vertices)]
    for fi, tri in enumerate(faces):
        for v in tri:
            vf[int(v)].append(fi)
    return vf


def boundary_vertices(faces: np.ndarray, n_vertices: int) -> set[int]:
    boundary: set[int] = set()
    for (a, b), incident in edge_faces(faces).items():
        if len(incident) == 1:
            boundary.add(a)
            boundary.add(b)
    return boundary


def mean_edge_length(mesh: Mesh) -> float:
    lengths = []
    v = mesh.vertices
    for a, b in iter_unique_edges(mesh.faces):
        lengths.append(float(np.linalg.norm(v[a] - v[b])))
    if not lengths:
        return 0.0
    return float(np.mean(lengths))


def cotangent_edge_weights(mesh: Mesh, min_weight: float = 1e-4) -> dict[tuple[int, int], float]:
    """Return cotangent weights for undirected edges.

    For edge (i, j), the weight is 0.5 * sum(cot opposite angles). Non-positive
    or invalid weights are clamped to min_weight, matching the paper's intent.
    """

    weights: dict[tuple[int, int], float] = defaultdict(float)
    v = mesh.vertices

    def cot_at(a: int, b: int, c: int) -> float:
        u = v[b] - v[a]
        w = v[c] - v[a]
        denom = np.linalg.norm(np.cross(u, w))
        if denom <= 1e-14:
            return 0.0
        return float(np.dot(u, w) / denom)

    for a, b, c in mesh.faces:
        a = int(a)
        b = int(b)
        c = int(c)
        for i, j, opposite in ((b, c, a), (c, a, b), (a, b, c)):
            if i > j:
                i, j = j, i
            weights[(i, j)] += cot_at(opposite, i, j)

    final = {}
    for edge, val in weights.items():
        w = 0.5 * val
        final[edge] = w if np.isfinite(w) and w > 0.0 else min_weight
    return final


def triangle_quality(mesh: Mesh) -> np.ndarray:
    v = mesh.vertices
    f = mesh.faces
    if len(f) == 0:
        return np.zeros(0, dtype=np.float64)
    p0 = v[f[:, 0]]
    p1 = v[f[:, 1]]
    p2 = v[f[:, 2]]
    l0 = np.linalg.norm(p1 - p0, axis=1)
    l1 = np.linalg.norm(p2 - p1, axis=1)
    l2 = np.linalg.norm(p0 - p2, axis=1)
    half_p = 0.5 * (l0 + l1 + l2)
    area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
    longest = np.maximum(l0, np.maximum(l1, l2))
    denom = half_p * longest
    return np.divide(6.0 * area, np.sqrt(3.0) * denom, out=np.zeros_like(area), where=denom > 1e-14)


def max_inner_angle(mesh: Mesh) -> np.ndarray:
    v = mesh.vertices
    vals = []
    for tri in mesh.faces:
        pts = v[tri]
        angles = []
        for i in range(3):
            a = pts[(i + 1) % 3] - pts[i]
            b = pts[(i + 2) % 3] - pts[i]
            na = np.linalg.norm(a)
            nb = np.linalg.norm(b)
            if na <= 1e-14 or nb <= 1e-14:
                angles.append(np.pi)
            else:
                angles.append(float(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))))
        vals.append(max(angles))
    return np.asarray(vals, dtype=np.float64)
