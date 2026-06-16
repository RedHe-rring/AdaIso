from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.spatial import cKDTree

from .geometry import (
    boundary_vertices,
    cotangent_edge_weights,
    edge_faces,
    face_normals_and_areas,
    max_inner_angle,
    mean_edge_length,
    normalize,
    vertex_faces,
    vertex_neighbors,
    vertex_normals,
)
from .mesh import Mesh, compact_vertices, iter_unique_edges, remove_degenerate_faces


Variant = Literal["paper", "code"]


@dataclass
class AdaIsoConfig:
    mesh_scale: float = 1.0
    variant: Variant = "paper"
    preprocess: bool = False
    global_iterations: int = 5
    csf_max_iterations: int = 20
    csf_tolerance: float = 0.005
    csf_mu: float = 0.5
    cot_min_weight: float = 1e-4
    histogram_bins: int = 40
    multipliers: tuple[float, float, float, float, float] = (1.8, 1.4, 1.0, 0.8, 0.6)
    subremesh_multiplier: float = 1.5
    max_subremesh_rounds: int = 8
    max_split_passes: int = 10
    max_collapse_passes: int = 10
    max_flip_passes: int = 20
    tangent_lambda: float = 0.6
    project_to_surface: bool = True
    mls_k: int = 8
    mls_iterations: int = 10
    mls_tolerance: float = 1e-4
    avoid_obtuse_flip: bool = True
    min_quality_after_collapse: float = 0.05
    verbose: bool = True

    @property
    def split_factor(self) -> float:
        return 5.0 / 3.0 if self.variant == "paper" else 1.33

    @property
    def collapse_factor(self) -> float:
        return 4.0 / 5.0

    @property
    def collapse_reference(self) -> Literal["max", "min"]:
        return "max" if self.variant == "paper" else "min"


class AdaIsoRemesher:
    def __init__(self, config: AdaIsoConfig | None = None) -> None:
        self.config = config or AdaIsoConfig()
        self._target_tree: cKDTree | None = None
        self._target_positions: np.ndarray | None = None
        self._target_lengths: np.ndarray | None = None
        self.stats: dict[str, float | int] = {}

    def remesh(self, mesh: Mesh) -> Mesh:
        cfg = self.config
        mesh = remove_degenerate_faces(mesh)
        original_for_projection = mesh.copy()

        if cfg.verbose:
            print(f"input: {mesh.vertex_count} vertices, {mesh.face_count} faces")

        if cfg.preprocess:
            mesh = self.sub_remesh(mesh)
            mesh = remove_degenerate_faces(mesh)
            if cfg.verbose:
                print(f"after sub-remesh: {mesh.vertex_count} vertices, {mesh.face_count} faces")

        self._build_target_lengths(mesh)

        for it in range(cfg.global_iterations):
            if cfg.verbose:
                print(f"iteration {it + 1}/{cfg.global_iterations}")
            split_count = self._split_long_edges(mesh)
            collapse_count = self._collapse_short_edges(mesh)
            mesh = remove_degenerate_faces(mesh)
            flip_count = self._flip_edges(mesh)
            self._tangent_smoothing(mesh)
            mesh = remove_degenerate_faces(mesh)
            if cfg.verbose:
                print(f"  split={split_count}, collapse={collapse_count}, flip={flip_count}, "
                      f"mesh={mesh.vertex_count}v/{mesh.face_count}f")

        if cfg.project_to_surface:
            mesh.vertices = self._mls_project(mesh.vertices, original_for_projection)
            mesh = remove_degenerate_faces(mesh)

        self.stats = {
            "vertices": mesh.vertex_count,
            "faces": mesh.face_count,
        }
        return mesh

    def sub_remesh(self, mesh: Mesh) -> Mesh:
        """Practical sub-remeshing: recursively split extremely long edges."""

        cfg = self.config
        mesh = mesh.copy()
        base = mean_edge_length(mesh)
        if base <= 0.0:
            return mesh
        threshold = 2.0 * cfg.subremesh_multiplier * base
        for _ in range(cfg.max_subremesh_rounds):
            e2f = edge_faces(mesh.faces)
            candidates = [
                (float(np.linalg.norm(mesh.vertices[a] - mesh.vertices[b])), a, b)
                for a, b in e2f
                if np.linalg.norm(mesh.vertices[a] - mesh.vertices[b]) > threshold
            ]
            if not candidates:
                break
            candidates.sort(reverse=True)
            _, a, b = candidates[0]
            self._split_one_edge(mesh, a, b)
            mesh = remove_degenerate_faces(mesh)
        return mesh

    def _build_target_lengths(self, mesh: Mesh) -> None:
        cfg = self.config
        base_len = mean_edge_length(mesh)
        curvature = self._compute_csf(mesh)
        multipliers = self._histogram_multipliers(curvature)
        target_lengths = base_len * cfg.mesh_scale * multipliers
        self._target_positions = mesh.vertices.copy()
        self._target_lengths = target_lengths
        self._target_tree = cKDTree(self._target_positions)
        if cfg.verbose:
            print(
                f"target length: base={base_len:.6g}, "
                f"min={target_lengths.min():.6g}, max={target_lengths.max():.6g}"
            )

    def _compute_csf(self, mesh: Mesh) -> np.ndarray:
        cfg = self.config
        normals = vertex_normals(mesh)
        neigh = vertex_neighbors(mesh.faces, mesh.vertex_count)
        curvature = np.zeros(mesh.vertex_count, dtype=np.float64)

        for i, nb in enumerate(neigh):
            if not nb:
                continue
            dots = normals[list(nb)] @ normals[i]
            dots = np.clip(dots, -1.0, 1.0)
            angles = np.arccos(dots)
            if cfg.variant == "paper":
                curvature[i] = float(np.max(angles))
            else:
                curvature[i] = float(np.mean(angles))

        weights = cotangent_edge_weights(mesh, cfg.cot_min_weight)
        vmax = int(np.argmax(curvature)) if len(curvature) else -1
        forbidden = set(neigh[vmax]) | {vmax} if vmax >= 0 else set()
        candidates = [i for i in range(mesh.vertex_count) if i not in forbidden]
        vmin = min(candidates, key=lambda idx: curvature[idx]) if candidates else int(np.argmin(curvature))
        fixed = {vmax, vmin}

        for it in range(cfg.csf_max_iterations):
            old = curvature.copy()
            deltas = []
            for i, nb in enumerate(neigh):
                if i in fixed or not nb:
                    continue
                total = 0.0
                accum = 0.0
                for j in nb:
                    edge = (i, j) if i < j else (j, i)
                    w = weights.get(edge, cfg.cot_min_weight)
                    total += w
                    accum += w * old[j]
                if total <= 0.0:
                    continue
                weighted_avg = accum / total
                delta = weighted_avg - old[i]
                curvature[i] = old[i] + (1.0 - cfg.csf_mu) * delta
                deltas.append(abs(delta))
            mean_delta = float(np.mean(deltas)) if deltas else 0.0
            if mean_delta < cfg.csf_tolerance:
                self.stats["csf_iterations"] = it + 1
                break
        return curvature

    def _histogram_multipliers(self, curvature: np.ndarray) -> np.ndarray:
        cfg = self.config
        if len(curvature) == 0:
            return np.zeros(0, dtype=np.float64)
        if np.allclose(curvature.max(), curvature.min()):
            return np.ones_like(curvature)

        counts, edges = np.histogram(curvature, bins=cfg.histogram_bins)
        max_count = counts.max()
        max_bins = np.flatnonzero(counts == max_count)
        base_bin = int(max_bins[len(max_bins) // 2])
        base = 0.5 * (edges[base_bin] + edges[base_bin + 1])
        sorted_vals = np.sort(curvature)

        if cfg.variant == "code":
            idx = int(np.searchsorted(sorted_vals, base, side="left"))
            idx = max(1, min(idx, len(sorted_vals) - 1))
            unit_right = max(1, (len(sorted_vals) - idx) // 5)
            cuts = [
                sorted_vals[max(0, min(len(sorted_vals) - 1, (idx // 5) * 2))],
                sorted_vals[max(0, min(len(sorted_vals) - 1, (idx // 5) * 4))],
                sorted_vals[max(0, min(len(sorted_vals) - 1, idx + unit_right))],
                sorted_vals[max(0, min(len(sorted_vals) - 1, idx + unit_right * 3))],
            ]
        else:
            left = np.sort(curvature[curvature <= base])
            right = np.sort(curvature[curvature > base])
            if len(left) == 0:
                left = sorted_vals[: max(1, len(sorted_vals) // 2)]
            if len(right) == 0:
                right = sorted_vals[max(0, len(sorted_vals) // 2) :]
            cuts = [
                float(np.quantile(left, 2.0 / 5.0)),
                float(np.quantile(left, 4.0 / 5.0)),
                float(np.quantile(right, 1.0 / 5.0)),
                float(np.quantile(right, 3.0 / 5.0)),
            ]

        m = np.empty_like(curvature, dtype=np.float64)
        m[curvature < cuts[0]] = cfg.multipliers[0]
        m[(curvature >= cuts[0]) & (curvature < cuts[1])] = cfg.multipliers[1]
        m[(curvature >= cuts[1]) & (curvature < cuts[2])] = cfg.multipliers[2]
        m[(curvature >= cuts[2]) & (curvature < cuts[3])] = cfg.multipliers[3]
        m[curvature >= cuts[3]] = cfg.multipliers[4]
        return m

    def _target_at(self, points: np.ndarray) -> np.ndarray:
        if self._target_tree is None or self._target_lengths is None:
            raise RuntimeError("target lengths have not been initialized")
        _, idx = self._target_tree.query(points, k=1)
        return self._target_lengths[np.asarray(idx, dtype=np.int64)]

    def _edge_target(self, mesh: Mesh, a: int, b: int, reference: Literal["min", "max"] = "min") -> float:
        vals = self._target_at(mesh.vertices[[a, b]])
        return float(np.max(vals) if reference == "max" else np.min(vals))

    def _split_long_edges(self, mesh: Mesh) -> int:
        cfg = self.config
        total = 0
        for _ in range(cfg.max_split_passes):
            candidates: list[tuple[float, int, int]] = []
            for a, b in edge_faces(mesh.faces):
                length = float(np.linalg.norm(mesh.vertices[a] - mesh.vertices[b]))
                target = self._edge_target(mesh, a, b, "min")
                if length > cfg.split_factor * target:
                    candidates.append((length, a, b))
            if not candidates:
                break
            candidates.sort(reverse=True)
            processed = 0
            for _, a, b in candidates:
                if not self._edge_exists(mesh, a, b):
                    continue
                length = float(np.linalg.norm(mesh.vertices[a] - mesh.vertices[b]))
                target = self._edge_target(mesh, a, b, "min")
                if length <= cfg.split_factor * target:
                    continue
                self._split_one_edge(mesh, a, b)
                processed += 1
                total += 1
            if processed == 0:
                break
        return total

    def _split_one_edge(self, mesh: Mesh, a: int, b: int) -> None:
        midpoint = 0.5 * (mesh.vertices[a] + mesh.vertices[b])
        m = mesh.vertex_count
        mesh.vertices = np.vstack([mesh.vertices, midpoint[None, :]])

        new_faces: list[list[int]] = []
        for tri in mesh.faces:
            tri_list = [int(x) for x in tri]
            if a in tri_list and b in tri_list:
                new_faces.extend(_split_face_on_edge(tri_list, a, b, m))
            else:
                new_faces.append(tri_list)
        mesh.faces = np.asarray(new_faces, dtype=np.int64)

    def _collapse_short_edges(self, mesh: Mesh) -> int:
        cfg = self.config
        total = 0
        for _ in range(cfg.max_collapse_passes):
            bverts = boundary_vertices(mesh.faces, mesh.vertex_count)
            candidates: list[tuple[float, int, int]] = []
            for a, b in edge_faces(mesh.faces):
                if a in bverts or b in bverts:
                    continue
                length = float(np.linalg.norm(mesh.vertices[a] - mesh.vertices[b]))
                target = self._edge_target(mesh, a, b, cfg.collapse_reference)
                if length < cfg.collapse_factor * target:
                    candidates.append((length, a, b))
            if not candidates:
                break
            candidates.sort()
            collapsed = False
            e2f = edge_faces(mesh.faces)
            for _, a, b in candidates:
                if not self._edge_exists(mesh, a, b):
                    continue
                if len(e2f.get((min(a, b), max(a, b)), [])) > 2:
                    continue
                if not self._collapse_valid(mesh, a, b):
                    continue
                self._collapse_one_edge(mesh, a, b)
                total += 1
                collapsed = True
                break
            if not collapsed:
                break
        return total

    def _collapse_valid(self, mesh: Mesh, a: int, b: int) -> bool:
        midpoint = 0.5 * (mesh.vertices[a] + mesh.vertices[b])
        neigh = vertex_neighbors(mesh.faces, mesh.vertex_count)
        e2f = edge_faces(mesh.faces)
        incident = e2f.get((min(a, b), max(a, b)), [])
        opposite = set()
        for fi in incident:
            opposite.update(int(x) for x in mesh.faces[fi] if int(x) not in (a, b))
        link_intersection = neigh[a].intersection(neigh[b]) - {a, b}
        if link_intersection != opposite:
            return False
        related = (neigh[a] | neigh[b]) - {a, b}
        if len(related) < 3:
            return False
        for n in related:
            length = float(np.linalg.norm(midpoint - mesh.vertices[n]))
            target = float(np.min(self._target_at(np.vstack([midpoint, mesh.vertices[n]]))))
            if length > self.config.split_factor * target:
                return False
        if self.config.min_quality_after_collapse > 0:
            vertices = mesh.vertices.copy()
            faces = mesh.faces.copy()
            keep = min(a, b)
            remove = max(a, b)
            vertices[keep] = midpoint
            faces[faces == remove] = keep
            candidate = remove_degenerate_faces(Mesh(vertices, faces))
            if candidate.face_count == 0:
                return False
            from .geometry import triangle_quality

            q = triangle_quality(candidate)
            if len(q) == 0 or float(np.min(q)) < self.config.min_quality_after_collapse:
                return False
        return True

    def _collapse_one_edge(self, mesh: Mesh, a: int, b: int) -> None:
        keep = min(a, b)
        remove = max(a, b)
        mesh.vertices[keep] = 0.5 * (mesh.vertices[a] + mesh.vertices[b])
        faces = mesh.faces.copy()
        faces[faces == remove] = keep
        tmp = Mesh(mesh.vertices, faces)
        tmp = remove_degenerate_faces(tmp)
        mesh.vertices = tmp.vertices
        mesh.faces = tmp.faces

    def _flip_edges(self, mesh: Mesh) -> int:
        cfg = self.config
        total = 0
        for _ in range(cfg.max_flip_passes):
            e2f = edge_faces(mesh.faces)
            bverts = boundary_vertices(mesh.faces, mesh.vertex_count)
            neigh = vertex_neighbors(mesh.faces, mesh.vertex_count)
            valence = np.asarray([len(n) for n in neigh], dtype=np.int64)
            flipped = 0
            for (a, b), incident in list(e2f.items()):
                if len(incident) != 2:
                    continue
                f1, f2 = incident
                tri1 = [int(x) for x in mesh.faces[f1]]
                tri2 = [int(x) for x in mesh.faces[f2]]
                cands1 = [x for x in tri1 if x not in (a, b)]
                cands2 = [x for x in tri2 if x not in (a, b)]
                if len(cands1) != 1 or len(cands2) != 1:
                    continue
                c = cands1[0]
                d = cands2[0]
                if c == d or self._edge_exists(mesh, c, d):
                    continue
                length = float(np.linalg.norm(mesh.vertices[c] - mesh.vertices[d]))
                target = self._edge_target(mesh, c, d, "min")
                if length < cfg.collapse_factor * target or length > cfg.split_factor * target:
                    continue
                if not self._flip_improves_valence(valence, bverts, a, b, c, d):
                    continue
                ref_n = _safe_face_normal(mesh.vertices[tri1]) + _safe_face_normal(mesh.vertices[tri2])
                new1 = _orient_triangle([c, d, b], mesh.vertices, ref_n)
                new2 = _orient_triangle([d, c, a], mesh.vertices, ref_n)
                if cfg.avoid_obtuse_flip:
                    tmp = Mesh(mesh.vertices, np.asarray([new1, new2], dtype=np.int64))
                    if np.any(max_inner_angle(tmp) > 0.5 * np.pi):
                        continue
                mesh.faces[f1] = np.asarray(new1, dtype=np.int64)
                mesh.faces[f2] = np.asarray(new2, dtype=np.int64)
                flipped += 1
                total += 1
            if flipped == 0:
                break
        return total

    @staticmethod
    def _flip_improves_valence(
        valence: np.ndarray,
        boundary: set[int],
        a: int,
        b: int,
        c: int,
        d: int,
    ) -> bool:
        def target(v: int) -> int:
            return 4 if v in boundary else 6

        before = (
            abs(int(valence[a]) - target(a))
            + abs(int(valence[b]) - target(b))
            + abs(int(valence[c]) - target(c))
            + abs(int(valence[d]) - target(d))
        )
        after = (
            abs(int(valence[a]) - 1 - target(a))
            + abs(int(valence[b]) - 1 - target(b))
            + abs(int(valence[c]) + 1 - target(c))
            + abs(int(valence[d]) + 1 - target(d))
        )
        return after < before

    def _tangent_smoothing(self, mesh: Mesh) -> None:
        cfg = self.config
        bverts = boundary_vertices(mesh.faces, mesh.vertex_count)
        vf = vertex_faces(mesh.faces, mesh.vertex_count)
        normals = vertex_normals(mesh)
        _, areas = face_normals_and_areas(mesh)
        new_vertices = mesh.vertices.copy()

        for i in range(mesh.vertex_count):
            if i in bverts or not vf[i]:
                continue
            weights: dict[int, float] = {}
            for fi in vf[i]:
                tri = [int(x) for x in mesh.faces[fi]]
                for n in tri:
                    if n != i:
                        weights[n] = weights.get(n, 0.0) + float(areas[fi])
            if not weights:
                continue
            neigh_idx = np.asarray(list(weights.keys()), dtype=np.int64)
            area_w = np.asarray([weights[int(j)] for j in neigh_idx], dtype=np.float64)
            length_w = self._target_at(mesh.vertices[neigh_idx])
            w = area_w * length_w
            if np.sum(w) <= 1e-14:
                continue
            centroid = np.sum(mesh.vertices[neigh_idx] * w[:, None], axis=0) / np.sum(w)
            disp = centroid - mesh.vertices[i]
            n = normals[i]
            tangent = disp - np.dot(disp, n) * n
            candidate = mesh.vertices[i] + cfg.tangent_lambda * tangent
            if np.all(np.isfinite(candidate)):
                new_vertices[i] = candidate
        mesh.vertices = new_vertices

    def _mls_project(self, points: np.ndarray, original: Mesh) -> np.ndarray:
        cfg = self.config
        if original.vertex_count == 0:
            return points
        normals = vertex_normals(original)
        tree = cKDTree(original.vertices)
        k = min(cfg.mls_k, original.vertex_count)
        out = points.copy()
        for pi, point in enumerate(points):
            dists, idx = tree.query(point, k=k)
            dists = np.atleast_1d(dists).astype(np.float64)
            idx = np.atleast_1d(idx).astype(np.int64)
            if dists[0] <= 1e-14:
                out[pi] = point
                continue
            px = point.copy()
            best_err = np.inf
            for _ in range(cfg.mls_iterations):
                dists, idx = tree.query(px, k=k)
                dists = np.atleast_1d(dists).astype(np.float64)
                idx = np.atleast_1d(idx).astype(np.int64)
                h = dists[-1]
                if len(dists) > 1:
                    h = h + max(0.0, h - dists[-2])
                h = max(float(h), 1e-12)
                w = np.exp(-((dists / h) ** 2))
                sw = float(np.sum(w))
                if sw <= 1e-14:
                    break
                a = np.sum(original.vertices[idx] * w[:, None], axis=0) / sw
                n = np.sum(normals[idx] * w[:, None], axis=0) / sw
                n_norm = np.linalg.norm(n)
                if n_norm <= 1e-14:
                    break
                n = n / n_norm
                err = abs(float(np.dot(n, point - a)))
                if err > best_err:
                    break
                best_err = err
                px = point - np.dot(n, point - a) * n
                if err < cfg.mls_tolerance:
                    break
            out[pi] = px
        return out

    @staticmethod
    def _edge_exists(mesh: Mesh, a: int, b: int) -> bool:
        key = (a, b) if a < b else (b, a)
        return key in edge_faces(mesh.faces)


def _split_face_on_edge(face: list[int], a: int, b: int, m: int) -> list[list[int]]:
    for offset in range(3):
        x = face[offset]
        y = face[(offset + 1) % 3]
        z = face[(offset + 2) % 3]
        if x == a and y == b:
            return [[a, m, z], [m, b, z]]
        if x == b and y == a:
            return [[b, m, z], [m, a, z]]
    raise ValueError("face does not contain requested edge")


def _safe_face_normal(points: np.ndarray) -> np.ndarray:
    n = np.cross(points[1] - points[0], points[2] - points[0])
    norm = np.linalg.norm(n)
    return n / norm if norm > 1e-14 else np.zeros(3, dtype=np.float64)


def _orient_triangle(tri: list[int], vertices: np.ndarray, ref_normal: np.ndarray) -> list[int]:
    if np.linalg.norm(ref_normal) <= 1e-14:
        return tri
    pts = vertices[np.asarray(tri, dtype=np.int64)]
    n = _safe_face_normal(pts)
    if np.dot(n, ref_normal) < 0:
        return [tri[0], tri[2], tri[1]]
    return tri
