from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from scipy.spatial import cKDTree


AREA_EPS = 1e-14


def json_escape_path(path: Path) -> str:
    return str(path)


def load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    loaded = trimesh.load(path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError(f"mesh scene has no geometry: {path}")
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    vertices = np.asarray(loaded.vertices, dtype=np.float64)
    faces = np.asarray(loaded.faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"mesh vertices are not Nx3: {path}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"mesh faces are not triangular: {path}")
    return remove_degenerate_faces(vertices, faces)


def remove_degenerate_faces(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(faces) == 0:
        return vertices.copy(), faces.copy()
    valid_index = np.all((faces >= 0) & (faces < len(vertices)), axis=1)
    faces = faces[valid_index]
    non_repeated = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 2] != faces[:, 0])
    faces = faces[non_repeated]
    if len(faces) == 0:
        return vertices.copy(), faces.copy()

    area2 = np.linalg.norm(np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                                    vertices[faces[:, 2]] - vertices[faces[:, 0]]), axis=1)
    faces = faces[area2 > AREA_EPS]
    if len(faces) == 0:
        return vertices.copy(), faces.copy()

    _, unique_indices = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)
    faces = faces[np.sort(unique_indices)]

    used = np.unique(faces.reshape(-1))
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    return vertices[used].copy(), remap[faces]


def unique_edges(faces: np.ndarray) -> np.ndarray:
    if len(faces) == 0:
        return np.empty((0, 2), dtype=np.int64)
    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges.sort(axis=1)
    return np.unique(edges, axis=0)


def edge_counts(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(faces) == 0:
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.int64)
    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges.sort(axis=1)
    return np.unique(edges, axis=0, return_counts=True)


def face_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    if len(faces) == 0:
        return np.empty(0, dtype=np.float64)
    return 0.5 * np.linalg.norm(np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                                         vertices[faces[:, 2]] - vertices[faces[:, 0]]), axis=1)


def triangle_qualities(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    if len(faces) == 0:
        return np.empty(0, dtype=np.float64)
    p0 = vertices[faces[:, 0]]
    p1 = vertices[faces[:, 1]]
    p2 = vertices[faces[:, 2]]
    l0 = np.linalg.norm(p1 - p0, axis=1)
    l1 = np.linalg.norm(p2 - p1, axis=1)
    l2 = np.linalg.norm(p0 - p2, axis=1)
    half_p = 0.5 * (l0 + l1 + l2)
    longest = np.maximum(l0, np.maximum(l1, l2))
    area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
    denom = math.sqrt(3.0) * half_p * longest
    return np.divide(6.0 * area, denom, out=np.zeros_like(area), where=denom > AREA_EPS)


def triangle_angles_degrees(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    if len(faces) == 0:
        return np.empty((0, 3), dtype=np.float64)
    pts = vertices[faces]
    result = np.empty((len(faces), 3), dtype=np.float64)
    for i in range(3):
        u = pts[:, (i + 1) % 3] - pts[:, i]
        v = pts[:, (i + 2) % 3] - pts[:, i]
        denom = np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)
        cosv = np.divide(np.einsum("ij,ij->i", u, v), denom, out=np.full(len(faces), -1.0), where=denom > AREA_EPS)
        result[:, i] = np.degrees(np.arccos(np.clip(cosv, -1.0, 1.0)))
    return result


def edge_length_summary(vertices: np.ndarray, faces: np.ndarray) -> dict[str, float | int]:
    edges = unique_edges(faces)
    if len(edges) == 0:
        return {"count": 0, "min": math.nan, "max": math.nan, "mean": math.nan, "p05": math.nan, "p50": math.nan, "p95": math.nan}
    lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    return {
        "count": int(len(lengths)),
        "min": float(np.min(lengths)),
        "max": float(np.max(lengths)),
        "mean": float(np.mean(lengths)),
        "p05": float(np.quantile(lengths, 0.05)),
        "p50": float(np.quantile(lengths, 0.50)),
        "p95": float(np.quantile(lengths, 0.95)),
    }


def connected_component_count(vertex_count: int, edges: np.ndarray) -> int:
    if vertex_count == 0:
        return 0
    parent = np.arange(vertex_count, dtype=np.int64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    for a, b in edges:
        ra = find(int(a))
        rb = find(int(b))
        if ra != rb:
            parent[rb] = ra
    return len({find(i) for i in range(vertex_count)})


def boundary_loop_count(boundary_edges: np.ndarray) -> int:
    if len(boundary_edges) == 0:
        return 0
    vertices = np.unique(boundary_edges.reshape(-1))
    index = {int(v): i for i, v in enumerate(vertices)}
    parent = np.arange(len(vertices), dtype=np.int64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    for a, b in boundary_edges:
        ia = index[int(a)]
        ib = index[int(b)]
        ra = find(ia)
        rb = find(ib)
        if ra != rb:
            parent[rb] = ra
    return len({find(i) for i in range(len(vertices))})


def regular_vertex_summary(vertex_count: int, faces: np.ndarray) -> dict[str, float | int]:
    if vertex_count == 0:
        return {"count": 0, "regular_count": 0, "regular_ratio": math.nan, "boundary_vertices": 0}
    edges, counts = edge_counts(faces)
    boundary_edges = edges[counts == 1]
    neighbors: list[set[int]] = [set() for _ in range(vertex_count)]
    for a, b in edges:
        ai = int(a)
        bi = int(b)
        neighbors[ai].add(bi)
        neighbors[bi].add(ai)
    boundary_vertices = set(int(v) for v in boundary_edges.reshape(-1)) if len(boundary_edges) else set()
    regular_count = 0
    for i, item in enumerate(neighbors):
        degree = len(item)
        if i in boundary_vertices:
            regular_count += 3 <= degree <= 5
        else:
            regular_count += 5 <= degree <= 7
    return {
        "count": int(vertex_count),
        "regular_count": int(regular_count),
        "regular_ratio": float(regular_count / vertex_count),
        "regular_ratio_pct": float(100.0 * regular_count / vertex_count),
        "boundary_vertices": int(len(boundary_vertices)),
    }


def bbox_summary(vertices: np.ndarray) -> dict[str, Any]:
    if len(vertices) == 0:
        return {"min": [math.nan, math.nan, math.nan], "max": [math.nan, math.nan, math.nan], "diagonal": math.nan}
    mn = np.min(vertices, axis=0)
    mx = np.max(vertices, axis=0)
    return {"min": mn.tolist(), "max": mx.tolist(), "diagonal": float(np.linalg.norm(mx - mn))}


def mesh_summary(vertices: np.ndarray, faces: np.ndarray, angle_threshold: float) -> dict[str, Any]:
    edges, counts = edge_counts(faces)
    boundary_edges = edges[counts == 1]
    areas = face_areas(vertices, faces)
    qualities = triangle_qualities(vertices, faces)
    angles = triangle_angles_degrees(vertices, faces)
    min_angles = np.min(angles, axis=1) if len(angles) else np.empty(0, dtype=np.float64)
    max_angles = np.max(angles, axis=1) if len(angles) else np.empty(0, dtype=np.float64)

    return {
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "edges": int(len(edges)),
        "components": int(connected_component_count(len(vertices), edges)),
        "boundary_edges": int(len(boundary_edges)),
        "boundary_loops": int(boundary_loop_count(boundary_edges)),
        "qavg": float(np.mean(qualities)) if len(qualities) else math.nan,
        "qmin": float(np.min(qualities)) if len(qualities) else math.nan,
        "area": {
            "total": float(np.sum(areas)) if len(areas) else 0.0,
            "min": float(np.min(areas)) if len(areas) else math.nan,
            "max": float(np.max(areas)) if len(areas) else math.nan,
            "mean": float(np.mean(areas)) if len(areas) else math.nan,
        },
        "angles": {
            "minimal_angle_deg": float(np.min(min_angles)) if len(min_angles) else math.nan,
            "average_minimal_angle_deg": float(np.mean(min_angles)) if len(min_angles) else math.nan,
            "maximal_angle_deg": float(np.max(max_angles)) if len(max_angles) else math.nan,
            "faces_below_threshold_ratio": float(np.mean(min_angles < angle_threshold)) if len(min_angles) else math.nan,
            "faces_below_threshold_pct": float(100.0 * np.mean(min_angles < angle_threshold)) if len(min_angles) else math.nan,
            "threshold_deg": float(angle_threshold),
        },
        "edge_lengths": edge_length_summary(vertices, faces),
        "regular_vertices": regular_vertex_summary(len(vertices), faces),
        "bbox": bbox_summary(vertices),
    }


def radical_inverse(index: int, base: int) -> float:
    result = 0.0
    factor = 1.0 / float(base)
    while index > 0:
        result += factor * float(index % base)
        index //= base
        factor /= float(base)
    return result


def area_sample_points(vertices: np.ndarray, faces: np.ndarray, samples: int) -> np.ndarray:
    if samples <= 0:
        raise ValueError("sample count must be positive")
    areas = face_areas(vertices, faces)
    positive = np.flatnonzero(areas > 0.0)
    if len(positive) == 0:
        raise ValueError("mesh has no positive-area triangles")
    cdf = np.cumsum(areas[positive])
    total_area = float(cdf[-1])
    points = np.empty((samples, 3), dtype=np.float64)
    for i in range(samples):
        seq = i + 1
        area_position = radical_inverse(seq, 2) * total_area
        face_index = int(positive[min(np.searchsorted(cdf, area_position, side="left"), len(cdf) - 1)])
        u = radical_inverse(seq, 3)
        v = radical_inverse(seq, 5)
        if u + v > 1.0:
            u = 1.0 - u
            v = 1.0 - v
        f = faces[face_index]
        a = vertices[f[0]]
        b = vertices[f[1]]
        c = vertices[f[2]]
        points[i] = a + u * (b - a) + v * (c - a)
    return points


def nearest_distance_summary(query_points: np.ndarray, target_points: np.ndarray) -> dict[str, float]:
    tree = cKDTree(target_points)
    distances, _ = tree.query(query_points, k=1)
    squared = distances * distances
    return {
        "msd": float(np.mean(squared)),
        "rmse": float(math.sqrt(float(np.mean(squared)))),
        "max": float(np.max(distances)),
        "p95": float(np.quantile(distances, 0.95)),
    }


def chamfer_summary(mesh_vertices: np.ndarray, mesh_faces: np.ndarray,
                    gt_vertices: np.ndarray, gt_faces: np.ndarray,
                    samples: int, normalization_diag: float) -> dict[str, Any]:
    mesh_points = area_sample_points(mesh_vertices, mesh_faces, samples)
    gt_points = area_sample_points(gt_vertices, gt_faces, samples)
    output_to_gt = nearest_distance_summary(mesh_points, gt_points)
    gt_to_output = nearest_distance_summary(gt_points, mesh_points)
    chamfer_distance = output_to_gt["msd"] + gt_to_output["msd"]
    symmetric_rmse = math.sqrt(0.5 * chamfer_distance)
    hausdorff = max(output_to_gt["max"], gt_to_output["max"])
    diag = normalization_diag if normalization_diag > 0.0 and math.isfinite(normalization_diag) else math.nan
    return {
        "method": "area_sampled",
        "samples_per_direction": int(samples),
        "output_to_gt_msd": output_to_gt["msd"],
        "gt_to_output_msd": gt_to_output["msd"],
        "chamfer_distance": float(chamfer_distance),
        "symmetric_rmse": float(symmetric_rmse),
        "output_to_gt_rmse": output_to_gt["rmse"],
        "gt_to_output_rmse": gt_to_output["rmse"],
        "output_to_gt_max": output_to_gt["max"],
        "gt_to_output_max": gt_to_output["max"],
        "sampled_hausdorff_distance": float(hausdorff),
        "output_to_gt_p95": output_to_gt["p95"],
        "gt_to_output_p95": gt_to_output["p95"],
        "symmetric_rmse_over_bbox_diag": float(symmetric_rmse / diag) if diag == diag else math.nan,
        "symmetric_rmse_ratio_bbox_pct": float(100.0 * symmetric_rmse / diag) if diag == diag else math.nan,
        "chamfer_distance_over_bbox_diag2": float(chamfer_distance / (diag * diag)) if diag == diag else math.nan,
        "sampled_hausdorff_over_bbox_diag": float(hausdorff / diag) if diag == diag else math.nan,
        "sampled_hausdorff_ratio_bbox_pct": float(100.0 * hausdorff / diag) if diag == diag else math.nan,
    }


def load_optional_json(path: Path | None) -> Any:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def final_iteration_from_summary(summary: dict[str, Any]) -> dict[str, Any] | None:
    iterations = summary.get("iterations")
    if isinstance(iterations, list) and iterations:
        return iterations[-1]
    return None


def build_metrics(args: argparse.Namespace) -> dict[str, Any]:
    mesh_vertices, mesh_faces = load_mesh(args.mesh)
    mesh_info = mesh_summary(mesh_vertices, mesh_faces, args.angle_threshold)

    result: dict[str, Any] = {
        "mesh": json_escape_path(args.mesh),
        "cd_samples": int(args.cd_samples),
        "mesh_summary": mesh_info,
    }

    gt_vertices = gt_faces = None
    gt_info = None
    if args.gt is not None:
        gt_vertices, gt_faces = load_mesh(args.gt)
        gt_info = mesh_summary(gt_vertices, gt_faces, args.angle_threshold)
        result["ground_truth"] = json_escape_path(args.gt)
        result["gt_summary"] = gt_info

        diag = float(gt_info["bbox"]["diagonal"])
        chamfer = chamfer_summary(mesh_vertices, mesh_faces, gt_vertices, gt_faces, args.cd_samples, diag)
        result["chamfer_distance"] = chamfer
        result["sampled_hausdorff"] = {
            "method": "area_sampled_pointset",
            "samples_per_direction": chamfer["samples_per_direction"],
            "output_to_gt_max": chamfer["output_to_gt_max"],
            "gt_to_output_max": chamfer["gt_to_output_max"],
            "symmetric": chamfer["sampled_hausdorff_distance"],
            "symmetric_ratio_bbox_pct": chamfer["sampled_hausdorff_ratio_bbox_pct"],
        }
        result["paper_like"] = {
            "maximal_error": chamfer["sampled_hausdorff_distance"],
            "maximal_error_ratio_pct": chamfer["sampled_hausdorff_ratio_bbox_pct"],
            "rms_face_error_proxy": chamfer["symmetric_rmse"],
            "rms_face_error_ratio_pct": chamfer["symmetric_rmse_ratio_bbox_pct"],
            "angles_smaller_than_threshold_pct": mesh_info["angles"]["faces_below_threshold_pct"],
            "angles_smaller_than_30_degree_pct": mesh_summary(mesh_vertices, mesh_faces, 30.0)["angles"]["faces_below_threshold_pct"],
            "regular_vertices_ratio_pct": mesh_info["regular_vertices"]["regular_ratio_pct"],
        }

    target_field_stats = load_optional_json(args.target_field_stats)
    if target_field_stats is not None:
        result["target_field_stats"] = target_field_stats

    adaiso_summary = load_optional_json(args.adaiso_summary_json)
    if isinstance(adaiso_summary, dict):
        result["adaiso_summary"] = {
            "parameters": adaiso_summary.get("parameters", {}),
            "stats": adaiso_summary.get("stats", {}),
            "final_iteration": final_iteration_from_summary(adaiso_summary),
        }

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute mesh metrics without using cpp_adaiso eval.cpp.")
    parser.add_argument("--mesh", type=Path, required=True, help="Mesh to evaluate.")
    parser.add_argument("--gt", type=Path, help="Reference/ground-truth mesh for CD/RMSE/Hausdorff proxies.")
    parser.add_argument("--cd-samples", type=int, default=100000, help="Area samples per CD direction. Matches eval.cpp default.")
    parser.add_argument("--angle-threshold", type=float, default=30.0, help="Angle threshold for small-angle face ratio.")
    parser.add_argument("--target-field-stats", type=Path, help="Optional cpp_adaiso target_field/field_stats.json to merge.")
    parser.add_argument("--adaiso-summary-json", type=Path, help="Optional cpp_adaiso summary.json to merge final iteration metrics.")
    parser.add_argument("--json", type=Path, help="Write JSON metrics to this path.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.cd_samples <= 0:
        raise ValueError("--cd-samples must be positive")
    metrics = build_metrics(args)
    text = json.dumps(metrics, ensure_ascii=False, indent=2 if args.pretty else None, sort_keys=False)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
