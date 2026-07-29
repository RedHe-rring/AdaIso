from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from mesh_metrics import bbox_summary, load_mesh, triangle_qualities


STAGES = ("split", "collapse", "flip", "smooth")
REGION_CODES = {"smooth": 0, "near_feature": 1, "feature": 2}

ColorStops = tuple[tuple[float, tuple[int, int, int]], ...]

ERROR_STOPS: ColorStops = (
    (0.0, (43, 64, 160)),
    (0.25, (86, 180, 233)),
    (0.5, (120, 210, 120)),
    (0.75, (255, 220, 70)),
    (1.0, (190, 30, 45)),
)


def json_path(path: Path) -> str:
    return str(path)


def mesh_from_arrays(vertices: np.ndarray, faces: np.ndarray) -> trimesh.Trimesh:
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    normals = np.zeros((len(faces), 3), dtype=np.float64)
    if len(faces) == 0:
        return normals
    cross = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]])
    lengths = np.linalg.norm(cross, axis=1)
    valid = lengths > 1e-14
    normals[valid] = cross[valid] / lengths[valid, None]
    return normals


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(vertices, dtype=np.float64)
    if len(faces) == 0:
        return normals
    cross = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]])
    for corner in range(3):
        np.add.at(normals, faces[:, corner], cross)
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1e-14
    normals[valid] /= lengths[valid, None]
    return normals


def edge_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def topology(vertices_count: int, faces: np.ndarray) -> tuple[dict[tuple[int, int], list[int]], list[set[int]]]:
    edge_faces: dict[tuple[int, int], list[int]] = {}
    neighbors: list[set[int]] = [set() for _ in range(vertices_count)]
    for face_index, (a_raw, b_raw, c_raw) in enumerate(faces):
        a = int(a_raw)
        b = int(b_raw)
        c = int(c_raw)
        for u, v in ((a, b), (b, c), (c, a)):
            edge_faces.setdefault(edge_key(u, v), []).append(face_index)
            neighbors[u].add(v)
            neighbors[v].add(u)
    return edge_faces, neighbors


def normal_variation(vertex_normals_array: np.ndarray, neighbors: list[set[int]]) -> np.ndarray:
    variation = np.zeros(len(vertex_normals_array), dtype=np.float64)
    for index, nbrs in enumerate(neighbors):
        if not nbrs:
            continue
        n0 = vertex_normals_array[index]
        values = []
        for nbr in nbrs:
            dot = float(np.clip(np.dot(n0, vertex_normals_array[nbr]), -1.0, 1.0))
            values.append(math.acos(dot))
        variation[index] = max(values) if values else 0.0
    return variation


def expand_mask(mask: np.ndarray, neighbors: list[set[int]], rings: int) -> np.ndarray:
    expanded = mask.copy()
    frontier = set(np.flatnonzero(mask).tolist())
    for _ in range(max(0, rings)):
        next_frontier: set[int] = set()
        for index in frontier:
            for nbr in neighbors[index]:
                if not expanded[nbr]:
                    expanded[nbr] = True
                    next_frontier.add(nbr)
        frontier = next_frontier
        if not frontier:
            break
    return expanded


def build_feature_regions(
    vertices: np.ndarray,
    faces: np.ndarray,
    dihedral_threshold_deg: float,
    curvature_top_quantile: float,
    near_rings: int,
) -> dict[str, Any]:
    edge_faces, neighbors = topology(len(vertices), faces)
    f_normals = face_normals(vertices, faces)
    v_normals = vertex_normals(vertices, faces)
    feature = np.zeros(len(vertices), dtype=bool)

    boundary_edge_count = 0
    dihedral_edge_count = 0
    threshold_rad = math.radians(dihedral_threshold_deg)
    for (a, b), adjacent in edge_faces.items():
        if len(adjacent) == 1:
            feature[a] = True
            feature[b] = True
            boundary_edge_count += 1
        elif len(adjacent) == 2:
            dot = float(np.clip(np.dot(f_normals[adjacent[0]], f_normals[adjacent[1]]), -1.0, 1.0))
            angle = math.acos(dot)
            if angle >= threshold_rad:
                feature[a] = True
                feature[b] = True
                dihedral_edge_count += 1

    variation = normal_variation(v_normals, neighbors)
    finite_variation = variation[np.isfinite(variation)]
    curvature_threshold = float(np.quantile(finite_variation, curvature_top_quantile)) if len(finite_variation) else math.nan
    high_curvature = np.zeros(len(vertices), dtype=bool)
    if math.isfinite(curvature_threshold):
        high_curvature = (variation >= curvature_threshold) & (variation > 0.0)
        if not np.any(high_curvature) and np.max(variation, initial=0.0) > 0.0:
            fallback_count = max(1, int(math.ceil(0.01 * len(vertices))))
            high_curvature[np.argsort(variation)[-fallback_count:]] = True
    feature |= high_curvature

    expanded = expand_mask(feature, neighbors, near_rings)
    near_feature = expanded & ~feature
    smooth = ~expanded
    labels = np.zeros(len(vertices), dtype=np.int32)
    labels[near_feature] = REGION_CODES["near_feature"]
    labels[feature] = REGION_CODES["feature"]

    masks = {"global": np.ones(len(vertices), dtype=bool), "feature": feature, "near_feature": near_feature, "smooth": smooth}
    return {
        "masks": masks,
        "labels": labels,
        "neighbors": neighbors,
        "vertex_normals": v_normals,
        "normal_variation": variation,
        "metadata": {
            "dihedral_threshold_deg": float(dihedral_threshold_deg),
            "curvature_top_quantile": float(curvature_top_quantile),
            "curvature_threshold_rad": curvature_threshold,
            "near_feature_rings": int(near_rings),
            "boundary_feature_edges": int(boundary_edge_count),
            "dihedral_feature_edges": int(dihedral_edge_count),
            "high_curvature_vertices": int(np.count_nonzero(high_curvature)),
            "feature_vertices": int(np.count_nonzero(feature)),
            "near_feature_vertices": int(np.count_nonzero(near_feature)),
            "smooth_vertices": int(np.count_nonzero(smooth)),
        },
    }


def colors_from_values(values: np.ndarray, color_max: float, stops: ColorStops = ERROR_STOPS) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if not math.isfinite(color_max) or color_max <= 0.0:
        color_max = float(np.max(values)) if len(values) else 1.0
    if color_max <= 0.0:
        color_max = 1.0
    t = np.clip(values / color_max, 0.0, 1.0)
    stop_pos = np.asarray([item[0] for item in stops], dtype=np.float64)
    stop_rgb = np.asarray([item[1] for item in stops], dtype=np.float64)
    colors = np.empty((len(values), 3), dtype=np.uint8)
    for channel in range(3):
        colors[:, channel] = np.interp(t, stop_pos, stop_rgb[:, channel]).astype(np.uint8)
    return colors


def feature_mask_colors(labels: np.ndarray) -> np.ndarray:
    colors = np.full((len(labels), 3), (170, 170, 170), dtype=np.uint8)
    colors[labels == REGION_CODES["near_feature"]] = np.array((245, 195, 55), dtype=np.uint8)
    colors[labels == REGION_CODES["feature"]] = np.array((220, 35, 45), dtype=np.uint8)
    return colors


def write_colored_ply(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray,
    *,
    errors: np.ndarray | None = None,
    region_labels: np.ndarray | None = None,
    comment: str = "",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as out:
        out.write("ply\n")
        out.write("format ascii 1.0\n")
        if comment:
            out.write(f"comment {comment}\n")
        out.write(f"element vertex {len(vertices)}\n")
        out.write("property float x\n")
        out.write("property float y\n")
        out.write("property float z\n")
        out.write("property uchar red\n")
        out.write("property uchar green\n")
        out.write("property uchar blue\n")
        if errors is not None:
            out.write("property float error\n")
        if region_labels is not None:
            out.write("property int region\n")
        out.write(f"element face {len(faces)}\n")
        out.write("property list uchar int vertex_indices\n")
        out.write("end_header\n")
        for index, (point, color) in enumerate(zip(vertices, colors)):
            values = [
                f"{point[0]:.17g}",
                f"{point[1]:.17g}",
                f"{point[2]:.17g}",
                str(int(color[0])),
                str(int(color[1])),
                str(int(color[2])),
            ]
            if errors is not None:
                values.append(f"{float(errors[index]):.17g}")
            if region_labels is not None:
                values.append(str(int(region_labels[index])))
            out.write(" ".join(values) + "\n")
        for face in faces:
            out.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")


def point_to_mesh_surface(query_points: np.ndarray, target_mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    query = trimesh.proximity.ProximityQuery(target_mesh)
    closest, distances, _ = query.on_surface(query_points)
    return closest.astype(np.float64, copy=False), distances.astype(np.float64, copy=False)


def ratio(value: float, bbox_diag: float) -> float:
    return float(100.0 * value / bbox_diag) if math.isfinite(value) and bbox_diag > 0.0 else math.nan


def region_error_summary(
    distances: np.ndarray,
    signed: np.ndarray,
    mask: np.ndarray,
    bbox_diag: float,
) -> dict[str, float | int]:
    count = int(np.count_nonzero(mask))
    if count == 0:
        return {
            "count": 0,
            "mean": math.nan,
            "rmse": math.nan,
            "p95": math.nan,
            "p99": math.nan,
            "max": math.nan,
            "mean_inward": math.nan,
            "p95_inward": math.nan,
            "max_inward": math.nan,
            "mean_ratio_bbox_pct": math.nan,
            "rmse_ratio_bbox_pct": math.nan,
            "p95_ratio_bbox_pct": math.nan,
            "p99_ratio_bbox_pct": math.nan,
            "max_ratio_bbox_pct": math.nan,
            "mean_inward_ratio_bbox_pct": math.nan,
            "p95_inward_ratio_bbox_pct": math.nan,
            "max_inward_ratio_bbox_pct": math.nan,
        }
    values = distances[mask]
    signed_values = signed[mask]
    inward = np.maximum(0.0, -signed_values)
    mean = float(np.mean(values))
    rmse = float(math.sqrt(float(np.mean(values * values))))
    p95 = float(np.quantile(values, 0.95))
    p99 = float(np.quantile(values, 0.99))
    max_value = float(np.max(values))
    mean_inward = float(np.mean(inward))
    p95_inward = float(np.quantile(inward, 0.95))
    max_inward = float(np.max(inward))
    return {
        "count": count,
        "mean": mean,
        "rmse": rmse,
        "p95": p95,
        "p99": p99,
        "max": max_value,
        "mean_inward": mean_inward,
        "p95_inward": p95_inward,
        "max_inward": max_inward,
        "mean_ratio_bbox_pct": ratio(mean, bbox_diag),
        "rmse_ratio_bbox_pct": ratio(rmse, bbox_diag),
        "p95_ratio_bbox_pct": ratio(p95, bbox_diag),
        "p99_ratio_bbox_pct": ratio(p99, bbox_diag),
        "max_ratio_bbox_pct": ratio(max_value, bbox_diag),
        "mean_inward_ratio_bbox_pct": ratio(mean_inward, bbox_diag),
        "p95_inward_ratio_bbox_pct": ratio(p95_inward, bbox_diag),
        "max_inward_ratio_bbox_pct": ratio(max_inward, bbox_diag),
    }


def mesh_quality(vertices: np.ndarray, faces: np.ndarray) -> dict[str, float | int]:
    qualities = triangle_qualities(vertices, faces)
    return {
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "qavg": float(np.mean(qualities)) if len(qualities) else math.nan,
        "qmin": float(np.min(qualities)) if len(qualities) else math.nan,
    }


def analyze_snapshot(
    snapshot_name: str,
    mesh_path: Path,
    reference_vertices: np.ndarray,
    masks: dict[str, np.ndarray],
    reference_normals: np.ndarray,
    bbox_diag: float,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    vertices, faces = load_mesh(mesh_path)
    target_mesh = mesh_from_arrays(vertices, faces)
    closest, distances = point_to_mesh_surface(reference_vertices, target_mesh)
    signed = np.einsum("ij,ij->i", closest - reference_vertices, reference_normals)
    regions = {
        name: region_error_summary(distances, signed, mask, bbox_diag)
        for name, mask in masks.items()
    }
    row = {
        "snapshot": snapshot_name,
        "mesh": json_path(mesh_path),
        **mesh_quality(vertices, faces),
        "regions": regions,
    }
    return row, distances, signed


def flatten_metric_row(row: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {
        "snapshot": row["snapshot"],
        "mesh": row["mesh"],
        "vertices": row["vertices"],
        "faces": row["faces"],
        "qavg": row["qavg"],
        "qmin": row["qmin"],
    }
    for region in ("global", "feature", "near_feature", "smooth"):
        summary = row["regions"][region]
        for key, value in summary.items():
            flat[f"{region}_{key}"] = value
    for key, value in row.get("deltas", {}).items():
        flat[key] = value
    return flat


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def stage_snapshots(iteration_dir: Path | None, stage_dir: Path | None) -> list[tuple[str, Path]]:
    snapshots: list[tuple[str, Path]] = []
    if iteration_dir is not None:
        iter000 = iteration_dir / "iter000.ply"
        if iter000.exists():
            snapshots.append(("iter000", iter000))
    if stage_dir is None or not stage_dir.exists():
        return snapshots
    pattern = re.compile(r"^iter(\d+)_after_(split|collapse|flip|smooth)\.ply$")
    by_iteration: dict[int, dict[str, Path]] = {}
    for path in stage_dir.glob("iter*_after_*.ply"):
        match = pattern.match(path.name)
        if match:
            by_iteration.setdefault(int(match.group(1)), {})[match.group(2)] = path
    for iteration in sorted(by_iteration):
        for stage in STAGES:
            path = by_iteration[iteration].get(stage)
            if path is not None:
                snapshots.append((f"iter{iteration:03d}_after_{stage}", path))
    return snapshots


def add_stage_deltas(rows: list[dict[str, Any]]) -> None:
    previous: dict[str, Any] | None = None
    for row in rows:
        if previous is None:
            row["deltas"] = {
                "delta_feature_rmse": 0.0,
                "delta_feature_p95": 0.0,
                "delta_feature_max": 0.0,
                "delta_feature_mean_inward": 0.0,
                "delta_qmin": 0.0,
            }
        else:
            row["deltas"] = {
                "delta_feature_rmse": float(row["regions"]["feature"]["rmse"] - previous["regions"]["feature"]["rmse"]),
                "delta_feature_p95": float(row["regions"]["feature"]["p95"] - previous["regions"]["feature"]["p95"]),
                "delta_feature_max": float(row["regions"]["feature"]["max"] - previous["regions"]["feature"]["max"]),
                "delta_feature_mean_inward": float(
                    row["regions"]["feature"]["mean_inward"] - previous["regions"]["feature"]["mean_inward"]
                ),
                "delta_qmin": float(row["qmin"] - previous["qmin"]),
            }
        previous = row


def make_final_outputs(
    args: argparse.Namespace,
    reference_vertices: np.ndarray,
    reference_faces: np.ndarray,
    remesh_vertices: np.ndarray,
    remesh_faces: np.ndarray,
    labels: np.ndarray,
    final_reference_distances: np.ndarray,
    bbox_diag: float,
) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_colored_ply(
        args.out_dir / "feature_mask.ply",
        reference_vertices,
        reference_faces,
        feature_mask_colors(labels),
        region_labels=labels,
        comment="region: 2=feature, 1=near_feature, 0=smooth",
    )

    color_max = args.color_max if args.color_max is not None and args.color_max > 0.0 else float(
        np.quantile(final_reference_distances, args.color_clip_percentile / 100.0)
    )
    write_colored_ply(
        args.out_dir / "reference_feature_error_to_remesh.ply",
        reference_vertices,
        reference_faces,
        colors_from_values(final_reference_distances, color_max),
        errors=final_reference_distances,
        region_labels=labels,
        comment="vertex error is distance from reference mesh vertex to remesh surface; region: 2=feature, 1=near_feature, 0=smooth",
    )

    reference_mesh = mesh_from_arrays(reference_vertices, reference_faces)
    _, remesh_distances = point_to_mesh_surface(remesh_vertices, reference_mesh)
    remesh_color_max = args.color_max if args.color_max is not None and args.color_max > 0.0 else float(
        np.quantile(remesh_distances, args.color_clip_percentile / 100.0)
    )
    write_colored_ply(
        args.out_dir / "remesh_error_to_reference.ply",
        remesh_vertices,
        remesh_faces,
        colors_from_values(remesh_distances, remesh_color_max),
        errors=remesh_distances,
        comment="vertex error is distance from remesh mesh vertex to reference surface",
    )


def stage_hint(args: argparse.Namespace) -> str:
    case_dir = args.remesh.parent
    return (
        f"Rerun remesher with: --export-iteration-meshes {case_dir / 'iteration_meshes'} "
        f"--export-all-stage-meshes {case_dir / 'stage_meshes'}"
    )


def build_diagnostics(args: argparse.Namespace) -> dict[str, Any]:
    reference_vertices, reference_faces = load_mesh(args.reference)
    remesh_vertices, remesh_faces = load_mesh(args.remesh)
    bbox_diag = float(bbox_summary(reference_vertices)["diagonal"])

    regions = build_feature_regions(
        reference_vertices,
        reference_faces,
        args.dihedral_threshold_deg,
        args.curvature_top_quantile,
        args.near_feature_rings,
    )
    masks = regions["masks"]
    labels = regions["labels"]
    reference_normals = regions["vertex_normals"]

    final_row, final_reference_distances, _ = analyze_snapshot(
        "final",
        args.remesh,
        reference_vertices,
        masks,
        reference_normals,
        bbox_diag,
    )
    make_final_outputs(
        args,
        reference_vertices,
        reference_faces,
        remesh_vertices,
        remesh_faces,
        labels,
        final_reference_distances,
        bbox_diag,
    )

    stage_rows: list[dict[str, Any]] = []
    snapshots = stage_snapshots(args.iteration_dir, args.stage_dir)
    has_stage_attribution = bool(args.stage_dir and args.stage_dir.exists() and any("after_" in name for name, _ in snapshots))
    if has_stage_attribution:
        for snapshot_name, snapshot_path in snapshots:
            row, _, _ = analyze_snapshot(snapshot_name, snapshot_path, reference_vertices, masks, reference_normals, bbox_diag)
            stage_rows.append(row)
        add_stage_deltas(stage_rows)

    flat_final = flatten_metric_row(final_row)
    flat_stage = [flatten_metric_row(row) for row in stage_rows]
    write_csv(args.out_dir / "feature_metrics.csv", [flat_final] + flat_stage)
    if has_stage_attribution:
        write_csv(args.out_dir / "stage_feature_metrics.csv", flat_stage)

    worst_stage = None
    if stage_rows:
        candidates = [row for row in stage_rows if row["snapshot"] != "iter000"]
        if candidates:
            worst_stage = max(candidates, key=lambda row: row.get("deltas", {}).get("delta_feature_mean_inward", -math.inf))

    summary = {
        "method": "geometry_feature_mask_reference_to_remesh_surface_distance",
        "reference": json_path(args.reference),
        "remesh": json_path(args.remesh),
        "bbox_diagonal": bbox_diag,
        "feature_detection": regions["metadata"],
        "region_codes": REGION_CODES,
        "outputs": {
            "feature_mask": json_path(args.out_dir / "feature_mask.ply"),
            "reference_feature_error_to_remesh": json_path(args.out_dir / "reference_feature_error_to_remesh.ply"),
            "remesh_error_to_reference": json_path(args.out_dir / "remesh_error_to_reference.ply"),
            "feature_metrics_csv": json_path(args.out_dir / "feature_metrics.csv"),
            "stage_feature_metrics_csv": json_path(args.out_dir / "stage_feature_metrics.csv") if has_stage_attribution else None,
        },
        "final": {
            "mesh_quality": {key: final_row[key] for key in ("vertices", "faces", "qavg", "qmin")},
            "global": final_row["regions"]["global"],
            "feature": final_row["regions"]["feature"],
            "near_feature": final_row["regions"]["near_feature"],
            "smooth": final_row["regions"]["smooth"],
        },
        "stage_attribution": {
            "enabled": has_stage_attribution,
            "snapshot_count": len(stage_rows),
            "worst_mean_inward_jump": {
                "snapshot": worst_stage["snapshot"],
                **worst_stage["deltas"],
            }
            if worst_stage is not None
            else None,
            "missing_stage_hint": None if has_stage_attribution else stage_hint(args),
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "feature_diagnostics_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def print_summary(summary: dict[str, Any]) -> None:
    final = summary["final"]
    feature = final["feature"]
    global_metrics = final["global"]
    print("Feature remesh diagnostics")
    print(f"  Global RMSE %bbox:  {global_metrics['rmse_ratio_bbox_pct']:.6f}")
    print(f"  Feature RMSE %bbox: {feature['rmse_ratio_bbox_pct']:.6f}")
    print(f"  Feature max %bbox:  {feature['max_ratio_bbox_pct']:.6f}")
    print(f"  Max inward %bbox:   {feature['max_inward_ratio_bbox_pct']:.6f}")
    stage = summary["stage_attribution"]
    if stage["enabled"] and stage["worst_mean_inward_jump"] is not None:
        worst = stage["worst_mean_inward_jump"]
        print(f"  Worst stage jump:   {worst['snapshot']} (delta mean inward={worst['delta_feature_mean_inward']:.6g})")
    else:
        print(f"  Stage attribution:  skipped")
        print(f"  {stage['missing_stage_hint']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose remesh feature preservation on final and optional stage meshes.")
    parser.add_argument("--reference", type=Path, required=True, help="Original reference mesh before remeshing.")
    parser.add_argument("--remesh", type=Path, required=True, help="Final remeshed mesh.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for diagnostics outputs.")
    parser.add_argument("--iteration-dir", type=Path, help="Optional directory containing iter000.ply, iter001.ply, ...")
    parser.add_argument("--stage-dir", type=Path, help="Optional directory containing iterXXX_after_<stage>.ply snapshots.")
    parser.add_argument("--dihedral-threshold-deg", type=float, default=45.0, help="Feature edge dihedral threshold in degrees.")
    parser.add_argument("--curvature-top-quantile", type=float, default=0.90, help="Top normal-variation quantile marked as feature.")
    parser.add_argument("--near-feature-rings", type=int, default=2, help="Graph rings around feature vertices marked near-feature.")
    parser.add_argument("--color-clip-percentile", type=float, default=99.0, help="Distance percentile mapped to hottest color.")
    parser.add_argument("--color-max", type=float, help="Absolute distance mapped to hottest color.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 < args.curvature_top_quantile < 1.0:
        raise ValueError("--curvature-top-quantile must be in (0, 1)")
    if not 0.0 < args.color_clip_percentile <= 100.0:
        raise ValueError("--color-clip-percentile must be in (0, 100]")
    if args.near_feature_rings < 0:
        raise ValueError("--near-feature-rings must be non-negative")
    if args.dihedral_threshold_deg < 0.0 or not math.isfinite(args.dihedral_threshold_deg):
        raise ValueError("--dihedral-threshold-deg must be finite and non-negative")
    summary = build_diagnostics(args)
    print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
