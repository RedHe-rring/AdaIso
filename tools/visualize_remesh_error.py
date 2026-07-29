from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from mesh_metrics import area_sample_points, bbox_summary, load_mesh


ColorStops = tuple[tuple[float, tuple[int, int, int]], ...]


DEFAULT_STOPS: ColorStops = (
    (0.0, (43, 64, 160)),
    (0.25, (86, 180, 233)),
    (0.5, (120, 210, 120)),
    (0.75, (255, 220, 70)),
    (1.0, (190, 30, 45)),
)


def json_path(path: Path) -> str:
    return str(path)


def error_summary(distances: np.ndarray, bbox_diag: float) -> dict[str, Any]:
    squared = distances * distances
    diag = bbox_diag if math.isfinite(bbox_diag) and bbox_diag > 0.0 else math.nan

    def q(value: float) -> float:
        return float(np.quantile(distances, value))

    result = {
        "count": int(len(distances)),
        "mean": float(np.mean(distances)),
        "rmse": float(math.sqrt(float(np.mean(squared)))),
        "p50": q(0.50),
        "p90": q(0.90),
        "p95": q(0.95),
        "p99": q(0.99),
        "max": float(np.max(distances)),
    }
    if diag == diag:
        result.update(
            {
                "mean_over_bbox_diag": float(result["mean"] / diag),
                "rmse_over_bbox_diag": float(result["rmse"] / diag),
                "p95_over_bbox_diag": float(result["p95"] / diag),
                "p99_over_bbox_diag": float(result["p99"] / diag),
                "max_over_bbox_diag": float(result["max"] / diag),
                "mean_ratio_bbox_pct": float(100.0 * result["mean"] / diag),
                "rmse_ratio_bbox_pct": float(100.0 * result["rmse"] / diag),
                "p95_ratio_bbox_pct": float(100.0 * result["p95"] / diag),
                "p99_ratio_bbox_pct": float(100.0 * result["p99"] / diag),
                "max_ratio_bbox_pct": float(100.0 * result["max"] / diag),
            }
        )
    return result


def colors_from_distances(distances: np.ndarray, color_max: float, stops: ColorStops = DEFAULT_STOPS) -> np.ndarray:
    if not math.isfinite(color_max) or color_max <= 0.0:
        color_max = float(np.max(distances)) if len(distances) else 1.0
    if color_max <= 0.0:
        color_max = 1.0

    t = np.clip(distances / color_max, 0.0, 1.0)
    colors = np.empty((len(distances), 4), dtype=np.uint8)
    stop_pos = np.asarray([item[0] for item in stops], dtype=np.float64)
    stop_rgb = np.asarray([item[1] for item in stops], dtype=np.float64)
    for channel in range(3):
        colors[:, channel] = np.interp(t, stop_pos, stop_rgb[:, channel]).astype(np.uint8)
    colors[:, 3] = 255
    return colors


def mesh_from_arrays(vertices: np.ndarray, faces: np.ndarray) -> trimesh.Trimesh:
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def export_colored_mesh(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray,
    distances: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as out:
        out.write("ply\n")
        out.write("format ascii 1.0\n")
        out.write("comment vertex error is distance to the opposite mesh surface\n")
        out.write(f"element vertex {len(vertices)}\n")
        out.write("property float x\n")
        out.write("property float y\n")
        out.write("property float z\n")
        out.write("property uchar red\n")
        out.write("property uchar green\n")
        out.write("property uchar blue\n")
        out.write("property float error\n")
        out.write(f"element face {len(faces)}\n")
        out.write("property list uchar int vertex_indices\n")
        out.write("end_header\n")
        for point, color, distance in zip(vertices, colors, distances):
            out.write(
                f"{point[0]:.17g} {point[1]:.17g} {point[2]:.17g} "
                f"{int(color[0])} {int(color[1])} {int(color[2])} {float(distance):.17g}\n"
            )
        for face in faces:
            out.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")


def export_colored_points(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    cloud = trimesh.PointCloud(points, colors=colors)
    path.parent.mkdir(parents=True, exist_ok=True)
    cloud.export(path)


def nearest_distances(query_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
    tree = cKDTree(target_points)
    distances, _ = tree.query(query_points, k=1, workers=-1)
    return distances.astype(np.float64, copy=False)


def point_to_mesh_surface_distances(query_points: np.ndarray, target_mesh: trimesh.Trimesh) -> np.ndarray:
    query = trimesh.proximity.ProximityQuery(target_mesh)
    _, distances, _ = query.on_surface(query_points)
    return distances.astype(np.float64, copy=False)


def direction_payload(
    name: str,
    distances: np.ndarray,
    bbox_diag: float,
    color_clip_percentile: float,
    explicit_color_max: float | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if explicit_color_max is None:
        color_max = float(np.quantile(distances, color_clip_percentile / 100.0))
    else:
        color_max = explicit_color_max
    colors = colors_from_distances(distances, color_max)
    summary = error_summary(distances, bbox_diag)
    summary.update(
        {
            "name": name,
            "color_max": float(color_max),
            "color_max_ratio_bbox_pct": float(100.0 * color_max / bbox_diag)
            if math.isfinite(bbox_diag) and bbox_diag > 0.0
            else math.nan,
        }
    )
    return distances, colors, summary


def build_visualization(args: argparse.Namespace) -> dict[str, Any]:
    reference_vertices, reference_faces = load_mesh(args.reference)
    remesh_vertices, remesh_faces = load_mesh(args.remesh)
    reference_mesh = mesh_from_arrays(reference_vertices, reference_faces)
    remesh_mesh = mesh_from_arrays(remesh_vertices, remesh_faces)
    reference_info = bbox_summary(reference_vertices)
    bbox_diag = float(reference_info["diagonal"])

    reference_samples = area_sample_points(reference_vertices, reference_faces, args.samples)
    remesh_samples = area_sample_points(remesh_vertices, remesh_faces, args.samples)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    color_max = args.color_max if args.color_max is not None and args.color_max > 0.0 else None

    remesh_vertex_surface_distances = point_to_mesh_surface_distances(remesh_vertices, reference_mesh)
    remesh_vertex_distances, remesh_vertex_colors, remesh_vertex_summary = direction_payload(
        "remesh_vertices_to_reference_mesh",
        remesh_vertex_surface_distances,
        bbox_diag,
        args.color_clip_percentile,
        color_max,
    )
    export_colored_mesh(
        args.out_dir / "remesh_mesh_error_to_reference.ply",
        remesh_vertices,
        remesh_faces,
        remesh_vertex_colors,
        remesh_vertex_distances,
    )

    reference_vertex_surface_distances = point_to_mesh_surface_distances(reference_vertices, remesh_mesh)
    reference_vertex_distances, reference_vertex_colors, reference_vertex_summary = direction_payload(
        "reference_vertices_to_remesh_mesh",
        reference_vertex_surface_distances,
        bbox_diag,
        args.color_clip_percentile,
        color_max,
    )
    export_colored_mesh(
        args.out_dir / "reference_mesh_error_to_remesh.ply",
        reference_vertices,
        reference_faces,
        reference_vertex_colors,
        reference_vertex_distances,
    )

    out_sample_point_distances = nearest_distances(remesh_samples, reference_samples)
    out_sample_distances, out_sample_colors, out_sample_summary = direction_payload(
        "remesh_surface_samples_to_reference_samples",
        out_sample_point_distances,
        bbox_diag,
        args.color_clip_percentile,
        color_max,
    )

    ref_sample_point_distances = nearest_distances(reference_samples, remesh_samples)
    ref_sample_distances, ref_sample_colors, ref_sample_summary = direction_payload(
        "reference_surface_samples_to_remesh_samples",
        ref_sample_point_distances,
        bbox_diag,
        args.color_clip_percentile,
        color_max,
    )
    if args.export_sample_clouds:
        export_colored_points(args.out_dir / "remesh_sample_error_to_reference.ply", remesh_samples, out_sample_colors)
        export_colored_points(args.out_dir / "reference_sample_error_to_remesh.ply", reference_samples, ref_sample_colors)

    chamfer_distance = float(np.mean(out_sample_distances * out_sample_distances) + np.mean(ref_sample_distances * ref_sample_distances))
    symmetric_rmse = float(math.sqrt(0.5 * chamfer_distance))
    sampled_hausdorff = float(max(np.max(out_sample_distances), np.max(ref_sample_distances)))

    summary = {
        "method": "mesh_vertex_to_opposite_triangle_surface_distance_with_sampled_symmetric_metrics",
        "reference": json_path(args.reference),
        "remesh": json_path(args.remesh),
        "samples_per_surface": int(args.samples),
        "bbox_diagonal": bbox_diag,
        "color_clip_percentile": float(args.color_clip_percentile),
        "outputs": {
            "remesh_error_mesh": json_path(args.out_dir / "remesh_mesh_error_to_reference.ply"),
            "reference_error_mesh": json_path(args.out_dir / "reference_mesh_error_to_remesh.ply"),
            "remesh_sample_error_cloud": json_path(args.out_dir / "remesh_sample_error_to_reference.ply")
            if args.export_sample_clouds
            else None,
            "reference_sample_error_cloud": json_path(args.out_dir / "reference_sample_error_to_remesh.ply")
            if args.export_sample_clouds
            else None,
        },
        "directions": {
            remesh_vertex_summary["name"]: remesh_vertex_summary,
            reference_vertex_summary["name"]: reference_vertex_summary,
            out_sample_summary["name"]: out_sample_summary,
            ref_sample_summary["name"]: ref_sample_summary,
        },
        "symmetric": {
            "chamfer_distance": chamfer_distance,
            "symmetric_rmse": symmetric_rmse,
            "sampled_hausdorff_distance": sampled_hausdorff,
            "symmetric_rmse_ratio_bbox_pct": float(100.0 * symmetric_rmse / bbox_diag)
            if math.isfinite(bbox_diag) and bbox_diag > 0.0
            else math.nan,
            "sampled_hausdorff_ratio_bbox_pct": float(100.0 * sampled_hausdorff / bbox_diag)
            if math.isfinite(bbox_diag) and bbox_diag > 0.0
            else math.nan,
        },
    }

    summary_path = args.out_dir / "error_visualization_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export colored PLY visualizations of remesh error against a reference mesh.")
    parser.add_argument("--reference", type=Path, required=True, help="Original mesh before remeshing.")
    parser.add_argument("--remesh", type=Path, required=True, help="Remeshed output mesh.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for colored PLY files and summary JSON.")
    parser.add_argument("--samples", type=int, default=200000, help="Area samples per surface for approximate closest-distance queries.")
    parser.add_argument("--export-sample-clouds", action="store_true", help="Also export auxiliary colored sample point clouds.")
    parser.add_argument(
        "--color-clip-percentile",
        type=float,
        default=99.0,
        help="Percentile used as the hot color limit unless --color-max is set.",
    )
    parser.add_argument("--color-max", type=float, help="Absolute distance mapped to the hottest color.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.samples <= 0:
        raise ValueError("--samples must be positive")
    if not 0.0 < args.color_clip_percentile <= 100.0:
        raise ValueError("--color-clip-percentile must be in (0, 100]")
    summary = build_visualization(args)
    print(json.dumps(summary["symmetric"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
