#!/usr/bin/env python3
"""Analyze high-curvature/small-target-h ROI components from target-field CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import deque
from pathlib import Path


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    data = sorted(values)
    if len(data) == 1:
        return data[0]
    pos = max(0.0, min(1.0, q)) * (len(data) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    t = pos - lo
    return data[lo] * (1.0 - t) + data[hi] * t


def read_vertices_csv(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "vertex_id": int(row["vertex_id"]),
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "z": float(row["z"]),
                    "H_physical": float(row["H_physical"]),
                    "raw_kappa": float(row["raw_kappa"]),
                    "csf_kappa": float(row["csf_kappa"]),
                    "capped_kappa": float(row["capped_kappa"]),
                    "target_h": float(row["target_h"]),
                    "h_over_base": float(row["h_over_base"]),
                    "is_capped": int(row["is_capped"]),
                }
            )
    return rows


def read_mesh(path: Path) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    suffix = path.suffix.lower()
    if suffix == ".off":
        return read_off(path)
    if suffix == ".ply":
        return read_ascii_ply(path)
    if suffix == ".obj":
        return read_obj(path)
    raise ValueError(f"unsupported mesh format: {path}")


def read_off(path: Path) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    with path.open(encoding="utf-8") as f:
        header = f.readline().strip()
        if header != "OFF":
            raise ValueError(f"not an OFF file: {path}")
        line = f.readline().strip()
        while not line or line.startswith("#"):
            line = f.readline().strip()
        nv, nf, _ = [int(x) for x in line.split()[:3]]
        vertices = [tuple(float(x) for x in f.readline().split()[:3]) for _ in range(nv)]
        faces: list[tuple[int, int, int]] = []
        for _ in range(nf):
            parts = [int(x) for x in f.readline().split()]
            if parts[0] < 3:
                continue
            ids = parts[1 : 1 + parts[0]]
            for i in range(1, len(ids) - 1):
                faces.append((ids[0], ids[i], ids[i + 1]))
        return vertices, faces


def read_ascii_ply(path: Path) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    with path.open(encoding="utf-8") as f:
        if f.readline().strip() != "ply":
            raise ValueError(f"not a PLY file: {path}")
        vertex_count = 0
        face_count = 0
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"unterminated PLY header: {path}")
            line = line.strip()
            if line == "end_header":
                break
            parts = line.split()
            if len(parts) >= 3 and parts[0] == "element" and parts[1] == "vertex":
                vertex_count = int(parts[2])
            if len(parts) >= 3 and parts[0] == "element" and parts[1] == "face":
                face_count = int(parts[2])
        vertices = [tuple(float(x) for x in f.readline().split()[:3]) for _ in range(vertex_count)]
        faces: list[tuple[int, int, int]] = []
        for _ in range(face_count):
            parts = [int(x) for x in f.readline().split()]
            if parts[0] < 3:
                continue
            ids = parts[1 : 1 + parts[0]]
            for i in range(1, len(ids) - 1):
                faces.append((ids[0], ids[i], ids[i + 1]))
        return vertices, faces


def read_obj(path: Path) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.startswith("v "):
                vertices.append(tuple(float(x) for x in line.split()[1:4]))
            elif line.startswith("f "):
                ids = [int(part.split("/")[0]) - 1 for part in line.split()[1:]]
                for i in range(1, len(ids) - 1):
                    faces.append((ids[0], ids[i], ids[i + 1]))
    return vertices, faces


def squared_distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2


def build_neighbors(vertex_count: int, faces: list[tuple[int, int, int]]) -> list[set[int]]:
    neigh = [set() for _ in range(vertex_count)]
    for a, b, c in faces:
        neigh[a].update((b, c))
        neigh[b].update((a, c))
        neigh[c].update((a, b))
    return neigh


def connected_components(selected: set[int], neigh: list[set[int]]) -> list[list[int]]:
    components: list[list[int]] = []
    remaining = set(selected)
    while remaining:
        start = remaining.pop()
        comp = [start]
        queue: deque[int] = deque([start])
        while queue:
            v = queue.popleft()
            for n in neigh[v]:
                if n in remaining:
                    remaining.remove(n)
                    comp.append(n)
                    queue.append(n)
        components.append(sorted(comp))
    components.sort(key=len, reverse=True)
    return components


def nearest_distances(
    source: list[tuple[float, float, float]],
    target: list[tuple[float, float, float]],
) -> list[float]:
    if not source or not target:
        return []
    distances: list[float] = []
    for p in source:
        best = min(squared_distance(p, q) for q in target)
        distances.append(math.sqrt(best))
    return distances


def output_vertices_near_roi(
    roi_points: list[tuple[float, float, float]],
    output_vertices: list[tuple[float, float, float]],
    radius: float,
) -> list[int]:
    if not roi_points or not output_vertices:
        return []
    r2 = radius * radius
    result = []
    for i, p in enumerate(output_vertices):
        if min(squared_distance(p, q) for q in roi_points) <= r2:
            result.append(i)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-mesh", required=True)
    parser.add_argument("--output-mesh", required=True)
    parser.add_argument("--target-field", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--csf-quantile", type=float, default=0.90)
    parser.add_argument("--h-quantile", type=float, default=0.10)
    parser.add_argument("--near-radius-factor", type=float, default=3.0)
    parser.add_argument("--max-components", type=int, default=40)
    args = parser.parse_args()

    target_dir = Path(args.target_field)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_vertices_csv(target_dir / "vertices.csv")
    input_vertices, input_faces = read_mesh(Path(args.input_mesh))
    output_vertices, output_faces = read_mesh(Path(args.output_mesh))
    if len(rows) != len(input_vertices):
        raise ValueError("target-field vertices.csv does not match input mesh vertex count")

    csf_threshold = quantile([r["csf_kappa"] for r in rows], args.csf_quantile)
    h_threshold = quantile([r["target_h"] for r in rows], args.h_quantile)
    selected = {
        r["vertex_id"]
        for r in rows
        if r["csf_kappa"] >= csf_threshold or r["target_h"] <= h_threshold
    }
    neigh = build_neighbors(len(input_vertices), input_faces)
    components = connected_components(selected, neigh)

    output_neigh = build_neighbors(len(output_vertices), output_faces)
    output_edges = {(min(i, j), max(i, j)) for i, ns in enumerate(output_neigh) for j in ns if i != j}

    report_rows = []
    for component_id, comp in enumerate(components[: max(0, args.max_components)]):
        comp_set = set(comp)
        roi_rows = [rows[i] for i in comp]
        roi_points = [input_vertices[i] for i in comp]
        mean_h = sum(r["target_h"] for r in roi_rows) / len(roi_rows)
        radius = args.near_radius_factor * mean_h
        near_ids = output_vertices_near_roi(roi_points, output_vertices, radius)
        near_set = set(near_ids)
        near_points = [output_vertices[i] for i in near_ids]
        d_in_out = nearest_distances(roi_points, output_vertices)
        d_out_in = nearest_distances(near_points, roi_points)
        local_rmse_terms = [d * d for d in d_in_out] + [d * d for d in d_out_in]
        local_rmse = math.sqrt(sum(local_rmse_terms) / len(local_rmse_terms)) if local_rmse_terms else 0.0

        input_roi_faces = sum(1 for f in input_faces if any(v in comp_set for v in f))
        output_roi_faces = sum(1 for f in output_faces if any(v in near_set for v in f))
        roi_edges = [(a, b) for a, b in output_edges if a in near_set and b in near_set]
        output_edge_lengths = [
            math.sqrt(squared_distance(output_vertices[a], output_vertices[b])) for a, b in roi_edges
        ]
        mean_output_edge = (
            sum(output_edge_lengths) / len(output_edge_lengths) if output_edge_lengths else 0.0
        )
        output_target_ratio = mean_output_edge / mean_h if mean_h > 0.0 else 0.0

        report_rows.append(
            {
                "component_id": component_id,
                "input_vertices": len(comp),
                "input_faces_any": input_roi_faces,
                "output_vertices_near": len(near_ids),
                "output_faces_any": output_roi_faces,
                "mean_csf": sum(r["csf_kappa"] for r in roi_rows) / len(roi_rows),
                "max_csf": max(r["csf_kappa"] for r in roi_rows),
                "mean_target_h": mean_h,
                "p05_target_h": quantile([r["target_h"] for r in roi_rows], 0.05),
                "p50_target_h": quantile([r["target_h"] for r in roi_rows], 0.50),
                "mean_h_over_base": sum(r["h_over_base"] for r in roi_rows) / len(roi_rows),
                "mean_output_edge": mean_output_edge,
                "output_edge_target_ratio": output_target_ratio,
                "local_input_to_output_mean": sum(d_in_out) / len(d_in_out) if d_in_out else 0.0,
                "local_output_to_input_mean": sum(d_out_in) / len(d_out_in) if d_out_in else 0.0,
                "local_symmetric_rmse": local_rmse,
                "near_radius": radius,
                "under_sampled": int(output_target_ratio > 1.65),
                "centroid_x": sum(p[0] for p in roi_points) / len(roi_points),
                "centroid_y": sum(p[1] for p in roi_points) / len(roi_points),
                "centroid_z": sum(p[2] for p in roi_points) / len(roi_points),
            }
        )

    csv_path = out_dir / "roi_components.csv"
    fieldnames = list(report_rows[0].keys()) if report_rows else ["component_id"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)

    with (out_dir / "roi_selection.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "csf_quantile": args.csf_quantile,
                "h_quantile": args.h_quantile,
                "csf_threshold": csf_threshold,
                "target_h_threshold": h_threshold,
                "selected_vertices": len(selected),
                "component_count": len(components),
                "reported_component_count": min(len(components), max(0, args.max_components)),
                "note": "Confirm T21 toe / T30 ear component_id manually from roi_components.csv centroids and mesh view.",
            },
            f,
            indent=2,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
