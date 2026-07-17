#!/usr/bin/env python3
"""Export colored PLY visualizations for high-curvature ROI components."""

from __future__ import annotations

import argparse
import csv
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
                    "csf_kappa": float(row["csf_kappa"]),
                    "target_h": float(row["target_h"]),
                }
            )
    return rows


def read_off(path: Path) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    with path.open(encoding="utf-8") as f:
        header = f.readline().strip()
        if header != "OFF":
            raise ValueError(f"not an OFF file: {path}")
        line = f.readline().strip()
        while not line or line.startswith("#"):
            line = f.readline().strip()
        vertex_count, face_count, _ = [int(x) for x in line.split()[:3]]
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


def write_colored_ply(
    path: Path,
    vertices: list[tuple[float, float, float]],
    faces: list[tuple[int, int, int]],
    highlighted: set[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for i, (x, y, z) in enumerate(vertices):
            if i in highlighted:
                color = (220, 30, 30)
            else:
                color = (180, 180, 180)
            f.write(f"{x:.17g} {y:.17g} {z:.17g} {color[0]} {color[1]} {color[2]}\n")
        for a, b, c in faces:
            f.write(f"3 {a} {b} {c}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-mesh", required=True)
    parser.add_argument("--target-field", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--component-id", type=int, default=0)
    parser.add_argument("--csf-quantile", type=float, default=0.90)
    parser.add_argument("--h-quantile", type=float, default=0.10)
    args = parser.parse_args()

    vertices, faces = read_off(Path(args.input_mesh))
    rows = read_vertices_csv(Path(args.target_field) / "vertices.csv")
    if len(rows) != len(vertices):
        raise ValueError("target-field vertices.csv does not match mesh vertex count")

    csf_threshold = quantile([r["csf_kappa"] for r in rows], args.csf_quantile)
    h_threshold = quantile([r["target_h"] for r in rows], args.h_quantile)
    selected = {
        r["vertex_id"]
        for r in rows
        if r["csf_kappa"] >= csf_threshold or r["target_h"] <= h_threshold
    }
    components = connected_components(selected, build_neighbors(len(vertices), faces))
    if args.component_id < 0 or args.component_id >= len(components):
        raise ValueError(f"component_id {args.component_id} out of range; count={len(components)}")

    write_colored_ply(Path(args.out), vertices, faces, set(components[args.component_id]))
    print(f"wrote {args.out} with component {args.component_id} ({len(components[args.component_id])} vertices)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
