#!/usr/bin/env python3
"""Export target-field scalar CSV columns as colored PLY heatmaps."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


PALETTE = [
    (45, 70, 180),
    (0, 170, 210),
    (80, 190, 80),
    (245, 210, 70),
    (210, 40, 40),
]


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


def read_vertices_csv(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items() if k != "vertex_id"})
    return rows


def lerp(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    return tuple(round((1.0 - t) * a[i] + t * b[i]) for i in range(3))


def heat_color(t: float) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    scaled = t * (len(PALETTE) - 1)
    idx = min(int(math.floor(scaled)), len(PALETTE) - 2)
    return lerp(PALETTE[idx], PALETTE[idx + 1], scaled - idx)


def colors_for(values: list[float], lo: float, hi: float, invert: bool = False) -> list[tuple[int, int, int]]:
    if hi <= lo or not math.isfinite(lo) or not math.isfinite(hi):
        return [heat_color(0.5) for _ in values]
    colors = []
    for value in values:
        t = (value - lo) / (hi - lo)
        if invert:
            t = 1.0 - t
        colors.append(heat_color(t))
    return colors


def write_ply(
    path: Path,
    vertices: list[tuple[float, float, float]],
    faces: list[tuple[int, int, int]],
    colors: list[tuple[int, int, int]],
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
        for (x, y, z), (r, g, b) in zip(vertices, colors):
            f.write(f"{x:.17g} {y:.17g} {z:.17g} {r} {g} {b}\n")
        for a, b, c in faces:
            f.write(f"3 {a} {b} {c}\n")


def export_scalar(
    out_dir: Path,
    name: str,
    values: list[float],
    vertices: list[tuple[float, float, float]],
    faces: list[tuple[int, int, int]],
    lo: float,
    hi: float,
    invert: bool = False,
) -> dict[str, float | str | bool]:
    write_ply(out_dir / f"{name}.ply", vertices, faces, colors_for(values, lo, hi, invert))
    return {"file": f"{name}.ply", "min": lo, "max": hi, "inverted": invert}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-mesh", required=True)
    parser.add_argument("--target-field", required=True)
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--low-q", type=float, default=0.02)
    parser.add_argument("--high-q", type=float, default=0.98)
    args = parser.parse_args()

    target_field = Path(args.target_field)
    out_dir = Path(args.out_dir) if args.out_dir else target_field / "heatmaps"
    vertices, faces = read_off(Path(args.input_mesh))
    rows = read_vertices_csv(target_field / "vertices.csv")
    if len(rows) != len(vertices):
        raise ValueError("vertices.csv row count does not match input mesh")

    columns = {
        "raw_kappa": [r["raw_kappa"] for r in rows],
        "csf_kappa": [r["csf_kappa"] for r in rows],
        "capped_kappa": [r["capped_kappa"] for r in rows],
        "target_h": [r["target_h"] for r in rows],
    }

    scales: dict[str, dict[str, float | str | bool]] = {}
    raw_csf_values = columns["raw_kappa"] + columns["csf_kappa"]
    shared_lo = quantile(raw_csf_values, args.low_q)
    shared_hi = quantile(raw_csf_values, args.high_q)
    scales["raw_kappa_heat_shared"] = export_scalar(
        out_dir, "raw_kappa_heat_shared", columns["raw_kappa"], vertices, faces, shared_lo, shared_hi
    )
    scales["csf_kappa_heat_shared"] = export_scalar(
        out_dir, "csf_kappa_heat_shared", columns["csf_kappa"], vertices, faces, shared_lo, shared_hi
    )

    for col in ("raw_kappa", "csf_kappa", "capped_kappa"):
        values = columns[col]
        lo = quantile(values, args.low_q)
        hi = quantile(values, args.high_q)
        scales[f"{col}_heat_independent"] = export_scalar(
            out_dir, f"{col}_heat_independent", values, vertices, faces, lo, hi
        )

    h_values = columns["target_h"]
    scales["target_h_heat_observed"] = export_scalar(
        out_dir,
        "target_h_heat_observed",
        h_values,
        vertices,
        faces,
        quantile(h_values, args.low_q),
        quantile(h_values, args.high_q),
        invert=True,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "color_scales.json").write_text(json.dumps(scales, indent=2), encoding="utf-8")
    print(f"wrote heatmaps to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
