#!/usr/bin/env python3
"""Run the v2.5.0 target-field gradation diagnostic matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import struct
from pathlib import Path


GROUPS = {
    "B0_raw_mapping": {
        "target_smooth_alpha": "0",
        "target_max_adjacent_ratio": "1.0",
        "target_gradation_iters": "0",
    },
    "B1_smooth_only": {
        "target_smooth_alpha": "0.3",
        "target_max_adjacent_ratio": "1.0",
        "target_gradation_iters": "0",
    },
    "V25_default": {
        "target_smooth_alpha": "0.3",
        "target_max_adjacent_ratio": "1.30",
        "target_gradation_iters": "20",
    },
    "V25_strict": {
        "target_smooth_alpha": "0.5",
        "target_max_adjacent_ratio": "1.25",
        "target_gradation_iters": "20",
    },
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_exe(root: Path) -> Path:
    candidates = [
        root / "cpp_adaiso" / "build" / "adaiso_cli.exe",
        root / "cpp_adaiso" / "build" / "Release" / "adaiso_cli.exe",
        root / "cpp_adaiso" / "build" / "Debug" / "adaiso_cli.exe",
        root / "cpp_adaiso" / "build" / "adaiso_cli",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def write_ridge_strip(path: Path, nu: int = 161, nv: int = 49) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices = []
    for j in range(nv):
        v = -0.3 + 0.6 * j / (nv - 1)
        for i in range(nu):
            u = -1.0 + 2.0 * i / (nu - 1)
            z = 0.12 * math.exp(-((u / 0.08) ** 2))
            vertices.append((u, v, z))

    faces = []
    for j in range(nv - 1):
        for i in range(nu - 1):
            a = j * nu + i + 1
            b = a + 1
            c = a + nu
            d = c + 1
            faces.append((a, b, d))
            faces.append((a, d, c))

    with path.open("w", encoding="utf-8") as out:
        for x, y, z in vertices:
            out.write(f"v {x:.17g} {y:.17g} {z:.17g}\n")
        for a, b, c in faces:
            out.write(f"f {a} {b} {c}\n")


def is_ascii_ply(path: Path) -> bool:
    with path.open("rb") as f:
        header = f.read(512)
    return b"format ascii" in header


def convert_simple_binary_ply_to_obj(path: Path, converted: Path) -> bool:
    """Convert the simple binary PLY layout used by Th2.ply to OBJ."""
    with path.open("rb") as f:
        header = []
        while True:
            line = f.readline()
            if not line:
                return False
            text = line.decode("ascii", errors="strict").strip()
            header.append(text)
            if text == "end_header":
                break

        if "format binary_little_endian 1.0" not in header:
            return False
        vertex_count = None
        face_count = None
        vertex_properties = []
        in_vertex = False
        in_face = False
        for line in header:
            parts = line.split()
            if len(parts) == 3 and parts[0] == "element":
                in_vertex = parts[1] == "vertex"
                in_face = parts[1] == "face"
                if in_vertex:
                    vertex_count = int(parts[2])
                if in_face:
                    face_count = int(parts[2])
            elif len(parts) == 3 and parts[0] == "property" and in_vertex:
                vertex_properties.append((parts[1], parts[2]))
            elif len(parts) == 5 and parts[0] == "property" and parts[1] == "list" and in_face:
                if parts[2:] != ["uchar", "int", "vertex_indices"]:
                    return False

        if vertex_count is None or face_count is None:
            return False
        if vertex_properties[:3] != [("float", "x"), ("float", "y"), ("float", "z")]:
            return False
        if len(vertex_properties) != 3:
            return False

        vertices = [struct.unpack("<fff", f.read(12)) for _ in range(vertex_count)]
        faces = []
        for _ in range(face_count):
            count_bytes = f.read(1)
            if len(count_bytes) != 1:
                return False
            count = struct.unpack("<B", count_bytes)[0]
            indices = struct.unpack("<" + "i" * count, f.read(4 * count))
            if count == 3:
                faces.append(indices)

    converted.parent.mkdir(parents=True, exist_ok=True)
    with converted.open("w", encoding="utf-8", newline="\n") as out:
        for x, y, z in vertices:
            out.write(f"v {x:.9g} {y:.9g} {z:.9g}\n")
        for a, b, c in faces:
            out.write(f"f {a + 1} {b + 1} {c + 1}\n")
    return True


def resolve_input_mesh(path: Path, converted_dir: Path) -> Path | None:
    if path.suffix.lower() != ".ply" or is_ascii_ply(path):
        return path

    converted = converted_dir / f"{path.stem}_ascii.obj"
    if converted.exists():
        return converted
    if convert_simple_binary_ply_to_obj(path, converted):
        return converted
    try:
        import trimesh  # type: ignore
    except ImportError:
        print(f"skip {path}: binary PLY requires trimesh for conversion")
        return None

    converted.parent.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.load(path, process=False)
    mesh.export(converted)
    return converted


def run_case(
    exe: Path,
    model_name: str,
    input_path: Path,
    group_name: str,
    params: dict[str, str],
    out_root: Path,
    iterations: int,
) -> dict:
    case_dir = out_root / group_name / model_name
    export_dir = case_dir / "target_field"
    output_mesh = case_dir / f"{model_name}_iters{iterations}.ply"
    summary_json = case_dir / "run_summary.json"
    case_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(exe),
        str(input_path),
        "-o",
        str(output_mesh),
        "--mesh-scale",
        "1.0",
        "--csf-alpha",
        "0.5",
        "--target-min-factor",
        "0.6",
        "--target-max-factor",
        "1.8",
        "--target-gamma",
        "1.0",
        "--target-smooth-alpha",
        params["target_smooth_alpha"],
        "--target-max-adjacent-ratio",
        params["target_max_adjacent_ratio"],
        "--target-gradation-iters",
        params["target_gradation_iters"],
        "--iters",
        str(iterations),
        "--project",
        "0",
        "--quiet",
        "--export-target-field",
        str(export_dir),
        "--summary-json",
        str(summary_json),
    ]
    with (case_dir / "run_stdout.log").open("w", encoding="utf-8") as stdout:
        subprocess.run(cmd, check=True, stdout=stdout, stderr=subprocess.STDOUT)

    stats_path = export_dir / "field_stats.json"
    with stats_path.open("r", encoding="utf-8") as f:
        stats = json.load(f)
    return {
        "group": group_name,
        "model": model_name,
        "iterations": iterations,
        "input": str(input_path),
        "export_dir": str(export_dir),
        "max_edge_ratio": stats.get("max_edge_ratio"),
        "p95_edge_ratio": stats.get("p95_edge_ratio"),
        "p99_edge_ratio": stats.get("p99_edge_ratio"),
        "violation_rate": stats.get("violation_rate"),
        "violation_count": stats.get("violation_count"),
        "range_ok": stats.get("range_ok"),
        "acceptance_pass": stats.get("acceptance_pass"),
        "max_grad_log_h_norm": stats.get("max_grad_log_h_norm"),
        "p95_grad_log_h_norm": stats.get("p95_grad_log_h_norm"),
        "theoretical_min_edge_rings": stats.get("theoretical_min_edge_rings"),
    }


def main() -> int:
    root = repo_root()
    parser = argparse.ArgumentParser(description="Run v2.5.0 target-field gradation diagnostics.")
    parser.add_argument("--exe", type=Path, default=default_exe(root), help="Path to adaiso_cli executable.")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=root / "cpp_adaiso" / "results" / "v2.5.0_target_field_gradation",
        help="Experiment output directory.",
    )
    parser.add_argument("--stop-on-failure", action="store_true", help="Stop after the first failed case.")
    parser.add_argument("--iters", type=int, default=0, help="Remeshing iterations to run. Default: 0.")
    args = parser.parse_args()

    if not args.exe.exists():
        raise SystemExit(f"adaiso_cli executable not found: {args.exe}")
    if args.iters < 0:
        raise SystemExit("--iters must be non-negative")

    input_dir = args.outdir / "_inputs"
    ridge = input_dir / "ridge_strip.obj"
    write_ridge_strip(ridge)

    model_paths = {
        "ridge_strip": ridge,
        "bunny": root / "cpp_adaiso" / "examples" / "bunny.obj",
        "hand": root / "cpp_adaiso" / "examples" / "hand.obj",
        "T1": root / "cpp_adaiso" / "examples" / "T1.off",
        "Th2": root / "cpp_adaiso" / "examples" / "Th2.ply",
    }

    converted_dir = args.outdir / "_converted"
    rows = []
    failures = []
    for group_name, params in GROUPS.items():
        for model_name, model_path in model_paths.items():
            if not model_path.exists():
                print(f"skip {model_name}: missing {model_path}")
                continue
            input_path = resolve_input_mesh(model_path, converted_dir)
            if input_path is None:
                continue
            print(f"run {group_name}/{model_name}")
            try:
                rows.append(run_case(args.exe, model_name, input_path, group_name, params, args.outdir, args.iters))
            except subprocess.CalledProcessError as exc:
                failures.append((group_name, model_name, exc.returncode))
                print(f"failed {group_name}/{model_name}: exit {exc.returncode}")
                if args.stop_on_failure:
                    raise

    summary_path = args.outdir / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "group",
        "model",
        "iterations",
        "input",
        "export_dir",
        "max_edge_ratio",
        "p95_edge_ratio",
        "p99_edge_ratio",
        "violation_rate",
        "violation_count",
        "range_ok",
        "acceptance_pass",
        "max_grad_log_h_norm",
        "p95_grad_log_h_norm",
        "theoretical_min_edge_rings",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {summary_path}")
    if failures:
        print(f"{len(failures)} cases failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
