from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = Path(__file__).resolve().parent
CLI_PATH = ROOT / "cpp_adaiso" / "build" / "adaiso_cli.exe"
INPUT_MESH = ROOT / "cpp_adaiso" / "examples" / "T21.off"
CONFIG_HEADER = ROOT / "cpp_adaiso" / "include" / "adaiso" / "AdaIsoRemesher.hpp"
RESULT_ROOT = ROOT / "cpp_adaiso" / "results" / "T21_inner5_hmin020_iter5_vs_10_shrink_compare"
REPORT_PATH = REPORT_DIR / "T21_inner5_hmin020_iter5_vs_10收缩对比实验报告.md"

STAGES = ("split", "collapse", "flip", "smooth")


@dataclass(frozen=True)
class Group:
    name: str
    iters: int
    project: bool

    @property
    def final_name(self) -> str:
        return "final_project1.ply" if self.project else "final_project0.ply"


GROUPS = (
    Group("iter5_project1", 5, True),
    Group("iter5_project0", 5, False),
    Group("iter10_project1", 10, True),
    Group("iter10_project0", 10, False),
)


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_experiments() -> None:
    if not CLI_PATH.exists():
        raise RuntimeError(f"missing CLI executable: {CLI_PATH}")
    if not INPUT_MESH.exists():
        raise RuntimeError(f"missing input mesh: {INPUT_MESH}")

    for group in GROUPS:
        group_dir = RESULT_ROOT / group.name
        group_dir.mkdir(parents=True, exist_ok=True)
        command = [
            rel(CLI_PATH),
            rel(INPUT_MESH),
            "-o",
            rel(group_dir / group.final_name),
            "--iters",
            str(group.iters),
            "--preprocess",
            "0",
            "--project",
            "1" if group.project else "0",
            "--curvature-mode",
            "robust-mean",
            "--csf-alpha",
            "2.0",
            "--target-min-factor",
            "0.20",
            "--target-max-factor",
            "1.8",
            "--target-size-gamma",
            "2.0",
            "--gt",
            rel(INPUT_MESH),
            "--cd-samples",
            "100000",
            "--export-target-field",
            rel(group_dir / "target_field"),
            "--export-iteration-meshes",
            rel(group_dir / "iteration_meshes"),
            "--export-all-stage-meshes",
            rel(group_dir / "stage_meshes"),
            "--summary-json",
            rel(group_dir / "summary.json"),
            "--quiet",
        ]
        print(f"running {group.name}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


def read_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    suffix = path.suffix.lower()
    if suffix == ".off":
        return read_off(path)
    if suffix == ".ply":
        return read_ply(path)
    raise RuntimeError(f"unsupported mesh format: {path}")


def read_off(path: Path) -> tuple[np.ndarray, np.ndarray]:
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not lines or lines[0] != "OFF":
        raise RuntimeError(f"invalid OFF file: {path}")
    counts = lines[1].split()
    vertex_count = int(counts[0])
    face_count = int(counts[1])
    vertices = np.array(
        [[float(item) for item in lines[2 + i].split()[:3]] for i in range(vertex_count)],
        dtype=np.float64,
    )
    faces = []
    offset = 2 + vertex_count
    for i in range(face_count):
        parts = lines[offset + i].split()
        count = int(parts[0])
        if count != 3:
            raise RuntimeError(f"non-triangle face in {path}")
        faces.append([int(parts[1]), int(parts[2]), int(parts[3])])
    return vertices, np.array(faces, dtype=np.int32)


def read_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    lines = path.read_text(encoding="utf-8").splitlines()
    vertex_count: int | None = None
    face_count: int | None = None
    end_header = None
    for index, line in enumerate(lines):
        if line.startswith("element vertex "):
            vertex_count = int(line.split()[2])
        elif line.startswith("element face "):
            face_count = int(line.split()[2])
        elif line == "end_header":
            end_header = index
            break
    if vertex_count is None or face_count is None or end_header is None:
        raise RuntimeError(f"invalid PLY file: {path}")
    first_vertex = end_header + 1
    first_face = first_vertex + vertex_count
    vertices = np.array(
        [[float(item) for item in lines[first_vertex + i].split()[:3]] for i in range(vertex_count)],
        dtype=np.float64,
    )
    faces = []
    for i in range(face_count):
        parts = lines[first_face + i].split()
        count = int(parts[0])
        if count != 3:
            raise RuntimeError(f"non-triangle face in {path}")
        faces.append([int(parts[1]), int(parts[2]), int(parts[3])])
    return vertices, np.array(faces, dtype=np.int32)


def write_colored_ply(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as out:
        out.write("ply\n")
        out.write("format ascii 1.0\n")
        out.write(f"element vertex {len(vertices)}\n")
        out.write("property float x\n")
        out.write("property float y\n")
        out.write("property float z\n")
        out.write("property uchar red\n")
        out.write("property uchar green\n")
        out.write("property uchar blue\n")
        out.write(f"element face {len(faces)}\n")
        out.write("property list uchar int vertex_indices\n")
        out.write("end_header\n")
        for vertex, color in zip(vertices, colors):
            out.write(
                f"{vertex[0]:.17g} {vertex[1]:.17g} {vertex[2]:.17g} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )
        for face in faces:
            out.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")


def triangle_quality(vertices: np.ndarray, faces: np.ndarray) -> tuple[float, float]:
    if len(faces) == 0:
        return 0.0, 0.0
    tri = vertices[faces]
    l0 = np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1)
    l1 = np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1)
    l2 = np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1)
    half_p = 0.5 * (l0 + l1 + l2)
    longest = np.maximum.reduce([l0, l1, l2])
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    denom = math.sqrt(3.0) * half_p * longest
    quality = np.zeros_like(area)
    ok = denom > 1e-14
    quality[ok] = 6.0 * area[ok] / denom[ok]
    return float(np.mean(quality)), float(np.min(quality))


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(vertices)
    tri = vertices[faces]
    raw = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    areas = 0.5 * np.linalg.norm(raw, axis=1)
    face_normals = raw.copy()
    lengths = np.linalg.norm(face_normals, axis=1)
    ok = lengths > 1e-14
    face_normals[ok] /= lengths[ok, None]
    weighted = face_normals * areas[:, None]
    for face, normal in zip(faces, weighted):
        normals[face[0]] += normal
        normals[face[1]] += normal
        normals[face[2]] += normal
    lengths = np.linalg.norm(normals, axis=1)
    ok = lengths > 1e-14
    normals[ok] /= lengths[ok, None]
    return normals


def closest_point_on_triangle(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = float(np.dot(ab, ap))
    d2 = float(np.dot(ac, ap))
    if d1 <= 0.0 and d2 <= 0.0:
        return a

    bp = p - b
    d3 = float(np.dot(ab, bp))
    d4 = float(np.dot(ac, bp))
    if d3 >= 0.0 and d4 <= d3:
        return b

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return a + v * ab

    cp = p - c
    d5 = float(np.dot(ab, cp))
    d6 = float(np.dot(ac, cp))
    if d6 >= 0.0 and d5 <= d6:
        return c

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return a + w * ac

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + w * (c - b)

    denom = va + vb + vc
    if abs(denom) <= 1e-14:
        return a
    v = vb / denom
    w = vc / denom
    return a + ab * v + ac * w


class MeshQuery:
    def __init__(self, vertices: np.ndarray, faces: np.ndarray) -> None:
        self.triangles = vertices[faces]
        self.centroids = self.triangles.mean(axis=1)
        self.tree = cKDTree(self.centroids)

    def closest_points(self, points: np.ndarray, candidate_count: int = 64) -> tuple[np.ndarray, np.ndarray]:
        k = min(candidate_count, len(self.triangles))
        _, indices = self.tree.query(points, k=k)
        if k == 1:
            indices = indices[:, None]
        closest = np.empty_like(points)
        sqdist = np.empty(len(points), dtype=np.float64)
        for i, point in enumerate(points):
            best_dist = float("inf")
            best_point = self.triangles[int(indices[i, 0]), 0]
            for face_index in indices[i]:
                tri = self.triangles[int(face_index)]
                candidate = closest_point_on_triangle(point, tri[0], tri[1], tri[2])
                dist = float(np.sum((candidate - point) ** 2))
                if dist < best_dist:
                    best_dist = dist
                    best_point = candidate
            closest[i] = best_point
            sqdist[i] = best_dist
        return closest, sqdist


def vertex_chamfer_proxy(vertices_a: np.ndarray, vertices_b: np.ndarray) -> float:
    tree_a = cKDTree(vertices_a)
    tree_b = cKDTree(vertices_b)
    da, _ = tree_b.query(vertices_a)
    db, _ = tree_a.query(vertices_b)
    return float(np.mean(da * da) + np.mean(db * db))


def load_target_vertices(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [
            {
                "vertex_id": int(row["vertex_id"]),
                "csf_curvature": float(row["csf_curvature"]),
                "target_h": float(row["target_h"]),
            }
            for row in reader
        ]


def selected_components(vertices: np.ndarray, faces: np.ndarray, selected: np.ndarray) -> list[np.ndarray]:
    adjacency: list[list[int]] = [[] for _ in range(len(vertices))]
    for face in faces:
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            if selected[a] and selected[b]:
                adjacency[a].append(int(b))
                adjacency[b].append(int(a))

    visited = np.zeros(len(vertices), dtype=bool)
    components: list[np.ndarray] = []
    for start in np.where(selected)[0]:
        if visited[start]:
            continue
        queue: deque[int] = deque([int(start)])
        visited[start] = True
        component = []
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in adjacency[current]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        components.append(np.array(component, dtype=np.int32))
    return components


def displacement_metrics(
    snapshot_path: Path,
    roi_indices: np.ndarray,
    input_vertices: np.ndarray,
    input_normals: np.ndarray,
) -> dict[str, float]:
    vertices, faces = read_mesh(snapshot_path)
    query = MeshQuery(vertices, faces)
    roi_points = input_vertices[roi_indices]
    roi_normals = input_normals[roi_indices]
    closest, sqdist = query.closest_points(roi_points)
    signed = np.sum((closest - roi_points) * roi_normals, axis=1)
    inward = np.maximum(0.0, -signed)
    return {
        "mean_signed": float(np.mean(signed)),
        "mean_inward": float(np.mean(inward)),
        "p95_inward": float(np.quantile(inward, 0.95)),
        "max_inward": float(np.max(inward)),
        "rms_distance": float(math.sqrt(float(np.mean(sqdist)))),
    }


def component_mask_colors(vertex_count: int, component: np.ndarray) -> np.ndarray:
    colors = np.full((vertex_count, 3), 175, dtype=np.uint8)
    colors[component] = np.array([220, 40, 40], dtype=np.uint8)
    return colors


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_roi(input_vertices: np.ndarray, input_faces: np.ndarray) -> tuple[np.ndarray, list[dict[str, object]]]:
    source_group_dir = RESULT_ROOT / "iter10_project1"
    target_rows = load_target_vertices(source_group_dir / "target_field" / "vertices.csv")
    csf = np.array([row["csf_curvature"] for row in target_rows], dtype=np.float64)
    target_h = np.array([row["target_h"] for row in target_rows], dtype=np.float64)
    csf_threshold = float(np.quantile(csf, 0.90))
    target_h_threshold = float(np.quantile(target_h, 0.10))
    selected = (csf >= csf_threshold) | (target_h <= target_h_threshold)
    components = selected_components(input_vertices, input_faces, selected)
    min_size = max(20, int(0.001 * len(input_vertices)))
    candidates = [component for component in components if len(component) >= min_size]
    if not candidates:
        candidates = sorted(components, key=len, reverse=True)[:1]
    candidates = sorted(candidates, key=len, reverse=True)[:50]

    final_path = source_group_dir / "final_project1.ply"
    input_normals = vertex_normals(input_vertices, input_faces)
    component_rows: list[dict[str, object]] = []
    for component_id, component in enumerate(candidates):
        metrics = displacement_metrics(final_path, component, input_vertices, input_normals)
        component_rows.append(
            {
                "component_id": component_id,
                "vertex_count": len(component),
                "csf_threshold": csf_threshold,
                "target_h_threshold": target_h_threshold,
                **metrics,
                "indices": component,
            }
        )

    selected_row = max(component_rows, key=lambda row: float(row["mean_inward"]))
    roi_dir = RESULT_ROOT / "roi"
    roi_dir.mkdir(parents=True, exist_ok=True)

    for rank, row in enumerate(sorted(component_rows, key=lambda item: float(item["mean_inward"]), reverse=True)[:8], 1):
        component = row["indices"]
        write_colored_ply(
            roi_dir / f"roi_component_{rank:02d}_id{row['component_id']}.ply",
            input_vertices,
            input_faces,
            component_mask_colors(len(input_vertices), component),
        )
    write_colored_ply(
        roi_dir / "selected_wing_edge_roi.ply",
        input_vertices,
        input_faces,
        component_mask_colors(len(input_vertices), selected_row["indices"]),
    )

    csv_rows = []
    for row in component_rows:
        csv_rows.append({key: value for key, value in row.items() if key != "indices"})
    write_csv(roi_dir / "roi_components.csv", csv_rows)
    metadata = {
        "selection": "max_iter10_project1_mean_inward",
        "selected_component_id": int(selected_row["component_id"]),
        "selected_vertex_count": int(selected_row["vertex_count"]),
        "csf_top_quantile": 0.90,
        "target_h_bottom_quantile": 0.10,
        "csf_threshold": csf_threshold,
        "target_h_threshold": target_h_threshold,
        "candidate_component_count": len(component_rows),
    }
    (roi_dir / "roi_selection.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return selected_row["indices"], component_rows


def snapshot_quality(path: Path, input_vertices: np.ndarray) -> tuple[int, int, float, float, float]:
    vertices, faces = read_mesh(path)
    qavg, qmin = triangle_quality(vertices, faces)
    return len(vertices), len(faces), qavg, qmin, vertex_chamfer_proxy(vertices, input_vertices)


def iteration_snapshots(group: Group, group_dir: Path) -> list[tuple[str, Path]]:
    return [
        (f"iter{i:03d}", group_dir / "iteration_meshes" / f"iter{i:03d}.ply")
        for i in range(0, group.iters + 1)
    ]


def stage_snapshots(group: Group, group_dir: Path) -> list[tuple[str, Path]]:
    snapshots = [("iter000", group_dir / "iteration_meshes" / "iter000.ply")]
    for iteration in range(1, group.iters + 1):
        for stage in STAGES:
            snapshots.append(
                (
                    f"iter{iteration:03d}_after_{stage}",
                    group_dir / "stage_meshes" / f"iter{iteration:03d}_after_{stage}.ply",
                )
            )
    return snapshots


def analyze_group(
    group: Group,
    roi_indices: np.ndarray,
    input_vertices: np.ndarray,
    input_normals: np.ndarray,
) -> dict[str, object]:
    group_dir = RESULT_ROOT / group.name
    summary = load_json(group_dir / "summary.json")
    iteration_by_index = {int(row["iteration"]): row for row in summary["iterations"]}
    final_path = group_dir / group.final_name

    iteration_rows: list[dict[str, object]] = []
    for snapshot, path in iteration_snapshots(group, group_dir):
        if not path.exists():
            raise RuntimeError(f"missing iteration snapshot: {path}")
        vertices_count, faces_count, qavg, qmin, cd_proxy = snapshot_quality(path, input_vertices)
        disp = displacement_metrics(path, roi_indices, input_vertices, input_normals)
        iteration_index = int(snapshot[-3:])
        stats = iteration_by_index.get(iteration_index, {})
        iteration_rows.append(
            {
                "snapshot": snapshot,
                "vertices": vertices_count,
                "faces": faces_count,
                "qavg": qavg,
                "qmin": qmin,
                "split_count": stats.get("split_count", 0),
                "collapse_count": stats.get("collapse_count", 0),
                "flip_count": stats.get("flip_count", 0),
                "violation_rate": stats.get("violation_rate", 0.0),
                "vertex_cd_proxy": cd_proxy,
                **disp,
            }
        )

    stage_rows: list[dict[str, object]] = []
    previous_mean = None
    previous_max = None
    for snapshot, path in stage_snapshots(group, group_dir):
        if not path.exists():
            raise RuntimeError(f"missing stage snapshot: {path}")
        vertices_count, faces_count, qavg, qmin, _ = snapshot_quality(path, input_vertices)
        disp = displacement_metrics(path, roi_indices, input_vertices, input_normals)
        mean_delta = 0.0 if previous_mean is None else disp["mean_inward"] - previous_mean
        max_delta = 0.0 if previous_max is None else disp["max_inward"] - previous_max
        previous_mean = disp["mean_inward"]
        previous_max = disp["max_inward"]
        stage_rows.append(
            {
                "snapshot": snapshot,
                "vertices": vertices_count,
                "faces": faces_count,
                "qavg": qavg,
                "qmin": qmin,
                "delta_mean_inward": mean_delta,
                "delta_max_inward": max_delta,
                **disp,
            }
        )

    final_disp = displacement_metrics(final_path, roi_indices, input_vertices, input_normals)
    final_vertices, final_faces, final_qavg, final_qmin, final_cd_proxy = snapshot_quality(final_path, input_vertices)
    last_iteration = summary["iterations"][-1]
    final_row = {
        "snapshot": group.name,
        "vertices": final_vertices,
        "faces": final_faces,
        "qavg": final_qavg,
        "qmin": final_qmin,
        "split_count": last_iteration["split_count"],
        "collapse_count": last_iteration["collapse_count"],
        "flip_count": last_iteration["flip_count"],
        "violation_rate": last_iteration["violation_rate"],
        "vertex_cd_proxy": final_cd_proxy,
        "area_sampled_cd": summary["chamfer_distance"]["chamfer_distance"],
        "area_sampled_rmse": summary["chamfer_distance"]["symmetric_rmse"],
        **final_disp,
    }

    metrics_dir = group_dir / "analysis"
    write_csv(metrics_dir / "iteration_metrics.csv", iteration_rows)
    write_csv(metrics_dir / "stage_metrics.csv", stage_rows)
    write_csv(metrics_dir / "final_metrics.csv", [final_row])

    return {
        "group": group,
        "summary": summary,
        "iteration_rows": iteration_rows,
        "stage_rows": stage_rows,
        "final_row": final_row,
        "max_jump": max(stage_rows[1:], key=lambda row: float(row["delta_mean_inward"])),
    }


def expected_iteration_count(group: Group) -> int:
    return group.iters + 1


def expected_stage_count(group: Group) -> int:
    return group.iters * len(STAGES)


def expected_files_exist(group: Group) -> tuple[int, int]:
    group_dir = RESULT_ROOT / group.name
    iteration_count = sum(1 for _, path in iteration_snapshots(group, group_dir) if path.exists())
    stage_count = sum(
        1
        for _, path in stage_snapshots(group, group_dir)
        if path.name != "iter000.ply" and path.exists()
    )
    return iteration_count, stage_count


def parse_inner_defaults() -> dict[str, int]:
    text = CONFIG_HEADER.read_text(encoding="utf-8")
    defaults: dict[str, int] = {}
    for name in ("max_split_passes", "max_collapse_passes", "max_flip_passes", "mls_iterations"):
        match = re.search(rf"\bint\s+{name}\s*=\s*(\d+)\s*;", text)
        if not match:
            raise RuntimeError(f"failed to parse {name} from {CONFIG_HEADER}")
        defaults[name] = int(match.group(1))
    return defaults


def verify_outputs(analyses: dict[str, dict[str, object]]) -> None:
    defaults = parse_inner_defaults()
    for key in ("max_split_passes", "max_collapse_passes", "max_flip_passes", "mls_iterations"):
        if defaults[key] != 5:
            raise RuntimeError(f"{key} is {defaults[key]}, expected 5")

    for group in GROUPS:
        group_dir = RESULT_ROOT / group.name
        iter_count, stage_count = expected_files_exist(group)
        if iter_count != expected_iteration_count(group):
            raise RuntimeError(f"{group.name}: expected {expected_iteration_count(group)} iteration meshes, got {iter_count}")
        if stage_count != expected_stage_count(group):
            raise RuntimeError(f"{group.name}: expected {expected_stage_count(group)} stage meshes, got {stage_count}")

        summary = analyses[group.name]["summary"]
        params = summary["parameters"]
        if int(params["global_iterations"]) != group.iters:
            raise RuntimeError(f"{group.name}: wrong global_iterations")
        if int(params["mls_iterations"]) != 5:
            raise RuntimeError(f"{group.name}: wrong mls_iterations")
        if abs(float(params["target_min_factor"]) - 0.20) > 1e-12:
            raise RuntimeError(f"{group.name}: wrong target_min_factor")
        if not bool(summary["stats"]["csf_solver_success"]):
            raise RuntimeError(f"{group.name}: CSF solver failed")

        for required in (
            group_dir / "summary.json",
            group_dir / "target_field" / "vertices.csv",
            group_dir / "analysis" / "iteration_metrics.csv",
            group_dir / "analysis" / "stage_metrics.csv",
            group_dir / "analysis" / "final_metrics.csv",
        ):
            if not required.exists():
                raise RuntimeError(f"missing required output: {required}")

    if not (RESULT_ROOT / "roi" / "selected_wing_edge_roi.ply").exists():
        raise RuntimeError("missing selected ROI PLY")
    if not REPORT_PATH.exists():
        raise RuntimeError(f"missing report: {REPORT_PATH}")


def fmt(value: object, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}"


def pct(value: float) -> str:
    return f"{value * 100.0:+.2f}%"


def sci(value: object) -> str:
    return f"{float(value):.6E}"


def table(rows: list[list[object]], headers: list[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def rel_change(new: float, base: float) -> str:
    if abs(base) <= 1e-30:
        return "n/a"
    return pct((new - base) / base)


def final_compare_rows(analyses: dict[str, dict[str, object]]) -> list[list[object]]:
    rows = []
    for group in GROUPS:
        row = analyses[group.name]["final_row"]
        rows.append(
            [
                group.name,
                f"{row['vertices']}/{row['faces']}",
                fmt(row["qmin"]),
                fmt(row["violation_rate"]),
                row["split_count"],
                row["collapse_count"],
                row["flip_count"],
                sci(row["area_sampled_cd"]),
                sci(row["area_sampled_rmse"]),
                sci(row["mean_inward"]),
                sci(row["p95_inward"]),
                sci(row["max_inward"]),
            ]
        )
    return rows


def synchronized_iter5_rows(analyses: dict[str, dict[str, object]]) -> list[list[object]]:
    rows = []
    pairs = [
        ("iter5_project1 final", analyses["iter5_project1"]["final_row"]),
        ("iter10_project1 iter005", analyses["iter10_project1"]["iteration_rows"][5]),
        ("iter5_project0 final", analyses["iter5_project0"]["final_row"]),
        ("iter10_project0 iter005", analyses["iter10_project0"]["iteration_rows"][5]),
    ]
    for name, row in pairs:
        rows.append(
            [
                name,
                f"{row['vertices']}/{row['faces']}",
                fmt(row["qmin"]),
                fmt(row["violation_rate"]),
                sci(row["vertex_cd_proxy"]),
                sci(row["mean_inward"]),
                sci(row["p95_inward"]),
                sci(row["max_inward"]),
            ]
        )
    return rows


def project_compare_rows(analyses: dict[str, dict[str, object]]) -> list[list[object]]:
    rows = []
    for iters in (5, 10):
        p1 = analyses[f"iter{iters}_project1"]["final_row"]
        p0 = analyses[f"iter{iters}_project0"]["final_row"]
        mean_diff = float(p1["mean_inward"]) - float(p0["mean_inward"])
        max_diff = float(p1["max_inward"]) - float(p0["max_inward"])
        cd_diff = float(p1["area_sampled_cd"]) - float(p0["area_sampled_cd"])
        rows.append(
            [
                f"iter{iters}",
                sci(p0["mean_inward"]),
                sci(p1["mean_inward"]),
                sci(mean_diff),
                sci(p0["max_inward"]),
                sci(p1["max_inward"]),
                sci(max_diff),
                sci(cd_diff),
            ]
        )
    return rows


def max_jump_rows(analyses: dict[str, dict[str, object]]) -> list[list[object]]:
    rows = []
    for group in GROUPS:
        jump = analyses[group.name]["max_jump"]
        rows.append(
            [
                group.name,
                jump["snapshot"],
                sci(jump["delta_mean_inward"]),
                sci(jump["delta_max_inward"]),
                sci(jump["mean_inward"]),
                sci(jump["max_inward"]),
            ]
        )
    return rows


def write_report(
    component_rows: list[dict[str, object]],
    analyses: dict[str, dict[str, object]],
) -> None:
    defaults = parse_inner_defaults()
    selected_meta = load_json(RESULT_ROOT / "roi" / "roi_selection.json")

    completeness_rows = []
    for group in GROUPS:
        iter_count, stage_count = expected_files_exist(group)
        completeness_rows.append(
            [
                group.name,
                f"{iter_count} / {expected_iteration_count(group)}",
                f"{stage_count} / {expected_stage_count(group)}",
                "yes" if (RESULT_ROOT / group.name / "summary.json").exists() else "no",
                "yes" if (RESULT_ROOT / group.name / "target_field" / "vertices.csv").exists() else "no",
                "yes" if (RESULT_ROOT / group.name / "analysis" / "stage_metrics.csv").exists() else "no",
            ]
        )

    component_table_rows = []
    for row in sorted(component_rows, key=lambda item: float(item["mean_inward"]), reverse=True)[:8]:
        component_table_rows.append(
            [
                row["component_id"],
                row["vertex_count"],
                sci(row["mean_inward"]),
                sci(row["p95_inward"]),
                sci(row["max_inward"]),
                sci(row["rms_distance"]),
            ]
        )

    iter5 = analyses["iter5_project1"]["final_row"]
    iter10 = analyses["iter10_project1"]["final_row"]
    iter10_iter5 = analyses["iter10_project1"]["iteration_rows"][5]
    mean5 = float(iter5["mean_inward"])
    mean10 = float(iter10["mean_inward"])
    mean10_at5 = float(iter10_iter5["mean_inward"])
    mean_delta = mean10 - mean5
    mean_delta_pct = (mean_delta / mean5) if mean5 > 1e-30 else 0.0
    sync_delta = mean10_at5 - mean5
    sync_delta_pct = (sync_delta / mean5) if mean5 > 1e-30 else 0.0

    if abs(mean_delta) <= max(1e-8, 0.05 * max(mean5, 1e-12)):
        iter_conclusion = "iter5 final 与 iter10 final 的 mean inward 接近，主要收缩已经发生在前 5 轮。"
    elif mean_delta > 0.0:
        iter_conclusion = "iter10 final 的 mean inward 明显高于 iter5 final，外轮继续迭代会持续放大 hmin020 收缩。"
    else:
        iter_conclusion = "iter10 final 的 mean inward 低于 iter5 final，后续外轮没有继续放大该 ROI 收缩。"

    project_conclusions = []
    for iters in (5, 10):
        p1 = analyses[f"iter{iters}_project1"]["final_row"]
        p0 = analyses[f"iter{iters}_project0"]["final_row"]
        diff = float(p1["mean_inward"]) - float(p0["mean_inward"])
        if abs(diff) <= max(1e-8, 0.05 * max(float(p0["mean_inward"]), 1e-12)):
            project_conclusions.append(f"iter{iters} project0/project1 接近")
        elif diff > 0.0:
            project_conclusions.append(f"iter{iters} project1 收缩更大")
        else:
            project_conclusions.append(f"iter{iters} project0 收缩更大")
    projection_conclusion = "；".join(project_conclusions) + "，最终 MLS projection 仍不是主要收缩来源。"

    collapse_groups = [
        group.name
        for group in GROUPS
        if "collapse" in str(analyses[group.name]["max_jump"]["snapshot"])
    ]
    if len(collapse_groups) == len(GROUPS):
        stage_conclusion = "四组最大 mean inward 跳变都发生在 collapse 后，collapse 仍是优先检查对象。"
    else:
        stage_conclusion = (
            "最大阶段跳变不完全由 collapse 主导，需结合 stage_metrics.csv 检查不同外轮/投影组。"
        )

    lines = [
        "# T21 inner=5 hmin020 iter5 vs iter10 收缩对比实验报告",
        "",
        "## 实验设置",
        "",
        "本实验固定 `target_min_factor=0.20`，使用当前代码的内轮默认值 5，比较外轮 `--iters 5` 与 `--iters 10` 对 T21 翅膀边缘 ROI inward shrink 的影响。",
        "",
        f"- 输出目录：`{rel(RESULT_ROOT)}/`",
        f"- 报告脚本：`{rel(Path(__file__))}`",
        "- CLI 参数：`--preprocess 0 --curvature-mode robust-mean --csf-alpha 2.0 --target-min-factor 0.20 --target-max-factor 1.8 --target-size-gamma 2.0 --cd-samples 100000`",
        "",
        "### inner=5 配置确认",
        "",
        table(
            [
                ["max_split_passes", defaults["max_split_passes"]],
                ["max_collapse_passes", defaults["max_collapse_passes"]],
                ["max_flip_passes", defaults["max_flip_passes"]],
                ["mls_iterations", defaults["mls_iterations"]],
            ],
            ["参数", "当前默认值"],
        ),
        "",
        "## 完整性检查",
        "",
        table(completeness_rows, ["Group", "iteration meshes", "stage meshes", "summary", "target_field", "metrics"]),
        "",
        "## ROI 选择",
        "",
        "ROI 候选来自输入 T21 上 `target_h` 最小 10% 与 CSF 曲率最高 10% 的并集连通分量。所有组共用同一 ROI，默认选择 `iter10_project1` final 中 mean inward 最大的候选分量。",
        "",
        f"- selected component id: `{selected_meta['selected_component_id']}`",
        f"- selected vertices: `{selected_meta['selected_vertex_count']}`",
        f"- candidate components: `{selected_meta['candidate_component_count']}`",
        f"- ROI 文件：`{rel(RESULT_ROOT / 'roi' / 'selected_wing_edge_roi.ply')}`",
        "",
        table(component_table_rows, ["component", "vertices", "mean inward", "p95 inward", "max inward", "rms dist"]),
        "",
        "## final 对比",
        "",
        table(
            final_compare_rows(analyses),
            ["Group", "Out V/F", "Qmin", "viol", "split", "collapse", "flip", "area CD", "RMSE", "mean inward", "p95 inward", "max inward"],
        ),
        "",
        f"- iter10_project1 final mean inward 相对 iter5_project1 final 变化：`{sci(mean_delta)}` ({pct(mean_delta_pct)})。",
        f"- iter10_project1 final area CD 相对 iter5_project1 final 变化：`{rel_change(float(iter10['area_sampled_cd']), float(iter5['area_sampled_cd']))}`。",
        "",
        "## iter5 同步快照对比",
        "",
        "这里比较 `iter5_project*` final 与 `iter10_project*` 的 `iter005` 快照，用于判断两次运行在第 5 外轮是否一致。",
        "",
        table(
            synchronized_iter5_rows(analyses),
            ["snapshot", "V/F", "Qmin", "viol", "vertex CD proxy", "mean inward", "p95 inward", "max inward"],
        ),
        "",
        f"- iter10_project1 `iter005` mean inward 相对 iter5_project1 final 变化：`{sci(sync_delta)}` ({pct(sync_delta_pct)})。",
        "",
        "## project0/project1 对比",
        "",
        table(
            project_compare_rows(analyses),
            ["iters", "project0 mean", "project1 mean", "Δmean", "project0 max", "project1 max", "Δmax", "Δarea CD"],
        ),
        "",
        projection_conclusion,
        "",
        "## 最大阶段跳变",
        "",
        table(max_jump_rows(analyses), ["Group", "snapshot", "Δmean inward", "Δmax inward", "mean inward", "max inward"]),
        "",
        stage_conclusion,
        "",
        "## 结论",
        "",
        f"1. {iter_conclusion}",
        f"2. {projection_conclusion}",
        f"3. {stage_conclusion}",
        "",
        "详细逐阶段数值见各组 `analysis/stage_metrics.csv`；逐轮数值见 `analysis/iteration_metrics.csv`。`vertex CD proxy` 是顶点采样的快速代理指标，不等同于 summary 中的 area-sampled CD。",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="run the four CLI experiments before analysis")
    args = parser.parse_args()

    if args.run:
        run_experiments()

    input_vertices, input_faces = read_mesh(INPUT_MESH)
    input_normals = vertex_normals(input_vertices, input_faces)
    roi_indices, component_rows = build_roi(input_vertices, input_faces)

    analyses = {}
    for group in GROUPS:
        analyses[group.name] = analyze_group(group, roi_indices, input_vertices, input_normals)

    write_report(component_rows, analyses)
    verify_outputs(analyses)
    print(f"wrote {REPORT_PATH}")
    print(f"wrote {RESULT_ROOT}")


if __name__ == "__main__":
    main()
