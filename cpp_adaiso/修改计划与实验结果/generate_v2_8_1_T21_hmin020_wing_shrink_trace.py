from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from collections import deque
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[2]
CLI_PATH = ROOT / "cpp_adaiso" / "build" / "adaiso_cli.exe"
INPUT_MESH = ROOT / "cpp_adaiso" / "examples" / "T21.off"
RESULT_ROOT = ROOT / "cpp_adaiso" / "results" / "v2.8.1_T21_hmin020_wing_shrink_trace"
REPORT_PATH = ROOT / "cpp_adaiso" / "修改计划与实验结果" / "v2.8.1 T21 hmin020翅膀边缘收缩诊断报告.md"

GROUPS = (
    ("hmin020_project1", True),
    ("hmin020_project0", False),
)
STAGES = ("split", "collapse", "flip", "smooth")


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_experiments() -> None:
    if not CLI_PATH.exists():
        raise RuntimeError(f"missing CLI executable: {CLI_PATH}")
    if not INPUT_MESH.exists():
        raise RuntimeError(f"missing input mesh: {INPUT_MESH}")

    for group, project in GROUPS:
        group_dir = RESULT_ROOT / group
        group_dir.mkdir(parents=True, exist_ok=True)
        output_name = "final_project1.ply" if project else "final_project0.ply"
        command = [
            rel(CLI_PATH),
            rel(INPUT_MESH),
            "-o",
            rel(group_dir / output_name),
            "--iters",
            "10",
            "--preprocess",
            "0",
            "--project",
            "1" if project else "0",
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
        print(f"running {group}", flush=True)
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
            best_point = self.triangles[indices[i, 0], 0]
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


def build_roi(input_vertices: np.ndarray, input_faces: np.ndarray, group_dir: Path) -> tuple[np.ndarray, list[dict[str, object]]]:
    target_rows = load_target_vertices(group_dir / "target_field" / "vertices.csv")
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

    final_path = RESULT_ROOT / "hmin020_project1" / "final_project1.ply"
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
        "selection": "max_final_project1_mean_inward",
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


def snapshot_quality(path: Path) -> tuple[int, int, float, float, float]:
    vertices, faces = read_mesh(path)
    qavg, qmin = triangle_quality(vertices, faces)
    return len(vertices), len(faces), qavg, qmin, vertex_chamfer_proxy(vertices, INPUT_VERTICES)


def iteration_snapshots(group_dir: Path) -> list[tuple[str, Path]]:
    return [
        (f"iter{i:03d}", group_dir / "iteration_meshes" / f"iter{i:03d}.ply")
        for i in range(0, 11)
    ]


def stage_snapshots(group_dir: Path) -> list[tuple[str, Path]]:
    snapshots = [("iter000", group_dir / "iteration_meshes" / "iter000.ply")]
    for iteration in range(1, 11):
        for stage in STAGES:
            snapshots.append(
                (
                    f"iter{iteration:03d}_after_{stage}",
                    group_dir / "stage_meshes" / f"iter{iteration:03d}_after_{stage}.ply",
                )
            )
    return snapshots


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


def analyze_group(
    group: str,
    project: bool,
    roi_indices: np.ndarray,
    input_vertices: np.ndarray,
    input_normals: np.ndarray,
) -> dict[str, object]:
    group_dir = RESULT_ROOT / group
    summary = load_json(group_dir / "summary.json")
    iteration_by_index = {int(row["iteration"]): row for row in summary["iterations"]}
    final_path = group_dir / ("final_project1.ply" if project else "final_project0.ply")

    iteration_rows: list[dict[str, object]] = []
    for snapshot, path in iteration_snapshots(group_dir):
        if not path.exists():
            raise RuntimeError(f"missing iteration snapshot: {path}")
        vertices_count, faces_count, qavg, qmin, cd_proxy = snapshot_quality(path)
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
    for snapshot, path in stage_snapshots(group_dir):
        if not path.exists():
            raise RuntimeError(f"missing stage snapshot: {path}")
        vertices_count, faces_count, qavg, qmin, _ = snapshot_quality(path)
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
    final_vertices, final_faces, final_qavg, final_qmin, final_cd_proxy = snapshot_quality(final_path)
    final_row = {
        "snapshot": "final_project1" if project else "final_project0",
        "vertices": final_vertices,
        "faces": final_faces,
        "qavg": final_qavg,
        "qmin": final_qmin,
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
        "summary": summary,
        "iteration_rows": iteration_rows,
        "stage_rows": stage_rows,
        "final_row": final_row,
        "max_jump": max(stage_rows[1:], key=lambda row: float(row["delta_mean_inward"])),
    }


def count_existing(pattern: str, root: Path) -> int:
    return len(list(root.glob(pattern)))


def fmt(value: object, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}"


def sci(value: object) -> str:
    return f"{float(value):.6E}"


def table(rows: list[list[object]], headers: list[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def write_report(
    component_rows: list[dict[str, object]],
    analyses: dict[str, dict[str, object]],
) -> None:
    selected_meta = load_json(RESULT_ROOT / "roi" / "roi_selection.json")
    completeness_rows = []
    for group, _ in GROUPS:
        group_dir = RESULT_ROOT / group
        completeness_rows.append(
            [
                group,
                f"{count_existing('iter*.ply', group_dir / 'iteration_meshes')} / 11",
                f"{count_existing('iter*_after_*.ply', group_dir / 'stage_meshes')} / 40",
                "yes" if (group_dir / "summary.json").exists() else "no",
                "yes" if (group_dir / "target_field" / "vertices.csv").exists() else "no",
                "yes" if (group_dir / "analysis" / "stage_metrics.csv").exists() else "no",
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

    final_rows = []
    for group, _ in GROUPS:
        final_row = analyses[group]["final_row"]
        final_rows.append(
            [
                group,
                f"{final_row['vertices']}/{final_row['faces']}",
                fmt(final_row["qmin"]),
                sci(final_row["area_sampled_cd"]),
                sci(final_row["mean_inward"]),
                sci(final_row["p95_inward"]),
                sci(final_row["max_inward"]),
            ]
        )

    iteration_rows = []
    for row in analyses["hmin020_project1"]["iteration_rows"]:
        iteration_rows.append(
            [
                row["snapshot"],
                f"{row['vertices']}/{row['faces']}",
                fmt(row["qmin"]),
                fmt(row["violation_rate"]),
                row["split_count"],
                row["collapse_count"],
                row["flip_count"],
                sci(row["vertex_cd_proxy"]),
                sci(row["mean_inward"]),
                sci(row["max_inward"]),
            ]
        )

    jump_rows = []
    for group, _ in GROUPS:
        jump = analyses[group]["max_jump"]
        jump_rows.append(
            [
                group,
                jump["snapshot"],
                sci(jump["delta_mean_inward"]),
                sci(jump["delta_max_inward"]),
                sci(jump["mean_inward"]),
                sci(jump["max_inward"]),
            ]
        )

    final_project1 = analyses["hmin020_project1"]["final_row"]
    final_project0 = analyses["hmin020_project0"]["final_row"]
    mean_diff = float(final_project1["mean_inward"]) - float(final_project0["mean_inward"])
    if abs(mean_diff) <= max(1e-8, 0.05 * max(float(final_project0["mean_inward"]), 1e-12)):
        projection_conclusion = "project=0 和 project=1 的 ROI 内缩均值接近，最终 MLS projection 不是主要来源。"
    elif mean_diff > 0.0:
        projection_conclusion = "project=1 的 ROI 内缩更大，最终 MLS projection 可能放大了内缩。"
    else:
        projection_conclusion = "project=1 的 ROI 内缩更小，最终 MLS projection 没有解释内缩。"

    stage_name = str(analyses["hmin020_project1"]["max_jump"]["snapshot"])
    if "collapse" in stage_name:
        stage_conclusion = "最大跳变发生在 collapse 后，优先检查 collapse 是否跨薄边或特征边合并。"
    elif "smooth" in stage_name:
        stage_conclusion = "最大跳变发生在 smooth 后，优先检查 tangent smoothing 是否把边缘拉向局部重心。"
    elif "split" in stage_name:
        stage_conclusion = "最大跳变发生在 split 后，优先检查新增点和局部三角形重构。"
    elif "flip" in stage_name:
        stage_conclusion = "最大跳变发生在 flip 后，优先检查翻边对薄片边缘拓扑的影响。"
    else:
        stage_conclusion = "最大跳变阶段需要结合 stage_metrics.csv 进一步确认。"

    lines = [
        "# v2.8.1 T21 hmin020 翅膀边缘收缩诊断报告",
        "",
        "## 实验设置",
        "",
        "本实验固定 `target_min_factor=0.20`，保存 T21 从 `iter000` 到 `iter010` 的迭代快照，并额外保存每轮 `split/collapse/flip/smooth` 后的阶段快照，用于定位翅膀边缘向内收缩的来源。",
        "",
        f"- 输出目录：`{rel(RESULT_ROOT)}/`",
        "- 参数：`iters=10`、`preprocess=0`、`curvature-mode=robust-mean`、`csf-alpha=2.0`、`target-min-factor=0.20`、`target-max-factor=1.8`、`target-size-gamma=2.0`、`cd-samples=100000`。",
        "- 对照：`hmin020_project1` 与 `hmin020_project0`。",
        "",
        "## 完整性检查",
        "",
        table(completeness_rows, ["Group", "iteration meshes", "stage meshes", "summary", "target_field", "metrics"]),
        "",
        "## ROI 选择",
        "",
        "ROI 候选来自输入 T21 上 `target_h` 最小 10% 与 CSF 曲率最高 10% 的并集连通分量。默认选择 `final_project1` 中平均 inward displacement 最大的候选分量作为翅膀边缘 ROI。",
        "",
        f"- selected component id: `{selected_meta['selected_component_id']}`",
        f"- selected vertices: `{selected_meta['selected_vertex_count']}`",
        f"- candidate components: `{selected_meta['candidate_component_count']}`",
        f"- ROI 文件：`{rel(RESULT_ROOT / 'roi' / 'selected_wing_edge_roi.ply')}`",
        "",
        table(component_table_rows, ["component", "vertices", "mean inward", "p95 inward", "max inward", "rms dist"]),
        "",
        "## project 对照",
        "",
        table(final_rows, ["Group", "Out V/F", "Qmin", "area CD", "mean inward", "p95 inward", "max inward"]),
        "",
        projection_conclusion,
        "",
        "## 主实验逐轮指标",
        "",
        table(
            iteration_rows,
            ["snapshot", "V/F", "Qmin", "viol", "split", "collapse", "flip", "vertex CD proxy", "mean inward", "max inward"],
        ),
        "",
        "## 最大阶段跳变",
        "",
        table(jump_rows, ["Group", "snapshot", "Δmean inward", "Δmax inward", "mean inward", "max inward"]),
        "",
        stage_conclusion,
        "",
        "## 结论",
        "",
        f"{projection_conclusion} {stage_conclusion}",
        "",
        "详细逐阶段数值见各组 `analysis/stage_metrics.csv`；逐轮数值见 `analysis/iteration_metrics.csv`。`vertex CD proxy` 是顶点采样的快速代理指标，不等同于 summary 中的 area-sampled CD。",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def verify_outputs() -> None:
    for group, _ in GROUPS:
        group_dir = RESULT_ROOT / group
        iter_count = count_existing("iter*.ply", group_dir / "iteration_meshes")
        stage_count = count_existing("iter*_after_*.ply", group_dir / "stage_meshes")
        if iter_count != 11:
            raise RuntimeError(f"{group}: expected 11 iteration meshes, got {iter_count}")
        if stage_count != 40:
            raise RuntimeError(f"{group}: expected 40 stage meshes, got {stage_count}")
        for required in (
            group_dir / "summary.json",
            group_dir / "target_field" / "vertices.csv",
            group_dir / "analysis" / "iteration_metrics.csv",
            group_dir / "analysis" / "stage_metrics.csv",
            group_dir / "analysis" / "final_metrics.csv",
        ):
            if not required.exists():
                raise RuntimeError(f"missing required output: {required}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="run project0/project1 experiments before analysis")
    args = parser.parse_args()

    if args.run:
        run_experiments()

    global INPUT_VERTICES
    input_vertices, input_faces = read_mesh(INPUT_MESH)
    INPUT_VERTICES = input_vertices
    input_normals = vertex_normals(input_vertices, input_faces)
    roi_indices, component_rows = build_roi(input_vertices, input_faces, RESULT_ROOT / "hmin020_project1")

    analyses = {}
    for group, project in GROUPS:
        analyses[group] = analyze_group(group, project, roi_indices, input_vertices, input_normals)

    write_report(component_rows, analyses)
    verify_outputs()
    print(f"wrote {REPORT_PATH}")
    print(f"wrote {RESULT_ROOT}")


INPUT_VERTICES = np.empty((0, 3), dtype=np.float64)


if __name__ == "__main__":
    main()
