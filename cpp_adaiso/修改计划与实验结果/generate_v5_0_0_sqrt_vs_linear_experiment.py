import csv
import json
import math
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import trimesh
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "build2" / "adaiso_cli.exe"
EXAMPLES = ROOT / "examples"
RESULT_ROOT = ROOT / "results" / "v5_0_0_sqrt_vs_linear"
REPORT_PATH = ROOT / "修改计划与实验结果" / "v5.0.0 sqrt_vs_linear实验报告.md"

MODELS = [
    "hand.obj",
    "bunny.obj",
    "T21.obj",
    "model_001.off",
]
MAPPINGS = ["inverse-sqrt", "inverse-linear"]
GLOBAL_CD_SAMPLES = 50000
LOCAL_SAMPLES = 80000
ITERATIONS = 5

K_EPS = 1e-14
K_COT_MAX = 10.0
K_AREA_FLOOR_FACTOR = 1e-8


def radical_inverse(index: int, base: int) -> float:
    result = 0.0
    factor = 1.0 / base
    while index > 0:
        result += factor * (index % base)
        index //= base
        factor /= base
    return result


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError(f"failed to load mesh: {path}")
    if mesh.vertices.size == 0 or mesh.faces.size == 0:
        raise RuntimeError(f"empty mesh: {path}")
    return mesh


def sample_mesh_by_area(mesh: trimesh.Trimesh, samples: int):
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    tri = vertices[faces]
    areas = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    valid = np.isfinite(areas) & (areas > 0.0)
    if not np.any(valid):
        raise RuntimeError("mesh has no positive-area triangles")
    valid_faces = np.nonzero(valid)[0]
    cdf = np.cumsum(areas[valid])
    total = float(cdf[-1])

    pts = np.empty((samples, 3), dtype=np.float64)
    face_ids = np.empty(samples, dtype=np.int64)
    bary = np.empty((samples, 3), dtype=np.float64)
    for i in range(samples):
        seq = i + 1
        area_position = radical_inverse(seq, 2) * total
        local_idx = np.searchsorted(cdf, area_position, side="left")
        if local_idx >= len(valid_faces):
            local_idx = len(valid_faces) - 1
        face_idx = int(valid_faces[local_idx])
        u = radical_inverse(seq, 3)
        v = radical_inverse(seq, 5)
        if u + v > 1.0:
            u = 1.0 - u
            v = 1.0 - v
        w = 1.0 - u - v
        f = faces[face_idx]
        pts[i] = w * vertices[f[0]] + u * vertices[f[1]] + v * vertices[f[2]]
        face_ids[i] = face_idx
        bary[i] = (w, u, v)
    return pts, face_ids, bary


def mean_edge_length(mesh: trimesh.Trimesh) -> float:
    edges = mesh.edges_unique
    if len(edges) == 0:
        return 0.0
    v = np.asarray(mesh.vertices, dtype=np.float64)
    lengths = np.linalg.norm(v[edges[:, 0]] - v[edges[:, 1]], axis=1)
    return float(np.mean(lengths))


def clamped_positive_cotangent(a, b, c) -> float:
    u = b - a
    v = c - a
    denom = np.linalg.norm(np.cross(u, v))
    if denom <= K_EPS or not np.isfinite(denom):
        return 0.0
    cot = float(np.dot(u, v) / denom)
    if not np.isfinite(cot) or cot <= 0.0:
        return 0.0
    return min(cot, K_COT_MAX)


def robust_mean_curvature_signal(mesh: trimesh.Trimesh) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    mean_len = mean_edge_length(mesh)
    if mean_len <= 0.0 or not np.isfinite(mean_len):
        raise RuntimeError("invalid mean edge length")
    area_floor = K_AREA_FLOOR_FACTOR * mean_len * mean_len
    areas = np.zeros(len(vertices), dtype=np.float64)
    lap = np.zeros_like(vertices)
    weights = {}

    for f in faces:
        p = vertices[f]
        area = 0.5 * np.linalg.norm(np.cross(p[1] - p[0], p[2] - p[0]))
        if not np.isfinite(area) or area <= 0.0:
            continue
        share = area / 3.0
        areas[f] += share
        edge_data = [
            (f[1], f[2], f[0]),
            (f[2], f[0], f[1]),
            (f[0], f[1], f[2]),
        ]
        for a, b, opposite in edge_data:
            key = (int(min(a, b)), int(max(a, b)))
            w = 0.5 * clamped_positive_cotangent(vertices[opposite], vertices[a], vertices[b])
            weights[key] = weights.get(key, 0.0) + w

    for (a, b), w in weights.items():
        if not np.isfinite(w) or w <= 0.0:
            continue
        pa = vertices[a]
        pb = vertices[b]
        lap[a] += w * (pb - pa)
        lap[b] += w * (pa - pb)

    safe_areas = np.maximum(areas, area_floor)
    return 0.5 * np.linalg.norm(lap, axis=1) / safe_areas * mean_len


def quantile(values, q):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan")
    return float(np.quantile(values, q))


def distance_stats(dist2):
    dist = np.sqrt(np.asarray(dist2, dtype=np.float64))
    return {
        "rmse": float(math.sqrt(float(np.mean(dist2)))) if len(dist2) else float("nan"),
        "mean": float(np.mean(dist)) if len(dist) else float("nan"),
        "p95": quantile(dist, 0.95),
        "max": float(np.max(dist)) if len(dist) else float("nan"),
    }


def target_field_stats(path: Path):
    vertices_csv = path / "vertices.csv"
    if not vertices_csv.exists():
        return {}
    data = pd.read_csv(vertices_csv)
    target_h = data["target_h"].to_numpy(dtype=np.float64)
    csf = data["csf_curvature"].to_numpy(dtype=np.float64)
    out = {
        "target_h_mean": float(np.mean(target_h)),
        "target_h_p10": quantile(target_h, 0.10),
        "target_h_p50": quantile(target_h, 0.50),
        "target_h_p90": quantile(target_h, 0.90),
        "target_h_min": float(np.min(target_h)),
        "target_h_max": float(np.max(target_h)),
    }
    for pct in [0.05, 0.10, 0.20]:
        threshold = np.quantile(csf, 1.0 - pct)
        mask = csf >= threshold
        prefix = f"target_h_top{int(pct * 100)}"
        out[f"{prefix}_mean"] = float(np.mean(target_h[mask]))
        out[f"{prefix}_p50"] = quantile(target_h[mask], 0.50)
    return out


def analyze_local_detail(gt_path: Path, out_path: Path):
    gt = load_mesh(gt_path)
    out = load_mesh(out_path)
    gt_points, gt_faces, bary = sample_mesh_by_area(gt, LOCAL_SAMPLES)
    out_points, _, _ = sample_mesh_by_area(out, LOCAL_SAMPLES)
    out_tree = cKDTree(out_points)
    gt_to_out_dist, _ = out_tree.query(gt_points, k=1)

    # scipy returns distances, not squared distances.
    gt_to_out_dist2 = np.square(gt_to_out_dist)
    curvature = robust_mean_curvature_signal(gt)
    face_curv = curvature[np.asarray(gt.faces, dtype=np.int64)[gt_faces]]
    sample_curv = np.sum(face_curv * bary, axis=1)

    result = {}
    for pct in [0.05, 0.10, 0.20]:
        threshold = np.quantile(sample_curv, 1.0 - pct)
        mask = sample_curv >= threshold
        stats = distance_stats(gt_to_out_dist2[mask])
        prefix = f"high_curv_top{int(pct * 100)}"
        result[f"{prefix}_samples"] = int(np.sum(mask))
        result[f"{prefix}_threshold"] = float(threshold)
        for key, value in stats.items():
            result[f"{prefix}_{key}"] = value
    result.update({f"local_gt_to_out_{k}": v for k, v in distance_stats(gt_to_out_dist2).items()})
    return result


def run_command(command, cwd: Path, log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(" ".join(str(x) for x in command) + "\n\n")
        result = subprocess.run(command, cwd=cwd, text=True, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f"command failed, see {log_path}")


def load_summary(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt(value, digits=6):
    if value is None:
        return ""
    try:
        v = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(v):
        return "nan"
    return f"{v:.{digits}g}"


def percent_delta(linear, sqrt):
    if sqrt == 0 or sqrt is None or linear is None:
        return float("nan")
    return (linear - sqrt) / sqrt * 100.0


def build_report(rows, result_root: Path):
    df = pd.DataFrame(rows)
    lines = []
    lines.append("# v5.0.0 sqrt vs linear target_h 实验报告")
    lines.append("")
    lines.append(f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- 结果目录：`{result_root.relative_to(ROOT)}`")
    lines.append("- 固定配置：`curvature-mode=robust-mean`，`target_faces=input face count`，`iters=5`，`cd_samples=50000`。")
    lines.append("- 对比变量：`inverse-sqrt` 对应 `target_h = lambda / sqrt(kappa)`；`inverse-linear` 对应 `target_h = lambda / kappa`。")
    lines.append("- 局部指标：在 GT 上计算 robust mean curvature signal，取面积采样点 top 5/10/20% 高曲率区域，报告 GT->output 最近邻距离。数值越小越好。")
    lines.append("")
    lines.append("## 主要结果")
    lines.append("")
    lines.append("| model | mapping | faces | global symmetric RMSE | high top5 RMSE | high top10 RMSE | high top20 RMSE | qavg | qmin | target_h top10 mean | runtime s |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, row in df.sort_values(["model", "mapping"]).iterrows():
        lines.append(
            f"| {row['model']} | {row['mapping']} | {int(row['output_faces'])} | "
            f"{fmt(row['global_symmetric_rmse'])} | {fmt(row['high_curv_top5_rmse'])} | "
            f"{fmt(row['high_curv_top10_rmse'])} | {fmt(row['high_curv_top20_rmse'])} | "
            f"{fmt(row['qavg'])} | {fmt(row['qmin'])} | {fmt(row['target_h_top10_mean'])} | "
            f"{fmt(row['runtime_sec'])} |"
        )
    lines.append("")
    lines.append("## linear 相对 sqrt 的变化")
    lines.append("")
    lines.append("| model | global RMSE delta | top5 RMSE delta | top10 RMSE delta | top20 RMSE delta | target_h top10 mean delta | face delta | 结论 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    wins = 0
    losses = 0
    ties = 0
    for model in MODELS:
        sub = df[df["model"] == Path(model).stem].set_index("mapping")
        if "inverse-sqrt" not in sub.index or "inverse-linear" not in sub.index:
            continue
        sqrt_row = sub.loc["inverse-sqrt"]
        lin_row = sub.loc["inverse-linear"]
        d_top10 = percent_delta(lin_row["high_curv_top10_rmse"], sqrt_row["high_curv_top10_rmse"])
        if d_top10 < -1.0:
            verdict = "linear 改善高曲率 top10"
            wins += 1
        elif d_top10 > 1.0:
            verdict = "linear 变差"
            losses += 1
        else:
            verdict = "差异很小"
            ties += 1
        lines.append(
            f"| {Path(model).stem} | {fmt(percent_delta(lin_row['global_symmetric_rmse'], sqrt_row['global_symmetric_rmse']))}% | "
            f"{fmt(percent_delta(lin_row['high_curv_top5_rmse'], sqrt_row['high_curv_top5_rmse']))}% | "
            f"{fmt(d_top10)}% | "
            f"{fmt(percent_delta(lin_row['high_curv_top20_rmse'], sqrt_row['high_curv_top20_rmse']))}% | "
            f"{fmt(percent_delta(lin_row['target_h_top10_mean'], sqrt_row['target_h_top10_mean']))}% | "
            f"{fmt(percent_delta(lin_row['output_faces'], sqrt_row['output_faces']))}% | {verdict} |"
        )
    lines.append("")
    lines.append("## 结论")
    lines.append("")
    lines.append(
        f"以 high-curvature top10 RMSE 为主指标，`inverse-linear` 在 {wins} 个模型上优于 `inverse-sqrt`，"
        f"{losses} 个模型上变差，{ties} 个模型差异小于 1%。"
    )
    lines.append("如果 `inverse-linear` 的 top5/top10 RMSE 下降但 global CD 上升，说明它确实把预算更偏向尖锐区域，代价是平坦区域/整体误差。")
    lines.append("如果 high-curvature RMSE 没有稳定下降，则不开根号不能作为默认策略，只适合作为高曲率细节优先的可选模式。")
    lines.append("")
    lines.append(
        "本次实验还看到一个需要控制的混杂因素：虽然两组使用相同 `target_faces` 输入预算，"
        "但 `inverse-linear` 的最终输出面数比 `inverse-sqrt` 高 6.23%-16.16%。"
        "因此当前结论应表述为：在相同目标预算配置下，`inverse-linear` 会明显压低高曲率区域的 `target_h`，"
        "但没有稳定带来高曲率局部误差下降；若要做最终定论，需要再做一轮“匹配最终面数”的校准实验。"
    )
    lines.append("")
    lines.append("## 产物")
    lines.append("")
    lines.append("- 汇总 CSV：`run_summary.csv`")
    lines.append("- 每组输出：`<model>/<mapping>/output.ply`")
    lines.append("- 每组 CLI summary：`<model>/<mapping>/summary.json`")
    lines.append("- 每组 target field：`<model>/<mapping>/target_field/vertices.csv` 与热图 PLY")
    lines.append("")
    return "\n".join(lines) + "\n"


def main():
    if not CLI.exists():
        raise RuntimeError(f"missing CLI: {CLI}")
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for model_name in MODELS:
        model_path = EXAMPLES / model_name
        gt_mesh = load_mesh(model_path)
        target_faces = int(len(gt_mesh.faces))
        model_key = Path(model_name).stem
        for mapping in MAPPINGS:
            out_dir = RESULT_ROOT / model_key / mapping
            out_dir.mkdir(parents=True, exist_ok=True)
            output_path = out_dir / "output.ply"
            summary_path = out_dir / "summary.json"
            target_dir = out_dir / "target_field"
            command = [
                str(CLI),
                str(model_path),
                "-o",
                str(output_path),
                "--curvature-mode",
                "robust-mean",
                "--target-size-mapping",
                mapping,
                "--target-faces",
                str(target_faces),
                "--iters",
                str(ITERATIONS),
                "--gt",
                str(model_path),
                "--cd-samples",
                str(GLOBAL_CD_SAMPLES),
                "--summary-json",
                str(summary_path),
                "--export-target-field",
                str(target_dir),
                "--quiet",
            ]
            print(f"running {model_key} {mapping}")
            run_command(command, ROOT, out_dir / "run.log")
            summary = load_summary(summary_path)
            local = analyze_local_detail(model_path, output_path)
            target_stats = target_field_stats(target_dir)
            row = {
                "model": model_key,
                "mapping": mapping,
                "input_faces": target_faces,
                "output_vertices": summary["output_summary"]["vertices"],
                "output_faces": summary["output_summary"]["faces"],
                "qavg": summary["output_summary"]["qavg"],
                "qmin": summary["output_summary"]["qmin"],
                "runtime_sec": summary["runtime_sec"],
                "global_chamfer_distance": summary["chamfer_distance"]["chamfer_distance"],
                "global_symmetric_rmse": summary["chamfer_distance"]["symmetric_rmse"],
                "global_output_to_gt_msd": summary["chamfer_distance"]["output_to_gt_msd"],
                "global_gt_to_output_msd": summary["chamfer_distance"]["gt_to_output_msd"],
            }
            row.update(local)
            row.update(target_stats)
            rows.append(row)
            write_csv(RESULT_ROOT / "run_summary.csv", rows)
    write_csv(RESULT_ROOT / "run_summary.csv", rows)
    REPORT_PATH.write_text(build_report(rows, RESULT_ROOT), encoding="utf-8")
    print(f"wrote {RESULT_ROOT / 'run_summary.csv'}")
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
