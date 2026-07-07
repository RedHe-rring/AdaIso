from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = ROOT / "cpp_adaiso" / "results" / "v2.8.0_hmin_cd_T21_T30"
REPORT_PATH = ROOT / "cpp_adaiso" / "修改计划与实验结果" / "v2.8.0 hmin-CD实验报告.md"
RUN_SUMMARY_PATH = RESULT_ROOT / "run_summary.csv"

MODELS = ("T21", "T30")
HMIN_GROUPS = (
    (0.60, "hmin060"),
    (0.45, "hmin045"),
    (0.30, "hmin030"),
    (0.20, "hmin020"),
)

NATIVE_TARGET_FIELD_FILES = (
    "raw_curvature_heat_shared.ply",
    "csf_curvature_heat_shared.ply",
    "raw_curvature_heat_independent.ply",
    "csf_curvature_heat_independent.ply",
    "target_h_heat.ply",
    "vertices.csv",
    "edges.csv",
    "faces.csv",
    "field_stats.json",
    "color_scales.json",
)

CSV_FIELDS = (
    "model",
    "hmin_factor",
    "h_min",
    "h_max",
    "observed_h_min",
    "observed_h_max",
    "output_vertices",
    "output_faces",
    "qavg",
    "qmin",
    "violation_rate",
    "chamfer_distance",
    "symmetric_rmse",
    "runtime_sec",
    "max_edge_target_ratio",
    "p95_edge_target_ratio",
    "max_grad_h_norm",
    "p95_grad_h_norm",
    "delta_cd_vs_hmin060",
    "delta_cd_pct_vs_hmin060",
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def format_float(value: float, digits: int = 10) -> str:
    return f"{value:.{digits}g}"


def format_sci(value: float) -> str:
    return f"{value:.6E}"


def label_for_delta(delta_pct: float) -> str:
    if delta_pct < -5.0:
        return "明显改善"
    if delta_pct > 5.0:
        return "变差"
    return "变化不显著"


def color_byte(value: float) -> int:
    return int(round(max(0.0, min(1.0, value)) * 255.0))


def lerp_color(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    return (
        color_byte((1.0 - t) * a[0] / 255.0 + t * b[0] / 255.0),
        color_byte((1.0 - t) * a[1] / 255.0 + t * b[1] / 255.0),
        color_byte((1.0 - t) * a[2] / 255.0 + t * b[2] / 255.0),
    )


def heat_color(t: float) -> tuple[int, int, int]:
    palette = (
        (45, 70, 180),
        (0, 170, 210),
        (80, 190, 80),
        (245, 210, 70),
        (210, 40, 40),
    )
    t = max(0.0, min(1.0, t))
    scaled = t * (len(palette) - 1)
    index = min(int(scaled), len(palette) - 2)
    return lerp_color(palette[index], palette[index + 1], scaled - index)


def read_vertices_csv(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, float]] = []
        for row in reader:
            rows.append(
                {
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "z": float(row["z"]),
                    "raw_curvature": float(row["raw_curvature"]),
                    "csf_curvature": float(row["csf_curvature"]),
                    "target_h": float(row["target_h"]),
                }
            )
        return rows


def read_face_lines_from_ply(path: Path) -> tuple[int, list[str]]:
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
        raise RuntimeError(f"invalid PLY header: {path}")
    first_face = end_header + 1 + vertex_count
    face_lines = lines[first_face : first_face + face_count]
    if len(face_lines) != face_count:
        raise RuntimeError(f"invalid PLY face count: {path}")
    return face_count, face_lines


def write_common_target_h_ply(
    target_field_dir: Path,
    vertices: list[dict[str, float]],
    common_min: float,
    common_max: float,
) -> None:
    face_count, face_lines = read_face_lines_from_ply(target_field_dir / "target_h_heat.ply")
    ply_path = target_field_dir / "target_h_heat_common.ply"
    with ply_path.open("w", encoding="utf-8", newline="\n") as out:
        out.write("ply\n")
        out.write("format ascii 1.0\n")
        out.write(f"element vertex {len(vertices)}\n")
        out.write("property float x\n")
        out.write("property float y\n")
        out.write("property float z\n")
        out.write("property uchar red\n")
        out.write("property uchar green\n")
        out.write("property uchar blue\n")
        out.write(f"element face {face_count}\n")
        out.write("property list uchar int vertex_indices\n")
        out.write("end_header\n")
        denom = common_max - common_min
        for vertex in vertices:
            normalized = 0.5 if denom <= 0.0 else (vertex["target_h"] - common_min) / denom
            t = 1.0 - max(0.0, min(1.0, normalized))
            r, g, b = heat_color(t)
            out.write(
                f"{vertex['x']:.17g} {vertex['y']:.17g} {vertex['z']:.17g} "
                f"{r} {g} {b}\n"
            )
        for line in face_lines:
            out.write(line + "\n")

    scale = {
        "palette": "heat",
        "normalization": "linear_clamp",
        "scope": "per_model_hmin_sweep_common_target_h",
        "scale": {
            "file": "target_h_heat_common.ply",
            "field": "target_h",
            "min": common_min,
            "max": common_max,
            "inverted": True,
        },
    }
    (target_field_dir / "common_color_scale.json").write_text(
        json.dumps(scale, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def validate_group_files(target_field_dir: Path) -> list[str]:
    missing = []
    for name in NATIVE_TARGET_FIELD_FILES:
        if not (target_field_dir / name).exists():
            missing.append(name)
    return missing


def build_rows_and_heatmaps() -> tuple[list[dict[str, object]], dict[str, dict[str, float]]]:
    rows: list[dict[str, object]] = []
    curvature_checks: dict[str, dict[str, float]] = {}

    for model in MODELS:
        baseline_cd: float | None = None
        model_entries = []
        baseline_vertices: list[dict[str, float]] | None = None
        max_raw_diff = 0.0
        max_csf_diff = 0.0

        for hmin_factor, label in HMIN_GROUPS:
            group_dir = RESULT_ROOT / model / label
            target_field_dir = group_dir / "target_field"
            summary_path = group_dir / "summary.json"
            field_stats_path = target_field_dir / "field_stats.json"
            missing = validate_group_files(target_field_dir)
            if missing:
                raise RuntimeError(f"missing target field files in {target_field_dir}: {missing}")
            if not summary_path.exists():
                raise RuntimeError(f"missing summary: {summary_path}")

            summary = load_json(summary_path)
            if not summary["stats"].get("csf_solver_success", False):
                raise RuntimeError(f"CSF solve failed: {summary_path}")
            actual_hmin = float(summary["parameters"]["target_min_factor"])
            if abs(actual_hmin - hmin_factor) > 1e-12:
                raise RuntimeError(
                    f"target_min_factor mismatch in {summary_path}: "
                    f"expected {hmin_factor}, got {actual_hmin}"
                )
            field_stats = load_json(field_stats_path)
            vertices = read_vertices_csv(target_field_dir / "vertices.csv")
            if baseline_vertices is None:
                baseline_vertices = vertices
            else:
                if len(vertices) != len(baseline_vertices):
                    raise RuntimeError(f"vertex count mismatch for curvature check: {model} {label}")
                for a, b in zip(baseline_vertices, vertices):
                    max_raw_diff = max(max_raw_diff, abs(a["raw_curvature"] - b["raw_curvature"]))
                    max_csf_diff = max(max_csf_diff, abs(a["csf_curvature"] - b["csf_curvature"]))

            common_min = float(field_stats["base_length"]) * 0.20
            common_max = float(field_stats["base_length"]) * 1.80
            write_common_target_h_ply(target_field_dir, vertices, common_min, common_max)

            output_summary = summary["output_summary"]
            last_iter = summary["iterations"][-1]
            chamfer = summary["chamfer_distance"]
            cd = float(chamfer["chamfer_distance"])
            if label == "hmin060":
                baseline_cd = cd
            model_entries.append(
                {
                    "model": model,
                    "hmin_factor": hmin_factor,
                    "h_min": float(field_stats["h_min"]),
                    "h_max": float(field_stats["h_max"]),
                    "observed_h_min": float(field_stats["observed_h_min"]),
                    "observed_h_max": float(field_stats["observed_h_max"]),
                    "output_vertices": int(output_summary["vertices"]),
                    "output_faces": int(output_summary["faces"]),
                    "qavg": float(output_summary["qavg"]),
                    "qmin": float(output_summary["qmin"]),
                    "violation_rate": float(last_iter["violation_rate"]),
                    "chamfer_distance": cd,
                    "symmetric_rmse": float(chamfer["symmetric_rmse"]),
                    "runtime_sec": float(summary["runtime_sec"]),
                    "max_edge_target_ratio": float(field_stats["max_edge_target_ratio"]),
                    "p95_edge_target_ratio": float(field_stats["p95_edge_target_ratio"]),
                    "max_grad_h_norm": float(field_stats["max_grad_h_norm"]),
                    "p95_grad_h_norm": float(field_stats["p95_grad_h_norm"]),
                    "group_label": label,
                }
            )

        if baseline_cd is None:
            raise RuntimeError(f"missing hmin060 baseline for {model}")
        for entry in model_entries:
            delta = float(entry["chamfer_distance"]) - baseline_cd
            entry["delta_cd_vs_hmin060"] = delta
            entry["delta_cd_pct_vs_hmin060"] = 100.0 * delta / baseline_cd
            rows.append(entry)
        curvature_checks[model] = {
            "max_raw_curvature_abs_diff": max_raw_diff,
            "max_csf_curvature_abs_diff": max_csf_diff,
        }

    return rows, curvature_checks


def write_run_summary(rows: list[dict[str, object]]) -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    with RUN_SUMMARY_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in CSV_FIELDS})


def markdown_table_for_model(rows: list[dict[str, object]], model: str) -> str:
    lines = [
        "| hmin | Out V/F | Qavg | Qmin | viol | CD | RMSE | ΔCD% | max edge ratio | max grad | 判定 | 风险 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in [item for item in rows if item["model"] == model]:
        delta_pct = float(row["delta_cd_pct_vs_hmin060"])
        decision = "基线" if row["group_label"] == "hmin060" else label_for_delta(delta_pct)
        improved = delta_pct < -5.0
        risky = improved and (
            float(row["qmin"]) < 0.10
            or float(row["violation_rate"]) > 0.06
            or float(row["max_edge_target_ratio"]) > 2.5
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{float(row['hmin_factor']):.2f}",
                    f"{row['output_vertices']}/{row['output_faces']}",
                    f"{float(row['qavg']):.6f}",
                    f"{float(row['qmin']):.6f}",
                    f"{float(row['violation_rate']):.6f}",
                    format_sci(float(row["chamfer_distance"])),
                    format_sci(float(row["symmetric_rmse"])),
                    f"{delta_pct:.2f}",
                    f"{float(row['max_edge_target_ratio']):.6f}",
                    f"{float(row['max_grad_h_norm']):.6f}",
                    decision,
                    "高" if risky else "-",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def model_conclusion(rows: list[dict[str, object]], model: str) -> str:
    model_rows = [row for row in rows if row["model"] == model and row["group_label"] != "hmin060"]
    improved_rows = [row for row in model_rows if float(row["delta_cd_pct_vs_hmin060"]) < -5.0]
    worsened_rows = [row for row in model_rows if float(row["delta_cd_pct_vs_hmin060"]) > 5.0]
    if improved_rows:
        best = min(model_rows, key=lambda row: float(row["chamfer_distance"]))
        return (
            f"{model}: 降低 hmin 至 {float(best['hmin_factor']):.2f} 时 CD 最低，"
            f"相对 hmin=0.60 变化 {float(best['delta_cd_pct_vs_hmin060']):.2f}%。"
        )
    if worsened_rows and len(worsened_rows) == len(model_rows):
        return f"{model}: 所有更低 hmin 组的 CD 都变差，降低 hmin 不支持该模型的 CD 改善。"
    return f"{model}: 更低 hmin 组的 CD 变化均未超过 ±5%，不能判定为明显改善。"


def verification_counts() -> dict[str, int]:
    counts = {
        "summary": 0,
        "csf_success": 0,
        "target_min_match": 0,
        "native_complete": 0,
        "common_ply": 0,
        "common_color_scale": 0,
        "total": len(MODELS) * len(HMIN_GROUPS),
    }
    for model in MODELS:
        for hmin_factor, label in HMIN_GROUPS:
            group_dir = RESULT_ROOT / model / label
            target_field_dir = group_dir / "target_field"
            summary_path = group_dir / "summary.json"
            if summary_path.exists():
                counts["summary"] += 1
                summary = load_json(summary_path)
                if summary["stats"].get("csf_solver_success", False):
                    counts["csf_success"] += 1
                if abs(float(summary["parameters"]["target_min_factor"]) - hmin_factor) <= 1e-12:
                    counts["target_min_match"] += 1
            if not validate_group_files(target_field_dir):
                counts["native_complete"] += 1
            if (target_field_dir / "target_h_heat_common.ply").exists():
                counts["common_ply"] += 1
            if (target_field_dir / "common_color_scale.json").exists():
                counts["common_color_scale"] += 1
    return counts


def write_report(rows: list[dict[str, object]], curvature_checks: dict[str, dict[str, float]]) -> None:
    improved_models = []
    for model in MODELS:
        if any(
            row["model"] == model
            and row["group_label"] != "hmin060"
            and float(row["delta_cd_pct_vs_hmin060"]) < -5.0
            for row in rows
        ):
            improved_models.append(model)

    if len(improved_models) == len(MODELS):
        overall = "T21 和 T30 都至少有一个更低 hmin 组 CD 明显下降，因此降低 hmin 有助于 CD，但需要同时检查质量代价。"
    elif improved_models:
        overall = "只有部分模型在更低 hmin 下 CD 明显下降，因此 hmin 对 CD 的收益是模型相关的。"
    else:
        overall = "T21 和 T30 在更低 hmin 下都没有超过 5% 的 CD 改善，因此本实验不支持降低 hmin 会稳定改善 CD。"

    lines: list[str] = [
        "# v2.8.0 hmin-CD 实验报告：T21/T30",
        "",
        "## 实验目的",
        "",
        "本实验用于判断 v2.8.0 饱和尺寸场中降低 `hmin`，即降低 `--target-min-factor`，是否会让 T21 和 T30 的 Chamfer Distance 下降。本文把“CD 提高”定义为几何精度提高，即 `chamfer_distance` 下降；表格同时保留 CD 数值变化方向。",
        "",
        "只改变 `target_min_factor`，其余参数固定：`iters=10`、`preprocess=0`、`project=1`、`curvature-mode=robust-mean`、`csf-alpha=2.0`、`target-max-factor=1.8`、`target-size-gamma=2.0`、`cd-samples=100000`。",
        "",
        f"实验输出目录：`{rel(RESULT_ROOT)}/`",
        "",
        f"汇总 CSV：`{rel(RUN_SUMMARY_PATH)}`",
        "",
        "## 判定标准",
        "",
        "- 以每个模型的 `hmin=0.60` 作为基线。",
        "- `chamfer_distance` 下降超过 5% 判定为明显改善。",
        "- `chamfer_distance` 变化在 ±5% 内判定为变化不显著。",
        "- `chamfer_distance` 上升超过 5% 判定为变差。",
        "- 若改善伴随 `qmin < 0.10`、`violation_rate > 0.06` 或 `max_edge_target_ratio > 2.5`，标记为高质量风险。",
        "",
        "## T21 结果",
        "",
        markdown_table_for_model(rows, "T21"),
        "",
        model_conclusion(rows, "T21"),
        "",
        "## T30 结果",
        "",
        markdown_table_for_model(rows, "T30"),
        "",
        model_conclusion(rows, "T30"),
        "",
        "## 可视化输出",
        "",
        "每组都保留原生尺寸场与曲率场诊断文件。额外生成的 `target_h_heat_common.ply` 使用同一模型内共同色标：`min = base_length * 0.20`，`max = base_length * 1.80`，并保持 inverted heat 显示。",
        "",
    ]

    for model in MODELS:
        lines.append(f"### {model}")
        lines.append("")
        lines.append("| hmin | common-scale target_h | raw curvature | csf curvature |")
        lines.append("|---:|---|---|---|")
        for hmin_factor, label in HMIN_GROUPS:
            target_field_dir = RESULT_ROOT / model / label / "target_field"
            lines.append(
                f"| {hmin_factor:.2f} | "
                f"`{rel(target_field_dir / 'target_h_heat_common.ply')}` | "
                f"`{rel(target_field_dir / 'raw_curvature_heat_shared.ply')}` | "
                f"`{rel(target_field_dir / 'csf_curvature_heat_shared.ply')}` |"
            )
        lines.append("")

    checks = verification_counts()
    total = checks["total"]
    lines.extend(
        [
            "## 完整性检查",
            "",
            "| 检查项 | 结果 |",
            "|---|---:|",
            f"| summary.json 数量 | {checks['summary']} / {total} |",
            f"| csf_solver_success=true | {checks['csf_success']} / {total} |",
            f"| target_min_factor 参数匹配 | {checks['target_min_match']} / {total} |",
            f"| 原生 target_field 诊断完整 | {checks['native_complete']} / {total} |",
            f"| common-scale PLY 数量 | {checks['common_ply']} / {total} |",
            f"| common_color_scale.json 数量 | {checks['common_color_scale']} / {total} |",
            "",
            "曲率场一致性检查：",
            "",
            "| Model | max raw curvature diff | max CSF curvature diff | 结论 |",
            "|---|---:|---:|---|",
        ]
    )
    for model in MODELS:
        raw_diff = curvature_checks[model]["max_raw_curvature_abs_diff"]
        csf_diff = curvature_checks[model]["max_csf_curvature_abs_diff"]
        ok = raw_diff <= 1e-12 and csf_diff <= 1e-12
        lines.append(
            f"| {model} | {raw_diff:.3E} | {csf_diff:.3E} | {'通过' if ok else '异常'} |"
        )

    lines.extend(
        [
            "",
            "## 总体结论",
            "",
            overall,
            "",
            "实验同时显示，降低 hmin 会增加输出顶点/面数，并可能推高尺寸场相邻边比例或降低 Qmin。因此即使 CD 下降，也不能单独作为默认参数选择依据，需要结合质量指标和局部可视化判断。",
            "",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows, curvature_checks = build_rows_and_heatmaps()
    write_run_summary(rows)
    write_report(rows, curvature_checks)
    print(f"wrote {RUN_SUMMARY_PATH}")
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
