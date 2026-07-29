from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import mesh_metrics


SAMPLES_PER_DIRECTION = 100000
METHOD_OURS = "cpp_adaiso_v5.1.3"
METHOD_CWF10K = "CWF10k"
METHOD_CWF2K = "CWF2k"
METHOD_CGAL = "CGAL2019"

RAW_COLUMNS = [
    "method",
    "model",
    "source_model",
    "status",
    "input_mesh",
    "output_mesh",
    "gt_mesh",
    "output_vertices",
    "output_faces",
    "faces_out_in",
    "qavg",
    "qmin",
    "min_angle_deg",
    "avg_min_angle_deg",
    "max_angle_deg",
    "angle_lt_30_pct",
    "angle_lt_20_pct",
    "angle_lt_10_pct",
    "edge_mean",
    "edge_std",
    "edge_p05",
    "edge_p50",
    "edge_p95",
    "cd",
    "cd_bbox2",
    "symmetric_rmse",
    "symmetric_rmse_bbox_pct",
    "sampled_hausdorff",
    "sampled_hausdorff_bbox_pct",
    "runtime_sec",
    "violation",
    "error",
]


@dataclass
class EvalCase:
    method: str
    model: str
    source_model: str
    status: str
    input_mesh: Path | None
    output_mesh: Path | None
    gt_mesh: Path | None
    runtime_sec: float | None
    input_faces: int | None = None
    violation: float | None = None
    error: str = ""


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: csv_value(row.get(column)) for column in columns})


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return value


def to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def to_int(value: Any) -> int | None:
    number = to_float(value)
    return int(number) if number is not None else None


def path_or_none(value: Any, root: Path) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = root / path
    return path


def stem(value: str) -> str:
    return Path(value).stem


def find_gt(root: Path, model: str, preferred: Path | None = None) -> Path | None:
    if preferred is not None and preferred.exists():
        return preferred
    examples = root / "cpp_adaiso" / "examples"
    for ext in (".off", ".obj", ".ply"):
        candidate = examples / f"{model}{ext}"
        if candidate.exists():
            return candidate
    return None


def ours_cases(root: Path) -> list[EvalCase]:
    summary_path = root / "cpp_adaiso" / "results" / "v5.1.3_T_model_best_params_full" / "run_summary.csv"
    cases: list[EvalCase] = []
    for row in read_csv(summary_path):
        model = row["model"]
        case_path = path_or_none(row.get("case_path"), root)
        output_mesh = (case_path / "output.ply") if case_path is not None else None
        gt_mesh = find_gt(root, model)
        cases.append(
            EvalCase(
                method=METHOD_OURS,
                model=model,
                source_model=model,
                status=row.get("status", ""),
                input_mesh=gt_mesh,
                output_mesh=output_mesh,
                gt_mesh=gt_mesh,
                runtime_sec=to_float(row.get("runtime_sec")),
                input_faces=to_int(row.get("input_faces")),
                violation=to_float(row.get("violation")),
                error=row.get("error", ""),
            )
        )
    return cases


def cwf_cases(root: Path, method: str, rel_csv: str) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for row in read_csv(root / rel_csv):
        model = row["model"]
        input_mesh = path_or_none(row.get("input_file"), root)
        gt_mesh = find_gt(root, model, input_mesh)
        cases.append(
            EvalCase(
                method=method,
                model=model,
                source_model=model,
                status=row.get("status", ""),
                input_mesh=input_mesh,
                output_mesh=path_or_none(row.get("output_mesh"), root),
                gt_mesh=gt_mesh,
                runtime_sec=to_float(row.get("runtime_sec")),
                input_faces=to_int(row.get("gt_faces")),
                error=row.get("error", ""),
            )
        )
    return cases


def dedupe_cgal_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[stem(row["model"])].append(row)

    selected: list[dict[str, str]] = []
    for model, items in grouped.items():
        if len(items) == 1:
            selected.append(items[0])
            continue
        success_items = [item for item in items if item.get("status") == "success"]
        candidates = success_items or items
        off_candidates = [item for item in candidates if Path(item["model"]).suffix.lower() == ".off"]
        selected.append((off_candidates or candidates)[0])
    return selected


def cgal_cases(root: Path) -> tuple[list[EvalCase], Counter[str]]:
    summary_path = root / "compare" / "2019_CGALRemeshing" / "results" / "default_examples_preprocessed" / "summary.csv"
    raw_rows = read_csv(summary_path)
    raw_status_counts = Counter(row.get("status", "") for row in raw_rows)
    cases: list[EvalCase] = []
    for row in dedupe_cgal_rows(raw_rows):
        model = stem(row["model"])
        input_mesh = path_or_none(row.get("input"), root)
        gt_mesh = find_gt(root, model, input_mesh)
        cases.append(
            EvalCase(
                method=METHOD_CGAL,
                model=model,
                source_model=row["model"],
                status=row.get("status", ""),
                input_mesh=input_mesh,
                output_mesh=path_or_none(row.get("output"), root),
                gt_mesh=gt_mesh,
                runtime_sec=to_float(row.get("runtime_seconds")),
                input_faces=to_int(row.get("input_faces")),
                error=row.get("error", ""),
            )
        )
    return cases, raw_status_counts


def angle_summary(vertices: np.ndarray, faces: np.ndarray) -> dict[str, float]:
    angles = mesh_metrics.triangle_angles_degrees(vertices, faces)
    if len(angles) == 0:
        return {
            "min_angle_deg": math.nan,
            "avg_min_angle_deg": math.nan,
            "max_angle_deg": math.nan,
            "angle_lt_30_pct": math.nan,
            "angle_lt_20_pct": math.nan,
            "angle_lt_10_pct": math.nan,
        }
    min_angles = np.min(angles, axis=1)
    max_angles = np.max(angles, axis=1)
    return {
        "min_angle_deg": float(np.min(min_angles)),
        "avg_min_angle_deg": float(np.mean(min_angles)),
        "max_angle_deg": float(np.max(max_angles)),
        "angle_lt_30_pct": float(100.0 * np.mean(min_angles < 30.0)),
        "angle_lt_20_pct": float(100.0 * np.mean(min_angles < 20.0)),
        "angle_lt_10_pct": float(100.0 * np.mean(min_angles < 10.0)),
    }


def edge_summary(vertices: np.ndarray, faces: np.ndarray) -> dict[str, float]:
    edges = mesh_metrics.unique_edges(faces)
    if len(edges) == 0:
        return {"edge_mean": math.nan, "edge_std": math.nan, "edge_p05": math.nan, "edge_p50": math.nan, "edge_p95": math.nan}
    lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    return {
        "edge_mean": float(np.mean(lengths)),
        "edge_std": float(np.std(lengths)),
        "edge_p05": float(np.quantile(lengths, 0.05)),
        "edge_p50": float(np.quantile(lengths, 0.50)),
        "edge_p95": float(np.quantile(lengths, 0.95)),
    }


def empty_row(case: EvalCase, status: str | None = None, error: str | None = None) -> dict[str, Any]:
    return {
        "method": case.method,
        "model": case.model,
        "source_model": case.source_model,
        "status": status if status is not None else case.status,
        "input_mesh": str(case.input_mesh) if case.input_mesh else "",
        "output_mesh": str(case.output_mesh) if case.output_mesh else "",
        "gt_mesh": str(case.gt_mesh) if case.gt_mesh else "",
        "runtime_sec": case.runtime_sec,
        "violation": case.violation,
        "error": error if error is not None else case.error,
    }


def evaluate_case(case: EvalCase, samples: int) -> dict[str, Any]:
    row = empty_row(case)
    if case.status != "success":
        return row
    if case.output_mesh is None or not case.output_mesh.exists():
        row["status"] = "missing_output"
        row["error"] = f"output mesh missing: {case.output_mesh}"
        return row
    if case.gt_mesh is None or not case.gt_mesh.exists():
        row["status"] = "missing_gt"
        row["error"] = f"ground truth missing: {case.gt_mesh}"
        return row

    try:
        vertices, faces = mesh_metrics.load_mesh(case.output_mesh)
        gt_vertices, gt_faces = mesh_metrics.load_mesh(case.gt_mesh)
        qualities = mesh_metrics.triangle_qualities(vertices, faces)
        gt_info = mesh_metrics.mesh_summary(gt_vertices, gt_faces, 30.0)
        chamfer = mesh_metrics.chamfer_summary(vertices, faces, gt_vertices, gt_faces, samples, float(gt_info["bbox"]["diagonal"]))
        row.update(
            {
                "output_vertices": int(len(vertices)),
                "output_faces": int(len(faces)),
                "faces_out_in": float(len(faces) / case.input_faces) if case.input_faces else math.nan,
                "qavg": float(np.mean(qualities)) if len(qualities) else math.nan,
                "qmin": float(np.min(qualities)) if len(qualities) else math.nan,
                "cd": chamfer["chamfer_distance"],
                "cd_bbox2": chamfer["chamfer_distance_over_bbox_diag2"],
                "symmetric_rmse": chamfer["symmetric_rmse"],
                "symmetric_rmse_bbox_pct": chamfer["symmetric_rmse_ratio_bbox_pct"],
                "sampled_hausdorff": chamfer["sampled_hausdorff_distance"],
                "sampled_hausdorff_bbox_pct": chamfer["sampled_hausdorff_ratio_bbox_pct"],
            }
        )
        row.update(angle_summary(vertices, faces))
        row.update(edge_summary(vertices, faces))
        return row
    except Exception as exc:  # noqa: BLE001 - batch report must preserve per-case failures.
        row["status"] = "eval_failed"
        row["error"] = str(exc)
        return row


def finite_values(rows: list[dict[str, Any]], field: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = to_float(row.get(field))
        if value is not None:
            values.append(value)
    return values


def mean(rows: list[dict[str, Any]], field: str) -> float:
    values = finite_values(rows, field)
    return float(np.mean(values)) if values else math.nan


def median(rows: list[dict[str, Any]], field: str) -> float:
    values = finite_values(rows, field)
    return float(np.median(values)) if values else math.nan


def minimum(rows: list[dict[str, Any]], field: str) -> float:
    values = finite_values(rows, field)
    return float(np.min(values)) if values else math.nan


def maximum(rows: list[dict[str, Any]], field: str) -> float:
    values = finite_values(rows, field)
    return float(np.max(values)) if values else math.nan


def success_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("status") == "success" and to_float(row.get("qavg")) is not None]


def summary_rows(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        by_method[str(row["method"])].append(row)

    for method, rows in sorted(by_method.items()):
        ok = success_rows(rows)
        result.append(
            {
                "method": method,
                "records": len(rows),
                "successes": len(ok),
                "success_rate": len(ok) / len(rows) if rows else math.nan,
                "qavg_mean": mean(ok, "qavg"),
                "qavg_median": median(ok, "qavg"),
                "qmin_mean": mean(ok, "qmin"),
                "qmin_min": minimum(ok, "qmin"),
                "min_angle_mean": mean(ok, "min_angle_deg"),
                "angle_lt_30_pct_mean": mean(ok, "angle_lt_30_pct"),
                "angle_lt_20_pct_mean": mean(ok, "angle_lt_20_pct"),
                "angle_lt_10_pct_mean": mean(ok, "angle_lt_10_pct"),
                "cd_bbox2_mean": mean(ok, "cd_bbox2"),
                "rmse_bbox_pct_mean": mean(ok, "symmetric_rmse_bbox_pct"),
                "hausdorff_bbox_pct_mean": mean(ok, "sampled_hausdorff_bbox_pct"),
                "output_faces_mean": mean(ok, "output_faces"),
                "faces_out_in_mean": mean(ok, "faces_out_in"),
                "runtime_sec_mean": mean(ok, "runtime_sec"),
            }
        )
    return result


def common_rows(raw_rows: list[dict[str, Any]], methods: set[str]) -> dict[str, list[dict[str, Any]]]:
    by_method_model: dict[tuple[str, str], dict[str, Any]] = {}
    models_by_method: dict[str, set[str]] = defaultdict(set)
    for row in success_rows(raw_rows):
        method = str(row["method"])
        model = str(row["model"])
        if method in methods:
            by_method_model[(method, model)] = row
            models_by_method[method].add(model)
    common_models = set.intersection(*(models_by_method[method] for method in methods))
    return {
        method: [by_method_model[(method, model)] for model in sorted(common_models)]
        for method in sorted(methods)
    }


def wins_against_ours(raw_rows: list[dict[str, Any]], methods: set[str]) -> list[dict[str, Any]]:
    grouped = common_rows(raw_rows, methods)
    if METHOD_OURS not in grouped:
        return []
    lower_is_better = {
        "qmin": False,
        "qavg": False,
        "min_angle_deg": False,
        "angle_lt_30_pct": True,
        "angle_lt_20_pct": True,
        "angle_lt_10_pct": True,
        "cd_bbox2": True,
        "symmetric_rmse_bbox_pct": True,
        "sampled_hausdorff_bbox_pct": True,
        "runtime_sec": True,
    }
    result: list[dict[str, Any]] = []
    ours_rows = grouped[METHOD_OURS]
    for method, rows in grouped.items():
        if method == METHOD_OURS:
            continue
        item: dict[str, Any] = {"vs": method, "common_models": len(rows)}
        for metric, lower in lower_is_better.items():
            wins = 0
            ties = 0
            for ours, other in zip(ours_rows, rows, strict=True):
                ov = to_float(ours.get(metric))
                tv = to_float(other.get(metric))
                if ov is None or tv is None:
                    continue
                if math.isclose(ov, tv, rel_tol=1e-12, abs_tol=1e-12):
                    ties += 1
                elif (ov < tv) if lower else (ov > tv):
                    wins += 1
            item[f"{metric}_wins"] = wins
            item[f"{metric}_ties"] = ties
        result.append(item)
    return result


def size_bucket(faces_out_in: float | None) -> str:
    if faces_out_in is None:
        return "unknown"
    if faces_out_in < 0.75:
        return "<0.75x"
    if faces_out_in < 1.25:
        return "0.75-1.25x"
    if faces_out_in < 2.0:
        return "1.25-2.0x"
    return ">=2.0x"


def size_summary_rows(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in success_rows(raw_rows):
        groups[(str(row["method"]), size_bucket(to_float(row.get("faces_out_in"))))].append(row)
    result: list[dict[str, Any]] = []
    for (method, bucket), rows in sorted(groups.items()):
        result.append(
            {
                "method": method,
                "faces_out_in_bucket": bucket,
                "cases": len(rows),
                "qavg_mean": mean(rows, "qavg"),
                "qmin_mean": mean(rows, "qmin"),
                "rmse_bbox_pct_mean": mean(rows, "symmetric_rmse_bbox_pct"),
                "hausdorff_bbox_pct_mean": mean(rows, "sampled_hausdorff_bbox_pct"),
                "runtime_sec_mean": mean(rows, "runtime_sec"),
            }
        )
    return result


def fmt(value: Any, digits: int = 5) -> str:
    number = to_float(value)
    if number is None:
        return ""
    if abs(number) != 0.0 and (abs(number) < 1e-4 or abs(number) >= 1e5):
        return f"{number:.6E}"
    return f"{number:.{digits}f}"


def markdown_table(columns: list[str], rows: list[dict[str, Any]], limit: int | None = None) -> list[str]:
    selected = rows[:limit] if limit is not None else rows
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in selected:
        lines.append("| " + " | ".join(fmt(row.get(column)) if isinstance(row.get(column), float) else str(row.get(column, "")) for column in columns) + " |")
    return lines


def write_markdown(
    path: Path,
    raw_rows: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    wins: list[dict[str, Any]],
    size_rows: list[dict[str, Any]],
    raw_cgal_counts: Counter[str],
) -> None:
    failures = [row for row in raw_rows if row.get("status") != "success"]
    lines: list[str] = [
        "# v5.1.3 cpp_adaiso vs CWF/CGAL 统一评估报告",
        "",
        "## 口径",
        "",
        f"- 统一 evaluator: `tools/mesh_metrics.py` imported by `tools/run_unified_eval.py`.",
        f"- CD/RMSE/Hausdorff: 每方向 `{SAMPLES_PER_DIRECTION}` 个面积采样点。",
        "- 误差归一化: 使用 GT mesh 的 bbox diagonal。",
        "- 不重跑 remeshing；只评估已有输出网格。",
        "- CGAL 原始状态统计保留所有记录；共同模型对比中 `T21` 采用 `T21.off` 行。",
        "",
        "## 表 1：成功率和总体均值",
        "",
    ]
    lines.extend(
        markdown_table(
            [
                "method",
                "records",
                "successes",
                "success_rate",
                "qavg_mean",
                "qmin_mean",
                "qmin_min",
                "min_angle_mean",
                "angle_lt_30_pct_mean",
                "rmse_bbox_pct_mean",
                "hausdorff_bbox_pct_mean",
                "output_faces_mean",
                "runtime_sec_mean",
            ],
            summaries,
        )
    )
    lines.extend(
        [
            "",
            f"CGAL 原始 status 计数：`{dict(raw_cgal_counts)}`。上表使用去重后的模型行，避免 `T21.obj`/`T21.off` 重复进入共同模型对比。",
            "",
            "## 表 2：cpp_adaiso 与 CWF10k/CGAL2019 共同成功模型",
            "",
        ]
    )
    common_primary = common_rows(raw_rows, {METHOD_OURS, METHOD_CWF10K, METHOD_CGAL})
    primary_summaries = []
    for method, rows in common_primary.items():
        primary_summaries.append(
            {
                "method": method,
                "common_models": len(rows),
                "qavg_mean": mean(rows, "qavg"),
                "qmin_mean": mean(rows, "qmin"),
                "min_angle_mean": mean(rows, "min_angle_deg"),
                "angle_lt_30_pct_mean": mean(rows, "angle_lt_30_pct"),
                "cd_bbox2_mean": mean(rows, "cd_bbox2"),
                "rmse_bbox_pct_mean": mean(rows, "symmetric_rmse_bbox_pct"),
                "hausdorff_bbox_pct_mean": mean(rows, "sampled_hausdorff_bbox_pct"),
                "output_faces_mean": mean(rows, "output_faces"),
                "runtime_sec_mean": mean(rows, "runtime_sec"),
            }
        )
    lines.extend(markdown_table(list(primary_summaries[0].keys()) if primary_summaries else [], primary_summaries))
    lines.extend(["", "## 表 3：逐模型胜负计数", ""])
    lines.extend(markdown_table(list(wins[0].keys()) if wins else [], wins))
    lines.extend(["", "## 表 4：按输出规模分组", ""])
    lines.extend(markdown_table(list(size_rows[0].keys()) if size_rows else [], size_rows))
    lines.extend(["", "## 表 5：失败/异常案例", ""])
    failure_columns = ["method", "model", "source_model", "status", "output_mesh", "error"]
    lines.extend(markdown_table(failure_columns, failures, limit=80))
    if len(failures) > 80:
        lines.append(f"\n仅展示前 80 条失败/异常；完整列表见 raw CSV，共 {len(failures)} 条。")
    lines.extend(
        [
            "",
            "## 内部判断",
            "",
            "- `cpp_adaiso_v5.1.3` 的成功率和速度是当前最明确优势。",
            "- 与 CWF 的质量对比必须看统一评估后的 `qavg/qmin/min_angle/angle_lt_*` 和归一化 RMSE/Hausdorff，不能再混用旧 CSV。",
            "- 如果 `cpp_adaiso` 的误差优势依赖明显更多输出面数，论文里只能写 trade-off，不能写全面优于。",
            "- 下一阶段如果需要更强结论，应做 target-face 或 target-vertex 对齐重跑。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    out_dir = root / "compare" / "unified_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    cgal, raw_cgal_counts = cgal_cases(root)
    cases = [
        *ours_cases(root),
        *cwf_cases(root, METHOD_CWF10K, "compare/CWF/results/results_10000/metrics_summary.csv"),
        *cwf_cases(root, METHOD_CWF2K, "compare/CWF/results/results_2000/metrics_summary.csv"),
        *cgal,
    ]

    if args.smoke:
        keep = {"T0", "model_001"}
        cases = [case for case in cases if case.model in keep]

    raw_rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case.method} {case.model} {case.status}", flush=True)
        raw_rows.append(evaluate_case(case, args.samples))

    raw_csv = out_dir / "v5.1.3_cwf_cgal_unified_metrics.csv"
    summary_csv = out_dir / "v5.1.3_cwf_cgal_unified_summary.csv"
    wins_csv = out_dir / "v5.1.3_cwf_cgal_unified_wins.csv"
    size_csv = out_dir / "v5.1.3_cwf_cgal_unified_size_summary.csv"
    report_md = out_dir / "v5.1.3_cwf_cgal_unified_metrics.md"

    summaries = summary_rows(raw_rows)
    wins = wins_against_ours(raw_rows, {METHOD_OURS, METHOD_CWF10K, METHOD_CGAL})
    size_rows = size_summary_rows(raw_rows)

    write_csv(raw_csv, raw_rows, RAW_COLUMNS)
    write_csv(summary_csv, summaries, list(summaries[0].keys()) if summaries else [])
    write_csv(wins_csv, wins, list(wins[0].keys()) if wins else [])
    write_csv(size_csv, size_rows, list(size_rows[0].keys()) if size_rows else [])
    write_markdown(report_md, raw_rows, summaries, wins, size_rows, raw_cgal_counts)

    print(f"wrote {raw_csv}")
    print(f"wrote {summary_csv}")
    print(f"wrote {wins_csv}")
    print(f"wrote {size_csv}")
    print(f"wrote {report_md}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run unified metrics for cpp_adaiso/CWF/CGAL outputs.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root.")
    parser.add_argument("--samples", type=int, default=SAMPLES_PER_DIRECTION, help="Area samples per CD direction.")
    parser.add_argument("--smoke", action="store_true", help="Evaluate only T0 and model_001 path mappings.")
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(run(parse_args()))
