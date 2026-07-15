from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = ROOT / "cpp_adaiso" / "examples"
RESULT_ROOT = ROOT / "cpp_adaiso" / "results" / "v2.11.1_T_all_iter10_20260714"
BASELINE_ROOT = ROOT / "cpp_adaiso" / "results" / "v2.11.0_T_all_iter10_20260714"
REPORT_PATH = ROOT / "cpp_adaiso" / "修改计划与实验结果" / "v2.11.1 T系列 iter10 vs v2.11.0实验报告.md"
RUN_SUMMARY_PATH = RESULT_ROOT / "run_summary.csv"
COMPARE_SUMMARY_PATH = RESULT_ROOT / "comparison_vs_v2.11.0.csv"
DEFAULT_CLI = ROOT / "cpp_adaiso" / "build2" / "adaiso_cli.exe"

CURRENT_MAPPING = "principal_curvature_screened_kappa_inverse_power_log_downward_gradation"
BASELINE_MAPPING = "principal_curvature_initial_h_screened_laplacian"
EXPECTED_EXPONENT = 0.75

CURRENT_TARGET_FIELD_FILES = (
    "color_scales.json",
    "edges.csv",
    "faces.csv",
    "field_stats.json",
    "initial_h_heat.ply",
    "k1_abs_heat.ply",
    "k2_abs_heat.ply",
    "limited_h_heat.ply",
    "raw_kappa_heat.ply",
    "smoothed_kappa_heat.ply",
    "target_h_heat.ply",
    "vertices.csv",
)

BASELINE_TARGET_FIELD_FILES = (
    "color_scales.json",
    "edges.csv",
    "faces.csv",
    "field_stats.json",
    "initial_h_heat.ply",
    "k1_abs_heat.ply",
    "k2_abs_heat.ply",
    "kappa_heat.ply",
    "smoothed_h_heat.ply",
    "target_h_heat.ply",
    "vertices.csv",
)

RUN_FIELDS = (
    "model",
    "input_vertices",
    "input_faces",
    "output_vertices",
    "output_faces",
    "target_faces",
    "qavg",
    "qmin",
    "iter10_violation",
    "cd",
    "symmetric_rmse",
    "runtime_sec",
    "target_size_mapping",
    "curvature_smoothing_alpha",
    "target_gradation_ratio",
    "solver_success",
    "observed_h_min",
    "h_p05",
    "h_p50",
    "h_p95",
    "observed_h_max",
    "h_p95_p05",
    "h_max_min",
    "max_edge_target_ratio",
    "p95_edge_target_ratio",
    "p99_edge_target_ratio",
    "max_grad_h_norm",
    "p95_grad_h_norm",
    "p99_grad_h_norm",
    "max_normalized_log_slope",
    "p95_normalized_log_slope",
    "p99_normalized_log_slope",
    "gradation_changed_vertex_rate",
    "final_face_estimate_relative_error",
    "split_total",
    "collapse_total",
    "flip_total",
    "global_iterations",
    "cd_samples",
    "principal_curvature_fallback_rate",
)

COMPARE_FIELDS = (
    "model",
    "v2111_cd",
    "v2110_cd",
    "delta_cd",
    "delta_cd_pct",
    "v2111_rmse",
    "v2110_rmse",
    "delta_rmse",
    "delta_rmse_pct",
    "v2111_output_faces",
    "v2110_output_faces",
    "delta_output_faces",
    "delta_output_faces_pct",
    "v2111_qavg",
    "v2110_qavg",
    "delta_qavg",
    "v2111_qmin",
    "v2110_qmin",
    "delta_qmin",
    "v2111_violation",
    "v2110_violation",
    "delta_violation",
    "v2111_max_edge_target_ratio",
    "v2110_max_edge_target_ratio",
    "delta_max_edge_target_ratio",
    "v2111_max_grad_h_norm",
    "v2110_max_grad_h_norm",
    "delta_max_grad_h_norm",
    "v2111_runtime_sec",
    "v2110_runtime_sec",
    "delta_runtime_sec",
    "delta_runtime_sec_pct",
)


def natural_model_key(path_or_name: Path | str) -> int:
    name = path_or_name.stem if isinstance(path_or_name, Path) else path_or_name
    match = re.fullmatch(r"T(\d+)", name)
    if not match:
        raise ValueError(f"unexpected T-series model name: {name}")
    return int(match.group(1))


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def rel_arg(path: Path) -> str:
    return str(path.relative_to(ROOT))


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def fmt(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def sci(value: float) -> str:
    return f"{value:.6E}"


def signed(value: float, digits: int = 6) -> str:
    return f"{value:+.{digits}f}"


def signed_pct(value: float) -> str:
    return f"{value:+.2f}%"


def pct_delta(current: float, baseline: float) -> float:
    if baseline == 0.0:
        return 0.0 if current == 0.0 else math.copysign(math.inf, current)
    return 100.0 * (current - baseline) / baseline


def discover_models() -> list[Path]:
    models = []
    for path in EXAMPLES_ROOT.glob("T*.off"):
        if re.fullmatch(r"T\d+", path.stem):
            models.append(path)
    models.sort(key=natural_model_key)
    if not models:
        raise RuntimeError(f"no T-series models found under {EXAMPLES_ROOT}")
    return models


def case_dir(root: Path, model: str) -> Path:
    return root / model


def required_target_files(root: Path) -> tuple[str, ...]:
    return CURRENT_TARGET_FIELD_FILES if root == RESULT_ROOT else BASELINE_TARGET_FIELD_FILES


def missing_target_files(root: Path, model: str) -> list[str]:
    target_dir = case_dir(root, model) / "target_field"
    return [name for name in required_target_files(root) if not (target_dir / name).is_file()]


def quantile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    position = fraction * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    t = position - lower
    return sorted_values[lower] * (1.0 - t) + sorted_values[upper] * t


def target_h_quantiles(root: Path, model: str) -> dict[str, float]:
    values: list[float] = []
    with (case_dir(root, model) / "target_field" / "vertices.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            values.append(float(row["target_h"]))
    values.sort()
    if not values or values[0] <= 0.0:
        raise RuntimeError(f"{model}: target_h values are empty or non-positive")
    return {
        "observed_h_min": values[0],
        "h_p05": quantile(values, 0.05),
        "h_p50": quantile(values, 0.50),
        "h_p95": quantile(values, 0.95),
        "observed_h_max": values[-1],
        "h_p95_p05": values[-1] / values[0] if values[0] == values[-1] else quantile(values, 0.95) / quantile(values, 0.05),
        "h_max_min": values[-1] / values[0],
    }


def read_case(root: Path, model: str) -> dict:
    directory = case_dir(root, model)
    summary = load_json(directory / "summary.json")
    field = load_json(directory / "target_field" / "field_stats.json")
    parameters = summary["parameters"]
    stats = summary["stats"]
    iterations = summary["iterations"]
    chamfer = summary["chamfer_distance"]
    output = summary["output_summary"]
    input_summary = summary["input_summary"]
    h = target_h_quantiles(root, model)
    current = root == RESULT_ROOT
    row = {
        "model": model,
        "input_vertices": int(input_summary["vertices"]),
        "input_faces": int(input_summary["faces"]),
        "output_vertices": int(output["vertices"]),
        "output_faces": int(output["faces"]),
        "target_faces": int(parameters["target_faces"]),
        "qavg": float(output["qavg"]),
        "qmin": float(output["qmin"]),
        "iter10_violation": float(iterations[-1]["violation_rate"]),
        "cd": float(chamfer["chamfer_distance"]),
        "symmetric_rmse": float(chamfer["symmetric_rmse"]),
        "runtime_sec": float(summary["runtime_sec"]),
        "target_size_mapping": str(parameters["target_size_mapping"]),
        "curvature_smoothing_alpha": float(
            parameters["curvature_smoothing_alpha"] if current else parameters["target_smoothing_alpha"]
        ),
        "target_gradation_ratio": float(parameters.get("target_gradation_ratio", 0.0)),
        "solver_success": bool(
            stats["curvature_smoothing_solver_success"] if current else stats["target_smoothing_solver_success"]
        ),
        **h,
        "max_edge_target_ratio": float(field["max_edge_target_ratio"]),
        "p95_edge_target_ratio": float(field["p95_edge_target_ratio"]),
        "p99_edge_target_ratio": float(field["p99_edge_target_ratio"]),
        "max_grad_h_norm": float(field["max_grad_h_norm"]),
        "p95_grad_h_norm": float(field["p95_grad_h_norm"]),
        "p99_grad_h_norm": float(field["p99_grad_h_norm"]),
        "max_normalized_log_slope": float(field.get("max_normalized_log_slope", math.nan)),
        "p95_normalized_log_slope": float(field.get("p95_normalized_log_slope", math.nan)),
        "p99_normalized_log_slope": float(field.get("p99_normalized_log_slope", math.nan)),
        "gradation_changed_vertex_rate": float(stats.get("gradation_changed_vertex_rate", 0.0)),
        "final_face_estimate_relative_error": float(stats.get("final_face_estimate_relative_error", 0.0)),
        "split_total": sum(int(item["split_count"]) for item in iterations),
        "collapse_total": sum(int(item["collapse_count"]) for item in iterations),
        "flip_total": sum(int(item["flip_count"]) for item in iterations),
        "global_iterations": int(parameters["global_iterations"]),
        "cd_samples": int(chamfer["samples_per_direction"]),
        "principal_curvature_fallback_rate": float(stats["principal_curvature_fallback_rate"]),
    }
    return row


def case_is_complete(root: Path, model: str, iters: int, cd_samples: int) -> bool:
    directory = case_dir(root, model)
    if not (directory / "summary.json").is_file() or not (directory / "output.ply").is_file():
        return False
    if missing_target_files(root, model):
        return False
    try:
        row = read_case(root, model)
    except (KeyError, ValueError, OSError, json.JSONDecodeError, RuntimeError):
        return False
    expected_mapping = CURRENT_MAPPING if root == RESULT_ROOT else BASELINE_MAPPING
    return (
        row["target_size_mapping"] == expected_mapping
        and row["global_iterations"] == iters
        and row["cd_samples"] == cd_samples
        and row["solver_success"]
    )


def run_case(model_path: Path, args: argparse.Namespace) -> None:
    model = model_path.stem
    directory = case_dir(RESULT_ROOT, model)
    target_dir = directory / "target_field"
    directory.mkdir(parents=True, exist_ok=True)
    command = [
        str(args.cli),
        rel_arg(model_path),
        "-o",
        rel_arg(directory / "output.ply"),
        "--preprocess",
        "0",
        "--iters",
        str(args.iters),
        "--curvature-smoothing-alpha",
        str(args.curvature_smoothing_alpha),
        "--target-gradation",
        "1",
        "--target-gradation-ratio",
        str(args.target_gradation_ratio),
        "--gt",
        rel_arg(model_path),
        "--cd-samples",
        str(args.cd_samples),
        "--export-target-field",
        rel_arg(target_dir),
        "--summary-json",
        rel_arg(directory / "summary.json"),
        "--quiet",
    ]
    (directory / "command.txt").write_text(
        subprocess.list2cmdline(command) + "\n", encoding="utf-8"
    )
    started = time.perf_counter()
    proc = subprocess.run(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    elapsed = time.perf_counter() - started
    (directory / "stdout.log").write_text(proc.stdout, encoding="utf-8")
    (directory / "stderr.log").write_text(proc.stderr, encoding="utf-8")
    print(f"[{model}] exit={proc.returncode} elapsed={elapsed:.1f}s", flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{model} failed; see {rel(directory / 'stderr.log')}")


def collect_rows(models: list[Path], root: Path, iters: int, cd_samples: int) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    errors: list[str] = []
    for path in models:
        model = path.stem
        if not case_is_complete(root, model, iters, cd_samples):
            errors.append(f"{model}: incomplete or parameter mismatch")
            continue
        try:
            rows.append(read_case(root, model))
        except Exception as error:  # noqa: BLE001 - report every damaged case
            errors.append(f"{model}: {error}")
    rows.sort(key=lambda row: natural_model_key(row["model"]))
    return rows, errors


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)


def compare_rows(current: dict, baseline: dict) -> dict:
    result = {"model": current["model"]}
    pairs = (
        ("cd", "cd"),
        ("rmse", "symmetric_rmse"),
        ("output_faces", "output_faces"),
        ("qavg", "qavg"),
        ("qmin", "qmin"),
        ("violation", "iter10_violation"),
        ("max_edge_target_ratio", "max_edge_target_ratio"),
        ("max_grad_h_norm", "max_grad_h_norm"),
        ("runtime_sec", "runtime_sec"),
    )
    for label, key in pairs:
        current_value = current[key]
        baseline_value = baseline[key]
        result[f"v2111_{label}"] = current_value
        result[f"v2110_{label}"] = baseline_value
        result[f"delta_{label}"] = current_value - baseline_value
        if label in {"cd", "rmse", "output_faces", "runtime_sec"}:
            result[f"delta_{label}_pct"] = pct_delta(current_value, baseline_value)
    return result


def average(rows: list[dict], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows) if rows else math.nan


def count_where(rows: list[dict], predicate: Callable[[dict], bool]) -> int:
    return sum(1 for row in rows if predicate(row))


def result_table(rows: list[dict]) -> str:
    lines = [
        "| Model | Out F | Qavg | Qmin | viol | CD | RMSE | h p95/p05 | max edge | max grad | max log slope | changed | runtime |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['output_faces']} | {fmt(row['qavg'])} | {fmt(row['qmin'])} | "
            f"{fmt(row['iter10_violation'])} | {sci(row['cd'])} | {sci(row['symmetric_rmse'])} | "
            f"{fmt(row['h_p95_p05'])} | {fmt(row['max_edge_target_ratio'])} | "
            f"{fmt(row['max_grad_h_norm'])} | {fmt(row['max_normalized_log_slope'])} | "
            f"{100.0 * row['gradation_changed_vertex_rate']:.1f}% | {row['runtime_sec']:.2f}s |"
        )
    return "\n".join(lines)


def comparison_table(rows: list[dict]) -> str:
    lines = [
        "| Model | dCD% | dRMSE% | dFaces% | dQavg | dQmin | dviol | dmax edge | dmax grad | dRuntime% |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {signed_pct(row['delta_cd_pct'])} | {signed_pct(row['delta_rmse_pct'])} | "
            f"{signed_pct(row['delta_output_faces_pct'])} | {signed(row['delta_qavg'])} | "
            f"{signed(row['delta_qmin'])} | {signed(row['delta_violation'])} | "
            f"{signed(row['delta_max_edge_target_ratio'])} | {signed(row['delta_max_grad_h_norm'])} | "
            f"{signed_pct(row['delta_runtime_sec_pct'])} |"
        )
    return "\n".join(lines)


def top_models(rows: list[dict], key: str, count: int, reverse: bool) -> str:
    selected = sorted(rows, key=lambda row: row[key], reverse=reverse)[:count]
    return ", ".join(f"`{row['model']}` ({row[key]:+.4g})" for row in selected)


def git_status_text() -> str:
    proc = subprocess.run(
        ["git", "status", "--short"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return proc.stdout.strip() or "clean"


def write_report(
    models: list[Path],
    current_rows: list[dict],
    baseline_rows: list[dict],
    comparisons: list[dict],
    current_errors: list[str],
    baseline_errors: list[str],
    args: argparse.Namespace,
) -> None:
    complete_pairs = len(comparisons)
    allowed_slope = math.log(args.target_gradation_ratio)
    gradation_pass = count_where(
        current_rows,
        lambda row: row["max_normalized_log_slope"] <= allowed_slope + 1e-10,
    )

    cd_better = count_where(comparisons, lambda row: row["delta_cd"] < 0.0)
    qavg_better = count_where(comparisons, lambda row: row["delta_qavg"] > 0.0)
    qmin_better = count_where(comparisons, lambda row: row["delta_qmin"] > 0.0)
    violation_better = count_where(comparisons, lambda row: row["delta_violation"] < 0.0)

    avg_dqavg = average(comparisons, "delta_qavg")
    avg_dqmin = average(comparisons, "delta_qmin")
    avg_dviol = average(comparisons, "delta_violation")
    current_avg_cd = average(current_rows, "cd")
    baseline_avg_cd = average(baseline_rows, "cd")
    current_avg_rmse = average(current_rows, "symmetric_rmse")
    baseline_avg_rmse = average(baseline_rows, "symmetric_rmse")
    current_avg_faces = average(current_rows, "output_faces")
    baseline_avg_faces = average(baseline_rows, "output_faces")
    current_avg_runtime = average(current_rows, "runtime_sec")
    baseline_avg_runtime = average(baseline_rows, "runtime_sec")
    if avg_dqavg > 0.0 and avg_dqmin > 0.0 and avg_dviol < 0.0:
        quality_conclusion = "三项平均质量指标同时改善"
    elif avg_dqavg < 0.0 and avg_dqmin < 0.0 and avg_dviol > 0.0:
        quality_conclusion = "三项平均质量指标同时退化"
    else:
        quality_conclusion = "质量指标呈混合变化"

    lines = [
        "# v2.11.1 T系列 iter10 vs v2.11.0 实验报告",
        "",
        "## 实验设置",
        "",
        "本实验使用当前 v2.11.1 管线，对全部 T 系列模型运行 10 轮重网格化，并与已归档的 v2.11.0 T 系列 iter10 结果比较。每个模型使用自身输入作为 GT，计算 area-sampled Chamfer Distance 和 symmetric RMSE。",
        "",
        "| 参数 | v2.11.1 | v2.11.0 |",
        "|---|---:|---:|",
        f"| 模型数量 | {len(models)} | {len(models)} |",
        f"| iters | {args.iters} | 10 |",
        "| preprocess | 0 | 0 |",
        f"| curvature smoothing alpha | {args.curvature_smoothing_alpha} | target smoothing alpha=2.0 |",
        f"| target gradation ratio | {args.target_gradation_ratio} | 无硬约束 |",
        f"| CD samples | {args.cd_samples} | 100000 |",
        f"| target_size_mapping | `{CURRENT_MAPPING}` | `{BASELINE_MAPPING}` |",
        f"| target curvature exponent | {EXPECTED_EXPONENT} | {EXPECTED_EXPONENT} |",
        "",
        f"- v2.11.1 输出目录：`{rel(RESULT_ROOT)}/`",
        f"- v2.11.1 汇总 CSV：`{rel(RUN_SUMMARY_PATH)}`",
        f"- 对比 CSV：`{rel(COMPARE_SUMMARY_PATH)}`",
        f"- v2.11.0 基线：`{rel(BASELINE_ROOT)}/`",
        "",
        "## 完整性与约束检查",
        "",
        "| 检查项 | 结果 |",
        "|---|---:|",
        f"| v2.11.1 完成模型 | {len(current_rows)} / {len(models)} |",
        f"| v2.11.0 基线完整 | {len(baseline_rows)} / {len(models)} |",
        f"| 可比较模型 | {complete_pairs} / {len(models)} |",
        f"| curvature solver success | {count_where(current_rows, lambda row: row['solver_success'])} / {len(models)} |",
        f"| gradation 硬约束通过 | {gradation_pass} / {len(models)} |",
        f"| 最大允许 normalized log slope | {allowed_slope:.10f} |",
        f"| 实测最大 normalized log slope | {max((row['max_normalized_log_slope'] for row in current_rows), default=math.nan):.10f} |",
        f"| 最大 target-face 连续估计相对误差 | {max((row['final_face_estimate_relative_error'] for row in current_rows), default=math.nan):.3E} |",
        "",
    ]
    if current_errors or baseline_errors:
        lines.extend(["异常：", ""])
        lines.extend(f"- {error}" for error in current_errors + baseline_errors)
        lines.append("")

    lines.extend(
        [
            "## v2.11.1 单模型结果",
            "",
            result_table(current_rows),
            "",
            "## 汇总对比",
            "",
            f"结论：**{quality_conclusion}**。相对 v2.11.0，Qavg 提高 {qavg_better}/{complete_pairs}，Qmin 提高 {qmin_better}/{complete_pairs}，violation 降低 {violation_better}/{complete_pairs}；CD 降低 {cd_better}/{complete_pairs}。",
            "",
            "| 指标 | v2.11.1 | v2.11.0 | 差值/变化 |",
            "|---|---:|---:|---:|",
            f"| 平均 CD | {sci(current_avg_cd)} | {sci(baseline_avg_cd)} | {signed_pct(pct_delta(current_avg_cd, baseline_avg_cd))} |",
            f"| 平均 RMSE | {sci(current_avg_rmse)} | {sci(baseline_avg_rmse)} | {signed_pct(pct_delta(current_avg_rmse, baseline_avg_rmse))} |",
            f"| 平均输出面数 | {current_avg_faces:.1f} | {baseline_avg_faces:.1f} | {signed_pct(pct_delta(current_avg_faces, baseline_avg_faces))} |",
            f"| 平均 Qavg | {fmt(average(current_rows, 'qavg'))} | {fmt(average(baseline_rows, 'qavg'))} | {signed(avg_dqavg)} |",
            f"| 平均 Qmin | {fmt(average(current_rows, 'qmin'))} | {fmt(average(baseline_rows, 'qmin'))} | {signed(avg_dqmin)} |",
            f"| 平均 violation | {fmt(average(current_rows, 'iter10_violation'))} | {fmt(average(baseline_rows, 'iter10_violation'))} | {signed(avg_dviol)} |",
            f"| 平均 max edge-target ratio | {fmt(average(current_rows, 'max_edge_target_ratio'))} | {fmt(average(baseline_rows, 'max_edge_target_ratio'))} | {signed(average(comparisons, 'delta_max_edge_target_ratio'))} |",
            f"| 平均 max grad_h_norm | {fmt(average(current_rows, 'max_grad_h_norm'))} | {fmt(average(baseline_rows, 'max_grad_h_norm'))} | {signed(average(comparisons, 'delta_max_grad_h_norm'))} |",
            f"| 平均 runtime | {current_avg_runtime:.2f}s | {baseline_avg_runtime:.2f}s | {signed_pct(pct_delta(current_avg_runtime, baseline_avg_runtime))} |",
            f"| 平均逐模型 dCD% | - | - | {signed_pct(average(comparisons, 'delta_cd_pct'))} |",
            "",
            "## 逐模型差值",
            "",
            comparison_table(comparisons),
            "",
            "## 重点模型",
            "",
            f"- CD 改善最大：{top_models(comparisons, 'delta_cd_pct', 5, False)}",
            f"- CD 退化最大：{top_models(comparisons, 'delta_cd_pct', 5, True)}",
            f"- Qmin 改善最大：{top_models(comparisons, 'delta_qmin', 5, True)}",
            f"- Qmin 退化最大：{top_models(comparisons, 'delta_qmin', 5, False)}",
            f"- violation 变化最小：{top_models(comparisons, 'delta_violation', 5, False)}",
            "",
            "## 总体判断",
            "",
            "v2.11.1 的硬 gradation 目标已在 31/31 个输入尺寸场上达到，但当前默认管线不适合作为 v2.11.0 的直接替代：Qavg 没有任何模型提高，violation 没有任何模型降低，平均 Qavg/Qmin 均下降，平均 violation、输出面数和几何误差均上升。",
            "",
            "其中 `T21` 和 `T36` 出现明显失稳：输出面数分别增加约 106.8% 和 49.1%，Qavg 降至约 0.387 和 0.528，CD 分别上升约 1282% 和 1713%。在解决这些模型的尺寸传播与 remesh 跟踪问题前，不建议将 v2.11.1 当前参数设为默认。",
            "",
            "## 解释边界",
            "",
            "- v2.11.1 的 gradation 验收发生在输入顶点目标尺寸场上；它证明逐边 normalized log slope 受限，不等价于保证 remeshing 后每个三角形的 Qmin 提高。",
            "- v2.11.1 相对 v2.11.0 同时改变了平滑对象、尺寸映射校准顺序并加入 limiter，因此本报告评估的是完整管线，不是单独的 limiter 消融。",
            "- 两组均使用相同 T 输入、target_faces、remesh 参数、迭代数和 CD 采样数；输出面数差异是管线收敛结果的一部分。",
            "",
            "## 运行时工作区状态",
            "",
            "```text",
            git_status_text(),
            "```",
            "",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run v2.11.1 T-series iter10 and compare with v2.11.0."
    )
    parser.add_argument("--cli", type=Path, default=DEFAULT_CLI)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--cd-samples", type=int, default=100000)
    parser.add_argument("--curvature-smoothing-alpha", type=float, default=0.75)
    parser.add_argument("--target-gradation-ratio", type=float, default=1.30)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.cli = args.cli.resolve()
    if not args.cli.is_file():
        raise RuntimeError(f"CLI not found: {args.cli}")
    if args.iters <= 0 or args.cd_samples <= 0:
        raise RuntimeError("iters and cd-samples must be positive")
    if args.curvature_smoothing_alpha < 0.0 or not math.isfinite(args.curvature_smoothing_alpha):
        raise RuntimeError("curvature smoothing alpha must be finite and non-negative")
    if args.target_gradation_ratio <= 1.0 or not math.isfinite(args.target_gradation_ratio):
        raise RuntimeError("target gradation ratio must be finite and greater than one")

    models = discover_models()
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    if not args.report_only:
        for index, model_path in enumerate(models, start=1):
            model = model_path.stem
            if not args.force and case_is_complete(RESULT_ROOT, model, args.iters, args.cd_samples):
                print(f"[{index}/{len(models)}] {model}: complete, skipped", flush=True)
                continue
            print(f"[{index}/{len(models)}] {model}: running", flush=True)
            run_case(model_path, args)

    current_rows, current_errors = collect_rows(
        models, RESULT_ROOT, args.iters, args.cd_samples
    )
    baseline_rows, baseline_errors = collect_rows(models, BASELINE_ROOT, 10, 100000)
    current_by_model = {row["model"]: row for row in current_rows}
    baseline_by_model = {row["model"]: row for row in baseline_rows}
    comparisons = [
        compare_rows(current_by_model[model.stem], baseline_by_model[model.stem])
        for model in models
        if model.stem in current_by_model and model.stem in baseline_by_model
    ]
    write_csv(RUN_SUMMARY_PATH, RUN_FIELDS, current_rows)
    write_csv(COMPARE_SUMMARY_PATH, COMPARE_FIELDS, comparisons)
    write_report(
        models,
        current_rows,
        baseline_rows,
        comparisons,
        current_errors,
        baseline_errors,
        args,
    )
    print(
        f"complete: current={len(current_rows)}/{len(models)}, "
        f"baseline={len(baseline_rows)}/{len(models)}, report={rel(REPORT_PATH)}",
        flush=True,
    )
    return 0 if len(current_rows) == len(models) and len(baseline_rows) == len(models) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:  # noqa: BLE001 - command-line error boundary
        print(f"generate_v2_11_1: {error}", file=sys.stderr)
        raise SystemExit(2)
