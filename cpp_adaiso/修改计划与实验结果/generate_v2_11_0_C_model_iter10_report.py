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
RESULT_ROOT = ROOT / "cpp_adaiso" / "results" / "v2.11.0_C_model_iter10_20260714"
REPORT_PATH = ROOT / "cpp_adaiso" / "修改计划与实验结果" / "v2.11.0 C系列和model系列 iter10实验报告.md"
RUN_SUMMARY_PATH = RESULT_ROOT / "run_summary.csv"
DEFAULT_CLI = ROOT / "cpp_adaiso" / "build2" / "adaiso_cli.exe"

EXPECTED_MAPPING = "principal_curvature_initial_h_screened_laplacian"
EXPECTED_EXPONENT = 0.75

TARGET_FIELD_FILES = (
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
    "group",
    "model",
    "input_path",
    "input_vertices",
    "input_faces",
    "output_vertices",
    "output_faces",
    "face_ratio",
    "target_faces",
    "qavg",
    "qmin",
    "iter10_violation",
    "cd",
    "symmetric_rmse",
    "runtime_sec",
    "target_size_mapping",
    "target_curvature_exponent",
    "target_smoothing_alpha",
    "target_bisection_iterations",
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
    "split_total",
    "collapse_total",
    "flip_total",
    "global_iterations",
    "cd_samples",
    "principal_curvature_fallback_rate",
)


def model_group(name: str) -> str:
    if re.fullmatch(r"C\d+", name):
        return "C"
    if re.fullmatch(r"model_\d+", name):
        return "model"
    raise ValueError(f"unexpected model name: {name}")


def model_key(path_or_name: Path | str) -> tuple[int, int]:
    name = path_or_name.stem if isinstance(path_or_name, Path) else path_or_name
    match = re.fullmatch(r"C(\d+)", name)
    if match:
        return (0, int(match.group(1)))
    match = re.fullmatch(r"model_(\d+)", name)
    if match:
        return (1, int(match.group(1)))
    raise ValueError(f"unexpected model name: {name}")


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


def discover_models() -> list[Path]:
    models: list[Path] = []
    for path in EXAMPLES_ROOT.iterdir():
        if not path.is_file() or path.suffix.lower() not in {".obj", ".off", ".ply"}:
            continue
        if re.fullmatch(r"C\d+", path.stem) or re.fullmatch(r"model_\d+", path.stem):
            models.append(path)
    return sorted(models, key=model_key)


def expected_missing(models: list[str]) -> list[str]:
    existing = set(models)
    missing: list[str] = []
    for i in range(10):
        name = f"C{i}"
        if name not in existing:
            missing.append(name)
    for i in range(1, 11):
        name = f"model_{i:03d}"
        if name not in existing:
            missing.append(name)
    return missing


def case_dir(model: str) -> Path:
    return RESULT_ROOT / model


def missing_target_files(model: str) -> list[str]:
    field_dir = case_dir(model) / "target_field"
    return [name for name in TARGET_FIELD_FILES if not (field_dir / name).exists()]


def quantile(sorted_values: list[float], t: float) -> float:
    if not sorted_values:
        return 0.0
    pos = (len(sorted_values) - 1) * t
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[lo]
    return sorted_values[lo] * (hi - pos) + sorted_values[hi] * (pos - lo)


def target_h_quantiles(model: str) -> dict[str, float]:
    path = case_dir(model) / "target_field" / "vertices.csv"
    values: list[float] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            values.append(float(row["target_h"]))
    values.sort()
    h_min = values[0] if values else 0.0
    h_max = values[-1] if values else 0.0
    h_p05 = quantile(values, 0.05)
    h_p50 = quantile(values, 0.50)
    h_p95 = quantile(values, 0.95)
    return {
        "observed_h_min": h_min,
        "h_p05": h_p05,
        "h_p50": h_p50,
        "h_p95": h_p95,
        "observed_h_max": h_max,
        "h_p95_p05": h_p95 / h_p05 if h_p05 > 0.0 else 0.0,
        "h_max_min": h_max / h_min if h_min > 0.0 else 0.0,
    }


def tail_violation(summary: dict) -> float:
    iterations = summary.get("iterations", [])
    return float(iterations[-1].get("violation_rate", 0.0)) if iterations else 0.0


def read_case(model: str) -> dict:
    current_dir = case_dir(model)
    summary = load_json(current_dir / "summary.json")
    field_stats = load_json(current_dir / "target_field" / "field_stats.json")
    h_stats = target_h_quantiles(model)

    input_summary = summary["input_summary"]
    output_summary = summary["output_summary"]
    chamfer = summary["chamfer_distance"]
    parameters = summary.get("parameters", {})
    stats = summary.get("stats", {})
    iterations = summary.get("iterations", [])
    input_faces = int(input_summary["faces"])
    output_faces = int(output_summary["faces"])

    row = {
        "group": model_group(model),
        "model": model,
        "input_path": summary.get("input", ""),
        "input_vertices": int(input_summary["vertices"]),
        "input_faces": input_faces,
        "output_vertices": int(output_summary["vertices"]),
        "output_faces": output_faces,
        "face_ratio": output_faces / input_faces if input_faces else 0.0,
        "target_faces": int(parameters.get("target_faces", 0) or 0),
        "qavg": float(output_summary["qavg"]),
        "qmin": float(output_summary["qmin"]),
        "iter10_violation": tail_violation(summary),
        "cd": float(chamfer["chamfer_distance"]),
        "symmetric_rmse": float(chamfer["symmetric_rmse"]),
        "runtime_sec": float(summary["runtime_sec"]),
        "target_size_mapping": parameters.get("target_size_mapping", ""),
        "target_curvature_exponent": float(parameters.get("target_curvature_exponent", 0.0)),
        "target_smoothing_alpha": float(parameters.get("target_smoothing_alpha", 0.0)),
        "target_bisection_iterations": int(parameters.get("target_bisection_iterations", 0)),
        "solver_success": bool(stats.get("target_smoothing_solver_success", False)),
        "max_edge_target_ratio": float(field_stats["max_edge_target_ratio"]),
        "p95_edge_target_ratio": float(field_stats["p95_edge_target_ratio"]),
        "p99_edge_target_ratio": float(field_stats.get("p99_edge_target_ratio", 0.0)),
        "max_grad_h_norm": float(field_stats["max_grad_h_norm"]),
        "p95_grad_h_norm": float(field_stats.get("p95_grad_h_norm", 0.0)),
        "p99_grad_h_norm": float(field_stats.get("p99_grad_h_norm", 0.0)),
        "split_total": sum(int(item.get("split_count", 0)) for item in iterations),
        "collapse_total": sum(int(item.get("collapse_count", 0)) for item in iterations),
        "flip_total": sum(int(item.get("flip_count", 0)) for item in iterations),
        "global_iterations": int(parameters.get("global_iterations", len(iterations))),
        "cd_samples": int(chamfer.get("samples_per_direction", 0)),
        "principal_curvature_fallback_rate": float(stats.get("principal_curvature_fallback_rate", 0.0)),
    }
    row.update(h_stats)
    return row


def summary_has_expected_iterations(summary: dict, expected_iters: int) -> bool:
    iterations = summary.get("iterations", [])
    parameters = summary.get("parameters", {})
    if int(parameters.get("global_iterations", -1)) != expected_iters:
        return False
    if expected_iters == 0:
        return len(iterations) == 0
    return bool(iterations) and int(iterations[-1].get("iteration", -1)) == expected_iters


def case_is_complete(model: str, expected_iters: int, expected_cd_samples: int) -> bool:
    current_dir = case_dir(model)
    if not (current_dir / "output.ply").exists():
        return False
    summary_path = current_dir / "summary.json"
    if not summary_path.exists():
        return False
    if missing_target_files(model):
        return False
    try:
        summary = load_json(summary_path)
        row = read_case(model)
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        return False

    if row["target_size_mapping"] != EXPECTED_MAPPING:
        return False
    if abs(row["target_curvature_exponent"] - EXPECTED_EXPONENT) > 1e-12:
        return False
    if not summary_has_expected_iterations(summary, expected_iters):
        return False
    if row["cd_samples"] != expected_cd_samples:
        return False
    if not row["solver_success"]:
        return False
    return all(
        finite(row[key])
        for key in (
            "qavg",
            "qmin",
            "iter10_violation",
            "cd",
            "symmetric_rmse",
            "runtime_sec",
            "h_p95_p05",
            "h_max_min",
            "max_edge_target_ratio",
            "max_grad_h_norm",
        )
    )


def collect_rows(models: list[str], expected_iters: int, expected_cd_samples: int) -> tuple[list[dict], dict[str, str]]:
    rows: list[dict] = []
    errors: dict[str, str] = {}
    for model in models:
        if not case_is_complete(model, expected_iters, expected_cd_samples):
            missing = missing_target_files(model)
            errors[model] = "missing target-field files: " + ",".join(missing) if missing else "incomplete summary/metrics"
            continue
        rows.append(read_case(model))
    rows.sort(key=lambda row: model_key(row["model"]))
    return rows, errors


def run_case(model_path: Path, args: argparse.Namespace) -> None:
    model = model_path.stem
    current_dir = case_dir(model)
    current_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(args.cli),
        rel_arg(model_path),
        "-o",
        rel_arg(current_dir / "output.ply"),
        "--iters",
        str(args.iters),
        "--preprocess",
        "1" if args.preprocess else "0",
        "--target-smoothing-alpha",
        str(args.target_smoothing_alpha),
        "--target-bisection-iters",
        str(args.target_bisection_iters),
        "--gt",
        rel_arg(model_path),
        "--cd-samples",
        str(args.cd_samples),
        "--export-target-field",
        rel_arg(current_dir / "target_field"),
        "--summary-json",
        rel_arg(current_dir / "summary.json"),
        "--quiet",
    ]
    started = time.perf_counter()
    proc = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    elapsed = time.perf_counter() - started
    (current_dir / "run.log").write_text(
        "\n".join(
            [
                "$ " + " ".join(command),
                f"returncode: {proc.returncode}",
                f"wall_time_sec: {elapsed:.3f}",
                "--- stdout ---",
                proc.stdout,
                "--- stderr ---",
                proc.stderr,
            ]
        ),
        encoding="utf-8",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{model} failed; see {rel(current_dir / 'run.log')}")


def write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def average(rows: list[dict], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows) if rows else 0.0


def count_where(rows: list[dict], predicate: Callable[[dict], bool]) -> int:
    return sum(1 for row in rows if predicate(row))


def top_models(rows: list[dict], key: str, count: int, reverse: bool = True) -> str:
    selected = sorted(rows, key=lambda row: row[key], reverse=reverse)[:count]
    return ", ".join(f"`{row['model']}` ({row[key]:.6g})" for row in selected)


def result_table(rows: list[dict]) -> str:
    lines = [
        "| Group | Model | In F | Out F | F ratio | Qavg | Qmin | viol | CD | RMSE | h p95/p05 | h max/min | max edge | max grad | runtime |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["group"],
                    row["model"],
                    str(row["input_faces"]),
                    str(row["output_faces"]),
                    fmt(row["face_ratio"]),
                    fmt(row["qavg"]),
                    fmt(row["qmin"]),
                    fmt(row["iter10_violation"]),
                    sci(row["cd"]),
                    sci(row["symmetric_rmse"]),
                    fmt(row["h_p95_p05"]),
                    fmt(row["h_max_min"]),
                    fmt(row["max_edge_target_ratio"]),
                    fmt(row["max_grad_h_norm"]),
                    f"{row['runtime_sec']:.2f}s",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def group_summary_table(rows: list[dict]) -> str:
    lines = [
        "| Group | Count | Avg CD | Max CD | Avg RMSE | Avg Qavg | Avg Qmin | Min Qmin | Avg viol | Avg F ratio | Avg runtime |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in ("C", "model", "all"):
        selected = rows if group == "all" else [row for row in rows if row["group"] == group]
        if not selected:
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    group,
                    str(len(selected)),
                    f"{average(selected, 'cd'):.6E}",
                    f"{max(row['cd'] for row in selected):.6E}",
                    f"{average(selected, 'symmetric_rmse'):.6E}",
                    fmt(average(selected, "qavg")),
                    fmt(average(selected, "qmin")),
                    fmt(min(row["qmin"] for row in selected)),
                    fmt(average(selected, "iter10_violation")),
                    fmt(average(selected, "face_ratio")),
                    f"{average(selected, 'runtime_sec'):.2f}s",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def git_status_text() -> str:
    proc = subprocess.run(
        ["git", "-c", "core.quotepath=false", "status", "--short"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if proc.returncode != 0:
        return f"git status failed: {proc.stderr.strip()}"
    return proc.stdout.strip() or "(clean)"


def write_report(
    models: list[str],
    missing_inputs: list[str],
    rows: list[dict],
    errors: dict[str, str],
    args: argparse.Namespace,
) -> None:
    complete_count = len(rows)
    target_complete = sum(1 for model in models if not missing_target_files(model))
    mapping_count = count_where(rows, lambda row: row["target_size_mapping"] == EXPECTED_MAPPING)
    exponent_count = count_where(rows, lambda row: abs(row["target_curvature_exponent"] - EXPECTED_EXPONENT) <= 1e-12)
    solver_count = count_where(rows, lambda row: row["solver_success"])

    lines = [
        "# v2.11.0 C系列和model系列 iter10 实验报告",
        "",
        "## 实验设置",
        "",
        "本实验使用当前工作区 v2.11.0 方案，对 `examples` 中实际存在的 C 系列和 model 系列模型运行 10 轮重网格化。每个模型用自身输入作为 GT 计算 area-sampled Chamfer Distance 和 symmetric RMSE。",
        "",
        "| 参数 | 值 |",
        "|---|---:|",
        f"| 实际运行模型数 | {len(models)} |",
        f"| C 系列模型数 | {sum(1 for model in models if model_group(model) == 'C')} |",
        f"| model 系列模型数 | {sum(1 for model in models if model_group(model) == 'model')} |",
        f"| iters | {args.iters} |",
        f"| preprocess | {1 if args.preprocess else 0} |",
        f"| target_smoothing_alpha | {args.target_smoothing_alpha} |",
        f"| target_bisection_iters | {args.target_bisection_iters} |",
        f"| cd-samples | {args.cd_samples} |",
        f"| target_size_mapping | {EXPECTED_MAPPING} |",
        f"| target_curvature_exponent | {EXPECTED_EXPONENT} |",
        "",
        f"- 输出目录：`{rel(RESULT_ROOT)}/`",
        f"- 汇总 CSV：`{rel(RUN_SUMMARY_PATH)}`",
        "",
        "当前工作区状态：",
        "",
        "```text",
        git_status_text(),
        "```",
        "",
    ]
    if missing_inputs:
        lines.extend(
            [
                "输入缺失样例：",
                "",
                "- " + ", ".join(f"`{name}`" for name in missing_inputs),
                "",
            ]
        )

    lines.extend(
        [
            "## 完整性检查",
            "",
            "| 检查项 | 结果 |",
            "|---|---:|",
            f"| 完成模型 | {complete_count} / {len(models)} |",
            f"| target_field 文件完整 | {target_complete} / {len(models)} |",
            f"| target_size_mapping 匹配 | {mapping_count} / {len(models)} |",
            f"| target_curvature_exponent=0.75 | {exponent_count} / {len(models)} |",
            f"| target_smoothing_solver_success=true | {solver_count} / {len(models)} |",
            "",
        ]
    )
    if errors:
        lines.extend(["未完成模型：", "", "| Model | 原因 |", "|---|---|"])
        for model in models:
            if model in errors:
                lines.append(f"| {model} | {errors[model]} |")
        lines.append("")

    lines.extend(["## 汇总统计", "", group_summary_table(rows) if rows else "_无完整结果。_", ""])
    if rows:
        lines.extend(
            [
                "重点模型：",
                "",
                f"- 最大 CD：{top_models(rows, 'cd', 5, reverse=True)}",
                f"- 最低 Qmin：{top_models(rows, 'qmin', 5, reverse=False)}",
                f"- 最大 violation：{top_models(rows, 'iter10_violation', 5, reverse=True)}",
                f"- 最大 target_h max/min：{top_models(rows, 'h_max_min', 5, reverse=True)}",
                f"- 最大 grad_h_norm：{top_models(rows, 'max_grad_h_norm', 5, reverse=True)}",
                "",
                "观察：",
                "",
                f"- C 组平均 CD 为 `{average([row for row in rows if row['group'] == 'C'], 'cd'):.6E}`，model 组平均 CD 为 `{average([row for row in rows if row['group'] == 'model'], 'cd'):.6E}`。",
                f"- 全部模型平均输出面数/输入面数比例为 `{average(rows, 'face_ratio'):.6f}`，说明当前参数整体接近输入面数预算，但具体模型会因 split/collapse 收敛状态有偏差。",
                f"- 全部模型最低 Qmin 出现在 {top_models(rows, 'qmin', 1, reverse=False)}；该模型建议后续做局部可视化检查。",
                "",
            ]
        )

    lines.extend(["## 单模型结果", "", result_table(rows) if rows else "_无完整结果。_", ""])

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v2.11.0 C/model iter10 experiments.")
    parser.add_argument("--cli", type=Path, default=DEFAULT_CLI)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--preprocess", type=int, choices=(0, 1), default=0)
    parser.add_argument("--target-smoothing-alpha", type=float, default=2.0)
    parser.add_argument("--target-bisection-iters", type=int, default=64)
    parser.add_argument("--cd-samples", type=int, default=100000)
    parser.add_argument("--skip-runs", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--models", default="", help="Comma-separated subset such as C0,model_001.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.cli = args.cli.resolve()
    args.preprocess = bool(args.preprocess)
    if args.iters < 0:
        print("--iters must be non-negative", file=sys.stderr)
        return 2
    if args.target_smoothing_alpha <= 0.0 or not math.isfinite(args.target_smoothing_alpha):
        print("--target-smoothing-alpha must be positive and finite", file=sys.stderr)
        return 2
    if args.target_bisection_iters < 0:
        print("--target-bisection-iters must be non-negative", file=sys.stderr)
        return 2
    if args.cd_samples <= 0:
        print("--cd-samples must be positive", file=sys.stderr)
        return 2
    if not args.skip_runs and not args.cli.exists():
        print(f"missing CLI executable: {args.cli}", file=sys.stderr)
        return 2

    all_model_paths = discover_models()
    if args.models:
        selected = {item.strip() for item in args.models.split(",") if item.strip()}
        model_paths = [path for path in all_model_paths if path.stem in selected]
        missing = sorted(selected.difference(path.stem for path in model_paths), key=model_key)
        if missing:
            print(f"unknown models: {', '.join(missing)}", file=sys.stderr)
            return 2
    else:
        model_paths = all_model_paths
    models = [path.stem for path in model_paths]
    missing_inputs = expected_missing([path.stem for path in all_model_paths]) if not args.models else []
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)

    failures: dict[str, str] = {}
    if not args.skip_runs:
        for model_path in model_paths:
            model = model_path.stem
            if not args.force and case_is_complete(model, args.iters, args.cd_samples):
                print(f"[skip] {model} already complete", flush=True)
                continue
            print(f"[run] {model}", flush=True)
            try:
                run_case(model_path, args)
            except Exception as exc:  # noqa: BLE001
                failures[model] = str(exc)
                print(f"[fail] {model}: {exc}", file=sys.stderr, flush=True)

    rows, errors = collect_rows(models, args.iters, args.cd_samples)
    errors.update(failures)
    write_csv(RUN_SUMMARY_PATH, RUN_FIELDS, rows)
    write_report(models, missing_inputs, rows, errors, args)

    print(f"summary: {rel(RUN_SUMMARY_PATH)}")
    print(f"report: {rel(REPORT_PATH)}")
    if errors:
        print(f"incomplete models: {', '.join(sorted(errors, key=model_key))}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
