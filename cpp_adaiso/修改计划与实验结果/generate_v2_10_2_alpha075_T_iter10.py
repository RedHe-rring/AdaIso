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


ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = ROOT / "cpp_adaiso" / "examples"
RESULT_ROOT = ROOT / "cpp_adaiso" / "results" / "v2.10.2_alpha075" / "T_all_iter10"
BASELINE_ROOT = ROOT / "cpp_adaiso" / "results" / "v2.10.0_target_faces" / "T_all_iter10"
REPORT_PATH = ROOT / "cpp_adaiso" / "修改计划与实验结果" / "v2.10.2 T系列 alpha075 iter10实验报告.md"
RUN_SUMMARY_PATH = RESULT_ROOT / "run_summary.csv"
COMPARE_SUMMARY_PATH = RESULT_ROOT / "comparison_vs_v2.10.0.csv"
DEFAULT_CLI = ROOT / "cpp_adaiso" / "build" / "adaiso_cli.exe"

EXPECTED_MAPPING = "target_faces_inverse_power_diffused_curvature"
EXPECTED_EXPONENT = 0.75
BASELINE_MAPPING = "target_faces_inverse_sqrt_diffused_curvature"

CURRENT_TARGET_FIELD_FILES = (
    "raw_curvature_heat_shared.ply",
    "csf_curvature_heat_shared.ply",
    "raw_curvature_heat_independent.ply",
    "csf_curvature_heat_independent.ply",
    "target_h_heat.ply",
    "target_h_heat_observed.ply",
    "vertices.csv",
    "edges.csv",
    "faces.csv",
    "field_stats.json",
    "color_scales.json",
)

BASELINE_TARGET_FIELD_FILES = tuple(
    name for name in CURRENT_TARGET_FIELD_FILES if name != "target_h_heat_observed.ply"
)

RUN_FIELDS = (
    "model",
    "input_faces",
    "output_faces",
    "target_faces",
    "qavg",
    "qmin",
    "iter10_violation",
    "cd",
    "symmetric_rmse",
    "runtime_sec",
    "target_size_mapping",
    "target_curvature_exponent",
    "csf_solver_success",
    "observed_h_min",
    "h_p05",
    "h_p50",
    "h_p95",
    "observed_h_max",
    "h_p95_p05",
    "h_max_min",
    "max_edge_target_ratio",
    "p95_edge_target_ratio",
    "max_grad_h_norm",
)

COMPARE_FIELDS = (
    "model",
    "v2102_cd",
    "v2100_cd",
    "delta_cd",
    "delta_cd_pct",
    "cd_lower",
    "v2102_qmin",
    "v2100_qmin",
    "delta_qmin",
    "v2102_violation",
    "v2100_violation",
    "delta_violation",
    "v2102_h_p95_p05",
    "v2100_h_p95_p05",
    "delta_h_p95_p05",
    "v2102_h_max_min",
    "v2100_h_max_min",
    "delta_h_max_min",
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


def discover_models() -> list[Path]:
    models = []
    for path in EXAMPLES_ROOT.glob("T*.off"):
        try:
            natural_model_key(path)
        except ValueError:
            continue
        models.append(path)
    return sorted(models, key=natural_model_key)


def case_dir(root: Path, model: str) -> Path:
    return root / model


def target_files_for(root: Path) -> tuple[str, ...]:
    return CURRENT_TARGET_FIELD_FILES if root == RESULT_ROOT else BASELINE_TARGET_FIELD_FILES


def missing_target_files(root: Path, model: str) -> list[str]:
    field_dir = case_dir(root, model) / "target_field"
    return [name for name in target_files_for(root) if not (field_dir / name).exists()]


def quantile(sorted_values: list[float], t: float) -> float:
    if not sorted_values:
        return 0.0
    pos = (len(sorted_values) - 1) * t
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[lo]
    return sorted_values[lo] * (hi - pos) + sorted_values[hi] * (pos - lo)


def target_h_quantiles(root: Path, model: str) -> dict[str, float]:
    path = case_dir(root, model) / "target_field" / "vertices.csv"
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


def read_case(root: Path, model: str) -> dict:
    current_dir = case_dir(root, model)
    summary = load_json(current_dir / "summary.json")
    field_stats = load_json(current_dir / "target_field" / "field_stats.json")
    h_stats = target_h_quantiles(root, model)

    input_summary = summary["input_summary"]
    output_summary = summary["output_summary"]
    chamfer = summary["chamfer_distance"]
    parameters = summary.get("parameters", {})
    stats = summary.get("stats", {})

    row = {
        "model": model,
        "input_faces": int(input_summary["faces"]),
        "output_faces": int(output_summary["faces"]),
        "target_faces": int(parameters.get("target_faces", 0) or 0),
        "qavg": float(output_summary["qavg"]),
        "qmin": float(output_summary["qmin"]),
        "iter10_violation": tail_violation(summary),
        "cd": float(chamfer["chamfer_distance"]),
        "symmetric_rmse": float(chamfer["symmetric_rmse"]),
        "runtime_sec": float(summary["runtime_sec"]),
        "target_size_mapping": parameters.get("target_size_mapping", ""),
        "target_curvature_exponent": float(parameters.get("target_curvature_exponent", 0.5)),
        "csf_solver_success": bool(stats.get("csf_solver_success", False)),
        "max_edge_target_ratio": float(field_stats["max_edge_target_ratio"]),
        "p95_edge_target_ratio": float(field_stats["p95_edge_target_ratio"]),
        "max_grad_h_norm": float(field_stats["max_grad_h_norm"]),
    }
    row.update(h_stats)
    return row


def case_is_complete(root: Path, model: str, expected_mapping: str | None) -> bool:
    current_dir = case_dir(root, model)
    if root == RESULT_ROOT and not (current_dir / "output.ply").exists():
        return False
    if not (current_dir / "summary.json").exists():
        return False
    if missing_target_files(root, model):
        return False
    try:
        row = read_case(root, model)
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        return False
    if expected_mapping is not None and row["target_size_mapping"] != expected_mapping:
        return False
    if root == RESULT_ROOT and abs(row["target_curvature_exponent"] - EXPECTED_EXPONENT) > 1e-12:
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
        )
    )


def collect_rows(models: list[str], root: Path, expected_mapping: str | None) -> tuple[list[dict], dict[str, str]]:
    rows: list[dict] = []
    errors: dict[str, str] = {}
    for model in models:
        if not case_is_complete(root, model, expected_mapping):
            missing = missing_target_files(root, model)
            errors[model] = "missing target-field files: " + ",".join(missing) if missing else "incomplete summary/metrics"
            continue
        rows.append(read_case(root, model))
    rows.sort(key=lambda row: natural_model_key(row["model"]))
    return rows, errors


def run_case(model_path: Path, args: argparse.Namespace) -> None:
    model = model_path.stem
    current_dir = case_dir(RESULT_ROOT, model)
    current_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(args.cli),
        rel_arg(model_path),
        "-o",
        rel_arg(current_dir / "output.ply"),
        "--iters",
        str(args.iters),
        "--preprocess",
        "0",
        "--curvature-mode",
        args.curvature_mode,
        "--csf-alpha",
        str(args.csf_alpha),
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


def compare_rows(current: dict, baseline: dict) -> dict:
    delta_cd = current["cd"] - baseline["cd"]
    delta_cd_pct = 100.0 * delta_cd / baseline["cd"] if baseline["cd"] else 0.0
    return {
        "model": current["model"],
        "v2102_cd": current["cd"],
        "v2100_cd": baseline["cd"],
        "delta_cd": delta_cd,
        "delta_cd_pct": delta_cd_pct,
        "cd_lower": delta_cd < 0.0,
        "v2102_qmin": current["qmin"],
        "v2100_qmin": baseline["qmin"],
        "delta_qmin": current["qmin"] - baseline["qmin"],
        "v2102_violation": current["iter10_violation"],
        "v2100_violation": baseline["iter10_violation"],
        "delta_violation": current["iter10_violation"] - baseline["iter10_violation"],
        "v2102_h_p95_p05": current["h_p95_p05"],
        "v2100_h_p95_p05": baseline["h_p95_p05"],
        "delta_h_p95_p05": current["h_p95_p05"] - baseline["h_p95_p05"],
        "v2102_h_max_min": current["h_max_min"],
        "v2100_h_max_min": baseline["h_max_min"],
        "delta_h_max_min": current["h_max_min"] - baseline["h_max_min"],
    }


def average(rows: list[dict], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def count_where(rows: list[dict], predicate) -> int:
    return sum(1 for row in rows if predicate(row))


def top_models(rows: list[dict], key: str, count: int, reverse: bool = True) -> str:
    selected = sorted(rows, key=lambda row: row[key], reverse=reverse)[:count]
    return ", ".join(f"`{row['model']}` ({row[key]:.6g})" for row in selected)


def result_table(rows: list[dict]) -> str:
    lines = [
        "| Model | Out F | Qavg | Qmin | viol | CD | RMSE | h p95/p05 | h max/min | max edge ratio | max grad |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["model"],
                    str(row["output_faces"]),
                    fmt(row["qavg"]),
                    fmt(row["qmin"]),
                    fmt(row["iter10_violation"]),
                    sci(row["cd"]),
                    sci(row["symmetric_rmse"]),
                    fmt(row["h_p95_p05"]),
                    fmt(row["h_max_min"]),
                    fmt(row["max_edge_target_ratio"]),
                    fmt(row["max_grad_h_norm"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def comparison_table(rows: list[dict]) -> str:
    lines = [
        "| Model | dCD% | CD降低 | dQmin | dviol | d h p95/p05 | d h max/min |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["model"],
                    signed_pct(row["delta_cd_pct"]),
                    "yes" if row["cd_lower"] else "no",
                    signed(row["delta_qmin"]),
                    signed(row["delta_violation"]),
                    signed(row["delta_h_p95_p05"]),
                    signed(row["delta_h_max_min"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def write_report(
    models: list[str],
    current_rows: list[dict],
    current_errors: dict[str, str],
    baseline_rows: list[dict],
    baseline_errors: dict[str, str],
    comparisons: list[dict],
    args: argparse.Namespace,
) -> None:
    model_count = len(models)
    current_complete = len(current_rows)
    baseline_complete = len(baseline_rows)
    target_complete = sum(1 for model in models if not missing_target_files(RESULT_ROOT, model))
    mapping_count = count_where(current_rows, lambda row: row["target_size_mapping"] == EXPECTED_MAPPING)
    exponent_count = count_where(current_rows, lambda row: abs(row["target_curvature_exponent"] - EXPECTED_EXPONENT) <= 1e-12)
    csf_count = count_where(current_rows, lambda row: row["csf_solver_success"])

    lines = [
        "# v2.10.2 T系列 alpha075 iter10实验报告",
        "",
        "## 实验设置",
        "",
        "v2.10.2 保留 v2.10.1 的 logless robust-mean 曲率信号，并将目标尺寸公式从 `1/sqrt(kappa)` 改为内部固定指数 `1/pow(kappa, 0.75)`。`target_faces` 仍默认使用清理后输入 mesh 面数。",
        "",
        "| 参数 | 值 |",
        "|---|---:|",
        f"| 模型数量 | {model_count} |",
        f"| iters | {args.iters} |",
        "| preprocess | 0 |",
        f"| curvature_mode | {args.curvature_mode} |",
        f"| csf_alpha | {args.csf_alpha} |",
        f"| cd-samples | {args.cd_samples} |",
        f"| target_size_mapping | {EXPECTED_MAPPING} |",
        f"| target_curvature_exponent | {EXPECTED_EXPONENT} |",
        "",
        f"- 输出目录：`{rel(RESULT_ROOT)}/`",
        f"- 汇总 CSV：`{rel(RUN_SUMMARY_PATH)}`",
        f"- 对比 CSV：`{rel(COMPARE_SUMMARY_PATH)}`",
        f"- v2.10.0 基线：`{rel(BASELINE_ROOT)}/`",
        "",
        "## 完整性检查",
        "",
        "| 检查项 | 结果 |",
        "|---|---:|",
        f"| v2.10.2 完成模型 | {current_complete} / {model_count} |",
        f"| target_field 文件完整 | {target_complete} / {model_count} |",
        f"| target_size_mapping 匹配 | {mapping_count} / {model_count} |",
        f"| target_curvature_exponent=0.75 | {exponent_count} / {model_count} |",
        f"| csf_solver_success=true | {csf_count} / {model_count} |",
        f"| v2.10.0 基线完整 | {baseline_complete} / {model_count} |",
        "",
    ]
    if current_errors:
        lines.extend(["未完成 v2.10.2 模型：", "", "| Model | 原因 |", "|---|---|"])
        for model in models:
            if model in current_errors:
                lines.append(f"| {model} | {current_errors[model]} |")
        lines.append("")
    if baseline_errors:
        lines.extend(["缺失 v2.10.0 基线：", "", "| Model | 原因 |", "|---|---|"])
        for model in models:
            if model in baseline_errors:
                lines.append(f"| {model} | {baseline_errors[model]} |")
        lines.append("")

    lines.extend(["## v2.10.2 结果", "", result_table(current_rows) if current_rows else "_无完整结果。_", ""])
    if current_rows:
        lines.extend(
            [
                "汇总统计：",
                "",
                "| 指标 | v2.10.2 |",
                "|---|---:|",
                f"| 平均 CD | {average(current_rows, 'cd'):.10E} |",
                f"| 最大 CD | {max(row['cd'] for row in current_rows):.10E} |",
                f"| 平均 RMSE | {average(current_rows, 'symmetric_rmse'):.10E} |",
                f"| 平均 Qavg | {average(current_rows, 'qavg'):.10f} |",
                f"| 平均 Qmin | {average(current_rows, 'qmin'):.10f} |",
                f"| 最低 Qmin | {min(row['qmin'] for row in current_rows):.10f} |",
                f"| 平均 violation | {average(current_rows, 'iter10_violation'):.10f} |",
                f"| 平均 h p95/p05 | {average(current_rows, 'h_p95_p05'):.10f} |",
                f"| 平均 h max/min | {average(current_rows, 'h_max_min'):.10f} |",
                f"| 平均 runtime | {average(current_rows, 'runtime_sec'):.3f} s |",
                "",
                f"- 最大 CD：{top_models(current_rows, 'cd', 5, reverse=True)}",
                f"- 最低 Qmin：{top_models(current_rows, 'qmin', 5, reverse=False)}",
                f"- 最大 target_h max/min：{top_models(current_rows, 'h_max_min', 5, reverse=True)}",
                f"- 最大 grad_h_norm：{top_models(current_rows, 'max_grad_h_norm', 5, reverse=True)}",
                "",
            ]
        )

    if len(comparisons) == model_count:
        cd_lower = count_where(comparisons, lambda row: row["cd_lower"])
        cd_higher = model_count - cd_lower
        avg_delta_cd_pct = average(comparisons, "delta_cd_pct")
        avg_delta_qmin = average(comparisons, "delta_qmin")
        avg_delta_viol = average(comparisons, "delta_violation")
        avg_delta_h_p95 = average(comparisons, "delta_h_p95_p05")
        avg_delta_h_max = average(comparisons, "delta_h_max_min")
        conclusion = (
            "CD 相较 v2.10.0 整体降低"
            if avg_delta_cd_pct < 0.0 and cd_lower > cd_higher
            else "CD 相较 v2.10.0 没有整体降低"
        )
        lines.extend(
            [
                "## 与 v2.10.0 的 CD 对比",
                "",
                f"结论：**{conclusion}**。v2.10.2 有 {cd_lower}/{model_count} 个模型 CD 低于 v2.10.0，平均 dCD% 为 `{signed_pct(avg_delta_cd_pct)}`。",
                "",
                comparison_table(comparisons),
                "",
                "对比统计：",
                "",
                "| 指标 | v2.10.2 - v2.10.0 |",
                "|---|---:|",
                f"| CD 降低模型数 | {cd_lower} / {model_count} |",
                f"| CD 升高模型数 | {cd_higher} / {model_count} |",
                f"| 平均 dCD% | {signed_pct(avg_delta_cd_pct)} |",
                f"| 平均 dQmin | {signed(avg_delta_qmin)} |",
                f"| 平均 dviolation | {signed(avg_delta_viol)} |",
                f"| 平均 d h p95/p05 | {signed(avg_delta_h_p95)} |",
                f"| 平均 d h max/min | {signed(avg_delta_h_max)} |",
                "",
            ]
        )
    else:
        lines.extend(["## 与 v2.10.0 的 CD 对比", "", "基线或当前结果不完整，未能给出全量 CD 结论。", ""])

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v2.10.2 alpha=0.75 T-series iter10 experiments.")
    parser.add_argument("--cli", type=Path, default=DEFAULT_CLI)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--cd-samples", type=int, default=100000)
    parser.add_argument("--curvature-mode", default="robust-mean")
    parser.add_argument("--csf-alpha", type=float, default=2.0)
    parser.add_argument("--skip-runs", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--models", default="", help="Comma-separated subset such as T0,T21.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.cli = args.cli.resolve()
    if not args.skip_runs and not args.cli.exists():
        print(f"missing CLI executable: {args.cli}", file=sys.stderr)
        return 2

    all_model_paths = discover_models()
    if args.models:
        selected = {item.strip() for item in args.models.split(",") if item.strip()}
        model_paths = [path for path in all_model_paths if path.stem in selected]
        missing = sorted(selected.difference(path.stem for path in model_paths), key=natural_model_key)
        if missing:
            print(f"unknown models: {', '.join(missing)}", file=sys.stderr)
            return 2
    else:
        model_paths = all_model_paths
    models = [path.stem for path in model_paths]
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)

    failures: dict[str, str] = {}
    if not args.skip_runs:
        for model_path in model_paths:
            model = model_path.stem
            if not args.force and case_is_complete(RESULT_ROOT, model, EXPECTED_MAPPING):
                print(f"[skip] {model} already complete")
                continue
            print(f"[run] {model}")
            try:
                run_case(model_path, args)
            except Exception as exc:  # noqa: BLE001
                failures[model] = str(exc)
                print(f"[fail] {model}: {exc}", file=sys.stderr)

    current_rows, current_errors = collect_rows(models, RESULT_ROOT, EXPECTED_MAPPING)
    current_errors.update(failures)
    baseline_rows, baseline_errors = collect_rows(models, BASELINE_ROOT, BASELINE_MAPPING)

    current_by_model = {row["model"]: row for row in current_rows}
    baseline_by_model = {row["model"]: row for row in baseline_rows}
    comparisons = [
        compare_rows(current_by_model[model], baseline_by_model[model])
        for model in models
        if model in current_by_model and model in baseline_by_model
    ]

    write_csv(RUN_SUMMARY_PATH, RUN_FIELDS, current_rows)
    write_csv(COMPARE_SUMMARY_PATH, COMPARE_FIELDS, comparisons)
    write_report(models, current_rows, current_errors, baseline_rows, baseline_errors, comparisons, args)

    print(f"summary: {rel(RUN_SUMMARY_PATH)}")
    print(f"comparison: {rel(COMPARE_SUMMARY_PATH)}")
    print(f"report: {rel(REPORT_PATH)}")
    if current_errors:
        print(f"incomplete models: {', '.join(sorted(current_errors, key=natural_model_key))}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
