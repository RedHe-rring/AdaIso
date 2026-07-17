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
CPP_ROOT = ROOT / "cpp_adaiso"
EXAMPLES_ROOT = CPP_ROOT / "examples"
RESULT_ROOT = CPP_ROOT / "results" / "v5.0.0_all_series"
REPORT_PATH = CPP_ROOT / "修改计划与实验结果" / "v5.0.0实验结果.md"
RUN_SUMMARY_PATH = RESULT_ROOT / "run_summary.csv"
SERIES_SUMMARY_PATH = RESULT_ROOT / "series_summary.csv"
DEFAULT_CLI = CPP_ROOT / "build2" / "adaiso_cli.exe"

TARGET_FIELD_FILES = (
    "vertices.csv",
    "edges.csv",
    "faces.csv",
    "field_stats.json",
    "color_scales.json",
    "raw_curvature_heat_shared.ply",
    "raw_curvature_heat_independent.ply",
    "csf_curvature_heat_shared.ply",
    "csf_curvature_heat_independent.ply",
    "target_h_heat.ply",
    "target_h_heat_observed.ply",
)

RUN_FIELDS = (
    "series",
    "model",
    "input_file",
    "input_vertices",
    "input_faces",
    "output_vertices",
    "output_faces",
    "target_faces",
    "target_faces_matches_input",
    "qavg",
    "qmin",
    "iter10_violation",
    "cd",
    "symmetric_rmse",
    "runtime_sec",
    "csf_solver_success",
    "csf_residual_scaled",
    "target_size_mapping",
    "base_length",
    "h_min",
    "h_max",
    "observed_h_min",
    "observed_h_max",
    "max_edge_target_ratio",
    "p95_edge_target_ratio",
    "p99_edge_target_ratio",
    "max_grad_h_norm",
    "p95_grad_h_norm",
    "p99_grad_h_norm",
    "case_path",
    "target_field_path",
)

SERIES_FIELDS = (
    "series",
    "models",
    "completed",
    "target_field_complete",
    "finite_metrics",
    "csf_solver_success",
    "avg_qavg",
    "avg_qmin",
    "min_qmin",
    "avg_cd",
    "max_cd",
    "avg_symmetric_rmse",
    "avg_iter10_violation",
    "max_iter10_violation",
    "avg_runtime_sec",
)

EXPECTED_MAPPING = "target_faces_inverse_sqrt_diffused_curvature"
SERIES_ORDER = {"T": 0, "C": 1, "model": 2}


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


def natural_key(path_or_name: Path | str) -> tuple[str, int]:
    name = path_or_name.stem if isinstance(path_or_name, Path) else path_or_name
    if match := re.fullmatch(r"T(\d+)", name):
        return ("T", int(match.group(1)))
    if match := re.fullmatch(r"C(\d+)", name):
        return ("C", int(match.group(1)))
    if match := re.fullmatch(r"model_(\d+)", name):
        return ("model", int(match.group(1)))
    raise ValueError(f"unexpected model name: {name}")


def series_name(model: str) -> str:
    return natural_key(model)[0]


def model_sort_key(path_or_name: Path | str) -> tuple[int, int]:
    series, index = natural_key(path_or_name)
    return (SERIES_ORDER[series], index)


def discover_models() -> list[Path]:
    patterns = ("T*.off", "C*.obj", "model_*.off")
    paths: list[Path] = []
    for pattern in patterns:
        for path in EXAMPLES_ROOT.glob(pattern):
            try:
                natural_key(path)
            except ValueError:
                continue
            paths.append(path)
    return sorted(paths, key=model_sort_key)


def case_dir(model: str) -> Path:
    return RESULT_ROOT / series_name(model) / model


def target_field_dir(model: str) -> Path:
    return case_dir(model) / "target_field"


def missing_target_files(model: str) -> list[str]:
    field_dir = target_field_dir(model)
    return [name for name in TARGET_FIELD_FILES if not (field_dir / name).exists()]


def tail_violation(summary: dict) -> float:
    iterations = summary.get("iterations", [])
    if not iterations:
        return 0.0
    return float(iterations[-1].get("violation_rate", 0.0))


def read_case(model: str) -> dict:
    current_dir = case_dir(model)
    summary = load_json(current_dir / "summary.json")
    field_stats = load_json(target_field_dir(model) / "field_stats.json")

    input_summary = summary["input_summary"]
    output_summary = summary["output_summary"]
    chamfer = summary["chamfer_distance"]
    parameters = summary.get("parameters", {})
    stats = summary.get("stats", {})
    input_faces = int(input_summary["faces"])
    target_faces = int(parameters.get("target_faces", 0) or 0)

    return {
        "series": series_name(model),
        "model": model,
        "input_file": summary.get("input", ""),
        "input_vertices": int(input_summary["vertices"]),
        "input_faces": input_faces,
        "output_vertices": int(output_summary["vertices"]),
        "output_faces": int(output_summary["faces"]),
        "target_faces": target_faces,
        "target_faces_matches_input": target_faces == input_faces,
        "qavg": float(output_summary["qavg"]),
        "qmin": float(output_summary["qmin"]),
        "iter10_violation": tail_violation(summary),
        "cd": float(chamfer["chamfer_distance"]),
        "symmetric_rmse": float(chamfer["symmetric_rmse"]),
        "runtime_sec": float(summary["runtime_sec"]),
        "csf_solver_success": bool(stats.get("csf_solver_success", False)),
        "csf_residual_scaled": float(stats.get("csf_residual_scaled", 0.0)),
        "target_size_mapping": parameters.get("target_size_mapping", ""),
        "base_length": float(field_stats["base_length"]),
        "h_min": float(field_stats["h_min"]),
        "h_max": float(field_stats["h_max"]),
        "observed_h_min": float(field_stats["observed_h_min"]),
        "observed_h_max": float(field_stats["observed_h_max"]),
        "max_edge_target_ratio": float(field_stats["max_edge_target_ratio"]),
        "p95_edge_target_ratio": float(field_stats["p95_edge_target_ratio"]),
        "p99_edge_target_ratio": float(field_stats["p99_edge_target_ratio"]),
        "max_grad_h_norm": float(field_stats["max_grad_h_norm"]),
        "p95_grad_h_norm": float(field_stats["p95_grad_h_norm"]),
        "p99_grad_h_norm": float(field_stats["p99_grad_h_norm"]),
        "case_path": rel(current_dir),
        "target_field_path": rel(target_field_dir(model)),
    }


def row_has_finite_metrics(row: dict) -> bool:
    keys = (
        "qavg",
        "qmin",
        "iter10_violation",
        "cd",
        "symmetric_rmse",
        "runtime_sec",
        "max_edge_target_ratio",
        "max_grad_h_norm",
    )
    return all(finite(row[key]) for key in keys)


def case_is_complete(model: str) -> bool:
    current_dir = case_dir(model)
    if not (current_dir / "output.ply").exists():
        return False
    if not (current_dir / "summary.json").exists():
        return False
    if missing_target_files(model):
        return False
    try:
        row = read_case(model)
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        return False
    return row["target_size_mapping"] == EXPECTED_MAPPING and row_has_finite_metrics(row)


def write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def average(rows: list[dict], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def count_where(rows: list[dict], predicate) -> int:
    return sum(1 for row in rows if predicate(row))


def top_models(rows: list[dict], key: str, count: int, reverse: bool = True) -> str:
    selected = sorted(rows, key=lambda row: row[key], reverse=reverse)[:count]
    return ", ".join(f"`{row['model']}` ({row[key]:.6g})" for row in selected)


def run_result_table(rows: list[dict]) -> str:
    lines = [
        "| Series | Model | In F | Out V/F | target_faces | Qavg | Qmin | viol | CD | RMSE | max edge ratio | p95 ratio | max grad | time(s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["series"],
                    row["model"],
                    str(row["input_faces"]),
                    f"{row['output_vertices']}/{row['output_faces']}",
                    str(row["target_faces"]),
                    fmt(row["qavg"]),
                    fmt(row["qmin"]),
                    fmt(row["iter10_violation"]),
                    sci(row["cd"]),
                    sci(row["symmetric_rmse"]),
                    fmt(row["max_edge_target_ratio"]),
                    fmt(row["p95_edge_target_ratio"]),
                    fmt(row["max_grad_h_norm"]),
                    fmt(row["runtime_sec"], 2),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def series_summary_rows(all_models: list[str], rows: list[dict]) -> list[dict]:
    result = []
    rows_by_series = {
        series: [row for row in rows if row["series"] == series]
        for series in ("T", "C", "model")
    }
    model_count_by_series = {
        series: count_where([{"model": model} for model in all_models], lambda item, s=series: series_name(item["model"]) == s)
        for series in ("T", "C", "model")
    }
    for series in ("T", "C", "model"):
        current = rows_by_series[series]
        if not current:
            result.append(
                {
                    "series": series,
                    "models": model_count_by_series[series],
                    "completed": 0,
                    "target_field_complete": 0,
                    "finite_metrics": 0,
                    "csf_solver_success": 0,
                    "avg_qavg": "",
                    "avg_qmin": "",
                    "min_qmin": "",
                    "avg_cd": "",
                    "max_cd": "",
                    "avg_symmetric_rmse": "",
                    "avg_iter10_violation": "",
                    "max_iter10_violation": "",
                    "avg_runtime_sec": "",
                }
            )
            continue
        result.append(
            {
                "series": series,
                "models": model_count_by_series[series],
                "completed": len(current),
                "target_field_complete": count_where(current, lambda row: not missing_target_files(row["model"])),
                "finite_metrics": count_where(current, row_has_finite_metrics),
                "csf_solver_success": count_where(current, lambda row: row["csf_solver_success"]),
                "avg_qavg": average(current, "qavg"),
                "avg_qmin": average(current, "qmin"),
                "min_qmin": min(row["qmin"] for row in current),
                "avg_cd": average(current, "cd"),
                "max_cd": max(row["cd"] for row in current),
                "avg_symmetric_rmse": average(current, "symmetric_rmse"),
                "avg_iter10_violation": average(current, "iter10_violation"),
                "max_iter10_violation": max(row["iter10_violation"] for row in current),
                "avg_runtime_sec": average(current, "runtime_sec"),
            }
        )
    return result


def series_summary_table(rows: list[dict]) -> str:
    lines = [
        "| Series | Done | target_field | finite | CSF ok | avg Qavg | avg Qmin | min Qmin | avg CD | max CD | avg viol | max viol | avg time(s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if not row["completed"]:
            lines.append(f"| {row['series']} | 0/{row['models']} | 0 | 0 | 0 | - | - | - | - | - | - | - | - |")
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    row["series"],
                    f"{row['completed']}/{row['models']}",
                    str(row["target_field_complete"]),
                    str(row["finite_metrics"]),
                    str(row["csf_solver_success"]),
                    fmt(row["avg_qavg"]),
                    fmt(row["avg_qmin"]),
                    fmt(row["min_qmin"]),
                    sci(row["avg_cd"]),
                    sci(row["max_cd"]),
                    fmt(row["avg_iter10_violation"]),
                    fmt(row["max_iter10_violation"]),
                    fmt(row["avg_runtime_sec"], 2),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def visual_field_table(models: list[str]) -> str:
    lines = [
        "| Series | Model | target_field path | files |",
        "|---|---|---|---:|",
    ]
    for model in models:
        missing = missing_target_files(model)
        complete = len(TARGET_FIELD_FILES) - len(missing)
        lines.append(f"| {series_name(model)} | {model} | `{rel(target_field_dir(model))}/` | {complete}/{len(TARGET_FIELD_FILES)} |")
    return "\n".join(lines)


def collect_rows(models: list[str]) -> tuple[list[dict], dict[str, str]]:
    rows: list[dict] = []
    errors: dict[str, str] = {}
    for model in models:
        if not case_is_complete(model):
            missing = missing_target_files(model)
            if missing:
                errors[model] = "missing target_field files: " + ", ".join(missing)
            else:
                errors[model] = "summary/output/metrics incomplete"
            continue
        try:
            rows.append(read_case(model))
        except Exception as exc:  # noqa: BLE001 - keep collecting other models.
            errors[model] = str(exc)
    rows.sort(key=lambda row: model_sort_key(row["model"]))
    return rows, errors


def run_case(model_path: Path, args: argparse.Namespace) -> None:
    model = model_path.stem
    current_dir = case_dir(model)
    current_dir.mkdir(parents=True, exist_ok=True)
    field_dir = target_field_dir(model)
    output_path = current_dir / "output.ply"
    summary_path = current_dir / "summary.json"

    command = [
        str(args.cli),
        rel_arg(model_path),
        "-o",
        rel_arg(output_path),
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
        rel_arg(field_dir),
        "--summary-json",
        rel_arg(summary_path),
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
    log = [
        "$ " + " ".join(command),
        "",
        f"returncode: {proc.returncode}",
        f"wall_time_sec: {elapsed:.3f}",
        "",
        "--- stdout ---",
        proc.stdout,
        "",
        "--- stderr ---",
        proc.stderr,
    ]
    (current_dir / "run.log").write_text("\n".join(log), encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"{model} failed with exit code {proc.returncode}; see {rel(current_dir / 'run.log')}")


def write_report(models: list[str], rows: list[dict], errors: dict[str, str], args: argparse.Namespace) -> None:
    series_rows = series_summary_rows(models, rows)
    completed_count = len(rows)
    target_complete_count = sum(1 for model in models if not missing_target_files(model))
    mapping_count = count_where(rows, lambda row: row["target_size_mapping"] == EXPECTED_MAPPING)
    target_faces_default_count = count_where(rows, lambda row: row["target_faces_matches_input"])
    csf_success_count = count_where(rows, lambda row: row["csf_solver_success"])
    finite_count = count_where(rows, row_has_finite_metrics)

    lines = [
        "# v5.0.0 实验结果",
        "",
        "## 实验设置",
        "",
        "本实验对 `cpp_adaiso/examples` 下的 T、C、model 三个系列运行 v5.0.0 当前实现。输出包含重网格结果、target-field 可视化文件、Chamfer Distance、symmetric RMSE、网格质量和迭代末 violation 指标。",
        "",
        "| 参数 | 值 |",
        "|---|---:|",
        f"| 模型总数 | {len(models)} |",
        f"| T 系列 | {count_where([{'model': model} for model in models], lambda item: series_name(item['model']) == 'T')} |",
        f"| C 系列 | {count_where([{'model': model} for model in models], lambda item: series_name(item['model']) == 'C')} |",
        f"| model 系列 | {count_where([{'model': model} for model in models], lambda item: series_name(item['model']) == 'model')} |",
        f"| iters | {args.iters} |",
        "| preprocess | 0 |",
        f"| curvature_mode | {args.curvature_mode} |",
        f"| csf_alpha | {args.csf_alpha} |",
        f"| cd-samples | {args.cd_samples} |",
        "| target_faces | 默认：清理后输入面数 |",
        "",
        f"- 输出目录：`{rel(RESULT_ROOT)}/`",
        f"- 汇总 CSV：`{rel(RUN_SUMMARY_PATH)}`",
        f"- 分系列 CSV：`{rel(SERIES_SUMMARY_PATH)}`",
        "",
        "命令模板：",
        "",
        "```powershell",
        "cpp_adaiso/build2/adaiso_cli.exe cpp_adaiso/examples/<model> `",
        "  -o cpp_adaiso/results/v5.0.0_all_series/<series>/<model>/output.ply `",
        f"  --iters {args.iters} --preprocess 0 `",
        f"  --curvature-mode {args.curvature_mode} --csf-alpha {args.csf_alpha} `",
        f"  --gt cpp_adaiso/examples/<model> --cd-samples {args.cd_samples} `",
        "  --export-target-field cpp_adaiso/results/v5.0.0_all_series/<series>/<model>/target_field `",
        "  --summary-json cpp_adaiso/results/v5.0.0_all_series/<series>/<model>/summary.json `",
        "  --quiet",
        "```",
        "",
        "## 完整性检查",
        "",
        "| 检查项 | 结果 |",
        "|---|---:|",
        f"| 模型总数 | {len(models)} |",
        f"| 完成模型 | {completed_count} / {len(models)} |",
        f"| target_field 文件完整 | {target_complete_count} / {len(models)} |",
        f"| target_size_mapping 匹配 | {mapping_count} / {len(models)} |",
        f"| target_faces 等于输入面数 | {target_faces_default_count} / {len(models)} |",
        f"| csf_solver_success=true | {csf_success_count} / {len(models)} |",
        f"| 指标 finite | {finite_count} / {len(models)} |",
        "",
        "`target_field/` 应包含：",
        "",
        "```text",
        "\n".join(TARGET_FIELD_FILES),
        "```",
        "",
    ]

    if errors:
        lines.extend(["未完成模型：", "", "| Series | Model | 原因 |", "|---|---|---|"])
        for model in models:
            if model in errors:
                lines.append(f"| {series_name(model)} | {model} | {errors[model]} |")
        lines.append("")

    lines.extend(
        [
            "## 分系列汇总",
            "",
            series_summary_table(series_rows),
            "",
            "## CD 尺度说明",
            "",
            "本报告中的 `CD` 和 `symmetric RMSE` 是未归一化的绝对坐标尺度指标。当前实现按面积采样计算双向最近点平方距离，并令 `chamfer_distance = output_to_gt_msd + gt_to_output_msd`、`symmetric_rmse = sqrt(0.5 * chamfer_distance)`；没有除以包围盒、平均边长或表面积尺度。",
            "",
            "因此不同系列之间的 CD 不宜直接横向比较。模型坐标整体放大 `k` 倍时，`symmetric RMSE` 约放大 `k` 倍，`CD` 约放大 `k^2` 倍。本次结果中 T 系列的 `base_length` 多在 0.008-0.030，而 C/model 系列常在 0.2-3.0，C/model 的绝对 CD 显著大于 T 系列主要受模型尺度差异影响，并不必然表示重网格质量更差。",
            "",
            "后续若要跨系列比较几何误差，建议补充归一化指标，例如 `symmetric_rmse / base_length` 和 `chamfer_distance / (base_length^2)`，或使用包围盒对角线归一化后的 RMSE/CD。",
            "",
            "## 每模型结果",
            "",
            run_result_table(rows) if rows else "_没有可汇总的完整结果。_",
            "",
        ]
    )

    if rows:
        lines.extend(
            [
                "## 总体统计",
                "",
                "| 指标 | v5.0.0 |",
                "|---|---:|",
                f"| 平均 Qavg | {average(rows, 'qavg'):.10f} |",
                f"| 平均 Qmin | {average(rows, 'qmin'):.10f} |",
                f"| 最低 Qmin | {min(row['qmin'] for row in rows):.10f} |",
                f"| 平均 CD | {average(rows, 'cd'):.10E} |",
                f"| 最大 CD | {max(row['cd'] for row in rows):.10E} |",
                f"| 平均 symmetric RMSE | {average(rows, 'symmetric_rmse'):.10E} |",
                f"| 平均 iter10 violation | {average(rows, 'iter10_violation'):.10f} |",
                f"| 最高 iter10 violation | {max(row['iter10_violation'] for row in rows):.10f} |",
                f"| 平均 runtime | {average(rows, 'runtime_sec'):.3f} s |",
                "",
                "Worst-case：",
                "",
                f"- 最低 Qmin：{top_models(rows, 'qmin', 5, reverse=False)}",
                f"- 最大 CD：{top_models(rows, 'cd', 5, reverse=True)}",
                f"- 最高 violation：{top_models(rows, 'iter10_violation', 5, reverse=True)}",
                f"- 最大 edge-target ratio：{top_models(rows, 'max_edge_target_ratio', 5, reverse=True)}",
                f"- 最大 target_h 梯度：{top_models(rows, 'max_grad_h_norm', 5, reverse=True)}",
                "",
            ]
        )

    lines.extend(
        [
            "## 可视化输出",
            "",
            "`target_field/` 中的 PLY 文件可直接用于查看 raw curvature、CSF curvature、target_h 和 observed target_h 的热力图；CSV 文件保存顶点、边、面级诊断数据。",
            "",
            visual_field_table(models),
            "",
        ]
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v5.0.0 T/C/model series iter10 experiments.")
    parser.add_argument("--cli", type=Path, default=DEFAULT_CLI)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--cd-samples", type=int, default=100000)
    parser.add_argument("--curvature-mode", default="robust-mean")
    parser.add_argument("--csf-alpha", type=float, default=2.0)
    parser.add_argument("--skip-runs", action="store_true", help="Only summarize existing outputs.")
    parser.add_argument("--force", action="store_true", help="Rerun every selected model even if complete.")
    parser.add_argument("--series", default="T,C,model", help="Comma-separated subset of T,C,model.")
    parser.add_argument("--models", default="", help="Comma-separated subset such as T0,C0,model_001.")
    return parser.parse_args()


def selected_model_paths(args: argparse.Namespace) -> list[Path]:
    all_paths = discover_models()
    selected_series = {item.strip() for item in args.series.split(",") if item.strip()}
    valid_series = {"T", "C", "model"}
    invalid_series = selected_series.difference(valid_series)
    if invalid_series:
        raise ValueError(f"unknown series: {', '.join(sorted(invalid_series))}")
    paths = [path for path in all_paths if series_name(path.stem) in selected_series]
    if args.models:
        selected_models = {item.strip() for item in args.models.split(",") if item.strip()}
        paths = [path for path in paths if path.stem in selected_models]
        missing = sorted(selected_models.difference(path.stem for path in paths), key=model_sort_key)
        if missing:
            raise ValueError(f"unknown models: {', '.join(missing)}")
    return sorted(paths, key=model_sort_key)


def main() -> int:
    args = parse_args()
    args.cli = args.cli.resolve()
    if not args.skip_runs and not args.cli.exists():
        print(f"missing CLI executable: {args.cli}", file=sys.stderr)
        return 2

    try:
        model_paths = selected_model_paths(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    models = [path.stem for path in model_paths]
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)

    failures: dict[str, str] = {}
    if not args.skip_runs:
        for index, model_path in enumerate(model_paths, start=1):
            model = model_path.stem
            if not args.force and case_is_complete(model):
                print(f"[skip {index}/{len(model_paths)}] {model} already complete", flush=True)
                continue
            print(f"[run {index}/{len(model_paths)}] {model}", flush=True)
            try:
                run_case(model_path, args)
            except Exception as exc:  # noqa: BLE001 - continue to summarize completed cases.
                failures[model] = str(exc)
                print(f"[fail] {model}: {exc}", file=sys.stderr, flush=True)

    rows, errors = collect_rows(models)
    errors.update(failures)
    series_rows = series_summary_rows(models, rows)
    write_csv(RUN_SUMMARY_PATH, RUN_FIELDS, rows)
    write_csv(SERIES_SUMMARY_PATH, SERIES_FIELDS, series_rows)
    write_report(models, rows, errors, args)

    print(f"summary: {rel(RUN_SUMMARY_PATH)}")
    print(f"series: {rel(SERIES_SUMMARY_PATH)}")
    print(f"report: {rel(REPORT_PATH)}")
    if errors:
        print(f"incomplete models: {', '.join(sorted(errors, key=model_sort_key))}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
