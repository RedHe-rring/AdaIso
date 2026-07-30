from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


HELPER_PATH = Path(__file__).with_name("generate_v5_1_3_hysteresis_probe.py")
SPEC = importlib.util.spec_from_file_location("v5_1_3_hysteresis_probe", HELPER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"failed to load helper script: {HELPER_PATH}")
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)

ROOT = helper.ROOT
CPP_ROOT = helper.CPP_ROOT
EXAMPLES_ROOT = helper.EXAMPLES_ROOT
RESULT_ROOT = CPP_ROOT / "results" / "v5.1.5_T_series_auto_delta_q50"
REPORT_PATH = CPP_ROOT / "修改计划与实验结果" / "v5.1.5 T系列尺寸场实验报告.md"
BASELINE_CSV = CPP_ROOT / "results" / "v5.1.3_T_model_best_params_full" / "run_summary.csv"
DEFAULT_CLI = helper.DEFAULT_CLI

FIXED_SPLIT_FACTOR = 1.2
FIXED_COLLAPSE_FACTOR = 0.6
FIXED_H_MAX_FACTOR = 3.0
AUTO_DELTA_QUANTILE = 0.5

CSV_FIELDS = (
    "series",
    "model",
    "status",
    "input_faces",
    "output_faces",
    "faces_out_in",
    "estimated_faces",
    "output_estimated",
    "target_error_bbox_ratio",
    "h_range",
    "auto_delta_quantile",
    "h_min_safe_factor",
    "rmse_pct_bbox",
    "cd_bbox2",
    "qavg",
    "qmin",
    "violation",
    "split_count",
    "collapse_count",
    "flip_count",
    "runtime_sec",
    "case_path",
    "error",
)

SUMMARY_FIELDS = (
    "cases",
    "successes",
    "avg_faces_out_in",
    "avg_output_estimated",
    "avg_target_error_bbox_ratio",
    "avg_h_range",
    "avg_rmse_pct_bbox",
    "avg_cd_bbox2",
    "avg_qavg",
    "avg_qmin",
    "min_qmin",
    "avg_violation",
    "max_violation",
    "avg_runtime_sec",
)

COMPARISON_FIELDS = (
    "model",
    "status",
    "faces_out_in_current",
    "faces_out_in_baseline",
    "faces_out_in_delta",
    "output_faces_current",
    "output_faces_baseline",
    "output_faces_delta",
    "rmse_pct_bbox_current",
    "rmse_pct_bbox_baseline",
    "rmse_pct_bbox_delta",
    "cd_bbox2_current",
    "cd_bbox2_baseline",
    "cd_bbox2_delta_pct",
    "qavg_current",
    "qavg_baseline",
    "qavg_delta",
    "qmin_current",
    "qmin_baseline",
    "qmin_delta",
    "violation_current",
    "violation_baseline",
    "violation_delta",
    "runtime_sec_current",
    "runtime_sec_baseline",
    "runtime_sec_delta",
)


def model_index(path_or_name: Path | str) -> int:
    name = path_or_name.stem if isinstance(path_or_name, Path) else path_or_name
    if not name.startswith("T"):
        raise ValueError(f"unexpected T model name: {name}")
    return int(name[1:])


def discover_t_models() -> list[Path]:
    return sorted(
        [path for path in EXAMPLES_ROOT.glob("T*.off") if path.stem[1:].isdigit()],
        key=model_index,
    )


def case_dir(model: str) -> Path:
    return RESULT_ROOT / model


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def to_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def average(rows: list[dict[str, Any]], key: str) -> float:
    values = [to_float(row.get(key)) for row in rows]
    values = [value for value in values if math.isfinite(value)]
    return sum(values) / len(values) if values else math.nan


def fmt(value: Any, digits: int = 6) -> str:
    number = to_float(value)
    if not math.isfinite(number):
        return ""
    return f"{number:.{digits}f}"


def sci(value: Any) -> str:
    number = to_float(value)
    if not math.isfinite(number):
        return ""
    return f"{number:.6E}"


def row_from_summary(model: str, summary: dict[str, Any]) -> dict[str, Any]:
    params = summary.get("parameters", {})
    input_summary = summary.get("input_summary", {})
    output_summary = summary.get("output_summary", {})
    chamfer = summary.get("chamfer_distance", {})
    iteration = helper.final_iteration(summary)
    total_split, total_collapse, total_flip = helper.operation_totals(summary)

    input_faces = int(input_summary.get("faces", 0) or 0)
    output_faces = int(output_summary.get("faces", 0) or 0)
    estimated_faces = to_float(params.get("estimated_faces"))
    bbox = to_float(params.get("bbox_diagonal"))
    h_min = to_float(params.get("h_min"))
    h_max = to_float(params.get("h_max"))
    rmse = to_float(chamfer.get("symmetric_rmse"))
    cd = to_float(chamfer.get("chamfer_distance"))

    return {
        "series": "T",
        "model": model,
        "status": "success",
        "input_faces": input_faces,
        "output_faces": output_faces,
        "faces_out_in": output_faces / input_faces if input_faces > 0 else math.nan,
        "estimated_faces": estimated_faces,
        "output_estimated": output_faces / estimated_faces if estimated_faces > 0 else math.nan,
        "target_error_bbox_ratio": params.get("target_error_bbox_ratio", math.nan),
        "h_range": h_max / h_min if h_min > 0.0 else math.nan,
        "auto_delta_quantile": params.get("auto_delta_quantile", math.nan),
        "h_min_safe_factor": params.get("h_min_safe_factor", math.nan),
        "rmse_pct_bbox": 100.0 * rmse / bbox if math.isfinite(rmse) and bbox > 0.0 else math.nan,
        "cd_bbox2": cd / (bbox * bbox) if math.isfinite(cd) and bbox > 0.0 else math.nan,
        "qavg": output_summary.get("qavg", math.nan),
        "qmin": output_summary.get("qmin", math.nan),
        "violation": iteration.get("violation_rate", 0.0),
        "split_count": total_split,
        "collapse_count": total_collapse,
        "flip_count": total_flip,
        "runtime_sec": summary.get("runtime_sec", math.nan),
        "case_path": str(case_dir(model).relative_to(ROOT)).replace("\\", "/"),
        "error": "",
    }


def failed_row(model: str, error: str) -> dict[str, Any]:
    row = {field: "" for field in CSV_FIELDS}
    row.update({
        "series": "T",
        "model": model,
        "status": "failed",
        "case_path": str(case_dir(model).relative_to(ROOT)).replace("\\", "/"),
        "error": error,
    })
    return row


def run_case(mesh: Path, args: argparse.Namespace) -> dict[str, Any]:
    model = mesh.stem
    current_dir = case_dir(model)
    current_dir.mkdir(parents=True, exist_ok=True)
    summary_path = current_dir / "summary.json"
    output_path = current_dir / "output.ply"
    log_path = current_dir / "run.log"
    status_path = current_dir / "status.json"

    if args.resume and summary_path.is_file() and output_path.is_file():
        try:
            return row_from_summary(model, helper.load_json(summary_path))
        except Exception:
            pass

    command = [
        str(args.cli),
        str(mesh),
        "-o",
        str(output_path),
        "--iters",
        str(args.iters),
        "--preprocess",
        "0",
        "--curvature-mode",
        "principal-max",
        "--csf-alpha",
        "2.0",
        "--split-factor",
        str(FIXED_SPLIT_FACTOR),
        "--collapse-factor",
        str(FIXED_COLLAPSE_FACTOR),
        "--target-h-max-factor",
        str(FIXED_H_MAX_FACTOR),
        "--auto-delta-quantile",
        str(AUTO_DELTA_QUANTILE),
        "--gt",
        str(mesh),
        "--cd-samples",
        str(args.cd_samples),
        "--summary-json",
        str(summary_path),
        "--quiet",
    ]

    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write("$ " + " ".join(command) + "\n\n")
        log.flush()
        try:
            completed = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=args.timeout_seconds,
                check=False,
            )
            returncode = completed.returncode
        except subprocess.TimeoutExpired:
            elapsed = time.perf_counter() - started
            log.write(f"\nTIMEOUT after {args.timeout_seconds} seconds\n")
            status_path.write_text(
                json.dumps({"status": "timeout", "runtime_sec": elapsed, "command": command}, ensure_ascii=False, indent=2)
                + "\n",
                encoding="utf-8",
            )
            return failed_row(model, "timeout")

    elapsed = time.perf_counter() - started
    if returncode != 0 or not summary_path.is_file() or not output_path.is_file():
        status_path.write_text(
            json.dumps({"status": "failed", "returncode": returncode, "runtime_sec": elapsed, "command": command}, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        return failed_row(model, f"adaiso_cli returned {returncode}")

    status_path.write_text(
        json.dumps({"status": "success", "returncode": 0, "runtime_sec": elapsed, "command": command}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return row_from_summary(model, helper.load_json(summary_path))


def write_csv(rows: list[dict[str, Any]], path: Path, fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def summary_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [row for row in rows if row.get("status") == "success"]
    qmins = [to_float(row.get("qmin")) for row in successes]
    qmins = [value for value in qmins if math.isfinite(value)]
    violations = [to_float(row.get("violation")) for row in successes]
    violations = [value for value in violations if math.isfinite(value)]
    return {
        "cases": len(rows),
        "successes": len(successes),
        "avg_faces_out_in": average(successes, "faces_out_in"),
        "avg_output_estimated": average(successes, "output_estimated"),
        "avg_target_error_bbox_ratio": average(successes, "target_error_bbox_ratio"),
        "avg_h_range": average(successes, "h_range"),
        "avg_rmse_pct_bbox": average(successes, "rmse_pct_bbox"),
        "avg_cd_bbox2": average(successes, "cd_bbox2"),
        "avg_qavg": average(successes, "qavg"),
        "avg_qmin": average(successes, "qmin"),
        "min_qmin": min(qmins) if qmins else math.nan,
        "avg_violation": average(successes, "violation"),
        "max_violation": max(violations) if violations else math.nan,
        "avg_runtime_sec": average(successes, "runtime_sec"),
    }


def read_baseline_rows() -> dict[str, dict[str, Any]]:
    with BASELINE_CSV.open("r", encoding="utf-8", newline="") as handle:
        return {row["model"]: row for row in csv.DictReader(handle) if row.get("series") == "T"}


def comparison_rows(rows: list[dict[str, Any]], baseline_by_model: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    for row in rows:
        model = row["model"]
        base = baseline_by_model.get(model)
        if row.get("status") != "success" or base is None:
            comparisons.append({"model": model, "status": "missing_baseline" if base is None else row.get("status")})
            continue

        def delta(current_key: str, baseline_key: str) -> tuple[float, float, float]:
            cur = to_float(row.get(current_key))
            old = to_float(base.get(baseline_key))
            return cur, old, cur - old if math.isfinite(cur) and math.isfinite(old) else math.nan

        faces_cur, faces_base, faces_delta = delta("faces_out_in", "faces_out_in")
        out_faces_cur, out_faces_base, out_faces_delta = delta("output_faces", "output_faces")
        rmse_cur, rmse_base, rmse_delta = delta("rmse_pct_bbox", "rmse_pct_bbox")
        cd_cur, cd_base, cd_delta = delta("cd_bbox2", "cd_bbox2")
        qavg_cur, qavg_base, qavg_delta = delta("qavg", "qavg")
        qmin_cur, qmin_base, qmin_delta = delta("qmin", "qmin")
        violation_cur, violation_base, violation_delta = delta("violation", "violation")
        runtime_cur, runtime_base, runtime_delta = delta("runtime_sec", "runtime_sec")
        comparisons.append({
            "model": model,
            "status": "success",
            "faces_out_in_current": faces_cur,
            "faces_out_in_baseline": faces_base,
            "faces_out_in_delta": faces_delta,
            "output_faces_current": out_faces_cur,
            "output_faces_baseline": out_faces_base,
            "output_faces_delta": out_faces_delta,
            "rmse_pct_bbox_current": rmse_cur,
            "rmse_pct_bbox_baseline": rmse_base,
            "rmse_pct_bbox_delta": rmse_delta,
            "cd_bbox2_current": cd_cur,
            "cd_bbox2_baseline": cd_base,
            "cd_bbox2_delta_pct": 100.0 * cd_delta / cd_base if math.isfinite(cd_delta) and cd_base != 0.0 else math.nan,
            "qavg_current": qavg_cur,
            "qavg_baseline": qavg_base,
            "qavg_delta": qavg_delta,
            "qmin_current": qmin_cur,
            "qmin_baseline": qmin_base,
            "qmin_delta": qmin_delta,
            "violation_current": violation_cur,
            "violation_baseline": violation_base,
            "violation_delta": violation_delta,
            "runtime_sec_current": runtime_cur,
            "runtime_sec_baseline": runtime_base,
            "runtime_sec_delta": runtime_delta,
        })
    return comparisons


def comparison_summary(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [row for row in comparisons if row.get("status") == "success"]
    summary: dict[str, Any] = {"cases": len(successes)}
    for key in (
        "faces_out_in",
        "output_faces",
        "rmse_pct_bbox",
        "cd_bbox2",
        "qavg",
        "qmin",
        "violation",
        "runtime_sec",
    ):
        summary[f"avg_{key}_current"] = average(successes, f"{key}_current")
        summary[f"avg_{key}_baseline"] = average(successes, f"{key}_baseline")
        summary[f"avg_{key}_delta"] = average(successes, f"{key}_delta")
    summary["avg_cd_bbox2_delta_pct"] = average(successes, "cd_bbox2_delta_pct")
    return summary


def better_counts(comparisons: list[dict[str, Any]]) -> dict[str, int]:
    successes = [row for row in comparisons if row.get("status") == "success"]
    return {
        "lower_faces": sum(1 for row in successes if to_float(row.get("faces_out_in_delta")) < 0.0),
        "lower_rmse": sum(1 for row in successes if to_float(row.get("rmse_pct_bbox_delta")) < 0.0),
        "lower_cd": sum(1 for row in successes if to_float(row.get("cd_bbox2_delta_pct")) < 0.0),
        "higher_qavg": sum(1 for row in successes if to_float(row.get("qavg_delta")) > 0.0),
        "higher_qmin": sum(1 for row in successes if to_float(row.get("qmin_delta")) > 0.0),
        "lower_violation": sum(1 for row in successes if to_float(row.get("violation_delta")) < 0.0),
        "faster": sum(1 for row in successes if to_float(row.get("runtime_sec_delta")) < 0.0),
    }


def write_report(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    comparisons: list[dict[str, Any]],
    comp_summary: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    successes = [row for row in rows if row.get("status") == "success"]
    failures = [row for row in rows if row.get("status") != "success"]
    counts = better_counts(comparisons)
    common_cases = int(comp_summary.get("cases", 0) or 0)

    lines: list[str] = []
    lines.append("# v5.1.5 T 系列尺寸场实验报告")
    lines.append("")
    lines.append("## 1. 结论摘要")
    lines.append("")
    lines.append(
        f"本轮只跑 T 系列 `{len(rows)}` 个模型，成功 `{len(successes)}/{len(rows)}`。"
        "v5.1.5 使用输入局部边长和 CSF 后曲率估计自动容差，旧的 `kappa_ref + h_ref_factor` 路径已经删除。"
    )
    lines.append("")
    lines.append(
        f"- 平均 faces out/in：`{fmt(summary.get('avg_faces_out_in'), 3)}`；"
        f"相对 v5.1.3 变化 `{fmt(comp_summary.get('avg_faces_out_in_delta'), 3)}`。"
    )
    lines.append(
        f"- 平均 RMSE %bbox：`{fmt(summary.get('avg_rmse_pct_bbox'), 5)}`；"
        f"相对 v5.1.3 变化 `{fmt(comp_summary.get('avg_rmse_pct_bbox_delta'), 5)}`。"
    )
    lines.append(
        f"- 平均 CD/bbox^2：`{sci(summary.get('avg_cd_bbox2'))}`；"
        f"相对 v5.1.3 平均变化 `{fmt(comp_summary.get('avg_cd_bbox2_delta_pct'), 2)}%`。"
    )
    lines.append(
        f"- 平均 qmin：`{fmt(summary.get('avg_qmin'), 5)}`；"
        f"相对 v5.1.3 变化 `{fmt(comp_summary.get('avg_qmin_delta'), 5)}`。"
    )
    lines.append(
        f"- 平均 violation：`{fmt(summary.get('avg_violation'), 5)}`；"
        f"相对 v5.1.3 变化 `{fmt(comp_summary.get('avg_violation_delta'), 5)}`。"
    )
    lines.append("")
    lines.append("## 2. 实验设置")
    lines.append("")
    lines.append(f"- CLI：`{args.cli}`")
    lines.append(f"- 输出目录：`{RESULT_ROOT.relative_to(ROOT).as_posix()}`")
    lines.append(f"- 当前汇总 CSV：`{(RESULT_ROOT / 'run_summary.csv').relative_to(ROOT).as_posix()}`")
    lines.append(f"- 对比 CSV：`{(RESULT_ROOT / 'v5.1.5_vs_v5.1.3_T_comparison.csv').relative_to(ROOT).as_posix()}`")
    lines.append(f"- 基线 CSV：`{BASELINE_CSV.relative_to(ROOT).as_posix()}`")
    lines.append(
        f"- 参数：`split={FIXED_SPLIT_FACTOR}, collapse={FIXED_COLLAPSE_FACTOR}, "
        f"hmax={FIXED_H_MAX_FACTOR}, auto_delta_quantile={AUTO_DELTA_QUANTILE}, iters={args.iters}`"
    )
    lines.append(f"- CD samples：`{args.cd_samples}`；workers：`{args.workers}`")
    lines.append("")
    lines.append("## 3. T 系列总体指标")
    lines.append("")
    lines.append("| cases | successes | faces out/in | output/estimated | delta/bbox | h range | RMSE %bbox | CD/bbox^2 | qavg | qmin avg/min | violation avg/max | runtime s |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    lines.append(
        f"| {summary['cases']} | {summary['successes']} | {fmt(summary['avg_faces_out_in'], 3)} | "
        f"{fmt(summary['avg_output_estimated'], 3)} | {sci(summary['avg_target_error_bbox_ratio'])} | "
        f"{fmt(summary['avg_h_range'], 2)} | {fmt(summary['avg_rmse_pct_bbox'], 5)} | "
        f"{sci(summary['avg_cd_bbox2'])} | {fmt(summary['avg_qavg'], 5)} | "
        f"{fmt(summary['avg_qmin'], 5)}/{fmt(summary['min_qmin'], 5)} | "
        f"{fmt(summary['avg_violation'], 5)}/{fmt(summary['max_violation'], 5)} | "
        f"{fmt(summary['avg_runtime_sec'], 2)} |"
    )
    lines.append("")
    lines.append("## 4. 与 v5.1.3 对比")
    lines.append("")
    lines.append(f"共同成功 T 模型数：`{common_cases}`。v5.1.3 参考报告为 `v5.1.3_vs_CWF_CGAL内部判断实验报告.md` 中的 T 系列结果。")
    lines.append("")
    lines.append("| metric | v5.1.5 | v5.1.3 | delta |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| faces out/in | {fmt(comp_summary.get('avg_faces_out_in_current'), 3)} | {fmt(comp_summary.get('avg_faces_out_in_baseline'), 3)} | {fmt(comp_summary.get('avg_faces_out_in_delta'), 3)} |")
    lines.append(f"| output faces | {fmt(comp_summary.get('avg_output_faces_current'), 2)} | {fmt(comp_summary.get('avg_output_faces_baseline'), 2)} | {fmt(comp_summary.get('avg_output_faces_delta'), 2)} |")
    lines.append(f"| RMSE %bbox | {fmt(comp_summary.get('avg_rmse_pct_bbox_current'), 5)} | {fmt(comp_summary.get('avg_rmse_pct_bbox_baseline'), 5)} | {fmt(comp_summary.get('avg_rmse_pct_bbox_delta'), 5)} |")
    lines.append(f"| CD/bbox^2 | {sci(comp_summary.get('avg_cd_bbox2_current'))} | {sci(comp_summary.get('avg_cd_bbox2_baseline'))} | {fmt(comp_summary.get('avg_cd_bbox2_delta_pct'), 2)}% |")
    lines.append(f"| qavg | {fmt(comp_summary.get('avg_qavg_current'), 5)} | {fmt(comp_summary.get('avg_qavg_baseline'), 5)} | {fmt(comp_summary.get('avg_qavg_delta'), 5)} |")
    lines.append(f"| qmin | {fmt(comp_summary.get('avg_qmin_current'), 5)} | {fmt(comp_summary.get('avg_qmin_baseline'), 5)} | {fmt(comp_summary.get('avg_qmin_delta'), 5)} |")
    lines.append(f"| violation | {fmt(comp_summary.get('avg_violation_current'), 5)} | {fmt(comp_summary.get('avg_violation_baseline'), 5)} | {fmt(comp_summary.get('avg_violation_delta'), 5)} |")
    lines.append(f"| runtime s | {fmt(comp_summary.get('avg_runtime_sec_current'), 2)} | {fmt(comp_summary.get('avg_runtime_sec_baseline'), 2)} | {fmt(comp_summary.get('avg_runtime_sec_delta'), 2)} |")
    lines.append("")
    lines.append("逐模型胜出计数：")
    lines.append("")
    lines.append("| criterion | v5.1.5 better / common |")
    lines.append("|---|---:|")
    lines.append(f"| lower faces out/in | {counts['lower_faces']}/{common_cases} |")
    lines.append(f"| lower RMSE %bbox | {counts['lower_rmse']}/{common_cases} |")
    lines.append(f"| lower CD/bbox^2 | {counts['lower_cd']}/{common_cases} |")
    lines.append(f"| higher qavg | {counts['higher_qavg']}/{common_cases} |")
    lines.append(f"| higher qmin | {counts['higher_qmin']}/{common_cases} |")
    lines.append(f"| lower violation | {counts['lower_violation']}/{common_cases} |")
    lines.append(f"| faster runtime | {counts['faster']}/{common_cases} |")
    lines.append("")
    lines.append("## 5. 变化最大的模型")
    lines.append("")
    lines.append("### 5.1 面数下降最多")
    lines.append("")
    lines.append("| model | faces out/in v5.1.5 | v5.1.3 | delta | RMSE delta | violation delta |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in sorted([r for r in comparisons if r.get("status") == "success"], key=lambda r: to_float(r.get("faces_out_in_delta")))[:8]:
        lines.append(
            f"| {row['model']} | {fmt(row['faces_out_in_current'], 3)} | {fmt(row['faces_out_in_baseline'], 3)} | "
            f"{fmt(row['faces_out_in_delta'], 3)} | {fmt(row['rmse_pct_bbox_delta'], 5)} | {fmt(row['violation_delta'], 5)} |"
        )
    lines.append("")
    lines.append("### 5.2 误差恶化最多")
    lines.append("")
    lines.append("| model | RMSE %bbox v5.1.5 | v5.1.3 | delta | faces delta | violation delta |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in sorted([r for r in comparisons if r.get("status") == "success"], key=lambda r: to_float(r.get("rmse_pct_bbox_delta")), reverse=True)[:8]:
        lines.append(
            f"| {row['model']} | {fmt(row['rmse_pct_bbox_current'], 5)} | {fmt(row['rmse_pct_bbox_baseline'], 5)} | "
            f"{fmt(row['rmse_pct_bbox_delta'], 5)} | {fmt(row['faces_out_in_delta'], 3)} | {fmt(row['violation_delta'], 5)} |"
        )
    lines.append("")
    lines.append("### 5.3 violation 最高")
    lines.append("")
    lines.append("| model | faces out/in | RMSE %bbox | qmin | violation | split/collapse/flip |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in sorted(successes, key=lambda r: to_float(r.get("violation")), reverse=True)[:10]:
        lines.append(
            f"| {row['model']} | {fmt(row['faces_out_in'], 3)} | {fmt(row['rmse_pct_bbox'], 5)} | "
            f"{fmt(row['qmin'], 5)} | {fmt(row['violation'], 5)} | "
            f"{row['split_count']}/{row['collapse_count']}/{row['flip_count']} |"
        )
    lines.append("")
    lines.append("## 6. 内部判断")
    lines.append("")
    if to_float(comp_summary.get("avg_faces_out_in_delta")) < 0:
        lines.append("- v5.1.5 的主要目标是让自动容差回到输入分辨率驱动；如果面数明显下降，说明这个方向确实降低了 v5.1.3 为稳定性付出的加密代价。")
    else:
        lines.append("- v5.1.5 没有降低平均输出规模，说明 q=0.5 的输入分辨率容差并不比 v5.1.3 的 best params 更松。")
    if to_float(comp_summary.get("avg_rmse_pct_bbox_delta")) > 0 or to_float(comp_summary.get("avg_violation_delta")) > 0:
        lines.append("- 代价是几何误差或 target metric violation 出现上升；下一步应优先检查 `local_delta_heat.ply` 与高 violation 模型的空间分布。")
    else:
        lines.append("- 几何误差和 violation 没有系统性恶化，可以继续把 v5.1.5 作为尺寸场主线推进。")
    if to_float(comp_summary.get("avg_qmin_delta")) < 0:
        lines.append("- 需要注意平均 qmin 下降，说明新尺寸场降低面数的同时没有改善最小单元质量；后续仍要单独处理坏角和 collapse/flip 质量控制。")
    lines.append("- 本报告只比较 T 系列。是否推广到 `model_*` 系列需要单独跑全量或至少高风险 model 小样本。")
    if failures:
        lines.append("")
        lines.append("## 7. 失败项")
        lines.append("")
        lines.append("| model | error |")
        lines.append("|---|---|")
        for row in failures:
            lines.append(f"| {row['model']} | {row['error']} |")
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v5.1.5 T-series sizing validation.")
    parser.add_argument("--cli", type=Path, default=DEFAULT_CLI)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--cd-samples", type=int, default=100000)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--only", nargs="*", help="Optional subset of T model names.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    meshes = discover_t_models()
    if args.only:
        wanted = set(args.only)
        meshes = [mesh for mesh in meshes if mesh.stem in wanted]

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_case, mesh, args): mesh for mesh in meshes}
        for index, future in enumerate(as_completed(futures), start=1):
            mesh = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = failed_row(mesh.stem, str(exc))
            rows.append(row)
            print(f"[{index}/{len(meshes)}] {mesh.stem} {row['status']}")

    rows.sort(key=lambda row: model_index(row["model"]))
    summary = summary_row(rows)
    baseline = read_baseline_rows()
    comparisons = comparison_rows(rows, baseline)
    comp_summary = comparison_summary(comparisons)
    write_csv(rows, RESULT_ROOT / "run_summary.csv", CSV_FIELDS)
    write_csv([summary], RESULT_ROOT / "summary.csv", SUMMARY_FIELDS)
    write_csv(comparisons, RESULT_ROOT / "v5.1.5_vs_v5.1.3_T_comparison.csv", COMPARISON_FIELDS)
    write_report(rows, summary, comparisons, comp_summary, args)
    print(f"wrote {RESULT_ROOT / 'run_summary.csv'}")
    print(f"wrote {RESULT_ROOT / 'summary.csv'}")
    print(f"wrote {RESULT_ROOT / 'v5.1.5_vs_v5.1.3_T_comparison.csv'}")
    print(f"wrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
