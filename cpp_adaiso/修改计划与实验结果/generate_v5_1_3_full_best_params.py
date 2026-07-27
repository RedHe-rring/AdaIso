from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import re
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
RESULT_ROOT = CPP_ROOT / "results" / "v5.1.3_T_model_best_params_full"
REPORT_PATH = CPP_ROOT / "修改计划与实验结果" / "v5.1.3 全T和model best params实验报告.md"
BASELINE_CSV = CPP_ROOT / "results" / "v5.1.0_T_model_auto_delta_full" / "v5.1.0_T_model_auto_delta_full_summary.csv"
DEFAULT_CLI = helper.DEFAULT_CLI

FIXED_SPLIT_FACTOR = 1.2
FIXED_COLLAPSE_FACTOR = 0.6
FIXED_H_MAX_FACTOR = 3.0
FIXED_H_REF_FACTOR = 0.45

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
SERIES_FIELDS = (
    "series",
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
    "series",
    "cases",
    "avg_faces_out_in_current",
    "avg_faces_out_in_baseline",
    "avg_faces_out_in_delta",
    "avg_rmse_pct_bbox_current",
    "avg_rmse_pct_bbox_baseline",
    "avg_rmse_pct_bbox_delta",
    "avg_cd_bbox2_current",
    "avg_cd_bbox2_baseline",
    "avg_cd_bbox2_delta_pct",
    "avg_qmin_current",
    "avg_qmin_baseline",
    "avg_qmin_delta",
    "avg_violation_current",
    "avg_violation_baseline",
    "avg_violation_delta",
)
SERIES_ORDER = {"T": 0, "model": 1, "all": 2}


def natural_key(path_or_name: Path | str) -> tuple[str, int]:
    name = path_or_name.stem if isinstance(path_or_name, Path) else path_or_name
    if match := re.fullmatch(r"T(\d+)", name):
        return ("T", int(match.group(1)))
    if match := re.fullmatch(r"model_(\d+)", name):
        return ("model", int(match.group(1)))
    raise ValueError(f"unexpected model name: {name}")


def model_sort_key(path_or_name: Path | str) -> tuple[int, int]:
    series, index = natural_key(path_or_name)
    return (SERIES_ORDER[series], index)


def series_name(model: str) -> str:
    return natural_key(model)[0]


def discover_models() -> list[Path]:
    paths: list[Path] = []
    for pattern in ("T*.off", "model_*.off"):
        for path in EXAMPLES_ROOT.glob(pattern):
            try:
                natural_key(path)
            except ValueError:
                continue
            paths.append(path)
    return sorted(paths, key=model_sort_key)


def case_dir(model: str) -> Path:
    return RESULT_ROOT / series_name(model) / model


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def average(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if finite(row.get(key))]
    return sum(values) / len(values) if values else math.nan


def fmt(value: Any, digits: int = 6) -> str:
    if not finite(value):
        return ""
    return f"{float(value):.{digits}f}"


def sci(value: Any) -> str:
    if not finite(value):
        return ""
    return f"{float(value):.6E}"


def row_from_summary(model: str, summary: dict[str, Any]) -> dict[str, Any]:
    params = summary.get("parameters", {})
    input_summary = summary.get("input_summary", {})
    output_summary = summary.get("output_summary", {})
    chamfer = summary.get("chamfer_distance", {})
    iteration = helper.final_iteration(summary)
    total_split, total_collapse, total_flip = helper.operation_totals(summary)

    input_faces = int(input_summary.get("faces", 0) or 0)
    output_faces = int(output_summary.get("faces", 0) or 0)
    estimated_faces = float(params.get("estimated_faces", 0.0) or 0.0)
    bbox = float(params.get("bbox_diagonal", 0.0) or 0.0)
    h_min = float(params.get("h_min", 0.0) or 0.0)
    h_max = float(params.get("h_max", 0.0) or 0.0)
    rmse = chamfer.get("symmetric_rmse")
    cd = chamfer.get("chamfer_distance")

    return {
        "series": series_name(model),
        "model": model,
        "status": "success",
        "input_faces": input_faces,
        "output_faces": output_faces,
        "faces_out_in": output_faces / input_faces if input_faces > 0 else math.nan,
        "estimated_faces": estimated_faces,
        "output_estimated": output_faces / estimated_faces if estimated_faces > 0 else math.nan,
        "target_error_bbox_ratio": params.get("target_error_bbox_ratio", math.nan),
        "h_range": h_max / h_min if h_min > 0.0 else math.nan,
        "rmse_pct_bbox": 100.0 * float(rmse) / bbox if finite(rmse) and bbox > 0.0 else math.nan,
        "cd_bbox2": float(cd) / (bbox * bbox) if finite(cd) and bbox > 0.0 else math.nan,
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
        "series": series_name(model),
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
        "--target-error-h-ref-factor",
        str(FIXED_H_REF_FACTOR),
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


def series_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    successes = [row for row in rows if row.get("status") == "success"]
    groups: dict[str, list[dict[str, Any]]] = {"T": [], "model": [], "all": []}
    for row in successes:
        groups[row["series"]].append(row)
        groups["all"].append(row)

    summaries: list[dict[str, Any]] = []
    for series in ("T", "model", "all"):
        group = groups[series]
        qmins = [float(row["qmin"]) for row in group if finite(row.get("qmin"))]
        violations = [float(row["violation"]) for row in group if finite(row.get("violation"))]
        summaries.append({
            "series": series,
            "cases": len([row for row in rows if row.get("series") == series]) if series != "all" else len(rows),
            "successes": len(group),
            "avg_faces_out_in": average(group, "faces_out_in"),
            "avg_output_estimated": average(group, "output_estimated"),
            "avg_target_error_bbox_ratio": average(group, "target_error_bbox_ratio"),
            "avg_h_range": average(group, "h_range"),
            "avg_rmse_pct_bbox": average(group, "rmse_pct_bbox"),
            "avg_cd_bbox2": average(group, "cd_bbox2"),
            "avg_qavg": average(group, "qavg"),
            "avg_qmin": average(group, "qmin"),
            "min_qmin": min(qmins) if qmins else math.nan,
            "avg_violation": average(group, "violation"),
            "max_violation": max(violations) if violations else math.nan,
            "avg_runtime_sec": average(group, "runtime_sec"),
        })
    return summaries


def read_baseline_rows() -> dict[str, dict[str, Any]]:
    if not BASELINE_CSV.is_file():
        return {}
    with BASELINE_CSV.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["model"]: row for row in rows}


def comparison_summary(rows: list[dict[str, Any]], baseline_by_model: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    current_by_model = {row["model"]: row for row in rows if row.get("status") == "success"}
    common = [model for model in current_by_model if model in baseline_by_model]
    groups = {"T": [], "model": [], "all": []}
    for model in common:
        series = series_name(model)
        item = (current_by_model[model], baseline_by_model[model])
        groups[series].append(item)
        groups["all"].append(item)

    def avg_pair(items: list[tuple[dict[str, Any], dict[str, Any]]], current_key: str, baseline_key: str) -> tuple[float, float]:
        current_values = [float(cur[current_key]) for cur, base in items if finite(cur.get(current_key)) and baseline_key in base]
        baseline_values = [float(base[baseline_key]) for cur, base in items if finite(cur.get(current_key)) and baseline_key in base]
        if not current_values or not baseline_values:
            return math.nan, math.nan
        return sum(current_values) / len(current_values), sum(baseline_values) / len(baseline_values)

    summaries: list[dict[str, Any]] = []
    for series in ("T", "model", "all"):
        items = groups[series]
        faces_cur, faces_base = avg_pair(items, "faces_out_in", "face_ratio_out_in")
        rmse_cur, rmse_base = avg_pair(items, "rmse_pct_bbox", "rmse_bbox_pct")
        cd_cur, cd_base = avg_pair(items, "cd_bbox2", "cd_bbox2")
        qmin_cur, qmin_base = avg_pair(items, "qmin", "qmin")
        violation_cur, violation_base = avg_pair(items, "violation", "final_violation")
        summaries.append({
            "series": series,
            "cases": len(items),
            "avg_faces_out_in_current": faces_cur,
            "avg_faces_out_in_baseline": faces_base,
            "avg_faces_out_in_delta": faces_cur - faces_base if finite(faces_cur) and finite(faces_base) else math.nan,
            "avg_rmse_pct_bbox_current": rmse_cur,
            "avg_rmse_pct_bbox_baseline": rmse_base,
            "avg_rmse_pct_bbox_delta": rmse_cur - rmse_base if finite(rmse_cur) and finite(rmse_base) else math.nan,
            "avg_cd_bbox2_current": cd_cur,
            "avg_cd_bbox2_baseline": cd_base,
            "avg_cd_bbox2_delta_pct": 100.0 * (cd_cur - cd_base) / cd_base if finite(cd_cur) and finite(cd_base) and cd_base != 0.0 else math.nan,
            "avg_qmin_current": qmin_cur,
            "avg_qmin_baseline": qmin_base,
            "avg_qmin_delta": qmin_cur - qmin_base if finite(qmin_cur) and finite(qmin_base) else math.nan,
            "avg_violation_current": violation_cur,
            "avg_violation_baseline": violation_base,
            "avg_violation_delta": violation_cur - violation_base if finite(violation_cur) and finite(violation_base) else math.nan,
        })
    return summaries


def write_report(rows: list[dict[str, Any]], summaries: list[dict[str, Any]], comparisons: list[dict[str, Any]], args: argparse.Namespace) -> None:
    successes = [row for row in rows if row.get("status") == "success"]
    failures = [row for row in rows if row.get("status") != "success"]
    by_series = {row["series"]: row for row in summaries}
    all_summary = by_series.get("all", {})
    model_summary = by_series.get("model", {})
    all_comparison = {row["series"]: row for row in comparisons}.get("all", {})
    model_comparison = {row["series"]: row for row in comparisons}.get("model", {})

    lines: list[str] = []
    lines.append("# v5.1.3 全 T 和 model best params 实验报告")
    lines.append("")
    lines.append("## 实验设置")
    lines.append("")
    lines.append(f"- CLI：`{args.cli}`")
    lines.append(f"- 参数：`split={FIXED_SPLIT_FACTOR}, collapse={FIXED_COLLAPSE_FACTOR}, hmax={FIXED_H_MAX_FACTOR}, href={FIXED_H_REF_FACTOR}`")
    lines.append(f"- 输出目录：`{RESULT_ROOT.relative_to(ROOT).as_posix()}`")
    lines.append(f"- 汇总 CSV：`{(RESULT_ROOT / 'run_summary.csv').relative_to(ROOT).as_posix()}`")
    lines.append(f"- 分组 CSV：`{(RESULT_ROOT / 'series_summary.csv').relative_to(ROOT).as_posix()}`")
    lines.append(f"- 对比 CSV：`{(RESULT_ROOT / 'v5.1.0_vs_v5.1.3_best_params_comparison.csv').relative_to(ROOT).as_posix()}`")
    lines.append(f"- iters：`{args.iters}`，cd samples：`{args.cd_samples}`，workers：`{args.workers}`")
    lines.append("")
    lines.append("## 结论快照")
    lines.append("")
    lines.append(
        f"- 全量成功：{len(successes)}/{len(rows)}；all avg violation `{fmt(all_summary.get('avg_violation'))}`，"
        f"avg RMSE %bbox `{fmt(all_summary.get('avg_rmse_pct_bbox'))}`，avg qmin `{fmt(all_summary.get('avg_qmin'))}`。"
    )
    if model_comparison:
        lines.append(
            f"- 相对 v5.1.0，model avg violation 变化 `{fmt(model_comparison.get('avg_violation_delta'))}`，"
            f"avg RMSE %bbox 变化 `{fmt(model_comparison.get('avg_rmse_pct_bbox_delta'))}`，"
            f"avg qmin 变化 `{fmt(model_comparison.get('avg_qmin_delta'))}`。"
        )
    if all_comparison:
        lines.append(
            f"- 相对 v5.1.0，all avg violation 变化 `{fmt(all_comparison.get('avg_violation_delta'))}`，"
            f"avg faces out/in 变化 `{fmt(all_comparison.get('avg_faces_out_in_delta'), 3)}`。"
        )
    lines.append("")
    lines.append("## 分组指标")
    lines.append("")
    lines.append("| series | cases | successes | faces out/in | output/estimated | delta/bbox | h range | rmse %bbox | CD/bbox^2 | qavg | qmin avg/min | violation avg/max | runtime s |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in summaries:
        lines.append(
            f"| {row['series']} | {row['cases']} | {row['successes']} | {fmt(row['avg_faces_out_in'], 3)} | "
            f"{fmt(row['avg_output_estimated'], 3)} | {sci(row['avg_target_error_bbox_ratio'])} | "
            f"{fmt(row['avg_h_range'], 2)} | {fmt(row['avg_rmse_pct_bbox'], 5)} | {sci(row['avg_cd_bbox2'])} | "
            f"{fmt(row['avg_qavg'], 5)} | {fmt(row['avg_qmin'], 5)}/{fmt(row['min_qmin'], 5)} | "
            f"{fmt(row['avg_violation'], 5)}/{fmt(row['max_violation'], 5)} | {fmt(row['avg_runtime_sec'], 2)} |"
        )
    if comparisons:
        lines.append("")
        lines.append("## 与 v5.1.0 对比")
        lines.append("")
        lines.append("| series | cases | faces out/in Δ | rmse %bbox Δ | CD/bbox2 Δ% | qmin Δ | violation Δ |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for row in comparisons:
            lines.append(
                f"| {row['series']} | {row['cases']} | {fmt(row['avg_faces_out_in_delta'], 3)} | "
                f"{fmt(row['avg_rmse_pct_bbox_delta'], 5)} | {fmt(row['avg_cd_bbox2_delta_pct'], 2)} | "
                f"{fmt(row['avg_qmin_delta'], 5)} | {fmt(row['avg_violation_delta'], 5)} |"
            )
    lines.append("")
    lines.append("## 高风险样例")
    lines.append("")
    lines.append("| model | faces out/in | output/estimated | rmse %bbox | CD/bbox^2 | qmin | violation | split/collapse/flip |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    high_risk = sorted(
        successes,
        key=lambda row: (
            -float(row["violation"]),
            -float(row["rmse_pct_bbox"]),
            float(row["qmin"]),
        ),
    )[:12]
    for row in high_risk:
        lines.append(
            f"| {row['model']} | {fmt(row['faces_out_in'], 3)} | {fmt(row['output_estimated'], 3)} | "
            f"{fmt(row['rmse_pct_bbox'], 5)} | {sci(row['cd_bbox2'])} | {fmt(row['qmin'], 5)} | "
            f"{fmt(row['violation'], 5)} | {row['split_count']}/{row['collapse_count']}/{row['flip_count']} |"
        )
    if failures:
        lines.append("")
        lines.append("## 失败项")
        lines.append("")
        lines.append("| model | error |")
        lines.append("|---|---|")
        for row in failures:
            lines.append(f"| {row['model']} | {row['error']} |")
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full T/model v5.1.3 best-parameter validation.")
    parser.add_argument("--cli", type=Path, default=DEFAULT_CLI)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--cd-samples", type=int, default=100000)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--only", nargs="*", help="Optional subset of model names.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    meshes = discover_models()
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

    rows.sort(key=lambda row: model_sort_key(row["model"]))
    summaries = series_summary(rows)
    comparisons = comparison_summary(rows, read_baseline_rows())
    write_csv(rows, RESULT_ROOT / "run_summary.csv", CSV_FIELDS)
    write_csv(summaries, RESULT_ROOT / "series_summary.csv", SERIES_FIELDS)
    write_csv(comparisons, RESULT_ROOT / "v5.1.0_vs_v5.1.3_best_params_comparison.csv", COMPARISON_FIELDS)
    write_report(rows, summaries, comparisons, args)
    print(f"wrote {RESULT_ROOT / 'run_summary.csv'}")
    print(f"wrote {RESULT_ROOT / 'series_summary.csv'}")
    print(f"wrote {RESULT_ROOT / 'v5.1.0_vs_v5.1.3_best_params_comparison.csv'}")
    print(f"wrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
