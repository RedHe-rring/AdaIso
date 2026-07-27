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
RESULT_ROOT = CPP_ROOT / "results" / "v5.1.4_kapparef_probe"
REPORT_PATH = CPP_ROOT / "修改计划与实验结果" / "v5.1.4 kapparef小样本实验报告.md"
DEFAULT_CLI = helper.DEFAULT_CLI
CASES = helper.CASES

FIXED_SPLIT_FACTOR = 1.2
FIXED_COLLAPSE_FACTOR = 0.6
FIXED_H_MAX_FACTOR = 3.0
FIXED_H_REF_FACTOR = 0.45
KAPPA_REFERENCES = ("max", "p99", "p95")

CSV_FIELDS = (
    "series",
    "model",
    "param_id",
    "kappa_reference",
    "status",
    "input_faces",
    "output_faces",
    "faces_out_in",
    "estimated_faces",
    "output_estimated",
    "target_error_bbox_ratio",
    "h_range",
    "kappa_ref",
    "kappa_max",
    "kappa_ref_over_max",
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
AGGREGATE_FIELDS = (
    "param_id",
    "series",
    "cases",
    "kappa_reference",
    "avg_faces_out_in",
    "avg_output_estimated",
    "avg_target_error_bbox_ratio",
    "avg_h_range",
    "avg_kappa_ref_over_max",
    "avg_rmse_pct_bbox",
    "avg_cd_bbox2",
    "avg_qmin",
    "avg_violation",
    "avg_runtime_sec",
)


def param_id(kappa_reference: str) -> str:
    return f"kref_{kappa_reference}"


def case_dir(model: str, kappa_reference: str) -> Path:
    return RESULT_ROOT / helper.series_name(model) / model / param_id(kappa_reference)


def row_from_summary(model: str, kappa_reference: str, summary: dict[str, Any]) -> dict[str, Any]:
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
    kappa_ref = float(params.get("kappa_ref", math.nan))
    kappa_max = float(params.get("kappa_max", math.nan))
    rmse = chamfer.get("symmetric_rmse")
    cd = chamfer.get("chamfer_distance")

    return {
        "series": helper.series_name(model),
        "model": model,
        "param_id": param_id(kappa_reference),
        "kappa_reference": params.get("kappa_reference", kappa_reference),
        "status": "success",
        "input_faces": input_faces,
        "output_faces": output_faces,
        "faces_out_in": output_faces / input_faces if input_faces > 0 else math.nan,
        "estimated_faces": estimated_faces,
        "output_estimated": output_faces / estimated_faces if estimated_faces > 0 else math.nan,
        "target_error_bbox_ratio": params.get("target_error_bbox_ratio", math.nan),
        "h_range": h_max / h_min if h_min > 0.0 else math.nan,
        "kappa_ref": kappa_ref,
        "kappa_max": kappa_max,
        "kappa_ref_over_max": kappa_ref / kappa_max if kappa_max > 0.0 else math.nan,
        "rmse_pct_bbox": 100.0 * float(rmse) / bbox if helper.finite(rmse) and bbox > 0.0 else math.nan,
        "cd_bbox2": float(cd) / (bbox * bbox) if helper.finite(cd) and bbox > 0.0 else math.nan,
        "qavg": output_summary.get("qavg", math.nan),
        "qmin": output_summary.get("qmin", math.nan),
        "violation": iteration.get("violation_rate", 0.0),
        "split_count": total_split,
        "collapse_count": total_collapse,
        "flip_count": total_flip,
        "runtime_sec": summary.get("runtime_sec", math.nan),
        "case_path": str(case_dir(model, kappa_reference).relative_to(ROOT)).replace("\\", "/"),
        "error": "",
    }


def failed_row(model: str, kappa_reference: str, error: str) -> dict[str, Any]:
    row = {field: "" for field in CSV_FIELDS}
    row.update({
        "series": helper.series_name(model),
        "model": model,
        "param_id": param_id(kappa_reference),
        "kappa_reference": kappa_reference,
        "status": "failed",
        "case_path": str(case_dir(model, kappa_reference).relative_to(ROOT)).replace("\\", "/"),
        "error": error,
    })
    return row


def run_case(model: str, kappa_reference: str, args: argparse.Namespace) -> dict[str, Any]:
    current_dir = case_dir(model, kappa_reference)
    current_dir.mkdir(parents=True, exist_ok=True)
    summary_path = current_dir / "summary.json"
    output_path = current_dir / "output.ply"
    log_path = current_dir / "run.log"
    status_path = current_dir / "status.json"

    if args.resume and summary_path.is_file() and output_path.is_file():
        try:
            return row_from_summary(model, kappa_reference, helper.load_json(summary_path))
        except Exception:
            pass

    mesh = helper.input_path(model)
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
        "--auto-delta-kappa-ref",
        kappa_reference,
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
            return failed_row(model, kappa_reference, "timeout")

    elapsed = time.perf_counter() - started
    if returncode != 0 or not summary_path.is_file() or not output_path.is_file():
        status_path.write_text(
            json.dumps({"status": "failed", "returncode": returncode, "runtime_sec": elapsed, "command": command}, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        return failed_row(model, kappa_reference, f"adaiso_cli returned {returncode}")

    status_path.write_text(
        json.dumps({"status": "success", "returncode": 0, "runtime_sec": elapsed, "command": command}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return row_from_summary(model, kappa_reference, helper.load_json(summary_path))


def write_csv(rows: list[dict[str, Any]], path: Path, fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def average(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if helper.finite(row.get(key))]
    return sum(values) / len(values) if values else math.nan


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    successes = [row for row in rows if row.get("status") == "success"]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in successes:
        grouped.setdefault((str(row["param_id"]), str(row["series"])), []).append(row)
        grouped.setdefault((str(row["param_id"]), "all"), []).append(row)

    aggregates: list[dict[str, Any]] = []
    for (param, series), group in sorted(grouped.items()):
        aggregates.append({
            "param_id": param,
            "series": series,
            "cases": len(group),
            "kappa_reference": group[0]["kappa_reference"],
            "avg_faces_out_in": average(group, "faces_out_in"),
            "avg_output_estimated": average(group, "output_estimated"),
            "avg_target_error_bbox_ratio": average(group, "target_error_bbox_ratio"),
            "avg_h_range": average(group, "h_range"),
            "avg_kappa_ref_over_max": average(group, "kappa_ref_over_max"),
            "avg_rmse_pct_bbox": average(group, "rmse_pct_bbox"),
            "avg_cd_bbox2": average(group, "cd_bbox2"),
            "avg_qmin": average(group, "qmin"),
            "avg_violation": average(group, "violation"),
            "avg_runtime_sec": average(group, "runtime_sec"),
        })
    return aggregates


def best(items: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not items:
        return None
    return min(
        items,
        key=lambda row: (
            float(row["avg_violation"]),
            float(row["avg_rmse_pct_bbox"]),
            -float(row["avg_qmin"]),
        ),
    )


def write_report(rows: list[dict[str, Any]], aggregates: list[dict[str, Any]], args: argparse.Namespace) -> None:
    successes = [row for row in rows if row.get("status") == "success"]
    failures = [row for row in rows if row.get("status") != "success"]
    model_aggs = [row for row in aggregates if row["series"] == "model"]
    all_aggs = [row for row in aggregates if row["series"] == "all"]
    model_best = best(model_aggs)
    all_best = best(all_aggs)

    lines: list[str] = []
    lines.append("# v5.1.4 kapparef小样本实验报告")
    lines.append("")
    lines.append("## 实验设置")
    lines.append("")
    lines.append(f"- CLI: `{args.cli}`")
    lines.append(f"- 输出目录: `{RESULT_ROOT.relative_to(ROOT).as_posix()}`")
    lines.append(f"- 汇总 CSV: `{(RESULT_ROOT / 'run_summary.csv').relative_to(ROOT).as_posix()}`")
    lines.append(f"- 聚合 CSV: `{(RESULT_ROOT / 'param_summary.csv').relative_to(ROOT).as_posix()}`")
    lines.append(f"- 样例: `{', '.join(CASES)}`")
    lines.append(
        f"- 固定参数: `split={FIXED_SPLIT_FACTOR}, collapse={FIXED_COLLAPSE_FACTOR}, "
        f"hmax={FIXED_H_MAX_FACTOR}, href={FIXED_H_REF_FACTOR}`"
    )
    lines.append(f"- kappa_ref: `{', '.join(KAPPA_REFERENCES)}`")
    lines.append(f"- iters: `{args.iters}`, cd samples: `{args.cd_samples}`, workers: `{args.workers}`")
    lines.append("")
    lines.append("## 结论快照")
    lines.append("")
    if model_best:
        lines.append(
            f"- model 组综合优先参数: `{model_best['param_id']}`, avg violation "
            f"`{helper.fmt(model_best['avg_violation'])}`, avg RMSE %bbox "
            f"`{helper.fmt(model_best['avg_rmse_pct_bbox'])}`, avg qmin "
            f"`{helper.fmt(model_best['avg_qmin'])}`."
        )
    if all_best:
        lines.append(
            f"- 全部样例综合优先参数: `{all_best['param_id']}`, avg violation "
            f"`{helper.fmt(all_best['avg_violation'])}`, avg RMSE %bbox "
            f"`{helper.fmt(all_best['avg_rmse_pct_bbox'])}`, avg qmin "
            f"`{helper.fmt(all_best['avg_qmin'])}`."
        )
    lines.append(f"- 成功: `{len(successes)}/{len(rows)}`, 失败: `{len(failures)}`.")
    lines.append(
        "- `p95` 明显降低 violation，但代价是输出面数增加、运行时间变长，且 qmin 下降；"
        "它更适合作为后续实验选项，不适合直接替换默认 `max`。"
    )
    lines.append(
        "- `p99` 是较温和折中，但本小样本中对 RMSE 的改善很小，主要收益仍来自更低 violation。"
    )
    lines.append("")
    lines.append("## 参数聚合")
    lines.append("")
    lines.append("| param | series | cases | faces out/in | output/estimated | delta/bbox | h range | kref/kmax | rmse %bbox | CD/bbox^2 | qmin | violation | runtime s |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in sorted(aggregates, key=lambda item: (item["series"] != "model", item["series"], item["avg_violation"])):
        lines.append(
            f"| {row['param_id']} | {row['series']} | {row['cases']} | {helper.fmt(row['avg_faces_out_in'], 3)} | "
            f"{helper.fmt(row['avg_output_estimated'], 3)} | {helper.sci(row['avg_target_error_bbox_ratio'])} | "
            f"{helper.fmt(row['avg_h_range'], 2)} | {helper.fmt(row['avg_kappa_ref_over_max'], 5)} | "
            f"{helper.fmt(row['avg_rmse_pct_bbox'], 5)} | {helper.sci(row['avg_cd_bbox2'])} | "
            f"{helper.fmt(row['avg_qmin'], 5)} | {helper.fmt(row['avg_violation'], 5)} | "
            f"{helper.fmt(row['avg_runtime_sec'], 2)} |"
        )
    lines.append("")
    lines.append("## 逐样例最优")
    lines.append("")
    lines.append("| model | best param | faces out/in | output/estimated | kref/kmax | rmse %bbox | CD/bbox^2 | qmin | violation | split/collapse/flip |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for model in CASES:
        group = [row for row in successes if row["model"] == model]
        if not group:
            continue
        winner = min(group, key=lambda row: (float(row["violation"]), float(row["rmse_pct_bbox"]), -float(row["qmin"])))
        lines.append(
            f"| {model} | {winner['param_id']} | {helper.fmt(winner['faces_out_in'], 3)} | "
            f"{helper.fmt(winner['output_estimated'], 3)} | {helper.fmt(winner['kappa_ref_over_max'], 5)} | "
            f"{helper.fmt(winner['rmse_pct_bbox'], 5)} | {helper.sci(winner['cd_bbox2'])} | "
            f"{helper.fmt(winner['qmin'], 5)} | {helper.fmt(winner['violation'], 5)} | "
            f"{winner['split_count']}/{winner['collapse_count']}/{winner['flip_count']} |"
        )
    if failures:
        lines.append("")
        lines.append("## 失败项")
        lines.append("")
        lines.append("| model | param | error |")
        lines.append("|---|---|---|")
        for row in failures:
            lines.append(f"| {row['model']} | {row['param_id']} | {row['error']} |")
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v5.1.4 kappa reference probe on selected high-risk meshes.")
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
    models = list(CASES)
    if args.only:
        wanted = set(args.only)
        models = [model for model in models if model in wanted]
    for model in models:
        helper.input_path(model)

    jobs = [(model, reference) for model in models for reference in KAPPA_REFERENCES]
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_case, model, reference, args): (model, reference)
            for model, reference in jobs
        }
        for index, future in enumerate(as_completed(futures), start=1):
            model, reference = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = failed_row(model, reference, str(exc))
            rows.append(row)
            print(f"[{index}/{len(jobs)}] {model} {param_id(reference)} {row['status']}")

    rows.sort(key=lambda row: (row["series"], row["model"], row["kappa_reference"]))
    aggregates = aggregate_rows(rows)
    write_csv(rows, RESULT_ROOT / "run_summary.csv", CSV_FIELDS)
    write_csv(aggregates, RESULT_ROOT / "param_summary.csv", AGGREGATE_FIELDS)
    write_report(rows, aggregates, args)
    print(f"wrote {RESULT_ROOT / 'run_summary.csv'}")
    print(f"wrote {RESULT_ROOT / 'param_summary.csv'}")
    print(f"wrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
