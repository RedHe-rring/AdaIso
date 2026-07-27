from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CPP_ROOT = ROOT / "cpp_adaiso"
EXAMPLES_ROOT = CPP_ROOT / "examples"
RESULT_ROOT = CPP_ROOT / "results" / "v5.1.3_hysteresis_probe"
REPORT_PATH = CPP_ROOT / "修改计划与实验结果" / "v5.1.3 hysteresis小样本实验报告.md"
DEFAULT_CLI = CPP_ROOT / "build2" / "adaiso_cli.exe"

CASES = (
    "model_005",
    "model_010",
    "model_009",
    "model_007",
    "T125",
    "T10",
    "T53",
    "T44",
)
SPLIT_FACTORS = (1.2, 1.33, 1.5)
COLLAPSE_FACTORS = (0.6, 0.7, 0.8)
CSV_FIELDS = (
    "series",
    "model",
    "param_id",
    "split_factor",
    "collapse_factor",
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
AGGREGATE_FIELDS = (
    "param_id",
    "series",
    "cases",
    "split_factor",
    "collapse_factor",
    "avg_faces_out_in",
    "avg_output_estimated",
    "avg_rmse_pct_bbox",
    "avg_cd_bbox2",
    "avg_qmin",
    "avg_violation",
    "avg_runtime_sec",
)


def series_name(model: str) -> str:
    return "model" if model.startswith("model_") else "T"


def input_path(model: str) -> Path:
    suffix = ".off"
    path = EXAMPLES_ROOT / f"{model}{suffix}"
    if not path.is_file():
        raise FileNotFoundError(f"missing input mesh: {path}")
    return path


def param_id(split_factor: float, collapse_factor: float) -> str:
    return f"split{split_factor:g}_collapse{collapse_factor:g}".replace(".", "p")


def case_dir(model: str, split_factor: float, collapse_factor: float) -> Path:
    return RESULT_ROOT / series_name(model) / model / param_id(split_factor, collapse_factor)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def fmt(value: Any, digits: int = 6) -> str:
    if not finite(value):
        return ""
    return f"{float(value):.{digits}f}"


def sci(value: Any) -> str:
    if not finite(value):
        return ""
    return f"{float(value):.6E}"


def final_iteration(summary: dict[str, Any]) -> dict[str, Any]:
    iterations = summary.get("iterations", [])
    if not iterations:
        return {}
    return iterations[-1]


def operation_totals(summary: dict[str, Any]) -> tuple[int, int, int]:
    split = collapse = flip = 0
    for item in summary.get("iterations", []):
        split += int(item.get("split_count", 0) or 0)
        collapse += int(item.get("collapse_count", 0) or 0)
        flip += int(item.get("flip_count", 0) or 0)
    return split, collapse, flip


def row_from_summary(model: str, split_factor: float, collapse_factor: float, summary: dict[str, Any]) -> dict[str, Any]:
    params = summary.get("parameters", {})
    input_summary = summary.get("input_summary", {})
    output_summary = summary.get("output_summary", {})
    chamfer = summary.get("chamfer_distance", {})
    iteration = final_iteration(summary)
    total_split, total_collapse, total_flip = operation_totals(summary)

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
        "param_id": param_id(split_factor, collapse_factor),
        "split_factor": split_factor,
        "collapse_factor": collapse_factor,
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
        "case_path": str(case_dir(model, split_factor, collapse_factor).relative_to(ROOT)).replace("\\", "/"),
        "error": "",
    }


def failed_row(model: str, split_factor: float, collapse_factor: float, error: str) -> dict[str, Any]:
    row = {field: "" for field in CSV_FIELDS}
    row.update({
        "series": series_name(model),
        "model": model,
        "param_id": param_id(split_factor, collapse_factor),
        "split_factor": split_factor,
        "collapse_factor": collapse_factor,
        "status": "failed",
        "case_path": str(case_dir(model, split_factor, collapse_factor).relative_to(ROOT)).replace("\\", "/"),
        "error": error,
    })
    return row


def run_case(model: str, split_factor: float, collapse_factor: float, args: argparse.Namespace) -> dict[str, Any]:
    current_dir = case_dir(model, split_factor, collapse_factor)
    current_dir.mkdir(parents=True, exist_ok=True)
    summary_path = current_dir / "summary.json"
    output_path = current_dir / "output.ply"
    log_path = current_dir / "run.log"
    status_path = current_dir / "status.json"

    if args.resume and summary_path.is_file() and output_path.is_file():
        try:
            return row_from_summary(model, split_factor, collapse_factor, load_json(summary_path))
        except Exception:
            pass

    mesh = input_path(model)
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
        str(split_factor),
        "--collapse-factor",
        str(collapse_factor),
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
            status = {"status": "timeout", "runtime_sec": elapsed, "command": command}
            status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            return failed_row(model, split_factor, collapse_factor, "timeout")

    elapsed = time.perf_counter() - started
    if returncode != 0 or not summary_path.is_file() or not output_path.is_file():
        status = {"status": "failed", "returncode": returncode, "runtime_sec": elapsed, "command": command}
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return failed_row(model, split_factor, collapse_factor, f"adaiso_cli returned {returncode}")

    status_path.write_text(
        json.dumps({"status": "success", "returncode": 0, "runtime_sec": elapsed, "command": command}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return row_from_summary(model, split_factor, collapse_factor, load_json(summary_path))


def write_csv(rows: list[dict[str, Any]], path: Path, fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def average(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if finite(row.get(key))]
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
            "split_factor": group[0]["split_factor"],
            "collapse_factor": group[0]["collapse_factor"],
            "avg_faces_out_in": average(group, "faces_out_in"),
            "avg_output_estimated": average(group, "output_estimated"),
            "avg_rmse_pct_bbox": average(group, "rmse_pct_bbox"),
            "avg_cd_bbox2": average(group, "cd_bbox2"),
            "avg_qmin": average(group, "qmin"),
            "avg_violation": average(group, "violation"),
            "avg_runtime_sec": average(group, "runtime_sec"),
        })
    return aggregates


def write_report(rows: list[dict[str, Any]], aggregates: list[dict[str, Any]], args: argparse.Namespace) -> None:
    successes = [row for row in rows if row.get("status") == "success"]
    failures = [row for row in rows if row.get("status") != "success"]
    all_aggs = [row for row in aggregates if row["series"] == "all"]
    model_aggs = [row for row in aggregates if row["series"] == "model"]

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

    lines: list[str] = []
    lines.append("# v5.1.3 hysteresis 小样本实验报告")
    lines.append("")
    lines.append("## 实验设置")
    lines.append("")
    lines.append(f"- CLI：`{args.cli}`")
    lines.append(f"- 输出目录：`{RESULT_ROOT.relative_to(ROOT).as_posix()}`")
    lines.append(f"- 汇总 CSV：`{(RESULT_ROOT / 'run_summary.csv').relative_to(ROOT).as_posix()}`")
    lines.append(f"- 聚合 CSV：`{(RESULT_ROOT / 'param_summary.csv').relative_to(ROOT).as_posix()}`")
    lines.append(f"- 样例：`{', '.join(CASES)}`")
    lines.append(f"- splitFactor：`{', '.join(str(v) for v in SPLIT_FACTORS)}`")
    lines.append(f"- collapseFactor：`{', '.join(str(v) for v in COLLAPSE_FACTORS)}`")
    lines.append(f"- iters：`{args.iters}`，cd samples：`{args.cd_samples}`，workers：`{args.workers}`")
    lines.append("")
    lines.append("## 结论快照")
    lines.append("")
    all_best = best(all_aggs)
    model_best = best(model_aggs)
    if all_best:
        lines.append(
            f"- 全部样例综合优先参数：`{all_best['param_id']}`，avg violation `{fmt(all_best['avg_violation'])}`，"
            f"avg RMSE %bbox `{fmt(all_best['avg_rmse_pct_bbox'])}`，avg qmin `{fmt(all_best['avg_qmin'])}`。"
        )
    if model_best:
        lines.append(
            f"- model 组综合优先参数：`{model_best['param_id']}`，avg violation `{fmt(model_best['avg_violation'])}`，"
            f"avg RMSE %bbox `{fmt(model_best['avg_rmse_pct_bbox'])}`，avg qmin `{fmt(model_best['avg_qmin'])}`。"
        )
    lines.append(f"- 成功：{len(successes)}/{len(rows)}，失败：{len(failures)}。")
    lines.append("")
    lines.append("## 参数聚合")
    lines.append("")
    lines.append("| param | series | cases | faces out/in | output/estimated | rmse %bbox | CD/bbox^2 | qmin | violation | runtime s |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in sorted(aggregates, key=lambda item: (item["series"] != "model", item["series"], item["avg_violation"])):
        lines.append(
            f"| {row['param_id']} | {row['series']} | {row['cases']} | {fmt(row['avg_faces_out_in'], 3)} | "
            f"{fmt(row['avg_output_estimated'], 3)} | {fmt(row['avg_rmse_pct_bbox'], 5)} | {sci(row['avg_cd_bbox2'])} | "
            f"{fmt(row['avg_qmin'], 5)} | {fmt(row['avg_violation'], 5)} | {fmt(row['avg_runtime_sec'], 2)} |"
        )
    lines.append("")
    lines.append("## 逐样例最佳")
    lines.append("")
    lines.append("| model | best param | faces out/in | output/estimated | rmse %bbox | CD/bbox^2 | qmin | violation | split/collapse/flip |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for model in CASES:
        group = [row for row in successes if row["model"] == model]
        if not group:
            continue
        winner = min(group, key=lambda row: (float(row["violation"]), float(row["rmse_pct_bbox"]), -float(row["qmin"])))
        lines.append(
            f"| {model} | {winner['param_id']} | {fmt(winner['faces_out_in'], 3)} | {fmt(winner['output_estimated'], 3)} | "
            f"{fmt(winner['rmse_pct_bbox'], 5)} | {sci(winner['cd_bbox2'])} | {fmt(winner['qmin'], 5)} | "
            f"{fmt(winner['violation'], 5)} | {winner['split_count']}/{winner['collapse_count']}/{winner['flip_count']} |"
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
    parser = argparse.ArgumentParser(description="Run v5.1.3 hysteresis probe on selected high-risk meshes.")
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
        input_path(model)

    jobs = [(model, split, collapse) for model in models for split in SPLIT_FACTORS for collapse in COLLAPSE_FACTORS]
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_case, model, split, collapse, args): (model, split, collapse)
            for model, split, collapse in jobs
        }
        for index, future in enumerate(as_completed(futures), start=1):
            model, split, collapse = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = failed_row(model, split, collapse, str(exc))
            rows.append(row)
            print(f"[{index}/{len(jobs)}] {model} {param_id(split, collapse)} {row['status']}")

    rows.sort(key=lambda row: (row["series"], row["model"], row["split_factor"], row["collapse_factor"]))
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
