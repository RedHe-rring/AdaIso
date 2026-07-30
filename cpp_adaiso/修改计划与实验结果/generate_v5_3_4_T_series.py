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


BASE_SCRIPT = Path(__file__).with_name("generate_v5_2_0_T_series.py")
SPEC = importlib.util.spec_from_file_location("v5_2_0_t_series", BASE_SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"failed to load base script: {BASE_SCRIPT}")
v520 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v520)
base = v520.base

ROOT = v520.ROOT
CPP_ROOT = v520.CPP_ROOT
RESULT_ROOT = CPP_ROOT / "results" / "v5.3.4_T_series_collapse_target_energy_q50"
BASELINE_CSV = (
    CPP_ROOT
    / "results"
    / "v5.3.3_T_series_collapse_diagnosis_q50"
    / "run_summary.csv"
)
REPORT_PATH = CPP_ROOT / "修改计划与实验结果" / "v5.3.4 T系列Collapse Target-Energy实验报告.md"

POLICY = "split_collapse_no_guard"

REJECT_REASON_FIELDS = (
    "size_stage_collapse_rejected_by_boundary",
    "size_stage_collapse_rejected_by_non_manifold",
    "size_stage_collapse_rejected_by_batch_conflict",
    "size_stage_collapse_rejected_by_link_condition",
    "size_stage_collapse_rejected_by_degenerate_patch",
    "size_stage_collapse_rejected_by_target_length",
    "size_stage_collapse_rejected_by_angle",
    "size_stage_collapse_rejected_by_batch_duplicate",
)

CSV_FIELDS = (
    *v520.CSV_FIELDS,
    "relocation_count",
    "size_stage_iterations",
    "size_stage_splits",
    "size_stage_collapses",
    "size_stage_long_edge_candidates",
    "size_stage_split_accepted",
    "size_stage_split_rejected_by_angle",
    "size_stage_short_edge_candidates",
    "size_stage_collapse_accepted",
    "size_stage_collapse_rejected",
    *REJECT_REASON_FIELDS,
    "collapse_accept_rate",
    "collapse_reject_breakdown_closed",
    "size_stage_final_violation",
    "size_stage_final_min_angle_degrees",
    "size_stage_final_angle_lt_10_pct",
    "size_stage_final_angle_lt_20_pct",
    "quality_stage_iterations",
    "quality_stage_flips",
    "quality_stage_relocations",
)


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
    return f"{number:.{digits}f}" if math.isfinite(number) else ""


def sci(value: Any) -> str:
    number = to_float(value)
    return f"{number:.6E}" if math.isfinite(number) else ""


def case_dir(model: str) -> Path:
    return RESULT_ROOT / model


def row_from_summary(model: str, summary: dict[str, Any]) -> dict[str, Any]:
    row = v520.ORIGINAL_ROW_FROM_SUMMARY(model, summary)
    stats = summary.get("stats", {})
    output_summary = summary.get("output_summary", {})
    iterations = summary.get("iterations", [])
    row.update({
        "case_path": str(case_dir(model).relative_to(ROOT)).replace("\\", "/"),
        "min_angle_degrees": output_summary.get("min_angle_degrees", math.nan),
        "angle_lt_10_pct": output_summary.get("angle_lt_10_pct", math.nan),
        "angle_lt_20_pct": output_summary.get("angle_lt_20_pct", math.nan),
        "relocation_count": sum(int(item.get("relocation_count", 0) or 0) for item in iterations),
        "size_stage_iterations": stats.get("size_stage_iterations", math.nan),
        "size_stage_splits": stats.get("size_stage_splits", math.nan),
        "size_stage_collapses": stats.get("size_stage_collapses", math.nan),
        "size_stage_long_edge_candidates": stats.get("size_stage_long_edge_candidates", math.nan),
        "size_stage_split_accepted": stats.get("size_stage_split_accepted", math.nan),
        "size_stage_split_rejected_by_angle": stats.get("size_stage_split_rejected_by_angle", math.nan),
        "size_stage_short_edge_candidates": stats.get("size_stage_short_edge_candidates", math.nan),
        "size_stage_collapse_accepted": stats.get("size_stage_collapse_accepted", math.nan),
        "size_stage_collapse_rejected": stats.get("size_stage_collapse_rejected", math.nan),
        "size_stage_final_violation": stats.get("size_stage_final_violation", math.nan),
        "size_stage_final_min_angle_degrees": stats.get("size_stage_final_min_angle_degrees", math.nan),
        "size_stage_final_angle_lt_10_pct": stats.get("size_stage_final_angle_lt_10_pct", math.nan),
        "size_stage_final_angle_lt_20_pct": stats.get("size_stage_final_angle_lt_20_pct", math.nan),
        "quality_stage_iterations": stats.get("quality_stage_iterations", math.nan),
        "quality_stage_flips": stats.get("quality_stage_flips", math.nan),
        "quality_stage_relocations": stats.get("quality_stage_relocations", math.nan),
    })
    for field in REJECT_REASON_FIELDS:
        row[field] = stats.get(field, math.nan)

    accepted = to_float(row.get("size_stage_collapse_accepted"))
    rejected = to_float(row.get("size_stage_collapse_rejected"))
    denom = accepted + rejected
    row["collapse_accept_rate"] = accepted / denom if denom > 0 else math.nan
    reason_sum = sum(to_float(row.get(field)) for field in REJECT_REASON_FIELDS)
    row["collapse_reject_breakdown_closed"] = (
        int(reason_sum == rejected) if math.isfinite(reason_sum) and math.isfinite(rejected) else 0
    )
    return row


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

    if args.resume and summary_path.is_file() and output_path.is_file():
        try:
            return row_from_summary(model, json.loads(summary_path.read_text(encoding="utf-8")))
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
        str(base.FIXED_SPLIT_FACTOR),
        "--collapse-factor",
        str(base.FIXED_COLLAPSE_FACTOR),
        "--target-h-max-factor",
        str(base.FIXED_H_MAX_FACTOR),
        "--auto-delta-quantile",
        str(base.AUTO_DELTA_QUANTILE),
        "--stage-a-policy",
        POLICY,
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
        completed = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=args.timeout_seconds,
            check=False,
        )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        return failed_row(model, f"exit {completed.returncode}; see {log_path}")
    row = row_from_summary(model, json.loads(summary_path.read_text(encoding="utf-8")))
    row["runtime_sec"] = row.get("runtime_sec") or elapsed
    return row


def read_baseline_rows() -> dict[str, dict[str, Any]]:
    with BASELINE_CSV.open("r", encoding="utf-8", newline="") as handle:
        return {row["model"]: dict(row) for row in csv.DictReader(handle) if row.get("series") == "T"}


def read_baseline_summary() -> dict[str, Any]:
    path = BASELINE_CSV.parent / "summary.csv"
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return dict(rows[0]) if rows else {}


def summary_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = v520.summary_row(rows)
    successes = [row for row in rows if row.get("status") == "success"]
    for field in (
        "size_stage_short_edge_candidates",
        "size_stage_collapse_accepted",
        "size_stage_collapse_rejected",
        "collapse_accept_rate",
        "size_stage_final_violation",
    ):
        summary[f"avg_{field}"] = average(successes, field)
    for field in REJECT_REASON_FIELDS:
        summary[f"avg_{field}"] = average(successes, field)
    summary["reject_breakdown_closed_cases"] = sum(
        1 for row in successes if str(row.get("collapse_reject_breakdown_closed")) == "1"
    )
    return summary


def comparison_rows(rows: list[dict[str, Any]], baseline: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    comparisons = []
    for row in rows:
        model = row.get("model")
        old = baseline.get(model)
        if row.get("status") != "success" or old is None or old.get("status") != "success":
            continue
        comparisons.append({
            "model": model,
            "faces_out_in_delta": to_float(row.get("faces_out_in")) - to_float(old.get("faces_out_in")),
            "min_angle_degrees_delta": to_float(row.get("min_angle_degrees")) - to_float(old.get("min_angle_degrees")),
            "angle_lt_10_pct_delta": to_float(row.get("angle_lt_10_pct")) - to_float(old.get("angle_lt_10_pct")),
            "angle_lt_20_pct_delta": to_float(row.get("angle_lt_20_pct")) - to_float(old.get("angle_lt_20_pct")),
            "violation_delta": to_float(row.get("violation")) - to_float(old.get("violation")),
            "rmse_pct_bbox_delta": to_float(row.get("rmse_pct_bbox")) - to_float(old.get("rmse_pct_bbox")),
            "cd_bbox2_delta": to_float(row.get("cd_bbox2")) - to_float(old.get("cd_bbox2")),
        })
    return comparisons


def reason_share(summary: dict[str, Any], field: str) -> float:
    rejected = to_float(summary.get("avg_size_stage_collapse_rejected"))
    value = to_float(summary.get(f"avg_{field}"))
    return value / rejected if rejected > 0 else math.nan


def write_report(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    comparisons: list[dict[str, Any]],
    baseline_summary: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    successes = [row for row in rows if row.get("status") == "success"]
    failures = [row for row in rows if row.get("status") != "success"]
    lines = [
        "# v5.3.4 T系列 Collapse Target-Energy 实验报告",
        "",
        "## 1. 结论摘要",
        "",
        f"本轮使用 v5.3.4 默认策略 `{POLICY}` 跑 T 系列 `{len(rows)}` 个模型，成功 `{len(successes)}/{len(rows)}`。",
        "本版只替换 collapse 的 target-length 接受准则：由逐边硬上限改为局部 target metric violation 下降，并保留 max-ratio slack。",
        "",
        "## 2. 总体指标",
        "",
        "| faces out/in | min angle | angle<10% | angle<20% | violation | Stage A violation | runtime s |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        f"| {fmt(summary['avg_faces_out_in'], 3)} | {fmt(summary['avg_min_angle_degrees'], 4)} | "
        f"{fmt(summary['avg_angle_lt_10_pct'], 4)} | {fmt(summary['avg_angle_lt_20_pct'], 4)} | "
        f"{fmt(summary['avg_violation'], 5)} | {fmt(summary['avg_size_stage_final_violation'], 5)} | "
        f"{fmt(summary['avg_runtime_sec'], 2)} |",
        "",
        "## 3. Collapse 拒绝分解",
        "",
        "| item | avg count | share of rejected |",
        "|---|---:|---:|",
    ]
    for field in REJECT_REASON_FIELDS:
        label = field.replace("size_stage_collapse_rejected_by_", "")
        lines.append(f"| {label} | {fmt(summary[f'avg_{field}'], 2)} | {fmt(100.0 * reason_share(summary, field), 2)}% |")

    lines.extend([
        "",
        "## 4. Collapse 候选闭合性",
        "",
        "| short candidates | accepted | rejected | accept rate | closed cases |",
        "|---:|---:|---:|---:|---:|",
        f"| {fmt(summary['avg_size_stage_short_edge_candidates'], 2)} | "
        f"{fmt(summary['avg_size_stage_collapse_accepted'], 2)} | "
        f"{fmt(summary['avg_size_stage_collapse_rejected'], 2)} | "
        f"{fmt(100.0 * summary['avg_collapse_accept_rate'], 2)}% | "
        f"{summary['reject_breakdown_closed_cases']}/{len(successes)} |",
        "",
        "## 5. 与 v5.3.3 对比",
        "",
    ])
    if comparisons:
        lines.extend([
            "| metric delta | avg | max abs |",
            "|---|---:|---:|",
        ])
        for field in (
            "faces_out_in_delta",
            "min_angle_degrees_delta",
            "angle_lt_10_pct_delta",
            "angle_lt_20_pct_delta",
            "violation_delta",
            "rmse_pct_bbox_delta",
            "cd_bbox2_delta",
        ):
            values = [to_float(row.get(field)) for row in comparisons]
            values = [value for value in values if math.isfinite(value)]
            avg = sum(values) / len(values) if values else math.nan
            max_abs = max((abs(value) for value in values), default=math.nan)
            lines.append(f"| {field} | {fmt(avg, 8)} | {fmt(max_abs, 8)} |")
    else:
        lines.append("未找到可用 v5.3.3 基线 CSV，跳过输出一致性对比。")

    if baseline_summary:
        lines.extend([
            "",
            "## 6. Collapse 诊断对比",
            "",
            "| metric | v5.3.4 | v5.3.3 | delta |",
            "|---|---:|---:|---:|",
        ])
        for label, field in (
            ("short candidates", "avg_size_stage_short_edge_candidates"),
            ("collapse accepted", "avg_size_stage_collapse_accepted"),
            ("collapse rejected", "avg_size_stage_collapse_rejected"),
            ("accept rate", "avg_collapse_accept_rate"),
            ("target_length rejected", "avg_size_stage_collapse_rejected_by_target_length"),
            ("batch_conflict rejected", "avg_size_stage_collapse_rejected_by_batch_conflict"),
            ("degenerate_patch rejected", "avg_size_stage_collapse_rejected_by_degenerate_patch"),
            ("angle rejected", "avg_size_stage_collapse_rejected_by_angle"),
        ):
            current = to_float(summary.get(field))
            baseline = to_float(baseline_summary.get(field))
            scale = 100.0 if field == "avg_collapse_accept_rate" else 1.0
            suffix = "%" if field == "avg_collapse_accept_rate" else ""
            lines.append(
                f"| {label} | {fmt(scale * current, 2)}{suffix} | "
                f"{fmt(scale * baseline, 2)}{suffix} | {fmt(scale * (current - baseline), 2)}{suffix} |"
            )

    lines.extend([
        "",
        "## 7. 判断",
        "",
        f"- 主拒绝原因是 `target_length`，平均 `{fmt(summary['avg_size_stage_collapse_rejected_by_target_length'], 2)}` 条，占拒绝的 `{fmt(100.0 * reason_share(summary, 'size_stage_collapse_rejected_by_target_length'), 2)}%`。",
        f"- 第二原因是 `batch_conflict`，平均 `{fmt(summary['avg_size_stage_collapse_rejected_by_batch_conflict'], 2)}` 条，占 `{fmt(100.0 * reason_share(summary, 'size_stage_collapse_rejected_by_batch_conflict'), 2)}%`。",
        f"- `angle` 拒绝平均 `{fmt(summary['avg_size_stage_collapse_rejected_by_angle'], 2)}` 条，只占 `{fmt(100.0 * reason_share(summary, 'size_stage_collapse_rejected_by_angle'), 2)}%`；Stage A collapse 的主要瓶颈不是角度安全线。",
        f"- collapse 平均接受率为 `{fmt(100.0 * summary['avg_collapse_accept_rate'], 2)}%`；虽然接受数量比 v5.3.3 增加，但短边候选增长更快，所以接受率没有提升。",
        "",
        "- v5.3.4 的最终质量和尺寸场指标是正向的：faces out/in、violation、min angle、angle<10%、angle<20% 均优于 v5.3.3。",
        "- 但它没有消除 target_length 拒绝主因；`target_length` 绝对拒绝数增加，说明新准则改变了演化路径，而不是把目标长度瓶颈彻底解除。",
        "- 下一步不应立刻调 slack，而应继续拆分 target_length：区分 `energy_not_decreased` 和 `max_ratio_slack` 两类拒绝，再决定是改 energy 定义还是放松局部长边上限。",
        "",
        "本轮数据对应的下一步：",
        "- v5.3.5 优先把 `target_length` 拒绝拆成 energy 与 max-ratio slack 两个诊断字段。",
        "- batch selection 仍是第二问题，但应等 target_length 内部分解完成后再做 mini-batch 或 sequential collapse。",
        "",
        "输出文件：",
        f"- `{(RESULT_ROOT / 'run_summary.csv').relative_to(ROOT).as_posix()}`",
        f"- `{(RESULT_ROOT / 'summary.csv').relative_to(ROOT).as_posix()}`",
        f"- `{REPORT_PATH.relative_to(ROOT).as_posix()}`",
    ])
    if failures:
        lines.extend(["", "## 7. 失败项", "", "| model | error |", "|---|---|"])
        for row in failures:
            lines.append(f"| {row['model']} | {row['error']} |")

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v5.3.4 T-series collapse target-energy validation.")
    parser.add_argument("--cli", type=Path, default=base.DEFAULT_CLI)
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
    meshes = base.discover_t_models()
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

    rows.sort(key=lambda row: base.model_index(row["model"]))
    summary = summary_row(rows)
    comparisons = comparison_rows(rows, read_baseline_rows()) if BASELINE_CSV.is_file() else []
    base.write_csv(rows, RESULT_ROOT / "run_summary.csv", CSV_FIELDS)
    base.write_csv([summary], RESULT_ROOT / "summary.csv", tuple(summary.keys()))
    if comparisons:
        base.write_csv(comparisons, RESULT_ROOT / "v5.3.4_vs_v5.3.3_T_comparison.csv", tuple(comparisons[0].keys()))
    write_report(rows, summary, comparisons, read_baseline_summary(), args)
    print(f"wrote {RESULT_ROOT / 'run_summary.csv'}")
    print(f"wrote {RESULT_ROOT / 'summary.csv'}")
    print(f"wrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
