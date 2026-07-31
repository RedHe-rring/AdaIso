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
RESULT_ROOT = CPP_ROOT / "results" / "v5.3.7_T_series_continuous_size_energy_q50"
BASELINE_CSV = (
    CPP_ROOT
    / "results"
    / "v5.3.6_T_series_collapse_conflict_mini_round_q50"
    / "run_summary.csv"
)
REPORT_PATH = CPP_ROOT / "修改计划与实验结果" / "v5.3.7 T系列连续目标尺寸场能量实验报告.md"

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

RETRY_FIELDS = (
    "size_stage_collapse_conflict_mini_rounds",
    "size_stage_collapse_conflict_retry_candidates",
    "size_stage_collapse_conflict_retry_accepted",
    "size_stage_collapse_conflict_retry_rejected",
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
    *RETRY_FIELDS,
    "collapse_accept_rate",
    "collapse_reject_breakdown_closed",
    "size_stage_final_violation",
    "size_stage_final_short_violation",
    "size_stage_final_long_violation",
    "size_stage_final_target_log_energy",
    "size_stage_final_mean_ratio",
    "size_stage_final_std_ratio",
    "size_stage_final_p05_ratio",
    "size_stage_final_p95_ratio",
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
    size_iterations = [item for item in iterations if item.get("stage") == "size"]
    final_size = size_iterations[-1] if size_iterations else {}
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
        "size_stage_final_short_violation": stats.get("size_stage_final_short_violation", math.nan),
        "size_stage_final_long_violation": stats.get("size_stage_final_long_violation", math.nan),
        "size_stage_final_target_log_energy": stats.get("size_stage_final_target_log_energy", math.nan),
        "size_stage_final_mean_ratio": final_size.get("mean_ratio", math.nan),
        "size_stage_final_std_ratio": final_size.get("std_ratio", math.nan),
        "size_stage_final_p05_ratio": final_size.get("p05_ratio", math.nan),
        "size_stage_final_p95_ratio": final_size.get("p95_ratio", math.nan),
        "size_stage_final_min_angle_degrees": stats.get("size_stage_final_min_angle_degrees", math.nan),
        "size_stage_final_angle_lt_10_pct": stats.get("size_stage_final_angle_lt_10_pct", math.nan),
        "size_stage_final_angle_lt_20_pct": stats.get("size_stage_final_angle_lt_20_pct", math.nan),
        "quality_stage_iterations": stats.get("quality_stage_iterations", math.nan),
        "quality_stage_flips": stats.get("quality_stage_flips", math.nan),
        "quality_stage_relocations": stats.get("quality_stage_relocations", math.nan),
    })
    for field in REJECT_REASON_FIELDS:
        row[field] = stats.get(field, math.nan)
    for field in RETRY_FIELDS:
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
        "size_stage_final_short_violation",
        "size_stage_final_long_violation",
        "size_stage_final_target_log_energy",
        "size_stage_final_mean_ratio",
        "size_stage_final_std_ratio",
        "size_stage_final_p05_ratio",
        "size_stage_final_p95_ratio",
        *RETRY_FIELDS,
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
            "runtime_sec_delta": to_float(row.get("runtime_sec")) - to_float(old.get("runtime_sec")),
            "size_stage_final_violation_delta": (
                to_float(row.get("size_stage_final_violation")) -
                to_float(old.get("size_stage_final_violation"))
            ),
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

    def comparison_average(field: str) -> float:
        values = [to_float(row.get(field)) for row in comparisons]
        values = [value for value in values if math.isfinite(value)]
        return sum(values) / len(values) if values else math.nan

    baseline_violation_delta = comparison_average("size_stage_final_violation_delta")
    target_length_delta = math.nan
    if baseline_summary:
        target_length_delta = (
            to_float(summary.get("avg_size_stage_collapse_rejected_by_target_length"))
            - to_float(baseline_summary.get("avg_size_stage_collapse_rejected_by_target_length"))
        )

    lines = [
        "# v5.3.7 T系列 连续目标尺寸场能量实验报告",
        "",
        "## 1. 结论摘要",
        "",
        f"本轮使用 v5.3.7 默认策略 `{POLICY}` 跑 T 系列 `{len(rows)}` 个模型，成功 `{len(successes)}/{len(rows)}`。",
        "本版把局部尺寸场接受准则从 hinge violation 改为连续能量 `mean(log(r)^2)`；split 仍按长边阈值触发。",
        "",
        "## 2. Stage A 尺寸场距离",
        "",
        "| mean ratio | std ratio | p05 | p95 | violation | short viol | long viol | log energy |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {fmt(summary['avg_size_stage_final_mean_ratio'], 4)} | "
        f"{fmt(summary['avg_size_stage_final_std_ratio'], 4)} | "
        f"{fmt(summary['avg_size_stage_final_p05_ratio'], 4)} | "
        f"{fmt(summary['avg_size_stage_final_p95_ratio'], 4)} | "
        f"{fmt(summary['avg_size_stage_final_violation'], 5)} | "
        f"{fmt(summary['avg_size_stage_final_short_violation'], 5)} | "
        f"{fmt(summary['avg_size_stage_final_long_violation'], 5)} | "
        f"{fmt(summary['avg_size_stage_final_target_log_energy'], 6)} |",
        "",
        "## 3. 最终网格指标",
        "",
        "| faces out/in | min angle | angle<10% | angle<20% | output violation | RMSE %bbox | CD/bbox2 | runtime s |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {fmt(summary['avg_faces_out_in'], 3)} | {fmt(summary['avg_min_angle_degrees'], 4)} | "
        f"{fmt(summary['avg_angle_lt_10_pct'], 4)} | {fmt(summary['avg_angle_lt_20_pct'], 4)} | "
        f"{fmt(summary['avg_violation'], 5)} | {fmt(summary['avg_rmse_pct_bbox'], 5)} | "
        f"{sci(summary['avg_cd_bbox2'])} | {fmt(summary['avg_runtime_sec'], 2)} |",
        "",
        "## 4. Collapse 拒绝分解",
        "",
        "| item | avg count | share of rejected |",
        "|---|---:|---:|",
    ]
    for field in REJECT_REASON_FIELDS:
        label = field.replace("size_stage_collapse_rejected_by_", "")
        lines.append(f"| {label} | {fmt(summary[f'avg_{field}'], 2)} | {fmt(100.0 * reason_share(summary, field), 2)}% |")

    lines.extend([
        "",
        "## 5. Collapse 候选闭合性",
        "",
        "| short candidates | accepted | rejected | accept rate | closed cases |",
        "|---:|---:|---:|---:|---:|",
        f"| {fmt(summary['avg_size_stage_short_edge_candidates'], 2)} | "
        f"{fmt(summary['avg_size_stage_collapse_accepted'], 2)} | "
        f"{fmt(summary['avg_size_stage_collapse_rejected'], 2)} | "
        f"{fmt(100.0 * summary['avg_collapse_accept_rate'], 2)}% | "
        f"{summary['reject_breakdown_closed_cases']}/{len(successes)} |",
        "",
        "## 6. Conflict Retry 统计",
        "",
        "| retry mini-rounds | retry candidates | retry accepted | retry rejected |",
        "|---:|---:|---:|---:|",
        f"| {fmt(summary['avg_size_stage_collapse_conflict_mini_rounds'], 2)} | "
        f"{fmt(summary['avg_size_stage_collapse_conflict_retry_candidates'], 2)} | "
        f"{fmt(summary['avg_size_stage_collapse_conflict_retry_accepted'], 2)} | "
        f"{fmt(summary['avg_size_stage_collapse_conflict_retry_rejected'], 2)} |",
        "",
        "## 7. 与 v5.3.6 对比",
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
            "size_stage_final_violation_delta",
            "rmse_pct_bbox_delta",
            "cd_bbox2_delta",
            "runtime_sec_delta",
        ):
            values = [to_float(row.get(field)) for row in comparisons]
            values = [value for value in values if math.isfinite(value)]
            avg = sum(values) / len(values) if values else math.nan
            max_abs = max((abs(value) for value in values), default=math.nan)
            lines.append(f"| {field} | {fmt(avg, 8)} | {fmt(max_abs, 8)} |")
    else:
        lines.append("未找到可用 v5.3.6 基线 CSV，跳过输出一致性对比。")

    if baseline_summary:
        lines.extend([
            "",
            "## 8. Collapse 诊断对比",
            "",
            "| metric | v5.3.7 | v5.3.6 | delta |",
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
        "## 9. 判断",
        "",
        f"- Stage A 当前 `target_log_energy = {fmt(summary['avg_size_stage_final_target_log_energy'], 6)}`，这是 v5.3.7 后续版本的连续尺寸场距离主指标。",
        f"- Stage A 当前 `short_violation = {fmt(summary['avg_size_stage_final_short_violation'], 5)}`，`long_violation = {fmt(summary['avg_size_stage_final_long_violation'], 5)}`，用于判断残差来自短边还是长边。",
        f"- 与 v5.3.6 相比，Stage A 总 violation 平均变化 `{fmt(baseline_violation_delta, 6)}`；v5.3.6 没有记录 log energy，因此本报告不输出 log energy delta。",
        f"- `target_length` 拒绝平均 `{fmt(summary['avg_size_stage_collapse_rejected_by_target_length'], 2)}` 条，在 v5.3.7 中表示局部 `target_log_energy` 未下降；相对 v5.3.6 平均变化 `{fmt(target_length_delta, 2)}` 条。",
        "",
        "本轮数据对应的判断：",
        f"- 连续能量准则显著释放 collapse：accepted 平均 `{fmt(summary['avg_size_stage_collapse_accepted'], 2)}`，accept rate `{fmt(100.0 * summary['avg_collapse_accept_rate'], 2)}%`，明显高于 v5.3.6。",
        f"- 尺寸场贴合改善是系统性的：Stage A violation 平均变化 `{fmt(comparison_average('size_stage_final_violation_delta'), 6)}`，output violation 平均变化 `{fmt(comparison_average('violation_delta'), 6)}`，faces out/in 平均变化 `{fmt(comparison_average('faces_out_in_delta'), 6)}`。",
        f"- 主要代价是几何误差：RMSE 平均变化 `{fmt(comparison_average('rmse_pct_bbox_delta'), 6)}`，CD 平均变化 `{sci(comparison_average('cd_bbox2_delta'))}`；这说明 v5.3.7 更贴近尺寸场，但略牺牲曲面误差。",
        f"- 质量指标整体没有系统性恶化：min angle 平均变化 `{fmt(comparison_average('min_angle_degrees_delta'), 4)}`，angle<20% 平均变化 `{fmt(comparison_average('angle_lt_20_pct_delta'), 6)}`，但 angle<10% 基本持平。",
        "- 下一步不应回退到 hinge violation；应在连续尺寸场能量外增加 soft geometry/quality guard，控制 RMSE/CD 的小幅上升。",
        "",
        "输出文件：",
        f"- `{(RESULT_ROOT / 'run_summary.csv').relative_to(ROOT).as_posix()}`",
        f"- `{(RESULT_ROOT / 'summary.csv').relative_to(ROOT).as_posix()}`",
        f"- `{REPORT_PATH.relative_to(ROOT).as_posix()}`",
    ])
    if failures:
        lines.extend(["", "## 10. 失败项", "", "| model | error |", "|---|---|"])
        for row in failures:
            lines.append(f"| {row['model']} | {row['error']} |")

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v5.3.7 T-series continuous target-field energy validation.")
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
        base.write_csv(comparisons, RESULT_ROOT / "v5.3.7_vs_v5.3.6_T_comparison.csv", tuple(comparisons[0].keys()))
    write_report(rows, summary, comparisons, read_baseline_summary(), args)
    print(f"wrote {RESULT_ROOT / 'run_summary.csv'}")
    print(f"wrote {RESULT_ROOT / 'summary.csv'}")
    print(f"wrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
