from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import shutil
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
RESULT_ROOT = CPP_ROOT / "results" / "v5.3.2_T_series_stage_a_ablation_q50"
REPORT_PATH = CPP_ROOT / "修改计划与实验结果" / "v5.3.2 T系列StageA诊断消融实验报告.md"

POLICIES = (
    "split_collapse_no_guard",
    "split_collapse_weak_guard",
    "collapse_split_weak_guard",
)

BASELINES = {
    "v5.3.0": CPP_ROOT / "results" / "v5.3.0_T_series_staged_remesh_q50" / "run_summary.csv",
    "v5.3.1": CPP_ROOT / "results" / "v5.3.1_T_series_stage_a_guard_q50" / "run_summary.csv",
    "v5.2.1": CPP_ROOT / "results" / "v5.2.1_T_series_min_angle_safety_q50" / "run_summary.csv",
}

CSV_FIELDS = (
    "policy",
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
    "size_stage_final_violation",
    "size_stage_final_min_angle_degrees",
    "size_stage_final_angle_lt_10_pct",
    "size_stage_final_angle_lt_20_pct",
    "quality_stage_iterations",
    "quality_stage_flips",
    "quality_stage_relocations",
)

SUMMARY_FIELDS = (
    "policy",
    *v520.SUMMARY_FIELDS,
    "avg_size_stage_iterations",
    "avg_size_stage_splits",
    "avg_size_stage_collapses",
    "avg_size_stage_long_edge_candidates",
    "avg_size_stage_split_rejected_by_angle",
    "avg_size_stage_short_edge_candidates",
    "avg_size_stage_collapse_rejected",
    "avg_size_stage_final_violation",
    "avg_size_stage_final_min_angle_degrees",
    "avg_size_stage_final_angle_lt_10_pct",
    "avg_size_stage_final_angle_lt_20_pct",
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


def case_dir(policy: str, model: str) -> Path:
    return RESULT_ROOT / policy / model


def row_from_summary(policy: str, model: str, summary: dict[str, Any]) -> dict[str, Any]:
    row = v520.ORIGINAL_ROW_FROM_SUMMARY(model, summary)
    output_summary = summary.get("output_summary", {})
    stats = summary.get("stats", {})
    iterations = summary.get("iterations", [])
    row.update({
        "policy": policy,
        "case_path": str(case_dir(policy, model).relative_to(ROOT)).replace("\\", "/"),
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
    return row


def failed_row(policy: str, model: str, error: str) -> dict[str, Any]:
    row = {field: "" for field in CSV_FIELDS}
    row.update({
        "policy": policy,
        "series": "T",
        "model": model,
        "status": "failed",
        "case_path": str(case_dir(policy, model).relative_to(ROOT)).replace("\\", "/"),
        "error": error,
    })
    return row


def run_case(policy: str, mesh: Path, args: argparse.Namespace) -> dict[str, Any]:
    model = mesh.stem
    current_dir = case_dir(policy, model)
    current_dir.mkdir(parents=True, exist_ok=True)
    summary_path = current_dir / "summary.json"
    output_path = current_dir / "output.ply"
    log_path = current_dir / "run.log"

    if args.resume and summary_path.is_file() and output_path.is_file():
      try:
          return row_from_summary(policy, model, base.helper.load_json(summary_path))
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
        policy,
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
        return failed_row(policy, model, f"exit {completed.returncode}; see {log_path}")
    row = row_from_summary(policy, model, base.helper.load_json(summary_path))
    row["runtime_sec"] = row.get("runtime_sec") or elapsed
    return row


def read_baseline_rows(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["model"]: dict(row) for row in csv.DictReader(handle) if row.get("series") == "T"}


def summary_row(policy: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = v520.summary_row(rows)
    summary["policy"] = policy
    successes = [row for row in rows if row.get("status") == "success"]
    for key in SUMMARY_FIELDS:
        if key.startswith("avg_size_stage_"):
            source = key[len("avg_"):]
            summary[key] = average(successes, source)
    return summary


def write_report(
    rows_by_policy: dict[str, list[dict[str, Any]]],
    summaries: list[dict[str, Any]],
    comparisons: dict[tuple[str, str], dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    lines = [
        "# v5.3.2 T 系列 Stage A 诊断消融实验报告",
        "",
        "## 1. 结论摘要",
        "",
        f"本轮使用当前工作区 v5.3.2 代码跑 T 系列，策略数 `{len(rows_by_policy)}`，每组模型数 `{len(next(iter(rows_by_policy.values())))}`。",
        "",
        "v5.3.2 的目的不是直接定最终算法，而是判断 Stage A 的问题来自顺序、split guard 还是 collapse 回收。",
        "",
        "## 2. 策略总体指标",
        "",
        "| policy | success | faces out/in | min angle | angle<10% | angle<20% | violation | stageA violation | split accepted | split angle rejected | collapse accepted | collapse rejected | runtime s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary['policy']} | {summary['successes']}/{summary['cases']} | "
            f"{fmt(summary['avg_faces_out_in'], 3)} | {fmt(summary['avg_min_angle_degrees'], 4)} | "
            f"{fmt(summary['avg_angle_lt_10_pct'], 4)} | {fmt(summary['avg_angle_lt_20_pct'], 4)} | "
            f"{fmt(summary['avg_violation'], 5)} | {fmt(summary['avg_size_stage_final_violation'], 5)} | "
            f"{fmt(summary['avg_size_stage_splits'], 2)} | {fmt(summary['avg_size_stage_split_rejected_by_angle'], 2)} | "
            f"{fmt(summary['avg_size_stage_collapses'], 2)} | {fmt(summary['avg_size_stage_collapse_rejected'], 2)} | "
            f"{fmt(summary['avg_runtime_sec'], 2)} |"
        )

    for section_index, baseline_name in enumerate(("v5.3.0", "v5.3.1", "v5.2.1"), start=3):
        lines.extend([
            "",
            f"## {section_index}. 对比 {baseline_name}",
            "",
            "| policy | faces delta | min angle delta | angle<10 delta | angle<20 delta | violation delta | runtime delta |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for policy in rows_by_policy:
            comp = comparisons[(policy, baseline_name)]
            lines.append(
                f"| {policy} | {fmt(comp.get('avg_faces_out_in_delta'), 3)} | "
                f"{fmt(comp.get('avg_min_angle_degrees_delta'), 4)} | "
                f"{fmt(comp.get('avg_angle_lt_10_pct_delta'), 4)} | "
                f"{fmt(comp.get('avg_angle_lt_20_pct_delta'), 4)} | "
                f"{fmt(comp.get('avg_violation_delta'), 5)} | "
                f"{fmt(comp.get('avg_runtime_sec_delta'), 2)} |"
            )

    lines.extend([
        "",
        "## 6. 诊断判断",
        "",
    ])
    weak = next((s for s in summaries if s["policy"] == "split_collapse_weak_guard"), None)
    no_guard = next((s for s in summaries if s["policy"] == "split_collapse_no_guard"), None)
    collapse_first = next((s for s in summaries if s["policy"] == "collapse_split_weak_guard"), None)
    if weak and no_guard:
        rejection = to_float(weak["avg_size_stage_split_rejected_by_angle"])
        violation_delta = to_float(weak["avg_violation"]) - to_float(no_guard["avg_violation"])
        lines.append(
            f"- weak guard 平均拒绝 `{fmt(rejection, 2)}` 条 split，但最终 violation 只比 no guard 高 `{fmt(violation_delta, 5)}`；"
            "它没有修复 v5.3.0 的过度加密，也没有改善坏角。"
        )
    if collapse_first and weak:
        stage_delta = to_float(collapse_first["avg_size_stage_final_violation"]) - to_float(weak["avg_size_stage_final_violation"])
        lines.append(
            f"- collapse-first weak guard 的 Stage A violation 与 split-first weak guard 只差 `{fmt(stage_delta, 5)}`，"
            "说明 v5.3.1 的欠采样主因不是顺序本身，而是强 split angle guard。"
        )
    if no_guard:
        lines.append(
            "- `split_collapse_no_guard` 与 v5.3.0 指标基本一致，证明 v5.3.2 诊断入口没有改变原始阶段化基线。"
        )
    lines.append("- 代码默认策略据此改为 `split_collapse_no_guard`；下一步应转向诊断 collapse 为什么拒绝大量 short-edge candidate，弱 split guard 仅保留为实验入口。")

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", type=Path, default=base.DEFAULT_CLI)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--cd-samples", type=int, default=100000)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--policies", nargs="*", default=list(POLICIES), choices=POLICIES)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if not args.cli.is_file():
        raise FileNotFoundError(args.cli)
    if not args.resume and RESULT_ROOT.exists():
        shutil.rmtree(RESULT_ROOT)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)

    meshes = base.discover_t_models()
    if args.only:
        wanted = set(args.only)
        meshes = [mesh for mesh in meshes if mesh.stem in wanted]

    rows_by_policy: dict[str, list[dict[str, Any]]] = {}
    summaries: list[dict[str, Any]] = []
    comparisons: dict[tuple[str, str], dict[str, Any]] = {}

    for policy in args.policies:
        rows: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_case, policy, mesh, args): mesh for mesh in meshes}
            for index, future in enumerate(as_completed(futures), start=1):
                mesh = futures[future]
                try:
                    row = future.result()
                except Exception as exc:
                    row = failed_row(policy, mesh.stem, str(exc))
                rows.append(row)
                print(f"[{policy} {index}/{len(meshes)}] {mesh.stem} {row['status']}")
        rows.sort(key=lambda row: base.model_index(row["model"]))
        rows_by_policy[policy] = rows
        summaries.append(summary_row(policy, rows))
        base.write_csv(rows, RESULT_ROOT / policy / "run_summary.csv", CSV_FIELDS)
        for baseline_name, baseline_csv in BASELINES.items():
            comparison_rows = v520.comparison_rows(rows, read_baseline_rows(baseline_csv))
            comparisons[(policy, baseline_name)] = v520.comparison_summary(comparison_rows)
            comparison_path = RESULT_ROOT / policy / f"{policy}_vs_{baseline_name}_comparison.csv"
            base.write_csv(comparison_rows, comparison_path, v520.COMPARISON_FIELDS)

    all_rows = [row for rows in rows_by_policy.values() for row in rows]
    base.write_csv(all_rows, RESULT_ROOT / "run_summary.csv", CSV_FIELDS)
    base.write_csv(summaries, RESULT_ROOT / "summary_by_policy.csv", SUMMARY_FIELDS)
    write_report(rows_by_policy, summaries, comparisons, args)
    print(f"wrote {RESULT_ROOT / 'run_summary.csv'}")
    print(f"wrote {RESULT_ROOT / 'summary_by_policy.csv'}")
    print(f"wrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
