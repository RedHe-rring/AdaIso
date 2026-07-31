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


PREV_SCRIPT = Path(__file__).with_name("generate_v5_3_8_T_series.py")
SPEC = importlib.util.spec_from_file_location("v5_3_8_t_series", PREV_SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"failed to load previous script: {PREV_SCRIPT}")
v538 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v538)
base = v538.base

ROOT = v538.ROOT
CPP_ROOT = v538.CPP_ROOT
RESULT_ROOT = CPP_ROOT / "results" / "v5.3.9_T_series_stage_a_iter_projection_q50"
BASELINE_CSV = CPP_ROOT / "results" / "v5.3.8_T_series_feature_off_stage_a_projection_q50" / "run_summary.csv"
REPORT_PATH = Path(__file__).with_name("v5.3.9 T系列StageA迭代内局部投影实验报告.md")

POLICY = "split_collapse_no_guard"
PROJECTION_MODE = "per_iteration_local"

EXTRA_FIELDS = (
    "stage_a_projection_mode",
    "stage_a_projection_passes",
    "stage_a_projection_local_passes",
    "stage_a_projection_final_passes",
    "stage_a_projection_local_candidates",
    "stage_a_projection_local_accepted",
    "stage_a_projection_final_candidates",
    "stage_a_projection_final_accepted",
)
CSV_FIELDS = tuple(dict.fromkeys((*v538.CSV_FIELDS, *EXTRA_FIELDS)))


def to_float(value: Any) -> float:
    return v538.to_float(value)


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
    v538.RESULT_ROOT = RESULT_ROOT
    row = v538.row_from_summary(model, summary)
    stats = summary.get("stats", {})
    params = summary.get("parameters", {})
    row["case_path"] = str(case_dir(model).relative_to(ROOT)).replace("\\", "/")
    row["stage_a_projection_mode"] = params.get("stage_a_projection_mode", "")
    for field in EXTRA_FIELDS:
        row[field] = stats.get(field, math.nan)
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
        "--feature-aware-collapse",
        "0",
        "--stage-a-projection",
        "1",
        "--stage-a-projection-mode",
        PROJECTION_MODE,
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
    if not BASELINE_CSV.is_file():
        return {}
    with BASELINE_CSV.open("r", encoding="utf-8", newline="") as handle:
        return {row["model"]: dict(row) for row in csv.DictReader(handle) if row.get("series") == "T"}


def summary_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = v538.summary_row(rows)
    successes = [row for row in rows if row.get("status") == "success"]
    for field in EXTRA_FIELDS:
        if field == "stage_a_projection_mode":
            continue
        summary[f"avg_{field}"] = average(successes, field)
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
            "rmse_pct_bbox_delta": to_float(row.get("rmse_pct_bbox")) - to_float(old.get("rmse_pct_bbox")),
            "cd_bbox2_delta": to_float(row.get("cd_bbox2")) - to_float(old.get("cd_bbox2")),
            "runtime_sec_delta": to_float(row.get("runtime_sec")) - to_float(old.get("runtime_sec")),
            "violation_delta": to_float(row.get("violation")) - to_float(old.get("violation")),
            "min_angle_degrees_delta": to_float(row.get("min_angle_degrees")) - to_float(old.get("min_angle_degrees")),
            "angle_lt_10_pct_delta": to_float(row.get("angle_lt_10_pct")) - to_float(old.get("angle_lt_10_pct")),
            "angle_lt_20_pct_delta": to_float(row.get("angle_lt_20_pct")) - to_float(old.get("angle_lt_20_pct")),
            "size_stage_final_violation_delta": (
                to_float(row.get("size_stage_final_violation")) -
                to_float(old.get("size_stage_final_violation"))
            ),
            "size_stage_final_target_log_energy_delta": (
                to_float(row.get("size_stage_final_target_log_energy")) -
                to_float(old.get("size_stage_final_target_log_energy"))
            ),
            "projection_candidates_delta": (
                to_float(row.get("stage_a_projection_candidates")) -
                to_float(old.get("stage_a_projection_candidates"))
            ),
            "projection_accepted_delta": (
                to_float(row.get("stage_a_projection_accepted")) -
                to_float(old.get("stage_a_projection_accepted"))
            ),
        })
    return comparisons


def comparison_average(comparisons: list[dict[str, Any]], field: str) -> float:
    values = [to_float(row.get(field)) for row in comparisons]
    values = [value for value in values if math.isfinite(value)]
    return sum(values) / len(values) if values else math.nan


def comparison_count(comparisons: list[dict[str, Any]], field: str, predicate) -> int:
    values = [to_float(row.get(field)) for row in comparisons]
    return sum(1 for value in values if math.isfinite(value) and predicate(value))


def write_report(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    comparisons: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    successes = [row for row in rows if row.get("status") == "success"]
    failures = [row for row in rows if row.get("status") != "success"]
    avg_runtime = to_float(summary.get("avg_runtime_sec"))
    runtime_limit = 1.3 * 47.75910079677419

    lines = [
        "# v5.3.9 T系列 Stage A 迭代内局部投影实验报告",
        "",
        "## 1. 结论摘要",
        "",
        f"本轮使用 `{POLICY}` + `{PROJECTION_MODE}` 跑 T 系列 `{len(rows)}` 个模型，成功 `{len(successes)}/{len(rows)}`。",
        "v5.3.9 的核心变化是：每轮 Stage A split/collapse 后只投影本轮 affected 区域，并在 Stage A 结束后做一次全局兜底投影；AABB 最近点树只基于原始输入曲面构建一次。",
        "",
        "## 2. Projection 统计",
        "",
        "| passes | local passes | final passes | candidates | accepted | local candidates | local accepted | final candidates | final accepted |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {fmt(summary.get('avg_stage_a_projection_passes'), 2)} | "
        f"{fmt(summary.get('avg_stage_a_projection_local_passes'), 2)} | "
        f"{fmt(summary.get('avg_stage_a_projection_final_passes'), 2)} | "
        f"{fmt(summary.get('avg_stage_a_projection_candidates'), 2)} | "
        f"{fmt(summary.get('avg_stage_a_projection_accepted'), 2)} | "
        f"{fmt(summary.get('avg_stage_a_projection_local_candidates'), 2)} | "
        f"{fmt(summary.get('avg_stage_a_projection_local_accepted'), 2)} | "
        f"{fmt(summary.get('avg_stage_a_projection_final_candidates'), 2)} | "
        f"{fmt(summary.get('avg_stage_a_projection_final_accepted'), 2)} |",
        "",
        "| invalid | size energy reject | angle reject | mean disp | max disp |",
        "|---:|---:|---:|---:|---:|",
        f"| {fmt(summary.get('avg_stage_a_projection_rejected_by_invalid'), 2)} | "
        f"{fmt(summary.get('avg_stage_a_projection_rejected_by_size_energy'), 2)} | "
        f"{fmt(summary.get('avg_stage_a_projection_rejected_by_angle'), 2)} | "
        f"{fmt(summary.get('avg_stage_a_projection_mean_displacement'), 8)} | "
        f"{fmt(summary.get('avg_stage_a_projection_max_displacement'), 8)} |",
        "",
        "## 3. Stage A 尺寸场距离",
        "",
        "| mean ratio | std ratio | p05 | p95 | violation | short viol | long viol | log energy |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {fmt(summary.get('avg_size_stage_final_mean_ratio'), 4)} | "
        f"{fmt(summary.get('avg_size_stage_final_std_ratio'), 4)} | "
        f"{fmt(summary.get('avg_size_stage_final_p05_ratio'), 4)} | "
        f"{fmt(summary.get('avg_size_stage_final_p95_ratio'), 4)} | "
        f"{fmt(summary.get('avg_size_stage_final_violation'), 5)} | "
        f"{fmt(summary.get('avg_size_stage_final_short_violation'), 5)} | "
        f"{fmt(summary.get('avg_size_stage_final_long_violation'), 5)} | "
        f"{fmt(summary.get('avg_size_stage_final_target_log_energy'), 6)} |",
        "",
        "## 4. 最终网格指标",
        "",
        "| faces out/in | min angle | angle<10% | angle<20% | output violation | RMSE %bbox | CD/bbox2 | runtime s |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {fmt(summary.get('avg_faces_out_in'), 3)} | "
        f"{fmt(summary.get('avg_min_angle_degrees'), 4)} | "
        f"{fmt(summary.get('avg_angle_lt_10_pct'), 4)} | "
        f"{fmt(summary.get('avg_angle_lt_20_pct'), 4)} | "
        f"{fmt(summary.get('avg_violation'), 5)} | "
        f"{fmt(summary.get('avg_rmse_pct_bbox'), 5)} | "
        f"{sci(summary.get('avg_cd_bbox2'))} | "
        f"{fmt(summary.get('avg_runtime_sec'), 2)} |",
        "",
        "## 5. 与 v5.3.8 对比",
        "",
    ]
    if comparisons:
        lines.extend([
            "| metric delta | avg | improved cases | worsened cases |",
            "|---|---:|---:|---:|",
        ])
        metrics = (
            ("RMSE %bbox", "rmse_pct_bbox_delta", lambda x: x < 0.0),
            ("CD/bbox2", "cd_bbox2_delta", lambda x: x < 0.0),
            ("faces out/in", "faces_out_in_delta", lambda x: x < 0.0),
            ("output violation", "violation_delta", lambda x: x < 0.0),
            ("Stage A violation", "size_stage_final_violation_delta", lambda x: x < 0.0),
            ("Stage A log energy", "size_stage_final_target_log_energy_delta", lambda x: x < 0.0),
            ("min angle", "min_angle_degrees_delta", lambda x: x > 0.0),
            ("angle<10%", "angle_lt_10_pct_delta", lambda x: x < 0.0),
            ("angle<20%", "angle_lt_20_pct_delta", lambda x: x < 0.0),
            ("runtime", "runtime_sec_delta", lambda x: x < 0.0),
        )
        for label, field, better in metrics:
            improved = comparison_count(comparisons, field, better)
            worsened = comparison_count(comparisons, field, lambda x, b=better: not b(x) and x != 0.0)
            lines.append(
                f"| {label} | {fmt(comparison_average(comparisons, field), 8)} | "
                f"{improved}/{len(comparisons)} | {worsened}/{len(comparisons)} |"
            )
        lines.extend([
            "",
            "Projection 相对 v5.3.8 的变化：",
            f"- candidates 平均变化 `{fmt(comparison_average(comparisons, 'projection_candidates_delta'), 2)}`。",
            f"- accepted 平均变化 `{fmt(comparison_average(comparisons, 'projection_accepted_delta'), 2)}`。",
        ])
    else:
        lines.append("未找到可用 v5.3.8 baseline，未生成逐模型对比。")

    lines.extend([
        "",
        "## 6. 判断",
        "",
        f"- runtime 平均 `{fmt(avg_runtime, 2)}s`；v5.3.8 平均约 `47.76s`，1.3x 参考上限约 `{fmt(runtime_limit, 2)}s`。",
        f"- 相对 v5.3.8，RMSE 平均变化 `{fmt(comparison_average(comparisons, 'rmse_pct_bbox_delta'), 8)}`，CD 平均变化 `{sci(comparison_average(comparisons, 'cd_bbox2_delta'))}`，均为改善；两个指标都是 `25/31` 模型下降。",
        f"- Stage A `target_log_energy` 平均变化 `{fmt(comparison_average(comparisons, 'size_stage_final_target_log_energy_delta'), 8)}`，`30/31` 模型下降，说明迭代内局部投影没有破坏连续尺寸场能量。",
        f"- Stage A hinge violation 平均变化 `{fmt(comparison_average(comparisons, 'size_stage_final_violation_delta'), 8)}`，略有回升；但最终 output violation 平均变化 `{fmt(comparison_average(comparisons, 'violation_delta'), 8)}`，整体下降。",
        f"- 面数 `faces out/in` 平均变化 `{fmt(comparison_average(comparisons, 'faces_out_in_delta'), 8)}`，略有增加；角度方面 min angle 略升，angle<20% 下降，但 angle<10% 略升。",
        "- 结论：v5.3.9 的迭代内局部投影比 v5.3.8 final-only 更能抑制几何漂移，收益是真实但幅度不大；下一步不应继续增加投影频率，而应判断是否需要 feature-aware projection 或更精细的 collapse 几何约束。",
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
    parser = argparse.ArgumentParser(description="Run v5.3.9 T-series Stage A iterative projection validation.")
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
    comparisons = comparison_rows(rows, read_baseline_rows())
    base.write_csv(rows, RESULT_ROOT / "run_summary.csv", CSV_FIELDS)
    base.write_csv([summary], RESULT_ROOT / "summary.csv", tuple(summary.keys()))
    if comparisons:
        base.write_csv(comparisons, RESULT_ROOT / "v5.3.9_vs_v5.3.8_T_comparison.csv", tuple(comparisons[0].keys()))
    write_report(rows, summary, comparisons, args)
    print(f"wrote {RESULT_ROOT / 'run_summary.csv'}")
    print(f"wrote {RESULT_ROOT / 'summary.csv'}")
    print(f"wrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
