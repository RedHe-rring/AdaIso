from __future__ import annotations

import csv
import importlib.util
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


BASE_SCRIPT = Path(__file__).with_name("generate_v5_2_0_T_series.py")
SPEC = importlib.util.spec_from_file_location("v5_2_0_t_series", BASE_SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"failed to load base script: {BASE_SCRIPT}")
v520 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v520)

ROOT = v520.ROOT
CPP_ROOT = v520.CPP_ROOT
RESULT_ROOT = CPP_ROOT / "results" / "v5.3.0_T_series_staged_remesh_q50"
BASELINE_ROOT = CPP_ROOT / "results" / "v5.2.1_T_series_min_angle_safety_q50"
BASELINE_CSV = BASELINE_ROOT / "run_summary.csv"
REPORT_PATH = CPP_ROOT / "修改计划与实验结果" / "v5.3.0 T系列阶段化remesh实验报告.md"

CSV_FIELDS = v520.CSV_FIELDS + (
    "relocation_count",
    "size_stage_iterations",
    "size_stage_splits",
    "size_stage_collapses",
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


def fmt(value: Any, digits: int = 6) -> str:
    number = to_float(value)
    return f"{number:.{digits}f}" if math.isfinite(number) else ""


def sci(value: Any) -> str:
    number = to_float(value)
    return f"{number:.6E}" if math.isfinite(number) else ""


def average(rows: list[dict[str, Any]], key: str) -> float:
    values = [to_float(row.get(key)) for row in rows]
    values = [value for value in values if math.isfinite(value)]
    return sum(values) / len(values) if values else math.nan


def case_dir(model: str) -> Path:
    return RESULT_ROOT / model


def row_from_summary(model: str, summary: dict[str, Any]) -> dict[str, Any]:
    row = v520.ORIGINAL_ROW_FROM_SUMMARY(model, summary)
    row["case_path"] = str(case_dir(model).relative_to(ROOT)).replace("\\", "/")
    output_summary = summary.get("output_summary", {})
    stats = summary.get("stats", {})
    iterations = summary.get("iterations", [])
    row["min_angle_degrees"] = output_summary.get("min_angle_degrees", math.nan)
    row["angle_lt_10_pct"] = output_summary.get("angle_lt_10_pct", math.nan)
    row["angle_lt_20_pct"] = output_summary.get("angle_lt_20_pct", math.nan)
    row["relocation_count"] = sum(int(item.get("relocation_count", 0) or 0) for item in iterations)
    row["size_stage_iterations"] = stats.get("size_stage_iterations", math.nan)
    row["size_stage_splits"] = stats.get("size_stage_splits", math.nan)
    row["size_stage_collapses"] = stats.get("size_stage_collapses", math.nan)
    row["quality_stage_iterations"] = stats.get("quality_stage_iterations", math.nan)
    row["quality_stage_flips"] = stats.get("quality_stage_flips", math.nan)
    row["quality_stage_relocations"] = stats.get("quality_stage_relocations", math.nan)
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


def read_baseline_rows() -> dict[str, dict[str, Any]]:
    with BASELINE_CSV.open("r", encoding="utf-8", newline="") as handle:
        rows = {row["model"]: dict(row) for row in csv.DictReader(handle) if row.get("series") == "T"}
    for model, row in rows.items():
        stats = v520.mesh_angle_stats(BASELINE_ROOT / model / "output.ply")
        row.update(stats)
    return rows


def write_report(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    comparisons: list[dict[str, Any]],
    comp: dict[str, Any],
    args: Any,
) -> None:
    successes = [row for row in rows if row.get("status") == "success"]
    failures = [row for row in rows if row.get("status") != "success"]
    common = int(comp.get("cases", 0) or 0)
    counts = v520.better_counts(comparisons)
    successful_comparisons = [row for row in comparisons if row.get("status") == "success"]

    lines = [
        "# v5.3.0 T 系列阶段化 remesh 实验报告",
        "",
        "## 1. 结论摘要",
        "",
        f"本轮使用当前工作区 v5.3.0 代码跑 T 系列 `{len(rows)}` 个模型，成功 `{len(successes)}/{len(rows)}`。对比基线为 v5.2.1 的 `v5.2.1_T_series_min_angle_safety_q50` 结果。",
        "",
        "v5.3.0 将 remesh 从单一混合循环拆成两个阶段：Stage A 只执行 split/collapse 逼近尺寸场，Stage B 只执行 flip/relocation 修复质量。实验重点是看这种结构拆分是否能在保住 v5.2.1 面数和 violation 优势的同时回收最小角质量。",
        "",
        f"- 平均最小角：v5.3.0 `{fmt(comp.get('avg_min_angle_degrees_current'), 4)}°`，v5.2.1 `{fmt(comp.get('avg_min_angle_degrees_baseline'), 4)}°`，变化 `{fmt(comp.get('avg_min_angle_degrees_delta'), 4)}°`。",
        f"- 平均 angle<10%：v5.3.0 `{fmt(comp.get('avg_angle_lt_10_pct_current'), 4)}%`，v5.2.1 `{fmt(comp.get('avg_angle_lt_10_pct_baseline'), 4)}%`，变化 `{fmt(comp.get('avg_angle_lt_10_pct_delta'), 4)}%`。",
        f"- 平均 angle<20%：v5.3.0 `{fmt(comp.get('avg_angle_lt_20_pct_current'), 4)}%`，v5.2.1 `{fmt(comp.get('avg_angle_lt_20_pct_baseline'), 4)}%`，变化 `{fmt(comp.get('avg_angle_lt_20_pct_delta'), 4)}%`。",
        f"- 平均 faces out/in：v5.3.0 `{fmt(comp.get('avg_faces_out_in_current'), 3)}`，v5.2.1 `{fmt(comp.get('avg_faces_out_in_baseline'), 3)}`，变化 `{fmt(comp.get('avg_faces_out_in_delta'), 3)}`。",
        f"- 平均 RMSE %bbox：v5.3.0 `{fmt(comp.get('avg_rmse_pct_bbox_current'), 5)}`，v5.2.1 `{fmt(comp.get('avg_rmse_pct_bbox_baseline'), 5)}`，变化 `{fmt(comp.get('avg_rmse_pct_bbox_delta'), 5)}`。",
        f"- 平均 CD/bbox^2：v5.3.0 `{sci(comp.get('avg_cd_bbox2_current'))}`，v5.2.1 `{sci(comp.get('avg_cd_bbox2_baseline'))}`，变化 `{fmt(comp.get('avg_cd_bbox2_delta_pct'), 2)}%`。",
        f"- 平均 violation：v5.3.0 `{fmt(comp.get('avg_violation_current'), 5)}`，v5.2.1 `{fmt(comp.get('avg_violation_baseline'), 5)}`，变化 `{fmt(comp.get('avg_violation_delta'), 5)}`。",
        "",
        "## 2. 实验设置",
        "",
        f"- CLI：`{args.cli}`",
        f"- 输出目录：`{RESULT_ROOT.relative_to(ROOT).as_posix()}`",
        f"- 当前汇总 CSV：`{(RESULT_ROOT / 'run_summary.csv').relative_to(ROOT).as_posix()}`",
        f"- 对比 CSV：`{(RESULT_ROOT / 'v5.3.0_vs_v5.2.1_T_comparison.csv').relative_to(ROOT).as_posix()}`",
        f"- 基线 CSV：`{BASELINE_CSV.relative_to(ROOT).as_posix()}`",
        f"- 参数：`split={v520.base.FIXED_SPLIT_FACTOR}, collapse={v520.base.FIXED_COLLAPSE_FACTOR}, hmax={v520.base.FIXED_H_MAX_FACTOR}, auto_delta_quantile={v520.base.AUTO_DELTA_QUANTILE}, iters={args.iters}`",
        f"- CD samples：`{args.cd_samples}`；workers：`{args.workers}`",
        "",
        "## 3. 总体指标",
        "",
        "| cases | successes | faces out/in | RMSE %bbox | CD/bbox^2 | qavg | qmin avg/min | min angle avg/min | angle<10% | angle<20% | violation avg/max | runtime s |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {summary['cases']} | {summary['successes']} | {fmt(summary['avg_faces_out_in'], 3)} | {fmt(summary['avg_rmse_pct_bbox'], 5)} | {sci(summary['avg_cd_bbox2'])} | {fmt(summary['avg_qavg'], 5)} | {fmt(summary['avg_qmin'], 5)}/{fmt(summary['min_qmin'], 5)} | {fmt(summary['avg_min_angle_degrees'], 4)}/{fmt(summary['min_min_angle_degrees'], 4)} | {fmt(summary['avg_angle_lt_10_pct'], 4)} | {fmt(summary['avg_angle_lt_20_pct'], 4)} | {fmt(summary['avg_violation'], 5)}/{fmt(summary['max_violation'], 5)} | {fmt(summary['avg_runtime_sec'], 2)} |",
        "",
        "阶段统计均值：",
        "",
        "| size iters | size split | size collapse | quality iters | quality flip | quality relocate |",
        "|---:|---:|---:|---:|---:|---:|",
        f"| {fmt(average(successes, 'size_stage_iterations'), 2)} | {fmt(average(successes, 'size_stage_splits'), 2)} | {fmt(average(successes, 'size_stage_collapses'), 2)} | {fmt(average(successes, 'quality_stage_iterations'), 2)} | {fmt(average(successes, 'quality_stage_flips'), 2)} | {fmt(average(successes, 'quality_stage_relocations'), 2)} |",
        "",
        "## 4. 与 v5.2.1 对比",
        "",
        f"共同成功 T 模型数：`{common}`。",
        "",
        "| metric | v5.3.0 | v5.2.1 | delta |",
        "|---|---:|---:|---:|",
        f"| min angle degrees | {fmt(comp.get('avg_min_angle_degrees_current'), 4)} | {fmt(comp.get('avg_min_angle_degrees_baseline'), 4)} | {fmt(comp.get('avg_min_angle_degrees_delta'), 4)} |",
        f"| angle<10 pct | {fmt(comp.get('avg_angle_lt_10_pct_current'), 4)} | {fmt(comp.get('avg_angle_lt_10_pct_baseline'), 4)} | {fmt(comp.get('avg_angle_lt_10_pct_delta'), 4)} |",
        f"| angle<20 pct | {fmt(comp.get('avg_angle_lt_20_pct_current'), 4)} | {fmt(comp.get('avg_angle_lt_20_pct_baseline'), 4)} | {fmt(comp.get('avg_angle_lt_20_pct_delta'), 4)} |",
        f"| faces out/in | {fmt(comp.get('avg_faces_out_in_current'), 3)} | {fmt(comp.get('avg_faces_out_in_baseline'), 3)} | {fmt(comp.get('avg_faces_out_in_delta'), 3)} |",
        f"| output faces | {fmt(comp.get('avg_output_faces_current'), 2)} | {fmt(comp.get('avg_output_faces_baseline'), 2)} | {fmt(comp.get('avg_output_faces_delta'), 2)} |",
        f"| RMSE %bbox | {fmt(comp.get('avg_rmse_pct_bbox_current'), 5)} | {fmt(comp.get('avg_rmse_pct_bbox_baseline'), 5)} | {fmt(comp.get('avg_rmse_pct_bbox_delta'), 5)} |",
        f"| CD/bbox^2 | {sci(comp.get('avg_cd_bbox2_current'))} | {sci(comp.get('avg_cd_bbox2_baseline'))} | {fmt(comp.get('avg_cd_bbox2_delta_pct'), 2)}% |",
        f"| qavg | {fmt(comp.get('avg_qavg_current'), 5)} | {fmt(comp.get('avg_qavg_baseline'), 5)} | {fmt(comp.get('avg_qavg_delta'), 5)} |",
        f"| qmin | {fmt(comp.get('avg_qmin_current'), 5)} | {fmt(comp.get('avg_qmin_baseline'), 5)} | {fmt(comp.get('avg_qmin_delta'), 5)} |",
        f"| violation | {fmt(comp.get('avg_violation_current'), 5)} | {fmt(comp.get('avg_violation_baseline'), 5)} | {fmt(comp.get('avg_violation_delta'), 5)} |",
        f"| runtime s | {fmt(comp.get('avg_runtime_sec_current'), 2)} | {fmt(comp.get('avg_runtime_sec_baseline'), 2)} | {fmt(comp.get('avg_runtime_sec_delta'), 2)} |",
        "",
        "逐模型胜出计数：",
        "",
        "| criterion | v5.3.0 better / common |",
        "|---|---:|",
        f"| higher min angle | {counts['higher_min_angle']}/{common} |",
        f"| lower angle<10 pct | {counts['lower_angle10']}/{common} |",
        f"| lower angle<20 pct | {counts['lower_angle20']}/{common} |",
        f"| lower faces out/in | {counts['lower_faces']}/{common} |",
        f"| lower RMSE %bbox | {counts['lower_rmse']}/{common} |",
        f"| lower CD/bbox^2 | {counts['lower_cd']}/{common} |",
        f"| higher qavg | {counts['higher_qavg']}/{common} |",
        f"| higher qmin | {counts['higher_qmin']}/{common} |",
        f"| lower violation | {counts['lower_violation']}/{common} |",
        f"| faster runtime | {counts['faster']}/{common} |",
        "",
        "## 5. 变化最大的模型",
        "",
        "### 5.1 最小角提升最多",
        "",
        "| model | min angle delta | angle<10 delta | angle<20 delta | faces delta | violation delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(successful_comparisons, key=lambda item: to_float(item.get("min_angle_degrees_delta")), reverse=True)[:10]:
        lines.append(f"| {row['model']} | {fmt(row['min_angle_degrees_delta'], 4)} | {fmt(row['angle_lt_10_pct_delta'], 4)} | {fmt(row['angle_lt_20_pct_delta'], 4)} | {fmt(row['faces_out_in_delta'], 3)} | {fmt(row['violation_delta'], 5)} |")
    lines.extend([
        "",
        "### 5.2 violation 恶化最多",
        "",
        "| model | violation delta | faces delta | min angle delta | RMSE delta | CD delta % |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in sorted(successful_comparisons, key=lambda item: to_float(item.get("violation_delta")), reverse=True)[:10]:
        lines.append(f"| {row['model']} | {fmt(row['violation_delta'], 5)} | {fmt(row['faces_out_in_delta'], 3)} | {fmt(row['min_angle_degrees_delta'], 4)} | {fmt(row['rmse_pct_bbox_delta'], 5)} | {fmt(row['cd_bbox2_delta_pct'], 2)} |")

    lines.extend(["", "## 6. 内部判断", ""])
    faces_delta = to_float(comp.get("avg_faces_out_in_delta"))
    violation_delta = to_float(comp.get("avg_violation_delta"))
    min_angle_delta = to_float(comp.get("avg_min_angle_degrees_delta"))
    angle10_delta = to_float(comp.get("avg_angle_lt_10_pct_delta"))
    if min_angle_delta > 0 and angle10_delta < 0:
        lines.append("- v5.3.0 的阶段拆分确实回收了坏角质量，说明把 flip/relocation 从尺寸场循环中拿出来作为质量阶段是有效方向。")
    elif min_angle_delta > 0:
        lines.append("- v5.3.0 提升了平均最小角，但低于 10° 的角比例没有同步改善，坏角修复仍然不够集中。")
    else:
        lines.append("- v5.3.0 没有取得平均最小角收益，说明 Stage B 当前接受准则或触发候选还不足。")
    if faces_delta <= 0 and violation_delta <= 0:
        lines.append("- v5.3.0 同时保住了 v5.2.1 的面数和尺寸场 violation 优势，阶段化结构可以作为主线推进。")
    elif violation_delta > 0:
        lines.append("- v5.3.0 的质量阶段对尺寸场 violation 有破坏，需要收紧 Stage B 的 metric 约束或减少无效 relocation。")
    else:
        lines.append("- v5.3.0 的面数或运行时间代价需要继续观察，但目标分层已经比混合循环更容易诊断。")
    lines.append("- 下一步应优先看 Stage A/Stage B 的分阶段指标：如果 Stage A 已经稳定贴合尺寸场，而 Stage B 改善质量不足，就应增强坏角候选选择；如果 Stage B 破坏 violation，则应降低质量阶段自由度。")
    if failures:
        lines.extend(["", "## 7. 失败项", "", "| model | error |", "|---|---|"])
        for row in failures:
            lines.append(f"| {row['model']} | {row['error']} |")
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = v520.base.parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")

    v520.base.RESULT_ROOT = RESULT_ROOT
    v520.base.REPORT_PATH = REPORT_PATH
    v520.base.BASELINE_CSV = BASELINE_CSV
    v520.base.CSV_FIELDS = CSV_FIELDS
    v520.base.SUMMARY_FIELDS = v520.SUMMARY_FIELDS
    v520.base.COMPARISON_FIELDS = v520.COMPARISON_FIELDS
    v520.base.case_dir = case_dir
    v520.base.row_from_summary = row_from_summary
    v520.base.failed_row = failed_row

    meshes = v520.base.discover_t_models()
    if args.only:
        wanted = set(args.only)
        meshes = [mesh for mesh in meshes if mesh.stem in wanted]

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(v520.base.run_case, mesh, args): mesh for mesh in meshes}
        for index, future in enumerate(as_completed(futures), start=1):
            mesh = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = failed_row(mesh.stem, str(exc))
            rows.append(row)
            print(f"[{index}/{len(meshes)}] {mesh.stem} {row['status']}")

    rows.sort(key=lambda row: v520.base.model_index(row["model"]))
    summary = v520.summary_row(rows)
    baseline = read_baseline_rows()
    comparisons = v520.comparison_rows(rows, baseline)
    comp_summary = v520.comparison_summary(comparisons)
    v520.base.write_csv(rows, RESULT_ROOT / "run_summary.csv", CSV_FIELDS)
    v520.base.write_csv([summary], RESULT_ROOT / "summary.csv", v520.SUMMARY_FIELDS)
    v520.base.write_csv(comparisons, RESULT_ROOT / "v5.3.0_vs_v5.2.1_T_comparison.csv", v520.COMPARISON_FIELDS)
    write_report(rows, summary, comparisons, comp_summary, args)
    print(f"wrote {RESULT_ROOT / 'run_summary.csv'}")
    print(f"wrote {RESULT_ROOT / 'summary.csv'}")
    print(f"wrote {RESULT_ROOT / 'v5.3.0_vs_v5.2.1_T_comparison.csv'}")
    print(f"wrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
