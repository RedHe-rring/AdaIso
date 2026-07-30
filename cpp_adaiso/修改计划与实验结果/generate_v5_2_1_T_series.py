from __future__ import annotations

import csv
import importlib.util
import json
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
RESULT_ROOT = CPP_ROOT / "results" / "v5.2.1_T_series_min_angle_safety_q50"
BASELINE_ROOT = CPP_ROOT / "results" / "v5.2.0_T_series_min_angle_q50"
BASELINE_CSV = BASELINE_ROOT / "run_summary.csv"
REPORT_PATH = CPP_ROOT / "修改计划与实验结果" / "v5.2.1 T系列最小角安全约束实验报告.md"


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
    row["min_angle_degrees"] = output_summary.get("min_angle_degrees", math.nan)
    row["angle_lt_10_pct"] = output_summary.get("angle_lt_10_pct", math.nan)
    row["angle_lt_20_pct"] = output_summary.get("angle_lt_20_pct", math.nan)
    return row


def failed_row(model: str, error: str) -> dict[str, Any]:
    row = {field: "" for field in v520.CSV_FIELDS}
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
        "# v5.2.1 T 系列最小角安全约束实验报告",
        "",
        "## 1. 结论摘要",
        "",
        f"本轮使用当前工作区 v5.2.1 代码跑 T 系列 `{len(rows)}` 个模型，成功 `{len(successes)}/{len(rows)}`。对比基线为 v5.2.0 的 `v5.2.0_T_series_min_angle_q50` 结果。",
        "",
        "总体判断：v5.2.1 把“必须提升最小角”替换为“禁止生成比当前局部更坏的角”。这降低了 v5.2.0 的强优化色彩，实验重点是看它能否保住最小角收益，同时回收面数、violation 和运行时间。",
        "",
        f"- 平均最小角：v5.2.1 `{fmt(comp.get('avg_min_angle_degrees_current'), 4)}°`，v5.2.0 `{fmt(comp.get('avg_min_angle_degrees_baseline'), 4)}°`，变化 `{fmt(comp.get('avg_min_angle_degrees_delta'), 4)}°`。",
        f"- 平均 angle<10%：v5.2.1 `{fmt(comp.get('avg_angle_lt_10_pct_current'), 4)}%`，v5.2.0 `{fmt(comp.get('avg_angle_lt_10_pct_baseline'), 4)}%`，变化 `{fmt(comp.get('avg_angle_lt_10_pct_delta'), 4)}%`。",
        f"- 平均 angle<20%：v5.2.1 `{fmt(comp.get('avg_angle_lt_20_pct_current'), 4)}%`，v5.2.0 `{fmt(comp.get('avg_angle_lt_20_pct_baseline'), 4)}%`，变化 `{fmt(comp.get('avg_angle_lt_20_pct_delta'), 4)}%`。",
        f"- 平均 faces out/in：v5.2.1 `{fmt(comp.get('avg_faces_out_in_current'), 3)}`，v5.2.0 `{fmt(comp.get('avg_faces_out_in_baseline'), 3)}`，变化 `{fmt(comp.get('avg_faces_out_in_delta'), 3)}`。",
        f"- 平均 RMSE %bbox：v5.2.1 `{fmt(comp.get('avg_rmse_pct_bbox_current'), 5)}`，v5.2.0 `{fmt(comp.get('avg_rmse_pct_bbox_baseline'), 5)}`，变化 `{fmt(comp.get('avg_rmse_pct_bbox_delta'), 5)}`。",
        f"- 平均 CD/bbox^2：v5.2.1 `{sci(comp.get('avg_cd_bbox2_current'))}`，v5.2.0 `{sci(comp.get('avg_cd_bbox2_baseline'))}`，变化 `{fmt(comp.get('avg_cd_bbox2_delta_pct'), 2)}%`。",
        f"- 平均 violation：v5.2.1 `{fmt(comp.get('avg_violation_current'), 5)}`，v5.2.0 `{fmt(comp.get('avg_violation_baseline'), 5)}`，变化 `{fmt(comp.get('avg_violation_delta'), 5)}`。",
        "",
        "## 2. 实验设置",
        "",
        f"- CLI：`{args.cli}`",
        f"- 输出目录：`{RESULT_ROOT.relative_to(ROOT).as_posix()}`",
        f"- 当前汇总 CSV：`{(RESULT_ROOT / 'run_summary.csv').relative_to(ROOT).as_posix()}`",
        f"- 对比 CSV：`{(RESULT_ROOT / 'v5.2.1_vs_v5.2.0_T_comparison.csv').relative_to(ROOT).as_posix()}`",
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
        "## 4. 与 v5.2.0 对比",
        "",
        f"共同成功 T 模型数：`{common}`。",
        "",
        "| metric | v5.2.1 | v5.2.0 | delta |",
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
        "| criterion | v5.2.1 better / common |",
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
        "### 5.1 面数下降最多",
        "",
        "| model | faces delta | min angle delta | angle<10 delta | violation delta | RMSE delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(successful_comparisons, key=lambda item: to_float(item.get("faces_out_in_delta")))[:10]:
        lines.append(f"| {row['model']} | {fmt(row['faces_out_in_delta'], 3)} | {fmt(row['min_angle_degrees_delta'], 4)} | {fmt(row['angle_lt_10_pct_delta'], 4)} | {fmt(row['violation_delta'], 5)} | {fmt(row['rmse_pct_bbox_delta'], 5)} |")
    lines.extend([
        "",
        "### 5.2 最小角恶化最多",
        "",
        "| model | min angle v5.2.1 | v5.2.0 | delta | angle<10 delta | faces delta |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in sorted(successful_comparisons, key=lambda item: to_float(item.get("min_angle_degrees_delta")))[:10]:
        lines.append(f"| {row['model']} | {fmt(row['min_angle_degrees_current'], 4)} | {fmt(row['min_angle_degrees_baseline'], 4)} | {fmt(row['min_angle_degrees_delta'], 4)} | {fmt(row['angle_lt_10_pct_delta'], 4)} | {fmt(row['faces_out_in_delta'], 3)} |")
    lines.extend([
        "",
        "## 6. 内部判断",
        "",
    ])
    faces_delta = to_float(comp.get("avg_faces_out_in_delta"))
    violation_delta = to_float(comp.get("avg_violation_delta"))
    min_angle_delta = to_float(comp.get("avg_min_angle_degrees_delta"))
    angle10_delta = to_float(comp.get("avg_angle_lt_10_pct_delta"))
    if faces_delta < 0 and violation_delta < 0:
        lines.append("- v5.2.1 相比 v5.2.0 回收了面数和尺寸场 violation，说明“禁止变坏”比“必须变好”更适合作为安全约束。")
    elif faces_delta < 0:
        lines.append("- v5.2.1 回收了部分面数，但 violation 仍未改善到理想水平；尺寸场贴合仍需检查。")
    else:
        lines.append("- v5.2.1 没有回收 v5.2.0 的面数代价，说明当前安全约束仍然过强或触发位置不对。")
    if min_angle_delta >= 0 or angle10_delta <= 0:
        lines.append("- 最小角收益基本保住，说明替换条件没有把质量控制直接退回 v5.1.5。")
    else:
        lines.append("- 最小角收益回撤明显，说明单纯禁止变坏不足以维持 v5.2.0 的坏角改善。")
    lines.append("- 下一步应同时看 v5.1.5/v5.2.0/v5.2.1 三者：v5.2.1 如果接近 v5.1.5 的面数和 violation，同时保留一部分最小角收益，就比 v5.2.0 更适合作为主线。")
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
    v520.base.CSV_FIELDS = v520.CSV_FIELDS
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
    v520.base.write_csv(rows, RESULT_ROOT / "run_summary.csv", v520.CSV_FIELDS)
    v520.base.write_csv([summary], RESULT_ROOT / "summary.csv", v520.SUMMARY_FIELDS)
    v520.base.write_csv(comparisons, RESULT_ROOT / "v5.2.1_vs_v5.2.0_T_comparison.csv", v520.COMPARISON_FIELDS)
    write_report(rows, summary, comparisons, comp_summary, args)
    print(f"wrote {RESULT_ROOT / 'run_summary.csv'}")
    print(f"wrote {RESULT_ROOT / 'summary.csv'}")
    print(f"wrote {RESULT_ROOT / 'v5.2.1_vs_v5.2.0_T_comparison.csv'}")
    print(f"wrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
