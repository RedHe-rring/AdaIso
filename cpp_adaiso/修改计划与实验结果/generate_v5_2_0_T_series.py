from __future__ import annotations

import csv
import importlib.util
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


BASE_SCRIPT = Path(__file__).with_name("generate_v5_1_5_T_series.py")
SPEC = importlib.util.spec_from_file_location("v5_1_5_t_series", BASE_SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"failed to load base script: {BASE_SCRIPT}")
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)
ORIGINAL_ROW_FROM_SUMMARY = base.row_from_summary

ROOT = base.ROOT
CPP_ROOT = base.CPP_ROOT
RESULT_ROOT = CPP_ROOT / "results" / "v5.2.0_T_series_min_angle_q50"
BASELINE_ROOT = CPP_ROOT / "results" / "v5.1.5_T_series_auto_delta_q50"
BASELINE_CSV = BASELINE_ROOT / "run_summary.csv"
REPORT_PATH = CPP_ROOT / "修改计划与实验结果" / "v5.2.0 T系列最小角质量控制实验报告.md"

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
    "min_angle_degrees",
    "angle_lt_10_pct",
    "angle_lt_20_pct",
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
    "avg_rmse_pct_bbox",
    "avg_cd_bbox2",
    "avg_qavg",
    "avg_qmin",
    "min_qmin",
    "avg_min_angle_degrees",
    "min_min_angle_degrees",
    "avg_angle_lt_10_pct",
    "avg_angle_lt_20_pct",
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
    "min_angle_degrees_current",
    "min_angle_degrees_baseline",
    "min_angle_degrees_delta",
    "angle_lt_10_pct_current",
    "angle_lt_10_pct_baseline",
    "angle_lt_10_pct_delta",
    "angle_lt_20_pct_current",
    "angle_lt_20_pct_baseline",
    "angle_lt_20_pct_delta",
    "violation_current",
    "violation_baseline",
    "violation_delta",
    "runtime_sec_current",
    "runtime_sec_baseline",
    "runtime_sec_delta",
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


def mesh_angle_stats(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        line = handle.readline().strip()
        if line != "ply":
            raise ValueError(f"not a PLY file: {path}")
        vertex_count = face_count = None
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"missing PLY header terminator: {path}")
            text = line.strip()
            if text == "end_header":
                break
            parts = text.split()
            if len(parts) == 3 and parts[0] == "element" and parts[1] == "vertex":
                vertex_count = int(parts[2])
            if len(parts) == 3 and parts[0] == "element" and parts[1] == "face":
                face_count = int(parts[2])
        if vertex_count is None or face_count is None:
            raise ValueError(f"missing vertex/face count in PLY: {path}")
        vertices = []
        for _ in range(vertex_count):
            parts = handle.readline().split()
            vertices.append((float(parts[0]), float(parts[1]), float(parts[2])))
        faces = []
        for _ in range(face_count):
            parts = handle.readline().split()
            n = int(parts[0])
            if n == 3:
                faces.append((int(parts[1]), int(parts[2]), int(parts[3])))

    def angle(p: tuple[float, float, float], q: tuple[float, float, float], r: tuple[float, float, float]) -> float:
        ux, uy, uz = q[0] - p[0], q[1] - p[1], q[2] - p[2]
        vx, vy, vz = r[0] - p[0], r[1] - p[1], r[2] - p[2]
        nu = math.sqrt(ux * ux + uy * uy + uz * uz)
        nv = math.sqrt(vx * vx + vy * vy + vz * vz)
        if nu <= 1e-14 or nv <= 1e-14:
            return 0.0
        cosv = max(-1.0, min(1.0, (ux * vx + uy * vy + uz * vz) / (nu * nv)))
        return math.degrees(math.acos(cosv))

    angles = []
    for i, j, k in faces:
        a, b, c = vertices[i], vertices[j], vertices[k]
        angles.extend((angle(a, b, c), angle(b, c, a), angle(c, a, b)))
    if not angles:
        return {"min_angle_degrees": math.nan, "angle_lt_10_pct": math.nan, "angle_lt_20_pct": math.nan}
    return {
        "min_angle_degrees": min(angles),
        "angle_lt_10_pct": 100.0 * sum(1 for value in angles if value < 10.0) / len(angles),
        "angle_lt_20_pct": 100.0 * sum(1 for value in angles if value < 20.0) / len(angles),
    }


def case_dir(model: str) -> Path:
    return RESULT_ROOT / model


def row_from_summary(model: str, summary: dict[str, Any]) -> dict[str, Any]:
    row = ORIGINAL_ROW_FROM_SUMMARY(model, summary)
    row["case_path"] = str(case_dir(model).relative_to(ROOT)).replace("\\", "/")
    output_summary = summary.get("output_summary", {})
    row["min_angle_degrees"] = output_summary.get("min_angle_degrees", math.nan)
    row["angle_lt_10_pct"] = output_summary.get("angle_lt_10_pct", math.nan)
    row["angle_lt_20_pct"] = output_summary.get("angle_lt_20_pct", math.nan)
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
        stats = mesh_angle_stats(BASELINE_ROOT / model / "output.ply")
        row.update(stats)
    return rows


def summary_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [row for row in rows if row.get("status") == "success"]
    qmins = [to_float(row.get("qmin")) for row in successes]
    qmins = [value for value in qmins if math.isfinite(value)]
    min_angles = [to_float(row.get("min_angle_degrees")) for row in successes]
    min_angles = [value for value in min_angles if math.isfinite(value)]
    violations = [to_float(row.get("violation")) for row in successes]
    violations = [value for value in violations if math.isfinite(value)]
    return {
        "cases": len(rows),
        "successes": len(successes),
        "avg_faces_out_in": average(successes, "faces_out_in"),
        "avg_output_estimated": average(successes, "output_estimated"),
        "avg_rmse_pct_bbox": average(successes, "rmse_pct_bbox"),
        "avg_cd_bbox2": average(successes, "cd_bbox2"),
        "avg_qavg": average(successes, "qavg"),
        "avg_qmin": average(successes, "qmin"),
        "min_qmin": min(qmins) if qmins else math.nan,
        "avg_min_angle_degrees": average(successes, "min_angle_degrees"),
        "min_min_angle_degrees": min(min_angles) if min_angles else math.nan,
        "avg_angle_lt_10_pct": average(successes, "angle_lt_10_pct"),
        "avg_angle_lt_20_pct": average(successes, "angle_lt_20_pct"),
        "avg_violation": average(successes, "violation"),
        "max_violation": max(violations) if violations else math.nan,
        "avg_runtime_sec": average(successes, "runtime_sec"),
    }


def comparison_rows(rows: list[dict[str, Any]], baseline_by_model: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    comparisons = []
    for row in rows:
        model = row["model"]
        base_row = baseline_by_model.get(model)
        if row.get("status") != "success" or base_row is None:
            comparisons.append({"model": model, "status": "missing_baseline" if base_row is None else row.get("status")})
            continue

        out = {"model": model, "status": "success"}
        for key in (
            "faces_out_in",
            "output_faces",
            "rmse_pct_bbox",
            "qavg",
            "qmin",
            "min_angle_degrees",
            "angle_lt_10_pct",
            "angle_lt_20_pct",
            "violation",
            "runtime_sec",
        ):
            cur = to_float(row.get(key))
            old = to_float(base_row.get(key))
            out[f"{key}_current"] = cur
            out[f"{key}_baseline"] = old
            out[f"{key}_delta"] = cur - old if math.isfinite(cur) and math.isfinite(old) else math.nan
        cd_cur = to_float(row.get("cd_bbox2"))
        cd_old = to_float(base_row.get("cd_bbox2"))
        out["cd_bbox2_current"] = cd_cur
        out["cd_bbox2_baseline"] = cd_old
        out["cd_bbox2_delta_pct"] = 100.0 * (cd_cur - cd_old) / cd_old if math.isfinite(cd_cur) and math.isfinite(cd_old) and cd_old else math.nan
        comparisons.append(out)
    return comparisons


def comparison_summary(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [row for row in comparisons if row.get("status") == "success"]
    summary: dict[str, Any] = {"cases": len(successes)}
    for key in (
        "faces_out_in",
        "output_faces",
        "rmse_pct_bbox",
        "qavg",
        "qmin",
        "min_angle_degrees",
        "angle_lt_10_pct",
        "angle_lt_20_pct",
        "violation",
        "runtime_sec",
    ):
        summary[f"avg_{key}_current"] = average(successes, f"{key}_current")
        summary[f"avg_{key}_baseline"] = average(successes, f"{key}_baseline")
        summary[f"avg_{key}_delta"] = average(successes, f"{key}_delta")
    summary["avg_cd_bbox2_current"] = average(successes, "cd_bbox2_current")
    summary["avg_cd_bbox2_baseline"] = average(successes, "cd_bbox2_baseline")
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
        "higher_min_angle": sum(1 for row in successes if to_float(row.get("min_angle_degrees_delta")) > 0.0),
        "lower_angle10": sum(1 for row in successes if to_float(row.get("angle_lt_10_pct_delta")) < 0.0),
        "lower_angle20": sum(1 for row in successes if to_float(row.get("angle_lt_20_pct_delta")) < 0.0),
        "lower_violation": sum(1 for row in successes if to_float(row.get("violation_delta")) < 0.0),
        "faster": sum(1 for row in successes if to_float(row.get("runtime_sec_delta")) < 0.0),
    }


def write_report(rows: list[dict[str, Any]], summary: dict[str, Any], comparisons: list[dict[str, Any]], comp: dict[str, Any], args: Any) -> None:
    successes = [row for row in rows if row.get("status") == "success"]
    failures = [row for row in rows if row.get("status") != "success"]
    common = int(comp.get("cases", 0) or 0)
    counts = better_counts(comparisons)

    lines = [
        "# v5.2.0 T 系列最小角质量控制实验报告",
        "",
        "## 1. 结论摘要",
        "",
        f"本轮使用当前工作区 v5.2.0 代码跑 T 系列 `{len(rows)}` 个模型，成功 `{len(successes)}/{len(rows)}`。对比基线为已提交 v5.1.5 的 `v5.1.5_T_series_auto_delta_q50` 结果。",
        "",
        "总体判断：v5.2.0 在“最小角/极小角”这个局部质量目标上有提升，但综合性能没有提升。主要代价是输出面数、target metric violation 和运行时间明显上升，平均 qavg 下降。",
        "",
        f"- 平均最小角：v5.2.0 `{fmt(comp.get('avg_min_angle_degrees_current'), 4)}°`，v5.1.5 `{fmt(comp.get('avg_min_angle_degrees_baseline'), 4)}°`，变化 `{fmt(comp.get('avg_min_angle_degrees_delta'), 4)}°`。",
        f"- 平均 angle<10%：v5.2.0 `{fmt(comp.get('avg_angle_lt_10_pct_current'), 4)}%`，v5.1.5 `{fmt(comp.get('avg_angle_lt_10_pct_baseline'), 4)}%`，变化 `{fmt(comp.get('avg_angle_lt_10_pct_delta'), 4)}%`。",
        f"- 平均 angle<20%：v5.2.0 `{fmt(comp.get('avg_angle_lt_20_pct_current'), 4)}%`，v5.1.5 `{fmt(comp.get('avg_angle_lt_20_pct_baseline'), 4)}%`，变化 `{fmt(comp.get('avg_angle_lt_20_pct_delta'), 4)}%`。",
        f"- 平均 faces out/in：v5.2.0 `{fmt(comp.get('avg_faces_out_in_current'), 3)}`，v5.1.5 `{fmt(comp.get('avg_faces_out_in_baseline'), 3)}`，变化 `{fmt(comp.get('avg_faces_out_in_delta'), 3)}`。",
        f"- 平均 RMSE %bbox：v5.2.0 `{fmt(comp.get('avg_rmse_pct_bbox_current'), 5)}`，v5.1.5 `{fmt(comp.get('avg_rmse_pct_bbox_baseline'), 5)}`，变化 `{fmt(comp.get('avg_rmse_pct_bbox_delta'), 5)}`。",
        f"- 平均 CD/bbox^2：v5.2.0 `{sci(comp.get('avg_cd_bbox2_current'))}`，v5.1.5 `{sci(comp.get('avg_cd_bbox2_baseline'))}`，变化 `{fmt(comp.get('avg_cd_bbox2_delta_pct'), 2)}%`。",
        f"- 平均 violation：v5.2.0 `{fmt(comp.get('avg_violation_current'), 5)}`，v5.1.5 `{fmt(comp.get('avg_violation_baseline'), 5)}`，变化 `{fmt(comp.get('avg_violation_delta'), 5)}`。",
        "",
        "## 2. 实验设置",
        "",
        f"- CLI：`{args.cli}`",
        f"- 输出目录：`{RESULT_ROOT.relative_to(ROOT).as_posix()}`",
        f"- 当前汇总 CSV：`{(RESULT_ROOT / 'run_summary.csv').relative_to(ROOT).as_posix()}`",
        f"- 对比 CSV：`{(RESULT_ROOT / 'v5.2.0_vs_v5.1.5_T_comparison.csv').relative_to(ROOT).as_posix()}`",
        f"- 基线 CSV：`{BASELINE_CSV.relative_to(ROOT).as_posix()}`",
        f"- 参数：`split={base.FIXED_SPLIT_FACTOR}, collapse={base.FIXED_COLLAPSE_FACTOR}, hmax={base.FIXED_H_MAX_FACTOR}, auto_delta_quantile={base.AUTO_DELTA_QUANTILE}, iters={args.iters}`",
        f"- CD samples：`{args.cd_samples}`；workers：`{args.workers}`",
        "",
        "## 3. 总体指标",
        "",
        "| cases | successes | faces out/in | RMSE %bbox | CD/bbox^2 | qavg | qmin avg/min | min angle avg/min | angle<10% | angle<20% | violation avg/max | runtime s |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {summary['cases']} | {summary['successes']} | {fmt(summary['avg_faces_out_in'], 3)} | {fmt(summary['avg_rmse_pct_bbox'], 5)} | {sci(summary['avg_cd_bbox2'])} | {fmt(summary['avg_qavg'], 5)} | {fmt(summary['avg_qmin'], 5)}/{fmt(summary['min_qmin'], 5)} | {fmt(summary['avg_min_angle_degrees'], 4)}/{fmt(summary['min_min_angle_degrees'], 4)} | {fmt(summary['avg_angle_lt_10_pct'], 4)} | {fmt(summary['avg_angle_lt_20_pct'], 4)} | {fmt(summary['avg_violation'], 5)}/{fmt(summary['max_violation'], 5)} | {fmt(summary['avg_runtime_sec'], 2)} |",
        "",
        "## 4. 与 v5.1.5 对比",
        "",
        f"共同成功 T 模型数：`{common}`。",
        "",
        "| metric | v5.2.0 | v5.1.5 | delta |",
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
        "| criterion | v5.2.0 better / common |",
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
        "| model | min angle v5.2.0 | v5.1.5 | delta | angle<10 delta | faces delta | RMSE delta |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    successful_comparisons = [row for row in comparisons if row.get("status") == "success"]
    for row in sorted(successful_comparisons, key=lambda item: to_float(item.get("min_angle_degrees_delta")), reverse=True)[:10]:
        lines.append(f"| {row['model']} | {fmt(row['min_angle_degrees_current'], 4)} | {fmt(row['min_angle_degrees_baseline'], 4)} | {fmt(row['min_angle_degrees_delta'], 4)} | {fmt(row['angle_lt_10_pct_delta'], 4)} | {fmt(row['faces_out_in_delta'], 3)} | {fmt(row['rmse_pct_bbox_delta'], 5)} |")
    lines.extend([
        "",
        "### 5.2 最小角恶化最多",
        "",
        "| model | min angle v5.2.0 | v5.1.5 | delta | angle<10 delta | violation delta |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in sorted(successful_comparisons, key=lambda item: to_float(item.get("min_angle_degrees_delta")))[:10]:
        lines.append(f"| {row['model']} | {fmt(row['min_angle_degrees_current'], 4)} | {fmt(row['min_angle_degrees_baseline'], 4)} | {fmt(row['min_angle_degrees_delta'], 4)} | {fmt(row['angle_lt_10_pct_delta'], 4)} | {fmt(row['violation_delta'], 5)} |")
    lines.extend([
        "",
        "### 5.3 几何误差恶化最多",
        "",
        "| model | RMSE v5.2.0 | v5.1.5 | delta | min angle delta | faces delta |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in sorted(successful_comparisons, key=lambda item: to_float(item.get("rmse_pct_bbox_delta")), reverse=True)[:10]:
        lines.append(f"| {row['model']} | {fmt(row['rmse_pct_bbox_current'], 5)} | {fmt(row['rmse_pct_bbox_baseline'], 5)} | {fmt(row['rmse_pct_bbox_delta'], 5)} | {fmt(row['min_angle_degrees_delta'], 4)} | {fmt(row['faces_out_in_delta'], 3)} |")
    lines.extend([
        "",
        "## 6. 内部判断",
        "",
    ])
    min_angle_delta = to_float(comp.get("avg_min_angle_degrees_delta"))
    angle10_delta = to_float(comp.get("avg_angle_lt_10_pct_delta"))
    rmse_delta = to_float(comp.get("avg_rmse_pct_bbox_delta"))
    violation_delta = to_float(comp.get("avg_violation_delta"))
    if min_angle_delta > 0 and angle10_delta < 0:
        lines.append("- v5.2.0 对最小角质量有正向效果：平均最小角上升，同时 angle<10 的比例下降。")
    elif min_angle_delta > 0:
        lines.append("- v5.2.0 提升了平均最小角，但 angle<10 的比例没有同步下降，说明改善集中在极端最小值而不是整体坏角分布。")
    else:
        lines.append("- v5.2.0 没有提升平均最小角，当前最小角接受准则需要重新检查，尤其是 relocation 只接受改善可能导致整体正则化不足。")
    if rmse_delta > 0 or violation_delta > 0:
        lines.append("- 需要关注副作用：target metric violation 明显上升；虽然 RMSE/CD 平均略降，但这是以更多面数和更慢运行换来的。")
    else:
        lines.append("- 几何误差和 violation 没有系统性恶化，说明最小角控制目前没有破坏 v5.1.5 尺寸场主线。")
    lines.append("- 因此，本版更像是验证了“最小角可被局部接受准则推动”，但还没有形成可接受的整体 tradeoff。下一步不应继续无条件卡最小角，而要把质量接受准则改成只针对真正坏角区域、并限制额外面数和 violation。")
    lines.append("- 是否值得进入下一步，取决于 T 系列外的 model_* 高风险样本是否也能看到小角比例下降；建议下一轮跑 T125/T53/T86/T82 加 model_005/model_010 小样本。")
    if failures:
        lines.extend(["", "## 7. 失败项", "", "| model | error |", "|---|---|"])
        for row in failures:
            lines.append(f"| {row['model']} | {row['error']} |")
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = base.parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")

    base.RESULT_ROOT = RESULT_ROOT
    base.REPORT_PATH = REPORT_PATH
    base.BASELINE_CSV = BASELINE_CSV
    base.CSV_FIELDS = CSV_FIELDS
    base.SUMMARY_FIELDS = SUMMARY_FIELDS
    base.COMPARISON_FIELDS = COMPARISON_FIELDS
    base.case_dir = case_dir
    base.row_from_summary = row_from_summary
    base.failed_row = failed_row
    base.read_baseline_rows = read_baseline_rows
    base.summary_row = summary_row
    base.comparison_rows = comparison_rows
    base.comparison_summary = comparison_summary
    base.better_counts = better_counts
    base.write_report = write_report

    meshes = base.discover_t_models()
    if args.only:
        wanted = set(args.only)
        meshes = [mesh for mesh in meshes if mesh.stem in wanted]

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(base.run_case, mesh, args): mesh for mesh in meshes}
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
    baseline = read_baseline_rows()
    comparisons = comparison_rows(rows, baseline)
    comp_summary = comparison_summary(comparisons)
    base.write_csv(rows, RESULT_ROOT / "run_summary.csv", CSV_FIELDS)
    base.write_csv([summary], RESULT_ROOT / "summary.csv", SUMMARY_FIELDS)
    base.write_csv(comparisons, RESULT_ROOT / "v5.2.0_vs_v5.1.5_T_comparison.csv", COMPARISON_FIELDS)
    write_report(rows, summary, comparisons, comp_summary, args)
    print(f"wrote {RESULT_ROOT / 'run_summary.csv'}")
    print(f"wrote {RESULT_ROOT / 'summary.csv'}")
    print(f"wrote {RESULT_ROOT / 'v5.2.0_vs_v5.1.5_T_comparison.csv'}")
    print(f"wrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
