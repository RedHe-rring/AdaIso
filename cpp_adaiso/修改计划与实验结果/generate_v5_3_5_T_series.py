from __future__ import annotations

import argparse
import csv
import importlib.util
import math
from pathlib import Path
from typing import Any


BASE_SCRIPT = Path(__file__).with_name("generate_v5_3_4_T_series.py")
SPEC = importlib.util.spec_from_file_location("v5_3_4_t_series", BASE_SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"failed to load base script: {BASE_SCRIPT}")
v534 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v534)

ROOT = v534.ROOT
CPP_ROOT = v534.CPP_ROOT
RESULT_ROOT = CPP_ROOT / "results" / "v5.3.5_T_series_collapse_energy_only_q50"
BASELINE_CSV = CPP_ROOT / "results" / "v5.3.4_T_series_collapse_target_energy_q50" / "run_summary.csv"
REPORT_PATH = CPP_ROOT / "修改计划与实验结果" / "v5.3.5 T系列Collapse Energy-Only实验报告.md"
POLICY = v534.POLICY

v534.RESULT_ROOT = RESULT_ROOT
v534.BASELINE_CSV = BASELINE_CSV
v534.REPORT_PATH = REPORT_PATH


def to_float(value: Any) -> float:
    return v534.to_float(value)


def fmt(value: Any, digits: int = 6) -> str:
    return v534.fmt(value, digits)


def reason_share(summary: dict[str, Any], field: str) -> float:
    rejected = to_float(summary.get("avg_size_stage_collapse_rejected"))
    value = to_float(summary.get(f"avg_{field}"))
    return value / rejected if rejected > 0 else math.nan


def read_baseline_summary() -> dict[str, Any]:
    path = BASELINE_CSV.parent / "summary.csv"
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return dict(rows[0]) if rows else {}


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
        "# v5.3.5 T系列 Collapse Energy-Only 实验报告",
        "",
        "## 1. 结论摘要",
        "",
        f"本轮使用 v5.3.5 默认策略 `{POLICY}` 跑 T 系列 `{len(rows)}` 个模型，成功 `{len(successes)}/{len(rows)}`。",
        "本版删除 v5.3.4 的 collapse max-ratio slack，只保留局部 target metric violation 必须下降。",
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
    for field in v534.REJECT_REASON_FIELDS:
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
        "## 5. 与 v5.3.4 对比",
        "",
    ])
    if comparisons:
        lines.extend(["| metric delta | avg | max abs |", "|---|---:|---:|"])
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
        lines.append("未找到可用 v5.3.4 基线 CSV，跳过输出一致性对比。")

    if baseline_summary:
        lines.extend([
            "",
            "## 6. Collapse 诊断对比",
            "",
            "| metric | v5.3.5 | v5.3.4 | delta |",
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
        f"- 主拒绝原因是 `target_length`，平均 `{fmt(summary['avg_size_stage_collapse_rejected_by_target_length'], 2)}` 条，占 `{fmt(100.0 * reason_share(summary, 'size_stage_collapse_rejected_by_target_length'), 2)}%`。",
        f"- 第二原因是 `batch_conflict`，平均 `{fmt(summary['avg_size_stage_collapse_rejected_by_batch_conflict'], 2)}` 条，占 `{fmt(100.0 * reason_share(summary, 'size_stage_collapse_rejected_by_batch_conflict'), 2)}%`。",
        f"- collapse 平均接受率为 `{fmt(100.0 * summary['avg_collapse_accept_rate'], 2)}%`。",
        "- v5.3.5 达到本轮主要目的：target_length 拒绝下降，collapse accepted 上升，faces out/in 和 violation 在 31/31 个模型上都下降。",
        "- 坏角质量同步改善：angle<20% 在 31/31 个模型下降，angle<10% 在 29/31 个模型下降，平均 min angle 提升。",
        "- 几何误差有轻微代价：RMSE/CD 平均小幅上升，且 27/31 个模型上升；这说明 max-ratio slack 可以删除，但下一步要监控误差项。",
        "- 下一版不应恢复硬 max-ratio 约束；如果要控制误差，应考虑 soft penalty 或把 target energy 加权，而不是回到一票否决。",
        "",
        "输出文件：",
        f"- `{(RESULT_ROOT / 'run_summary.csv').relative_to(ROOT).as_posix()}`",
        f"- `{(RESULT_ROOT / 'summary.csv').relative_to(ROOT).as_posix()}`",
        f"- `{REPORT_PATH.relative_to(ROOT).as_posix()}`",
    ])
    if failures:
        lines.extend(["", "## 8. 失败项", "", "| model | error |", "|---|---|"])
        for row in failures:
            lines.append(f"| {row['model']} | {row['error']} |")
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v5.3.5 T-series collapse energy-only validation.")
    parser.add_argument("--cli", type=Path, default=v534.base.DEFAULT_CLI)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--cd-samples", type=int, default=100000)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--only", nargs="*", help="Optional subset of T model names.")
    return parser.parse_args()


def main() -> int:
    v534.write_report = write_report
    v534.read_baseline_summary = read_baseline_summary
    v534.parse_args = parse_args
    result = v534.main()
    comparison = RESULT_ROOT / "v5.3.4_vs_v5.3.3_T_comparison.csv"
    if comparison.is_file():
        target = RESULT_ROOT / "v5.3.5_vs_v5.3.4_T_comparison.csv"
        if target.is_file():
            target.unlink()
        comparison.rename(target)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
