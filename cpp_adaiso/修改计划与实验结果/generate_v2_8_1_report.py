from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = ROOT / "cpp_adaiso" / "results" / "v2.8.1_percentile_target" / "T_all_iter10"
BASELINE_ROOT = ROOT / "cpp_adaiso" / "results" / "v2.8.0_saturated_target" / "T_all_iter10"
EXAMPLES_ROOT = ROOT / "cpp_adaiso" / "examples"
REPORT_PATH = ROOT / "cpp_adaiso" / "修改计划与实验结果" / "v2.8.1实验报告.md"
RUN_SUMMARY_PATH = RESULT_ROOT / "run_summary.csv"
COMPARE_SUMMARY_PATH = RESULT_ROOT / "comparison_vs_v2.8.0.csv"

TARGET_FIELD_FILES = (
    "raw_curvature_heat_shared.ply",
    "csf_curvature_heat_shared.ply",
    "raw_curvature_heat_independent.ply",
    "csf_curvature_heat_independent.ply",
    "target_h_heat.ply",
    "vertices.csv",
    "edges.csv",
    "faces.csv",
    "field_stats.json",
    "color_scales.json",
)

RUN_FIELDS = (
    "model",
    "input_vertices",
    "input_faces",
    "output_vertices",
    "output_faces",
    "qavg",
    "qmin",
    "iter10_violation",
    "cd",
    "symmetric_rmse",
    "runtime_sec",
    "csf_solver_success",
    "csf_residual_scaled",
    "target_size_mapping",
    "target_size_gamma",
    "base_length",
    "h_min",
    "h_max",
    "observed_h_min",
    "observed_h_max",
    "max_edge_target_ratio",
    "p95_edge_target_ratio",
    "p99_edge_target_ratio",
    "max_grad_h_norm",
    "p95_grad_h_norm",
    "p99_grad_h_norm",
)

COMPARE_FIELDS = (
    "model",
    "v281_output_vertices",
    "v280_output_vertices",
    "delta_vertices",
    "v281_qavg",
    "v280_qavg",
    "delta_qavg",
    "v281_qmin",
    "v280_qmin",
    "delta_qmin",
    "v281_violation",
    "v280_violation",
    "delta_violation",
    "v281_cd",
    "v280_cd",
    "delta_cd",
    "delta_cd_pct",
    "v281_rmse",
    "v280_rmse",
    "delta_rmse_pct",
    "v281_max_edge_target_ratio",
    "v280_max_edge_target_ratio",
    "delta_max_edge_target_ratio",
    "v281_max_grad_h_norm",
    "v280_max_grad_h_norm",
    "delta_max_grad_h_norm",
    "judgment",
)


def natural_model_key(path_or_name: Path | str) -> int:
    name = path_or_name.stem if isinstance(path_or_name, Path) else path_or_name
    return int(name[1:])


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def fmt(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def sci(value: float) -> str:
    return f"{value:.6E}"


def pct(value: float) -> str:
    return f"{value:.2f}%"


def signed(value: float, digits: int = 6) -> str:
    return f"{value:+.{digits}f}"


def signed_sci(value: float) -> str:
    return f"{value:+.6E}"


def signed_pct(value: float) -> str:
    return f"{value:+.2f}%"


def tail_violation(summary: dict) -> float:
    iterations = summary["iterations"]
    if not iterations:
        return 0.0
    return float(iterations[-1]["violation_rate"])


def read_case(root: Path, model: str) -> dict:
    case_dir = root / model
    summary_path = case_dir / "summary.json"
    field_dir = case_dir / "target_field"
    field_stats_path = field_dir / "field_stats.json"
    summary = load_json(summary_path)
    field_stats = load_json(field_stats_path)
    output_summary = summary["output_summary"]
    input_summary = summary["input_summary"]
    chamfer = summary["chamfer_distance"]
    row = {
        "model": model,
        "input_vertices": int(input_summary["vertices"]),
        "input_faces": int(input_summary["faces"]),
        "output_vertices": int(output_summary["vertices"]),
        "output_faces": int(output_summary["faces"]),
        "qavg": float(output_summary["qavg"]),
        "qmin": float(output_summary["qmin"]),
        "iter10_violation": tail_violation(summary),
        "cd": float(chamfer["chamfer_distance"]),
        "symmetric_rmse": float(chamfer["symmetric_rmse"]),
        "runtime_sec": float(summary["runtime_sec"]),
        "csf_solver_success": bool(summary["stats"].get("csf_solver_success", False)),
        "csf_residual_scaled": float(summary["stats"]["csf_residual_scaled"]),
        "target_size_mapping": summary["parameters"].get("target_size_mapping", ""),
        "target_size_gamma": float(summary["parameters"]["target_size_gamma"]),
        "base_length": float(field_stats["base_length"]),
        "h_min": float(field_stats["h_min"]),
        "h_max": float(field_stats["h_max"]),
        "observed_h_min": float(field_stats["observed_h_min"]),
        "observed_h_max": float(field_stats["observed_h_max"]),
        "max_edge_target_ratio": float(field_stats["max_edge_target_ratio"]),
        "p95_edge_target_ratio": float(field_stats["p95_edge_target_ratio"]),
        "p99_edge_target_ratio": float(field_stats["p99_edge_target_ratio"]),
        "max_grad_h_norm": float(field_stats["max_grad_h_norm"]),
        "p95_grad_h_norm": float(field_stats["p95_grad_h_norm"]),
        "p99_grad_h_norm": float(field_stats["p99_grad_h_norm"]),
    }
    return row


def missing_target_files(root: Path, model: str) -> list[str]:
    field_dir = root / model / "target_field"
    return [name for name in TARGET_FIELD_FILES if not (field_dir / name).exists()]


def compare_rows(current: dict, baseline: dict) -> dict:
    delta_cd = current["cd"] - baseline["cd"]
    delta_cd_pct = 100.0 * delta_cd / baseline["cd"] if baseline["cd"] else 0.0
    delta_rmse_pct = (
        100.0 * (current["symmetric_rmse"] - baseline["symmetric_rmse"]) / baseline["symmetric_rmse"]
        if baseline["symmetric_rmse"]
        else 0.0
    )
    delta_qmin = current["qmin"] - baseline["qmin"]
    delta_violation = current["iter10_violation"] - baseline["iter10_violation"]
    delta_max_edge_ratio = current["max_edge_target_ratio"] - baseline["max_edge_target_ratio"]

    if delta_cd_pct < -5.0 and delta_qmin > -0.05 and delta_violation <= 0.03:
        judgment = "v2.8.1 better"
    elif delta_cd_pct > 5.0 or delta_qmin < -0.10 or delta_violation > 0.05 or delta_max_edge_ratio > 0.50:
        judgment = "v2.8.0 steadier"
    else:
        judgment = "tradeoff"

    return {
        "model": current["model"],
        "v281_output_vertices": current["output_vertices"],
        "v280_output_vertices": baseline["output_vertices"],
        "delta_vertices": current["output_vertices"] - baseline["output_vertices"],
        "v281_qavg": current["qavg"],
        "v280_qavg": baseline["qavg"],
        "delta_qavg": current["qavg"] - baseline["qavg"],
        "v281_qmin": current["qmin"],
        "v280_qmin": baseline["qmin"],
        "delta_qmin": delta_qmin,
        "v281_violation": current["iter10_violation"],
        "v280_violation": baseline["iter10_violation"],
        "delta_violation": delta_violation,
        "v281_cd": current["cd"],
        "v280_cd": baseline["cd"],
        "delta_cd": delta_cd,
        "delta_cd_pct": delta_cd_pct,
        "v281_rmse": current["symmetric_rmse"],
        "v280_rmse": baseline["symmetric_rmse"],
        "delta_rmse_pct": delta_rmse_pct,
        "v281_max_edge_target_ratio": current["max_edge_target_ratio"],
        "v280_max_edge_target_ratio": baseline["max_edge_target_ratio"],
        "delta_max_edge_target_ratio": delta_max_edge_ratio,
        "v281_max_grad_h_norm": current["max_grad_h_norm"],
        "v280_max_grad_h_norm": baseline["max_grad_h_norm"],
        "delta_max_grad_h_norm": current["max_grad_h_norm"] - baseline["max_grad_h_norm"],
        "judgment": judgment,
    }


def write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def average(rows: list[dict], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def count_where(rows: list[dict], predicate) -> int:
    return sum(1 for row in rows if predicate(row))


def run_result_table(rows: list[dict]) -> str:
    lines = [
        "| Model | Out V/F | Qavg | Qmin | viol | CD | RMSE | max edge ratio | max grad |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["model"],
                    f"{row['output_vertices']}/{row['output_faces']}",
                    fmt(row["qavg"]),
                    fmt(row["qmin"]),
                    fmt(row["iter10_violation"]),
                    sci(row["cd"]),
                    sci(row["symmetric_rmse"]),
                    fmt(row["max_edge_target_ratio"]),
                    fmt(row["max_grad_h_norm"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def comparison_table(rows: list[dict]) -> str:
    lines = [
        "| Model | ΔCD% | ΔRMSE% | ΔQmin | Δviol | ΔV | Δmax edge ratio | 判断 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["model"],
                    signed_pct(row["delta_cd_pct"]),
                    signed_pct(row["delta_rmse_pct"]),
                    signed(row["delta_qmin"]),
                    signed(row["delta_violation"]),
                    f"{row['delta_vertices']:+d}",
                    signed(row["delta_max_edge_target_ratio"]),
                    row["judgment"],
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def top_models(rows: list[dict], key: str, count: int, reverse: bool = True) -> str:
    selected = sorted(rows, key=lambda row: row[key], reverse=reverse)[:count]
    return ", ".join(f"`{row['model']}` ({row[key]:.6g})" for row in selected)


def write_report(current_rows: list[dict], baseline_rows: list[dict], comparisons: list[dict]) -> None:
    model_count = len(current_rows)
    summary_count = sum(1 for row in current_rows if (RESULT_ROOT / row["model"] / "summary.json").exists())
    baseline_summary_count = sum(1 for row in baseline_rows if (BASELINE_ROOT / row["model"] / "summary.json").exists())
    target_complete_count = sum(1 for row in current_rows if not missing_target_files(RESULT_ROOT, row["model"]))
    raw_count = sum(1 for row in current_rows if (RESULT_ROOT / row["model"] / "target_field" / "raw_curvature_heat_shared.ply").exists())
    csf_count = sum(1 for row in current_rows if (RESULT_ROOT / row["model"] / "target_field" / "csf_curvature_heat_shared.ply").exists())
    mapping_count = count_where(current_rows, lambda row: row["target_size_mapping"] == "percentile_normalized_power")
    csf_success_count = count_where(current_rows, lambda row: row["csf_solver_success"])
    finite_count = count_where(
        current_rows,
        lambda row: all(
            finite(row[key])
            for key in (
                "qavg",
                "qmin",
                "iter10_violation",
                "cd",
                "symmetric_rmse",
                "runtime_sec",
                "max_edge_target_ratio",
                "max_grad_h_norm",
            )
        ),
    )

    cd_better = count_where(comparisons, lambda row: row["delta_cd_pct"] < -5.0)
    cd_worse = count_where(comparisons, lambda row: row["delta_cd_pct"] > 5.0)
    cd_similar = model_count - cd_better - cd_worse
    qmin_down = count_where(comparisons, lambda row: row["delta_qmin"] < -0.05)
    qmin_up = count_where(comparisons, lambda row: row["delta_qmin"] > 0.05)
    viol_up = count_where(comparisons, lambda row: row["delta_violation"] > 0.03)
    max_ratio_up = count_where(comparisons, lambda row: row["delta_max_edge_target_ratio"] > 0.50)
    judged_v281 = count_where(comparisons, lambda row: row["judgment"] == "v2.8.1 better")
    judged_v280 = count_where(comparisons, lambda row: row["judgment"] == "v2.8.0 steadier")
    judged_tradeoff = model_count - judged_v281 - judged_v280

    avg_cd_delta_pct = average(comparisons, "delta_cd_pct")
    avg_rmse_delta_pct = average(comparisons, "delta_rmse_pct")
    avg_qmin_delta = average(comparisons, "delta_qmin")
    avg_violation_delta = average(comparisons, "delta_violation")

    conclusion = (
        "整体看，v2.8.1 的分位数归一化不是 v2.8.0 的稳定改进："
        f"CD 明显改善 {cd_better} 个、明显变差 {cd_worse} 个、基本持平 {cd_similar} 个。"
        f"同时 Qmin 明显下降 {qmin_down} 个，violation 明显上升 {viol_up} 个。"
        "因此默认结论是 v2.8.0 更稳；v2.8.1 只适合作为个别模型的几何误差优化候选。"
    )
    if judged_v281 > judged_v280:
        conclusion = (
            "整体看，v2.8.1 在更多模型上取得可接受收益，"
            "但仍需要结合 Qmin、violation 和 edge-target ratio 排除局部质量退化。"
        )

    lines = [
        "# v2.8.1 实验报告：分位数归一化尺寸场 T 系列 iter10",
        "",
        "## 实验设置",
        "",
        "v2.8.1 将 v2.8.0 的 min/max 曲率归一化改为 `Q5/Q95` 分位数归一化，然后继续使用幂函数映射：",
        "",
        "```text",
        "t_i = clamp((k_i - Q5) / (Q95 - Q5), 0, 1)",
        "h_i = h_min + (h_max - h_min) * (1 - t_i)^gamma",
        "```",
        "",
        "| 参数 | 值 |",
        "|---|---:|",
        "| curvature_mode | robust-mean |",
        "| csf_alpha | 2.0 |",
        "| target_size_mapping | percentile_normalized_power |",
        "| target_min_factor | 0.6 |",
        "| target_max_factor | 1.8 |",
        "| target_size_gamma | 2.0 |",
        "| iters | 10 |",
        "| preprocess | 0 |",
        "| project | 1 |",
        "| cd-samples | 100000 |",
        "",
        f"- v2.8.1 输出目录：`{rel(RESULT_ROOT)}/`",
        f"- v2.8.1 汇总 CSV：`{rel(RUN_SUMMARY_PATH)}`",
        f"- v2.8.0 基线目录：`{rel(BASELINE_ROOT)}/`",
        f"- 对比 CSV：`{rel(COMPARE_SUMMARY_PATH)}`",
        "",
        "每个模型命令形式：",
        "",
        "```powershell",
        "cpp_adaiso/build/adaiso_cli.exe cpp_adaiso/examples/<T>.off `",
        "  -o cpp_adaiso/results/v2.8.1_percentile_target/T_all_iter10/<T>/<T>.ply `",
        "  --iters 10 --preprocess 0 --project 1 `",
        "  --gt cpp_adaiso/examples/<T>.off --cd-samples 100000 `",
        "  --export-target-field cpp_adaiso/results/v2.8.1_percentile_target/T_all_iter10/<T>/target_field `",
        "  --summary-json cpp_adaiso/results/v2.8.1_percentile_target/T_all_iter10/<T>/summary.json `",
        "  --quiet",
        "```",
        "",
        "## 完整性检查",
        "",
        "| 检查项 | 结果 |",
        "|---|---:|",
        f"| example T 模型数量 | {model_count} |",
        f"| v2.8.1 summary.json 数量 | {summary_count} / {model_count} |",
        f"| v2.8.0 基线 summary.json 数量 | {baseline_summary_count} / {model_count} |",
        f"| target_size_mapping 匹配数量 | {mapping_count} / {model_count} |",
        f"| target_field 诊断完整数量 | {target_complete_count} / {model_count} |",
        f"| raw 曲率热力图数量 | {raw_count} / {model_count} |",
        f"| CSF 曲率热力图数量 | {csf_count} / {model_count} |",
        f"| csf_solver_success=true | {csf_success_count} / {model_count} |",
        f"| 指标 finite | {finite_count} / {model_count} |",
        "",
        "每个 `<T>/target_field/` 均应包含：",
        "",
        "```text",
        "\n".join(TARGET_FIELD_FILES),
        "```",
        "",
        "其中 `raw_curvature_*` 是曲率扩散前可视化，`csf_curvature_*` 是 CSF 扩散后可视化；shared 色标用于同模型内 raw/CSF 直接对比。",
        "",
        "## v2.8.1 全 T 系列结果",
        "",
        run_result_table(current_rows),
        "",
        "整体统计：",
        "",
        "| 指标 | v2.8.1 |",
        "|---|---:|",
        f"| 平均 Qavg | {average(current_rows, 'qavg'):.10f} |",
        f"| 平均 Qmin | {average(current_rows, 'qmin'):.10f} |",
        f"| 最低 Qmin | {min(row['qmin'] for row in current_rows):.10f} |",
        f"| 平均 CD | {average(current_rows, 'cd'):.10E} |",
        f"| 最大 CD | {max(row['cd'] for row in current_rows):.10E} |",
        f"| 平均 iter10 violation | {average(current_rows, 'iter10_violation'):.10f} |",
        f"| 平均 runtime | {average(current_rows, 'runtime_sec'):.3f} s |",
        "",
        f"- 最低 Qmin：{top_models(current_rows, 'qmin', 5, reverse=False)}",
        f"- 最大 CD：{top_models(current_rows, 'cd', 5, reverse=True)}",
        f"- 最大 edge-target ratio：{top_models(current_rows, 'max_edge_target_ratio', 5, reverse=True)}",
        "",
        "## 与 v2.8.0 对比",
        "",
        "判定规则：`ΔCD% < -5%` 视为几何误差明显改善，`ΔCD% > 5%` 视为变差；同时检查 Qmin、violation 和 edge-target ratio 的退化。",
        "",
        comparison_table(comparisons),
        "",
        "对比统计：",
        "",
        "| 指标 | 结果 |",
        "|---|---:|",
        f"| CD 明显改善 | {cd_better} / {model_count} |",
        f"| CD 明显变差 | {cd_worse} / {model_count} |",
        f"| CD 基本持平 | {cd_similar} / {model_count} |",
        f"| Qmin 明显下降 | {qmin_down} / {model_count} |",
        f"| Qmin 明显上升 | {qmin_up} / {model_count} |",
        f"| violation 明显上升 | {viol_up} / {model_count} |",
        f"| max edge-target ratio 明显上升 | {max_ratio_up} / {model_count} |",
        f"| 判定 v2.8.1 better | {judged_v281} / {model_count} |",
        f"| 判定 v2.8.0 steadier | {judged_v280} / {model_count} |",
        f"| 判定 tradeoff | {judged_tradeoff} / {model_count} |",
        f"| 平均 ΔCD% | {signed_pct(avg_cd_delta_pct)} |",
        f"| 平均 ΔRMSE% | {signed_pct(avg_rmse_delta_pct)} |",
        f"| 平均 ΔQmin | {signed(avg_qmin_delta)} |",
        f"| 平均 Δviolation | {signed(avg_violation_delta)} |",
        "",
        "重点模型：",
        "",
    ]

    for model in ("T21", "T30"):
        row = next(item for item in comparisons if item["model"] == model)
        lines.extend(
            [
                f"### {model}",
                "",
                "| 指标 | v2.8.0 | v2.8.1 | 变化 |",
                "|---|---:|---:|---:|",
                f"| output vertices | {row['v280_output_vertices']} | {row['v281_output_vertices']} | {row['delta_vertices']:+d} |",
                f"| Qavg | {row['v280_qavg']:.10f} | {row['v281_qavg']:.10f} | {row['delta_qavg']:+.10f} |",
                f"| Qmin | {row['v280_qmin']:.10f} | {row['v281_qmin']:.10f} | {row['delta_qmin']:+.10f} |",
                f"| iter10 violation | {row['v280_violation']:.10f} | {row['v281_violation']:.10f} | {row['delta_violation']:+.10f} |",
                f"| CD | {sci(row['v280_cd'])} | {sci(row['v281_cd'])} | {signed_sci(row['delta_cd'])} ({signed_pct(row['delta_cd_pct'])}) |",
                f"| RMSE | {sci(row['v280_rmse'])} | {sci(row['v281_rmse'])} | {signed_pct(row['delta_rmse_pct'])} |",
                f"| max edge-target ratio | {row['v280_max_edge_target_ratio']:.10f} | {row['v281_max_edge_target_ratio']:.10f} | {row['delta_max_edge_target_ratio']:+.10f} |",
                f"| max grad_h_norm | {row['v280_max_grad_h_norm']:.10f} | {row['v281_max_grad_h_norm']:.10f} | {row['delta_max_grad_h_norm']:+.10f} |",
                "",
                f"判断：`{row['judgment']}`。",
                "",
            ]
        )

    lines.extend(
        [
            "## 总体结论",
            "",
            conclusion,
            "",
            "具体来说，分位数归一化会把曲率两端的 5% 样本压到端点，能让高曲率区域更容易进入小尺寸平台；这会在部分模型上降低 CD/RMSE，但也更容易增加局部尺寸场梯度和边长约束违例。后续若继续推进 v2.8.1，建议先围绕 Qmin 低、violation 高、max edge-target ratio 高的模型做局部可视化检查。",
            "",
        ]
    )

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    models = sorted((path.stem for path in EXAMPLES_ROOT.glob("T*.off")), key=natural_model_key)
    current_rows = [read_case(RESULT_ROOT, model) for model in models]
    baseline_rows = [read_case(BASELINE_ROOT, model) for model in models]
    baseline_by_model = {row["model"]: row for row in baseline_rows}
    comparisons = [compare_rows(row, baseline_by_model[row["model"]]) for row in current_rows]

    for row in current_rows:
        missing = missing_target_files(RESULT_ROOT, row["model"])
        if missing:
            raise RuntimeError(f"missing target field files for {row['model']}: {missing}")
        if row["target_size_mapping"] != "percentile_normalized_power":
            raise RuntimeError(f"unexpected target mapping for {row['model']}: {row['target_size_mapping']}")
        if not row["csf_solver_success"]:
            raise RuntimeError(f"CSF solve failed for {row['model']}")

    write_csv(RUN_SUMMARY_PATH, RUN_FIELDS, current_rows)
    write_csv(COMPARE_SUMMARY_PATH, COMPARE_FIELDS, comparisons)
    write_report(current_rows, baseline_rows, comparisons)
    print(f"wrote {rel(RUN_SUMMARY_PATH)}")
    print(f"wrote {rel(COMPARE_SUMMARY_PATH)}")
    print(f"wrote {rel(REPORT_PATH)}")


if __name__ == "__main__":
    main()
