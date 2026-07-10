from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = ROOT / "cpp_adaiso" / "examples"
RESULT_ROOT = ROOT / "cpp_adaiso" / "results" / "v2.10.0_target_faces" / "T_all_iter10"
BASELINE_ROOT = ROOT / "cpp_adaiso" / "results" / "v2.8.1_percentile_target" / "T_all_iter10"
REPORT_PATH = ROOT / "cpp_adaiso" / "修改计划与实验结果" / "v2.10.0 T系列 target-face iter10实验报告.md"
RUN_SUMMARY_PATH = RESULT_ROOT / "run_summary.csv"
COMPARE_SUMMARY_PATH = RESULT_ROOT / "comparison_vs_v2.8.1.csv"
DEFAULT_CLI = ROOT / "cpp_adaiso" / "build" / "adaiso_cli.exe"

EXPECTED_MAPPING = "target_faces_inverse_sqrt_diffused_curvature"

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
    "target_faces",
    "target_faces_matches_input",
    "qavg",
    "qmin",
    "iter10_violation",
    "cd",
    "symmetric_rmse",
    "runtime_sec",
    "csf_solver_success",
    "csf_residual_scaled",
    "target_size_mapping",
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
    "v210_output_faces",
    "v281_output_faces",
    "delta_faces",
    "v210_target_faces",
    "v210_qavg",
    "v281_qavg",
    "delta_qavg",
    "v210_qmin",
    "v281_qmin",
    "delta_qmin",
    "v210_violation",
    "v281_violation",
    "delta_violation",
    "v210_cd",
    "v281_cd",
    "delta_cd",
    "delta_cd_pct",
    "v210_rmse",
    "v281_rmse",
    "delta_rmse_pct",
    "v210_max_edge_target_ratio",
    "v281_max_edge_target_ratio",
    "delta_max_edge_target_ratio",
    "v210_max_grad_h_norm",
    "v281_max_grad_h_norm",
    "delta_max_grad_h_norm",
    "judgment",
)


def natural_model_key(path_or_name: Path | str) -> int:
    name = path_or_name.stem if isinstance(path_or_name, Path) else path_or_name
    match = re.fullmatch(r"T(\d+)", name)
    if not match:
        raise ValueError(f"unexpected T-series model name: {name}")
    return int(match.group(1))


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def rel_arg(path: Path) -> str:
    return str(path.relative_to(ROOT))


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


def discover_models() -> list[Path]:
    models = []
    for path in EXAMPLES_ROOT.glob("T*.off"):
        try:
            natural_model_key(path)
        except ValueError:
            continue
        models.append(path)
    return sorted(models, key=natural_model_key)


def case_dir(root: Path, model: str) -> Path:
    return root / model


def missing_target_files(root: Path, model: str) -> list[str]:
    field_dir = case_dir(root, model) / "target_field"
    return [name for name in TARGET_FIELD_FILES if not (field_dir / name).exists()]


def tail_violation(summary: dict) -> float:
    iterations = summary.get("iterations", [])
    if not iterations:
        return 0.0
    return float(iterations[-1].get("violation_rate", 0.0))


def read_case(root: Path, model: str) -> dict:
    current_dir = case_dir(root, model)
    summary = load_json(current_dir / "summary.json")
    field_stats = load_json(current_dir / "target_field" / "field_stats.json")

    input_summary = summary["input_summary"]
    output_summary = summary["output_summary"]
    chamfer = summary["chamfer_distance"]
    parameters = summary.get("parameters", {})
    stats = summary.get("stats", {})
    target_faces = int(parameters.get("target_faces", 0) or 0)
    input_faces = int(input_summary["faces"])

    return {
        "model": model,
        "input_vertices": int(input_summary["vertices"]),
        "input_faces": input_faces,
        "output_vertices": int(output_summary["vertices"]),
        "output_faces": int(output_summary["faces"]),
        "target_faces": target_faces,
        "target_faces_matches_input": target_faces == input_faces,
        "qavg": float(output_summary["qavg"]),
        "qmin": float(output_summary["qmin"]),
        "iter10_violation": tail_violation(summary),
        "cd": float(chamfer["chamfer_distance"]),
        "symmetric_rmse": float(chamfer["symmetric_rmse"]),
        "runtime_sec": float(summary["runtime_sec"]),
        "csf_solver_success": bool(stats.get("csf_solver_success", False)),
        "csf_residual_scaled": float(stats.get("csf_residual_scaled", 0.0)),
        "target_size_mapping": parameters.get("target_size_mapping", ""),
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


def case_is_complete(root: Path, model: str, *, expected_mapping: str | None) -> bool:
    current_dir = case_dir(root, model)
    if not (current_dir / "output.ply").exists() and root == RESULT_ROOT:
        return False
    if not (current_dir / "summary.json").exists():
        return False
    if missing_target_files(root, model):
        return False
    try:
        row = read_case(root, model)
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        return False
    if expected_mapping is not None and row["target_size_mapping"] != expected_mapping:
        return False
    return all(
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
    )


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


def top_models(rows: list[dict], key: str, count: int, reverse: bool = True) -> str:
    selected = sorted(rows, key=lambda row: row[key], reverse=reverse)[:count]
    return ", ".join(f"`{row['model']}` ({row[key]:.6g})" for row in selected)


def run_result_table(rows: list[dict]) -> str:
    lines = [
        "| Model | In F | Out V/F | target_faces | Qavg | Qmin | viol | CD | RMSE | max edge ratio | p95 ratio | max grad | time(s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["model"],
                    str(row["input_faces"]),
                    f"{row['output_vertices']}/{row['output_faces']}",
                    str(row["target_faces"]),
                    fmt(row["qavg"]),
                    fmt(row["qmin"]),
                    fmt(row["iter10_violation"]),
                    sci(row["cd"]),
                    sci(row["symmetric_rmse"]),
                    fmt(row["max_edge_target_ratio"]),
                    fmt(row["p95_edge_target_ratio"]),
                    fmt(row["max_grad_h_norm"]),
                    fmt(row["runtime_sec"], 2),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def visual_field_table(models: list[str], root: Path) -> str:
    lines = [
        "| Model | target_field path | files |",
        "|---|---|---:|",
    ]
    for model in models:
        missing = missing_target_files(root, model)
        complete = len(TARGET_FIELD_FILES) - len(missing)
        lines.append(
            f"| {model} | `{rel(case_dir(root, model) / 'target_field')}/` | {complete}/{len(TARGET_FIELD_FILES)} |"
        )
    return "\n".join(lines)


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
    delta_edge_ratio = current["max_edge_target_ratio"] - baseline["max_edge_target_ratio"]

    if delta_cd_pct < -5.0 and delta_qmin > -0.05 and delta_violation <= 0.03:
        judgment = "v2.10 geometry better"
    elif delta_cd_pct > 5.0 or delta_qmin < -0.10 or delta_violation > 0.05 or delta_edge_ratio > 0.50:
        judgment = "v2.8.1 steadier"
    else:
        judgment = "tradeoff"

    return {
        "model": current["model"],
        "v210_output_faces": current["output_faces"],
        "v281_output_faces": baseline["output_faces"],
        "delta_faces": current["output_faces"] - baseline["output_faces"],
        "v210_target_faces": current["target_faces"],
        "v210_qavg": current["qavg"],
        "v281_qavg": baseline["qavg"],
        "delta_qavg": current["qavg"] - baseline["qavg"],
        "v210_qmin": current["qmin"],
        "v281_qmin": baseline["qmin"],
        "delta_qmin": delta_qmin,
        "v210_violation": current["iter10_violation"],
        "v281_violation": baseline["iter10_violation"],
        "delta_violation": delta_violation,
        "v210_cd": current["cd"],
        "v281_cd": baseline["cd"],
        "delta_cd": delta_cd,
        "delta_cd_pct": delta_cd_pct,
        "v210_rmse": current["symmetric_rmse"],
        "v281_rmse": baseline["symmetric_rmse"],
        "delta_rmse_pct": delta_rmse_pct,
        "v210_max_edge_target_ratio": current["max_edge_target_ratio"],
        "v281_max_edge_target_ratio": baseline["max_edge_target_ratio"],
        "delta_max_edge_target_ratio": delta_edge_ratio,
        "v210_max_grad_h_norm": current["max_grad_h_norm"],
        "v281_max_grad_h_norm": baseline["max_grad_h_norm"],
        "delta_max_grad_h_norm": current["max_grad_h_norm"] - baseline["max_grad_h_norm"],
        "judgment": judgment,
    }


def comparison_table(rows: list[dict]) -> str:
    lines = [
        "| Model | dCD% | dRMSE% | dQmin | dviol | dF | dmax edge ratio | judgment |",
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
                    f"{row['delta_faces']:+d}",
                    signed(row["delta_max_edge_target_ratio"]),
                    row["judgment"],
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def collect_rows(models: list[str], root: Path, *, expected_mapping: str | None) -> tuple[list[dict], dict[str, str]]:
    rows: list[dict] = []
    errors: dict[str, str] = {}
    for model in models:
        if not case_is_complete(root, model, expected_mapping=expected_mapping):
            missing = missing_target_files(root, model)
            reason = "incomplete target_field" if missing else "summary/output/metrics incomplete"
            errors[model] = reason
            continue
        try:
            rows.append(read_case(root, model))
        except Exception as exc:  # noqa: BLE001 - report should keep collecting other models.
            errors[model] = str(exc)
    rows.sort(key=lambda row: natural_model_key(row["model"]))
    return rows, errors


def run_case(model_path: Path, args: argparse.Namespace) -> None:
    model = model_path.stem
    current_dir = case_dir(RESULT_ROOT, model)
    current_dir.mkdir(parents=True, exist_ok=True)
    field_dir = current_dir / "target_field"
    output_path = current_dir / "output.ply"
    summary_path = current_dir / "summary.json"

    command = [
        str(args.cli),
        rel_arg(model_path),
        "-o",
        rel_arg(output_path),
        "--iters",
        str(args.iters),
        "--preprocess",
        "0",
        "--curvature-mode",
        args.curvature_mode,
        "--csf-alpha",
        str(args.csf_alpha),
        "--gt",
        rel_arg(model_path),
        "--cd-samples",
        str(args.cd_samples),
        "--export-target-field",
        rel_arg(field_dir),
        "--summary-json",
        rel_arg(summary_path),
        "--quiet",
    ]

    started = time.perf_counter()
    proc = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    elapsed = time.perf_counter() - started
    log = [
        "$ " + " ".join(command),
        "",
        f"returncode: {proc.returncode}",
        f"wall_time_sec: {elapsed:.3f}",
        "",
        "--- stdout ---",
        proc.stdout,
        "",
        "--- stderr ---",
        proc.stderr,
    ]
    (current_dir / "run.log").write_text("\n".join(log), encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"{model} failed with exit code {proc.returncode}; see {rel(current_dir / 'run.log')}")


def write_report(
    models: list[str],
    current_rows: list[dict],
    current_errors: dict[str, str],
    baseline_rows: list[dict],
    baseline_errors: dict[str, str],
    comparisons: list[dict],
    args: argparse.Namespace,
) -> None:
    model_count = len(models)
    completed_count = len(current_rows)
    target_complete_count = sum(1 for model in models if not missing_target_files(RESULT_ROOT, model))
    mapping_count = count_where(current_rows, lambda row: row["target_size_mapping"] == EXPECTED_MAPPING)
    target_faces_default_count = count_where(current_rows, lambda row: row["target_faces_matches_input"])
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

    lines = [
        "# v2.10.0 T系列 target-face iter10实验报告",
        "",
        "## 实验设置",
        "",
        "本实验对 `cpp_adaiso/examples` 下的 T 系列 `.off` 模型运行 v2.10.0 当前实现。尺寸场使用 `target_faces_inverse_sqrt_diffused_curvature`：`kappa` 来自 CSF 扩散后的曲率信号，未显式传入 `--target-faces` 时默认目标面数为清理后输入 mesh 的面数。",
        "",
        "| 参数 | 值 |",
        "|---|---:|",
        f"| 模型数量 | {model_count} |",
        f"| iters | {args.iters} |",
        "| preprocess | 0 |",
        f"| curvature_mode | {args.curvature_mode} |",
        f"| csf_alpha | {args.csf_alpha} |",
        f"| cd-samples | {args.cd_samples} |",
        "| target_faces | 默认：清理后输入面数 |",
        "",
        f"- 输出目录：`{rel(RESULT_ROOT)}/`",
        f"- 汇总 CSV：`{rel(RUN_SUMMARY_PATH)}`",
        f"- v2.8.1 基线目录：`{rel(BASELINE_ROOT)}/`",
        f"- 对比 CSV：`{rel(COMPARE_SUMMARY_PATH)}`",
        "",
        "命令模板：",
        "",
        "```powershell",
        "cpp_adaiso/build/adaiso_cli.exe cpp_adaiso/examples/<T>.off `",
        "  -o cpp_adaiso/results/v2.10.0_target_faces/T_all_iter10/<T>/output.ply `",
        f"  --iters {args.iters} --preprocess 0 `",
        f"  --curvature-mode {args.curvature_mode} --csf-alpha {args.csf_alpha} `",
        f"  --gt cpp_adaiso/examples/<T>.off --cd-samples {args.cd_samples} `",
        "  --export-target-field cpp_adaiso/results/v2.10.0_target_faces/T_all_iter10/<T>/target_field `",
        "  --summary-json cpp_adaiso/results/v2.10.0_target_faces/T_all_iter10/<T>/summary.json `",
        "  --quiet",
        "```",
        "",
        "## 完整性检查",
        "",
        "| 检查项 | 结果 |",
        "|---|---:|",
        f"| T 系列 `.off` 模型 | {model_count} |",
        f"| v2.10.0 完成模型 | {completed_count} / {model_count} |",
        f"| target_field 文件完整 | {target_complete_count} / {model_count} |",
        f"| target_size_mapping 匹配 | {mapping_count} / {model_count} |",
        f"| target_faces 等于输入面数 | {target_faces_default_count} / {model_count} |",
        f"| csf_solver_success=true | {csf_success_count} / {model_count} |",
        f"| 指标 finite | {finite_count} / {model_count} |",
        f"| v2.8.1 基线完整 | {len(baseline_rows)} / {model_count} |",
        "",
        "`target_field/` 应包含：",
        "",
        "```text",
        "\n".join(TARGET_FIELD_FILES),
        "```",
        "",
    ]

    if current_errors:
        lines.extend(
            [
                "未完成模型：",
                "",
                "| Model | 原因 |",
                "|---|---|",
            ]
        )
        for model in models:
            if model in current_errors:
                lines.append(f"| {model} | {current_errors[model]} |")
        lines.append("")

    lines.extend(
        [
            "## v2.10.0 结果",
            "",
            run_result_table(current_rows) if current_rows else "_没有可汇总的完整结果。_",
            "",
        ]
    )

    if current_rows:
        lines.extend(
            [
                "汇总统计：",
                "",
                "| 指标 | v2.10.0 |",
                "|---|---:|",
                f"| 平均 Qavg | {average(current_rows, 'qavg'):.10f} |",
                f"| 平均 Qmin | {average(current_rows, 'qmin'):.10f} |",
                f"| 最低 Qmin | {min(row['qmin'] for row in current_rows):.10f} |",
                f"| 平均 CD | {average(current_rows, 'cd'):.10E} |",
                f"| 最大 CD | {max(row['cd'] for row in current_rows):.10E} |",
                f"| 平均 symmetric RMSE | {average(current_rows, 'symmetric_rmse'):.10E} |",
                f"| 平均 iter10 violation | {average(current_rows, 'iter10_violation'):.10f} |",
                f"| 最高 iter10 violation | {max(row['iter10_violation'] for row in current_rows):.10f} |",
                f"| 平均 runtime | {average(current_rows, 'runtime_sec'):.3f} s |",
                "",
                "Worst-case：",
                "",
                f"- 最低 Qmin：{top_models(current_rows, 'qmin', 5, reverse=False)}",
                f"- 最大 CD：{top_models(current_rows, 'cd', 5, reverse=True)}",
                f"- 最高 violation：{top_models(current_rows, 'iter10_violation', 5, reverse=True)}",
                f"- 最大 edge-target ratio：{top_models(current_rows, 'max_edge_target_ratio', 5, reverse=True)}",
                f"- 最大 target_h 梯度：{top_models(current_rows, 'max_grad_h_norm', 5, reverse=True)}",
                "",
            ]
        )

    lines.extend(
        [
            "## 可视化场输出",
            "",
            "`vertices.csv` 保持字段 `vertex_id,x,y,z,raw_curvature,csf_curvature,target_h`。其中 `target_h` 由 `csf_curvature` 生成；`raw_curvature_*` 仅作为扩散前对照场。",
            "",
            visual_field_table(models, RESULT_ROOT),
            "",
        ]
    )

    if len(baseline_rows) == model_count and len(comparisons) == model_count:
        cd_better = count_where(comparisons, lambda row: row["delta_cd_pct"] < -5.0)
        cd_worse = count_where(comparisons, lambda row: row["delta_cd_pct"] > 5.0)
        qmin_down = count_where(comparisons, lambda row: row["delta_qmin"] < -0.05)
        viol_up = count_where(comparisons, lambda row: row["delta_violation"] > 0.03)
        judged_v210 = count_where(comparisons, lambda row: row["judgment"] == "v2.10 geometry better")
        judged_v281 = count_where(comparisons, lambda row: row["judgment"] == "v2.8.1 steadier")
        judged_tradeoff = model_count - judged_v210 - judged_v281

        lines.extend(
            [
                "## 与 v2.8.1 对比",
                "",
                "对比基线为 `v2.8.1_percentile_target/T_all_iter10`。`d` 值均为 v2.10.0 减 v2.8.1；CD/RMSE 百分比为相对 v2.8.1 的变化。",
                "",
                comparison_table(comparisons),
                "",
                "对比统计：",
                "",
                "| 指标 | 结果 |",
                "|---|---:|",
                f"| CD 明显改善 | {cd_better} / {model_count} |",
                f"| CD 明显变差 | {cd_worse} / {model_count} |",
                f"| Qmin 明显下降 | {qmin_down} / {model_count} |",
                f"| violation 明显上升 | {viol_up} / {model_count} |",
                f"| 判定 v2.10 geometry better | {judged_v210} / {model_count} |",
                f"| 判定 v2.8.1 steadier | {judged_v281} / {model_count} |",
                f"| 判定 tradeoff | {judged_tradeoff} / {model_count} |",
                f"| 平均 dCD% | {signed_pct(average(comparisons, 'delta_cd_pct'))} |",
                f"| 平均 dRMSE% | {signed_pct(average(comparisons, 'delta_rmse_pct'))} |",
                f"| 平均 dQmin | {signed(average(comparisons, 'delta_qmin'))} |",
                f"| 平均 dviolation | {signed(average(comparisons, 'delta_violation'))} |",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## 与 v2.8.1 对比",
                "",
                f"未做完整基线对比：v2.8.1 完整结果为 {len(baseline_rows)} / {model_count}。",
                "",
            ]
        )
        if baseline_errors:
            lines.extend(["缺失基线：", "", "| Model | 原因 |", "|---|---|"])
            for model in models:
                if model in baseline_errors:
                    lines.append(f"| {model} | {baseline_errors[model]} |")
            lines.append("")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v2.10.0 target-face T-series iter10 experiments.")
    parser.add_argument("--cli", type=Path, default=DEFAULT_CLI)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--cd-samples", type=int, default=100000)
    parser.add_argument("--curvature-mode", default="robust-mean")
    parser.add_argument("--csf-alpha", type=float, default=2.0)
    parser.add_argument("--skip-runs", action="store_true", help="Only summarize existing outputs.")
    parser.add_argument("--force", action="store_true", help="Rerun every model even if complete.")
    parser.add_argument("--models", default="", help="Comma-separated subset such as T0,T21. Empty means all T*.off.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.cli = args.cli.resolve()
    if not args.skip_runs and not args.cli.exists():
        print(f"missing CLI executable: {args.cli}", file=sys.stderr)
        return 2

    all_model_paths = discover_models()
    if args.models:
        selected = {item.strip() for item in args.models.split(",") if item.strip()}
        model_paths = [path for path in all_model_paths if path.stem in selected]
        missing = sorted(selected.difference(path.stem for path in model_paths), key=natural_model_key)
        if missing:
            print(f"unknown models: {', '.join(missing)}", file=sys.stderr)
            return 2
    else:
        model_paths = all_model_paths

    models = [path.stem for path in model_paths]
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)

    failures: dict[str, str] = {}
    if not args.skip_runs:
        for model_path in model_paths:
            model = model_path.stem
            if not args.force and case_is_complete(RESULT_ROOT, model, expected_mapping=EXPECTED_MAPPING):
                print(f"[skip] {model} already complete")
                continue
            print(f"[run] {model}")
            try:
                run_case(model_path, args)
            except Exception as exc:  # noqa: BLE001 - continue to summarize completed cases.
                failures[model] = str(exc)
                print(f"[fail] {model}: {exc}", file=sys.stderr)

    current_rows, current_errors = collect_rows(models, RESULT_ROOT, expected_mapping=EXPECTED_MAPPING)
    current_errors.update(failures)
    baseline_rows, baseline_errors = collect_rows(models, BASELINE_ROOT, expected_mapping=None)

    baseline_by_model = {row["model"]: row for row in baseline_rows}
    current_by_model = {row["model"]: row for row in current_rows}
    comparisons = [
        compare_rows(current_by_model[model], baseline_by_model[model])
        for model in models
        if model in current_by_model and model in baseline_by_model
    ]

    write_csv(RUN_SUMMARY_PATH, RUN_FIELDS, current_rows)
    if len(baseline_rows) == len(models):
        write_csv(COMPARE_SUMMARY_PATH, COMPARE_FIELDS, comparisons)
    write_report(models, current_rows, current_errors, baseline_rows, baseline_errors, comparisons, args)

    print(f"summary: {rel(RUN_SUMMARY_PATH)}")
    print(f"report: {rel(REPORT_PATH)}")
    if current_errors:
        print(f"incomplete models: {', '.join(sorted(current_errors, key=natural_model_key))}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
