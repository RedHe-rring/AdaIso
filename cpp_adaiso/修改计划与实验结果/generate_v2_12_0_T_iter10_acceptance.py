from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = ROOT / "cpp_adaiso" / "examples"
RESULT_ROOT = ROOT / "cpp_adaiso" / "results" / "v2.12.0_T_all_iter10"
BASELINE_ROOT = ROOT / "cpp_adaiso" / "results" / "v2.10.0_target_faces" / "T_all_iter10"
REPORT_PATH = ROOT / "cpp_adaiso" / "修改计划与实验结果" / "v2.12.0 T系列 iter10验收实验报告.md"
RUN_SUMMARY_PATH = RESULT_ROOT / "run_summary.csv"
COMPARE_SUMMARY_PATH = RESULT_ROOT / "comparison_vs_v2.10.0.csv"
DEFAULT_CLI = ROOT / "cpp_adaiso" / "build2" / "adaiso_cli.exe"

EXPECTED_MAPPING = "target_faces_inverse_sqrt_diffused_curvature_cap0995"

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
    "target_size_mapping",
    "csf_solver_success",
    "csf_residual_scaled",
    "target_lambda",
    "target_estimated_faces",
    "target_h_min",
    "target_h_max",
    "target_observed_h_min",
    "target_observed_h_max",
    "curvature_cap_value",
    "curvature_cap_changed_rate",
    "split_total",
    "collapse_total",
    "flip_total",
)

COMPARE_FIELDS = (
    "model",
    "v212_output_faces",
    "v210_output_faces",
    "delta_output_faces",
    "delta_output_faces_pct",
    "v212_qavg",
    "v210_qavg",
    "delta_qavg",
    "v212_qmin",
    "v210_qmin",
    "delta_qmin",
    "v212_violation",
    "v210_violation",
    "delta_violation",
    "delta_violation_pct",
    "v212_cd",
    "v210_cd",
    "delta_cd",
    "delta_cd_pct",
    "v212_rmse",
    "v210_rmse",
    "delta_rmse",
    "delta_rmse_pct",
    "v212_runtime_sec",
    "v210_runtime_sec",
    "delta_runtime_sec",
    "delta_runtime_sec_pct",
    "v212_cap_changed_rate",
)


def natural_model_key(path_or_name: Path | str) -> int:
    name = path_or_name.stem if isinstance(path_or_name, Path) else str(path_or_name)
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


def fmt(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def sci(value: float) -> str:
    return f"{value:.6E}"


def signed(value: float, digits: int = 6) -> str:
    return f"{value:+.{digits}f}"


def signed_pct(value: float) -> str:
    return f"{value:+.2f}%"


def pct_delta(current: float, baseline: float) -> float:
    if baseline == 0.0:
        return 0.0 if current == 0.0 else math.copysign(math.inf, current)
    return 100.0 * (current - baseline) / baseline


def discover_models() -> list[Path]:
    models: list[Path] = []
    for path in EXAMPLES_ROOT.glob("T*.off"):
        try:
            natural_model_key(path)
        except ValueError:
            continue
        models.append(path)
    models.sort(key=natural_model_key)
    if len(models) != 31:
        raise RuntimeError(f"expected 31 T-series models, found {len(models)}")
    return models


def case_dir(root: Path, model: str) -> Path:
    return root / model


def tail_violation(summary: dict) -> float:
    iterations = summary.get("iterations", [])
    if not iterations:
        return 0.0
    return float(iterations[-1].get("violation_rate", 0.0))


def operator_total(summary: dict, key: str, stat_key: str) -> int:
    stats = summary.get("stats", {})
    if stat_key in stats:
        return int(stats[stat_key])
    return sum(int(item.get(key, 0)) for item in summary.get("iterations", []))


def read_case(root: Path, model: str) -> dict:
    summary = load_json(case_dir(root, model) / "summary.json")
    parameters = summary.get("parameters", {})
    stats = summary.get("stats", {})
    input_summary = summary["input_summary"]
    output_summary = summary["output_summary"]
    chamfer = summary["chamfer_distance"]
    input_faces = int(input_summary["faces"])
    target_faces = int(parameters.get("target_faces", input_faces))
    row = {
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
        "target_size_mapping": str(parameters.get("target_size_mapping", "")),
        "csf_solver_success": bool(stats.get("csf_solver_success", False)),
        "csf_residual_scaled": float(stats.get("csf_residual_scaled", 0.0)),
        "target_lambda": float(stats.get("target_lambda", 0.0)),
        "target_estimated_faces": float(stats.get("target_estimated_faces", 0.0)),
        "target_h_min": float(stats.get("target_h_min", 0.0)),
        "target_h_max": float(stats.get("target_h_max", 0.0)),
        "target_observed_h_min": float(stats.get("target_observed_h_min", 0.0)),
        "target_observed_h_max": float(stats.get("target_observed_h_max", 0.0)),
        "curvature_cap_value": float(stats.get("curvature_cap_value", 0.0)),
        "curvature_cap_changed_rate": float(stats.get("curvature_cap_changed_rate", 0.0)),
        "split_total": operator_total(summary, "split_count", "split_total"),
        "collapse_total": operator_total(summary, "collapse_count", "collapse_total"),
        "flip_total": operator_total(summary, "flip_count", "flip_total"),
        "global_iterations": int(parameters.get("global_iterations", len(summary.get("iterations", [])))),
        "cd_samples": int(chamfer.get("samples_per_direction", 0)),
    }
    return row


def case_is_complete(root: Path, model: str, *, iters: int, cd_samples: int, expected_mapping: str | None) -> bool:
    directory = case_dir(root, model)
    if not (directory / "summary.json").is_file() or not (directory / "output.ply").is_file():
        return False
    try:
        row = read_case(root, model)
    except (KeyError, ValueError, OSError, json.JSONDecodeError, RuntimeError):
        return False
    if row["global_iterations"] != iters or row["cd_samples"] != cd_samples:
        return False
    if expected_mapping is not None and row["target_size_mapping"] != expected_mapping:
        return False
    return bool(row["csf_solver_success"])


def run_case(model_path: Path, args: argparse.Namespace) -> None:
    model = model_path.stem
    directory = case_dir(RESULT_ROOT, model)
    directory.mkdir(parents=True, exist_ok=True)
    command = [
        str(args.cli),
        rel_arg(model_path),
        "-o",
        rel_arg(directory / "output.ply"),
        "--preprocess",
        "0",
        "--iters",
        str(args.iters),
        "--gt",
        rel_arg(model_path),
        "--cd-samples",
        str(args.cd_samples),
        "--summary-json",
        rel_arg(directory / "summary.json"),
        "--quiet",
    ]
    (directory / "command.txt").write_text(subprocess.list2cmdline(command) + "\n", encoding="utf-8")
    started = time.perf_counter()
    proc = subprocess.run(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    elapsed = time.perf_counter() - started
    (directory / "stdout.log").write_text(proc.stdout, encoding="utf-8")
    (directory / "stderr.log").write_text(proc.stderr, encoding="utf-8")
    print(f"[{model}] exit={proc.returncode} elapsed={elapsed:.1f}s", flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{model} failed; see {rel(directory / 'stderr.log')}")


def collect_rows(models: list[Path], root: Path, *, iters: int, cd_samples: int, expected_mapping: str | None) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    errors: list[str] = []
    for path in models:
        model = path.stem
        if not case_is_complete(root, model, iters=iters, cd_samples=cd_samples, expected_mapping=expected_mapping):
            errors.append(f"{model}: incomplete or parameter mismatch")
            continue
        try:
            rows.append(read_case(root, model))
        except Exception as error:  # noqa: BLE001
            errors.append(f"{model}: {error}")
    rows.sort(key=lambda row: natural_model_key(row["model"]))
    return rows, errors


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def compare_rows(current: dict, baseline: dict) -> dict:
    result = {
        "model": current["model"],
        "v212_cap_changed_rate": current["curvature_cap_changed_rate"],
    }
    for label, key in (
        ("output_faces", "output_faces"),
        ("qavg", "qavg"),
        ("qmin", "qmin"),
        ("violation", "iter10_violation"),
        ("cd", "cd"),
        ("rmse", "symmetric_rmse"),
        ("runtime_sec", "runtime_sec"),
    ):
        cv = float(current[key])
        bv = float(baseline[key])
        result[f"v212_{label}"] = cv
        result[f"v210_{label}"] = bv
        result[f"delta_{label}"] = cv - bv
        if label in {"output_faces", "violation", "cd", "rmse", "runtime_sec"}:
            result[f"delta_{label}_pct"] = pct_delta(cv, bv)
    return result


def average(rows: list[dict], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows) if rows else math.nan


def count_where(rows: list[dict], predicate) -> int:
    return sum(1 for row in rows if predicate(row))


def result_table(rows: list[dict]) -> str:
    lines = [
        "| Model | Out F | Qavg | Qmin | viol | CD | RMSE | cap changed | split/collapse/flip | runtime |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['output_faces']} | {fmt(row['qavg'])} | {fmt(row['qmin'])} | "
            f"{fmt(row['iter10_violation'])} | {sci(row['cd'])} | {sci(row['symmetric_rmse'])} | "
            f"{100.0 * row['curvature_cap_changed_rate']:.3f}% | "
            f"{row['split_total']}/{row['collapse_total']}/{row['flip_total']} | {row['runtime_sec']:.2f}s |"
        )
    return "\n".join(lines)


def comparison_table(rows: list[dict]) -> str:
    lines = [
        "| Model | dCD% | dRMSE% | dFaces% | dQavg | dQmin | dviol | dviol% | dRuntime% |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {signed_pct(row['delta_cd_pct'])} | {signed_pct(row['delta_rmse_pct'])} | "
            f"{signed_pct(row['delta_output_faces_pct'])} | {signed(row['delta_qavg'])} | "
            f"{signed(row['delta_qmin'])} | {signed(row['delta_violation'])} | "
            f"{signed_pct(row['delta_violation_pct'])} | {signed_pct(row['delta_runtime_sec_pct'])} |"
        )
    return "\n".join(lines)


def top_models(rows: list[dict], key: str, count: int, reverse: bool) -> str:
    selected = sorted(rows, key=lambda row: row[key], reverse=reverse)[:count]
    return ", ".join(f"`{row['model']}` ({row[key]:+.4g})" for row in selected)


def git_status_text() -> str:
    proc = subprocess.run(
        ["git", "status", "--short"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return proc.stdout.strip() or "clean"


def write_report(
    models: list[Path],
    current_rows: list[dict],
    baseline_rows: list[dict],
    comparisons: list[dict],
    current_errors: list[str],
    baseline_errors: list[str],
    args: argparse.Namespace,
) -> None:
    baseline_by_model = {row["model"]: row for row in baseline_rows}
    current_by_model = {row["model"]: row for row in current_rows}
    key_models = [m for m in ("T21", "T30", "T36", "T44", "T87", "T124") if m in current_by_model and m in baseline_by_model]

    current_avg_cd = average(current_rows, "cd")
    baseline_avg_cd = average(baseline_rows, "cd")
    current_avg_violation = average(current_rows, "iter10_violation")
    baseline_avg_violation = average(baseline_rows, "iter10_violation")
    current_min_qmin = min((row["qmin"] for row in current_rows), default=math.nan)
    current_avg_runtime = average(current_rows, "runtime_sec")
    baseline_avg_runtime = average(baseline_rows, "runtime_sec")

    cd_limit = 1.05 * baseline_avg_cd
    violation_limit = 1.20 * baseline_avg_violation
    qmin_limit = 0.02
    runtime_limit = 1.20 * baseline_avg_runtime

    cd_pass = current_avg_cd <= cd_limit
    violation_pass = current_avg_violation <= violation_limit
    qmin_pass = current_min_qmin > qmin_limit
    runtime_pass = current_avg_runtime <= runtime_limit
    overall_pass = cd_pass and violation_pass and qmin_pass and runtime_pass

    lines = [
        "# v2.12.0 T系列 iter10验收实验报告",
        "",
        "## 实验设置",
        "",
        "本实验使用当前 v2.12.0 管线，对 examples 中 31 个 T 系列模型运行 10 轮重网格化。每个模型使用自身输入作为 GT，计算 area-sampled Chamfer Distance 和 symmetric RMSE，并与 v2.10.0 target_faces iter10 基线比较。",
        "",
        "| 项目 | v2.12.0 | v2.10.0 基线 |",
        "|---|---:|---:|",
        f"| 模型数量 | {len(models)} | {len(models)} |",
        f"| iters | {args.iters} | 10 |",
        "| preprocess | 0 | 0 |",
        f"| CD samples | {args.cd_samples} | 100000 |",
        f"| target_size_mapping | `{EXPECTED_MAPPING}` | `target_faces_inverse_sqrt_diffused_curvature` |",
        "| curvature cap | q=0.995，默认开启 | 无 |",
        "| log-domain smoothing | 已实现内部函数，默认关闭 | 无 |",
        "",
        f"- v2.12.0 输出目录：`{rel(RESULT_ROOT)}/`",
        f"- run summary：`{rel(RUN_SUMMARY_PATH)}`",
        f"- comparison CSV：`{rel(COMPARE_SUMMARY_PATH)}`",
        f"- v2.10.0 基线目录：`{rel(BASELINE_ROOT)}/`",
        "",
        "## 完整性检查",
        "",
        "| 检查项 | 结果 |",
        "|---|---:|",
        f"| v2.12.0 完成模型 | {len(current_rows)} / {len(models)} |",
        f"| v2.10.0 基线完整 | {len(baseline_rows)} / {len(models)} |",
        f"| 可比较模型 | {len(comparisons)} / {len(models)} |",
        f"| CSF solver success | {count_where(current_rows, lambda row: row['csf_solver_success'])} / {len(models)} |",
        f"| target_faces 匹配输入面数 | {count_where(current_rows, lambda row: row['target_faces_matches_input'])} / {len(models)} |",
        f"| 平均 curvature cap changed rate | {100.0 * average(current_rows, 'curvature_cap_changed_rate'):.4f}% |",
        "",
    ]

    if current_errors or baseline_errors:
        lines.extend(["异常项：", ""])
        lines.extend(f"- {error}" for error in current_errors + baseline_errors)
        lines.append("")

    lines.extend(
        [
            "## 预设目标验收",
            "",
            "| 目标 | 阈值 | 实测 | 结论 |",
            "|---|---:|---:|---:|",
            f"| 平均 CD 不超过 v2.10.0 超过 5% | <= {sci(cd_limit)} | {sci(current_avg_cd)} ({signed_pct(pct_delta(current_avg_cd, baseline_avg_cd))}) | {'通过' if cd_pass else '未通过'} |",
            f"| 平均 violation 不超过 v2.10.0 超过 20% | <= {fmt(violation_limit)} | {fmt(current_avg_violation)} ({signed_pct(pct_delta(current_avg_violation, baseline_avg_violation))}) | {'通过' if violation_pass else '未通过'} |",
            f"| 最低 Qmin 高于 0.02 | > {qmin_limit:.2f} | {fmt(current_min_qmin)} | {'通过' if qmin_pass else '未通过'} |",
            f"| runtime 不明显劣化 | <= v2.10.0 +20% ({current_avg_runtime:.2f}s / {baseline_avg_runtime:.2f}s) | {signed_pct(pct_delta(current_avg_runtime, baseline_avg_runtime))} | {'通过' if runtime_pass else '未通过'} |",
            "",
            f"总体判断：**{'达到预设目标' if overall_pass else '未达到预设目标'}**。",
            "",
            "## 汇总对比",
            "",
            "| 指标 | v2.12.0 | v2.10.0 | 变化 |",
            "|---|---:|---:|---:|",
            f"| 平均 CD | {sci(current_avg_cd)} | {sci(baseline_avg_cd)} | {signed_pct(pct_delta(current_avg_cd, baseline_avg_cd))} |",
            f"| 平均 RMSE | {sci(average(current_rows, 'symmetric_rmse'))} | {sci(average(baseline_rows, 'symmetric_rmse'))} | {signed_pct(pct_delta(average(current_rows, 'symmetric_rmse'), average(baseline_rows, 'symmetric_rmse')))} |",
            f"| 平均输出面数 | {average(current_rows, 'output_faces'):.1f} | {average(baseline_rows, 'output_faces'):.1f} | {signed_pct(pct_delta(average(current_rows, 'output_faces'), average(baseline_rows, 'output_faces')))} |",
            f"| 平均 Qavg | {fmt(average(current_rows, 'qavg'))} | {fmt(average(baseline_rows, 'qavg'))} | {signed(average(comparisons, 'delta_qavg'))} |",
            f"| 平均 Qmin | {fmt(average(current_rows, 'qmin'))} | {fmt(average(baseline_rows, 'qmin'))} | {signed(average(comparisons, 'delta_qmin'))} |",
            f"| 最低 Qmin | {fmt(current_min_qmin)} | {fmt(min(row['qmin'] for row in baseline_rows))} | {signed(current_min_qmin - min(row['qmin'] for row in baseline_rows))} |",
            f"| 平均 violation | {fmt(current_avg_violation)} | {fmt(baseline_avg_violation)} | {signed_pct(pct_delta(current_avg_violation, baseline_avg_violation))} |",
            f"| 平均 runtime | {current_avg_runtime:.2f}s | {baseline_avg_runtime:.2f}s | {signed_pct(pct_delta(current_avg_runtime, baseline_avg_runtime))} |",
            "",
            "## v2.12.0 单模型结果",
            "",
            result_table(current_rows),
            "",
            "## 逐模型对比 v2.12.0 - v2.10.0",
            "",
            comparison_table(comparisons),
            "",
            "## 重点模型验收",
            "",
            "| Model | v2.12 CD | v2.10 CD | dCD% | v2.12 Qmin | v2.10 Qmin | v2.12 viol | v2.10 viol |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for model in key_models:
        current = current_by_model[model]
        baseline = baseline_by_model[model]
        lines.append(
            f"| {model} | {sci(current['cd'])} | {sci(baseline['cd'])} | "
            f"{signed_pct(pct_delta(current['cd'], baseline['cd']))} | "
            f"{fmt(current['qmin'])} | {fmt(baseline['qmin'])} | "
            f"{fmt(current['iter10_violation'])} | {fmt(baseline['iter10_violation'])} |"
        )

    lines.extend(
        [
            "",
            "## 变化最大的模型",
            "",
            f"- CD 改善最大：{top_models(comparisons, 'delta_cd_pct', 5, False)}",
            f"- CD 退化最大：{top_models(comparisons, 'delta_cd_pct', 5, True)}",
            f"- Qmin 改善最大：{top_models(comparisons, 'delta_qmin', 5, True)}",
            f"- Qmin 退化最大：{top_models(comparisons, 'delta_qmin', 5, False)}",
            f"- violation 增长最大：{top_models(comparisons, 'delta_violation_pct', 5, True)}",
            "",
            "## 结论",
            "",
        ]
    )
    if overall_pass:
        lines.append("v2.12.0 在当前 31 个 T 模型 iter10 实验中达到预设验收目标。curvature cap 开启后，平均 CD、平均 violation、最低 Qmin 和平均 runtime 均在设定阈值内。")
    else:
        failed = []
        if not cd_pass:
            failed.append("平均 CD")
        if not violation_pass:
            failed.append("平均 violation")
        if not qmin_pass:
            failed.append("最低 Qmin")
        if not runtime_pass:
            failed.append("runtime")
        lines.append(
            "v2.12.0 在当前 31 个 T 模型 iter10 实验中未达到预设验收目标，未通过项为："
            + "、".join(failed)
            + "。"
        )

    lines.extend(
        [
            "",
            "## 工作区状态",
            "",
            "```text",
            git_status_text(),
            "```",
            "",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", type=Path, default=DEFAULT_CLI)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--cd-samples", type=int, default=100000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-run", action="store_true")
    args = parser.parse_args()

    models = discover_models()
    if not args.cli.is_file():
        raise RuntimeError(f"CLI executable not found: {args.cli}")

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    if not args.skip_run:
        for model_path in models:
            model = model_path.stem
            complete = case_is_complete(
                RESULT_ROOT,
                model,
                iters=args.iters,
                cd_samples=args.cd_samples,
                expected_mapping=EXPECTED_MAPPING,
            )
            if complete and not args.force:
                print(f"[{model}] skip complete", flush=True)
                continue
            run_case(model_path, args)

    current_rows, current_errors = collect_rows(
        models,
        RESULT_ROOT,
        iters=args.iters,
        cd_samples=args.cd_samples,
        expected_mapping=EXPECTED_MAPPING,
    )
    baseline_rows, baseline_errors = collect_rows(
        models,
        BASELINE_ROOT,
        iters=10,
        cd_samples=100000,
        expected_mapping="target_faces_inverse_sqrt_diffused_curvature",
    )

    write_csv(RUN_SUMMARY_PATH, RUN_FIELDS, current_rows)
    baseline_by_model = {row["model"]: row for row in baseline_rows}
    comparisons = [
        compare_rows(row, baseline_by_model[row["model"]])
        for row in current_rows
        if row["model"] in baseline_by_model
    ]
    comparisons.sort(key=lambda row: natural_model_key(row["model"]))
    write_csv(COMPARE_SUMMARY_PATH, COMPARE_FIELDS, comparisons)
    write_report(models, current_rows, baseline_rows, comparisons, current_errors, baseline_errors, args)
    print(f"wrote {rel(RUN_SUMMARY_PATH)}", flush=True)
    print(f"wrote {rel(COMPARE_SUMMARY_PATH)}", flush=True)
    print(f"wrote {rel(REPORT_PATH)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
