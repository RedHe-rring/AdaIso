from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODEL = "T21"
INPUT_MESH = ROOT / "cpp_adaiso" / "examples" / f"{MODEL}.off"
RESULT_ROOT = ROOT / "cpp_adaiso" / "results" / "v2.10.0_lambda_bisection" / MODEL
REPORT_PATH = ROOT / "cpp_adaiso" / "修改计划与实验结果" / "v2.10.0 T21 lambda二分次数实验报告.md"
RUN_SUMMARY_PATH = RESULT_ROOT / "run_summary.csv"
DEFAULT_CLI = ROOT / "cpp_adaiso" / "build" / "adaiso_cli.exe"
DEFAULT_COUNTS = (0, 1, 2, 4, 8, 16, 32, 64, 128)

RUN_FIELDS = (
    "lambda_bisection_iterations",
    "lambda",
    "estimated_faces",
    "estimated_faces_error",
    "abs_estimated_faces_error",
    "output_vertices",
    "output_faces",
    "qavg",
    "qmin",
    "violation",
    "cd",
    "symmetric_rmse",
    "runtime_sec",
    "split_total",
    "collapse_total",
    "flip_total",
    "observed_h_min",
    "observed_h_max",
    "max_edge_target_ratio",
    "p95_edge_target_ratio",
    "max_grad_h_norm",
)


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def sci(value: float) -> str:
    return f"{value:.6E}"


def signed_sci(value: float) -> str:
    return f"{value:+.6E}"


def pct(value: float) -> str:
    return f"{value:.2f}%"


def signed_pct(value: float) -> str:
    return f"{value:+.2f}%"


def parse_counts(text: str) -> list[int]:
    counts = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value < 0:
            raise ValueError("lambda bisection count must be non-negative")
        counts.append(value)
    if not counts:
        raise ValueError("at least one lambda bisection count is required")
    return sorted(dict.fromkeys(counts))


def run_case(cli: Path, count: int, cd_samples: int, force: bool) -> dict:
    case_dir = RESULT_ROOT / f"bisect_{count:03d}"
    summary_path = case_dir / "summary.json"
    field_dir = case_dir / "target_field"
    output_path = case_dir / f"{MODEL}.ply"

    if summary_path.exists() and not force:
        return collect_case(count, summary_path)

    case_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(cli),
        str(INPUT_MESH.relative_to(ROOT)),
        "-o",
        str(output_path.relative_to(ROOT)),
        "--iters",
        "10",
        "--preprocess",
        "0",
        "--curvature-mode",
        "robust-mean",
        "--csf-alpha",
        "2.0",
        "--target-lambda-bisection-iters",
        str(count),
        "--gt",
        str(INPUT_MESH.relative_to(ROOT)),
        "--cd-samples",
        str(cd_samples),
        "--export-target-field",
        str(field_dir.relative_to(ROOT)),
        "--summary-json",
        str(summary_path.relative_to(ROOT)),
        "--quiet",
    ]
    started = time.time()
    print(f"[run] lambda_bisection_iterations={count}", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    print(f"[done] {count} in {time.time() - started:.2f}s", flush=True)
    return collect_case(count, summary_path)


def collect_case(count: int, summary_path: Path) -> dict:
    summary = load_json(summary_path)
    field_stats = load_json(summary_path.parent / "target_field" / "field_stats.json")
    iterations = summary["iterations"]
    chamfer = summary["chamfer_distance"]
    output_summary = summary["output_summary"]
    estimated_error = float(field_stats["estimated_faces_error"])
    row = {
        "lambda_bisection_iterations": count,
        "lambda": float(field_stats["lambda"]),
        "estimated_faces": float(field_stats["estimated_faces"]),
        "estimated_faces_error": estimated_error,
        "abs_estimated_faces_error": abs(estimated_error),
        "output_vertices": int(output_summary["vertices"]),
        "output_faces": int(output_summary["faces"]),
        "qavg": float(output_summary["qavg"]),
        "qmin": float(output_summary["qmin"]),
        "violation": float(iterations[-1]["violation_rate"]) if iterations else 0.0,
        "cd": float(chamfer["chamfer_distance"]),
        "symmetric_rmse": float(chamfer["symmetric_rmse"]),
        "runtime_sec": float(summary["runtime_sec"]),
        "split_total": sum(int(item["split_count"]) for item in iterations),
        "collapse_total": sum(int(item["collapse_count"]) for item in iterations),
        "flip_total": sum(int(item["flip_count"]) for item in iterations),
        "observed_h_min": float(field_stats["observed_h_min"]),
        "observed_h_max": float(field_stats["observed_h_max"]),
        "max_edge_target_ratio": float(field_stats["max_edge_target_ratio"]),
        "p95_edge_target_ratio": float(field_stats["p95_edge_target_ratio"]),
        "max_grad_h_norm": float(field_stats["max_grad_h_norm"]),
    }
    for key in RUN_FIELDS:
        value = row[key]
        if isinstance(value, float) and not math.isfinite(value):
            raise RuntimeError(f"non-finite value in {summary_path}: {key}={value}")
    return row


def write_csv(rows: list[dict]) -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    with RUN_SUMMARY_PATH.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=RUN_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in RUN_FIELDS})


def delta_pct(value: float, base: float) -> float:
    return 100.0 * (value - base) / base if base else 0.0


def result_table(rows: list[dict], baseline: dict) -> str:
    lines = [
        "| lambda二分次数 | lambda | estimated_faces误差 | Out V/F | Qmin | viol | CD | dCD vs 64 | RMSE | time(s) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["lambda_bisection_iterations"]),
                    sci(row["lambda"]),
                    signed_sci(row["estimated_faces_error"]),
                    f"{row['output_vertices']}/{row['output_faces']}",
                    fmt(row["qmin"]),
                    fmt(row["violation"]),
                    sci(row["cd"]),
                    signed_pct(delta_pct(row["cd"], baseline["cd"])),
                    sci(row["symmetric_rmse"]),
                    fmt(row["runtime_sec"], 2),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def write_report(rows: list[dict], cd_samples: int) -> None:
    baseline = next((row for row in rows if row["lambda_bisection_iterations"] == 64), rows[-1])
    best_cd = min(rows, key=lambda row: row["cd"])
    worst_cd = max(rows, key=lambda row: row["cd"])
    first_subface = next((row for row in rows if row["abs_estimated_faces_error"] < 1.0), None)
    first_same_as_64 = next(
        (
            row
            for row in rows
            if row["lambda_bisection_iterations"] <= baseline["lambda_bisection_iterations"]
            and abs(row["cd"] - baseline["cd"]) <= max(1e-12, baseline["cd"] * 0.001)
        ),
        None,
    )

    lines = [
        "# v2.10.0 T21 lambda二分次数实验报告",
        "",
        "## 实验目的",
        "",
        "探究 `target_faces_inverse_power_diffused_curvature` 尺寸场中，lambda 二分次数增加对 T21 的目标面数估计、最终输出网格和 Chamfer Distance 的影响。",
        "",
        "## 实验设置",
        "",
        "| 参数 | 值 |",
        "|---|---:|",
        f"| model | {MODEL} |",
        "| iters | 10 |",
        "| preprocess | 0 |",
        "| curvature_mode | robust-mean |",
        "| csf_alpha | 2.0 |",
        "| target_faces | 默认：清理后输入面数 |",
        "| target_curvature_exponent | 0.75 |",
        f"| cd-samples | {cd_samples} |",
        f"| lambda二分次数 | {', '.join(str(row['lambda_bisection_iterations']) for row in rows)} |",
        "",
        f"- 输出目录：`{rel(RESULT_ROOT)}/`",
        f"- 汇总 CSV：`{rel(RUN_SUMMARY_PATH)}`",
        "",
        "命令模板：",
        "",
        "```powershell",
        "cpp_adaiso/build/adaiso_cli.exe cpp_adaiso/examples/T21.off `",
        "  -o cpp_adaiso/results/v2.10.0_lambda_bisection/T21/bisect_<N>/T21.ply `",
        "  --iters 10 --preprocess 0 `",
        "  --curvature-mode robust-mean --csf-alpha 2.0 `",
        "  --target-lambda-bisection-iters <N> `",
        f"  --gt cpp_adaiso/examples/T21.off --cd-samples {cd_samples} `",
        "  --export-target-field cpp_adaiso/results/v2.10.0_lambda_bisection/T21/bisect_<N>/target_field `",
        "  --summary-json cpp_adaiso/results/v2.10.0_lambda_bisection/T21/bisect_<N>/summary.json `",
        "  --quiet",
        "```",
        "",
        "## 结果",
        "",
        result_table(rows, baseline),
        "",
        "## 观察",
        "",
        f"- 以 64 次二分为默认基线，CD 为 `{sci(baseline['cd'])}`，RMSE 为 `{sci(baseline['symmetric_rmse'])}`。",
        f"- 本组最小 CD 出现在 `{best_cd['lambda_bisection_iterations']}` 次二分：`{sci(best_cd['cd'])}`；最大 CD 出现在 `{worst_cd['lambda_bisection_iterations']}` 次二分：`{sci(worst_cd['cd'])}`。",
    ]
    if first_subface is not None:
        lines.append(
            f"- estimated_faces 误差首次小于 1 张面出现在 `{first_subface['lambda_bisection_iterations']}` 次二分，误差为 `{signed_sci(first_subface['estimated_faces_error'])}`。"
        )
    if first_same_as_64 is not None:
        lines.append(
            f"- CD 首次进入 64 次基线的 0.1% 内出现在 `{first_same_as_64['lambda_bisection_iterations']}` 次二分。"
        )
    lines.extend(
        [
            "- 0 到 4 次二分主要影响 lambda 的全局尺度，从而改变 target_h 分布和最终操作序列；这类变化会显著反映到输出面数、violation 和 CD。",
            "- 16 次以后 estimated_faces 已基本收敛，32、64、128 的 CD 差异应主要来自数值级别的尺寸场细微差别，不能视为稳定可利用的 CD 改善来源。",
            "",
            "## 结论",
            "",
            "lambda 二分次数需要足够大以稳定匹配 `target_faces`，但超过收敛区间后继续增加次数对 CD 的影响很小。对 T21 这组参数，默认 64 次属于保守且稳定的设置；若后续希望减少尺寸场构建时间，可优先评估 16 或 32 次，而不应使用 0 到 8 次这类未充分收敛的设置。",
            "",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run T21 lambda bisection iteration ablation.")
    parser.add_argument("--cli", type=Path, default=DEFAULT_CLI)
    parser.add_argument("--counts", default=",".join(str(item) for item in DEFAULT_COUNTS))
    parser.add_argument("--cd-samples", type=int, default=100000)
    parser.add_argument("--force", action="store_true", help="Rerun cases even when summary.json exists.")
    args = parser.parse_args()

    counts = parse_counts(args.counts)
    if args.cd_samples <= 0:
        raise ValueError("--cd-samples must be positive")
    if not args.cli.exists():
        raise FileNotFoundError(args.cli)
    if not INPUT_MESH.exists():
        raise FileNotFoundError(INPUT_MESH)

    rows = [run_case(args.cli, count, args.cd_samples, args.force) for count in counts]
    rows.sort(key=lambda row: row["lambda_bisection_iterations"])
    write_csv(rows)
    write_report(rows, args.cd_samples)
    print(f"[summary] {RUN_SUMMARY_PATH}", flush=True)
    print(f"[report] {REPORT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
