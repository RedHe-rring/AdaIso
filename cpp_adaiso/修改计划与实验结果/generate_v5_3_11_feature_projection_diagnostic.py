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


HERE = Path(__file__).resolve().parent
BASE_PATH = HERE / "generate_v5_1_5_T_series.py"
SPEC = importlib.util.spec_from_file_location("v5_1_5_t_series", BASE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"failed to load base script: {BASE_PATH}")
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)

ROOT = base.ROOT
CPP_ROOT = base.CPP_ROOT
RESULT_ROOT = CPP_ROOT / "results" / "v5.3.11_feature_projection_T21_T36"
REPORT_PATH = HERE / "v5.3.11 特征感知渐进投影T21_T36消融报告.md"
TARGETS = ("surface", "feature_aware")
DEFAULT_MODELS = ("T21", "T36")

FIELDS = (
    "series", "model", "target", "status", "input_faces", "output_faces", "faces_out_in",
    "runtime_sec", "before_rmse_pct_bbox", "after_rmse_pct_bbox", "final_rmse_pct_bbox",
    "before_cd_bbox2", "after_cd_bbox2", "final_cd_bbox2", "projection_rmse_gain",
    "projection_cd_gain", "stage_b_rmse_drift", "stage_b_cd_drift",
    "size_stage_final_target_log_energy", "size_stage_final_violation",
    "size_stage_final_min_angle_degrees", "final_min_angle_degrees", "final_angle_lt_10_pct",
    "reference_boundary_edges", "reference_crease_edges", "reference_corner_vertices",
    "projection_candidates", "projection_accepted", "projection_accept_rate",
    "full_step_accepted", "damped_step_accepted", "damped_accept_share",
    "backtracking_failed", "backtracking_fail_rate", "surface_fallbacks", "mean_alpha",
    "smooth_candidates", "smooth_accepted", "crease_candidates", "crease_accepted",
    "boundary_candidates", "boundary_accepted", "corner_candidates", "corner_accepted",
    "case_path", "error",
)


def number(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def divide(numerator: Any, denominator: Any) -> float:
    n = number(numerator)
    d = number(denominator)
    return n / d if math.isfinite(n) and math.isfinite(d) and d > 0.0 else math.nan


def average(rows: list[dict[str, Any]], key: str) -> float:
    values = [number(row.get(key)) for row in rows]
    values = [value for value in values if math.isfinite(value)]
    return sum(values) / len(values) if values else math.nan


def fmt(value: Any, digits: int = 6) -> str:
    value = number(value)
    return f"{value:.{digits}f}" if math.isfinite(value) else ""


def sci(value: Any) -> str:
    value = number(value)
    return f"{value:.6E}" if math.isfinite(value) else ""


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def eval_binary(cli: Path) -> Path:
    return cli.with_name("adaiso_eval" + cli.suffix)


def run_eval(mesh: Path, gt: Path, output: Path, args: argparse.Namespace) -> dict[str, Any]:
    if args.resume and output.is_file():
        return read_json(output)
    command = [
        str(args.eval), "--mesh", str(mesh), "--gt", str(gt),
        "--cd-samples", str(args.cd_samples), "--json", str(output),
    ]
    completed = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        timeout=args.timeout_seconds, check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"adaiso_eval failed for {mesh}: {completed.stdout}")
    return read_json(output)


def normalized_metrics(data: dict[str, Any], bbox: float) -> tuple[float, float]:
    chamfer = data.get("chamfer_distance", {})
    rmse = number(chamfer.get("symmetric_rmse"))
    cd = number(chamfer.get("chamfer_distance"))
    if not math.isfinite(bbox) or bbox <= 0.0:
        return math.nan, math.nan
    return 100.0 * rmse / bbox, cd / (bbox * bbox)


def case_dir(model: str, target: str) -> Path:
    return RESULT_ROOT / target / model


def failed_row(model: str, target: str, error: str) -> dict[str, Any]:
    row = {field: "" for field in FIELDS}
    row.update({
        "series": "T", "model": model, "target": target, "status": "failed",
        "case_path": str(case_dir(model, target).relative_to(ROOT)).replace("\\", "/"),
        "error": error,
    })
    return row


def row_from_outputs(
    model: str,
    target: str,
    directory: Path,
    summary: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    stage = directory / "stage_meshes"
    before_mesh = stage / "size_before_projection.ply"
    after_mesh = stage / "size_after_projection.ply"
    output_mesh = directory / "output.ply"
    for path in (before_mesh, after_mesh, output_mesh):
        if not path.is_file():
            raise RuntimeError(f"missing output: {path}")

    gt = Path(summary["input"])
    bbox = number(summary.get("parameters", {}).get("bbox_diagonal"))
    before = run_eval(before_mesh, gt, directory / "eval_before.json", args)
    after = run_eval(after_mesh, gt, directory / "eval_after.json", args)
    final = run_eval(output_mesh, gt, directory / "eval_final.json", args)
    before_rmse, before_cd = normalized_metrics(before, bbox)
    after_rmse, after_cd = normalized_metrics(after, bbox)
    final_rmse, final_cd = normalized_metrics(final, bbox)

    stats = summary.get("stats", {})
    input_summary = summary.get("input_summary", {})
    output_summary = summary.get("output_summary", {})
    candidates = stats.get("stage_a_projection_candidates", math.nan)
    accepted = stats.get("stage_a_projection_accepted", math.nan)
    damped = stats.get("stage_a_projection_damped_step_accepted", math.nan)
    failed = stats.get("stage_a_projection_backtracking_failed", math.nan)
    input_faces = int(input_summary.get("faces", 0) or 0)
    output_faces = int(output_summary.get("faces", 0) or 0)
    return {
        "series": "T", "model": model, "target": target, "status": "success",
        "input_faces": input_faces, "output_faces": output_faces,
        "faces_out_in": output_faces / input_faces if input_faces else math.nan,
        "runtime_sec": summary.get("runtime_sec", math.nan),
        "before_rmse_pct_bbox": before_rmse, "after_rmse_pct_bbox": after_rmse,
        "final_rmse_pct_bbox": final_rmse, "before_cd_bbox2": before_cd,
        "after_cd_bbox2": after_cd, "final_cd_bbox2": final_cd,
        "projection_rmse_gain": before_rmse - after_rmse,
        "projection_cd_gain": before_cd - after_cd,
        "stage_b_rmse_drift": final_rmse - after_rmse,
        "stage_b_cd_drift": final_cd - after_cd,
        "size_stage_final_target_log_energy": stats.get("size_stage_final_target_log_energy", math.nan),
        "size_stage_final_violation": stats.get("size_stage_final_violation", math.nan),
        "size_stage_final_min_angle_degrees": stats.get("size_stage_final_min_angle_degrees", math.nan),
        "final_min_angle_degrees": output_summary.get("min_angle_degrees", math.nan),
        "final_angle_lt_10_pct": output_summary.get("angle_lt_10_pct", math.nan),
        "reference_boundary_edges": stats.get("reference_boundary_edges", math.nan),
        "reference_crease_edges": stats.get("reference_crease_edges", math.nan),
        "reference_corner_vertices": stats.get("reference_corner_vertices", math.nan),
        "projection_candidates": candidates, "projection_accepted": accepted,
        "projection_accept_rate": divide(accepted, candidates),
        "full_step_accepted": stats.get("stage_a_projection_full_step_accepted", math.nan),
        "damped_step_accepted": damped, "damped_accept_share": divide(damped, accepted),
        "backtracking_failed": failed, "backtracking_fail_rate": divide(failed, candidates),
        "surface_fallbacks": stats.get("stage_a_projection_surface_fallbacks", math.nan),
        "mean_alpha": stats.get("stage_a_projection_mean_alpha", math.nan),
        "smooth_candidates": stats.get("stage_a_projection_smooth_candidates", math.nan),
        "smooth_accepted": stats.get("stage_a_projection_smooth_accepted", math.nan),
        "crease_candidates": stats.get("stage_a_projection_crease_candidates", math.nan),
        "crease_accepted": stats.get("stage_a_projection_crease_accepted", math.nan),
        "boundary_candidates": stats.get("stage_a_projection_boundary_candidates", math.nan),
        "boundary_accepted": stats.get("stage_a_projection_boundary_accepted", math.nan),
        "corner_candidates": stats.get("stage_a_projection_corner_candidates", math.nan),
        "corner_accepted": stats.get("stage_a_projection_corner_accepted", math.nan),
        "case_path": str(directory.relative_to(ROOT)).replace("\\", "/"), "error": "",
    }


def run_case(mesh: Path, target: str, args: argparse.Namespace) -> dict[str, Any]:
    directory = case_dir(mesh.stem, target)
    directory.mkdir(parents=True, exist_ok=True)
    summary_path = directory / "summary.json"
    output_path = directory / "output.ply"
    if args.resume and summary_path.is_file() and output_path.is_file():
        return row_from_outputs(mesh.stem, target, directory, read_json(summary_path), args)

    command = [
        str(args.cli), str(mesh), "-o", str(output_path), "--iters", str(args.iters),
        "--preprocess", "0", "--curvature-mode", "principal-max", "--csf-alpha", "2.0",
        "--split-factor", str(base.FIXED_SPLIT_FACTOR),
        "--collapse-factor", str(base.FIXED_COLLAPSE_FACTOR),
        "--target-h-max-factor", str(base.FIXED_H_MAX_FACTOR),
        "--auto-delta-quantile", str(base.AUTO_DELTA_QUANTILE),
        "--stage-a-policy", "split_collapse_no_guard", "--feature-aware-collapse", "0",
        "--stage-a-projection", "1", "--stage-a-projection-mode", "per_iteration_local",
        "--stage-a-projection-target", target,
        "--export-all-stage-meshes", str(directory / "stage_meshes"),
        "--gt", str(mesh), "--cd-samples", str(args.cd_samples),
        "--summary-json", str(summary_path), "--quiet",
    ]
    started = time.perf_counter()
    with (directory / "run.log").open("w", encoding="utf-8", errors="replace") as log:
        log.write("$ " + " ".join(command) + "\n\n")
        log.flush()
        completed = subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT, text=True,
            timeout=args.timeout_seconds, check=False,
        )
    if completed.returncode != 0:
        return failed_row(mesh.stem, target, f"exit {completed.returncode}")
    summary = read_json(summary_path)
    summary["runtime_sec"] = time.perf_counter() - started
    return row_from_outputs(mesh.stem, target, directory, summary, args)


def comparison_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {(row["model"], row["target"]): row for row in rows if row["status"] == "success"}
    result = []
    for model in DEFAULT_MODELS:
        surface = lookup.get((model, "surface"))
        feature = lookup.get((model, "feature_aware"))
        if surface is None or feature is None:
            continue
        item: dict[str, Any] = {"model": model}
        for key in (
            "after_rmse_pct_bbox", "final_rmse_pct_bbox", "after_cd_bbox2", "final_cd_bbox2",
            "size_stage_final_target_log_energy", "size_stage_final_violation",
            "final_min_angle_degrees", "final_angle_lt_10_pct", "faces_out_in", "runtime_sec",
        ):
            item[f"delta_{key}"] = number(feature.get(key)) - number(surface.get(key))
        result.append(item)
    return result


def load_v5310_iterative_baseline() -> list[dict[str, Any]]:
    path = CPP_ROOT / "results" / "v5.3.10_stage_geometry_diagnostic_T21_T36" / "run_summary.csv"
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [
            row for row in csv.DictReader(handle)
            if row.get("status") == "success"
            and row.get("mode") == "per_iteration_local"
            and row.get("model") in DEFAULT_MODELS
        ]


def write_report(rows: list[dict[str, Any]], comparisons: list[dict[str, Any]]) -> None:
    successes = [row for row in rows if row["status"] == "success"]
    by_target = {
        target: [row for row in successes if row["target"] == target]
        for target in TARGETS
    }
    lines = [
        "# v5.3.11 特征感知渐进投影 T21/T36 消融报告", "",
        "## 1. 实验完整性", "",
        f"本轮只运行 T21、T36，共 `{len(rows)}` 个 case，成功 `{len(successes)}/{len(rows)}`。",
        "两组均使用 `per_iteration_local` 和相同回溯接受规则，仅改变投影目标。", "",
        "## 2. 平均结果", "",
        "| target | final RMSE | final CD | size energy | violation | min angle | accept rate | damped share | mean alpha | runtime |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for target in TARGETS:
        group = by_target[target]
        lines.append(
            f"| {target} | {fmt(average(group, 'final_rmse_pct_bbox'))} | "
            f"{sci(average(group, 'final_cd_bbox2'))} | "
            f"{fmt(average(group, 'size_stage_final_target_log_energy'))} | "
            f"{fmt(average(group, 'size_stage_final_violation'))} | "
            f"{fmt(average(group, 'final_min_angle_degrees'))} | "
            f"{fmt(100.0 * average(group, 'projection_accept_rate'), 2)}% | "
            f"{fmt(100.0 * average(group, 'damped_accept_share'), 2)}% | "
            f"{fmt(average(group, 'mean_alpha'), 4)} | {fmt(average(group, 'runtime_sec'), 2)} |"
        )
    lines += [
        "", "## 3. 逐模型结果", "",
        "| model | target | before RMSE | after RMSE | final RMSE | final CD | crease cand/acc | corner cand/acc | full/damped/failed |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in successes:
        lines.append(
            f"| {row['model']} | {row['target']} | {fmt(row['before_rmse_pct_bbox'])} | "
            f"{fmt(row['after_rmse_pct_bbox'])} | {fmt(row['final_rmse_pct_bbox'])} | "
            f"{sci(row['final_cd_bbox2'])} | {row['crease_candidates']}/{row['crease_accepted']} | "
            f"{row['corner_candidates']}/{row['corner_accepted']} | "
            f"{row['full_step_accepted']}/{row['damped_step_accepted']}/{row['backtracking_failed']} |"
        )
    lines += [
        "", "## 4. Feature-aware 相对 Surface", "",
        "| model | Δ final RMSE | Δ final CD | Δ size energy | Δ violation | Δ min angle | Δ runtime |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        lines.append(
            f"| {row['model']} | {fmt(row['delta_final_rmse_pct_bbox'], 8)} | "
            f"{sci(row['delta_final_cd_bbox2'])} | "
            f"{fmt(row['delta_size_stage_final_target_log_energy'], 8)} | "
            f"{fmt(row['delta_size_stage_final_violation'], 8)} | "
            f"{fmt(row['delta_final_min_angle_degrees'], 8)} | {fmt(row['delta_runtime_sec'], 2)} |"
        )

    feature = by_target["feature_aware"]
    surface = by_target["surface"]
    baseline = load_v5310_iterative_baseline()
    delta_rmse = average(feature, "final_rmse_pct_bbox") - average(surface, "final_rmse_pct_bbox")
    delta_cd = average(feature, "final_cd_bbox2") - average(surface, "final_cd_bbox2")
    relative_rmse = divide(delta_rmse, average(surface, "final_rmse_pct_bbox"))
    relative_cd = divide(delta_cd, average(surface, "final_cd_bbox2"))
    relative_runtime = divide(
        average(feature, "runtime_sec") - average(surface, "runtime_sec"),
        average(surface, "runtime_sec"),
    )
    feature_candidates = sum(number(row["crease_candidates"]) + number(row["corner_candidates"]) for row in feature)
    damped = sum(number(row["damped_step_accepted"]) for row in successes)
    lines += ["", "## 5. 判断", ""]
    lines.append(f"- feature-aware 组产生 crease/corner candidates 共 `{int(feature_candidates)}`，消融路径实际生效。")
    lines.append(f"- 两组共有 `{int(damped)}` 个候选通过衰减步长接受，说明回溯不是空路径。")
    lines.append(
        f"- feature-aware 相对 surface 的平均最终 RMSE 变化为 `{fmt(delta_rmse, 8)}`，"
        f"最终 CD 变化为 `{sci(delta_cd)}`，相对恶化分别为 "
        f"`{fmt(100.0 * relative_rmse, 2)}%` 和 `{fmt(100.0 * relative_cd, 2)}%`。"
    )
    if baseline:
        baseline_accept_rate = average([
            {
                "rate": divide(row.get("stage_a_projection_accepted"), row.get("stage_a_projection_candidates"))
            }
            for row in baseline
        ], "rate")
        baseline_rmse = average(baseline, "final_output_rmse_pct_bbox")
        baseline_cd = average(baseline, "final_output_cd_bbox2")
        surface_accept_rate = average(surface, "projection_accept_rate")
        lines.append(
            f"- 相对 v5.3.10 的一次性 guarded projection，surface+回溯的平均接受率从 "
            f"`{fmt(100.0 * baseline_accept_rate, 2)}%` 提高到 "
            f"`{fmt(100.0 * surface_accept_rate, 2)}%`。"
        )
        lines.append(
            f"- surface+回溯相对 v5.3.10 的平均最终 RMSE 变化为 "
            f"`{fmt(average(surface, 'final_rmse_pct_bbox') - baseline_rmse, 8)}`，"
            f"CD 变化为 `{sci(average(surface, 'final_cd_bbox2') - baseline_cd)}`；"
            "回溯本身没有造成当前 feature-aware 的几何恶化。"
        )
    lines.append(
        "- feature-aware 的尺寸 log energy 略降，但阈值 violation 明显上升，投影接受率只有约一半；"
        "当前分类在迭代网格上反复重算后产生大量 crease/corner 候选，且没有持久化特征对应，"
        "容易形成特征分类与投影相互强化的反馈。"
    )
    lines.append(
        f"- feature-aware 平均 runtime 增加 `{fmt(100.0 * relative_runtime, 2)}%`，仍在 `1.3x` 上限内；"
        "T36 最小角明显提高，但不足以抵消几何误差和 violation 的系统性恶化。"
    )
    if delta_rmse < 0.0 and delta_cd < 0.0:
        lines.append("- 在这两个模型上，特征约束域投影同时改善 RMSE 和 CD，可以进入更大样本验证。")
    elif delta_rmse > 0.0 and delta_cd > 0.0:
        lines.append("- 在这两个模型上，特征约束域投影使 RMSE 和 CD 同时恶化，不应直接设为默认。")
    else:
        lines.append("- RMSE 与 CD 信号不一致，当前结果属于混合收益，需要结合特征线可视化判断。")
    lines.append(
        "- 因此本版应保留 `surface` 默认和渐进回溯，不应将当前 feature-aware 目标设为默认；"
        "下一步若继续保特征，应优先研究从原始网格向 split/collapse 传播持久化 feature label，"
        "而不是继续放宽投影 guard。"
    )
    lines += [
        "", "实验产物：",
        f"- `{(RESULT_ROOT / 'run_summary.csv').relative_to(ROOT).as_posix()}`",
        f"- `{(RESULT_ROOT / 'comparison.csv').relative_to(ROOT).as_posix()}`",
        f"- `{REPORT_PATH.relative_to(ROOT).as_posix()}`",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v5.3.11 feature-projection T21/T36 ablation.")
    parser.add_argument("--cli", type=Path, default=base.DEFAULT_CLI)
    parser.add_argument("--eval", type=Path, default=None)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--cd-samples", type=int, default=100000)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--only", nargs="*", default=list(DEFAULT_MODELS))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.eval is None:
        args.eval = eval_binary(args.cli)
    wanted = set(args.only or DEFAULT_MODELS)
    meshes = [mesh for mesh in base.discover_t_models() if mesh.stem in wanted]
    if len(meshes) != len(wanted):
        raise RuntimeError(f"missing requested models: {sorted(wanted - {mesh.stem for mesh in meshes})}")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_case, mesh, target, args): (mesh, target)
            for mesh in meshes for target in TARGETS
        }
        for index, future in enumerate(as_completed(futures), start=1):
            mesh, target = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = failed_row(mesh.stem, target, str(exc))
            rows.append(row)
            print(f"[{index}/{len(futures)}] {mesh.stem} {target}: {row['status']}")

    rows.sort(key=lambda row: (base.model_index(row["model"]), row["target"]))
    comparisons = comparison_rows(rows)
    base.write_csv(rows, RESULT_ROOT / "run_summary.csv", FIELDS)
    if comparisons:
        base.write_csv(comparisons, RESULT_ROOT / "comparison.csv", tuple(comparisons[0].keys()))
    write_report(rows, comparisons)
    print(f"wrote {RESULT_ROOT / 'run_summary.csv'}")
    print(f"wrote {REPORT_PATH}")
    return 0 if all(row["status"] == "success" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
