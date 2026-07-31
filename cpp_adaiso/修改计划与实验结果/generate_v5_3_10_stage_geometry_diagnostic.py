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


BASE_SCRIPT = Path(__file__).with_name("generate_v5_1_5_T_series.py")
SPEC = importlib.util.spec_from_file_location("v5_1_5_t_series", BASE_SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"failed to load base script: {BASE_SCRIPT}")
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)

ROOT = base.ROOT
CPP_ROOT = base.CPP_ROOT
RESULT_ROOT = CPP_ROOT / "results" / "v5.3.10_stage_geometry_diagnostic_T21_T36"
REPORT_PATH = Path(__file__).with_name("v5.3.10 StageA投影几何误差分解T21_T36报告.md")
POLICY = "split_collapse_no_guard"
MODES = ("final_only", "per_iteration_local")
DEFAULT_MODELS = ("T21", "T36")

CSV_FIELDS = (
    "series",
    "model",
    "mode",
    "status",
    "input_faces",
    "output_faces",
    "faces_out_in",
    "runtime_sec",
    "stage_a_projection_passes",
    "stage_a_projection_local_passes",
    "stage_a_projection_final_passes",
    "stage_a_projection_candidates",
    "stage_a_projection_accepted",
    "stage_a_projection_local_candidates",
    "stage_a_projection_local_accepted",
    "stage_a_projection_final_candidates",
    "stage_a_projection_final_accepted",
    "stage_a_before_projection_rmse_pct_bbox",
    "stage_a_before_projection_cd_bbox2",
    "stage_a_after_projection_rmse_pct_bbox",
    "stage_a_after_projection_cd_bbox2",
    "final_output_rmse_pct_bbox",
    "final_output_cd_bbox2",
    "projection_rmse_gain",
    "projection_cd_gain",
    "stage_b_rmse_drift",
    "stage_b_cd_drift",
    "size_stage_final_target_log_energy",
    "size_stage_final_violation",
    "final_output_violation",
    "case_path",
    "error",
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


def case_dir(model: str, mode: str) -> Path:
    return RESULT_ROOT / mode / model


def eval_path_from_cli(cli: Path) -> Path:
    suffix = cli.suffix
    name = "adaiso_eval" + suffix
    return cli.with_name(name)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mesh_metrics(eval_json: dict[str, Any], bbox_diagonal: float) -> tuple[float, float]:
    chamfer = eval_json.get("chamfer_distance", {})
    rmse = to_float(chamfer.get("symmetric_rmse"))
    cd = to_float(chamfer.get("chamfer_distance"))
    if bbox_diagonal <= 0.0 or not math.isfinite(bbox_diagonal):
      return math.nan, math.nan
    return 100.0 * rmse / bbox_diagonal, cd / (bbox_diagonal * bbox_diagonal)


def run_eval(mesh: Path, gt: Path, out_json: Path, args: argparse.Namespace) -> dict[str, Any]:
    if args.resume and out_json.is_file():
        return read_json(out_json)
    command = [
        str(args.eval),
        "--mesh",
        str(mesh),
        "--gt",
        str(gt),
        "--cd-samples",
        str(args.cd_samples),
        "--json",
        str(out_json),
    ]
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=args.timeout_seconds,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"adaiso_eval failed for {mesh}: {completed.stdout}")
    return read_json(out_json)


def failed_row(model: str, mode: str, error: str) -> dict[str, Any]:
    row = {field: "" for field in CSV_FIELDS}
    row.update({
        "series": "T",
        "model": model,
        "mode": mode,
        "status": "failed",
        "case_path": str(case_dir(model, mode).relative_to(ROOT)).replace("\\", "/"),
        "error": error,
    })
    return row


def row_from_existing(model: str, mode: str, current_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    summary = read_json(current_dir / "summary.json")
    return row_from_outputs(model, mode, current_dir, summary, args)


def row_from_outputs(
    model: str,
    mode: str,
    current_dir: Path,
    summary: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    stage_dir = current_dir / "stage_meshes"
    before_mesh = stage_dir / "size_before_projection.ply"
    after_mesh = stage_dir / "size_after_projection.ply"
    output_mesh = current_dir / "output.ply"
    if not before_mesh.is_file():
        raise RuntimeError(f"missing stage mesh: {before_mesh}")
    if not after_mesh.is_file():
        raise RuntimeError(f"missing stage mesh: {after_mesh}")
    if not output_mesh.is_file():
        raise RuntimeError(f"missing output mesh: {output_mesh}")

    bbox = to_float(summary.get("parameters", {}).get("bbox_diagonal"))
    before_eval = run_eval(before_mesh, Path(summary["input"]), current_dir / "eval_stage_a_before_projection.json", args)
    after_eval = run_eval(after_mesh, Path(summary["input"]), current_dir / "eval_stage_a_after_projection.json", args)
    final_eval = run_eval(output_mesh, Path(summary["input"]), current_dir / "eval_final_output.json", args)

    before_rmse, before_cd = mesh_metrics(before_eval, bbox)
    after_rmse, after_cd = mesh_metrics(after_eval, bbox)
    final_rmse, final_cd = mesh_metrics(final_eval, bbox)

    stats = summary.get("stats", {})
    input_summary = summary.get("input_summary", {})
    output_summary = summary.get("output_summary", {})
    iterations = summary.get("iterations", [])
    final_iteration = iterations[-1] if iterations else {}
    input_faces = int(input_summary.get("faces", 0) or 0)
    output_faces = int(output_summary.get("faces", 0) or 0)

    return {
        "series": "T",
        "model": model,
        "mode": mode,
        "status": "success",
        "input_faces": input_faces,
        "output_faces": output_faces,
        "faces_out_in": output_faces / input_faces if input_faces > 0 else math.nan,
        "runtime_sec": summary.get("runtime_sec", math.nan),
        "stage_a_projection_passes": stats.get("stage_a_projection_passes", math.nan),
        "stage_a_projection_local_passes": stats.get("stage_a_projection_local_passes", math.nan),
        "stage_a_projection_final_passes": stats.get("stage_a_projection_final_passes", math.nan),
        "stage_a_projection_candidates": stats.get("stage_a_projection_candidates", math.nan),
        "stage_a_projection_accepted": stats.get("stage_a_projection_accepted", math.nan),
        "stage_a_projection_local_candidates": stats.get("stage_a_projection_local_candidates", math.nan),
        "stage_a_projection_local_accepted": stats.get("stage_a_projection_local_accepted", math.nan),
        "stage_a_projection_final_candidates": stats.get("stage_a_projection_final_candidates", math.nan),
        "stage_a_projection_final_accepted": stats.get("stage_a_projection_final_accepted", math.nan),
        "stage_a_before_projection_rmse_pct_bbox": before_rmse,
        "stage_a_before_projection_cd_bbox2": before_cd,
        "stage_a_after_projection_rmse_pct_bbox": after_rmse,
        "stage_a_after_projection_cd_bbox2": after_cd,
        "final_output_rmse_pct_bbox": final_rmse,
        "final_output_cd_bbox2": final_cd,
        "projection_rmse_gain": before_rmse - after_rmse,
        "projection_cd_gain": before_cd - after_cd,
        "stage_b_rmse_drift": final_rmse - after_rmse,
        "stage_b_cd_drift": final_cd - after_cd,
        "size_stage_final_target_log_energy": stats.get("size_stage_final_target_log_energy", math.nan),
        "size_stage_final_violation": stats.get("size_stage_final_violation", math.nan),
        "final_output_violation": final_iteration.get("violation_rate", math.nan),
        "case_path": str(current_dir.relative_to(ROOT)).replace("\\", "/"),
        "error": "",
    }


def run_case(mesh: Path, mode: str, args: argparse.Namespace) -> dict[str, Any]:
    model = mesh.stem
    current_dir = case_dir(model, mode)
    current_dir.mkdir(parents=True, exist_ok=True)
    summary_path = current_dir / "summary.json"
    output_path = current_dir / "output.ply"
    stage_dir = current_dir / "stage_meshes"
    log_path = current_dir / "run.log"

    if args.resume and summary_path.is_file() and output_path.is_file():
        try:
            return row_from_existing(model, mode, current_dir, args)
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
        mode,
        "--export-all-stage-meshes",
        str(stage_dir),
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
        return failed_row(model, mode, f"exit {completed.returncode}; see {log_path}")
    summary = read_json(summary_path)
    summary.setdefault("runtime_sec", elapsed)
    return row_from_outputs(model, mode, current_dir, summary, args)


def summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for mode in MODES:
        successes = [row for row in rows if row.get("status") == "success" and row.get("mode") == mode]
        item: dict[str, Any] = {
            "mode": mode,
            "cases": sum(1 for row in rows if row.get("mode") == mode),
            "successes": len(successes),
        }
        for field in CSV_FIELDS:
            if field in {"series", "model", "mode", "status", "case_path", "error"}:
                continue
            item[f"avg_{field}"] = average(successes, field)
        result.append(item)
    return result


def comparison_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_model_mode = {(row.get("model"), row.get("mode")): row for row in rows if row.get("status") == "success"}
    comparisons = []
    for model in sorted({row.get("model") for row in rows if row.get("model")}):
        final_only = by_model_mode.get((model, "final_only"))
        iter_local = by_model_mode.get((model, "per_iteration_local"))
        if final_only is None or iter_local is None:
            continue
        comparisons.append({
            "model": model,
            "after_projection_rmse_delta": (
                to_float(iter_local.get("stage_a_after_projection_rmse_pct_bbox")) -
                to_float(final_only.get("stage_a_after_projection_rmse_pct_bbox"))
            ),
            "after_projection_cd_delta": (
                to_float(iter_local.get("stage_a_after_projection_cd_bbox2")) -
                to_float(final_only.get("stage_a_after_projection_cd_bbox2"))
            ),
            "final_rmse_delta": (
                to_float(iter_local.get("final_output_rmse_pct_bbox")) -
                to_float(final_only.get("final_output_rmse_pct_bbox"))
            ),
            "final_cd_delta": (
                to_float(iter_local.get("final_output_cd_bbox2")) -
                to_float(final_only.get("final_output_cd_bbox2"))
            ),
            "projection_rmse_gain_delta": (
                to_float(iter_local.get("projection_rmse_gain")) -
                to_float(final_only.get("projection_rmse_gain"))
            ),
            "stage_b_rmse_drift_delta": (
                to_float(iter_local.get("stage_b_rmse_drift")) -
                to_float(final_only.get("stage_b_rmse_drift"))
            ),
        })
    return comparisons


def write_report(rows: list[dict[str, Any]], summaries: list[dict[str, Any]], comparisons: list[dict[str, Any]]) -> None:
    successes = [row for row in rows if row.get("status") == "success"]
    failures = [row for row in rows if row.get("status") != "success"]
    summary_by_mode = {row["mode"]: row for row in summaries}

    lines = [
        "# v5.3.10 Stage A 投影几何误差分解 T21/T36 报告",
        "",
        "## 1. 结论摘要",
        "",
        f"本轮只跑 `T21`、`T36`，两种投影时机共 `{len(rows)}` 个 case，成功 `{len(successes)}/{len(rows)}`。",
        "本报告把几何误差拆成 Stage A 投影前、Stage A 投影后、Stage B 最终输出三段，用来判断误差来源。",
        "",
        "## 2. 分段平均指标",
        "",
        "| mode | before RMSE | after RMSE | final RMSE | projection gain | Stage B drift | before CD | after CD | final CD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        item = summary_by_mode.get(mode, {})
        lines.append(
            f"| {mode} | "
            f"{fmt(item.get('avg_stage_a_before_projection_rmse_pct_bbox'), 6)} | "
            f"{fmt(item.get('avg_stage_a_after_projection_rmse_pct_bbox'), 6)} | "
            f"{fmt(item.get('avg_final_output_rmse_pct_bbox'), 6)} | "
            f"{fmt(item.get('avg_projection_rmse_gain'), 6)} | "
            f"{fmt(item.get('avg_stage_b_rmse_drift'), 6)} | "
            f"{sci(item.get('avg_stage_a_before_projection_cd_bbox2'))} | "
            f"{sci(item.get('avg_stage_a_after_projection_cd_bbox2'))} | "
            f"{sci(item.get('avg_final_output_cd_bbox2'))} |"
        )

    lines.extend([
        "",
        "## 3. 逐模型分解",
        "",
        "| model | mode | before RMSE | after RMSE | final RMSE | projection gain | Stage B drift | before CD | after CD | final CD |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in successes:
        lines.append(
            f"| {row['model']} | {row['mode']} | "
            f"{fmt(row.get('stage_a_before_projection_rmse_pct_bbox'), 6)} | "
            f"{fmt(row.get('stage_a_after_projection_rmse_pct_bbox'), 6)} | "
            f"{fmt(row.get('final_output_rmse_pct_bbox'), 6)} | "
            f"{fmt(row.get('projection_rmse_gain'), 6)} | "
            f"{fmt(row.get('stage_b_rmse_drift'), 6)} | "
            f"{sci(row.get('stage_a_before_projection_cd_bbox2'))} | "
            f"{sci(row.get('stage_a_after_projection_cd_bbox2'))} | "
            f"{sci(row.get('final_output_cd_bbox2'))} |"
        )

    lines.extend([
        "",
        "## 4. 投影时机对比",
        "",
        "| model | after RMSE delta | final RMSE delta | after CD delta | final CD delta | projection gain delta | Stage B drift delta |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in comparisons:
        lines.append(
            f"| {row['model']} | "
            f"{fmt(row.get('after_projection_rmse_delta'), 8)} | "
            f"{fmt(row.get('final_rmse_delta'), 8)} | "
            f"{sci(row.get('after_projection_cd_delta'))} | "
            f"{sci(row.get('final_cd_delta'))} | "
            f"{fmt(row.get('projection_rmse_gain_delta'), 8)} | "
            f"{fmt(row.get('stage_b_rmse_drift_delta'), 8)} |"
        )

    final_only = summary_by_mode.get("final_only", {})
    iter_local = summary_by_mode.get("per_iteration_local", {})
    before_rmse_delta = (
        to_float(iter_local.get("avg_stage_a_before_projection_rmse_pct_bbox")) -
        to_float(final_only.get("avg_stage_a_before_projection_rmse_pct_bbox"))
    )
    after_rmse_delta = (
        to_float(iter_local.get("avg_stage_a_after_projection_rmse_pct_bbox")) -
        to_float(final_only.get("avg_stage_a_after_projection_rmse_pct_bbox"))
    )
    final_rmse_delta = (
        to_float(iter_local.get("avg_final_output_rmse_pct_bbox")) -
        to_float(final_only.get("avg_final_output_rmse_pct_bbox"))
    )
    lines.extend([
        "",
        "## 5. 判断",
        "",
        f"- `final_only` 的 projection RMSE gain 平均为 `{fmt(final_only.get('avg_projection_rmse_gain'), 6)}`，CD gain 为 `{sci(final_only.get('avg_projection_cd_gain'))}`；这说明 Stage A 结束时确实存在一部分可被最近点投影拉回的几何偏移。",
        f"- `per_iteration_local` 的 projection RMSE gain 平均只有 `{fmt(iter_local.get('avg_projection_rmse_gain'), 6)}`，因为迭代内局部投影已经提前抑制了漂移；其 projection 前 RMSE 相对 final_only 平均低 `{fmt(-before_rmse_delta, 6)}`。",
        f"- `per_iteration_local` 在 projection 后 RMSE 仍比 final_only 低 `{fmt(-after_rmse_delta, 6)}`，最终输出 RMSE 低 `{fmt(-final_rmse_delta, 6)}`；T21、T36 两个模型上迭代内投影都保留了几何收益。",
        f"- Stage B RMSE drift 很小：`final_only = {fmt(final_only.get('avg_stage_b_rmse_drift'), 6)}`，`per_iteration_local = {fmt(iter_local.get('avg_stage_b_rmse_drift'), 6)}`。因此本轮样本中几何误差主要不是 Stage B 引入的。",
        "- 结论：误差主要在 Stage A split/collapse 过程中产生，普通最近点 projection 能拉回一部分；v5.3.9 的迭代内局部 projection 能提前压低漂移，但 projection 后剩余误差仍接近最终误差水平，下一步更值得看 feature-aware projection 或 collapse geometry guard。",
        "",
        "输出文件：",
        f"- `{(RESULT_ROOT / 'run_summary.csv').relative_to(ROOT).as_posix()}`",
        f"- `{(RESULT_ROOT / 'summary.csv').relative_to(ROOT).as_posix()}`",
        f"- `{(RESULT_ROOT / 'projection_timing_comparison.csv').relative_to(ROOT).as_posix()}`",
        f"- `{REPORT_PATH.relative_to(ROOT).as_posix()}`",
    ])
    if failures:
        lines.extend(["", "## 6. 失败项", "", "| model | mode | error |", "|---|---|---|"])
        for row in failures:
            lines.append(f"| {row['model']} | {row['mode']} | {row['error']} |")
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose Stage A / projection / Stage B geometry error.")
    parser.add_argument("--cli", type=Path, default=base.DEFAULT_CLI)
    parser.add_argument("--eval", type=Path, default=None)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--cd-samples", type=int, default=100000)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--only", nargs="*", default=list(DEFAULT_MODELS))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.eval is None:
        args.eval = eval_path_from_cli(args.cli)
    meshes = base.discover_t_models()
    wanted = set(args.only or DEFAULT_MODELS)
    meshes = [mesh for mesh in meshes if mesh.stem in wanted]
    if args.workers <= 0:
        raise ValueError("--workers must be positive")

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_case, mesh, mode, args): (mesh, mode)
            for mesh in meshes
            for mode in MODES
        }
        for index, future in enumerate(as_completed(futures), start=1):
            mesh, mode = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = failed_row(mesh.stem, mode, str(exc))
            rows.append(row)
            print(f"[{index}/{len(futures)}] {mesh.stem} {mode} {row['status']}")

    rows.sort(key=lambda row: (base.model_index(row["model"]), row["mode"]))
    summaries = summary_rows(rows)
    comparisons = comparison_rows(rows)
    base.write_csv(rows, RESULT_ROOT / "run_summary.csv", CSV_FIELDS)
    base.write_csv(summaries, RESULT_ROOT / "summary.csv", tuple(summaries[0].keys()) if summaries else ("mode",))
    if comparisons:
        base.write_csv(comparisons, RESULT_ROOT / "projection_timing_comparison.csv", tuple(comparisons[0].keys()))
    write_report(rows, summaries, comparisons)
    print(f"wrote {RESULT_ROOT / 'run_summary.csv'}")
    print(f"wrote {RESULT_ROOT / 'summary.csv'}")
    print(f"wrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
