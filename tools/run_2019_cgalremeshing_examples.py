from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT / "cpp_adaiso" / "examples"
DEFAULT_EXE = ROOT / "compare" / "2019_CGALRemeshing" / "noqt" / "build-vs2022-vcpkg" / "Release" / "minangle_remesh_cli.exe"
DEFAULT_METRICS_SCRIPT = ROOT / "tools" / "mesh_metrics.py"
DEFAULT_OUT_DIR = ROOT / "compare" / "2019_CGALRemeshing" / "results" / "default_examples_preprocessed"

from mesh_metrics import load_mesh

ALGORITHM_DEFAULTS = {
    "delta_percent": 0.2,
    "theta_degrees": 30.0,
    "max_vertices": 100000000,
    "samples_in": 10,
    "samples_out": 10,
    "max_samples_per_area": 10000,
    "min_samples_per_triangle": 1,
    "bvd_iterations": 1,
    "sample_number": "fixed",
    "sample_strategy": "adaptive",
    "initial_simplification": True,
    "final_relocation": True,
    "edge_flip": True,
    "decrease_max_errors": True,
}


def mesh_files(input_dir: Path) -> list[Path]:
    suffixes = {".obj", ".off", ".ply"}
    return sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in suffixes)


def select_inputs(inputs: list[Path], args: argparse.Namespace) -> list[Path]:
    selected = inputs
    if args.only:
        wanted = {item.lower() for item in args.only}
        selected = [path for path in selected if path.name.lower() in wanted or path.stem.lower() in wanted]
    if args.limit is not None:
        selected = selected[:args.limit]
    return selected


def duplicate_stems(inputs: list[Path]) -> set[str]:
    counts: dict[str, int] = {}
    for path in inputs:
        counts[path.stem] = counts.get(path.stem, 0) + 1
    return {stem for stem, count in counts.items() if count > 1}


def case_id(mesh: Path, duplicate_stem_set: set[str]) -> str:
    if mesh.stem not in duplicate_stem_set:
        return mesh.stem
    return f"{mesh.stem}_{mesh.suffix.lower().lstrip('.')}"


def run_command(command: list[str], log_path: Path, timeout_seconds: int | None) -> tuple[int, float]:
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write("$ " + " ".join(command) + "\n\n")
        log.flush()
        try:
            completed = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
            return completed.returncode, time.perf_counter() - started
        except subprocess.TimeoutExpired:
            log.write(f"\nTIMEOUT after {timeout_seconds} seconds\n")
            return 124, time.perf_counter() - started


def write_off(path: Path, vertices: Any, faces: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("OFF\n")
        handle.write(f"{len(vertices)} {len(faces)} 0\n")
        for vertex in vertices:
            handle.write(f"{vertex[0]:.17g} {vertex[1]:.17g} {vertex[2]:.17g}\n")
        for face in faces:
            handle.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")


def preprocess_mesh(mesh: Path, output_path: Path, log_path: Path) -> tuple[bool, str | None]:
    started = time.perf_counter()
    try:
        vertices, faces = load_mesh(mesh)
        write_off(output_path, vertices, faces)
        log_path.write_text(
            "\n".join([
                f"input: {mesh}",
                f"output: {output_path}",
                f"vertices: {len(vertices)}",
                f"faces: {len(faces)}",
                f"runtime_seconds: {time.perf_counter() - started:.6f}",
                "",
            ]),
            encoding="utf-8",
        )
        return True, None
    except Exception as exc:
        log_path.write_text(f"input: {mesh}\nerror: {exc}\n", encoding="utf-8")
        return False, str(exc)


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def case_skippable(status_path: Path, rerun_failures: bool) -> bool:
    status = read_json(status_path)
    if not isinstance(status, dict):
        return False
    if status.get("status") == "success":
        return True
    return not rerun_failures


def run_case(mesh: Path, args: argparse.Namespace, dirs: dict[str, Path]) -> dict[str, Any]:
    cid = case_id(mesh, args.duplicate_stems)
    preprocessed_mesh = dirs["preprocessed"] / f"{cid}.off"
    preprocess_log = dirs["logs"] / f"{cid}.preprocess.log"
    output_mesh = dirs["meshes"] / f"{cid}.off"
    remesh_log = dirs["logs"] / f"{cid}.remesh.log"
    metrics_log = dirs["logs"] / f"{cid}.metrics.log"
    metrics_json = dirs["metrics"] / f"{cid}.metrics.json"
    status_json = dirs["status"] / f"{cid}.status.json"

    if args.resume and case_skippable(status_json, args.rerun_failures):
        status = read_json(status_json)
        assert status is not None
        return status

    status: dict[str, Any] = {
        "model": mesh.name,
        "input": str(mesh),
        "preprocessed_input": str(preprocessed_mesh),
        "output": str(output_mesh),
        "preprocess_log": str(preprocess_log),
        "metrics_json": str(metrics_json),
        "remesh_log": str(remesh_log),
        "metrics_log": str(metrics_log),
        "status": "pending",
        "remesh_returncode": None,
        "metrics_returncode": None,
        "runtime_seconds": None,
        "metrics_runtime_seconds": None,
        "error": None,
    }

    remesher_input = mesh
    metrics_reference = mesh
    if args.preprocess_inputs:
        ok, error = preprocess_mesh(mesh, preprocessed_mesh, preprocess_log)
        if not ok:
            status["status"] = "preprocess_failed"
            status["error"] = error
            status_json.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            return status
        remesher_input = preprocessed_mesh
        metrics_reference = preprocessed_mesh

    command = [
        str(args.exe),
        "--input",
        str(remesher_input),
        "--output",
        str(output_mesh),
        "--quiet",
    ]
    if args.no_properties:
        command.append("--no-properties")

    returncode, runtime = run_command(command, remesh_log, args.timeout_seconds)
    status["remesh_returncode"] = returncode
    status["runtime_seconds"] = runtime
    if returncode != 0 or not output_mesh.is_file():
        status["status"] = "remesh_timeout" if returncode == 124 else "remesh_failed"
        status["error"] = "remesher timed out" if returncode == 124 else f"remesher returned {returncode}"
        status_json.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return status

    metrics_command = [
        sys.executable,
        str(args.metrics_script),
        "--mesh",
        str(output_mesh),
        "--gt",
        str(metrics_reference),
        "--cd-samples",
        str(args.cd_samples),
        "--json",
        str(metrics_json),
    ]
    metrics_returncode, metrics_runtime = run_command(metrics_command, metrics_log, args.metrics_timeout_seconds)
    status["metrics_returncode"] = metrics_returncode
    status["metrics_runtime_seconds"] = metrics_runtime
    if metrics_returncode != 0 or not metrics_json.is_file():
        status["status"] = "metrics_failed"
        status["error"] = f"metrics returned {metrics_returncode}"
    else:
        status["status"] = "success"
    status_json.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return status


def flatten_case(status: dict[str, Any]) -> dict[str, Any]:
    metrics = read_json(Path(status["metrics_json"])) if status.get("metrics_json") else None
    row: dict[str, Any] = {
        "model": status.get("model"),
        "status": status.get("status"),
        "runtime_seconds": status.get("runtime_seconds"),
        "metrics_runtime_seconds": status.get("metrics_runtime_seconds"),
        "remesh_returncode": status.get("remesh_returncode"),
        "metrics_returncode": status.get("metrics_returncode"),
        "input": status.get("input"),
        "preprocessed_input": status.get("preprocessed_input"),
        "output": status.get("output"),
        "error": status.get("error"),
    }
    if not isinstance(metrics, dict):
        return row

    mesh = metrics.get("mesh_summary", {})
    gt = metrics.get("gt_summary", {})
    angles = mesh.get("angles", {})
    edges = mesh.get("edge_lengths", {})
    regular = mesh.get("regular_vertices", {})
    chamfer = metrics.get("chamfer_distance", {})
    paper = metrics.get("paper_like", {})

    row.update({
        "input_vertices": gt.get("vertices"),
        "input_faces": gt.get("faces"),
        "output_vertices": mesh.get("vertices"),
        "output_faces": mesh.get("faces"),
        "output_edges": mesh.get("edges"),
        "components": mesh.get("components"),
        "boundary_edges": mesh.get("boundary_edges"),
        "boundary_loops": mesh.get("boundary_loops"),
        "qavg": mesh.get("qavg"),
        "qmin": mesh.get("qmin"),
        "minimal_angle_deg": angles.get("minimal_angle_deg"),
        "average_minimal_angle_deg": angles.get("average_minimal_angle_deg"),
        "maximal_angle_deg": angles.get("maximal_angle_deg"),
        "angles_smaller_than_30_degree_pct": paper.get("angles_smaller_than_30_degree_pct"),
        "regular_vertices_ratio_pct": paper.get("regular_vertices_ratio_pct"),
        "mean_edge_length": edges.get("mean"),
        "edge_length_p05": edges.get("p05"),
        "edge_length_p50": edges.get("p50"),
        "edge_length_p95": edges.get("p95"),
        "sampled_hausdorff_distance": paper.get("maximal_error"),
        "sampled_hausdorff_ratio_bbox_pct": paper.get("maximal_error_ratio_pct"),
        "chamfer_distance": chamfer.get("chamfer_distance"),
        "symmetric_rmse": chamfer.get("symmetric_rmse"),
        "symmetric_rmse_ratio_bbox_pct": chamfer.get("symmetric_rmse_ratio_bbox_pct"),
        "rms_face_error_proxy": paper.get("rms_face_error_proxy"),
        "rms_face_error_ratio_pct": paper.get("rms_face_error_ratio_pct"),
    })
    return row


def mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if finite(row.get(key))]
    return float(sum(values) / len(values)) if values else math.nan


def minimum(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if finite(row.get(key))]
    return float(min(values)) if values else math.nan


def maximum(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if finite(row.get(key))]
    return float(max(values)) if values else math.nan


def format_num(value: Any, digits: int = 4) -> str:
    if not finite(value):
        return "-"
    return f"{float(value):.{digits}f}"


def write_csv(rows: list[dict[str, Any]], csv_path: Path) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_report(rows: list[dict[str, Any]], args: argparse.Namespace, report_path: Path) -> None:
    successes = [row for row in rows if row.get("status") == "success"]
    failures = [row for row in rows if row.get("status") != "success"]
    lines: list[str] = []
    lines.append("# 2019 CGALRemeshing 论文版本默认参数实验报告")
    lines.append("")
    lines.append(f"- 实验时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- 输入目录：`{args.input_dir}`")
    lines.append(f"- 输出目录：`{args.out_dir}`")
    lines.append(f"- 输入预处理：{'开启（三角化/清理/转 OFF 后作为论文版输入）' if args.preprocess_inputs else '关闭'}")
    lines.append(f"- 论文版程序：`{args.exe}`")
    lines.append(f"- 指标脚本：`{args.metrics_script}`")
    lines.append(f"- 模型数量：{len(rows)}，成功：{len(successes)}，失败：{len(failures)}")
    lines.append(f"- CD/RMSE 采样数：每个方向 {args.cd_samples}")
    lines.append(f"- Remeshing 超时保护：{args.timeout_seconds if args.timeout_seconds is not None else '无'} 秒/模型")
    lines.append(f"- 并发数：{args.workers}")
    lines.append("")
    lines.append("## 默认参数")
    lines.append("")
    lines.append("| 参数 | 值 |")
    lines.append("|---|---:|")
    for key, value in ALGORITHM_DEFAULTS.items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.append("")
    lines.append("## 汇总结果")
    lines.append("")
    lines.append("| 指标 | 平均 | 最小 | 最大 |")
    lines.append("|---|---:|---:|---:|")
    summary_keys = [
        ("运行时间 s", "runtime_seconds", 2),
        ("输出顶点数", "output_vertices", 1),
        ("输出面数", "output_faces", 1),
        ("qavg", "qavg", 6),
        ("qmin", "qmin", 6),
        ("最小角 deg", "minimal_angle_deg", 4),
        ("平均最小角 deg", "average_minimal_angle_deg", 4),
        ("<30deg 面比例 %", "angles_smaller_than_30_degree_pct", 4),
        ("规则顶点比例 %", "regular_vertices_ratio_pct", 4),
        ("采样 Hausdorff %bbox", "sampled_hausdorff_ratio_bbox_pct", 6),
        ("RMS 误差 %bbox", "rms_face_error_ratio_pct", 6),
        ("Chamfer distance", "chamfer_distance", 8),
    ]
    for label, key, digits in summary_keys:
        lines.append(
            f"| {label} | {format_num(mean(successes, key), digits)} | "
            f"{format_num(minimum(successes, key), digits)} | {format_num(maximum(successes, key), digits)} |"
        )
    lines.append("")
    lines.append("## 单模型结果")
    lines.append("")
    lines.append("| 模型 | 状态 | 时间(s) | V_in | F_in | V_out | F_out | qavg | qmin | min angle | <30deg% | regular% | H %bbox | RMS %bbox |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| {row.get('model')} | {row.get('status')} | {format_num(row.get('runtime_seconds'), 2)} | "
            f"{row.get('input_vertices', '-')} | {row.get('input_faces', '-')} | "
            f"{row.get('output_vertices', '-')} | {row.get('output_faces', '-')} | "
            f"{format_num(row.get('qavg'), 6)} | {format_num(row.get('qmin'), 6)} | "
            f"{format_num(row.get('minimal_angle_deg'), 4)} | "
            f"{format_num(row.get('angles_smaller_than_30_degree_pct'), 4)} | "
            f"{format_num(row.get('regular_vertices_ratio_pct'), 4)} | "
            f"{format_num(row.get('sampled_hausdorff_ratio_bbox_pct'), 6)} | "
            f"{format_num(row.get('rms_face_error_ratio_pct'), 6)} |"
        )
    if failures:
        lines.append("")
        lines.append("## 失败项")
        lines.append("")
        lines.append("| 模型 | 阶段 | 错误 | 日志 |")
        lines.append("|---|---|---|---|")
        for row in failures:
            model_path = Path(str(row.get("input"))) if row.get("input") else Path(str(row.get("model")))
            log_path = Path(args.out_dir) / "logs" / f"{case_id(model_path, args.duplicate_stems)}.remesh.log"
            lines.append(f"| {row.get('model')} | {row.get('status')} | {row.get('error')} | `{log_path}` |")
    lines.append("")
    lines.append("## 指标说明")
    lines.append("")
    lines.append("- `qavg/qmin` 使用 `eval.cpp` 相同三角形质量公式：`6A/(sqrt(3) * s * lmax)`，其中 `s` 是半周长。")
    lines.append("- `Chamfer distance/RMSE` 使用与 `eval.cpp` 一致的面积采样点集近邻公式，默认每方向 100000 个样本。")
    lines.append("- `H %bbox` 是同一采样点集上的双向最大近邻距离除以输入 bbox 对角线，属于采样 Hausdorff 代理指标。")
    lines.append("- `RMS %bbox` 是对称 RMSE 除以输入 bbox 对角线；可作为论文 RMS face error 的一致采样代理。")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the no-Qt 2019 CGALRemeshing paper baseline on cpp_adaiso examples.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--exe", type=Path, default=DEFAULT_EXE)
    parser.add_argument("--metrics-script", type=Path, default=DEFAULT_METRICS_SCRIPT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--cd-samples", type=int, default=100000)
    parser.add_argument("--timeout-seconds", type=int, default=None, help="Optional timeout per remeshing case.")
    parser.add_argument("--metrics-timeout-seconds", type=int, default=None, help="Optional timeout per metrics case.")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rerun-failures", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--preprocess-inputs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--only", nargs="*", help="Optional model names/stems to run.")
    parser.add_argument("--limit", type=int, help="Optional limit after sorting/filtering.")
    parser.add_argument("--workers", type=int, default=1, help="Number of models to run concurrently.")
    parser.add_argument("--no-properties", action="store_true", help="Suppress property blocks in remesher logs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.exe.is_file():
        raise FileNotFoundError(f"remesher executable not found: {args.exe}")
    if not args.metrics_script.is_file():
        raise FileNotFoundError(f"metrics script not found: {args.metrics_script}")
    if args.cd_samples <= 0:
        raise ValueError("--cd-samples must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")

    dirs = {
        "preprocessed": args.out_dir / "preprocessed",
        "meshes": args.out_dir / "meshes",
        "logs": args.out_dir / "logs",
        "metrics": args.out_dir / "metrics",
        "status": args.out_dir / "status",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    inputs = select_inputs(mesh_files(args.input_dir), args)
    args.duplicate_stems = duplicate_stems(inputs)
    rows_by_model: dict[str, dict[str, Any]] = {}

    def record_result(mesh: Path, status: dict[str, Any], done_index: int) -> None:
        row = flatten_case(status)
        rows_by_model[mesh.name] = row
        rows = [rows_by_model[path.name] for path in inputs if path.name in rows_by_model]
        print(
            f"[{done_index}/{len(inputs)}] {mesh.name} -> {row.get('status')} "
            f"runtime={format_num(row.get('runtime_seconds'), 2)}s",
            flush=True,
        )
        write_csv(rows, args.out_dir / "summary.csv")
        write_report(rows, args, args.out_dir / "experiment_report.md")

    if args.workers == 1:
        for index, mesh in enumerate(inputs, 1):
            print(f"[{index}/{len(inputs)}] {mesh.name}", flush=True)
            record_result(mesh, run_case(mesh, args, dirs), index)
    else:
        print(f"Running {len(inputs)} models with {args.workers} workers", flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_case, mesh, args, dirs): mesh for mesh in inputs}
            for done_index, future in enumerate(as_completed(futures), 1):
                mesh = futures[future]
                try:
                    status = future.result()
                except Exception as exc:
                    status = {
                        "model": mesh.name,
                        "input": str(mesh),
                        "output": None,
                        "metrics_json": None,
                        "status": "runner_failed",
                        "runtime_seconds": None,
                        "metrics_runtime_seconds": None,
                        "remesh_returncode": None,
                        "metrics_returncode": None,
                        "error": str(exc),
                    }
                record_result(mesh, status, done_index)

    rows = [rows_by_model[path.name] for path in inputs if path.name in rows_by_model]

    (args.out_dir / "run_config.json").write_text(
        json.dumps(
            {
                "input_dir": str(args.input_dir),
                "exe": str(args.exe),
                "metrics_script": str(args.metrics_script),
                "out_dir": str(args.out_dir),
                "cd_samples": args.cd_samples,
                "timeout_seconds": args.timeout_seconds,
                "metrics_timeout_seconds": args.metrics_timeout_seconds,
                "workers": args.workers,
                "preprocess_inputs": args.preprocess_inputs,
                "algorithm_defaults": ALGORITHM_DEFAULTS,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0 if len(rows) == len(inputs) and all(row.get("status") == "success" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
