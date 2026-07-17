from __future__ import annotations

import argparse
import csv
import json
import math
import re
import struct
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CPP_ROOT = ROOT / "cpp_adaiso"
EXAMPLES_ROOT = CPP_ROOT / "examples"
RESULT_ROOT = CPP_ROOT / "results" / "v2.12.0_remaining_examples_iter10"
REPORT_PATH = CPP_ROOT / "修改计划与实验结果" / "v2.12.0 T系列 iter10验收实验报告.md"
RUN_SUMMARY_PATH = RESULT_ROOT / "run_summary.csv"
DEFAULT_CLI = CPP_ROOT / "build2" / "adaiso_cli.exe"

EXPECTED_MAPPING = "target_faces_inverse_sqrt_diffused_curvature_cap0995"
REPORT_HEADING = "## 补充实验：剩余 examples v2.12.0 iter10"

TARGET_MODELS = (
    "C0",
    "C1",
    "C2",
    "C3",
    "C4",
    "C6",
    "C7",
    "model_001",
    "model_002",
    "model_003",
    "model_004",
    "model_005",
    "model_006",
    "model_007",
    "model_009",
    "model_010",
    "hand",
    "bunny",
    "dragon",
    "Th1",
    "Th2",
)

RUN_FIELDS = (
    "group",
    "model",
    "input_path",
    "input_vertices",
    "input_faces",
    "output_vertices",
    "output_faces",
    "face_ratio",
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


def finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def model_group(model: str) -> str:
    if re.fullmatch(r"C\d+", model):
        return "C"
    if re.fullmatch(r"model_\d+", model):
        return "model"
    if model in {"hand", "bunny", "dragon"}:
        return "standard"
    if re.fullmatch(r"Th\d+", model):
        return "Th"
    raise ValueError(f"unexpected model name: {model}")


def model_sort_key(model: str) -> tuple[int, int | str]:
    group = model_group(model)
    group_order = {"C": 0, "model": 1, "standard": 2, "Th": 3}[group]
    if group == "C":
        return (group_order, int(model[1:]))
    if group == "model":
        return (group_order, int(model.split("_", 1)[1]))
    if group == "standard":
        return (group_order, {"hand": 0, "bunny": 1, "dragon": 2}[model])
    return (group_order, int(model[2:]))


def discover_models() -> list[Path]:
    by_stem = {
        path.stem: path
        for path in EXAMPLES_ROOT.iterdir()
        if path.is_file() and path.suffix.lower() in {".obj", ".off", ".ply"}
    }
    missing = [model for model in TARGET_MODELS if model not in by_stem]
    if missing:
        raise RuntimeError("missing expected examples: " + ", ".join(missing))
    return [by_stem[model] for model in TARGET_MODELS]


def case_dir(model: str) -> Path:
    return RESULT_ROOT / model


def tail_violation(summary: dict) -> float:
    iterations = summary.get("iterations", [])
    return float(iterations[-1].get("violation_rate", 0.0)) if iterations else 0.0


def operator_total(summary: dict, iteration_key: str, stats_key: str) -> int:
    stats = summary.get("stats", {})
    if stats_key in stats:
        return int(stats[stats_key])
    return sum(int(item.get(iteration_key, 0)) for item in summary.get("iterations", []))


def read_case(model: str) -> dict:
    summary = load_json(case_dir(model) / "summary.json")
    parameters = summary.get("parameters", {})
    stats = summary.get("stats", {})
    input_summary = summary["input_summary"]
    output_summary = summary["output_summary"]
    chamfer = summary["chamfer_distance"]

    input_faces = int(input_summary["faces"])
    output_faces = int(output_summary["faces"])
    target_faces = int(parameters.get("target_faces", input_faces))

    return {
        "group": model_group(model),
        "model": model,
        "input_path": summary.get("input", ""),
        "input_vertices": int(input_summary["vertices"]),
        "input_faces": input_faces,
        "output_vertices": int(output_summary["vertices"]),
        "output_faces": output_faces,
        "face_ratio": output_faces / input_faces if input_faces else 0.0,
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
        "ground_truth": summary.get("ground_truth", ""),
    }


def case_is_complete(model: str, expected_iters: int, expected_cd_samples: int) -> bool:
    directory = case_dir(model)
    if not (directory / "summary.json").is_file() or not (directory / "output.ply").is_file():
        return False
    try:
        row = read_case(model)
    except (KeyError, ValueError, OSError, json.JSONDecodeError, RuntimeError):
        return False
    if row["global_iterations"] != expected_iters:
        return False
    if row["cd_samples"] != expected_cd_samples:
        return False
    if row["target_size_mapping"] != EXPECTED_MAPPING:
        return False
    if not row["csf_solver_success"]:
        return False
    required = (
        "qavg",
        "qmin",
        "iter10_violation",
        "cd",
        "symmetric_rmse",
        "runtime_sec",
        "target_lambda",
        "target_estimated_faces",
        "curvature_cap_changed_rate",
    )
    return all(finite(row[key]) for key in required)


def run_case(model_path: Path, args: argparse.Namespace) -> None:
    model = model_path.stem
    directory = case_dir(model)
    directory.mkdir(parents=True, exist_ok=True)
    cli_input = prepare_cli_input(model_path, directory)
    command = [
        str(args.cli),
        rel_arg(cli_input),
        "-o",
        rel_arg(directory / "output.ply"),
        "--preprocess",
        "0",
        "--iters",
        str(args.iters),
        "--gt",
        rel_arg(cli_input),
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
        check=False,
    )
    elapsed = time.perf_counter() - started
    (directory / "stdout.log").write_text(proc.stdout, encoding="utf-8")
    (directory / "stderr.log").write_text(proc.stderr, encoding="utf-8")
    print(f"[{model}] exit={proc.returncode} elapsed={elapsed:.1f}s", flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{model} failed; see {rel(directory / 'stderr.log')}")


def prepare_cli_input(model_path: Path, directory: Path) -> Path:
    if model_path.suffix.lower() != ".ply":
        return model_path
    header = read_ply_header(model_path)
    if header["format"] == "ascii":
        return model_path
    if header["format"] != "binary_little_endian":
        raise RuntimeError(f"unsupported PLY format for conversion: {header['format']}")
    converted = directory / "input_ascii.ply"
    if not converted.exists() or converted.stat().st_mtime < model_path.stat().st_mtime:
        convert_binary_little_ply_to_ascii(model_path, converted, header)
    return converted


def read_ply_header(path: Path) -> dict:
    with path.open("rb") as handle:
        first = handle.readline().decode("ascii", errors="replace").strip()
        if first != "ply":
            raise RuntimeError(f"not a PLY file: {path}")
        result = {
            "format": "",
            "vertex_count": 0,
            "face_count": 0,
            "header_end": 0,
        }
        while True:
            line_bytes = handle.readline()
            if not line_bytes:
                raise RuntimeError(f"unexpected EOF in PLY header: {path}")
            line = line_bytes.decode("ascii", errors="replace").strip()
            words = line.split()
            if len(words) >= 3 and words[0] == "format":
                result["format"] = words[1]
            elif len(words) >= 3 and words[0] == "element":
                if words[1] == "vertex":
                    result["vertex_count"] = int(words[2])
                elif words[1] == "face":
                    result["face_count"] = int(words[2])
            elif line == "end_header":
                result["header_end"] = handle.tell()
                break
    if not result["format"] or result["vertex_count"] <= 0 or result["face_count"] <= 0:
        raise RuntimeError(f"unsupported or incomplete PLY header: {path}")
    return result


def convert_binary_little_ply_to_ascii(path: Path, output_path: Path, header: dict) -> None:
    vertex_count = int(header["vertex_count"])
    face_count = int(header["face_count"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("rb") as source, output_path.open("w", encoding="utf-8", newline="\n") as out:
        source.seek(int(header["header_end"]))
        out.write("ply\n")
        out.write("format ascii 1.0\n")
        out.write("comment converted from binary_little_endian by generate_v2_12_0_remaining_iter10_acceptance.py\n")
        out.write(f"element vertex {vertex_count}\n")
        out.write("property float x\n")
        out.write("property float y\n")
        out.write("property float z\n")
        out.write(f"element face {face_count}\n")
        out.write("property list uchar int vertex_indices\n")
        out.write("end_header\n")

        for _ in range(vertex_count):
            data = source.read(12)
            if len(data) != 12:
                raise RuntimeError(f"unexpected EOF in binary PLY vertices: {path}")
            x, y, z = struct.unpack("<fff", data)
            out.write(f"{x:.9g} {y:.9g} {z:.9g}\n")

        for _ in range(face_count):
            count_data = source.read(1)
            if len(count_data) != 1:
                raise RuntimeError(f"unexpected EOF in binary PLY faces: {path}")
            count = struct.unpack("<B", count_data)[0]
            data = source.read(4 * count)
            if len(data) != 4 * count:
                raise RuntimeError(f"unexpected EOF in binary PLY face indices: {path}")
            indices = struct.unpack("<" + "i" * count, data)
            out.write(str(count))
            for index in indices:
                out.write(f" {index}")
            out.write("\n")


def collect_rows(models: list[str], expected_iters: int, expected_cd_samples: int) -> tuple[list[dict], dict[str, str]]:
    rows: list[dict] = []
    errors: dict[str, str] = {}
    for model in models:
        if not case_is_complete(model, expected_iters, expected_cd_samples):
            errors[model] = f"incomplete or parameter mismatch; see {rel(case_dir(model))}/"
            continue
        try:
            rows.append(read_case(model))
        except Exception as error:  # noqa: BLE001
            errors[model] = str(error)
    rows.sort(key=lambda row: model_sort_key(row["model"]))
    return rows, errors


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def average(rows: list[dict], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows) if rows else math.nan


def count_where(rows: list[dict], predicate) -> int:
    return sum(1 for row in rows if predicate(row))


def group_summary_table(rows: list[dict]) -> str:
    lines = [
        "| Group | Count | Avg CD | Max CD | Avg RMSE | Avg Qavg | Avg Qmin | Min Qmin | Avg viol | Avg F ratio | Avg runtime |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in ("C", "model", "standard", "Th", "all"):
        selected = rows if group == "all" else [row for row in rows if row["group"] == group]
        if not selected:
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    group,
                    str(len(selected)),
                    sci(average(selected, "cd")),
                    sci(max(row["cd"] for row in selected)),
                    sci(average(selected, "symmetric_rmse")),
                    fmt(average(selected, "qavg")),
                    fmt(average(selected, "qmin")),
                    fmt(min(row["qmin"] for row in selected)),
                    fmt(average(selected, "iter10_violation")),
                    fmt(average(selected, "face_ratio")),
                    f"{average(selected, 'runtime_sec'):.2f}s",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def result_table(rows: list[dict]) -> str:
    lines = [
        "| Group | Model | In F | Out F | F ratio | Qavg | Qmin | viol | CD | RMSE | cap changed | split/collapse/flip | runtime |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['group']} | {row['model']} | {row['input_faces']} | {row['output_faces']} | "
            f"{fmt(row['face_ratio'])} | {fmt(row['qavg'])} | {fmt(row['qmin'])} | "
            f"{fmt(row['iter10_violation'])} | {sci(row['cd'])} | {sci(row['symmetric_rmse'])} | "
            f"{100.0 * row['curvature_cap_changed_rate']:.3f}% | "
            f"{row['split_total']}/{row['collapse_total']}/{row['flip_total']} | "
            f"{row['runtime_sec']:.2f}s |"
        )
    return "\n".join(lines)


def top_models(rows: list[dict], key: str, count: int, reverse: bool = True) -> str:
    selected = sorted(rows, key=lambda row: row[key], reverse=reverse)[:count]
    return ", ".join(f"`{row['model']}` ({row[key]:.6g})" for row in selected)


def build_report_section(
    models: list[str],
    rows: list[dict],
    errors: dict[str, str],
    args: argparse.Namespace,
) -> str:
    complete_count = len(rows)
    target_match_count = count_where(rows, lambda row: row["target_faces_matches_input"])
    solver_count = count_where(rows, lambda row: row["csf_solver_success"])
    mapping_count = count_where(rows, lambda row: row["target_size_mapping"] == EXPECTED_MAPPING)

    lines = [
        REPORT_HEADING,
        "",
        "### 实验设置",
        "",
        "本补充实验使用当前 v2.12.0 管线，对 `examples` 中剩余 C 系列、model 系列、标准模型和 Th 模型运行 10 轮重网格化。每个模型使用自身输入作为 GT，计算 area-sampled Chamfer Distance 和 symmetric RMSE；本节只报告 v2.12.0 绝对指标，不做旧版本对比。",
        "",
        "| 参数 | 值 |",
        "|---|---:|",
        f"| 目标模型数 | {len(models)} |",
        f"| 完成模型数 | {complete_count} |",
        f"| iters | {args.iters} |",
        "| preprocess | 0 |",
        f"| CD samples | {args.cd_samples} |",
        f"| target_size_mapping | `{EXPECTED_MAPPING}` |",
        "",
        f"- 输出目录：`{rel(RESULT_ROOT)}/`",
        f"- run summary：`{rel(RUN_SUMMARY_PATH)}`",
        "",
        "### 完整性检查",
        "",
        "| 检查项 | 结果 |",
        "|---|---:|",
        f"| 完成模型 | {complete_count} / {len(models)} |",
        f"| target_size_mapping 匹配 | {mapping_count} / {len(models)} |",
        f"| CSF solver success | {solver_count} / {len(models)} |",
        f"| target_faces 匹配输入面数 | {target_match_count} / {len(models)} |",
        f"| 平均 curvature cap changed rate | {100.0 * average(rows, 'curvature_cap_changed_rate'):.4f}% |",
        "",
    ]
    if errors:
        lines.extend(["未完成模型：", "", "| Model | 原因 |", "|---|---|"])
        for model in models:
            if model in errors:
                lines.append(f"| {model} | {errors[model]} |")
        lines.append("")

    lines.extend(["### 分组汇总", "", group_summary_table(rows) if rows else "_无完整结果。_", ""])
    if rows:
        low_qmin = top_models(rows, "qmin", 5, reverse=False)
        high_violation = top_models(rows, "iter10_violation", 5, reverse=True)
        high_cd = top_models(rows, "cd", 5, reverse=True)
        high_runtime = top_models(rows, "runtime_sec", 5, reverse=True)
        lines.extend(
            [
                "### 重点模型",
                "",
                f"- 最低 Qmin：{low_qmin}",
                f"- 最大 violation：{high_violation}",
                f"- 最大 CD：{high_cd}",
                f"- 最长 runtime：{high_runtime}",
                "",
                "### 单模型结果",
                "",
                result_table(rows),
                "",
                "### 结论",
                "",
                f"本次补充实验完成 `{complete_count}/{len(models)}` 个模型。全部完成模型平均 CD 为 `{sci(average(rows, 'cd'))}`，平均 Qavg 为 `{fmt(average(rows, 'qavg'))}`，平均 Qmin 为 `{fmt(average(rows, 'qmin'))}`，平均 violation 为 `{fmt(average(rows, 'iter10_violation'))}`。",
                f"最低 Qmin 出现在 {top_models(rows, 'qmin', 1, reverse=False)}，最高 violation 出现在 {top_models(rows, 'iter10_violation', 1, reverse=True)}。",
                "",
            ]
        )
    else:
        lines.extend(["### 结论", "", "本次补充实验尚无完整结果。", ""])
    return "\n".join(lines).rstrip() + "\n"


def replace_report_section(section: str) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if REPORT_PATH.exists():
        original = REPORT_PATH.read_text(encoding="utf-8")
    else:
        original = "# v2.12.0 T系列 iter10验收实验报告\n"

    start = original.find(REPORT_HEADING)
    if start == -1:
        text = original.rstrip() + "\n\n" + section
    else:
        text = original[:start].rstrip() + "\n\n" + section
    REPORT_PATH.write_text(text.rstrip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v2.12.0 remaining examples iter10 experiments.")
    parser.add_argument("--cli", type=Path, default=DEFAULT_CLI)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--cd-samples", type=int, default=100000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument("--models", default="", help="Comma-separated subset of target model names.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.cli = args.cli.resolve()
    if args.iters < 0:
        raise RuntimeError("--iters must be non-negative")
    if args.cd_samples <= 0:
        raise RuntimeError("--cd-samples must be positive")
    if not args.skip_run and not args.cli.is_file():
        raise RuntimeError(f"CLI executable not found: {args.cli}")

    model_paths = discover_models()
    if args.models:
        requested = [item.strip() for item in args.models.split(",") if item.strip()]
        known = {path.stem: path for path in model_paths}
        unknown = [model for model in requested if model not in known]
        if unknown:
            raise RuntimeError("unknown requested models: " + ", ".join(unknown))
        model_paths = [known[model] for model in requested]

    models = [path.stem for path in model_paths]
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)

    failures: dict[str, str] = {}
    if not args.skip_run:
        for model_path in model_paths:
            model = model_path.stem
            if not args.force and case_is_complete(model, args.iters, args.cd_samples):
                print(f"[{model}] skip complete", flush=True)
                continue
            try:
                run_case(model_path, args)
            except Exception as error:  # noqa: BLE001
                failures[model] = str(error)
                print(f"[{model}] failed: {error}", flush=True)

    rows, errors = collect_rows(models, args.iters, args.cd_samples)
    errors.update(failures)
    write_csv(RUN_SUMMARY_PATH, RUN_FIELDS, rows)
    replace_report_section(build_report_section(models, rows, errors, args))

    print(f"wrote {rel(RUN_SUMMARY_PATH)}", flush=True)
    print(f"updated {rel(REPORT_PATH)}", flush=True)
    if errors:
        print("incomplete models: " + ", ".join(sorted(errors, key=model_sort_key)), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
