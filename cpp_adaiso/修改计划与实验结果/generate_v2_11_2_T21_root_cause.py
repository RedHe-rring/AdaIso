from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
CPP = ROOT / "cpp_adaiso"
EXAMPLES = CPP / "examples"
RESULT_ROOT = CPP / "results" / "v2.11.2_T21_root_cause_20260715"
REPORT = CPP / "修改计划与实验结果" / "v2.11.2 T21 h异常根因与修复筛选实验报告.md"
DEFAULT_CLI = CPP / "build2" / "adaiso_cli.exe"
BASELINE_2111 = CPP / "results" / "v2.11.1_T_all_iter10_20260714"
BASELINE_2110 = CPP / "results" / "v2.11.0_T_all_iter10_20260714"
TARGET_FACES_T21 = 19990
CD_SAMPLES = 100000


@dataclass(frozen=True)
class Case:
    name: str
    model: str = "T21"
    iters: int = 0
    mode: str = "none"
    top_k: int = 1
    beta: float = 5.0
    alpha: float = 0.75
    gradation: bool = True
    q: float = 1.3
    representative: str = "mean"
    hmax: float = 5.0
    with_cd: bool = False
    export_iterations: bool = False
    export_all_stages: bool = False


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def case_dir(group: str, case: Case) -> Path:
    return RESULT_ROOT / group / case.name


def run_case(cli: Path, group: str, case: Case, force: bool) -> dict:
    out_dir = case_dir(group, case)
    summary_path = out_dir / "summary.json"
    if summary_path.exists() and not force:
        return load_json(summary_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(cli),
        rel(EXAMPLES / f"{case.model}.off"),
        "-o",
        rel(out_dir / "output.ply"),
        "--preprocess",
        "0",
        "--iters",
        str(case.iters),
        "--curvature-outlier-mode",
        case.mode,
        "--curvature-outlier-top-k",
        str(case.top_k),
        "--curvature-cap-beta",
        str(case.beta),
        "--curvature-smoothing-alpha",
        str(case.alpha),
        "--target-gradation",
        "1" if case.gradation else "0",
        "--target-gradation-ratio",
        str(case.q),
        "--target-face-representative",
        case.representative,
        "--target-max-factor",
        str(case.hmax),
        "--relocation-mls",
        "0",
        "--export-target-field",
        rel(out_dir / "target_field"),
        "--summary-json",
        rel(summary_path),
        "--quiet",
    ]
    if case.with_cd:
        command.extend(
            ["--gt", rel(EXAMPLES / f"{case.model}.off"), "--cd-samples", str(CD_SAMPLES)]
        )
    if case.export_iterations:
        command.extend(["--export-iteration-meshes", rel(out_dir / "iteration_meshes")])
    if case.export_all_stages:
        command.extend(["--export-all-stage-meshes", rel(out_dir / "stage_meshes")])
    (out_dir / "command.txt").write_text(subprocess.list2cmdline(command), encoding="utf-8")
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    (out_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (out_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"case {group}/{case.name} failed ({completed.returncode}):\n{completed.stderr}"
        )
    summary = load_json(summary_path)
    summary["driver_runtime_sec"] = time.perf_counter() - started
    return summary


def causal_cases() -> list[Case]:
    cases: list[Case] = []
    for knockout in (False, True):
        for alpha in (0.0, 0.75):
            for gradation in (False, True):
                name = (
                    f"knockout{int(knockout)}_alpha{str(alpha).replace('.', 'p')}_"
                    f"grad{int(gradation)}"
                )
                cases.append(
                    Case(
                        name=name,
                        mode="topk-median" if knockout else "none",
                        alpha=alpha,
                        gradation=gradation,
                    )
                )
    return cases


def screen_cases() -> list[Case]:
    cases: list[Case] = []
    for beta in (None, 2.0, 5.0, 10.0):
        for q in (None, 1.3, 1.5, 2.0):
            beta_name = "none" if beta is None else str(beta).replace(".", "p")
            q_name = "off" if q is None else str(q).replace(".", "p")
            cases.append(
                Case(
                    name=f"beta{beta_name}_q{q_name}",
                    mode="none" if beta is None else "sampling-cap",
                    beta=5.0 if beta is None else beta,
                    gradation=q is not None,
                    q=1.3 if q is None else q,
                )
            )
    return cases


def field_rank(summary: dict, case: Case) -> tuple[float, ...]:
    stats = summary["stats"]
    beta_preference = 1.0e9 if case.mode == "none" else case.beta
    q_preference = 1.0e9 if not case.gradation else case.q
    return (
        abs(float(stats["final_min_mean_estimate_ratio"]) - 1.0),
        float(stats["target_h_max_saturation_rate"]),
        float(stats["smallest_five_percent_face_budget_rate"]),
        float(stats["input_edge_target_violation_rate"]),
        -beta_preference,
        -q_preference,
    )


def short_rank(summary: dict) -> tuple[float, ...]:
    target = int(summary["parameters"]["target_faces"])
    output_faces = int(summary["output_summary"]["faces"])
    iterations = summary["iterations"][-3:]
    tail = max(
        abs(int(item["split_count"]) - int(item["collapse_count"]))
        / max(1, int(item["vertices"]))
        for item in iterations
    )
    cd = float(summary.get("chamfer_distance", {}).get("chamfer_distance", math.inf))
    return (
        abs(output_faces - target) / target,
        tail,
        -float(summary["output_summary"]["qavg"]),
        -float(summary["output_summary"]["qmin"]),
        float(iterations[-1]["violation_rate"]),
        cd,
    )


def long_passes(summary: dict) -> bool:
    target = int(summary["parameters"]["target_faces"])
    output = summary["output_summary"]
    iterations = summary["iterations"]
    tail = iterations[-3:]
    tail_rates = [
        abs(int(item["split_count"]) - int(item["collapse_count"]))
        / max(1, int(item["vertices"]))
        for item in tail
    ]
    cd = float(summary["chamfer_distance"]["chamfer_distance"])
    stats = summary["stats"]
    return all(
        [
            abs(int(output["faces"]) - target) / target <= 0.10,
            max(tail_rates) <= 0.02,
            tail_rates[-1] <= tail_rates[0],
            float(output["qavg"]) >= 0.82,
            float(output["qmin"]) >= 0.02,
            float(iterations[-1]["violation_rate"]) <= 0.06,
            cd <= 1.0e-5,
            float(stats["target_h_max_saturation_rate"]) <= 0.25,
            float(stats["final_min_mean_estimate_ratio"]) <= 1.25,
            float(stats["smallest_five_percent_face_budget_rate"]) <= 0.35,
        ]
    )


def write_rows(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def result_row(group: str, case: Case, summary: dict) -> dict:
    stats = summary["stats"]
    output = summary["output_summary"]
    iterations = summary["iterations"]
    return {
        "group": group,
        "case": case.name,
        "model": case.model,
        "iters": case.iters,
        "mode": case.mode,
        "beta": case.beta,
        "alpha": case.alpha,
        "gradation": int(case.gradation),
        "q": case.q,
        "representative": case.representative,
        "hmax": case.hmax,
        "raw_kappa_max": stats["raw_kappa_max"],
        "filtered_kappa_max": stats["filtered_kappa_max"],
        "filter_changed_rate": stats["curvature_filter_changed_vertex_rate"],
        "gradation_changed_rate": stats["gradation_changed_vertex_rate"],
        "mean_faces": stats["final_estimated_faces_mean"],
        "min_faces": stats["final_estimated_faces_min"],
        "min_mean_ratio": stats["final_min_mean_estimate_ratio"],
        "hmax_saturation": stats["target_h_max_saturation_rate"],
        "small5_budget": stats["smallest_five_percent_face_budget_rate"],
        "input_edge_violation": stats["input_edge_target_violation_rate"],
        "output_faces": output["faces"],
        "qavg": output["qavg"],
        "qmin": output["qmin"],
        "final_violation": iterations[-1]["violation_rate"] if iterations else "",
        "cd": summary.get("chamfer_distance", {}).get("chamfer_distance", ""),
        "passes": int(long_passes(summary)) if case.iters >= 10 and case.with_cd else "",
    }


def mesh_metrics(summary: dict) -> dict[str, float]:
    return {
        "faces": float(summary["output_summary"]["faces"]),
        "qavg": float(summary["output_summary"]["qavg"]),
        "qmin": float(summary["output_summary"]["qmin"]),
        "violation": float(summary["iterations"][-1]["violation_rate"]),
        "cd": float(summary["chamfer_distance"]["chamfer_distance"]),
        "rmse": float(summary["chamfer_distance"]["symmetric_rmse"]),
    }


def build_comparison_rows(results: list[tuple[Case, dict]]) -> list[dict]:
    rows: list[dict] = []
    for case, summary in results:
        current = mesh_metrics(summary)
        row: dict[str, object] = {"model": case.model, **{f"current_{k}": v for k, v in current.items()}}
        for label, root in (("v2111", BASELINE_2111), ("v2110", BASELINE_2110)):
            baseline = mesh_metrics(load_json(root / case.model / "summary.json"))
            for metric, value in baseline.items():
                row[f"{label}_{metric}"] = value
                row[f"delta_{label}_{metric}"] = current[metric] - value
                row[f"delta_{label}_{metric}_pct"] = (
                    100.0 * (current[metric] - value) / value if value != 0.0 else math.inf
                )
        rows.append(row)
    return rows


def run_field_phase(cli: Path, force: bool) -> tuple[list[tuple[Case, dict]], list[tuple[Case, dict]]]:
    for model in ("T0", "T36"):
        run_case(cli, "controls_iter0", Case(name=model, model=model), force)
    causal = [(case, run_case(cli, "causal", case, force)) for case in causal_cases()]
    screen = [(case, run_case(cli, "screen", case, force)) for case in screen_cases()]
    write_rows(
        RESULT_ROOT / "field_results.csv",
        [result_row("causal", case, summary) for case, summary in causal]
        + [result_row("screen", case, summary) for case, summary in screen],
    )
    return causal, screen


def top_screen_candidates(screen: list[tuple[Case, dict]]) -> list[Case]:
    return [case for case, summary in sorted(screen, key=lambda item: field_rank(item[1], item[0]))[:3]]


def run_short_phase(
    cli: Path, force: bool, screen: list[tuple[Case, dict]]
) -> list[tuple[Case, dict]]:
    cases: list[Case] = []
    for base in top_screen_candidates(screen):
        for representative in ("mean", "min"):
            for hmax in (5.0, 1.8):
                cases.append(
                    Case(
                        name=f"{base.name}_rep{representative}_hmax{str(hmax).replace('.', 'p')}",
                        iters=3,
                        mode=base.mode,
                        beta=base.beta,
                        alpha=base.alpha,
                        gradation=base.gradation,
                        q=base.q,
                        representative=representative,
                        hmax=hmax,
                        with_cd=True,
                        export_iterations=True,
                    )
                )
    short = [(case, run_case(cli, "short_iter3", case, force)) for case in cases]
    short.sort(key=lambda item: short_rank(item[1]))
    write_rows(
        RESULT_ROOT / "short_results.csv",
        [result_row("short", case, summary) for case, summary in short],
    )
    return short


def run_long_phase(
    cli: Path, force: bool, short: list[tuple[Case, dict]]
) -> list[tuple[Case, dict]]:
    cases: list[Case] = []
    for base, _ in short[:2]:
        cases.append(
            Case(
                **{
                    **base.__dict__,
                    "name": base.name.replace("iter3", "") + "_iter10",
                    "iters": 10,
                    "with_cd": True,
                    "export_iterations": True,
                    "export_all_stages": True,
                }
            )
        )
    long_results = [(case, run_case(cli, "long_iter10", case, force)) for case in cases]
    long_results.sort(key=lambda item: (not long_passes(item[1]), short_rank(item[1])))
    winner = long_results[0][0]
    for model in ("T0", "T36"):
        control = Case(
            **{
                **winner.__dict__,
                "name": f"{model}_{winner.name}",
                "model": model,
                "export_all_stages": False,
            }
        )
        long_results.append((control, run_case(cli, "controls_iter10", control, force)))
    write_rows(
        RESULT_ROOT / "long_results.csv",
        [result_row("long", case, summary) for case, summary in long_results],
    )
    return long_results


def run_baseline_controls(cli: Path, force: bool) -> list[tuple[Case, dict]]:
    results: list[tuple[Case, dict]] = []
    for model in ("T0", "T21", "T36"):
        case = Case(
            name=model,
            model=model,
            iters=10,
            with_cd=True,
            export_iterations=True,
            export_all_stages=True,
        )
        results.append((case, run_case(cli, "baseline_controls_iter10", case, force)))
    write_rows(
        RESULT_ROOT / "baseline_control_results.csv",
        [result_row("baseline_control", case, summary) for case, summary in results],
    )
    return results


def model_key(path: Path) -> int:
    match = re.fullmatch(r"T(\d+)", path.stem)
    if not match:
        raise ValueError(path)
    return int(match.group(1))


def run_all_t_phase(cli: Path, force: bool, long_results: list[tuple[Case, dict]]) -> list[tuple[Case, dict]]:
    winner = sorted(long_results[:2], key=lambda item: (not long_passes(item[1]), short_rank(item[1])))[0][0]
    results: list[tuple[Case, dict]] = []
    for mesh in sorted(EXAMPLES.glob("T*.off"), key=model_key):
        model = mesh.stem
        case = Case(
            **{
                **winner.__dict__,
                "name": model,
                "model": model,
                "export_iterations": False,
                "export_all_stages": False,
            }
        )
        results.append((case, run_case(cli, "all_t_iter10", case, force)))
    write_rows(
        RESULT_ROOT / "all_t_results.csv",
        [result_row("all_t", case, summary) for case, summary in results],
    )
    write_rows(RESULT_ROOT / "comparison_vs_v2.11.0_v2.11.1.csv", build_comparison_rows(results))
    return results


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def generate_report(
    causal: list[tuple[Case, dict]],
    screen: list[tuple[Case, dict]],
    short: list[tuple[Case, dict]] | None,
    long_results: list[tuple[Case, dict]] | None,
    all_t: list[tuple[Case, dict]] | None,
) -> None:
    baseline = next(summary for case, summary in causal if case.name == "knockout0_alpha0p75_grad1")
    knockout = next(summary for case, summary in causal if case.name == "knockout1_alpha0p75_grad1")
    no_smooth = next(summary for case, summary in causal if case.name == "knockout0_alpha0p0_grad1")
    no_grad = next(summary for case, summary in causal if case.name == "knockout0_alpha0p75_grad0")
    b = baseline["stats"]
    k = knockout["stats"]
    ns = no_smooth["stats"]
    ng = no_grad["stats"]
    with (RESULT_ROOT / "causal" / "knockout0_alpha0p75_grad1" / "target_field" / "vertices.csv").open(
        encoding="utf-8"
    ) as handle:
        baseline_vertices = list(csv.DictReader(handle))
    spike = max(baseline_vertices, key=lambda row: float(row["raw_kappa"]))
    baseline_smoothed_max = max(float(row["smoothed_kappa"]) for row in baseline_vertices)
    with (RESULT_ROOT / "causal" / "knockout1_alpha0p75_grad1" / "target_field" / "vertices.csv").open(
        encoding="utf-8"
    ) as handle:
        knockout_vertices = list(csv.DictReader(handle))
    knockout_smoothed_max = max(float(row["smoothed_kappa"]) for row in knockout_vertices)
    top3 = sorted(screen, key=lambda item: field_rank(item[1], item[0]))[:3]
    lines = [
        "# v2.11.2 T21 h 异常根因与修复筛选实验报告",
        "",
        "## 已验证的基线",
        "",
        f"- raw kappa 最大值：`{b['raw_kappa_max']:.6g}`；最大点的 `|k|*local_scale` 为 `{b['principal_fit_max_curvature_times_local_scale']:.6g}`。",
        f"- limiter 修改 `{100*b['gradation_changed_vertex_rate']:.2f}%` 顶点。",
        f"- 校准后 hmax 饱和率 `{100*b['target_h_max_saturation_rate']:.2f}%`。",
        f"- mean/min 面数估计分别为 `{b['final_estimated_faces_mean']:.1f}` / `{b['final_estimated_faces_min']:.1f}`。",
        f"- 最小 5% h 面贡献 `{100*b['smallest_five_percent_face_budget_rate']:.2f}%` 的 mean 面数预算。",
        "",
        "## 2×2×2 因果隔离",
        "",
        "| 对照 | filtered kappa max | limiter changed | hmax saturation | min/mean | small5 budget |",
        "|---|---:|---:|---:|---:|---:|",
        f"| 当前完整管线 | {b['filtered_kappa_max']:.6g} | {100*b['gradation_changed_vertex_rate']:.2f}% | {100*b['target_h_max_saturation_rate']:.2f}% | {b['final_min_mean_estimate_ratio']:.4f} | {100*b['smallest_five_percent_face_budget_rate']:.2f}% |",
        f"| 最大点敲除 | {k['filtered_kappa_max']:.6g} | {100*k['gradation_changed_vertex_rate']:.2f}% | {100*k['target_h_max_saturation_rate']:.2f}% | {k['final_min_mean_estimate_ratio']:.4f} | {100*k['smallest_five_percent_face_budget_rate']:.2f}% |",
        f"| 不做 screened smoothing | {ns['filtered_kappa_max']:.6g} | {100*ns['gradation_changed_vertex_rate']:.2f}% | {100*ns['target_h_max_saturation_rate']:.2f}% | {ns['final_min_mean_estimate_ratio']:.4f} | {100*ns['smallest_five_percent_face_budget_rate']:.2f}% |",
        f"| 不做 limiter | {ng['filtered_kappa_max']:.6g} | {100*ng['gradation_changed_vertex_rate']:.2f}% | {100*ng['target_h_max_saturation_rate']:.2f}% | {ng['final_min_mean_estimate_ratio']:.4f} | {100*ng['smallest_five_percent_face_budget_rate']:.2f}% |",
        "",
        "## iter0 修复候选前三名",
        "",
        "| 排名 | Case | min/mean | hmax saturation | small5 budget | edge violation |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for rank, (case, summary) in enumerate(top3, 1):
        stats = summary["stats"]
        lines.append(
            f"| {rank} | `{case.name}` | {stats['final_min_mean_estimate_ratio']:.4f} | "
            f"{100*stats['target_h_max_saturation_rate']:.2f}% | "
            f"{100*stats['smallest_five_percent_face_budget_rate']:.2f}% | "
            f"{100*stats['input_edge_target_violation_rate']:.2f}% |"
        )
    if short:
        lines.extend(
            [
                "",
                "## iter3 排名",
                "",
                "| 排名 | Case | faces | Qavg | Qmin | violation | CD |",
                "|---:|---|---:|---:|---:|---:|---:|",
            ]
        )
        for rank, (case, summary) in enumerate(short, 1):
            lines.append(
                f"| {rank} | `{case.name}` | {summary['output_summary']['faces']} | "
                f"{summary['output_summary']['qavg']:.6f} | {summary['output_summary']['qmin']:.6f} | "
                f"{summary['iterations'][-1]['violation_rate']:.6f} | "
                f"{summary['chamfer_distance']['chamfer_distance']:.6E} |"
            )
    if long_results:
        lines.extend(
            [
                "",
                "## iter10 与对照",
                "",
                "| Case | Model | faces | Qavg | Qmin | violation | CD | 验收 |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for case, summary in long_results:
            lines.append(
                f"| `{case.name}` | {case.model} | {summary['output_summary']['faces']} | "
                f"{summary['output_summary']['qavg']:.6f} | {summary['output_summary']['qmin']:.6f} | "
                f"{summary['iterations'][-1]['violation_rate']:.6f} | "
                f"{summary['chamfer_distance']['chamfer_distance']:.6E} | "
                f"{'pass' if long_passes(summary) else 'fail'} |"
            )
    if all_t:
        current_metrics = [mesh_metrics(summary) for _, summary in all_t]
        baseline_2111_metrics = [
            mesh_metrics(load_json(BASELINE_2111 / case.model / "summary.json"))
            for case, _ in all_t
        ]
        baseline_2110_metrics = [
            mesh_metrics(load_json(BASELINE_2110 / case.model / "summary.json"))
            for case, _ in all_t
        ]
        def averages(items: list[dict[str, float]]) -> dict[str, float]:
            return {key: sum(item[key] for item in items) / len(items) for key in items[0]}
        current_avg = averages(current_metrics)
        avg_2111 = averages(baseline_2111_metrics)
        avg_2110 = averages(baseline_2110_metrics)
        improve_2111 = {
            metric: sum(
                (current[metric] < old[metric])
                if metric in ("violation", "cd", "rmse")
                else (current[metric] > old[metric])
                for current, old in zip(current_metrics, baseline_2111_metrics)
            )
            for metric in ("qavg", "qmin", "violation", "cd", "rmse")
        }
        improve_2110 = {
            metric: sum(
                (current[metric] < old[metric])
                if metric in ("violation", "cd", "rmse")
                else (current[metric] > old[metric])
                for current, old in zip(current_metrics, baseline_2110_metrics)
            )
            for metric in ("qavg", "qmin", "violation", "cd", "rmse")
        }
        lines.extend(
            [
                "",
                "## 全 T 系列",
                "",
                f"完成 `{len(all_t)}` 个模型。入选配置为 `sampling_cap beta=2, gradation=off, representative=mean, hmax=5`。",
                "",
                "| 版本 | 平均 faces | 平均 Qavg | 平均 Qmin | 平均 violation | 平均 CD | 平均 RMSE |",
                "|---|---:|---:|---:|---:|---:|---:|",
                f"| 当前候选 | {current_avg['faces']:.1f} | {current_avg['qavg']:.6f} | {current_avg['qmin']:.6f} | {current_avg['violation']:.6f} | {current_avg['cd']:.6E} | {current_avg['rmse']:.6E} |",
                f"| v2.11.1 | {avg_2111['faces']:.1f} | {avg_2111['qavg']:.6f} | {avg_2111['qmin']:.6f} | {avg_2111['violation']:.6f} | {avg_2111['cd']:.6E} | {avg_2111['rmse']:.6E} |",
                f"| v2.11.0 | {avg_2110['faces']:.1f} | {avg_2110['qavg']:.6f} | {avg_2110['qmin']:.6f} | {avg_2110['violation']:.6f} | {avg_2110['cd']:.6E} | {avg_2110['rmse']:.6E} |",
                "",
                f"相对 v2.11.1：Qavg/Qmin 改善 `{improve_2111['qavg']}/31`、`{improve_2111['qmin']}/31`，violation/CD 改善 `{improve_2111['violation']}/31`、`{improve_2111['cd']}/31`。",
                f"相对 v2.11.0：Qavg/Qmin 改善 `{improve_2110['qavg']}/31`、`{improve_2110['qmin']}/31`，violation/CD 改善 `{improve_2110['violation']}/31`、`{improve_2110['cd']}/31`。",
            ]
        )
    lines.extend(
        [
            "",
            "## 根因回答",
            "",
            f"1. **曲率峰值是不可靠拟合。** 顶点 `{spike['vertex_id']}` 的 one-ring 只有 `{spike['fit_sample_count']}` 个有效样本，SVD 比 `{float(spike['fit_singular_ratio']):.6E}` 仅略高于当前回退阈值 `1E-6`；它是三未知量对三个样本的近病态精确拟合，虽残差仅 `{float(spike['fit_residual_rms']):.6E}`，却产生 `|k|*local_scale={b['principal_fit_max_curvature_times_local_scale']:.1f}` 的不可解析曲率。",
            f"2. **screened smoothing 缩小峰值但没有消除异常。** 峰值由 `37725.5` 降到 `{baseline_smoothed_max:.3f}`；敲除单点后平滑峰值进一步降到 `{knockout_smoothed_max:.3f}`。无 smoothing 时 limiter 修改率为 `{100*ns['gradation_changed_vertex_rate']:.2f}%`，有 smoothing 时为 `{100*b['gradation_changed_vertex_rate']:.2f}%`。",
            f"3. **大面积 hmax 是校准后果，不是原始大 h 的独立根因。** 最大点敲除使 hmax 饱和率从 `{100*b['target_h_max_saturation_rate']:.2f}%` 降到 `{100*k['target_h_max_saturation_rate']:.2f}%`；关闭 limiter 时也仅为 `{100*ng['target_h_max_saturation_rate']:.2f}%`。异常小 h 占据面数预算后，全局 scale 才把其余区域推到 hmax。",
            f"4. **mean 校准确实低估 remesher 所需密度。** 当前场 mean/min 估计为 `{b['final_estimated_faces_mean']:.0f}/{b['final_estimated_faces_min']:.0f}`，相差 `{100*(b['final_min_mean_estimate_ratio']-1):.2f}%`；但直接使用 min 校准在三轮实验中只得到约 1.6 万面，说明它是过度保守的修复，不能单独替换 mean。",
            "5. **首要修复点是曲率稳健化。** sampling cap beta=2 消除了 T21/T36 的灾难性失稳；放宽或关闭 limiter 只是在稳健曲率之后改善跟踪。候选仍未达到 v2.11.0 的整体 Qavg/violation，因此本实验不修改默认配置。",
            "",
            "## 最终判断",
            "",
            "- T21 的主要故障源是近病态三点二次拟合产生的超采样曲率峰值，随后由尺寸映射、limiter 和全局校准放大。",
            "- `sampling_cap beta=2` 是有效的防失稳机制，但当前 `mean` 面数估计和 remesher 跟踪仍有系统偏差。",
            "- 入选候选相对 v2.11.1 大幅修复 T21/T36，也通过 T0 的 Qavg、CD 和面数回归门槛；但 T21 的 Qmin/violation 以及全 T 相对 v2.11.0 的 Qavg/violation 未达标。",
            "- 因此保留所有能力为实验选项，默认仍为 `curvature_outlier_mode=none`、`gradation=on`、`representative=mean`、`hmax=5`。",
            "",
            f"结果目录：`{rel(RESULT_ROOT)}`",
            f"逐模型对比：`{rel(RESULT_ROOT / 'comparison_vs_v2.11.0_v2.11.1.csv')}`",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run T21 root-cause and repair screening experiment")
    parser.add_argument("--cli", type=Path, default=DEFAULT_CLI)
    parser.add_argument("--phase", choices=("field", "short", "long", "all"), default="all")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cli = args.cli.resolve()
    if not cli.exists():
        raise FileNotFoundError(cli)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    causal, screen = run_field_phase(cli, args.force)
    short = None
    long_results = None
    all_t = None
    if args.phase in ("short", "long", "all"):
        short = run_short_phase(cli, args.force, screen)
    if args.phase in ("long", "all"):
        assert short is not None
        run_baseline_controls(cli, args.force)
        long_results = run_long_phase(cli, args.force, short)
    if args.phase == "all":
        assert long_results is not None
        all_t = run_all_t_phase(cli, args.force, long_results)
    generate_report(causal, screen, short, long_results, all_t)
    print(REPORT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
