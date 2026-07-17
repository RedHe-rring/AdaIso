#!/usr/bin/env python3
"""Generate a Chinese Markdown report for the high-curvature target_h experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CPP = ROOT / "cpp_adaiso"
DEFAULT_OUT_ROOT = CPP / "results" / "v2.12_high_curvature_detail"
DEFAULT_REPORT = CPP / "修改计划与实验结果" / "v2.12 高曲率细节target_h响应实验报告.md"

CASE_LABELS = {
    "baseline_log_dimless_csf_cap0995": "baseline log_dimless + CSF + cap",
    "no_log_dimless_csf_cap0995": "no-log dimless + CSF + cap",
    "physical_csf_cap0995": "physical + CSF + cap",
    "no_cap_dimless_csf": "dimless + CSF + no-cap",
    "raw_dimless_cap0995": "raw dimless + cap",
}

CASE_ORDER = list(CASE_LABELS)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def ffloat(value: object, digits: int = 6) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return ""
    if abs(x) != 0.0 and (abs(x) < 1e-4 or abs(x) >= 1e4):
        return f"{x:.{digits}E}"
    return f"{x:.{digits}f}"


def fint(value: object) -> str:
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return ""


def pct_delta(new: object, old: object) -> str:
    try:
        n = float(new)
        o = float(old)
    except (TypeError, ValueError):
        return ""
    if o == 0.0:
        return ""
    return f"{(n / o - 1.0) * 100.0:+.2f}%"


def case_dir_from_row(row: dict[str, str]) -> Path:
    roi_path = Path(row["roi_components"])
    if not roi_path.is_absolute():
        roi_path = ROOT / roi_path
    return roi_path.parent.parent


def enrich_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    enriched: list[dict[str, object]] = []
    for row in rows:
        case_dir = case_dir_from_row(row)
        summary = read_json(case_dir / "summary.json")
        field_stats = read_json(case_dir / "target_field" / "field_stats.json")
        roi_rows = read_csv(case_dir / "roi" / "roi_components.csv")
        final_iteration = summary["iterations"][-1] if summary["iterations"] else {}
        item: dict[str, object] = dict(row)
        item["case_dir"] = case_dir
        item["field_stats"] = field_stats
        item["final_violation"] = final_iteration.get("violation_rate", 0.0)
        item["summary_csf_success"] = summary["stats"].get("csf_solver_success", False)
        item["roi_rows"] = roi_rows
        enriched.append(item)
    return enriched


def row_by(rows: list[dict[str, object]], model: str, iters: int, case: str) -> dict[str, object]:
    for row in rows:
        if row["model"] == model and int(str(row["iters"])) == iters and row["case"] == case:
            return row
    raise KeyError((model, iters, case))


def global_table(rows: list[dict[str, object]], model: str, iters: int) -> list[str]:
    lines = [
        "| Case | Faces | Qavg | Qmin | final viol | CD | RMSE | h min/max | cap changed | ROI0 ratio | ROI0 local RMSE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in CASE_ORDER:
        r = row_by(rows, model, iters, case)
        h_range = f"{ffloat(r['target_h_min'])} / {ffloat(r['target_h_max'])}"
        lines.append(
            "| "
            + " | ".join(
                [
                    CASE_LABELS[case],
                    fint(r["output_faces"]),
                    ffloat(r["qavg"]),
                    ffloat(r["qmin"]),
                    ffloat(r["final_violation"]),
                    ffloat(r["cd"]),
                    ffloat(r["rmse"]),
                    h_range,
                    ffloat(r["cap_changed"]),
                    ffloat(r["roi0_output_edge_target_ratio"]),
                    ffloat(r["roi0_local_rmse"]),
                ]
            )
            + " |"
        )
    return lines


def roi_table(rows: list[dict[str, object]], model: str, iters: int) -> list[str]:
    lines = [
        "| Case | ROI0 V | mean h | p05 h | h/base | output edge/h | local RMSE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for case in CASE_ORDER:
        r = row_by(rows, model, iters, case)
        lines.append(
            "| "
            + " | ".join(
                [
                    CASE_LABELS[case],
                    fint(r["roi0_vertices"]),
                    ffloat(r["roi0_mean_h"]),
                    ffloat(r["roi0_p05_h"]),
                    ffloat(r["roi0_h_over_base"]),
                    ffloat(r["roi0_output_edge_target_ratio"]),
                    ffloat(r["roi0_local_rmse"]),
                ]
            )
            + " |"
        )
    return lines


def top5_roi_table(rows: list[dict[str, object]], model: str, iters: int) -> list[str]:
    lines = [
        "| Case | comp | V | mean h | h/base | output edge/h | local RMSE | centroid |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for case in CASE_ORDER:
        r = row_by(rows, model, iters, case)
        roi_rows = r["roi_rows"]
        assert isinstance(roi_rows, list)
        for roi in roi_rows[:5]:
            centroid = (
                f"({ffloat(roi['centroid_x'], 4)}, "
                f"{ffloat(roi['centroid_y'], 4)}, "
                f"{ffloat(roi['centroid_z'], 4)})"
            )
            lines.append(
                "| "
                + " | ".join(
                    [
                        CASE_LABELS[case],
                        roi["component_id"],
                        roi["input_vertices"],
                        ffloat(roi["mean_target_h"]),
                        ffloat(roi["mean_h_over_base"]),
                        ffloat(roi["output_edge_target_ratio"]),
                        ffloat(roi["local_symmetric_rmse"]),
                        centroid,
                    ]
                )
                + " |"
            )
    return lines


def model_conclusions(rows: list[dict[str, object]], model: str) -> list[str]:
    base0 = row_by(rows, model, 0, "baseline_log_dimless_csf_cap0995")
    dim0 = row_by(rows, model, 0, "no_log_dimless_csf_cap0995")
    phys0 = row_by(rows, model, 0, "physical_csf_cap0995")
    nocap0 = row_by(rows, model, 0, "no_cap_dimless_csf")
    raw0 = row_by(rows, model, 0, "raw_dimless_cap0995")
    base10 = row_by(rows, model, 10, "baseline_log_dimless_csf_cap0995")
    dim10 = row_by(rows, model, 10, "no_log_dimless_csf_cap0995")
    phys10 = row_by(rows, model, 10, "physical_csf_cap0995")
    raw10 = row_by(rows, model, 10, "raw_dimless_cap0995")

    phys_dim_diff_h = abs(float(phys0["roi0_mean_h"]) - float(dim0["roi0_mean_h"]))
    phys_dim_diff_cd = abs(float(phys10["cd"]) - float(dim10["cd"]))
    lines = [
        f"- `physical` 与 `dimless` 在 iter0 的 ROI0 mean h 差值为 `{phys_dim_diff_h:.3E}`，iter10 CD 差值为 `{phys_dim_diff_cd:.3E}`；可视为常数尺度被 lambda 校准抵消。",
        f"- 去掉 `log1p` 后，ROI0 mean h 相对 baseline 变化 `{pct_delta(dim0['roi0_mean_h'], base0['roi0_mean_h'])}`，p05 h 变化 `{pct_delta(dim0['roi0_p05_h'], base0['roi0_p05_h'])}`。",
        f"- 关闭 cap 相对 no-log 的 ROI0 mean h 变化 `{pct_delta(nocap0['roi0_mean_h'], dim0['roi0_mean_h'])}`；cap changed rate 从 `{ffloat(dim0['cap_changed'])}` 到 `0`。",
        f"- 不做 CSF 的 raw-field 让 ROI0 mean h 相对 no-log 变化 `{pct_delta(raw0['roi0_mean_h'], dim0['roi0_mean_h'])}`，但 iter10 Qavg 从 `{ffloat(dim10['qavg'])}` 降到 `{ffloat(raw10['qavg'])}`，Qmin 从 `{ffloat(dim10['qmin'])}` 降到 `{ffloat(raw10['qmin'])}`。",
        f"- baseline iter10 的 ROI0 output edge/h 为 `{ffloat(base10['roi0_output_edge_target_ratio'])}`；no-log 为 `{ffloat(dim10['roi0_output_edge_target_ratio'])}`。大于 split 阈值 1.65 时说明局部仍有欠跟踪风险。",
    ]
    return lines


def completeness(rows: list[dict[str, object]], out_root: Path) -> list[str]:
    summary_count = len(list(out_root.rglob("summary.json")))
    roi_count = len(list(out_root.rglob("roi_components.csv")))
    output_count = len(list(out_root.rglob("output.ply")))
    vertex_csv_count = len(list(out_root.rglob("vertices.csv")))
    failures = []
    for row in rows:
        if not row["summary_csf_success"]:
            failures.append(f"{row['model']} iter{row['iters']} {row['case']}")
    return [
        "| Check | Result |",
        "|---|---:|",
        f"| run_summary rows | {len(rows)} / 20 |",
        f"| summary.json | {summary_count} / 20 |",
        f"| output.ply | {output_count} / 20 |",
        f"| target_field vertices.csv | {vertex_csv_count} / 20 |",
        f"| roi_components.csv | {roi_count} / 20 |",
        f"| csf_solver_success false | {len(failures)} |",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    args = parser.parse_args()

    out_root = Path(args.out_root)
    report_path = Path(args.report)
    rows = enrich_rows(read_csv(out_root / "run_summary.csv"))

    lines: list[str] = [
        "# v2.12 高曲率细节 target_h 响应实验报告",
        "",
        "## 实验目的",
        "",
        "验证 T21 鸟脚趾与 T30 马耳朵一类高曲率细节中，target_h 偏大或细节不足是否来自曲率无量纲化、log1p 动态范围压缩、CSF 平滑/cap，还是 remesh 对小 h 的跟踪不足。",
        "",
        "## 实验设置",
        "",
        "| 项目 | 值 |",
        "|---|---|",
        "| 模型 | T21, T30 |",
        "| iters | 0, 10 |",
        "| CD samples | 100000 |",
        "| ROI 自动候选 | CSF top 10% ∪ target_h bottom 10%，按连通分量排序 |",
        f"| 结果目录 | `{out_root}` |",
        f"| run summary | `{out_root / 'run_summary.csv'}` |",
        "",
        "## 完整性检查",
        "",
        *completeness(rows, out_root),
        "",
        "## 总体结论",
        "",
    ]

    for model in ("T21", "T30"):
        lines.extend([f"### {model}", ""])
        lines.extend(model_conclusions(rows, model))
        lines.append("")

    lines.extend(
        [
            "综合判断：本实验不支持“物理曲率变成无量纲曲率这一常数尺度变化本身导致高曲率 h 不够小”。`physical` 与 `dimless` 基本重合，说明 lambda 面数校准抵消了常数缩放。更直接的因素是 `log1p` 压缩与 CSF 平滑会抬高或扩散局部高曲率 ROI 的 h；完全绕过 CSF 虽能显著减小 ROI h，但会带来更差的全局质量和更多面数。",
            "",
            "## T21 结果",
            "",
            "### T21 iter0 target-field / ROI0",
            "",
            *roi_table(rows, "T21", 0),
            "",
            "### T21 iter10 remesh 指标",
            "",
            *global_table(rows, "T21", 10),
            "",
            "### T21 iter10 前 5 个 ROI 候选",
            "",
            *top5_roi_table(rows, "T21", 10),
            "",
            "## T30 结果",
            "",
            "### T30 iter0 target-field / ROI0",
            "",
            *roi_table(rows, "T30", 0),
            "",
            "### T30 iter10 remesh 指标",
            "",
            *global_table(rows, "T30", 10),
            "",
            "### T30 iter10 前 5 个 ROI 候选",
            "",
            *top5_roi_table(rows, "T30", 10),
            "",
            "## 判定",
            "",
            "1. `physical` 与 `dimless` 的 target_h 和 ROI 指标基本一致，曲率无量纲化的常数缩放不是主因。",
            "2. `log_dimless` 到 `dimless` 会降低 ROI h，说明 `log1p` 确实压缩了高曲率响应。",
            "3. `no-cap` 相对 `dimless` 的变化较小，q0.995 cap 对主 ROI 不是最主要因素。",
            "4. `raw_dimless` 的 ROI h 最小，但 iter10 输出面数增加、Qavg/Qmin 明显下降，说明直接取消 CSF 会牺牲全局网格质量。",
            "5. 若某个候选 ROI 的 `output edge/h` 仍高于 1.65，应优先检查 remesh 跟踪、collapse/smoothing 或局部预算，而不是继续归因于无量纲曲率。",
            "",
            "## 备注",
            "",
            "ROI0 是自动候选中最大的连通分量，不一定就是最终人工确认的 T21 脚趾或 T30 马耳朵。报告列出前 5 个候选的 centroid，后续可在 mesh viewer 中确认目标部位并固定对应 component_id 复算局部结论。",
            "",
        ]
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
