#!/usr/bin/env python3
"""Run the T21/T30 high-curvature target_h response experiment matrix."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CPP = ROOT / "cpp_adaiso"
ANALYZE = CPP / "tools" / "analyze_high_curvature_roi.py"

CASES = [
    {
        "case": "baseline_log_dimless_csf_cap0995",
        "curvature_signal": "log_dimless",
        "use_csf": "1",
        "curvature_cap": "0.995",
    },
    {
        "case": "no_log_dimless_csf_cap0995",
        "curvature_signal": "dimless",
        "use_csf": "1",
        "curvature_cap": "0.995",
    },
    {
        "case": "physical_csf_cap0995",
        "curvature_signal": "physical",
        "use_csf": "1",
        "curvature_cap": "0.995",
    },
    {
        "case": "no_cap_dimless_csf",
        "curvature_signal": "dimless",
        "use_csf": "1",
        "curvature_cap": "off",
    },
    {
        "case": "raw_dimless_cap0995",
        "curvature_signal": "dimless",
        "use_csf": "0",
        "curvature_cap": "0.995",
    },
]


def find_cli(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if path.exists():
            return path
        raise FileNotFoundError(path)
    candidates = [
        CPP / "build2" / "adaiso_cli.exe",
        CPP / "build" / "adaiso_cli.exe",
        CPP / "build" / "Release" / "adaiso_cli.exe",
        CPP / "build_analysis" / "adaiso_cli.exe",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("could not find adaiso_cli.exe; pass --cli")


def run(cmd: list[str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(" ".join(cmd) + "\n\n")
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        log.write(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}")


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def first_roi_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return next(reader)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", default="")
    parser.add_argument("--models", default="T21,T30")
    parser.add_argument("--iters", default="0,10")
    parser.add_argument("--cd-samples", type=int, default=100000)
    parser.add_argument(
        "--out-root",
        default=str(CPP / "results" / "v2.12_high_curvature_detail"),
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    cli = find_cli(args.cli or None)
    out_root = Path(args.out_root)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    iters = [int(x.strip()) for x in args.iters.split(",") if x.strip()]
    rows: list[dict[str, object]] = []

    for model in models:
        input_mesh = CPP / "examples" / f"{model}.off"
        if not input_mesh.exists():
            raise FileNotFoundError(input_mesh)
        for it in iters:
            for case in CASES:
                case_dir = out_root / model / f"iter{it:02d}" / case["case"]
                output_mesh = case_dir / "output.ply"
                summary = case_dir / "summary.json"
                target_field = case_dir / "target_field"
                roi_dir = case_dir / "roi"
                cmd = [
                    str(cli),
                    str(input_mesh),
                    "-o",
                    str(output_mesh),
                    "--iters",
                    str(it),
                    "--preprocess",
                    "0",
                    "--curvature-signal",
                    case["curvature_signal"],
                    "--use-csf",
                    case["use_csf"],
                    "--curvature-cap",
                    case["curvature_cap"],
                    "--gt",
                    str(input_mesh),
                    "--cd-samples",
                    str(args.cd_samples),
                    "--export-target-field",
                    str(target_field),
                    "--summary-json",
                    str(summary),
                ]
                if args.quiet:
                    cmd.append("--quiet")
                run(cmd, ROOT, case_dir / "run.log")

                analyze_cmd = [
                    sys.executable,
                    str(ANALYZE),
                    "--input-mesh",
                    str(input_mesh),
                    "--output-mesh",
                    str(output_mesh),
                    "--target-field",
                    str(target_field),
                    "--out-dir",
                    str(roi_dir),
                ]
                run(analyze_cmd, ROOT, case_dir / "roi.log")

                summary_json = read_json(summary)
                roi0 = first_roi_row(roi_dir / "roi_components.csv")
                stats = summary_json["stats"]
                output_summary = summary_json["output_summary"]
                chamfer = summary_json.get("chamfer_distance", {})
                rows.append(
                    {
                        "model": model,
                        "iters": it,
                        "case": case["case"],
                        "curvature_signal": case["curvature_signal"],
                        "use_csf": case["use_csf"],
                        "curvature_cap": case["curvature_cap"],
                        "output_faces": output_summary["faces"],
                        "qavg": output_summary["qavg"],
                        "qmin": output_summary["qmin"],
                        "cd": chamfer.get("chamfer_distance", ""),
                        "rmse": chamfer.get("symmetric_rmse", ""),
                        "target_h_min": stats["target_observed_h_min"],
                        "target_h_max": stats["target_observed_h_max"],
                        "cap_changed": stats["curvature_cap_changed_rate"],
                        "roi0_vertices": roi0["input_vertices"],
                        "roi0_mean_h": roi0["mean_target_h"],
                        "roi0_p05_h": roi0["p05_target_h"],
                        "roi0_h_over_base": roi0["mean_h_over_base"],
                        "roi0_output_edge_target_ratio": roi0["output_edge_target_ratio"],
                        "roi0_local_rmse": roi0["local_symmetric_rmse"],
                        "roi_components": str(roi_dir / "roi_components.csv"),
                    }
                )

    out_root.mkdir(parents=True, exist_ok=True)
    summary_csv = out_root / "run_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    report = out_root / "README.md"
    with report.open("w", encoding="utf-8") as f:
        f.write("# T21/T30 high-curvature target_h response experiment\n\n")
        f.write(f"- CLI: `{cli}`\n")
        f.write(f"- Summary CSV: `{summary_csv}`\n")
        f.write("- ROI components are auto-selected from CSF top 10% union target_h bottom 10%.\n")
        f.write("- Manually confirm the T21 toe and T30 ear component ids from each `roi_components.csv`.\n\n")
        f.write("## Cases\n\n")
        for case in CASES:
            f.write(
                f"- `{case['case']}`: signal={case['curvature_signal']}, "
                f"use_csf={case['use_csf']}, cap={case['curvature_cap']}\n"
            )
    print(f"wrote {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
