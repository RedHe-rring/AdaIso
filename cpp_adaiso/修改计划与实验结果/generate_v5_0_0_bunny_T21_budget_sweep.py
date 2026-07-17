import csv
import importlib.util
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


HERE = Path(__file__).resolve().parent
BASE_SCRIPT = HERE / "generate_v5_0_0_sqrt_vs_linear_experiment.py"

spec = importlib.util.spec_from_file_location("sqrt_vs_linear_base", BASE_SCRIPT)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


MODELS = {
    "bunny": "bunny.obj",
    "T21": "T21.off",
}
BUDGET_RATIOS = [0.25, 0.50, 0.75, 1.00, 1.50]
MAPPINGS = ["inverse-sqrt", "inverse-linear"]

RESULT_ROOT = base.ROOT / "results" / "v5_0_0_bunny_T21_budget_sweep"
REPORT_PATH = HERE / "v5.0.0 bunny_T21_budget_sweep_report.md"
GLOBAL_CD_SAMPLES = 50000
LOCAL_SAMPLES = 80000
ITERATIONS = 5

# Reuse the existing local-detail analysis sample count.
base.LOCAL_SAMPLES = LOCAL_SAMPLES


def ratio_label(ratio: float) -> str:
    return f"{int(round(ratio * 100)):03d}pct"


def target_faces_for(model_path: Path, ratio: float) -> int:
    mesh = base.load_mesh(model_path)
    return max(1, int(round(len(mesh.faces) * ratio)))


def run_one(model_key: str, model_name: str, ratio: float, mapping: str):
    model_path = base.EXAMPLES / model_name
    target_faces = target_faces_for(model_path, ratio)
    out_dir = RESULT_ROOT / model_key / ratio_label(ratio) / mapping
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "output.ply"
    summary_path = out_dir / "summary.json"
    target_dir = out_dir / "target_field"

    if not (output_path.exists() and summary_path.exists() and (target_dir / "vertices.csv").exists()):
        command = [
            str(base.CLI),
            str(model_path),
            "-o",
            str(output_path),
            "--curvature-mode",
            "robust-mean",
            "--target-size-mapping",
            mapping,
            "--target-faces",
            str(target_faces),
            "--iters",
            str(ITERATIONS),
            "--gt",
            str(model_path),
            "--cd-samples",
            str(GLOBAL_CD_SAMPLES),
            "--summary-json",
            str(summary_path),
            "--export-target-field",
            str(target_dir),
            "--quiet",
        ]
        print(f"running {model_key} ratio={ratio:g} mapping={mapping} target_faces={target_faces}")
        base.run_command(command, base.ROOT, out_dir / "run.log")
    else:
        print(f"reusing {model_key} ratio={ratio:g} mapping={mapping}")

    summary = base.load_summary(summary_path)
    local = base.analyze_local_detail(model_path, output_path)
    target_stats = base.target_field_stats(target_dir)
    row = {
        "model": model_key,
        "input_mesh": model_name,
        "budget_ratio": ratio,
        "budget_label": ratio_label(ratio),
        "mapping": mapping,
        "target_faces_requested": target_faces,
        "output_vertices": summary["output_summary"]["vertices"],
        "output_faces": summary["output_summary"]["faces"],
        "output_face_ratio": summary["output_summary"]["faces"] / target_faces,
        "qavg": summary["output_summary"]["qavg"],
        "qmin": summary["output_summary"]["qmin"],
        "runtime_sec": summary["runtime_sec"],
        "global_chamfer_distance": summary["chamfer_distance"]["chamfer_distance"],
        "global_symmetric_rmse": summary["chamfer_distance"]["symmetric_rmse"],
        "global_output_to_gt_msd": summary["chamfer_distance"]["output_to_gt_msd"],
        "global_gt_to_output_msd": summary["chamfer_distance"]["gt_to_output_msd"],
    }
    row.update(local)
    row.update(target_stats)
    return row


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def percent_delta(linear, sqrt):
    if sqrt == 0 or sqrt is None or linear is None:
        return float("nan")
    return (linear - sqrt) / sqrt * 100.0


def build_pairwise(rows):
    df = pd.DataFrame(rows)
    out = []
    for (model, ratio), subdf in df.groupby(["model", "budget_ratio"]):
        sub = subdf.set_index("mapping")
        if "inverse-sqrt" not in sub.index or "inverse-linear" not in sub.index:
            continue
        sqrt = sub.loc["inverse-sqrt"]
        linear = sub.loc["inverse-linear"]
        row = {
            "model": model,
            "budget_ratio": ratio,
            "target_faces_requested": int(sqrt["target_faces_requested"]),
            "sqrt_output_faces": int(sqrt["output_faces"]),
            "linear_output_faces": int(linear["output_faces"]),
            "output_faces_delta_pct": percent_delta(linear["output_faces"], sqrt["output_faces"]),
        }
        for metric in [
            "global_symmetric_rmse",
            "high_curv_top5_rmse",
            "high_curv_top10_rmse",
            "high_curv_top20_rmse",
            "target_h_top10_mean",
            "qavg",
            "qmin",
        ]:
            row[f"{metric}_sqrt"] = float(sqrt[metric])
            row[f"{metric}_linear"] = float(linear[metric])
            row[f"{metric}_delta_pct"] = percent_delta(linear[metric], sqrt[metric])
        out.append(row)
    return sorted(out, key=lambda r: (r["model"], r["budget_ratio"]))


def fmt(value, digits=5):
    if value is None:
        return ""
    try:
        value = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(value):
        return "nan"
    return f"{value:.{digits}g}"


def plot_curves(rows, pairwise_rows):
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    pdf = pd.DataFrame(pairwise_rows)
    for model in MODELS:
        sub = df[df["model"] == model]
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for mapping in MAPPINGS:
            m = sub[sub["mapping"] == mapping].sort_values("budget_ratio")
            axes[0].plot(m["budget_ratio"], m["global_symmetric_rmse"], marker="o", label=mapping)
            axes[1].plot(m["budget_ratio"], m["high_curv_top10_rmse"], marker="o", label=mapping)
            axes[2].plot(m["budget_ratio"], m["output_faces"], marker="o", label=mapping)
        axes[0].set_title("Global symmetric RMSE")
        axes[1].set_title("High-curvature top10 RMSE")
        axes[2].set_title("Output faces")
        for ax in axes:
            ax.set_xlabel("target face budget ratio")
            ax.grid(True, alpha=0.3)
            ax.legend()
        fig.tight_layout()
        fig.savefig(RESULT_ROOT / f"{model}_budget_curves.png", dpi=160)
        plt.close(fig)

        psub = pdf[pdf["model"] == model].sort_values("budget_ratio")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.plot(psub["budget_ratio"], psub["global_symmetric_rmse_delta_pct"], marker="o", label="global RMSE delta")
        ax.plot(psub["budget_ratio"], psub["high_curv_top10_rmse_delta_pct"], marker="o", label="top10 RMSE delta")
        ax.plot(psub["budget_ratio"], psub["output_faces_delta_pct"], marker="o", label="output faces delta")
        ax.set_xlabel("target face budget ratio")
        ax.set_ylabel("linear vs sqrt delta (%)")
        ax.set_title(f"{model}: linear relative to sqrt")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(RESULT_ROOT / f"{model}_linear_vs_sqrt_delta.png", dpi=160)
        plt.close(fig)


def build_report(rows, pairwise_rows):
    df = pd.DataFrame(rows)
    pdf = pd.DataFrame(pairwise_rows)
    lines = []
    lines.append("# v5.0.0 bunny/T21 budget sweep: inverse-sqrt vs inverse-linear")
    lines.append("")
    lines.append(f"- Results dir: `{RESULT_ROOT.relative_to(base.ROOT)}`")
    lines.append(f"- Models: `bunny.obj`, `T21.off`")
    lines.append(f"- Budget ratios: `{', '.join(str(x) for x in BUDGET_RATIOS)}` times input face count")
    lines.append(f"- Fixed params: `curvature-mode=robust-mean`, `iters={ITERATIONS}`, `cd_samples={GLOBAL_CD_SAMPLES}`")
    lines.append("- Compared mappings: `inverse-sqrt` (`h=lambda/sqrt(kappa)`) vs `inverse-linear` (`h=lambda/kappa`).")
    lines.append("- Local metric: GT high-curvature sample points to output sampled points, using top5/top10/top20 GT robust-mean curvature signal.")
    lines.append("")
    lines.append("## Pairwise Delta")
    lines.append("")
    lines.append("Negative RMSE delta means `inverse-linear` is better than `inverse-sqrt`; positive face delta means `inverse-linear` produced more triangles.")
    lines.append("")
    lines.append("| model | budget | requested faces | sqrt faces | linear faces | face delta | global RMSE delta | top5 RMSE delta | top10 RMSE delta | top20 RMSE delta | target_h top10 delta |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in pairwise_rows:
        lines.append(
            f"| {row['model']} | {fmt(row['budget_ratio'], 2)} | {row['target_faces_requested']} | "
            f"{row['sqrt_output_faces']} | {row['linear_output_faces']} | {fmt(row['output_faces_delta_pct'])}% | "
            f"{fmt(row['global_symmetric_rmse_delta_pct'])}% | {fmt(row['high_curv_top5_rmse_delta_pct'])}% | "
            f"{fmt(row['high_curv_top10_rmse_delta_pct'])}% | {fmt(row['high_curv_top20_rmse_delta_pct'])}% | "
            f"{fmt(row['target_h_top10_mean_delta_pct'])}% |"
        )
    lines.append("")
    lines.append("## Raw Metrics")
    lines.append("")
    lines.append("| model | budget | mapping | output faces | global RMSE | high top5 RMSE | high top10 RMSE | high top20 RMSE | qavg | qmin | target_h top10 mean |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, row in df.sort_values(["model", "budget_ratio", "mapping"]).iterrows():
        lines.append(
            f"| {row['model']} | {fmt(row['budget_ratio'], 2)} | {row['mapping']} | {int(row['output_faces'])} | "
            f"{fmt(row['global_symmetric_rmse'])} | {fmt(row['high_curv_top5_rmse'])} | "
            f"{fmt(row['high_curv_top10_rmse'])} | {fmt(row['high_curv_top20_rmse'])} | "
            f"{fmt(row['qavg'])} | {fmt(row['qmin'])} | {fmt(row['target_h_top10_mean'])} |"
        )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    for model in MODELS:
        sub = pdf[pdf["model"] == model]
        top10_wins = int((sub["high_curv_top10_rmse_delta_pct"] < -1.0).sum())
        global_worse = int((sub["global_symmetric_rmse_delta_pct"] > 1.0).sum())
        mean_top10 = sub["high_curv_top10_rmse_delta_pct"].mean()
        mean_global = sub["global_symmetric_rmse_delta_pct"].mean()
        mean_faces = sub["output_faces_delta_pct"].mean()
        lines.append(
            f"- `{model}`: top10 high-curvature RMSE delta mean `{fmt(mean_top10)}%`, "
            f"`inverse-linear` improves >1% in `{top10_wins}/{len(sub)}` budgets; "
            f"global RMSE delta mean `{fmt(mean_global)}%`, global worse >1% in `{global_worse}/{len(sub)}` budgets; "
            f"output faces delta mean `{fmt(mean_faces)}%`."
        )
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    for model in MODELS:
        lines.append(f"- `{model}_budget_curves.png`: absolute curves for global RMSE, high-curvature top10 RMSE, and output faces.")
        lines.append(f"- `{model}_linear_vs_sqrt_delta.png`: percent delta curves for linear relative to sqrt.")
    lines.append("")
    return "\n".join(lines)


def main():
    if not base.CLI.exists():
        raise RuntimeError(f"missing CLI: {base.CLI}")
    rows = []
    for model_key, model_name in MODELS.items():
        for ratio in BUDGET_RATIOS:
            for mapping in MAPPINGS:
                rows.append(run_one(model_key, model_name, ratio, mapping))
                write_csv(RESULT_ROOT / "run_summary.csv", rows)
    pairwise_rows = build_pairwise(rows)
    write_csv(RESULT_ROOT / "run_summary.csv", rows)
    write_csv(RESULT_ROOT / "pairwise_delta_summary.csv", pairwise_rows)
    plot_curves(rows, pairwise_rows)
    REPORT_PATH.write_text(build_report(rows, pairwise_rows), encoding="utf-8")
    print(f"wrote {RESULT_ROOT / 'run_summary.csv'}")
    print(f"wrote {RESULT_ROOT / 'pairwise_delta_summary.csv'}")
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
