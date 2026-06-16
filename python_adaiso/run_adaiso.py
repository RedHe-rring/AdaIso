from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from adaiso import AdaIsoConfig, AdaIsoRemesher, load_mesh, write_obj
from adaiso.metrics import mesh_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Python AdaIso remeshing")
    parser.add_argument("input", type=Path, help="input OBJ/PLY mesh")
    parser.add_argument("-o", "--output", type=Path, default=None, help="output OBJ path")
    parser.add_argument("--mesh-scale", type=float, default=1.0, help="target edge length scale")
    parser.add_argument("--variant", choices=["paper", "code"], default="paper")
    parser.add_argument("--preprocess", action="store_true", help="enable long-edge sub-remeshing")
    parser.add_argument("--iters", type=int, default=5, help="global remeshing iterations")
    parser.add_argument("--csf-iters", type=int, default=20, help="CSF smoothing max iterations")
    parser.add_argument("--no-project", action="store_true", help="disable MLS projection to original surface")
    parser.add_argument(
        "--min-collapse-quality",
        type=float,
        default=0.05,
        help="reject edge collapses whose resulting Qmin is below this value",
    )
    parser.add_argument(
        "--allow-obtuse-flip",
        action="store_true",
        help="allow edge flips that create obtuse triangles",
    )
    parser.add_argument("--quiet", action="store_true", help="reduce console output")
    parser.add_argument("--summary-json", type=Path, default=None, help="optional JSON summary path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output
    if output is None:
        output = args.input.with_name(f"{args.input.stem}_pyadaiso_{args.mesh_scale:.6f}.obj")

    mesh = load_mesh(args.input)
    cfg = AdaIsoConfig(
        mesh_scale=args.mesh_scale,
        variant=args.variant,
        preprocess=args.preprocess,
        global_iterations=args.iters,
        csf_max_iterations=args.csf_iters,
        project_to_surface=not args.no_project,
        min_quality_after_collapse=args.min_collapse_quality,
        avoid_obtuse_flip=not args.allow_obtuse_flip,
        verbose=not args.quiet,
    )
    remesher = AdaIsoRemesher(cfg)

    start = time.perf_counter()
    out_mesh = remesher.remesh(mesh)
    elapsed = time.perf_counter() - start
    write_obj(output, out_mesh)

    summary = {
        "input": str(args.input),
        "output": str(output),
        "variant": args.variant,
        "mesh_scale": args.mesh_scale,
        "preprocess": args.preprocess,
        "runtime_sec": elapsed,
        "input_summary": mesh_summary(mesh),
        "output_summary": mesh_summary(out_mesh),
    }
    if not args.quiet:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
