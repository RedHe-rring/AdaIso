from __future__ import annotations

import argparse
import json
from pathlib import Path

from adaiso import load_mesh
from adaiso.metrics import mesh_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate mesh quality summary")
    parser.add_argument("mesh", type=Path)
    args = parser.parse_args()
    mesh = load_mesh(args.mesh)
    print(json.dumps(mesh_summary(mesh), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
