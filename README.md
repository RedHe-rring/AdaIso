# AdaIso

**Curvature-aware isotropic surface remeshing in C++17**

AdaIso is a research implementation inspired by Lv et al., *Adaptively Isotropic Remeshing Based on Curvature Smoothed Field* (2024). It investigates curvature-guided target sizing and local mesh operations for isotropic surface remeshing.

This repository is a practical research implementation, not a line-by-line port of the original code. Experimental results should be interpreted with their settings and limitations in mind.

## Overview

The current pipeline consists of:

1. **Mesh loading and cleanup** — read a supported surface mesh and remove degenerate faces.
2. **Curvature estimation** — compute a robust mean-curvature signal using a selectable representation.
3. **Optional curvature smoothing** — solve a screened Laplacian sparse system to diffuse the curvature signal.
4. **Target-size construction** — cap extreme curvature values when enabled, then derive local target edge lengths from curvature and a target face budget.
5. **Iterative remeshing** — split long edges, batch-collapse short edges, flip edges, and relocate vertices with feature-aware handling.
6. **Evaluation and diagnostics** — report mesh-quality and edge-length statistics, solver diagnostics, and optional Chamfer Distance against a ground-truth mesh.

Target face count guides the sizing field; the final output face count is not guaranteed to equal the requested budget exactly.

## Main features

- C++17 implementation with a modular CMake build.
- Curvature signal modes: `log_dimless`, `dimless`, and `physical`.
- Optional screened Laplacian curvature smoothing using a sparse linear solve.
- Quantile-based curvature capping and target sizing guided by a requested face budget.
- Local split, batch collapse, flip, and tangent-smoothing operations.
- JSON run summaries with per-iteration statistics and CSF solver diagnostics.
- Optional target-field CSV diagnostics and area-sampled Chamfer Distance evaluation.
- Four focused CTest targets for curvature, screened Laplacian, target sizing, and local operators.

## Repository layout

```text
AdaIso/
├── CMakeLists.txt
├── CHANGELOG.md
├── include/adaiso/       # Public and internal headers
├── src/                  # Core implementation and CLI tools
├── tests/                # Focused C++ tests
├── examples/             # Example surface meshes
└── cpp_adaiso/           # Supplementary scripts and historical experiment materials
```

The `cpp_adaiso/` directory is not required to build the core implementation. Its remaining scripts and historical reports are supplementary and are being curated separately.

## Requirements

- A C++17-compatible compiler
- CMake 3.20 or later
- Eigen3
- nanoflann
- Libigl with a CMake package discoverable by `find_package(Libigl CONFIG REQUIRED)`

CMake attempts to fetch Eigen3 and nanoflann if suitable installed packages are not found. Libigl must be installed and discoverable by CMake. FetchContent may require network access.

## Build and test

Run these commands from the repository root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

With a single-configuration generator such as Ninja or Unix Makefiles, the executable is usually at `build/adaiso_cli`. With a multi-configuration generator such as Visual Studio, it is usually at `build/Release/adaiso_cli.exe` after building the Release configuration.

## Run the remesher

Example using the included `T0.off` mesh:

```bash
./build/adaiso_cli examples/T0.off \
  -o results/T0_remeshed.obj \
  --target-faces 10000 \
  --curvature-signal log_dimless \
  --use-csf 1 \
  --curvature-cap 0.995 \
  --iters 5 \
  --summary-json results/T0_summary.json
```

On Windows with a Visual Studio generator, replace `./build/adaiso_cli` with `build/Release/adaiso_cli.exe`. Adjust `--target-faces` for the size and purpose of your input mesh.

### Evaluate against a ground-truth mesh

```bash
./build/adaiso_eval \
  --mesh results/T0_remeshed.obj \
  --gt examples/T0.off \
  --cd-samples 100000 \
  --json results/T0_metrics.json
```

The evaluator reports mesh summaries and area-sampled Chamfer Distance. Use the same sampling settings and normalization when comparing different methods.

## Command-line options

| Option | Description |
|---|---|
| `--target-faces <n>` | Target face budget used to construct the sizing field. Defaults to the cleaned input face count. |
| `--curvature-signal <mode>` | Curvature representation: `log_dimless`, `dimless`, or `physical`. Default: `log_dimless`. |
| `--use-csf <0|1>` | Enable or disable screened curvature smoothing. Default: `1`. |
| `--curvature-cap <q|off>` | Quantile used to cap extreme curvature values, or `off` to disable capping. Default: `0.995`. |
| `--export-target-field <dir>` | Export target-field CSV diagnostics to a directory. |
| `--preprocess <0|1>` | Enable or disable long-edge pre-splitting. Default: `0`. |
| `--iters <n>` | Number of global remeshing iterations. Default: `5`. |
| `--gt <mesh>` | Optional ground-truth mesh for Chamfer Distance evaluation. |
| `--cd-samples <n>` | Samples per Chamfer Distance direction. Default: `100000`. |
| `--summary-json <path>` | Write a JSON run summary. |
| `--quiet` | Suppress progress logs. |

Use `adaiso_cli --help` or `adaiso_eval --help` to view the command-line usage printed by the executable.

## Supported formats

- **Input:** OBJ, OFF, and ASCII PLY.
- **Output:** OBJ and ASCII PLY.
- **Not supported:** Binary PLY.

Polygonal faces are triangulated by a fan during loading. This is suitable for simple polygons, but it is not a general polygon triangulation algorithm for arbitrary non-convex faces.

## Tests

The CMake build defines four focused test targets:

- `screened_laplacian_tests`
- `target_face_sizing_tests`
- `raw_curvature_tests`
- `local_operators_tests`

Run all enabled tests with `ctest --test-dir build -C Release --output-on-failure`.

## Reported metrics

The run summary includes vertex and face counts, average and minimum triangle-quality values (`qavg`, `qmin`), per-iteration split/collapse/flip counts, edge-length-to-target statistics, target-sizing diagnostics, and screened-Laplacian timing and residual statistics. When `--gt` is provided, it also reports area-sampled Chamfer Distance metrics.

These metrics describe the output and the configured experiment; they do not by themselves establish superiority over another method. No general performance or quality improvement is claimed here without a reproducible, like-for-like comparison.

## Limitations

- The project is experimental and is not a complete reproduction of every data structure or baseline method from the referenced paper.
- Binary PLY input is not supported.
- The requested target face count is a sizing target, not an exact output-face-count guarantee.
- Mesh quality and geometric fidelity depend on the input mesh and parameter settings; inspect the output and metrics for each use case.
- Historical experiment reports may use different versions and settings. Do not compare their numbers without checking the configurations.

## Reference

Lv et al. *Adaptively Isotropic Remeshing Based on Curvature Smoothed Field* (2024).

Please consult the original publication for the method being reproduced. The implementation and experiments in this repository should be cited and described separately from the original paper.