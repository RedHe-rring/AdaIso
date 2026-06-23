# cpp_adaiso

C++17/CMake reproduction of the AdaIso remeshing workflow described in:

Lv et al., 2024, *Adaptively Isotropic Remeshing Based on Curvature Smoothed Field*.

This is version 2.1.1. The CSF stage is computed exclusively with a Dirichlet
Laplace sparse solve. The old explicit CSF iteration parameters and iteration
statistics were removed as a breaking interface change.

Version 2.1.1 adds ASCII PLY output, per-iteration JSON summary records, and
optional area-sampled Chamfer Distance evaluation against a ground-truth mesh.

## Scope

Implemented:

- OBJ input with polygon fan triangulation.
- ASCII PLY input with polygon fan triangulation.
- ASCII OFF input with polygon fan triangulation.
- OBJ and ASCII PLY output.
- Degenerate and duplicate unoriented triangle cleanup.
- Area-weighted vertex normals.
- Curvature-smoothed field from vertex normal variation and a Dirichlet Laplace sparse solve.
- Histogram-based local target edge length assignment.
- KD-tree nearest target length lookup via nanoflann.
- Global remeshing loop: split, batch collapse, cleanup, flip, tangent smoothing.
- MLS projection to the original mesh.
- Optional area-sampled Chamfer Distance against a GT mesh.
- CLI executable: `adaiso_cli`.

Not implemented:

- Binary PLY.
- VCG/PCL/CGAL/CUDA dependencies.
- The author's original `pointNeighbor` pair-list data structure.
- Baseline methods from the paper.

## Build

Install one C++ compiler toolchain first:

- MSVC Build Tools, or
- MinGW-w64, or
- clang/clang-cl.

Then configure and build:

```powershell
cmake -S cpp_adaiso -B cpp_adaiso/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp_adaiso/build --config Release
ctest --test-dir cpp_adaiso/build --output-on-failure
```

With the Visual Studio developer environment and vcpkg in `e:/dev/vcpkg`, the
validated command is:

```powershell
cmd /c "call ""C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat"" -arch=x64 && cmake -S cpp_adaiso -B cpp_adaiso/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=e:/dev/vcpkg/scripts/buildsystems/vcpkg.cmake -DVCPKG_TARGET_TRIPLET=x64-windows"
cmd /c "call ""C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat"" -arch=x64 && cmake --build cpp_adaiso/build --config Release"
cmd /c "call ""C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat"" -arch=x64 && ctest --test-dir cpp_adaiso/build --output-on-failure"
```

`Eigen3` and `nanoflann` are first resolved with `find_package`, so vcpkg
packages are used when a vcpkg toolchain file is supplied. If they are not
available, CMake falls back to `FetchContent`.

## CLI

```powershell
cpp_adaiso/build/Release/adaiso_cli.exe cpp_adaiso/examples/pyramid.obj `
  -o cpp_adaiso/out.obj `
  --mesh-scale 0.8 `
  --variant code `
  --preprocess 1 `
  --iters 2 `
  --project 1 `
  --summary-json cpp_adaiso/summary.json
```

On single-config generators such as Ninja or Makefiles, the executable is
usually located at:

```powershell
cpp_adaiso/build/adaiso_cli.exe
```

OFF input is supported:

```powershell
cpp_adaiso/build/adaiso_cli.exe cpp_adaiso/examples/T0.off `
  -o cpp_adaiso/results/T0_T10/T0_iter10_project1.obj `
  --variant code `
  --mesh-scale 1.0 `
  --iters 10 `
  --preprocess 0 `
  --project 1 `
  --summary-json cpp_adaiso/results/T0_T10/T0_iter10_project1.json
```

## Parameters

- `--mesh-scale`: global target edge length scale. Smaller values create denser meshes.
- `--variant code|paper`: `code` uses source-like split/collapse thresholds and length-scaled CSF weights; `paper` uses the paper split/collapse convention and cotangent CSF weights.
- `--preprocess 0|1`: enables long-edge pre-splitting.
- `--iters`: global split/collapse/flip/smooth iterations.
- `--project 0|1`: enables MLS projection back to the original mesh.
- `--gt`: optional ground-truth mesh for area-sampled Chamfer Distance evaluation.
- `--cd-samples`: samples per Chamfer Distance direction. Default: `100000`.
- `--summary-json`: writes a JSON run summary.

The summary contains input/output vertex counts, face counts, `qavg`, `qmin`,
runtime, core parameters, per-iteration remesh statistics, and CSF sparse-solve
timing/residual statistics. When `--gt` is provided it also contains GT summary
and area-sampled Chamfer Distance metrics.

## Smoke Test

After building:

```powershell
cpp_adaiso/build/Release/adaiso_cli.exe cpp_adaiso/examples/pyramid.obj `
  -o cpp_adaiso/out.obj `
  --variant code `
  --mesh-scale 0.8 `
  --iters 2 `
  --preprocess 1 `
  --summary-json cpp_adaiso/summary.json
```

Expected acceptance:

- `cpp_adaiso/out.obj` exists.
- All face indices are valid.
- `summary.json` reports positive `qavg` and `qmin`, per-iteration stats, successful CSF solve status, and CSF residual statistics.
- The basic `adaiso_tests` target passes.

## Notes On Reproduction

This is a practical reproduction rather than a line-by-line port. It avoids
known fragile behavior from the original code, including uninitialized MLS
bandwidth. MLS uses:

```text
h = dK + max(0, dK - dK-1)
```

MLS projection is guarded by a global triangle-quality check: if projection
would reduce `Qmin` below the pre-projection mesh, the projected positions are
rejected.

The result should be close in behavior to the Python reproduction under the same
parameters, but individual vertex positions are not expected to match exactly.
Version 2.0.0 no longer attempts to reproduce finite-step CSF iteration
intermediates; it solves the corresponding Dirichlet Laplace steady-state system.
Version 2.1.0 keeps that CSF behavior unchanged and improves collapse throughput
with batch passes. Version 2.1.1 keeps the algorithm behavior unchanged and adds
PLY output, iteration logging, and optional GT Chamfer Distance evaluation.
