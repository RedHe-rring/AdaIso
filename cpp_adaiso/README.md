# cpp_adaiso

C++17/CMake reproduction of the AdaIso remeshing workflow described in:

Lv et al., 2024, *Adaptively Isotropic Remeshing Based on Curvature Smoothed Field*.

This is version 2.5.0. The CSF stage is computed exclusively with a screened
Laplace sparse solve, `(M + alpha h^2 L)c = M r`, using barycentric vertex
masses and clamped cotangent weights. The old Dirichlet harmonic CSF solve and
its fixed/free anchor statistics have been removed.

Version 2.5.0 builds a surface-aware log-size target field: CSF curvature is
mapped to target log length, smoothed with screened regularization, constrained
by a local gradation projection, and queried by closest-triangle barycentric
interpolation on the input surface.

## Scope

Implemented:

- OBJ input with polygon fan triangulation.
- ASCII PLY input with polygon fan triangulation.
- ASCII OFF input with polygon fan triangulation.
- OBJ and ASCII PLY output.
- Degenerate and duplicate unoriented triangle cleanup.
- Area-weighted vertex normals.
- Curvature-smoothed field from mean vertex-normal variation and a screened Laplace sparse solve.
- Continuous empirical-quantile local target log-length assignment.
- Surface-aware target lookup with libigl AABB closest-triangle queries and barycentric log-size interpolation.
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

`Eigen3`, `nanoflann`, and `Libigl` are resolved with `find_package` when a
vcpkg toolchain file is supplied. `Eigen3` and `nanoflann` still have
`FetchContent` fallbacks; `Libigl` is required for the surface target field.

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
- `--variant code|paper`: controls remeshing thresholds and collapse reference behavior; CSF curvature smoothing is shared by both variants.
- `--csf-alpha`: screened CSF smoothing strength in `lambda = alpha h^2`. Default: `0.5`.
- `--target-min-factor`: minimum continuous target length factor. Default: `0.6`.
- `--target-max-factor`: maximum continuous target length factor. Default: `1.8`.
- `--target-gamma`: continuous target curvature response exponent. Default: `1.0`.
- `--target-smooth-alpha`: screened target log-size smoothing strength. Default: `0.3`.
- `--target-max-adjacent-ratio`: maximum adjacent input-edge target length ratio. Default: `1.30`.
- `--target-gradation-iters`: gradation projection iterations. Default: `20`.
- `--export-target-field`: optional directory for target-field diagnostics. It writes vertex/edge/face CSV files, edge-ratio statistics, and colored PLY heatmaps for raw curvature, CSF curvature, raw/smoothed/final target size, target delta, edge-ratio hotspots, violation masks, and face log-gradient.
- `--preprocess 0|1`: enables long-edge pre-splitting.
- `--iters`: global split/collapse/flip/smooth iterations.
- `--project 0|1`: enables MLS projection back to the original mesh.
- `--gt`: optional ground-truth mesh for area-sampled Chamfer Distance evaluation.
- `--cd-samples`: samples per Chamfer Distance direction. Default: `100000`.
- `--summary-json`: writes a JSON run summary.

The summary contains input/output vertex counts, face counts, `qavg`, `qmin`,
runtime, core parameters, per-iteration remesh statistics, edge-ratio stability
metrics, and CSF sparse-solve timing/residual statistics. When `--gt` is
provided it also contains GT summary and area-sampled Chamfer Distance metrics.

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
Version 2.2.0 keeps the same CSF and remeshing stages, but smooths the target
edge-length field by default with Gaussian kNN interpolation.
Version 2.3.0 replaces the Dirichlet harmonic CSF solve with screened smoothing
that preserves a data term at every vertex and no longer selects fixed anchor
vertices.
Version 2.4.0 replaces the five-level target-length assignment with a continuous
mapping: each vertex uses the empirical quantile of its CSF curvature, raises it
by `target_gamma`, and interpolates in log length between `target_max_factor`
and `target_min_factor`.
Version 2.5.0 smooths the target field in `log(h)`, applies an adjacent-edge
gradation limit, and evaluates the field on the input surface through libigl
AABB closest-triangle queries and barycentric interpolation.
