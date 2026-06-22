# cpp_adaiso

C++17/CMake reproduction of the AdaIso remeshing workflow described in:

Lv et al., 2024, *Adaptively Isotropic Remeshing Based on Curvature Smoothed Field*.

This is version 3.0.0. The remeshing size field now uses maximum absolute
principal curvature and an approximation-error tolerance directly. The old
remesher-level CSF histogram/multiplier size control is no longer used.

## Scope

Implemented:

- OBJ input with polygon fan triangulation.
- ASCII PLY input with polygon fan triangulation.
- ASCII OFF input with polygon fan triangulation.
- OBJ output.
- Degenerate and duplicate unoriented triangle cleanup.
- Area-weighted vertex normals.
- Maximum absolute principal curvature estimation from cotangent Laplacian and angle defect.
- Error-driven local target edge length assignment from curvature.
- Per-vertex target length maintenance through split and collapse operations.
- Global remeshing loop: split, collapse, cleanup, flip, tangent smoothing.
- MLS projection to the original mesh.
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
  --epsilon 0 `
  --min-edge-length 0 `
  --max-edge-length 0 `
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
  -o cpp_adaiso/results/T0_T10/T0_iter1_project1.obj `
  --variant code `
  --epsilon 0 `
  --min-edge-length 0 `
  --max-edge-length 0 `
  --iters 1 `
  --preprocess 0 `
  --project 1 `
  --summary-json cpp_adaiso/results/T0_T10/T0_iter1_project1.json
```

## Parameters

- `--epsilon`: approximation-error tolerance. `0` selects `0.05 * meanEdgeLength`.
- `--min-edge-length`: minimum target edge length. `0` selects `0.5 * meanEdgeLength`.
- `--max-edge-length`: maximum target edge length. `0` selects `4/3 * meanEdgeLength`.
- `--mesh-scale`: retained for CLI compatibility; v3.0.0 does not use it for target edge length.
- `--variant code|paper`: controls split/collapse threshold conventions only.
- `--preprocess 0|1`: enables long-edge pre-splitting.
- `--iters`: global split/collapse/flip/smooth iterations.
- `--project 0|1`: enables MLS projection back to the original mesh.
- `--summary-json`: writes a JSON run summary.

The summary contains input/output vertex counts, face counts, `qavg`, `qmin`,
runtime, core parameters, sizing-field statistics, and per-iteration remeshing
statistics.

## Smoke Test

After building:

```powershell
cpp_adaiso/build/Release/adaiso_cli.exe cpp_adaiso/examples/pyramid.obj `
  -o cpp_adaiso/out.obj `
  --variant code `
  --epsilon 0 `
  --min-edge-length 0 `
  --max-edge-length 0 `
  --iters 2 `
  --preprocess 1 `
  --summary-json cpp_adaiso/summary.json
```

Expected acceptance:

- `cpp_adaiso/out.obj` exists.
- All face indices are valid.
- `summary.json` reports positive `qavg` and `qmin`, sizing statistics, and per-iteration statistics.
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
Version 3.0.0 intentionally diverges from CSF histogram sizing by using an
error-driven curvature size field.
