# AdaIso

**A C++17 Research Implementation of Curvature-Aware Isotropic Surface Remeshing**

AdaIso is a research codebase for isotropic surface remeshing guided by curvature-smoothed fields. It explores how curvature information can guide local target sizing and how remeshing operations can improve triangle mesh quality while maintaining geometric fidelity.

The project is implemented in C++17 and uses CMake for building.

## Features

- **Curvature processing:** Compute curvature signals and apply screened Laplacian smoothing.
- **Target sizing:** Derive local target sizes from curvature information, with configurable curvature signals and target face budgets.
- **Mesh operations:** Implement local remeshing operators, including edge splitting, collapsing, flipping, and vertex smoothing.
- **Mesh evaluation:** Report mesh statistics and quality metrics to support experimental analysis.
- **Automated tests:** Provide focused tests for curvature processing, target sizing, and local mesh operations.

## Repository Structure

- `cpp_adaiso/` — C++ implementation, build configuration, tests, and example meshes.

## Build

### Requirements

- A C++17-compatible compiler
- CMake 3.20 or later
- Eigen3 and nanoflann
- Libigl, discoverable by CMake

CMake can fetch Eigen3 and nanoflann if suitable installed packages are unavailable. Libigl must be installed and discoverable.

### Configure and Build

Run the following commands from the repository root:

```bash
cmake -S cpp_adaiso -B cpp_adaiso/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp_adaiso/build --config Release
```

### Run Tests

```bash
ctest --test-dir cpp_adaiso/build --output-on-failure
```

## Research Scope

AdaIso is an experimental implementation inspired by curvature-smoothed-field-based remeshing research. Its implementation choices and results should be evaluated through reproducible experiments rather than assumed to match the original method exactly.

Quantitative comparisons, runtime measurements, and detailed experimental conclusions will be documented after validation.
