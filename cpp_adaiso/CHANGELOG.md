# Changelog

## 2.3.0

Breaking change:
CSF now uses a screened Laplace sparse solve instead of Dirichlet harmonic
interpolation.

- Replaced the fixed-anchor Dirichlet CSF system with `(M + alpha h^2 L)c = M r`.
- Added `--csf-alpha` and `AdaIsoConfig::csf_smoothing_alpha`; default is `0.5`.
- Unified raw CSF curvature to mean vertex-normal angle for all variants.
- Removed fixed/free CSF vertex statistics from the public summary.
- Removed the old Dirichlet solver entry point and anchor-selection helpers.

## 2.2.0

Feature update:
Smooths the target edge-length field by default and adds stability metrics for
nearest-vs-Gaussian target lookup experiments.

- Added `nearest` and `gaussian` target lookup modes; `gaussian` is the new default.
- Added Gaussian kNN interpolation controls for target lookup.
- Added per-iteration edge-ratio statistics and violation rates to JSON output.
- Documented that Gaussian kNN is locally continuous for a fixed neighbor set,
  while truncated kNN does not imply strict global continuity.

## 2.1.1

Feature update:
Adds engineering output and evaluation support without changing the AdaIso
remeshing algorithm.

- Added ASCII PLY output and extension-based mesh writer dispatch.
- Added per-global-iteration statistics to the JSON summary.
- Added optional area-sampled Chamfer Distance evaluation against a GT mesh.

## 2.1.0

Release update:
Collapse now supports batch processing within each collapse pass. The pass
selects non-conflicting short edges, collapses them together, and keeps the
existing local validity checks.

- Changed `max_collapse_passes` semantics from "up to 10 single-edge collapses"
  to "up to 10 batch collapse passes".
- Added regression coverage proving one collapse pass can collapse multiple
  disconnected components.
- Added v2.1.0 experiment plan and result records.

## 2.0.1

Documentation hotfix:
Updated version metadata and documentation notes only. Algorithm behavior and
the recorded v2.0.0 implementation plan and experiment results are unchanged.

## 2.0.0

Breaking change:
CSF iterative smoothing and its CLI/configuration parameters have been removed.
CSF is now computed exclusively using a Dirichlet Laplace sparse solve.

- Replaced explicit CSF smoothing iterations with a per-component Dirichlet Laplace solve.
- Removed CSF iteration statistics from the run summary.
- Added CSF sparse-solve timing, residual, matrix-size, vertex-count, and success statistics.
