# Changelog

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
