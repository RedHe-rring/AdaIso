# Changelog

## 3.1.0

Performance change:
Short-edge collapse now runs in batch passes instead of collapsing only one edge
per pass.

- Added batch collapse selection for one-ring-disjoint legal short edges.
- Reuses existing collapse validity checks for link condition, post-collapse edge length, and local quality.
- Rewrites affected faces once per batch and performs a single cleanup/remap of `target_lengths_`.
- `max_collapse_passes` now limits batch passes; `iteration_stats.collapse` still reports actual collapsed edge count.

## 3.0.0

Breaking change:
Remesher-level CSF histogram/multiplier target edge length control has been
replaced by an error-driven curvature size field.

- Added maximum absolute principal curvature estimation from cotangent Laplacian and angle defect.
- Added curvature-to-target-length mapping controlled by approximation tolerance and min/max target lengths.
- Added CLI parameters `--epsilon`, `--min-edge-length`, and `--max-edge-length`.
- Added sizing and per-iteration remeshing statistics to the run summary.
- Retuned auto sizing defaults to `epsilon=0.05*meanEdgeLength`, `min=0.5*meanEdgeLength`, and `max=4/3*meanEdgeLength`.
- Kept the standalone CSF sparse solver for internal tests, but removed it from remesher target sizing.

## 2.0.0

Breaking change:
CSF iterative smoothing and its CLI/configuration parameters have been removed.
CSF is now computed exclusively using a Dirichlet Laplace sparse solve.

- Replaced explicit CSF smoothing iterations with a per-component Dirichlet Laplace solve.
- Removed CSF iteration statistics from the run summary.
- Added CSF sparse-solve timing, residual, matrix-size, vertex-count, and success statistics.
