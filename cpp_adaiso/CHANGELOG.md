# Changelog

## 5.1.1

Feature update:
Exposes remeshing hysteresis factors for controlled experiments without
changing defaults.

- Added `--split-factor` and `--collapse-factor` CLI overrides.
- Added `AdaIsoConfig::split_factor_override` and
  `AdaIsoConfig::collapse_factor_override`; `0` keeps the current defaults.
- Summary JSON continues to report the effective split and collapse factors.

## 5.1.0

Breaking change:
Target sizing is now driven by an absolute geometric error tolerance instead of
a requested face budget.

- Added `principal-max` curvature mode using libigl principal curvature with
  `max(abs(k1), abs(k2))` as the raw curvature signal.
- The target field now uses CSF-smoothed principal curvature and
  `h_min = sqrt(6 * delta / kappa_max - 3 * delta^2)`.
- Removed `--target-faces` and `--target-size-mapping`.
- Added `--target-error-delta`; the default is automatically inferred from
  `kappa_max` and `0.5 * mean_edge_length`.
- Summary and target-field diagnostics now report delta, bbox diagonal,
  `kappa_max`, `h_min`, `h_max`, and estimated face count.

## 5.0.2

Behavior update:
Split, collapse, flip-length guards, and edge-ratio diagnostics now use a
single target metric edge length instead of separate endpoint min/max target
references.

- Added linear-target metric length evaluation,
  `M = length * (log(h_b) - log(h_a)) / (h_b - h_a)`, with equal-target
  fallback to `length / h`.
- Split uses `M > splitFactor()` and collapse uses `M < collapseFactor()`.
- Collapse validity and flip candidate checks use the same metric interval.
- Output edge diagnostics now report `target_metric_edge_length`.

## 2.8.1

Breaking change:
Target edge lengths now use percentile-normalized power mapping from CSF
curvature to target size.

- Replaced `saturatedTargetLengths` with `percentileTargetLengths`.
- The mapping uses fixed linearly interpolated `Q5` and `Q95` curvature
  quantiles, then evaluates
  `h = h_min + (h_max - h_min) * (1 - t)^target_size_gamma`.
- Degenerate percentile ranges fall back to the neutral
  `mean_edge_length * mesh_scale` target size clamped into `[h_min, h_max]`.
- Added `--export-all-stage-meshes` for every-iteration split/collapse/flip/smooth
  diagnostic snapshots. Iteration mesh export now also writes `iter000.ply`.

## 2.8.0

Breaking change:
Target edge lengths now use a saturated curvature-to-size mapping with
`target_size_gamma` instead of the direct linear helper.

- Replaced `linearTargetLengths` with `saturatedTargetLengths`.
- Added `--target-size-gamma` and `AdaIsoConfig::target_size_gamma`; default is
  `2.0`.
- `target_size_gamma = 1.0` reproduces the v2.7 linear response as a formula
  special case, without keeping the old mapping API.

## 2.7.0

Breaking change:
Target edge lengths now use direct linear mapping from the CSF curvature range
to `[target_min_factor, target_max_factor] * mean_edge_length * mesh_scale`.

- Removed empirical-quantile/log target mapping, target smoothing, and gradation
  projection configuration.
- `TargetLengthField` now stores real target lengths and interpolates `h`
  directly on the input surface.
- Target-field diagnostics now export one `target_h_heat.ply` and no longer
  write v2.6 staged target fields.

## 2.4.0

Breaking change:
Target edge lengths now use a continuous empirical-quantile mapping instead of
the old five-level curvature histogram assignment.

- Added `--target-min-factor`, `--target-max-factor`, and `--target-gamma`.
- Removed histogram bin and five-multiplier target assignment configuration.
- Constant CSF curvature fields now map to `mean_edge_length * mesh_scale`.
- Kept CSF screened smoothing, Gaussian kNN target lookup, and the remeshing
  loop behavior unchanged.

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
