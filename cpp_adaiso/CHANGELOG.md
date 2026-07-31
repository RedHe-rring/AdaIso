# Changelog

## 5.3.8

Behavior update:
Adds a guarded closest-point projection after Stage A and makes collapse
placement feature-agnostic by default.

- `feature_aware_collapse` now defaults to `false`; the CLI override remains
  available for compatibility and ablation.
- Added `--stage-a-projection <0|1>` to run one guarded closest-point
  projection pass after Stage A and before Stage B.
- The projection candidate is the closest point on the original input triangle
  surface, queried with an AABB; it is accepted only when the local patch stays
  valid, target log energy does not increase beyond tolerance, and the local
  minimum-angle safety bound is preserved.
- Summary JSON reports Stage A projection candidates, accepted projections,
  rejection reasons, and displacement statistics.

## 5.3.7

Behavior update:
Uses a continuous target-field distance energy for local size-field decisions.

- Edge-ratio diagnostics now report short/long violation rates and
  `target_log_energy = mean(log(target_metric_edge_length)^2)`.
- Stage A collapse now accepts candidates only when local target log energy
  decreases; split candidate collection still uses the existing long-edge
  threshold.
- Stage B flip and relocation keep their quality objectives, but their
  size-field protection now uses target log energy instead of hinge violation.
- Summary JSON reports Stage A final short violation, long violation, and
  target log energy.

## 5.3.6

Behavior update:
Adds one conflict-region collapse mini-round inside each Stage A collapse pass.

- Collapse still uses the v5.3.5 energy-only acceptance rule; split, Stage B,
  and the public CLI are unchanged.
- After a normal collapse batch, candidates skipped by batch conflict seed one
  local retry round that rebuilds topology and re-evaluates current short
  edges adjacent to the conflict region.
- Summary JSON reports conflict retry mini-rounds, retry candidates, retry
  accepted collapses, and retry rejected candidates.

## 5.3.5

Behavior update:
Removes the collapse max-ratio slack constraint added in v5.3.4.

- Collapse now uses local target-metric violation decrease as its only
  size-field acceptance condition.
- Angle safety, Stage A ordering, batch selection, CLI options, and rejection
  diagnostics remain unchanged.
- `size_stage_collapse_rejected_by_target_length` now means the collapse did
  not reduce local target-metric violation.

## 5.3.4

Behavior update:
Replaces the collapse target-length hard cap with a local target-energy
acceptance rule.

- Collapse candidates now compare local target-metric violation before and
  after the operation instead of rejecting any candidate that creates one edge
  above the split threshold.
- A small internal max-ratio slack prevents collapse from creating much longer
  local edges while still allowing the split/collapse pair to reduce local size
  field error.
- Stage A policy, batch selection, CLI options, and collapse rejection
  diagnostics are otherwise unchanged.

## 5.3.3

Behavior update:
Adds collapse rejection reason diagnostics for the staged size-field fitting
phase.

- Collapse now reports why short-edge candidates were rejected, split into
  boundary, non-manifold, batch conflict, link condition, degenerate patch,
  target length, angle, and batch duplicate categories.
- The v5.3.3 diagnostics do not change collapse acceptance rules; they only
  make Stage A recovery failures measurable in summary JSON.
- Project version metadata updated to `5.3.3`.

## 5.3.2

Behavior update:
Adds Stage A diagnostics and ablation policies for staged remeshing.

- Default Stage A policy is `split_collapse_no_guard`, restoring the v5.3.0
  size-field fitting baseline after the v5.3.2 ablation showed the weak split
  guard did not improve quality or violation.
- Added experimental `--stage-a-policy` values for
  `split_collapse_no_guard`, `split_collapse_weak_guard`, and
  `collapse_split_weak_guard`.
- Experimental weak split angle guard rejects only invalid patches or
  candidates below the internal split minimum-angle floor.
- Summary JSON now reports Stage A long/short edge candidates, accepted/rejected
  operations, and final Stage A violation and angle statistics.

## 5.3.1

Behavior update:
Stabilizes the staged remesh size-field fitting phase.

- Stage A now runs collapse before split, so existing over-dense edges are
  recovered before adding new samples for long edges.
- Long-edge split candidates now use a local angle guard and are rejected when
  they would lower the affected patch minimum angle.
- Split candidates also reject newly created or worsened obtuse local patches,
  while still allowing already obtuse patches when the maximum angle does not
  increase.

## 5.3.0

Behavior update:
Splits remeshing into a size-field fitting stage followed by a quality repair
stage.

- Stage A now runs split and collapse only, using minimum-angle safety as an
  admissibility constraint while fitting the target size field.
- Stage B now runs flip and relocation only, using local minimum-angle repair as
  the quality objective under a target-metric violation constraint.
- Iteration summaries now include a `stage` label, `relocation_count`, and
  aggregate size/quality stage operation counts.
- Stage mesh snapshots now use stage-qualified names such as
  `size_iter001_after_split.ply` and `quality_iter001_after_smooth.ply`.

## 5.2.1

Behavior update:
Replaces mandatory minimum-angle improvement with a minimum-angle safety
constraint for local operation acceptance.

- Collapse, flip, and relocation now reject candidates only when they would
  create a local minimum angle below `min(before_min_angle, angle_floor)`.
- Flip no longer accepts an operation solely because it improves minimum angle;
  target-metric violation and valence remain the deciding objectives after the
  angle safety check.
- Relocation now allows non-degrading smoothing candidates instead of requiring
  strict one-ring minimum-angle improvement.

## 5.2.0

Behavior update:
Adds minimum-angle-first acceptance checks for local remeshing operations.

- Replaced collapse-only `min_quality_after_collapse` with the internal
  `AdaIsoConfig::min_angle_floor_degrees` safety floor.
- Collapse now compares local minimum angle before and after a candidate
  operation instead of only checking a triangle-quality floor.
- Flip now treats minimum-angle improvement as the primary acceptance objective;
  target-metric violation and valence are only secondary tie-breaks.
- Tangent smoothing and MLS relocation now rollback candidates that do not
  improve one-ring minimum angle or that worsen local target-metric violation.
- Mesh summaries and iteration summaries now report `min_angle_degrees`,
  `angle_lt_10_pct`, and `angle_lt_20_pct`.

## 5.1.5

Behavior update:
Replaces the previous `kappa_ref + h_ref_factor` auto-delta sizing with an
input-resolution-aware CSF curvature sizing field.

- Auto delta now comes from an area-weighted quantile of per-vertex local
  tolerances estimated from CSF curvature and incident-edge median length.
- Added `--auto-delta-quantile` and `AdaIsoConfig::auto_delta_quantile`.
- Removed `--target-error-h-ref-factor`, `--auto-delta-kappa-ref`, and the
  corresponding `AdaIsoConfig`/diagnostic fields.
- Target-field diagnostics now export `local_edge_median`, `local_delta`, and
  `local_delta_heat.ply`.

## 5.1.4

Feature update:
Exposes the curvature reference used by auto-delta target-error sizing for
controlled experiments without changing defaults.

- Added `--auto-delta-kappa-ref <max|p99|p95>` and
  `AdaIsoConfig::target_error_curvature_reference`.
- Target-error sizing now reports both the selected `kappa_ref` and observed
  `kappa_max`.
- Summary JSON and target-field diagnostics now report `kappa_reference`,
  `kappa_ref`, and `kappa_max`.

## 5.1.3

Feature update:
Exposes the auto-delta reference edge-length factor for controlled experiments
without changing defaults.

- Added `--target-error-h-ref-factor` and
  `AdaIsoConfig::target_error_h_ref_factor`.
- Replaced the hard-coded `0.5 * mean_edge_length` auto-delta reference length
  with the configured factor.
- Summary JSON now reports `target_error_h_ref_factor`.

## 5.1.2

Feature update:
Exposes the target sizing upper clamp for controlled experiments without
changing defaults.

- Added `--target-h-max-factor` and `AdaIsoConfig::target_h_max_factor`.
- Replaced the hard-coded `5.0 * mean_edge_length` upper target-length clamp
  with the configured factor.
- Summary JSON and target-field diagnostics now report `h_max_factor`.

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
