# Workspace Inventory - 2026-09-18

This is a non-destructive snapshot made before reorganizing the AdaIso workflow.

## Git state

- Current branch: `develop/v6`
- Current commit: `3af46f8` (`test: record v6.0.0 baseline behavior`)
- v5 head: `3baccea` (v5.3.11 feature-aware Stage A projection)
- v5.3.7 code: `1ea06a8` (continuous size energy)
- v5 and v6 have diverged intentionally from an older common history.
- Four stashes exist. The newest stash contains mixed documentation, comparison
  script, and Python implementation changes and must be reviewed before removal.

## Removed local build identities

| Directory | Cached project version | Generator |
|---|---:|---|
| `cpp_adaiso/build2` | 5.3.11 | Ninja |
| `cpp_adaiso/build_analysis` | 2.12.0 | Visual Studio 17 2022 |
| `cpp_adaiso/build_v6` | 6.0.0 | Visual Studio 17 2022 |

These stale build directories were removed during cleanup. Future build
directories should belong to one worktree and should not be reused across code
lines.

## Large local directories

At inventory time:

- `compare/`: approximately 73.7 GB and 374,000 files.
- `cpp_adaiso/results/`: approximately 19.6 GB and 18,000 files.
- removed local build directories: approximately 172 MB combined.

The main `results/` archive was retained. Three explicitly temporary debug
directories totaling approximately 42.5 MB were removed:
`debug_high_curvature_matrix`, `debug_high_curvature_smoke`, and
`_debug_flip_dihedral`.

## Items requiring an explicit decision

- `notes/cpp_adaiso_v5_3_7_overall_flow_plan.md`: preserve as a tracked record
  after confirming whether it describes commit `1ea06a8` exactly.
- local T-series meshes and `wine_glass24_ascii.ply`: hashes are recorded in
  `notes/local_mesh_dataset_manifest.csv`; provenance still needs confirmation
  before deciding whether to use Git LFS or external artifact storage.
- `tools/colorize_csv.py`: review and either commit as a reusable tool or move to
  local-only utilities.
- newest stash: split into coherent commits or dedicated WIP branches; do not
  drop it during routine cleanup.

## Cleanup performed

- Added repository working rules in `AGENTS.md`.
- Added the version and experiment workflow.
- Added an experiment manifest template and commit-message template.
- Registered local benchmark meshes by relative path, size, and SHA-256.
- Extended ignore rules for local build variants, comparison checkouts, temporary
  tools, Office lock files, and generated inspection data.
- Removed stale build trees, the obsolete Python reproduction, weekly meeting
  files, and three explicitly named debug result directories.
