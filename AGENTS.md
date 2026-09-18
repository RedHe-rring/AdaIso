# Repository Working Rules

These rules apply to human and automated work in this repository.

## Before changing code

1. Record the current branch, commit, and `git status`.
2. Identify the target code line and the experiment or issue being addressed.
3. Do not switch branches with a dirty worktree. A stash is temporary transport,
   not durable storage.
4. Use separate Git worktrees for long-lived v5, v6, and record-keeping work.

## Versions and experiments

1. A software version names a reproducible code milestone.
2. An experiment ID names a hypothesis, configuration, and set of runs.
3. Do not bump the software version solely to distinguish parameter sweeps or
   ablation cells.
4. Stable baselines and releases must be identified by an annotated tag.
5. Every experiment record must include the exact Git commit and whether the
   source tree was dirty.

## Commits

1. Keep one algorithmic purpose per commit.
2. Separate implementation, generated results, and broad documentation cleanup
   when they can be reviewed independently.
3. Use commit subjects in the form `type(scope): summary`, where useful types
   include `feat`, `fix`, `experiment`, `test`, `docs`, and `chore`.
4. Before committing code, run the tests appropriate to the affected code path.
5. Never commit build outputs, caches, raw run directories, or large local
   comparison checkouts.

## Experiments

1. Start from a clean, committed source state for any result used in comparison.
2. Create an experiment manifest from `notes/experiment_manifest.template.yaml`.
3. Record inputs and their checksums, full parameters, toolchain, random seeds,
   commands, output location, metrics, failures, and conclusion.
4. Keep plans, manifests, compact summaries, and conclusions in Git.
5. Keep raw meshes, per-iteration dumps, binaries, and large plots in local or
   external artifact storage referenced by the manifest.
6. Prefer one parameterized runner and an explicit ablation matrix over copying
   a new runner script for every experiment.

## Builds and results

1. A build directory belongs to one worktree. Do not run a binary left by a
   different branch or worktree.
2. Rebuild after changing commits and record the executable's source commit in
   the experiment manifest.
3. Do not delete or relocate historical results until their manifest and compact
   summary have been preserved.

The detailed process is documented in
`notes/versioning_and_experiment_workflow.md`.
