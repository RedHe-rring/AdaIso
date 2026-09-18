# Version and Experiment Workflow

## 1. Purpose

AdaIso has multiple development lines and a large body of experimental output.
This workflow keeps three identities separate:

- **Code identity:** an immutable Git commit, optionally marked by a release or
  baseline tag.
- **Experiment identity:** a stable ID for one hypothesis and its run matrix.
- **Artifact identity:** the path or external location containing raw outputs.

The code may move backward or branch independently. Plans and conclusions must
remain readable without relying on whichever code branch is currently checked
out.

## 2. Worktree model

Long-lived code lines should exist in separate Git worktrees instead of being
checked out repeatedly in one directory. A target layout is:

```text
adaiso-records/        persistent plans, manifests, summaries, and conclusions
adaiso-v5/             worktree for develop/v5
adaiso-v6/             worktree for develop/v6
adaiso-artifacts/      ignored or external raw run outputs
```

The records location should eventually become either a small independent Git
repository or a dedicated records worktree. Until then, records added in this
repository must be deliberately propagated to each active development line.

Do not use an untracked file as the only durable copy of a plan or result. Do
not use stash as a long-term branch, archive, or experiment registry.

## 3. Naming

### 3.1 Software versions

Use semantic versions only for reproducible code milestones. A version change
means the source behavior or published interface has reached a named milestone.

Recommended annotated tags:

```text
baseline/v6.0.0
release/v5.4.0
```

Historical names may be retained, but their exact commit must be recorded.

### 3.2 Experiment IDs

Use IDs independent of the software version:

```text
E-2026-001-stage-a-continuous-energy
E-2026-002-feature-projection-ablation
```

An experiment may compare several commits, but every run cell must name exactly
one commit and configuration.

## 4. Experiment lifecycle

### 4.1 Plan

Before implementation or execution:

1. State the hypothesis and expected observation.
2. Name the baseline commit and comparison commit or feature flag.
3. Define fixed parameters and the variables being changed.
4. Define datasets, metrics, acceptance criteria, and stop conditions.
5. Create the manifest from `notes/experiment_manifest.template.yaml`.

### 4.2 Implement

Keep the algorithmic change reviewable. Prefer configuration flags for planned
ablations when both paths can remain clear and tested. Avoid accumulating many
near-identical experiment runner scripts.

### 4.3 Validate

Before a formal run:

1. Confirm `git status` is clean.
2. Build in the directory owned by the current worktree.
3. Run unit and smoke tests.
4. Record the compiler, generator, dependencies, and exact executable path.
5. Verify input checksums.

Exploratory dirty-tree runs are allowed, but they must be marked exploratory
and cannot serve as a reported baseline.

### 4.4 Run and conclude

Record every run, including failures. Preserve compact summaries in Git and raw
outputs outside Git. The conclusion must distinguish observation from inference
and list unresolved questions.

## 5. Ablation rules

An ablation should change one declared factor at a time or use a declared
factorial matrix. All other parameters must be frozen in the manifest.

The preferred structure is one reusable runner plus a data file describing the
matrix. Each output row should contain at least:

- experiment ID and run ID;
- source commit and dirty flag;
- dataset ID and checksum;
- complete parameter set and seed;
- build/toolchain identity;
- success or failure status;
- primary metrics and runtime;
- artifact location.

## 6. Commit and tag policy

Use small commits with a clear role:

```text
feat(remesh): add optional continuous size-energy guard
test(remesh): cover collapse energy acceptance
experiment(stage-a): run E-2026-001 matrix
docs(experiment): record E-2026-001 conclusion
```

Not every experiment commit needs a version bump or tag. Tag only the baselines
that must remain easy to reproduce or cite. Tags should be annotated and should
state the baseline purpose and validation status.

## 7. Artifact policy

Commit:

- source code and reusable runners;
- experiment plans and manifests;
- compact CSV/JSON summaries;
- conclusions and a limited number of publication-quality figures;
- dataset manifests and checksums where licensing permits.

Do not commit:

- CMake build trees and binaries;
- dependency checkouts;
- per-iteration meshes and raw diagnostic dumps;
- duplicate datasets or large benchmark outputs;
- temporary Office and inspection files.

Large artifacts must have a stable location referenced by the experiment
manifest. Important external data should have a checksum and provenance entry.

## 8. Returning to historical code

Prefer opening the relevant worktree. If a detached historical checkout is
needed, create a new temporary worktree at the required commit. Never overwrite
the current worktree or depend on an old build directory.

For comparison, use commit hashes and experiment summaries. A directory name or
the version string embedded in an old binary is supporting evidence, not the
primary identity.

## 9. Current migration checklist

- Preserve and review the existing stashes before dropping any of them.
- Establish separate v5 and v6 worktrees.
- Choose a persistent records repository or records worktree.
- Add immutable tags for the selected v5 and v6 baselines.
- Register local T-series and wine-glass inputs in a dataset manifest.
- Convert repeated v5 experiment generators into a reusable matrix runner when
  the next ablation series begins.
- Keep `compare/`, build trees, and raw `results/` outside normal Git status.
