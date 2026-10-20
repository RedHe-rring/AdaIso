# Supplementary Research Materials

This directory contains supporting scripts and historical experiment records. The active C++ implementation, build configuration, public headers, tests, and example meshes are located at the repository root. Start with [`../README.md`](../README.md) to build and run AdaIso.

## Contents

### `tools/`

Python utilities for targeted diagnostics and experiments, including high-curvature region analysis, ROI mesh export, target-field heatmap export, and high-curvature detail experiment reporting.

These scripts are experiment-specific rather than a general-purpose command-line interface. Before running one, inspect its input/output paths and Python dependencies; some may need local path or configuration changes.

### `修改计划与实验结果/`

Historical development plans, version-specific experiment reports, diagnostic scripts, generated figures, and presentation materials from the v2.x development process.

## How to interpret these materials

- They are **not required** to configure, build, test, or run the core C++ project.
- Results belong to the code version and parameter settings recorded in each report. They should not be treated as results of the current version unless the report explicitly says so.
- A report documenting a completed run is not, by itself, evidence that the experiment can be reproduced from a clean environment today.
- Before comparing reported metrics, verify the input meshes, build version, parameters, evaluation method, and sampling settings.

## Curation status

This directory is retained temporarily for research traceability while the public portfolio is being organized. The intended final portfolio should keep only reusable diagnostic tools and a small number of representative, reproducible experiment reports. Redundant plans, one-off outputs, and materials that are not suitable for public release can be moved out of this branch after review.

Do not use this directory as the main project entry point; use the root README instead.