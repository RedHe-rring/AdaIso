# Python AdaIso Reimplementation

This is a Python reimplementation of the main idea in Lv et al. 2024,
**Adaptively Isotropic Remeshing Based on Curvature Smoothed Field**.

It implements:

- OBJ/PLY loading through `trimesh`
- long-edge sub-remeshing
- curvature smoothed field (CSF)
- histogram-based local target edge lengths
- edge split, collapse, flip, and tangent smoothing
- optional MLS projection to the original surface
- basic mesh quality evaluation (`Qavg`, `Qmin`)

Run:

```powershell
python python_adaiso/run_adaiso.py input.obj -o output.obj --variant paper --mesh-scale 1 --preprocess
```

Useful options:

```powershell
--variant paper   # use paper-faithful thresholds and CSF curvature
--variant code    # use code-faithful histogram/threshold tendencies
--no-project      # skip MLS projection
--iters 5         # global remeshing iterations
--min-collapse-quality 0.05
```

For a quick smoke test:

```powershell
python python_adaiso/run_adaiso.py python_adaiso/examples/pyramid.obj `
  -o python_adaiso/examples/pyramid_remesh_code.obj `
  --variant code --mesh-scale 0.8 --iters 2 --preprocess
```

Evaluate an output mesh:

```powershell
python python_adaiso/eval_mesh.py output.obj
```

Notes:

- This is a clean Python implementation, not a line-by-line port of the original C++ pair-list topology code.
- For large meshes, pure Python edge collapse/flip is slower than the original C++ implementation.
- The default `paper` variant follows the equations and thresholds described in the paper. The `code` variant follows several tendencies observed in the public source code, including `1.33` split/flip upper threshold and `min` target length for collapse.
- Very small meshes are sensitive to `mesh_scale`; use the `code` variant or reduce `mesh_scale` for smoke tests.
