# PINN residual bounds

A physics-informed neural network (PINN, Raissi, Perdikaris and Karniadakis, arXiv:1711.10561)
is trained so that a PDE residual built from the network and its derivatives is small at sample
points. Here we bound that residual on an input box. Interval passes through the graph and its
tangent graphs enclose `u` and its first and second tangents along each axis, and CROWN can tighten
the bound on `u`. A small PDE language combines them into one residual interval, in host `Float`
with each endpoint of a sum, difference or product moved one binary64 value outward from the
nearest result. We read the tangents as derivatives, and no theorem relates a tangent graph to a
derivative.

- `Architecture.lean`: `PINN.Architecture`, its `model` graph and the `HiddenActivation` choices.
- `TorchLeanExamples/Verification/PINN.lean`: `referenceModel` (two tanh layers of width 16), the
  tangent graphs of `Derivatives.ofGraph`, `axisDerivativeBounds`, `finiteDifferenceResidual`,
  `parseCertificate` and `certTol`.
- `Certificate.lean`: the `pinn-cert` replay, with `parseModel` and `checkDomain`.
- `CLI.lean`: `pinn-cli`, with `parseFlags`, `computePrimsAt` and `loadWeightsOrDefault`.
- `PDEAST.lean`: the residual language `PINN.Expr` and its interval evaluator `PINN.Expr.eval`.
- `PDEParse.lean`: `parseExpr`, which reads residual strings such as `"uxx + uyy"` with the
  parser of `../ODE/ArithParse.lean`.
- `ResidualAffine.lean`: CROWN bounds for `u` (`crownUBoundsNodewise`, `crownUBoundsBackward`).
- `Dataset.lean`: sample points (`Sample`), `loadSection` and `containsWithTol`.
- `DatasetCheck.lean`: `pinn-dataset-check`, with `loadModel` and `checkSection`.
- `PyTorch/Load.lean`: `Checkpoint.ofJson?`, which reads a `layers.<i>.weight`/`.bias` state dict.
- `PyTorch/Model.lean`: `TorchLean.Examples.PINN.Checkpoint.model`, the graph of a loaded
  `PINN.Checkpoint`.
- `train_pinn.py`: PyTorch training of a candidate network, saved as a state dict.
- `export_pinn_cert.py`: writes `pinn_cert.json` for a tanh network with fixed weights.
- `import_burgers_shock_mat.py`: turns the Burgers shock data of Raissi et al. into a dataset.
- `pinn_cert.json`: the bundled certificate, for the residual `uxx` at 0.25, 0.5 and 0.75.
- `sample_dataset_1d.json`: nine (x, t) points in collocation, initial, boundary and data sections.

## Commands

```bash
scripts/lake.sh exe verify -- pinn-cert
scripts/lake.sh exe verify -- pinn-cli -- "u_t + u*u_x - 0.01*u_xx" 0.0 0.5 0.01
scripts/lake.sh exe verify -- pinn-dataset-check
```

`pinn-cert` reads `pinn_cert.json` unless it's given a path. A certificate's `model` must be a
state dict with one input and one output, and every point of `pinn.points` must lie in
`pinn.domain`. Lean rebuilds the network and, for each point `x`, runs IBP on boxes of radius
`eps` around `x - h`, `x` and `x + h`. It compares the exported intervals for `u`, the
finite-difference residual and the `pinn.pde` residual with its own, within `certTol` (1e-5). So
acceptance is a numerical tolerance test and no proof: an accepted certificate doesn't enclose the
exact-real residual or bound the error of a PDE solution. IBP bounds each tanh layer by [-1, 1], so
the bundled file has `u` within ±2.8, and the division by h² = 1e-4 widens the finite-difference
residual to ±112000.

`pinn-cli` takes the residual, then a box center and radius: `x eps` for a network with one input,
`x y eps` for two. Write the second axis `t` or `y` (`u_t` and `u_y` name the same derivative).
Without `--weights PATH` it bounds `referenceModel` with built-in parameters, and a weights file
that fails to load or has the wrong input size falls back to them with a warning.
`--method=ibp|crown-fwd|crown-bwd` picks the bound for `u` (`crown` means `crown-bwd`). With
`--split-depth=N` the box is cut into 2^N pieces in 1D or 4^N in 2D, and the hull of their bounds
is intersected with the unsplit bound.

`pinn-dataset-check` reads `sample_dataset_1d.json` unless it's given a path (first, or as
`--dataset=PATH`). It takes the first `--max` points (200) of each of the initial, boundary and
data sections and bounds the network on a box of radius `--eps` (0) around each `(x, t)`. It
counts the points whose `u` lies in the resulting interval, widened by `--tol` (1e-3) at each end.
Without `--weights` it uses `referenceModel 2` with demonstration parameters. It prints one line
per section and succeeds even when points miss, so add `--strict` when a miss should fail the
command. A pass says that the network's intervals contain the data, and nothing about the PDE.

## Training and regeneration

```bash
python3 TorchLeanExamples/Verification/PINN/train_pinn.py evolution
scripts/lake.sh exe verify -- pinn-cli -- --weights=_external/pinn/checkpoints/pinn1d.json \
  "u_t + u*u_x - 0.01*u_xx" 0.0 0.5 0.01
python3 TorchLeanExamples/Verification/PINN/export_pinn_cert.py
```

`evolution` trains on `(x, t)` in [-1, 1] × [0, 1] for Burgers' equation with `nu = 0.01` and the
initial condition `-sin(πx)`, and writes `_external/pinn/checkpoints/pinn1d.json`. `stationary`
trains on `(x, y)` in [-1, 1]² with the residual `uxx + uyy` and writes `pinn2d.json` next to it.
`--pde-expr`, `--bc-expr` and `--hidden-widths` change the problem and the network, and
`train_pinn.py export --in-dim 1 --out PATH` writes a freshly initialized 16-16 tanh model, or the
checkpoint of `--ckpt PATH`. Training needs PyTorch and certifies nothing. NumPy and SciPy are
needed by `import_burgers_shock_mat.py --mat PATH --out PATH`. Run all of these from the repository
root, since the scripts write `_external/pinn/checkpoints/` and `pinn_cert.json` by relative path.
