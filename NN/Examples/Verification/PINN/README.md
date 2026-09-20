# PINN Verification Assets

PINN examples are scientific ML verification artifacts. A physics-informed neural network is trained
so that its outputs satisfy data constraints and differential-equation residual constraints. The
verification question is sharper than the training curve: which residual, dataset region, weights,
and bounds were checked?

Reusable Lean code for PINN verification lives under `NN/Verification/PINN`. The files in this
folder are compact checked fixtures:

- `pinn_cert.json`: a small residual certificate checked by `lake exe verify -- pinn-cert`.
- `sample_dataset_1d.json`, `sample_dataset_2d.json`: pointwise dataset artifacts for
  `lake exe verify -- pinn-dataset-check`. The dataset command is diagnostic by default: it prints
  contained/missed counts and exits successfully unless `--strict` is passed.

Python producers live under `scripts/verification/pinn/`:

- `export_pinn_cert.py`: regenerates the compact certificate artifact.
- `train_pinn_1d.py`, `train_pinn_2d.py`: train local PINN checkpoints and TorchLean weight JSONs.
- `export_pinn_weights.py`: converts a PyTorch checkpoint or fresh model to TorchLean JSON.
- `import_burgers_shock_mat.py`: converts the external viscous Burgers `.mat` reference dataset to
  JSON.

Check the bundled residual fixture first; no training is required:

```bash
lake exe verify -- pinn-cert
```

This recomputes the fixture's solution and residual intervals at its stated sample points and
compares the reported intervals. It does not establish that an arbitrary trained network solves
the PDE everywhere. `pinn-dataset-check` is a separate containment diagnostic: read the counts,
or pass `--strict` when missed points should make the command fail.

The producer commands are for recreating data, not prerequisites for the bundled check:

```bash
python3 scripts/verification/pinn/export_pinn_cert.py
python3 scripts/verification/pinn/train_pinn_1d.py --steps 25
python3 scripts/verification/pinn/train_pinn_2d.py --steps 25
```

For Burgers-style examples, the PDE has the form

```text
u_t + u u_x = nu u_xx
```

with samples on a space-time grid. TorchLean's PINN path keeps the equation text, constants,
weights, sample points, and residual bounds explicit so the scientific claim can be separated from
the training run that produced the candidate network.
