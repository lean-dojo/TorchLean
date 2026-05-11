# PINN Verification Assets

Reusable Lean code for PINN verification lives under `NN/Verification/PINN`.
This folder contains only small checked assets. Python producers live under `scripts/verification/pinn/`:

- `pinn_cert.json`: compact certificate fixture checked by `lake exe verify -- pinn-cert`.
- `sample_dataset_1d.json`, `sample_dataset_2d.json`: compact dataset fixtures for
  `lake exe verify -- pinn-dataset-check`.
- `export_pinn_cert.py`: regenerates the compact certificate fixture.
- `train_pinn_1d.py`, `train_pinn_2d.py`: train local PINN checkpoints/weight JSONs.
- `export_pinn_weights.py`: converts a PyTorch checkpoint or fresh model to TorchLean JSON.
- `import_burgers_shock_mat.py`: converts the external Burgers `.mat` dataset to JSON.

Generated checkpoints and trained weight dumps belong in `checkpoints/`, `/tmp`, or another ignored
output directory. They should not be committed as source fixtures.

Useful commands:

```bash
python3 scripts/verification/regenerate_assets.py --group pinn-small --run
python3 scripts/verification/regenerate_assets.py --group pinn-train --run
lake exe verify -- pinn-cert
lake exe verify -- pinn-dataset-check
```
