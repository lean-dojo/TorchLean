# Two-Stage TorchLean Verification

This folder contains the TorchLean pieces of a two stage Van der Pol verification workflow.
The Python producer exports a trained seed; TorchLean consumes its parameters in Stage 2.

## Files

- The Python producer lives here; Lean workflows live under `NN/MLTheory/CROWN/Lyapunov/TwoStage`.
- Regenerated weights and stage outputs go under `_external/`, `/tmp`, or another local output
  directory.
- Tiny reproducible fixtures can live here when a checker or doc page uses them directly.

## Scripts

`scripts/verification/two_stage/export_van_stage1_bits.py`

Trains a compact PyTorch Stage 1 controller/Lyapunov seed and exports exact IEEE-754 binary32 bit
patterns. The bit export avoids JSON decimal conversion error when TorchLean reconstructs Float32 parameters.

```bash
python3 scripts/verification/two_stage/export_van_stage1_bits.py \
  --width 100 \
  --steps 50 \
  --out _external/van_stage1_w100_bits.json
```

## Trust Boundary

The Stage 1 bit file is an imported parameter artifact. Exact bit export solves reproducibility, not
soundness. The soundness claim comes only from the TorchLean/Lean checker that consumes those
parameters and verifies the stated condition.
