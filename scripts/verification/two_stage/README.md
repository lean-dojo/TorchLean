# Two-Stage TorchLean Verification

This folder contains the TorchLean pieces of a two stage Van der Pol verification workflow.
The Python scripts are producers and baselines; their outputs are not trusted until a Lean checker
parses and verifies the relevant property.

## What Belongs In Git

- Source code belongs here: Lean modules, Python producers, and small documentation.
- Regenerated weights and stage outputs should go under `_external/`, `/tmp`, or another ignored
  local output directory.
- Do not commit large stage checkpoints or generated JSON weight dumps unless they are deliberately
  promoted to a tiny reproducible fixture.

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

`scripts/verification/two_stage/cegis_van_stage2_python_baseline.py`

Runs a compact PyTorch Stage 2 reference loop using the same scalar loss and parameter pack order.
This is useful for comparing behavior with TorchLean Stage 2 code, not for producing a trusted
certificate.

```bash
python3 scripts/verification/two_stage/cegis_van_stage2_python_baseline.py \
  --weights _external/van_stage1_w100_bits.json \
  --stage2_rounds 5 \
  --candidates 8 \
  --pgd_steps 10
```

You can also print these commands through the verification asset catalog:

```bash
python3 scripts/verification/regenerate_assets.py --group two-stage
```

## Trust Boundary

The Stage 1 bit file is an imported parameter artifact. Exact bit export solves reproducibility, not
soundness. The soundness claim comes only from the TorchLean/Lean checker that consumes those
parameters and verifies the stated condition.
