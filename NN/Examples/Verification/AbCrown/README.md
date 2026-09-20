# Check an alpha-beta-CROWN leaf report

Run this small, offline format check from the repository root:

```bash
lake exe verify -- abcrown-leaf
```

The bundled report describes the box `[-1, 1] × [-1, 1]`, claims a lower bound of `1`, and gives
an unsafe threshold of `0`. The checker confirms that the leaf stays inside the root box,
`1 > 0`, and the reported margin equals `1 - 0`. The expected summary is
`Checked 1 leaves: ok=1, bad=0`.

This example teaches an **export/import sanity check**. It loads no network and does not recompute
the claimed lower bound. It also does not check that a collection of leaves covers the whole input
region. A made-up bound can pass these checks, so acceptance alone is not a robustness proof.
For a model-to-bound workflow, start with `lake exe verify -- torchlean-ibp` instead.

The two fixtures show the format boundary:

- `example_raw_leaf_dump.json` uses the field names an external search might export.
- `sample_abcrown_leaf_artifact_v0_1.json` uses TorchLean's checked schema.

Convert the raw fixture and run the same checker:

```bash
python3 scripts/verification/abcrown/export_leaf_artifact.py \
  --input NN/Examples/Verification/AbCrown/example_raw_leaf_dump.json \
  --out _external/abcrown/leaf_artifact.json \
  --check
```

The converter accepts aliases such as `x_L`, `x_U`, `lower_bounds`, and `thresholds`; vanilla
alpha-beta-CROWN does not directly emit TorchLean's schema. Real runs must export their terminal
leaf data before conversion. The checker implementation is
`NN/Verification/Cert/AbCrownLeafCert.lean`.
