# Check an alpha-beta-CROWN leaf report

Run this small, offline format check from the repository root:

```bash
scripts/lake.sh exe verify -- abcrown-leaf
```

The bundled report describes the box `[-1, 1] × [-1, 1]`, claims a lower bound of `1`, and gives
an unsafe threshold of `0`. The checker confirms that the leaf stays inside the root box, that the
leaves cover the root, `1 > 0`, and that the reported margin equals `1 - 0`. The expected output
is

```text
[artifact] Checked 1 leaves: ok=1, bad=0
[artifact] consistent: the leaves cover the root and every leaf clears its threshold.
[artifact] The lower bounds are the producer's claims; TorchLean did not recompute them.
```

This example teaches an **export/import consistency check**. It loads no network and does not
recompute the claimed lower bounds. A made-up bound can pass these checks, so acceptance alone is
not a robustness proof.
For a model-to-bound workflow, start with `scripts/lake.sh exe verify -- torchlean-ibp` instead.

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

To export terminal leaves from an instrumented verifier, call the adapter after its search:

```python
from scripts.verification.abcrown.export_leaf_artifact import write_abcrown_leaf_artifact

write_abcrown_leaf_artifact(
    root_lo=original_property_lo,
    root_hi=original_property_hi,
    leaves=terminal_leaves,
    out_path="leaf_artifact.json",
)
```

Use the original property box for a robustness or safety claim. When no root is supplied,
the converter infers an envelope from the leaves; that checks a different region.
If `out_path` is omitted, set `ABCROWN_ARTIFACT_OUT`. This is a TorchLean adapter convention;
setting it alone does not make an external alpha-beta-CROWN run export anything.
