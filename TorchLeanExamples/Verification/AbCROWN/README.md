# Check an alpha-beta-CROWN leaf report

Run this small, offline format check from the repository root:

```bash
scripts/lake.sh exe verify -- abcrown-leaf
```

The bundled report describes the box `[-1, 1] × [-1, 1]`, claims a lower bound of `1`, and gives
an unsafe threshold of `0`. The checker confirms that the leaf stays inside the root box, that the
leaves cover the root, that `1 > 0`, and that the reported margin equals `1 - 0`. The expected
output is:

```text
[artifact] Checked 1 leaves: ok=1, bad=0
[artifact] consistent: the leaves cover the root and every leaf clears its threshold.
[artifact] The lower bounds are the producer's claims; TorchLean did not recompute them.
```

`abcrown-leaf` is an **export/import consistency check**. It loads no network and does not
recompute the claimed lower bounds, so a made-up bound can pass and acceptance alone proves no
robustness. For a workflow that computes bounds from a model, start with
`scripts/lake.sh exe verify -- torchlean-ibp`.

The two fixtures show the format boundary:

- `example_raw_leaf_dump.json` uses the field names an external search might export.
- `sample_abcrown_leaf_artifact_v0_1.json` uses TorchLean's checked schema.

Convert the raw fixture and run the same checker:

```bash
python3 scripts/verification/abcrown/export_leaf_artifact.py \
  --input TorchLeanExamples/Verification/AbCROWN/example_raw_leaf_dump.json \
  --out _external/abcrown/leaf_artifact.json \
  --check
```

Vanilla alpha-beta-CROWN does not emit TorchLean's schema, so the converter accepts aliases such
as `x_L`, `x_U`, `lower_bounds`, and `thresholds`. A real run has to export its terminal leaf data
before conversion. The checker is `TorchLean/Verify/Certificate/AbCROWNLeaf.lean`.

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

Use the original property box for a robustness or safety claim. Without a root, the converter
infers an envelope from the leaves, and that checks a different region. If `out_path` is omitted,
the adapter writes to the path in `ABCROWN_ARTIFACT_OUT` and fails when that is unset. The
variable is a TorchLean adapter convention: setting it does not make an external
alpha-beta-CROWN run export anything.
