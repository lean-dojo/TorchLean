# Exporting alpha-beta-CROWN Leaves for TorchLean's Consistency Check

TorchLean's Lean checker consumes `abcrown_leaf_artifact_v0_1` JSON files. The external
alpha-beta-CROWN project does not natively emit that TorchLean schema, so this folder contains the
small producer-side bridge. The Lean check confirms that the leaves cover the root box and clear
their thresholds; it does not recompute the lower bounds.

Convert a raw terminal-domain dump:

```bash
python3 scripts/verification/abcrown/export_leaf_artifact.py \
  --input NN/Examples/Verification/AbCrown/example_raw_leaf_dump.json \
  --out _external/abcrown/leaf_artifact.json \
  --check
```

For an external verifier integration, import the writer and call it after the verifier has terminal
verified leaves:

```python
from scripts.verification.abcrown.export_leaf_artifact import write_abcrown_leaf_artifact

write_abcrown_leaf_artifact(
    root_lo=root_lo,
    root_hi=root_hi,
    leaves=terminal_leaves,
)
```

For real robustness or safety claims, pass the original input-property box as `root_lo`/`root_hi`.
If no root is supplied, the converter can infer the componentwise envelope of the represented
leaves, which is useful for standalone structural fixtures but is not a substitute for the original
property domain.

If `out_path` is omitted, the helper writes to `ABCROWN_ARTIFACT_OUT`. That environment variable is a
TorchLean helper convention; setting it does not modify alpha-beta-CROWN unless the external run
imports or calls this helper.
