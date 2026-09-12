# LiRPA Verification Artifacts

These files are small offline artifacts for TorchLean's LiRPA/IBP certificate checkers. The
producer side writes finite JSON objects: graph metadata, node ids, input boxes, intermediate
bounds, and the final property being checked. Lean then parses the artifact and recomputes or
checks the represented bound condition.

The bundled artifacts cover several graph shapes:

- `mlp_cert.json` from `scripts/verification/lirpa/export_mlp_cert.py`
- `cnn_cert.json` from `scripts/verification/lirpa/export_cnn_cert.py`
- `attention_softmax_cert.json` from `scripts/verification/lirpa/export_attention_cert.py`
- `gru_gate_cert.json` from `scripts/verification/lirpa/export_gru_cert.py`
- `transformer_encoder_cert.json` from `scripts/verification/lirpa/export_crown_cert.py`

Start with `lake exe verify -- lirpa-mlp`. No Python producer needs to run first: the JSON
fixtures are already bundled. The checker reconstructs the supported network fragment and compares
its propagated bounds with the reported result. A mismatch raises an error.

To regenerate the fixtures deliberately, run:

```bash
python3 scripts/verification/lirpa/export_mlp_cert.py
python3 scripts/verification/lirpa/export_cnn_cert.py
python3 scripts/verification/lirpa/export_attention_cert.py
python3 scripts/verification/lirpa/export_gru_cert.py
python3 scripts/verification/lirpa/export_crown_cert.py
```

This invokes the external producers and can replace the checked-in fixture files.

Or run a single checker through the unified verifier:

```bash
lake exe verify -- lirpa-mlp
lake exe verify -- lirpa-cnn
lake exe verify -- lirpa-attention
lake exe verify -- lirpa-gru
lake exe verify -- lirpa-encoder
```
