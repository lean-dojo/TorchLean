# LiRPA Verification Artifacts

These files are small offline artifacts for TorchLean's LiRPA/IBP certificate checkers. The
producers write JSON containing a graph name, input box and reported output bounds. Lean parses
each artifact and recomputes the bounds for the corresponding graph.

The bundled artifacts cover several graph shapes:

- `mlp_cert.json`: two-layer MLP
- `cnn_cert.json`: convolution and linear head
- `attention_softmax_cert.json`: softmax and value projection
- `gru_gate_cert.json`: sigmoid and tanh gates
- `transformer_encoder_cert.json`: transformer-like graph with a final LayerNorm

Start with `scripts/lake.sh exe verify -- lirpa-mlp`. No Python producer needs to run first: the JSON
fixtures are already bundled. The checker reconstructs the supported network fragment and compares
its propagated bounds with the reported result. A mismatch raises an error.

To regenerate the fixtures deliberately, run:

```bash
python3 scripts/verification/lirpa/export_cert.py all
```

This can replace the checked-in fixture files. To regenerate just one, replace `all` with
`mlp`, `cnn`, `attention`, `gru`, or `transformer`. Use `--out-dir PATH` to keep the bundled
fixtures untouched.

Or run a single checker through the unified verifier:

```bash
scripts/lake.sh exe verify -- lirpa-mlp
scripts/lake.sh exe verify -- lirpa-cnn
scripts/lake.sh exe verify -- lirpa-attention
scripts/lake.sh exe verify -- lirpa-gru
scripts/lake.sh exe verify -- lirpa-encoder
```
