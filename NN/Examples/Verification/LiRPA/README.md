# LiRPA Verification Fixtures

These files are compact offline fixtures for checking TorchLean's graph IBP certificate
checker. The reusable checker is `NN.Verification.Cert.IBPCert`; the Lean files here define small
model fixtures and call that checker.

The JSON files are intentionally small and reproducible:

- `mlp_cert.json` from `export_mlp_cert.py`
- `cnn_cert.json` from `export_cnn_cert.py`
- `attention_softmax_cert.json` from `export_attention_cert.py`
- `gru_gate_cert.json` from `export_gru_cert.py`
- `transformer_encoder_cert.json` from `export_crown_cert.py`

Regenerate and check them with:

```bash
python3 scripts/verification/regenerate_assets.py --group lirpa --run
lake exe verify -- all
```

Do not add large model dumps here. Larger LiRPA/auto LiRPA experiments should write to `_external/`,
`/tmp`, or a documented generated data directory.
