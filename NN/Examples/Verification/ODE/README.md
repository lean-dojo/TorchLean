# ODE Verification Fixtures

The reusable ODE verifier lives under `NN/Verification/ODE`. This folder contains compact JSON
fixtures consumed by the unified verifier CLI.

The JSON files here are compact fixtures for small checks and examples:

- `sample_ode_cert.json`
- `sin_cert.json`
- `logistic_trivial_cert.json`
- `logistic_learned_cert.json`
- `zero_mlp.json`, `one_mlp.json`, `zero_siren.json`
- `logistic_lower_learned.json`, `logistic_upper_learned.json`

Check a bundled certificate directly with:

```bash
lake exe verify -- ode --cert=NN/Examples/Verification/ODE/sample_ode_cert.json
```

Regenerate the curated ODE assets with:

```bash
python3 scripts/verification/regenerate_assets.py --group ode --run
```

These files are small enough to keep as public fixtures. Larger learned ODE/PINN weights should be
stored outside git and passed to `lake exe verify -- ode ...` explicitly.
