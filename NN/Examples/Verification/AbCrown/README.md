# AbCrown Verification Assets

The reusable Lean checker lives in `NN/Verification/Cert/AbCrownLeafCert.lean`.
This folder keeps only the compact offline sample artifact:

- `sample_abcrown_leaf_cert_v0_1.json`

Run it with:

```bash
lake exe verify -- abcrown-leaf
```

For real alpha beta CROWN runs, write the exported leaf certificate to an ignored local path such
as `/tmp/abcrown_leaf_cert.json` or `_external/abcrown_leaf_cert.json`, then pass that path to the
checker:

```bash
lake exe verify -- abcrown-leaf /tmp/abcrown_leaf_cert.json
```
