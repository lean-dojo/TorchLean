# VNN-COMP Verification Assets

`MnistFcVerify.lean` is a runnable checker for VNN-COMP style JSON exports, but large benchmark
snapshots do not belong in the source tree.

Expected local layout:

```text
_external/vnncomp/mnist_fc/model_weights.json
_external/vnncomp/mnist_fc/suite.json
_external/vnncomp/mnist_fc/alphas_crownobj.json   # optional
```

Run with:

```bash
lake exe verify -- vnncomp-mnistfc \
  --weights=_external/vnncomp/mnist_fc/model_weights.json \
  --suite=_external/vnncomp/mnist_fc/suite.json \
  --max=2
```

If you keep the artifacts somewhere else, pass `--weights=...`, `--suite=...`, and optionally
`--alphas=...`.

The source tree should not contain full VNN-COMP model dumps or suite exports. Keep only conversion
scripts and documentation here.
