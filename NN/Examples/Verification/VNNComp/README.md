# VNN-COMP-Style MNIST-FC Artifacts

This folder documents TorchLean's current VNN-COMP-facing shape: exported JSON artifacts for a
small MNIST fully-connected benchmark family, checked by Lean through
`NN.Verification.VNNComp.MnistFC`.

VNN-COMP uses standard benchmark packages, network formats, property files, timeouts, and solver
reporting rules. TorchLean currently converts a benchmark instance into explicit JSON objects, then
runs a Lean checker over the exported weights, properties, and optional verifier metadata.

This is an advanced external-input workflow. The files below are not bundled, so the command will
not run from a fresh checkout alone. Start with `lake exe verify -- torchlean-ibp` for a self-contained
example. Once you have converted benchmark inputs, use this local layout:

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

If the artifacts live somewhere else, pass `--weights=...`, `--suite=...`, and optionally
`--alphas=...`.

The checked object is the exported suite item: network weights, input region, expected label or
margin property, and any optional bound data attached to the item. The external preparation step is
responsible for converting the original benchmark files into this JSON shape. Full benchmark dumps,
large model files, and solver outputs belong in `_external/` or another local data directory rather
than in git.

Acceptance concerns the supplied JSON representation. It does not check whether an external
converter faithfully translated the original network and property files.
