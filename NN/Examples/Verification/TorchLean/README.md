# Native TorchLean Verification

This folder contains runnable workflows where the model is written in TorchLean, compiled to the
verifier IR, and checked by Lean bound propagation. These files should stay thin: reusable
compiler, IR, CROWN, and certificate logic belongs under `NN/Verification/TorchLean` or
`NN/MLTheory/CROWN`.

Run the maintained entry points through the unified verifier:

```bash
lake exe verify -- torchlean-ibp
lake exe verify -- torchlean-crown-ops
lake exe verify -- torchlean-transformer-ibp
lake exe verify -- torchlean-mlp-workflow --dtype float
```

The model training examples elsewhere in `NN/Examples/Models` exercise the CUDA eager runtime. The
workflows here are verifier workflows: after any training step, the parameters must be available as
Lean tensors so the verifier can compile and check the graph. Keep generated runtime logs,
checkpoints, and exported artifacts out of this directory; put them under an ignored `generated/`,
`outputs/`, or `_external/` directory if needed.
