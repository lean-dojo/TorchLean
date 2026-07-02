# TorchLean Dependency Audit

Inspired by Li, Peng, Severini, and Shafto, "The Network Structure of Mathlib" (arXiv:2604.24797).

## Summary

- Modules: `1157`
- Import edges: `3927`
- Internal import edges: `3392`
- Public imports: `3613`
- Private imports: `314`
- Critical-path length over internal imports: `121`
- Findings: `0` (`0` errors, `0` warnings)
- Lean files: `1157`
- Lean source lines: `299996`
- Declaration headers: `12326`
- Theorem/lemma headers: `2171`

## Top Fan-In Modules

- `NN`: `78` incoming imports
- `NN.Spec.Core.Tensor`: `39` incoming imports
- `NN.Spec.Core.TensorOps`: `36` incoming imports
- `NN.MLTheory.CROWN.Graph`: `34` incoming imports
- `NN.Spec.Core.Context`: `32` incoming imports
- `NN.Spec.Core.TensorReductionShape`: `31` incoming imports
- `NN.Spec.Layers.Activation`: `31` incoming imports
- `NN.Floats.IEEEExec.Exec32`: `27` incoming imports
- `NN.Spec.Module.SpecModule`: `24` incoming imports
- `NN.MLTheory.CROWN.Core`: `23` incoming imports

## Top Fan-Out Modules

- `NN.CI.All`: `333` imports
- `NN.Entrypoint.Proofs`: `42` imports
- `NN.Spec.Module`: `28` imports
- `NN.Spec.Models`: `26` imports
- `NN.Examples.Zoo`: `23` imports
- `NN.MLTheory.API`: `23` imports
- `NN.Tests.Runtime.Floats.ModelsCheck`: `19` imports
- `NN.Proofs.Autograd.Coverage`: `18` imports
- `NN.Spec.Layers`: `18` imports
- `NN.API.Public.Facade.Base.ModelZoo`: `17` imports

## Layer Edges

- `NN.API` -> `NN.API`: `365`
- `NN.Runtime` -> `NN.Runtime`: `348`
- `NN.Spec` -> `NN.Spec`: `315`
- `NN.Proofs` -> `NN.Proofs`: `282`
- `NN.MLTheory` -> `NN.MLTheory`: `222`
- `NN.Floats` -> `NN.Floats`: `168`
- `NN.Verification` -> `NN.Verification`: `162`
- `NN.Examples` -> `NN.Examples`: `143`
- `NN.CI` -> `NN.Spec`: `88`
- `NN.MLTheory` -> `NN.Spec`: `76`
- `NN.CI` -> `NN.MLTheory`: `69`
- `NN.Tests` -> `NN.Tests`: `61`
- `NN.Proofs` -> `NN.Spec`: `60`
- `NN.Tests` -> `NN.Runtime`: `52`
- `NN.Examples` -> `NN`: `50`
- `NN.CI` -> `NN.Floats`: `49`
- `NN.Entrypoint` -> `NN.Proofs`: `42`
- `NN.API` -> `NN.Runtime`: `39`
- `NN.Runtime` -> `NN.Spec`: `39`
- `NN.API` -> `NN.Spec`: `36`

## Findings

No findings.
