# Floating-point deep dives

These examples execute and prove facts about the floating-point behavior of TorchLean's scalar and
tensor operations. [Verify/Floats](../../../TorchLean/Verify/Floats/README.md) documents the
FloatLib bridges and APIs they use.

| File | What it shows |
| --- | --- |
| `EffectiveRounding.lean` | One tensor addition under `FP32` and under `Binary32`. On a finite result the IEEE bridge and effective rounding give the same canonical mantissa and exponent. |
| `Float32Semantics.lean` | A compact MLP forward pass and VJP under native `Float32` and configured binary32, with their largest absolute difference. |
| `GraphNumericalCertificate.lean` | Range certificates with binary32 interval endpoints for a four-node scalar graph and a two-layer MLP graph, rejection of a tampered artifact and bit-level binary32 replay. |
| `RoundingBugs.lean` | Floating-point bugs in standard formulas, with kernel-checked counterexamples and theorems about the stable formulas TorchLean computes. |
| `LibraryBugs.lean` | Kernel-checked rounding failures in formulas from PyTorch, JAX, Hugging Face Transformers, TRL, Stable-Baselines3 and diffusers, each paired with a formula for the same quantity that does not fail. |
| `ProbBugs/SurvivalComplement.lean` | The same for formulas from PyTorch, PyTorch Geometric and TensorFlow Probability. |
| `LossBugs/FocalPower.lean` | The same for gradients and decay factors from torchvision, xLSTM and PyTorch. |

`Stable/` holds worked proofs of `TorchLean.Verify.Stable`, whose programs round a real target once
and carry the certificate of `Once.claim` in every IEEE binary format.

| File | What it shows |
| --- | --- |
| `Stable/Exp.lean` | The sigmoid, the soft-margin loss gradient and the Weibull survival function rounded once, and the soft-margin gradient repaired where `LossBugs.SoftMargin.lossBackward` returns zero in its band. |
| `Stable/K3.lean` | TRL's k3 KL estimator rounded once and positive in its band, where TRL's formula and its `expm1` rewrite are rejected. |
| `Stable/Variance.lean` | The population and sample variance rounded once, `+0` at equal inputs, where PyG's one-pass formula returns `-2` in binary32. |
