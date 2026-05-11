/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Analysis.Dropout
public import NN.Proofs.Analysis.Fft
public import NN.Proofs.Analysis.FftBridge
public import NN.Proofs.Analysis.InductiveProperties
public import NN.Proofs.Analysis.Lipschitz
public import NN.Proofs.Analysis.Normalization
public import NN.Proofs.Analysis.Softmax

/-!
# Analysis Proofs

Curated import surface for TorchLean's real-analysis and numerics-facing proof utilities.

This is the “math facts about the spec” layer. It should not contain executable CUDA checks,
empirical approximations, or model examples. A good rule of thumb:

- if the statement is a reusable theorem about `Tensor` norms, real-valued layer specs, softmax,
  dropout, normalization, or exact complex DFT algebra, it belongs here;
- if the statement is a runtime approximation theorem, a CUDA contract test, or a verifier artifact
  checker, it belongs in the corresponding runtime/verification/proof folder instead.

The split between these files is intentional:

- `Lipschitz` carries the reusable real-valued norm library (`‖·‖₂`, Cauchy-Schwarz, ReLU
  Lipschitzness, matrix/operator bounds, and composition lemmas).
- `InductiveProperties` carries structural tensor-induction patterns and dimension-lifting lemmas
  that depend on those norm facts.
- `Softmax`, `Dropout`, and `Normalization` are small layer-specific sanity theorems.
- `Fft` is pure mathlib `ℂ` DFT algebra with no TorchLean runtime dependency.
- `FftBridge` is the transport layer from TorchLean runtime FFT twiddle definitions to the exact
  `Fft` matrices. Keeping it separate prevents the pure DFT theorem file from importing the runtime
  stack.

Trust boundary: these are Lean theorems about TorchLean specs and mathlib objects. CUDA/cuFFT or
other native fast paths are tested against their contracts elsewhere; they are not proved by this
module.
-/

@[expose] public section
