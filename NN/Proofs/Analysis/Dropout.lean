/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Dropout
public import Mathlib.Basic.Real.Basic

/-!
# Dropout analysis properties

TorchLean splits stochastic training-mode dropout into two pieces:

- a mask/seed producer, treated as non-differentiated data in autograd proofs, and
- a deterministic tensor map once the mask or inference probability is fixed.

This file records the spec-level identity for the deterministic inference map. The fixed-mask
training-mode derivative infrastructure lives with the autograd tape-node proofs.

Reference: Srivastava et al., 2014, “Dropout: A Simple Way to Prevent Neural Networks from
Overfitting”.
-/

@[expose] public section

namespace Proofs

open Spec TorchLean
open TorchLean.Tensor

noncomputable section

/-- Evaluation-mode dropout is the identity for every configured training probability. -/
theorem dropoutInferenceSpec_eq_id {s : Shape} (p : ℝ) (x : Tensor ℝ s) :
    Spec.dropoutInferenceSpec (α := ℝ) (s := s) p x = x := by
  rfl

end

end Proofs
