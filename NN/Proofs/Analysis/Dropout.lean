/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Dropout
public import NN.Runtime.Context

/-!
# Dropout analysis properties

TorchLean splits stochastic training-mode dropout into two pieces:

- a mask/seed producer, treated as non-differentiated data in autograd proofs, and
- a deterministic tensor map once the mask or inference probability is fixed.

This file records small spec-level identities for the deterministic inference map. The fixed-mask
training-mode derivative infrastructure lives with the autograd tape-node proofs.

Reference: Srivastava et al., 2014, “Dropout: A Simple Way to Prevent Neural Networks from
Overfitting”.
-/

@[expose] public section

namespace Proofs

open _root_.Spec
open _root_.Spec.Tensor

noncomputable section

/--
Mapping the identity function over a tensor is the identity.

This is private because the public dropout theorem below is the intended reusable fact; if we need a
general tensor-map identity elsewhere, it should move to the tensor proof library rather than stay
hidden in the dropout file.
-/
private theorem map_spec_id {s : Shape} (x : Tensor ℝ s) :
    Spec.Tensor.mapSpec (α := ℝ) (s := s) (fun x => x) x = x := by
  induction x with
  | scalar x =>
      rfl
  | dim g ih =>
      simp [Spec.Tensor.mapSpec, ih]

/--
Deterministic dropout inference scaling is the identity when `p = 0`.

Inference dropout multiplies activations by the keep/dropout scaling factor from the spec. At zero
dropout probability that factor is `1`, so the whole tensor is unchanged.
-/
theorem dropout_inference_spec_p0_eq_id {s : Shape} (x : Tensor ℝ s) :
    Spec.dropoutInferenceSpec (α := ℝ) (s := s) (p := (0 : ℝ)) x = x := by
  -- Reduce to pointwise scaling by `1`, then simplify.
  simp [Spec.dropoutInferenceSpec, Spec.Tensor.scaleSpec, map_spec_id]

end

end Proofs
