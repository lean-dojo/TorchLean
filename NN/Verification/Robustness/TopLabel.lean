/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Context
public import NN.Spec.Core.Tensor.Core
public import NN.Tensor.Internal.Elab.TensorLiteral

/-!
# Certified label checks

Shared predicates for certified classification from output bounds.

The checker uses one rule: the claimed label's lower bound must be strictly above every other
class upper bound. Bounds use equally shaped tensors, including after JSON decoding.
-/

@[expose] public section

namespace NN.Verification.Robustness.TopLabel

open Spec TorchLean

/-- Check that the label's lower bound strictly exceeds every competing upper bound.
An out-of-range label or unordered comparison returns false. -/
def check {α : Type} [Storage α] [Context α] {n : Nat}
    (lo hi : Tensor α [n]) (label : Nat) : Bool :=
  if h : label < n then
    let y : Fin n := ⟨label, h⟩
    let loY := Tensor.getScalar lo y
    (List.finRange n).all fun i => i == y || Context.gtBool loY (Tensor.getScalar hi i)
  else
    false

end NN.Verification.Robustness.TopLabel
