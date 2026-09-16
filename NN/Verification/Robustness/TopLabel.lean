/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

import Mathlib.Tactic.Positivity.Finset
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
open TorchLean.Tensor

/-- Strict label certificate over any indexed lower/upper bounds. -/
def strictTopLabelBy {α : Type} (n label : Nat)
    (lo hi : Fin n → α) (gt : α → α → Bool) : Bool :=
  if h : label < n then
    let y : Fin n := ⟨label, h⟩
    let loY := lo y
    (List.finRange n).all fun i => i == y || gt loY (hi i)
  else
    false

/-- Check a label directly from tensor lower/upper bounds. -/
def certifiesLabelFromTensorBounds {α : Type} [TorchLean.Storage α] [Context α] {n : Nat}
    (lo hi : Tensor α [n]) (label : Nat) : Bool :=
  strictTopLabelBy n label
    (fun i => Tensor.getScalar lo i)
    (fun i => Tensor.getScalar hi i)
    Context.gtBool

end NN.Verification.Robustness.TopLabel
