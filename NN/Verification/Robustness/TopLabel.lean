/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Algebra.Order.Field.Basic
import Mathlib.Tactic.NormNum.Inv
import Mathlib.Tactic.NormNum.Pow
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
def strictTopLabelBy {α : Type} [Max α] (n label : Nat)
    (lo hi : Fin n → α) (gt : α → α → Bool) : Bool :=
  if h : label < n then
    let y : Fin n := ⟨label, h⟩
    let loY := lo y
    let maxOther? :=
      (List.finRange n).foldl (fun (acc : Option α) i =>
        if i = y then acc
        else
          match acc with
          | none => some (hi i)
          | some m => some (max m (hi i))) none
    match maxOther? with
    | none => true
    | some m => gt loY m
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
