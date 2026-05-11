/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Tensor.Core

/-!
# Metrics

TorchLean metrics helpers.

These are non-differentiable evaluation helpers for demos (e.g. accuracy).
-/

@[expose] public section


namespace Runtime
namespace Autograd
namespace TorchLean

open Spec
open Tensor

namespace Metrics

/-- Index of the maximum entry in a length-`n` vector, if `n > 0`.

This is a small evaluation helper (not differentiable). It is written against `Tensor` directly so
it can be used with multiple scalar types.
-/
def argmax? {α : Type} [LT α] [DecidableRel ((· > ·) : α → α → Prop)]
    {n : Nat} (y : Tensor α (.dim n .scalar)) : Option (Fin n) :=
  match y with
  | Tensor.dim f =>
      if h0 : 0 < n then
        let init : Fin n := ⟨0, h0⟩
        let (bestIdx, _bestVal) := (List.finRange n).foldl (fun (acc : Fin n × α) k =>
          let (_, bestVal) := acc
          let vk := match f k with | Tensor.scalar v => v
          if vk > bestVal then (k, vk) else acc
        ) (init, match f init with | Tensor.scalar v => v)
        some bestIdx
      else
        none

/-- Class index of a one-hot target (implemented as `argmax?`). -/
def classOfOneHot? {α : Type} [LT α] [DecidableRel ((· > ·) : α → α → Prop)]
    {n : Nat} (yOneHot : Tensor α (.dim n .scalar)) : Option (Fin n) :=
  argmax? (α := α) (n := n) yOneHot

/-- Compare predicted `argmax` against a one-hot target; returns `none` when `n = 0`. -/
def correctOneHot? {α : Type} [LT α] [DecidableRel ((· > ·) : α → α → Prop)]
    {n : Nat} (logits : Tensor α (.dim n .scalar)) (targetOneHot : Tensor α (.dim n .scalar)) :
    Option Bool := do
  let p ← argmax? (α := α) (n := n) logits
  let y ← classOfOneHot? (α := α) (n := n) targetOneHot
  pure (p = y)

end Metrics

end TorchLean
end Autograd
end Runtime
