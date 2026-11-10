/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.CROWN.BoundOps.Binary32 -- shake: keep
public import TorchLean.Verify.CROWN
public import TorchLean.Tensor.Conversion

/-!
# CROWN Graph

Flattened input regions and affine bounds for the bound passes on graphs of registered operators
(`TorchLean.CROWN.Op`). `FlatBox.lInfBox` flattens a center and a radius into a box, rounding both
endpoints outward, and `FlatAffineBounds` holds the lower and upper affine forms of a node output
with respect to one flattened input. The module does not depend on the IR.

Main declarations: `FlatBox.lInfBox`, `FlatBox.lInfBall`, `FlatAffineBounds`,
`FlatAffineBounds.evalOnFlatBox`.

References:
- auto_LiRPA: Xu et al.,
  "Automatic Perturbation Analysis for Scalable Certified Robustness and Beyond",
  NeurIPS 2020, arXiv:2002.12920 (https://arxiv.org/abs/2002.12920), Sec. 3.2.
- CROWN: Zhang et al.,
  "Efficient Neural Network Robustness Certification with General Activation Functions",
  NeurIPS 2018, arXiv:1811.00866 (https://arxiv.org/abs/1811.00866), Corollary 3.3.
-/

@[expose] public section


open TorchLean.Shape (size)

namespace TorchLean.CROWN

open _root_.TorchLean.Tensor

namespace FlatBox

/-- Flatten a center/radius pair, rounding both interval endpoints outward. -/
def lInfBox {α : Type} [Storage α] [Context α] [BoundOps α] {s : Shape}
    (center radius : Tensor α s) : FlatBox α :=
  { dim := Shape.size s
    lo := (Tensor.zipWith (α := α) BoundOps.subDown center radius).reshape [_]
      (Shape.size_vector _).symm
    hi := (Tensor.zipWith (α := α) BoundOps.addUp center radius).reshape [_]
      (Shape.size_vector _).symm }

/-- The `ℓ∞` ball of radius `eps` around `center`, as a flat box rounded outward. -/
def lInfBall {α : Type} [Storage α] [Context α] [BoundOps α] {s : Shape}
    (center : Tensor α s) (eps : α) : FlatBox α :=
  lInfBox (α := α) center (Tensor.full (α := α) s eps)

/-- Flattening an `ℓ∞` ball preserves the number of tensor entries as its dimension. -/
@[simp] theorem lInfBall_dim {α : Type} [Storage α] [Context α] [BoundOps α]
    {s : Shape} (center : Tensor α s) (eps : α) : (lInfBall center eps).dim = s.size := by
  simp [lInfBall, lInfBox]

end FlatBox

end TorchLean.CROWN

namespace TorchLean.CROWN.Graph

open _root_.TorchLean.Tensor
open Std

/-- Lower and upper affine bounds on a node output in the flattened value of a chosen input node. -/
structure FlatAffineBounds (α : Type) [Storage α] [Context α] where
  /-- Flattened input dimension. -/
  inDim  : Nat
  /-- Flattened output dimension. -/
  outDim : Nat
  /-- Lower affine bound form. -/
  loAff  : Linear α inDim outDim
  /-- Upper affine bound form. -/
  hiAff  : Linear α inDim outDim

/-- Affine bounds are equal when their dimensions and both affine forms are. -/
instance {α : Type} [Storage α] [Context α] [DecidableEq α] :
    DecidableEq (FlatAffineBounds α)
  | ⟨n, m, lo, hi⟩, ⟨n', m', lo', hi'⟩ =>
    if h : n = n' ∧ m = m' then by
      obtain ⟨rfl, rfl⟩ := h
      exact decidable_of_iff (lo = lo' ∧ hi = hi') (by simp)
    else isFalse fun e =>
      h ⟨congrArg FlatAffineBounds.inDim e, congrArg FlatAffineBounds.outDim e⟩

namespace FlatAffineBounds

variable {α : Type} [Storage α] [Context α] [BoundOps α]

/--
Output box of a pair of affine bounds over a flat input box: the lower endpoints come from the lower
form and the upper endpoints from the upper form. CROWN concretizes its linear bounds this way.

On an `ℓ∞` ball of radius `ε` around `x₀`, Corollary 3.3 of Zhang et al. puts the upper end of a
row `Λ` at `Λ x₀ + ε ‖Λ‖₁` plus its constant. We evaluate on the box endpoints with `IBP.linear`, so
every product rounds outward and the box need not be a ball (`Directed.affineBoundsEval_encloses`).
-/
def evalOnFlatBox (bounds : FlatAffineBounds α) (xB : FlatBox α)
    (hIn : xB.dim = bounds.inDim) : Box α (.dim bounds.outDim .scalar) :=
  { lo := (bounds.loAff.evalOnFlatBox xB hIn).lo
    hi := (bounds.hiAff.evalOnFlatBox xB hIn).hi }

/-- `evalOnFlatBox`, with the output box cast to dimension `m` along `hOut : bounds.outDim = m`. -/
def evalOnFlatBoxAsDim (bounds : FlatAffineBounds α) (xB : FlatBox α)
    (hIn : xB.dim = bounds.inDim) {m : Nat} (hOut : bounds.outDim = m) :
    Box α (.dim m .scalar) :=
  let out := bounds.evalOnFlatBox xB hIn
  { lo := Tensor.castShape out.lo (congrArg (fun extent => Shape.dim extent .scalar) hOut)
    hi := Tensor.castShape out.hi (congrArg (fun extent => Shape.dim extent .scalar) hOut) }

end FlatAffineBounds

end TorchLean.CROWN.Graph
