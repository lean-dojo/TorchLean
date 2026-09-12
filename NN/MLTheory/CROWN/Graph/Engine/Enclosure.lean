/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.BoundOps.Lawful
public import NN.MLTheory.CROWN.Graph.Engine.Base

/-!
# Boxes that enclose a real vector

The engine in `NN.MLTheory.CROWN.Graph.Engine.Base` propagates boxes; this file relates a box to the
real vector it is supposed to contain, and shows that a lawful scalar transfer stays sound when it
is applied coordinatewise to a flat box.

The relation is stated in ℝ, which is why it is not part of the engine: the IBP and CROWN passes,
the certificate checker, and everything above them run on backend endpoints only.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Graph

open _root_.Spec _root_.TorchLean
open _root_.TorchLean.Tensor
open NN.MLTheory.CROWN

variable {α : Type} [TorchLean.Storage α] [Context α] [BoundOps α]

/--
The real value represented by each coordinate of `x` lies between the interpreted endpoints of
`B`. This is the semantic relation used to connect executable endpoint arithmetic to the real graph
semantics.
-/
def EnclosesReal [LawfulBoundOps α]
    (B : FlatBox α) (x : Tensor ℝ [B.dim]) : Prop :=
  ∀ i,
    LawfulBoundOps.toReal (TorchLean.Tensor.getScalar B.lo i) ≤ TorchLean.Tensor.getScalar x i ∧
      TorchLean.Tensor.getScalar x i ≤ LawfulBoundOps.toReal (TorchLean.Tensor.getScalar B.hi i)

/-- Dimension-aware enclosure of a real vector by a backend box. -/
def EnclosesRealValue [LawfulBoundOps α] {n : Nat}
    (B : FlatBox α) (x : Tensor ℝ [n]) : Prop :=
  ∃ h : B.dim = n, EnclosesReal B (h.symm ▸ x)

/-- A lawful scalar transfer remains sound when applied coordinatewise to a flat graph box. -/
theorem boxUnaryEnclosure?_enclosesReal [LawfulBoundOps α] [NonlinearBoundOps α]
    (f : ℝ → ℝ) (enclose : α → α → Option (α × α))
    (henclose : UnaryEnclosure (α := α) f enclose) (B : FlatBox α)
    (x : Tensor ℝ [B.dim]) (hx : EnclosesReal B x)
    {out : FlatBox α} (hout : boxUnaryEnclosure? (α := α) enclose B = some out) :
    EnclosesRealValue out (Tensor.mapSpec f x) := by
  simp only [boxUnaryEnclosure?] at hout
  obtain ⟨bounds, hbounds, hout⟩ := Option.bind_eq_some_iff.mp hout
  have hpoint := Internal.traverseFin_eq_some_iff.mp hbounds
  have houtEq :
      out =
        { dim := B.dim
          lo := Tensor.ofFn fun i => (bounds i).1
          hi := Tensor.ofFn fun i => (bounds i).2 } := by
    exact (Option.some.inj hout).symm
  subst out
  refine ⟨rfl, ?_⟩
  intro i
  have hscalar := henclose (hpoint i) (hx i).1 (hx i).2
  simpa [EnclosesReal, Tensor.mapSpec] using hscalar

end NN.MLTheory.CROWN.Graph
