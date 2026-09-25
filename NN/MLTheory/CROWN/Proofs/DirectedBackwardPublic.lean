/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.DirectedBackwardFallback

/-!
# Public rounded backward bounds

The public objective result encloses its real objective whether it comes from the directed
reverse sweep or from the output-box fallback.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Graph.DirectedBackward

open Spec TorchLean NN.IR
open TorchLean.Tensor
open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Graph.Internal
open scoped BigOperators

noncomputable section

variable {α : Type} [Storage α] [Context α] [BoundOps α] [LawfulBoundOps α]

local notation "value" => LawfulBoundOps.toReal (α := α)

/-- Real coordinate enclosure by the two stored affine rows. -/
def AffineRowsEnclose (bounds : FlatAffineBounds α) (x y : Nat → ℝ) : Prop :=
  ∀ i : Fin bounds.outDim,
    (∑ j : Fin bounds.inDim, value (Spec.get2 bounds.loAff.A i j) * x j.val) +
        value (bounds.loAff.c.getScalar i) ≤ y i.val ∧
      y i.val ≤
        (∑ j : Fin bounds.inDim, value (Spec.get2 bounds.hiAff.A i j) * x j.val) +
          value (bounds.hiAff.c.getScalar i)

/-- A scalar pair gives the corresponding one-row enclosure. -/
theorem scalarBounds_enclose {n : Nat} {lower upper : AffineVec α n 1}
    {x : Nat → ℝ} {z : ℝ}
    (h : affineValue lower (fun i => x i.val) ≤ z ∧
      z ≤ affineValue upper (fun i => x i.val)) :
    AffineRowsEnclose { inDim := n, outDim := 1, loAff := lower, hiAff := upper }
      x (fun _ => z) := by
  intro i
  have hi : i = 0 := Subsingleton.elim _ _
  subst i
  exact h

/-- The public rounded objective API returns a sound scalar affine enclosure, including its
output-box fallback. The exact-reassociation branch has a separate arithmetic contract. -/
theorem runCROWNBackwardObjective_encloses (hzero : value (0 : α) = 0)
    (hrounded : BoundOps.supportsExactAffineReassociation (α := α) = false)
    {g : Graph} {ps : ParamStore α} {ibp : Array (Option (FlatBox α))}
    {ctx : AffineCtx} {dims : Nat → Nat} {v : Nat → Nat → ℝ}
    (point : GraphPoint g.nodes ps ibp ctx dims v)
    (output : Nat) (houtput : output < g.nodes.size) (obj : FlatTensor α)
    (hdim : obj.n = dims output) {bounds : FlatAffineBounds α}
    (hresult : runCROWNBackwardObjective g ps ctx ibp output obj = some bounds) :
    bounds.inDim = ctx.inputDim ∧ bounds.outDim = 1 ∧
      AffineRowsEnclose bounds (v ctx.inputId)
        (fun _ => dot (dims output) (fun i => value (getAtOrZero obj.v [i])) (v output)) := by
  by_cases hsupported : crownGraphSemanticsSupported (α := α) g ps = true
  · simp only [runCROWNBackwardObjective, hsupported, hrounded, ↓reduceIte,
      Bool.false_eq_true] at hresult
    cases hr : runDirectedBackwardObjective g ps ctx ibp output obj with
    | some pair =>
        rcases pair with ⟨lower, upper⟩
        simp only [hr, Option.some.injEq] at hresult
        subst bounds
        exact ⟨rfl, rfl, scalarBounds_enclose
          (runDirectedBackwardObjective_encloses hzero point output houtput obj hdim hr)⟩
    | none =>
        simp only [hr] at hresult
        cases hlo : objectiveFromOutputBox .lower ibp output ctx.inputDim obj with
        | none => simp [hlo] at hresult
        | some lower =>
            cases hhi : objectiveFromOutputBox .upper ibp output ctx.inputDim obj with
            | none => simp [hlo, hhi] at hresult
            | some upper =>
                simp only [hlo, hhi, Option.some.injEq] at hresult
                subst bounds
                have hb (box : FlatBox α) (hbox : ibp[output]! = some box) :
                    RowEncloses box obj.n (v output) := by
                  rw [hdim]
                  exact point.ibp_encloses output houtput box hbox
                have hl := objectiveFromOutputBox_encloses hzero .lower ibp output
                  ctx.inputDim obj (v output) hb hlo (fun i => v ctx.inputId i.val)
                have hu := objectiveFromOutputBox_encloses hzero .upper ibp output
                  ctx.inputDim obj (v output) hb hhi (fun i => v ctx.inputId i.val)
                dsimp only at hl hu
                rw (occs := .pos [1]) [hdim] at hl hu
                exact ⟨rfl, rfl, scalarBounds_enclose ⟨hl, hu⟩⟩
  · simp [runCROWNBackwardObjective, hsupported] at hresult

end

end NN.MLTheory.CROWN.Graph.DirectedBackward
