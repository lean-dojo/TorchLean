/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.DirectedBackwardNodeBounds

/-!
# Directed evaluation of backward affine bounds

The returned affine functions have a real interpretation. Their executable interval evaluator
rounds each product, sum, and final constant outwards before returning a scalar box.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Graph.DirectedBackward

open Spec TorchLean NN.IR
open TorchLean.Tensor
open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Graph.Internal
open LayerNormDirected (value_min2 value_max2)
open scoped BigOperators

noncomputable section

variable {α : Type} [Storage α] [Context α] [BoundOps α] [LawfulBoundOps α]

local notation "value" => LawfulBoundOps.toReal (α := α)

/-- Directed affine interval evaluation encloses each real affine row. -/
theorem affineEvalOnBox_encloses (hzero : value (0 : α) = 0)
    {n m : Nat} (aff : AffineVec α n m) (box : Box α (.dim n .scalar))
    (x : Fin n → ℝ)
    (hx : ∀ j, value (box.lo.getScalar j) ≤ x j ∧ x j ≤ value (box.hi.getScalar j))
    (i : Fin m) :
    value ((aff.evalOnBox box).lo.getScalar i) ≤
        (∑ j, value (Spec.get2 aff.A i j) * x j) + value (aff.c.getScalar i) ∧
      (∑ j, value (Spec.get2 aff.A i j) * x j) + value (aff.c.getScalar i) ≤
        value ((aff.evalOnBox box).hi.getScalar i) := by
  let lower (j : Fin n) := BoundOps.min2
    (BoundOps.mulDown (Spec.get2 aff.A i j) (box.lo.getScalar j))
    (BoundOps.mulDown (Spec.get2 aff.A i j) (box.hi.getScalar j))
  let upper (j : Fin n) := BoundOps.max2
    (BoundOps.mulUp (Spec.get2 aff.A i j) (box.lo.getScalar j))
    (BoundOps.mulUp (Spec.get2 aff.A i j) (box.hi.getScalar j))
  have ht (j : Fin n) :
      value (lower j) ≤ value (Spec.get2 aff.A i j) * x j ∧
        value (Spec.get2 aff.A i j) * x j ≤ value (upper j) := by
    have h := intervalMul_encloses
      (le_refl (value (Spec.get2 aff.A i j))) (le_refl (value (Spec.get2 aff.A i j)))
      (hx j).1 (hx j).2
    simpa only [directedIntervalMul, value_min2, value_max2, min_self, max_self,
      lower, upper] using h
  have hs := sum_encloses hzero lower upper
    (fun j => value (Spec.get2 aff.A i j) * x j) ht
  simp only [AffineVec.evalOnBox, Tensor.getScalar_dim]
  exact ⟨(LawfulBoundOps.addDown_le _ _).trans (add_le_add hs.1 le_rfl),
    (add_le_add hs.2 le_rfl).trans (LawfulBoundOps.le_addUp _ _)⟩

/-- Evaluating a pair of real affine enclosures on an enclosing input box retains enclosure. -/
theorem affineBoundsEval_encloses (hzero : value (0 : α) = 0)
    (bounds : FlatAffineBounds α) (box : FlatBox α) (x y : Nat → ℝ)
    (hx : RowEncloses box bounds.inDim x) (hy : AffineRowsEnclose bounds x y) :
    RowEncloses
      { dim := bounds.outDim
        lo := (bounds.evalOnFlatBox box hx.1).lo
        hi := (bounds.evalOnFlatBox box hx.1).hi }
      bounds.outDim y := by
  obtain ⟨n, lo, hi⟩ := box
  have hd := hx.1
  change n = bounds.inDim at hd
  subst n
  have hinput (j : Fin bounds.inDim) :
      value (lo.getScalar j) ≤ x j.val ∧ x j.val ≤ value (hi.getScalar j) := by
    simpa only [read_fin] using hx.2 j
  refine ⟨rfl, ?_⟩
  intro i
  have hl := affineEvalOnBox_encloses hzero bounds.loAff
    { lo := lo, hi := hi } (fun j => x j.val) hinput i
  have hu := affineEvalOnBox_encloses hzero bounds.hiAff
    { lo := lo, hi := hi } (fun j => x j.val) hinput i
  simpa only [read_fin, FlatAffineBounds.evalOnFlatBox, FlatBox.getScalarBox,
    FlatBox.loAsDim, FlatBox.hiAsDim, Tensor.cast_shape_rfl] using
      And.intro (hl.1.trans (hy i).1) ((hy i).2.trans hu.2)

/-- The public scalar-box evaluator encloses every point enclosed by its affine argument. -/
theorem evalBackwardObjectiveBox_encloses (hzero : value (0 : α) = 0)
    (bounds : FlatAffineBounds α) (xB : FlatBox α) (inputDim : Nat)
    (x y : Nat → ℝ) (hx : RowEncloses xB bounds.inDim x)
    (hy : AffineRowsEnclose bounds x y) {result : FlatBox α}
    (hresult : evalBackwardObjectiveBox? bounds xB inputDim = .ok result) :
    RowEncloses result 1 y := by
  by_cases hi : bounds.inDim = inputDim
  · by_cases hxDim : xB.dim = inputDim
    · by_cases ho : bounds.outDim = 1
      · simp only [evalBackwardObjectiveBox?, hi, hxDim, ho, ↓reduceDIte] at hresult
        have he := Except.ok.inj hresult
        subst result
        have h := affineBoundsEval_encloses hzero bounds xB x y hx hy
        cases bounds with
        | mk n m lower upper =>
            dsimp only at ho
            subst m
            simpa only [FlatAffineBounds.evalOnFlatBoxAsDim, Tensor.cast_shape_rfl] using h
      · simp [evalBackwardObjectiveBox?, hi, hxDim, ho] at hresult
    · simp [evalBackwardObjectiveBox?, hi, hxDim] at hresult
  · simp [evalBackwardObjectiveBox?, hi] at hresult

/-- The complete rounded objective workflow returns an interval containing its real output
objective, including all directed arithmetic in both propagation and final evaluation. -/
theorem backwardObjectiveBox_encloses (hzero : value (0 : α) = 0)
    (hrounded : BoundOps.supportsExactAffineReassociation (α := α) = false)
    {g : Graph} {ps : ParamStore α} {ibp : Array (Option (FlatBox α))}
    {ctx : AffineCtx} {dims : Nat → Nat} {v : Nat → Nat → ℝ}
    (point : GraphPoint g.nodes ps ibp ctx dims v)
    (xB : FlatBox α) (hx : RowEncloses xB ctx.inputDim (v ctx.inputId))
    (output : Nat) (houtput : output < g.nodes.size) (obj : FlatTensor α)
    (hdim : obj.n = dims output) {result : FlatBox α}
    (hresult : backwardObjectiveBox? g ps ctx ibp xB output obj = .ok result) :
    RowEncloses result 1
      (fun _ => dot (dims output) (fun i => value (getAtOrZero obj.v [i])) (v output)) := by
  unfold backwardObjectiveBox? at hresult
  cases hb : runCROWNBackwardObjective g ps ctx ibp output obj with
  | none => simp [hb] at hresult
  | some bounds =>
      simp only [hb] at hresult
      have hs := runCROWNBackwardObjective_encloses hzero hrounded point
        output houtput obj hdim hb
      exact evalBackwardObjectiveBox_encloses hzero bounds xB ctx.inputDim _ _
        (by simpa only [hs.1] using hx) hs.2.2 hresult

end

end NN.MLTheory.CROWN.Graph.DirectedBackward
