/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.DirectedBackwardPublic
public import NN.Tensor.Internal.Laws.Sequence

/-!
# Nodewise rounded CROWN bounds

Sequencing the coordinate objectives produces a lower and upper affine row for every output.
If any sweep fails, the returned constant rows retain the output's IBP enclosure.
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

/-- A constant pair of affine maps retains an interval enclosure. -/
theorem boundsConst_encloses (hzero : value (0 : α) = 0)
    (n : Nat) (box : FlatBox α) (x y : Nat → ℝ)
    (hy : RowEncloses box box.dim y) :
    AffineRowsEnclose (boundsConst n box.dim box.lo box.hi) x y := by
  dsimp only [AffineRowsEnclose, boundsConst]
  intro i
  simpa only [Spec.get2_full, hzero, zero_mul, Finset.sum_const_zero,
    zero_add, read_fin] using hy.2 i

/-- The nodewise public API returns real affine enclosures for every coordinate. Both the
coordinate-sweep branch and the constant IBP fallback are covered. -/
theorem directedNodeBounds_encloses (hzero : value (0 : α) = 0)
    (hone : value (1 : α) = 1)
    {g : Graph} {ps : ParamStore α} {ibp : Array (Option (FlatBox α))}
    {ctx : AffineCtx} {dims : Nat → Nat} {v : Nat → Nat → ℝ}
    (point : GraphPoint g.nodes ps ibp ctx dims v)
    (output : Nat) (houtput : output < g.nodes.size)
    (hdim : dims output = g.nodes[output]!.outShape.size)
    {bounds : FlatAffineBounds α}
    (hresult : directedNodeBounds? g ps ctx ibp output = some bounds) :
    bounds.inDim = ctx.inputDim ∧ bounds.outDim = dims output ∧
      AffineRowsEnclose bounds (v ctx.inputId) (v output) := by
  by_cases hsupported : crownGraphSemanticsSupported (α := α) g ps = true
  · simp only [directedNodeBounds?, hsupported, ↓reduceIte,
      Option.bind_eq_bind, Option.pure_def] at hresult
    obtain ⟨node, hn, hresult⟩ := Option.bind_eq_some_iff.mp hresult
    have hnode : g.nodes[output]! = node := by
      obtain ⟨hindex, hget⟩ := Array.getElem?_eq_some_iff.mp hn
      simpa only [getElem!_pos (c := g.nodes) (i := output) hindex] using hget
    rw [hnode] at hdim
    let objective (i : Fin node.outShape.size) : FlatTensor α :=
      { n := node.outShape.size, v := Tensor.ofFn fun j => if i = j then 1 else 0 }
    cases hrows : Tensor.Internal.sequenceFinM
        (fun i : Fin node.outShape.size =>
          runDirectedBackwardObjective g ps ctx ibp output (objective i)) with
    | some rows =>
        simp only [objective] at hrows
        simp only [hrows, Option.some.injEq] at hresult
        subst bounds
        refine ⟨rfl, hdim.symm, ?_⟩
        intro i
        have hr := Tensor.Internal.sequenceFinM_get_of_eq_some hrows i
        have hs := runDirectedBackwardObjective_encloses hzero point output houtput
          (objective i) hdim.symm hr
        rw [hdim] at hs
        simpa [objective, dot, affineValue, read_fin, apply_ite, hzero, hone,
          Tensor.matrix, Spec.get2] using hs
    | none =>
        simp only [objective] at hrows
        simp only [hrows] at hresult
        obtain ⟨entry, he, hresult⟩ := Option.bind_eq_some_iff.mp hresult
        obtain ⟨box, hb, hresult⟩ := Option.bind_eq_some_iff.mp hresult
        cases Option.some.inj hresult
        rw [hb] at he
        have hlookup : ibp[output]! = some box := by
          obtain ⟨hindex, hget⟩ := Array.getElem?_eq_some_iff.mp he
          simpa only [getElem!_pos (c := ibp) (i := output) hindex] using hget
        have hy := point.ibp_encloses output houtput box hlookup
        exact ⟨rfl, hy.1, boundsConst_encloses hzero ctx.inputDim box
          (v ctx.inputId) (v output) (by simpa only [hy.1] using hy)⟩
  · simp [directedNodeBounds?, hsupported] at hresult

end

end NN.MLTheory.CROWN.Graph.DirectedBackward
