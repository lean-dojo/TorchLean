/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.GraphCertSoundness.Main.Extraction
public import NN.MLTheory.CROWN.Proofs.GraphCertSoundness.IntervalLemmas

/-!
# Binary Matrix Product Enclosures

The runtime and certificate semantics share a flat coordinate map for batch broadcasting and
vector promotion. Each product is enclosed by its four endpoint products, then accumulated in
the same order as the value evaluator. Empty contractions and empty output shapes need no
positivity assumptions.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Graph.CertSoundness

open Spec TorchLean TorchLean.Tensor
open NN.MLTheory.CROWN

noncomputable section

/-- Zero-extended reads preserve enclosure, including coordinates outside the stored vector. -/
private theorem enclosed_flat_read {B : FlatBox ℝ} {v : Val}
    (h : EnclosesBox B v) (index : Nat) :
    getAtOrZero B.lo [index] ≤ getAtOrZero v.v [index] ∧
      getAtOrZero v.v [index] ≤ getAtOrZero B.hi [index] := by
  rcases B with ⟨n, lower, upper⟩
  rcases v with ⟨m, value⟩
  obtain ⟨hDim, h⟩ := h
  dsimp only at hDim
  subst m
  by_cases hi : index < n
  · simpa [get_at_or_zero_dim_cons, hi, Tensor.getScalar, Spec.get,
      castDimScalar_self] using h ⟨index, hi⟩
  · simp [get_at_or_zero_dim_cons, hi]

/-- The four directed endpoint products enclose real multiplication. -/
theorem intervalMul_encloses {lx ux ly uy x y : ℝ}
    (hx : lx ≤ x) (hx' : x ≤ ux) (hy : ly ≤ y) (hy' : y ≤ uy) :
    (intervalMul lx ux ly uy).1 ≤ x * y ∧ x * y ≤ (intervalMul lx ux ly uy).2 := by
  simpa [intervalMul, min2_eq_min, max2_eq_max, BoundOps.mulDown, BoundOps.mulUp]
    using interval_mul_bounds hx hx' hy hy'

/-- Simultaneously accumulate lower endpoints, values, and upper endpoints in list order. -/
private theorem ordered_sum_encloses (indices : List Nat)
    (bounds : Nat → ℝ × ℝ) (value : Nat → ℝ)
    (h : ∀ i, (bounds i).1 ≤ value i ∧ value i ≤ (bounds i).2)
    (acc : ℝ × ℝ) (total : ℝ) (hacc : acc.1 ≤ total ∧ total ≤ acc.2) :
    (indices.foldl (fun a i => (a.1 + (bounds i).1, a.2 + (bounds i).2)) acc).1 ≤
        indices.foldl (fun a i => a + value i) total ∧
      indices.foldl (fun a i => a + value i) total ≤
        (indices.foldl (fun a i => (a.1 + (bounds i).1, a.2 + (bounds i).2)) acc).2 := by
  induction indices generalizing acc total with
  | nil => exact hacc
  | cons i indices ih =>
      exact ih _ _ ⟨add_le_add hacc.1 (h i).1, add_le_add hacc.2 (h i).2⟩

/-- The executable binary interval contraction encloses the shared real value contraction. -/
theorem binaryMatmulBox_encloses (dims : NN.IR.OpContracts.MatmulDims)
    {left right : FlatBox ℝ} {x y : Val}
    (hx : EnclosesBox left x) (hy : EnclosesBox right y) :
    EnclosesBox (binaryMatmulBox dims left right)
      ⟨dims.outShape.size, NN.IR.Graph.matmulFlat dims x.v y.v⟩ := by
  refine ⟨rfl, ?_⟩
  intro output
  change Fin dims.outShape.size at output
  simp only [binaryMatmulBox, NN.IR.Graph.matmulFlat, castDimScalar_self,
    Tensor.getScalar_ofFn, BoundOps.addDown, BoundOps.addUp]
  apply ordered_sum_encloses
  · intro inner
    exact intervalMul_encloses
      (enclosed_flat_read hx (dims.leftIndex output.val inner)).1
      (enclosed_flat_read hx (dims.leftIndex output.val inner)).2
      (enclosed_flat_read hy (dims.rightIndex output.val inner)).1
      (enclosed_flat_read hy (dims.rightIndex output.val inner)).2
  · exact ⟨le_rfl, le_rfl⟩

/-- Successful binary value and interval transfers have the same layout and are enclosed. -/
theorem ibpBinaryMatmul_encloses {leftShape rightShape : Shape}
    {left right B : FlatBox ℝ} {x y v : Val}
    (hx : EnclosesBox left x) (hy : EnclosesBox right y)
    (hB : ibpBinaryMatmul? leftShape rightShape left right = some B)
    (hv : evalBinaryMatmul? leftShape rightShape x y = some v) :
    EnclosesBox B v := by
  cases hd : (NN.IR.OpContracts.matmulDims leftShape rightShape).toOption with
  | none => simp only [ibpBinaryMatmul?, hd, reduceCtorEq] at hB
  | some dims =>
      simp only [ibpBinaryMatmul?, hd] at hB
      simp only [evalBinaryMatmul?, hd] at hv
      split at hB
      · split at hv
        · cases Option.some.inj hB
          cases Option.some.inj hv
          exact binaryMatmulBox_encloses dims hx hy
        · cases hv
      · cases hB

/-- Binary matmul's local certificate step preserves both parent enclosures. -/
theorem binary_matmul_node_encloses {nodes : Array Node} {ps : ParamStore ℝ}
    {cert : Array (Option (FlatBox ℝ))} {inputs : Std.HashMap Nat Val}
    {vals : Array (Option Val)} {k : Nat} {B : FlatBox ℝ} {v : Val}
    (hkKind : (nodes[k]!).kind = .matmul)
    (hUnary : NN.IR.unaryParent? (nodes[k]!).parents = none)
    (hcertStep : certStepNode? nodes ps cert k = some B)
    (hvalStep : evalNode? nodes ps inputs vals k = some v)
    (hpe : ParentsEnclosed nodes cert vals k) : EnclosesBox B v := by
  simp only [certStepNode?, hkKind, hUnary] at hcertStep
  simp only [evalNode?, hkKind, hUnary] at hvalStep
  rcases hp : NN.IR.binaryParents? (nodes[k]!).parents with _ | ⟨p1, p2⟩ <;>
    simp only [hp, reduceCtorEq] at hcertStep hvalStep
  rcases hb1 : getBox? cert p1 with _ | left <;>
    simp only [hb1, reduceCtorEq] at hcertStep
  rcases hb2 : getBox? cert p2 with _ | right <;>
    simp only [hb2, reduceCtorEq] at hcertStep
  rcases hv1 : getVal? vals p1 with _ | x <;>
    simp only [hv1, reduceCtorEq] at hvalStep
  rcases hv2 : getVal? vals p2 with _ | y <;>
    simp only [hv2, reduceCtorEq] at hvalStep
  obtain ⟨hx, hy⟩ := parents_enclosed_binary hpe hp hb1 hb2 hv1 hv2
  exact ibpBinaryMatmul_encloses hx hy hcertStep hvalStep

end

end NN.MLTheory.CROWN.Graph.CertSoundness
