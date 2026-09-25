/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.DirectedBackwardStructural

/-!
# The reverse sweep and its boundary conditions

The loop visits every graph position exactly once in reverse order. Failure is absorbing.
The frontier starts at the output objective and ends at the designated input affine form.
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

omit [LawfulBoundOps α] in
/-- Folding coefficient contributions preserves the size of the graph table. -/
@[simp] theorem addParents_size (parents : Array (Nat × FlatBox α))
    (st : DirectedBackwardState α) :
    (parents.foldl (fun state entry => addDirectedCoeff state entry.1 entry.2) st).coeffs.size =
      st.coeffs.size := by
  apply Array.foldl_induction
    (fun (_ : Nat) (state : DirectedBackwardState α) => state.coeffs.size = st.coeffs.size) rfl
  intro i state h
  simpa only [addCoeff_size] using h

omit [LawfulBoundOps α] in
/-- A backward node preserves the graph table size on every execution path. -/
@[simp] theorem backwardNode_size (nodes : Array Node) (ps : ParamStore α)
    (ibp : Array (Option (FlatBox α))) (ctx : AffineCtx)
    (st : DirectedBackwardState α) (id : Nat) :
    (directedBackwardNode nodes ps ibp ctx st id).coeffs.size = st.coeffs.size := by
  unfold directedBackwardNode
  split
  · rfl
  · split
    · rfl
    · dsimp only
      cases nodes[id]!.kind <;> dsimp only
      all_goals
        repeat' first
          | rfl
          | (solve | simp only [addDirectedConstant, addCoeff_size, consume_size, addParents_size])
          | split

omit [LawfulBoundOps α] in
/-- Once a directed sweep fails, later graph nodes leave it failed. -/
theorem backwardNode_failed (nodes : Array Node) (ps : ParamStore α)
    (ibp : Array (Option (FlatBox α))) (ctx : AffineCtx)
    (st : DirectedBackwardState α) (id : Nat) (h : st.failed = true) :
    (directedBackwardNode nodes ps ibp ctx st id).failed = true := by
  simp [directedBackwardNode, h]

/-- The sweep either has failed or encloses an exact pending objective in a table of fixed size. -/
def SweepInvariant (dims : Nat → Nat) (v : Nat → Nat → ℝ) (input size k : Nat)
    (z : ℝ) (st : DirectedBackwardState α) : Prop :=
  st.failed = true ∨ st.coeffs.size = size ∧ Represents dims v input k st z

/-- An invariant preserved at each position is preserved by the engine's reverse `finRange` fold. -/
theorem reverseSweep_preserves {β : Type} (step : β → Nat → β) (P : Nat → β → Prop)
    (size : Nat) (hstep : ∀ k, k < size → ∀ st, P (k + 1) st → P k (step st k)) :
    ∀ n, n ≤ size → ∀ st, P n st →
      P 0 ((List.finRange n).reverse.foldl (fun state i => step state i.val) st) := by
  intro n
  induction n with
  | zero =>
      intro _ st h
      exact h
  | succ n ih =>
      intro hn st h
      have hs := hstep n (by omega) st h
      simpa only [List.finRange_succ_last, List.reverse_append, List.reverse_cons,
        List.reverse_nil, List.cons_append, List.nil_append, List.foldl_cons,
        ← List.map_reverse, List.foldl_map,
        Fin.val_last, Fin.val_castSucc] using ih (by omega) (step st n) hs

/-- A single output row represents exactly the requested output objective. -/
theorem initial_represents (hzero : value (0 : α) = 0)
    (dims : Nat → Nat) (v : Nat → Nat → ℝ) (input size output : Nat)
    (houtput : output < size) (obj : FlatTensor α) (hdim : obj.n = dims output) :
    SweepInvariant dims v input size size
      (dot (dims output) (fun i => value (getAtOrZero obj.v [i])) (v output))
      { coeffs := (Array.replicate size none).set! output (some (pointCoeffBox obj))
        cstLo := 0, cstHi := 0 } := by
  right
  refine ⟨by simp, _, 0, initial_encloses hzero dims size output obj hdim, ?_⟩
  have hmem : output ∈ pending input size := by simp [pending, houtput]
  have hrow (id : Nat) :
      dot (dims id)
        (fun i => if id = output then value (getAtOrZero obj.v [i]) else 0) (v id) =
        if id = output then
          dot (dims output) (fun i => value (getAtOrZero obj.v [i])) (v output) else 0 := by
    by_cases h : id = output
    · subst id
      simp
    · simp [h, dot]
  simp only [frontierValue, hrow, Finset.sum_ite_eq', hmem, ↓reduceIte, add_zero]

/-- The dimension-checked affine conversion encloses an arbitrary enclosed input objective. -/
theorem inputAffines_row_encloses (hzero : value (0 : α) = 0)
    {n : Nat} {xB aB : FlatBox α} {x a : Nat → ℝ} {c : ℝ} {lo hi : α}
    (hx : RowEncloses xB n x) (ha : RowEncloses aB n a)
    (hc : value lo ≤ c ∧ c ≤ value hi) {lower upper : AffineVec α n 1}
    (hresult : directedInputAffines n xB aB lo hi = some (lower, upper)) :
    affineValue lower (fun i => x i.val) ≤ dot n a x + c ∧
      dot n a x + c ≤ affineValue upper (fun i => x i.val) := by
  obtain ⟨xdim, xlo, xhi⟩ := xB
  obtain ⟨adim, alo, ahi⟩ := aB
  obtain ⟨hxDim, hx⟩ := hx
  obtain ⟨haDim, ha⟩ := ha
  dsimp only at hxDim haDim
  subst xdim
  subst adim
  exact inputAffines_encloses hzero xlo xhi alo ahi lo hi
    (fun i => x i.val) (fun i => a i.val) c
    (by simpa only [read_fin] using hx) (by simpa only [read_fin] using ha) hc hresult

/-- The default zero input row encloses any coefficient represented by an absent array entry. -/
theorem inputRow_encloses (hzero : value (0 : α) = 0)
    {dims : Nat → Nat} {st : DirectedBackwardState α} {f : Nat → Nat → ℝ} {c : ℝ}
    (h : StateEncloses dims st f c) (input n : Nat)
    (hinput : input < st.coeffs.size) (hdim : dims input = n) :
    RowEncloses
      (st.coeffs[input]!.getD
        { dim := n
          lo := Tensor.full (α := α) (.dim n .scalar) 0
          hi := Tensor.full (α := α) (.dim n .scalar) 0 })
      n (f input) := by
  have hf := h.1 input hinput
  rw [hdim] at hf
  cases he : st.coeffs[input]! with
  | none =>
      simp only [he, Option.getD_none] at hf ⊢
      refine ⟨rfl, ?_⟩
      intro i
      simp only [read_fin, Tensor.getScalar_full, hzero, hf i, le_refl, and_self]
  | some box =>
      simpa only [he, Option.getD_some] using hf

end

end NN.MLTheory.CROWN.Graph.DirectedBackward
