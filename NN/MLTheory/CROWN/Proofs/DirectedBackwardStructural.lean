/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.DirectedBackwardSemantics

/-!
# Structural transfers in the directed backward sweep

Structural routing changes which node owes a coefficient. Addition and subtraction accumulate
both parent occurrences, including when their graph identifiers coincide.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Graph.DirectedBackward

open Spec TorchLean
open TorchLean.Tensor
open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Graph.Internal
open scoped BigOperators

noncomputable section

variable {α : Type} [Storage α] [Context α] [BoundOps α] [LawfulBoundOps α]

local notation "value" => LawfulBoundOps.toReal (α := α)

/-- Extending a finite vector by zero avoids dependent casts in the graph's coefficient table. -/
def extendFin {n : Nat} (f : Fin n → ℝ) (i : Nat) : ℝ :=
  if hi : i < n then f ⟨i, hi⟩ else 0

/-- The extension agrees with the finite vector at every valid coordinate. -/
@[simp] theorem extendFin_val {n : Nat} (f : Fin n → ℝ) (i : Fin n) :
    extendFin f i.val = f i := by simp [extendFin, i.isLt]

/-- Copying a coefficient to a value-preserving parent preserves its exact contribution. -/
theorem represents_copy {dims : Nat → Nat} {v : Nat → Nat → ℝ} {input k id pid : Nat}
    {st : DirectedBackwardState α} {z : ℝ}
    (h : Represents dims v input k st z) (hp : pid < st.coeffs.size)
    (hlive : pid ∈ pending input k) {aB : FlatBox α} {a : Nat → ℝ}
    (ha : RowEncloses aB (dims id) a) (heq : CopyEquation dims v id pid) :
    Represents dims v input k (addDirectedCoeff st pid aB)
      (z + dot (dims id) a (v id)) := by
  have hb : RowEncloses aB (dims pid) a := by simpa only [heq.1] using ha
  have hd : dot (dims pid) a (v pid) = dot (dims id) a (v id) := by
    rw [heq.1]
    simp only [dot, heq.2]
  simpa only [hd] using represents_add h pid hp hlive aB a hb

/-- An addition routes the same coefficient to both parents and sums both contributions. -/
theorem represents_addParents {dims : Nat → Nat} {v : Nat → Nat → ℝ}
    {input k id p q : Nat} {st : DirectedBackwardState α} {z : ℝ}
    (h : Represents dims v input k st z) (hp : p < st.coeffs.size)
    (hq : q < st.coeffs.size) (hplive : p ∈ pending input k) (hqlive : q ∈ pending input k)
    {aB : FlatBox α} {a : Nat → ℝ} (ha : RowEncloses aB (dims id) a)
    (hdp : dims p = dims id) (hdq : dims q = dims id)
    (hy : ∀ i : Fin (dims id), v id i.val = v p i.val + v q i.val) :
    Represents dims v input k (addDirectedCoeff (addDirectedCoeff st p aB) q aB)
      (z + dot (dims id) a (v id)) := by
  have hap : RowEncloses aB (dims p) a := by simpa only [hdp] using ha
  have haq : RowEncloses aB (dims q) a := by simpa only [hdq] using ha
  have h₁ := represents_add h p hp hplive aB a hap
  have h₂ := represents_add h₁ q (by simpa using hq) hqlive aB a haq
  have hd : dot (dims id) a (v id) =
      dot (dims p) a (v p) + dot (dims q) a (v q) := by
    rw [hdp, hdq]
    simp only [dot, hy, mul_add, Finset.sum_add_distrib]
  simpa only [hd, add_assoc] using h₂

/-- Subtraction routes a directed negation to the right parent, including repeated-parent
cancellation without assuming that the rounded coefficient interval is a point. -/
theorem represents_subParents (hzero : value (0 : α) = 0)
    {dims : Nat → Nat} {v : Nat → Nat → ℝ} {input k id p q : Nat}
    {st : DirectedBackwardState α} {z : ℝ}
    (h : Represents dims v input k st z) (hp : p < st.coeffs.size)
    (hq : q < st.coeffs.size) (hplive : p ∈ pending input k) (hqlive : q ∈ pending input k)
    {aB : FlatBox α} {a : Nat → ℝ} (ha : RowEncloses aB (dims id) a)
    (hdp : dims p = dims id) (hdq : dims q = dims id)
    (hy : ∀ i : Fin (dims id), v id i.val = v p i.val - v q i.val) :
    Represents dims v input k
      (addDirectedCoeff (addDirectedCoeff st p aB) q (negateDirectedCoeff aB))
      (z + dot (dims id) a (v id)) := by
  have hap : RowEncloses aB (dims p) a := by simpa only [hdp] using ha
  have haq : RowEncloses (negateDirectedCoeff aB) (dims q) (fun i => -a i) := by
    simpa only [hdq] using negateCoeff_encloses hzero ha
  have h₁ := represents_add h p hp hplive aB a hap
  have h₂ := represents_add h₁ q (by simpa using hq) hqlive
    (negateDirectedCoeff aB) (fun i => -a i) haq
  have hd : dot (dims id) a (v id) =
      dot (dims p) a (v p) + dot (dims q) (fun i => -a i) (v q) := by
    rw [hdp, hdq]
    simp only [dot, hy, mul_sub, Finset.sum_sub_distrib,
      neg_mul, Finset.sum_neg_distrib]
    ring
  simpa only [hd, add_assoc] using h₂

/-- Broadcasting a scalar coefficient to all coordinates encloses the exact sum pullback. -/
theorem broadcast_encloses {n : Nat} (lo hi : Tensor α [1]) (a : ℝ)
    (ha : value (lo.getScalar 0) ≤ a ∧ a ≤ value (hi.getScalar 0)) :
    RowEncloses
      { dim := n
        lo := Tensor.full (α := α) (.dim n .scalar) (getAtOrZero lo [0])
        hi := Tensor.full (α := α) (.dim n .scalar) (getAtOrZero hi [0]) }
      n (fun _ => a) := by
  refine ⟨rfl, ?_⟩
  intro i
  simp only [read_fin, Tensor.getScalar_full]
  rw [show getAtOrZero lo [0] = lo.getScalar 0 from read_fin lo 0,
    show getAtOrZero hi [0] = hi.getScalar 0 from read_fin hi 0]
  exact ha

/-- A coefficient gathered along a coordinate map encloses the same gathered real vector. -/
theorem gather_encloses {m n : Nat} (lo hi : Tensor α [m]) (a : Nat → ℝ)
    (ha : RowEncloses { dim := m, lo := lo, hi := hi } m a)
    (index : Fin n → Fin m) :
    RowEncloses
      { dim := n
        lo := Tensor.ofFn (fun i => lo.getScalar (index i))
        hi := Tensor.ofFn (fun i => hi.getScalar (index i)) }
      n (extendFin (fun i => a (index i).val)) := by
  refine ⟨rfl, ?_⟩
  intro i
  simpa only [read_fin, Tensor.getScalar_ofFn, extendFin_val] using ha.2 (index i)

/-- A bijective coordinate change preserves an exact real objective. -/
theorem dot_permutation {n : Nat} (perm : Fin n → Fin n) (hperm : Function.Bijective perm)
    (a x y : Nat → ℝ) (hy : ∀ i : Fin n, y (perm i).val = x i.val) :
    dot n a y = dot n (extendFin (fun i => a (perm i).val)) x := by
  unfold dot
  symm
  apply Fintype.sum_equiv (Equiv.ofBijective perm hperm)
  intro i
  simp only [extendFin_val, Equiv.ofBijective_apply, hy]

/-- The actual axis-permutation pullback encloses coefficients with the same real objective. -/
theorem axisPermutation_encloses {dims : Nat → Nat} {v : Nat → Nat → ℝ}
    {id pid : Nat} {aB aX : FlatBox α} {a : Nat → ℝ}
    (ha : RowEncloses aB (dims id) a) (shape : Spec.Shape) (forward : Array Nat)
    (heq : PermutationEquation dims v id pid shape forward)
    (hresult : directedAxisPermutation? shape forward aB = some aX) :
    ∃ a', RowEncloses aX (dims pid) a' ∧
      dot (dims id) a (v id) = dot (dims pid) a' (v pid) := by
  obtain ⟨adim, alo, ahi⟩ := aB
  obtain ⟨hdim, ha⟩ := ha
  dsimp only at hdim
  subst adim
  by_cases hz : dims id = 0
  · simp only [directedAxisPermutation?, hz, ↓reduceIte] at hresult
    change some { dim := dims id, lo := alo, hi := ahi } = some aX at hresult
    cases hresult
    refine ⟨a, ⟨heq.1.symm, ?_⟩, ?_⟩
    · rw [heq.1]
      exact ha
    · rw [heq.1, hz]
      simp [dot]
  · simp only [directedAxisPermutation?, hz, ↓reduceIte] at hresult
    change ((NN.IR.OpContracts.inversePerm forward).toOption.bind fun inverse =>
      (flatAxisPermutation? shape inverse (dims id)).bind fun perm =>
        some
          { dim := dims id
            lo := backwardPermuteVec perm alo
            hi := backwardPermuteVec perm ahi }) = some aX at hresult
    obtain ⟨inverse, hinverse, hresult⟩ := Option.bind_eq_some_iff.mp hresult
    obtain ⟨perm, hperm, hresult⟩ := Option.bind_eq_some_iff.mp hresult
    cases hresult
    obtain ⟨hbij, hy⟩ := heq.2 inverse hinverse perm hperm
    refine ⟨extendFin (fun i => a (perm i).val), ?_, ?_⟩
    · rw [heq.1]
      exact gather_encloses alo ahi a ⟨rfl, ha⟩ perm
    · rw [heq.1]
      exact dot_permutation perm hbij a (v pid) (v id) hy

end

end NN.MLTheory.CROWN.Graph.DirectedBackward
