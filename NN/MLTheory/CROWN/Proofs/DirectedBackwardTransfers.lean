/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.DirectedBackwardFrontier

/-!
# Soundness of the executable directed transfers

These lemmas connect the interval kernels with the exact objective at a sweep frontier.
All scalar operations in the coefficient and constant accumulators are the engine's directed
operations; the real identities describe the graph computed by the stored parameters.
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

/-- A flat interval inner product encloses the ordinary real inner product. -/
theorem dotBox_row_encloses
    {n : Nat} {aB xB : FlatBox α} {a x : Nat → ℝ}
    (ha : RowEncloses aB n a) (hx : RowEncloses xB n x) {lo hi : α}
    (hresult : directedDotBox aB xB = some (lo, hi)) :
    value lo ≤ dot n a x ∧ dot n a x ≤ value hi := by
  obtain ⟨adim, alo, ahi⟩ := aB
  obtain ⟨xdim, xlo, xhi⟩ := xB
  obtain ⟨haDim, ha⟩ := ha
  obtain ⟨hxDim, hx⟩ := hx
  dsimp only at haDim hxDim
  subst adim
  subst xdim
  exact dotBox_encloses alo ahi xlo xhi (fun i => a i.val) (fun i => x i.val)
    (by simpa only [read_fin] using ha) (by simpa only [read_fin] using hx) hresult

/-- Discharging an enclosed node value either fails or adds its exact objective to the constant. -/
theorem represents_consume
    {dims : Nat → Nat} {v : Nat → Nat → ℝ} {input k n : Nat}
    {st : DirectedBackwardState α} {z : ℝ}
    (h : Represents dims v input k st z) {aB xB : FlatBox α} {a x : Nat → ℝ}
    (ha : RowEncloses aB n a) (hx : RowEncloses xB n x) :
    (consumeDirectedObjective st aB xB).failed = true ∨
      Represents dims v input k (consumeDirectedObjective st aB xB) (z + dot n a x) := by
  cases hresult : directedDotBox aB xB with
  | none =>
      left
      simp [consumeDirectedObjective, hresult, DirectedBackwardState.fail]
  | some bounds =>
      rcases bounds with ⟨lo, hi⟩
      right
      simp only [consumeDirectedObjective, hresult]
      exact represents_addConstant h lo hi (dot n a x)
        (dotBox_row_encloses ha hx hresult)

/-- The exact coefficient produced by transposing a stored matrix. -/
def transposedCoeff {m n : Nat} (W : Tensor α [m, n]) (a : Nat → ℝ) (j : Nat) : ℝ :=
  ∑ i : Fin m, a i.val * value (getAtOrZero W [i.val, j])

/-- The exact constant split off by a stored linear bias. -/
def biasDot {m : Nat} (b : Tensor α [m]) (a : Nat → ℝ) : ℝ :=
  ∑ i : Fin m, a i.val * value (b.getScalar i)

/-- The flat linear-transfer result encloses its transposed coefficient and bias contribution. -/
theorem linear_row_encloses
    {m n : Nat} {aB : FlatBox α} (W : Tensor α [m, n]) (b : Tensor α [m])
    {a : Nat → ℝ} (ha : RowEncloses aB m a)
    {aX : FlatBox α} {lo hi : α}
    (hresult : directedBackwardLinear aB W b = some (aX, (lo, hi))) :
    RowEncloses aX n (transposedCoeff W a) ∧
      value lo ≤ biasDot b a ∧ biasDot b a ≤ value hi := by
  obtain ⟨adim, alo, ahi⟩ := aB
  obtain ⟨hdim, ha⟩ := ha
  dsimp only at hdim
  subst adim
  have h := linear_encloses alo ahi W b (fun i => a i.val)
    (by simpa only [read_fin] using ha) hresult
  refine ⟨⟨h.1, ?_⟩, h.2.2⟩
  intro j
  simpa [transposedCoeff, get_at_or_zero_dim_cons, j.isLt, Spec.get2,
    Tensor.getScalar, Spec.get] using h.2.1 j

/-- Reassociating a linear node is an exact real identity before interval approximation. -/
theorem dot_linear {m n : Nat} (W : Tensor α [m, n]) (b : Tensor α [m])
    (a x y : Nat → ℝ)
    (hy : ∀ i : Fin m, y i.val =
      (∑ j : Fin n, value (Spec.get2 W i j) * x j.val) + value (b.getScalar i)) :
    dot m a y = dot n (transposedCoeff W a) x + biasDot b a := by
  have hread (i : Fin m) (j : Fin n) :
      getAtOrZero W [i.val, j.val] = Spec.get2 W i j := by
    simp [get_at_or_zero_dim_cons, i.isLt, j.isLt, Spec.get2, Tensor.getScalar, Spec.get]
  simp only [dot, hy, mul_add, Finset.sum_add_distrib, Finset.mul_sum,
    transposedCoeff, Finset.sum_mul, hread, biasDot]
  congr 1
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro j _
  apply Finset.sum_congr rfl
  intro i _
  ring

/-- A successful linear transfer preserves the represented objective and the coefficient table
size, including accumulation into an already-active parent. -/
theorem represents_linear
    {dims : Nat → Nat} {v : Nat → Nat → ℝ} {input k m n : Nat}
    {st : DirectedBackwardState α} {z : ℝ}
    (h : Represents dims v input k st z) (pid : Nat)
    (hp : pid < st.coeffs.size) (hlive : pid ∈ pending input k) (hdim : dims pid = n)
    {aB : FlatBox α} {a y : Nat → ℝ} (ha : RowEncloses aB m a)
    (W : Tensor α [m, n]) (b : Tensor α [m])
    (hy : ∀ i : Fin m, y i.val =
      (∑ j : Fin n, value (Spec.get2 W i j) * v pid j.val) + value (b.getScalar i))
    {aX : FlatBox α} {lo hi : α}
    (hresult : directedBackwardLinear aB W b = some (aX, (lo, hi))) :
    Represents dims v input k (addDirectedConstant (addDirectedCoeff st pid aX) lo hi)
        (z + dot m a y) ∧
      (addDirectedConstant (addDirectedCoeff st pid aX) lo hi).failed = st.failed ∧
      (addDirectedConstant (addDirectedCoeff st pid aX) lo hi).coeffs.size = st.coeffs.size := by
  have he := linear_row_encloses W b ha hresult
  have hr : RowEncloses aX (dims pid) (transposedCoeff W a) := by
    simpa only [hdim] using he.1
  have hadd := represents_add h pid hp hlive aX (transposedCoeff W a) hr
  have hc := represents_addConstant hadd lo hi (biasDot b a) he.2
  obtain ⟨f, c, hs, _⟩ := h
  have hstate := addCoeff_encloses hs pid hp aX (transposedCoeff W a) hr
  refine ⟨?_, hstate.2.1, hstate.2.2⟩
  simpa only [hdim, dot_linear W b a (v pid) y hy, add_assoc] using hc

end

end NN.MLTheory.CROWN.Graph.DirectedBackward
