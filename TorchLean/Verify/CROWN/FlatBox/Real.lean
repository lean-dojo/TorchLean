/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.CROWN.FlatBox.Directed
public import TorchLean.NN.Gradients.Activation
public import TorchLean.Tensor.Layers.Linear
import TorchLean.Tensor.Algebra
import TorchLean.Tensor.Rep.Traversal.Sequence

/-!
# Real facts for the flat-box transfers

`IBP.linear_sound_real`: over `ℝ`, the interval linear rule `IBP.linear` on shaped boxes contains
the affine image of every point its input and bias boxes contain. `rowEncloses_of_contains_map`
reads a transfer on vector boxes on flat boxes: when the transfer contains `F` of every vector its
input box contains, its flat box encloses `F` of every coordinate the input box encloses
(`Directed.RowEncloses`). `Context.sigmoid_monotone_real` gives the monotonicity the minmax transfer
of the sigmoid needs, and `CROWN.tanh_monotone` gives it for `tanh`. `contains_getScalarBox_iff`:
viewing a flat box at a checked dimension (`FlatBox.getScalarBox`) and casting a point along the
same equality keep containment.
-/

@[expose] public section

open FloatLib.Numerics (MathFunctions)

namespace TorchLean.CROWN

open Tensor

/--
Soundness of `IBP.linear` over `ℝ`.

If `x ∈ xB` and `b ∈ bB`, then `W*x + b` lies in the interval box computed by
`IBP.linear W xB bB`.
-/
theorem IBP.linear_sound_real {m n : Nat}
  (W : Tensor ℝ [m, n])
  (xB : Box ℝ (.dim n .scalar))
  (bB : Box ℝ (.dim m .scalar))
  (x : Tensor ℝ [n]) (b : Tensor ℝ [m])
  (hx : Box.Contains (α:=ℝ) xB x) (hb : Box.Contains (α:=ℝ) bB b) :
  Box.Contains (α:=ℝ) (IBP.linear (α:=ℝ) W xB bB)
    (Linear.forward (α:=ℝ) { weights := W, bias := b } x) := by
  classical
  intro i
  change
    getScalar (IBP.linear (α := ℝ) W xB bB).lo i ≤
        getScalar (Linear.forward (α := ℝ) { weights := W, bias := b } x) i ∧
      getScalar (Linear.forward (α := ℝ) { weights := W, bias := b } x) i ≤
        getScalar (IBP.linear (α := ℝ) W xB bB).hi i
  simp only [IBP.linear, getScalar_dim, Linear.forward]
  rw [show
    getScalar (Tensor.contract W x _ _ [m] .matVec + b) i =
      getScalar (Tensor.contract W x _ _ [m] .matVec) i + getScalar b i by
        simp [getScalar_eq_apply, add_def]]
  rw [Tensor.getScalar_contract_matVec_eq_sum]
  simp only [BoundOps.addDown, BoundOps.addUp, BoundOps.mulDown, BoundOps.mulUp]
  let lower : Fin n → ℝ := fun j =>
    min (get2 W i j * getScalar xB.lo j) (get2 W i j * getScalar xB.hi j)
  let middle : Fin n → ℝ := fun j => get2 W i j * getScalar x j
  let upper : Fin n → ℝ := fun j =>
    max (get2 W i j * getScalar xB.lo j) (get2 W i j * getScalar xB.hi j)
  have termBounds (j : Fin n) : lower j ≤ middle j ∧ middle j ≤ upper j := by
    have hj := hx j
    change getScalar xB.lo j ≤ getScalar x j ∧
      getScalar x j ≤ getScalar xB.hi j at hj
    by_cases hw : 0 ≤ get2 W i j
    · exact ⟨
        le_trans (min_le_left _ _)
          (mul_le_mul_of_nonneg_left hj.1 hw),
        le_trans (mul_le_mul_of_nonneg_left hj.2 hw)
          (le_max_right _ _)⟩
    · have hw' : get2 W i j ≤ 0 := le_of_not_ge hw
      exact ⟨
        le_trans (min_le_right _ _)
          (mul_le_mul_of_nonpos_left hj.2 hw'),
        le_trans (mul_le_mul_of_nonpos_left hj.1 hw')
          (le_max_left _ _)⟩
  have hLower :
      (List.finRange n).foldl (fun acc j => acc + lower j) 0 ≤
        ∑ j : Fin n, middle j := by
    rw [List.finRange_foldl_add_eq_add_sum, zero_add]
    exact Finset.sum_le_sum fun j _ => (termBounds j).1
  have hUpper :
      (∑ j : Fin n, middle j) ≤
        (List.finRange n).foldl (fun acc j => acc + upper j) 0 := by
    rw [List.finRange_foldl_add_eq_add_sum, zero_add]
    exact Finset.sum_le_sum fun j _ => (termBounds j).2
  have hbI := hb i
  change getScalar bB.lo i ≤ getScalar b i ∧
    getScalar b i ≤ getScalar bB.hi i at hbI
  simpa [lower, middle, upper, Context.minimum_real, Context.maximum_real] using
    And.intro (add_le_add hLower hbI.1) (add_le_add hUpper hbI.2)

end TorchLean.CROWN

namespace TorchLean

open TorchLean.Tensor
open CROWN

noncomputable section

/-! ### Monotonicity of the scalar activations over the reals -/

/-- The scalar sigmoid specification is monotone over `ℝ`. -/
theorem Context.sigmoid_monotone_real : Monotone (Context.sigmoid (α := ℝ)) := by
  intro a b hab
  simpa only [sigmoid_eq_inv_exp, Real.sigmoid_def] using Real.sigmoid_monotone hab

/-! ### Transfers on vector boxes read on flat boxes -/

/-- Over `ℝ`, when the box `T` contains `F` of every vector the vector box of `B` contains, the flat
box of `T` encloses `F` of every coordinate `B` encloses. -/
theorem CROWN.Graph.rowEncloses_of_contains_map {F : ℝ → ℝ} {B : FlatBox ℝ} {n : Nat} {g : Nat → ℝ}
    {T : Box ℝ (.dim B.dim .scalar)}
    (hT : ∀ x : Tensor ℝ [B.dim], Box.Contains (ofFlatBox (α := ℝ) B) x →
      Box.Contains T (Tensor.map F x))
    (h : Directed.RowEncloses B n g) :
    Directed.RowEncloses (toFlatBox (α := ℝ) B.dim T) n fun i => F (g i) := by
  obtain ⟨d, lo, hi⟩ := B
  obtain rfl : d = n := h.1
  have hc : ∀ i : Fin d, getScalar lo i ≤ getScalar (Tensor.ofFn fun i : Fin d => g i) i ∧
      getScalar (Tensor.ofFn fun i : Fin d => g i) i ≤ getScalar hi i := fun i => by
    simpa only [Tensor.getAtOrZero_eq_getScalar, Tensor.getScalar_ofFn, LawfulBoundOps.toReal,
      instLawfulBoundOpsReal, id_eq] using h.2 i
  have hx : ∀ i : Fin d,
      getScalar T.lo i ≤ getScalar (Tensor.map F (Tensor.ofFn fun i : Fin d => g i)) i ∧
        getScalar (Tensor.map F (Tensor.ofFn fun i : Fin d => g i)) i ≤ getScalar T.hi i :=
    hT (Tensor.ofFn fun i : Fin d => g i) hc
  refine ⟨rfl, fun i => ?_⟩
  simpa only [toFlatBox, Tensor.getAtOrZero_eq_getScalar, getScalar_map, Tensor.getScalar_ofFn,
    LawfulBoundOps.toReal, instLawfulBoundOpsReal, id_eq] using hx i

/-! ### Viewing a flat box at a checked dimension -/

/-- Viewing a flat box at a checked dimension and casting a point along the same equality does not
change containment. -/
theorem CROWN.Graph.contains_getScalarBox_iff (B : FlatBox ℝ) {m : Nat} (h : B.dim = m)
    (x : Tensor ℝ [B.dim]) :
    Box.Contains (α := ℝ) (B.getScalarBox h) (castDimScalar (α := ℝ) h x) ↔
      Box.Contains (α := ℝ) (ofFlatBox (α := ℝ) B) x := by
  subst h
  exact Iff.rfl

end

end TorchLean
