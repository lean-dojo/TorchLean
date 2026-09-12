/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Algebra.BigOperators.Field
public import NN.Proofs.Tensor.Basic.LinearAlgebra
public import NN.Proofs.Utils.MathFunctions
public import NN.Spec.Layers.Activation
public import NN.Spec.Core.Context.Real

/-!
# Softmax analysis properties

This module proves theorem-level facts about TorchLean's spec-level softmax operators. The
definitions themselves live in `NN.Spec.Layers.Activation`; this file belongs under
`NN.Proofs.Analysis` because it imports real-analysis and finite-sum proof machinery to establish
properties of those definitions.

Current theorem surface:

- `getScalar_le_maxVecSpec` and `exists_getScalar_eq_maxVecSpec`: the shared stable-shift maximum
  is an attained upper bound;
- `softmax_shift_nonpos` and `exists_softmax_shift_eq_zero`: shifted logits are nonpositive and
  one is exactly zero;
- `softmax_shift_exp_le_one` and `softmax_shift_denom_bounds`: exponentials cannot overflow and
  their denominator lies between `1` and the axis length;
- `softmax_vec_spec_normalized`: exposes the positive normalized weights used by the stable
  max-shifted implementation;
- `softmax_vec_spec_pos`: every coordinate is strictly positive;
- `softmax_vec_spec_mem_unitInterval`: every coordinate lies in `[0,1]`;
- `sum_spec_softmax_vec_spec`: a nonempty vector softmax sums to `1`;
- `sum_spec_softmax_backward_spec`: the concrete stable softmax VJP has coordinate sum zero;
- `abs_getScalar_softmax_backward_spec_le_two_mul`: bounded upstream coordinates give a
  dimension-independent coordinate bound for that VJP;
- `sum_spec_softmax_spec_row`: axis-`1` matrix softmax has rows summing to `1` when the key
  dimension is nonempty;
- `getScalar_softmaxVecSpec_eq_exp_div` and `getScalar_logSoftmaxVecSpec_eq_sub_log`: over `ℝ`
  the stable kernels agree coordinatewise with the textbook formulas, which is what connects them
  to the analytic `softmaxVec` and `logSoftmaxVec` in `NN.Proofs.Autograd.FDeriv.SoftmaxSpec`;
- `softmaxSpec_zero_vec`, `softmaxSpec_one_matrix`, and `get_softmaxSpec_one`: the axis-parametric
  operators reduce to the vector kernels on vectors and matrix rows.

We intentionally state these over `ℝ`: positivity of `exp` and division by a positive denominator
are the mathematical facts that make the probabilistic interpretation precise.
-/

@[expose] public section

open scoped BigOperators

noncomputable section

namespace Proofs

open Spec TorchLean
open TorchLean TorchLean.Tensor
open Activation

/-! ## Scalar helpers

`softmaxVecSpec` is written over tensors, so even one coordinate has type `Tensor ℝ .scalar`.
Local helper definitions expose scalar coordinates to the proof without adding public API.
-/

/--
Eliminate a scalar tensor using the same matcher as `Activation.softmaxVecSpec`.

This local eliminator avoids depending on compiler-generated matcher names, which are not a stable
interface and can change when an earlier definition is inserted in `Activation.lean`.
-/
private def scalarElim {β : Sort _} (t : Tensor ℝ .scalar) (k : ℝ → β) : β :=
  k t.item

@[simp] private theorem scalarElim_scalar {β : Sort _} (k : ℝ → β) (v : ℝ) :
    scalarElim (β := β) (Tensor.scalar v) k = k v := by simp [scalarElim]

/-- Extract the real value from a scalar tensor for local proof steps. -/
private abbrev scalarVal (t : Tensor ℝ .scalar) : ℝ :=
  scalarElim (β := ℝ) t (fun v => v)

/-! ## Stable max shift -/

/-- Every coordinate is bounded above by the exact maximum used by softmax and log-softmax. -/
theorem getScalar_le_maxVecSpec {n : Nat}
    (t : Tensor ℝ [Nat.succ n]) (i : Fin (Nat.succ n)) :
    TorchLean.Tensor.getScalar t i <= Tensor.item (Activation.maxVecSpec t) := by
  change t.getScalar i <=
    (List.finRange (Nat.succ n)).foldl
      (fun acc j => max acc (t.getScalar j))
      (t.getScalar ⟨0, Nat.succ_pos n⟩)
  exact List.le_foldl_max_of_mem (List.finRange (Nat.succ n))
    (fun j => t.getScalar j)
    (acc := t.getScalar ⟨0, Nat.succ_pos n⟩) (i := i)
    (List.mem_finRange i)

/-- The maximum used by stable softmax is attained by an input coordinate. -/
theorem exists_getScalar_eq_maxVecSpec {n : Nat}
    (t : Tensor ℝ [Nat.succ n]) :
    ∃ i, TorchLean.Tensor.getScalar t i = Tensor.item (Activation.maxVecSpec t) := by
  let firstIndex : Fin (Nat.succ n) := ⟨0, Nat.succ_pos n⟩
  let value : Fin (Nat.succ n) -> ℝ := fun i => t.getScalar i
  change ∃ i, value i =
    (List.finRange (Nat.succ n)).foldl (fun acc j => max acc (value j))
      (value firstIndex)
  rcases List.foldl_max_eq_init_or_mem (List.finRange (Nat.succ n)) value
      (value firstIndex) with hfirst | ⟨i, hi, hvalue⟩
  · exact ⟨firstIndex, hfirst.symm⟩
  · exact ⟨i, hvalue.symm⟩

/-- Every logit shifted by the implementation's maximum is nonpositive. -/
theorem softmax_shift_nonpos {n : Nat}
    (t : Tensor ℝ [Nat.succ n]) (i : Fin (Nat.succ n)) :
    TorchLean.Tensor.getScalar
      (TorchLean.Tensor.subSpec t (Spec.replicate (Activation.maxVecSpec t))) i <= 0 := by
  have hle := getScalar_le_maxVecSpec t i
  rw [show TorchLean.Tensor.subSpec t (Spec.replicate (Activation.maxVecSpec t)) =
    TorchLean.Tensor.map2Spec (· - ·) t (Spec.replicate (Activation.maxVecSpec t)) by rfl]
  rw [TorchLean.Tensor.getScalar_map2Spec]
  simpa [Spec.replicate] using sub_nonpos.mpr hle

/-- At least one max-shifted logit is exactly zero. -/
theorem exists_softmax_shift_eq_zero {n : Nat}
    (t : Tensor ℝ [Nat.succ n]) :
    ∃ i, TorchLean.Tensor.getScalar
      (TorchLean.Tensor.subSpec t (Spec.replicate (Activation.maxVecSpec t))) i = 0 := by
  rcases exists_getScalar_eq_maxVecSpec t with ⟨i, hi⟩
  refine ⟨i, ?_⟩
  rw [show TorchLean.Tensor.subSpec t (Spec.replicate (Activation.maxVecSpec t)) =
    TorchLean.Tensor.map2Spec (· - ·) t (Spec.replicate (Activation.maxVecSpec t)) by rfl]
  rw [TorchLean.Tensor.getScalar_map2Spec]
  simpa [Spec.replicate] using sub_eq_zero.mpr hi

/-- Exponentiating a max-shifted real logit produces a value at most one. This is the central
overflow-prevention fact behind stable softmax. -/
theorem softmax_shift_exp_le_one {n : Nat}
    (t : Tensor ℝ [Nat.succ n]) (i : Fin (Nat.succ n)) :
    TorchLean.Tensor.getScalar (Activation.maxShiftedExpVecSpec t) i <= 1 := by
  have hshift := softmax_shift_nonpos t i
  simpa [Activation.maxShiftedExpVecSpec, TorchLean.Tensor.expSpec,
    mathfunc_exp_eq_rexp] using (Real.exp_le_one_iff.mpr hshift)

/-- Max-shifted real exponentials remain strictly positive. -/
theorem softmax_shift_exp_pos {n : Nat}
    (t : Tensor ℝ [Nat.succ n]) (i : Fin (Nat.succ n)) :
    0 < TorchLean.Tensor.getScalar (Activation.maxShiftedExpVecSpec t) i := by
  simpa [Activation.maxShiftedExpVecSpec, TorchLean.Tensor.expSpec,
    mathfunc_exp_eq_rexp] using
    Real.exp_pos (TorchLean.Tensor.getScalar
      (TorchLean.Tensor.subSpec t (Spec.replicate (Activation.maxVecSpec t))) i)

/-- One max-shifted exponential is exactly one, because the maximum is attained. -/
theorem exists_softmax_shift_exp_eq_one {n : Nat}
    (t : Tensor ℝ [Nat.succ n]) :
    ∃ i, TorchLean.Tensor.getScalar (Activation.maxShiftedExpVecSpec t) i = 1 := by
  rcases exists_softmax_shift_eq_zero t with ⟨i, hzero⟩
  refine ⟨i, ?_⟩
  simpa [Activation.maxShiftedExpVecSpec, TorchLean.Tensor.expSpec,
    mathfunc_exp_eq_rexp] using (Real.exp_eq_one_iff _).mpr hzero

/-- The stable softmax denominator lies in `[1,n]` for a nonempty vector of length `n`.

The lower bound rules out division by zero. The upper bound follows because every shifted
exponential is at most one. Together with `softmax_shift_exp_le_one`, this makes overflow
prevention an explicit theorem of the max-shifted implementation rather than an empirical claim.
-/
theorem softmax_shift_denom_bounds {n : Nat}
    (t : Tensor ℝ [Nat.succ n]) :
    1 <= TorchLean.Tensor.sumSpec (Activation.maxShiftedExpVecSpec t) ∧
      TorchLean.Tensor.sumSpec (Activation.maxShiftedExpVecSpec t) <= Nat.succ n := by
  classical
  let ex := Activation.maxShiftedExpVecSpec t
  have hpos : ∀ i, 0 <= TorchLean.Tensor.getScalar ex i :=
    fun i => le_of_lt (softmax_shift_exp_pos t i)
  have hle : ∀ i, TorchLean.Tensor.getScalar ex i <= 1 := fun i => softmax_shift_exp_le_one t i
  rcases exists_softmax_shift_exp_eq_one t with ⟨witness, hwitness⟩
  rw [Spec.sum_spec_vec]
  constructor
  · calc
      1 = TorchLean.Tensor.getScalar ex witness := hwitness.symm
      _ <= ∑ i, TorchLean.Tensor.getScalar ex i :=
        Finset.single_le_sum (fun i _ => hpos i) (Finset.mem_univ witness)
  · calc
      (∑ i, TorchLean.Tensor.getScalar ex i) <= ∑ _i : Fin (Nat.succ n), (1 : ℝ) := by
        exact Finset.sum_le_sum fun i _ => hle i
      _ = Nat.succ n := by simp

/-! ## Normalized coordinates -/

/--
The stable vector softmax has positive weights normalized by their sum.

This lemma exposes exactly one reusable algebraic description of the implementation. The weights
are the max-shifted exponentials computed by `softmaxVecSpec`; subsequent proofs of positivity,
range, and normalization do not unfold the implementation again.
-/
theorem softmax_vec_spec_normalized {n : Nat}
    (t : Tensor ℝ [Nat.succ n]) :
    ∃ weights : Fin (Nat.succ n) → ℝ,
      (∀ i, 0 < weights i) ∧
      ∀ i,
        TorchLean.Tensor.getScalar (Activation.softmaxVecSpec (α := ℝ) (n := Nat.succ n) t) i =
          weights i / ∑ j, weights j := by
  classical
  let exponentials := Activation.maxShiftedExpVecSpec t
  let weights : Fin (Nat.succ n) → ℝ := fun j => exponentials.getScalar j
  have hweightsPos : ∀ j, 0 < weights j := fun j => softmax_shift_exp_pos t j
  let denom : ℝ := TorchLean.Tensor.sumSpec exponentials
  have hdenom : denom = ∑ j : Fin (Nat.succ n), weights j := by
    simpa [denom, weights] using Spec.sum_spec_vec exponentials
  refine ⟨weights, hweightsPos, ?_⟩
  intro i
  change TorchLean.Tensor.getScalar
      (TorchLean.Tensor.map2Spec (· / ·) exponentials
        (Spec.replicate (Tensor.scalar denom))) i =
    weights i / ∑ j, weights j
  rw [TorchLean.Tensor.getScalar_map2Spec]
  simp [Spec.replicate, weights, hdenom]

/-- Coordinate equation for the concrete stable vector softmax.

This is the small unfolding lemma that downstream algebraic proofs should use. It exposes the
max-shifted numerator and its tensor sum while hiding the implementation chosen for tensor
reduction. -/
theorem getScalar_softmaxVecSpec {n : Nat}
    (t : Tensor ℝ [Nat.succ n]) (i : Fin (Nat.succ n)) :
    TorchLean.Tensor.getScalar (Activation.softmaxVecSpec (α := ℝ) (n := Nat.succ n) t) i =
      TorchLean.Tensor.getScalar (Activation.maxShiftedExpVecSpec t) i /
        TorchLean.Tensor.sumSpec (Activation.maxShiftedExpVecSpec t) := by
  change TorchLean.Tensor.getScalar
      (TorchLean.Tensor.map2Spec (· / ·) (Activation.maxShiftedExpVecSpec t)
        (Spec.replicate
          (Tensor.scalar (TorchLean.Tensor.sumSpec (Activation.maxShiftedExpVecSpec t))))) i = _
  rw [TorchLean.Tensor.getScalar_map2Spec]
  simp [Spec.replicate]

/-- Coordinate formula for the max-shifted exponentials. -/
theorem getScalar_maxShiftedExpVecSpec {n : Nat}
    (t : Tensor ℝ [Nat.succ n]) (j : Fin (Nat.succ n)) :
    TorchLean.Tensor.getScalar (Activation.maxShiftedExpVecSpec t) j =
      Real.exp (TorchLean.Tensor.getScalar t j - Tensor.item (Activation.maxVecSpec t)) := by
  simp only [Activation.maxShiftedExpVecSpec, TorchLean.Tensor.expSpec, TorchLean.Tensor.subSpec,
    TorchLean.Tensor.getScalar_mapSpec, TorchLean.Tensor.getScalar_map2Spec]
  simp [Spec.replicate, mathfunc_exp_eq_rexp]

/-- The stable softmax denominator is the plain exponential sum times the shift factor. -/
theorem sum_spec_maxShiftedExpVecSpec {n : Nat} (t : Tensor ℝ [Nat.succ n]) :
    TorchLean.Tensor.sumSpec (Activation.maxShiftedExpVecSpec t) =
      (∑ j, Real.exp (TorchLean.Tensor.getScalar t j)) *
        Real.exp (-Tensor.item (Activation.maxVecSpec t)) := by
  rw [Spec.sum_spec_vec, Finset.sum_mul]
  refine Finset.sum_congr rfl ?_
  intro j _
  rw [getScalar_maxShiftedExpVecSpec, Real.exp_sub, Real.exp_neg, div_eq_mul_inv]

/-- Over `ℝ`, the stable max-shifted softmax agrees coordinatewise with the textbook formula
`exp(xᵢ) / ∑ⱼ exp(xⱼ)`.

This is the lemma that lets analytic developments work with the unshifted formula while the spec
keeps its overflow-safe implementation. -/
theorem getScalar_softmaxVecSpec_eq_exp_div {n : Nat}
    (t : Tensor ℝ [Nat.succ n]) (i : Fin (Nat.succ n)) :
    TorchLean.Tensor.getScalar (Activation.softmaxVecSpec (α := ℝ) (n := Nat.succ n) t) i =
      Real.exp (TorchLean.Tensor.getScalar t i) /
        ∑ j, Real.exp (TorchLean.Tensor.getScalar t j) := by
  rw [getScalar_softmaxVecSpec, sum_spec_maxShiftedExpVecSpec, getScalar_maxShiftedExpVecSpec,
    Real.exp_sub, Real.exp_neg, div_eq_mul_inv]
  exact mul_div_mul_right _ _ (inv_ne_zero (Real.exp_ne_zero _))

/-- Over `ℝ`, the stable log-softmax agrees coordinatewise with `xᵢ - log ∑ⱼ exp(xⱼ)`. -/
theorem getScalar_logSoftmaxVecSpec_eq_sub_log {n : Nat}
    (t : Tensor ℝ [Nat.succ n]) (i : Fin (Nat.succ n)) :
    TorchLean.Tensor.getScalar (Activation.logSoftmaxVecSpec (α := ℝ) (n := Nat.succ n) t) i =
      TorchLean.Tensor.getScalar t i -
        Real.log (∑ j, Real.exp (TorchLean.Tensor.getScalar t j)) := by
  have hsumPos : 0 < ∑ j, Real.exp (TorchLean.Tensor.getScalar t j) :=
    Finset.sum_pos (fun j _ => Real.exp_pos _) Finset.univ_nonempty
  have hlog : MathFunctions.log (TorchLean.Tensor.sumSpec (Activation.maxShiftedExpVecSpec t)) =
      Real.log (∑ j, Real.exp (TorchLean.Tensor.getScalar t j)) -
        Tensor.item (Activation.maxVecSpec t) := by
    change Real.log _ = _
    rw [sum_spec_maxShiftedExpVecSpec, Real.log_mul (ne_of_gt hsumPos) (Real.exp_ne_zero _),
      Real.log_exp, sub_eq_add_neg]
  change TorchLean.Tensor.getScalar
    (TorchLean.Tensor.map2Spec (· - ·)
      (TorchLean.Tensor.map2Spec (· - ·) t (Spec.replicate (Activation.maxVecSpec t)))
      (Spec.replicate (Tensor.scalar
        (MathFunctions.log (TorchLean.Tensor.sumSpec (Activation.maxShiftedExpVecSpec t)))))) i = _
  rw [TorchLean.Tensor.getScalar_map2Spec, TorchLean.Tensor.getScalar_map2Spec, hlog]
  simp only [Spec.replicate, TorchLean.Tensor.getScalar_const, Tensor.item_scalar]
  ring

/-! ## Axis softmax on vectors and matrices

The axis-parametric operators move the selected axis to the innermost position and back. For a
vector (axis `0`) and for the key axis of a matrix (axis `1`) no permutation is needed, so the
operators reduce definitionally to the vector kernels. These lemmas record that reduction so that
downstream files do not depend on how the permutation bookkeeping is implemented. -/

/-- Axis-`0` softmax on a vector is the vector kernel. -/
theorem softmaxSpec_zero_vec {n : Nat} (t : Tensor ℝ [n]) :
    Activation.softmaxSpec (α := ℝ) (s := [n]) 0 t = Activation.softmaxVecSpec t := rfl

/-- Axis-`0` softmax backward on a vector is the vector kernel. -/
theorem softmaxBackwardSpec_zero_vec {n : Nat} (x dY : Tensor ℝ [n]) :
    Activation.softmaxBackwardSpec (α := ℝ) (s := [n]) 0 x dY =
      Activation.Internal.softmaxInnermostBackwardSpec x dY := rfl

/-- Axis-`0` log-softmax on a vector is the vector kernel. -/
theorem logSoftmaxSpec_zero_vec {n : Nat} (t : Tensor ℝ [n]) :
    Activation.logSoftmaxSpec (α := ℝ) (s := [n]) 0 t = Activation.logSoftmaxVecSpec t := rfl

/-- Axis-`0` log-softmax backward on a vector is the vector kernel. -/
theorem logSoftmaxBackwardSpec_zero_vec {n : Nat} (y dY : Tensor ℝ [n]) :
    Activation.logSoftmaxBackwardSpec (α := ℝ) (s := [n]) 0 y dY =
      Activation.Internal.logSoftmaxInnermostBackwardSpec y dY := rfl

/-- Axis-`1` softmax on a matrix is the row-wise innermost kernel. -/
theorem softmaxSpec_one_matrix {m n : Nat} (A : Tensor ℝ [m, n]) :
    Activation.softmaxSpec (α := ℝ) (s := [m, n]) 1 A =
      Activation.Internal.softmaxInnermostSpec A := rfl

/-- Row `i` of an axis-`1` matrix softmax is the vector softmax of row `i`. -/
theorem get_softmaxSpec_one {m n : Nat} (A : Tensor ℝ [m, n]) (i : Fin m) :
    Spec.get (Activation.softmaxSpec (α := ℝ) (s := [m, n]) 1 A) i =
      Activation.softmaxVecSpec (Spec.get A i) := by
  rw [softmaxSpec_one_matrix]
  exact Activation.unstack_softmaxInnermostSpec_matrix A i

/-! ## Probability-simplex properties -/

/-- Every coordinate of a nonempty real softmax vector is strictly positive. -/
theorem softmax_vec_spec_pos {n : Nat}
    (t : Tensor ℝ [Nat.succ n]) (i : Fin (Nat.succ n)) :
    0 < TorchLean.Tensor.getScalar (Activation.softmaxVecSpec (α := ℝ) (n := Nat.succ n) t) i := by
  classical
  rcases softmax_vec_spec_normalized t with ⟨weights, hpos, hcoord⟩
  rw [hcoord i]
  exact div_pos (hpos i) (Finset.sum_pos (fun j _ => hpos j) Finset.univ_nonempty)

/-- `softmaxVecSpec` produces a vector whose entries sum to `1` over `ℝ`. -/
theorem sum_spec_softmax_vec_spec {n : Nat}
    (t : Tensor ℝ [Nat.succ n]) :
    TorchLean.Tensor.sumSpec (Activation.softmaxVecSpec (α := ℝ) (n := Nat.succ n) t) = 1 := by
  classical
  rcases softmax_vec_spec_normalized t with ⟨weights, hpos, hcoord⟩
  rw [Spec.sum_spec_vec]
  simp_rw [hcoord]
  calc
    (∑ i, weights i / ∑ j, weights j) = (∑ i, weights i) / ∑ j, weights j := by
      simpa using
        (Finset.sum_div (s := (Finset.univ : Finset (Fin (Nat.succ n))))
          (f := weights) (a := ∑ j, weights j)).symm
    _ = 1 := div_self (ne_of_gt (Finset.sum_pos (fun j _ => hpos j) Finset.univ_nonempty))

/-- Every coordinate of a nonempty real softmax vector lies in the closed unit interval. -/
theorem softmax_vec_spec_mem_unitInterval {n : Nat}
    (t : Tensor ℝ [Nat.succ n]) (i : Fin (Nat.succ n)) :
    TorchLean.Tensor.getScalar (Activation.softmaxVecSpec (α := ℝ) (n := Nat.succ n) t) i
      ∈ Set.Icc 0 1 := by
  classical
  let y := Activation.softmaxVecSpec (α := ℝ) (n := Nat.succ n) t
  have hpos : ∀ j, 0 < TorchLean.Tensor.getScalar y j := fun j => softmax_vec_spec_pos t j
  have hsum : ∑ j, TorchLean.Tensor.getScalar y j = 1 := by
    simpa [Spec.sum_spec_vec] using sum_spec_softmax_vec_spec t
  constructor
  · exact le_of_lt (hpos i)
  · calc
      TorchLean.Tensor.getScalar y i <= ∑ j, TorchLean.Tensor.getScalar y j :=
        Finset.single_le_sum (fun j _ => le_of_lt (hpos j)) (Finset.mem_univ i)
      _ = 1 := hsum

/-! ## Backward conservation -/

/-- The concrete stable softmax backward is tangent to the probability simplex.

`Activation.softmaxBackwardSpec 0` is the vector VJP used by the spec and tape layers. Its
coordinate sum is zero because the stable forward weights sum to one. This statement is about the
actual tensor definition, not the separate analytic `EuclideanSpace` presentation of the same
derivative. -/
theorem sum_spec_softmax_backward_spec {n : Nat}
    (x dY : Tensor ℝ [Nat.succ n]) :
    TorchLean.Tensor.sumSpec
      (Activation.softmaxBackwardSpec (α := ℝ) (s := [Nat.succ n]) 0 x dY) = 0 := by
  change TorchLean.Tensor.sumSpec
    (Activation.Internal.softmaxInnermostBackwardSpec x dY) = 0
  classical
  let y := Activation.softmaxVecSpec (α := ℝ) (n := Nat.succ n) x
  let s : ℝ := TorchLean.Tensor.sumSpec (TorchLean.Tensor.mulSpec dY y)
  have hy : (∑ i, TorchLean.Tensor.getScalar y i) = 1 := by
    simpa [y, Spec.sum_spec_vec] using sum_spec_softmax_vec_spec x
  have hs : s = ∑ i, TorchLean.Tensor.getScalar y i * TorchLean.Tensor.getScalar dY i := by
    rw [show s = TorchLean.Tensor.sumSpec (TorchLean.Tensor.mulSpec dY y) by rfl,
      Spec.sum_spec_vec]
    refine Finset.sum_congr rfl ?_
    intro i _
    rw [Spec.getScalar_mul_spec]
    ring
  have hsub : ∀ i,
      TorchLean.Tensor.getScalar
        (TorchLean.Tensor.subSpec dY (Spec.replicate (Tensor.scalar s))) i =
          TorchLean.Tensor.getScalar dY i - s := by
    intro i
    change TorchLean.Tensor.getScalar
      (TorchLean.Tensor.map2Spec (· - ·) dY (Spec.replicate (Tensor.scalar s))) i =
        dY.getScalar i - s
    rw [TorchLean.Tensor.getScalar_map2Spec]
    simp [Spec.replicate]
  rw [show Activation.Internal.softmaxInnermostBackwardSpec x dY =
      TorchLean.Tensor.mulSpec y
        (TorchLean.Tensor.subSpec dY (Spec.replicate (Tensor.scalar s))) by
          simp [Activation.Internal.softmaxInnermostBackwardSpec, y, s]]
  rw [Spec.sum_spec_vec]
  simp_rw [Spec.getScalar_mul_spec, hsub]
  calc
    (∑ i, TorchLean.Tensor.getScalar y i * (TorchLean.Tensor.getScalar dY i - s)) =
        (∑ i, TorchLean.Tensor.getScalar y i * TorchLean.Tensor.getScalar dY i)
          - s * (∑ i, TorchLean.Tensor.getScalar y i) := by
      calc
        (∑ i, TorchLean.Tensor.getScalar y i * (TorchLean.Tensor.getScalar dY i - s)) =
            ∑ i, (TorchLean.Tensor.getScalar y i * TorchLean.Tensor.getScalar dY i
              - s * TorchLean.Tensor.getScalar y i) := by
          refine Finset.sum_congr rfl ?_
          intro i _
          ring
        _ = (∑ i, TorchLean.Tensor.getScalar y i * TorchLean.Tensor.getScalar dY i) -
            ∑ i, s * TorchLean.Tensor.getScalar y i := by rw [Finset.sum_sub_distrib]
        _ = (∑ i, TorchLean.Tensor.getScalar y i * TorchLean.Tensor.getScalar dY i) -
            s * (∑ i, TorchLean.Tensor.getScalar y i) := by rw [Finset.mul_sum]
    _ = 0 := by rw [hy, hs]; ring

/-- Coordinatewise bound for the concrete stable softmax VJP.

If every upstream coordinate has magnitude at most `G`, each input-gradient coordinate has
magnitude at most `2G`. The estimate does not grow with the axis length because the softmax output
is a nonnegative vector of total mass one. -/
theorem abs_getScalar_softmax_backward_spec_le_two_mul {n : Nat}
    (x dY : Tensor ℝ [Nat.succ n]) (G : ℝ)
    (hdY : ∀ i, |TorchLean.Tensor.getScalar dY i| <= G) (i : Fin (Nat.succ n)) :
    |TorchLean.Tensor.getScalar
      (Activation.softmaxBackwardSpec (α := ℝ) (s := [Nat.succ n]) 0 x dY) i| <=
        2 * G := by
  change |TorchLean.Tensor.getScalar
    (Activation.Internal.softmaxInnermostBackwardSpec x dY) i| <= 2 * G
  classical
  let y := Activation.softmaxVecSpec (α := ℝ) (n := Nat.succ n) x
  let s : ℝ := TorchLean.Tensor.sumSpec (TorchLean.Tensor.mulSpec dY y)
  have hyPos : ∀ j, 0 < TorchLean.Tensor.getScalar y j := by
    intro j
    exact softmax_vec_spec_pos x j
  have hySum : (∑ j, TorchLean.Tensor.getScalar y j) = 1 := by
    simpa [y, Spec.sum_spec_vec] using sum_spec_softmax_vec_spec x
  have hs : s = ∑ j, TorchLean.Tensor.getScalar y j * TorchLean.Tensor.getScalar dY j := by
    rw [show s = TorchLean.Tensor.sumSpec (TorchLean.Tensor.mulSpec dY y) by rfl,
      Spec.sum_spec_vec]
    refine Finset.sum_congr rfl ?_
    intro j _
    rw [Spec.getScalar_mul_spec]
    ring
  have hsAbs : |s| <= G := by
    rw [hs]
    calc
      |∑ j, TorchLean.Tensor.getScalar y j * TorchLean.Tensor.getScalar dY j| <=
          ∑ j, |TorchLean.Tensor.getScalar y j * TorchLean.Tensor.getScalar dY j| :=
        Finset.abs_sum_le_sum_abs _ _
      _ = ∑ j, TorchLean.Tensor.getScalar y j * |TorchLean.Tensor.getScalar dY j| := by
        refine Finset.sum_congr rfl ?_
        intro j _
        rw [abs_mul, abs_of_pos (hyPos j)]
      _ <= ∑ j, TorchLean.Tensor.getScalar y j * G := by
        refine Finset.sum_le_sum ?_
        intro j _
        exact mul_le_mul_of_nonneg_left (hdY j) (le_of_lt (hyPos j))
      _ = G := by rw [← Finset.sum_mul, hySum, one_mul]
  have hyLeOne : TorchLean.Tensor.getScalar y i <= 1 := by
    calc
      TorchLean.Tensor.getScalar y i <= ∑ j, TorchLean.Tensor.getScalar y j :=
        Finset.single_le_sum (fun j _ => le_of_lt (hyPos j)) (Finset.mem_univ i)
      _ = 1 := hySum
  have hsub : TorchLean.Tensor.getScalar
      (TorchLean.Tensor.subSpec dY (Spec.replicate (Tensor.scalar s))) i =
        TorchLean.Tensor.getScalar dY i - s := by
    change TorchLean.Tensor.getScalar
      (TorchLean.Tensor.map2Spec (· - ·) dY (Spec.replicate (Tensor.scalar s))) i =
        dY.getScalar i - s
    rw [TorchLean.Tensor.getScalar_map2Spec]
    simp [Spec.replicate]
  have hbackward :
      Activation.Internal.softmaxInnermostBackwardSpec x dY =
        TorchLean.Tensor.mulSpec y
          (TorchLean.Tensor.subSpec dY (Spec.replicate (Tensor.scalar s))) := by
    simp [Activation.Internal.softmaxInnermostBackwardSpec, y, s]
  have hdiff : |TorchLean.Tensor.getScalar dY i - s| <= 2 * G := by
    calc
      |TorchLean.Tensor.getScalar dY i - s| <= |TorchLean.Tensor.getScalar dY i| + |s| :=
        abs_sub _ _
      _ <= G + G := add_le_add (hdY i) hsAbs
      _ = 2 * G := by ring
  rw [hbackward, Spec.getScalar_mul_spec, hsub, abs_mul, abs_of_pos (hyPos i)]
  calc
    TorchLean.Tensor.getScalar y i * |TorchLean.Tensor.getScalar dY i - s| <=
        1 * |TorchLean.Tensor.getScalar dY i - s| :=
      mul_le_mul_of_nonneg_right hyLeOne (abs_nonneg _)
    _ <= 1 * (2 * G) := mul_le_mul_of_nonneg_left hdiff zero_le_one
    _ = 2 * G := one_mul _

/-!
Axis-`1` softmax on matrices is rowwise, so each row sums to `1`.

This is the attention-shaped theorem: for score matrices, the key axis is the last/vector axis, and
softmax is applied independently to every query row.
-/
theorem sum_spec_softmax_spec_row {nQ nK : Nat} (hK : nK ≠ 0)
    (scores : Tensor ℝ [nQ, nK]) (i : Fin nQ) :
    TorchLean.Tensor.sumSpec
        (Spec.get (Activation.softmaxSpec (α := ℝ) (s := [nQ, nK]) 1 scores) i)
      = 1 := by
  cases nK with
  | zero => exact (hK rfl).elim
  | succ nK' =>
      have hrow :
          Spec.get (Activation.softmaxSpec (α := ℝ) (s := [nQ, nK' + 1]) 1 scores) i =
            Activation.softmaxVecSpec (TorchLean.Tensor.unstack scores i) := by
        have hswaps :
            Shape.moveAxisToInnermostSwaps (Shape.rank [nQ, nK' + 1]) 1 = [] := by
          rfl
        unfold Activation.softmaxSpec
        rw [hswaps]
        simp only [TorchLean.Tensor.permuteByAdjacentSwaps, List.reverse_nil, Spec.get]
        exact Activation.unstack_softmaxInnermostSpec_matrix scores i
      rw [hrow]
      exact sum_spec_softmax_vec_spec (TorchLean.Tensor.unstack scores i)

end Proofs
