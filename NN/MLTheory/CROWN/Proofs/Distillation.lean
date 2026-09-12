/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Models.Mlp

/-!
# Distillation / Equivalence certificates (TwoLayerMLP)

This module adds a distillation-style certificate:

> prove that a Student network matches a Teacher network up to `ε`
> on an input box, i.e. `|T(x) - S(x)| ≤ ε` componentwise for all inputs `x` in the domain.

The implementation is kept simple and reuses TorchLean's existing, proved
IBP soundness theorem for 2-layer MLPs (`NN.MLTheory.CROWN.Theorems.bound_ibp_sound`).
-/

@[expose] public section


namespace NN.MLTheory.CROWN.Distillation

open Spec TorchLean
open TorchLean.Tensor
open NN.MLTheory.CROWN

/-! ## Small vector helpers -/

@[simp] theorem getScalar_sub {n : Nat}
    (x y : Tensor ℝ [n]) (i : Fin n) :
    (Tensor.subSpec x y).getScalar i = x.getScalar i - y.getScalar i := by
  simp [Tensor.subSpec]

/-! ## Interval arithmetic on output boxes -/

/-- Interval subtraction: `boxSub T S` encloses every pointwise difference $x-y$ with $x\in T$ and
$y\in S$. -/
def boxSub {n : Nat}
    (T S : Box ℝ (.dim n .scalar)) : Box ℝ (.dim n .scalar) :=
  { lo := Tensor.subSpec T.lo S.hi
    hi := Tensor.subSpec T.hi S.lo }

/-- Predicate asserting all coordinates of `B` lie in `[-eps, eps]`. -/
def boxWithinAbs {n : Nat} (B : Box ℝ (.dim n .scalar)) (eps : ℝ) : Prop :=
  ∀ i : Fin n, (-eps ≤ B.lo.getScalar i) ∧ (B.hi.getScalar i ≤ eps)

/-- Classical decision procedure for `boxWithinAbs`.

Noncomputable and `decide`-based because the carrier here is `ℝ`; the point is to have a
`Bool`-valued
checker to state agreement against, not to run it. The executable float version lives in the CROWN
engine. -/
noncomputable def checkBoxWithinAbs {n : Nat} (B : Box ℝ (.dim n .scalar)) (eps : ℝ) : Bool := by
  classical
  exact decide (boxWithinAbs (n := n) B eps)

/-- Correctness of `checkBoxWithinAbs`. -/
theorem checkBoxWithinAbs_spec {n : Nat} {B : Box ℝ (.dim n .scalar)} {eps : ℝ} :
    checkBoxWithinAbs (n := n) B eps = true ↔ boxWithinAbs (n := n) B eps := by
  classical
  simp [checkBoxWithinAbs, decide_eq_true_eq]

/-- If `x ∈ T` and `y ∈ S`, then `x - y ∈ boxSub T S`. -/
theorem boxSub_contains {n : Nat}
    {T S : Box ℝ (.dim n .scalar)}
    {x y : Tensor ℝ [n]}
    (hx : Box.contains (α := ℝ) T x)
    (hy : Box.contains (α := ℝ) S y) :
    Box.contains (α := ℝ) (boxSub (n := n) T S) (Tensor.subSpec x y) := by
  intro i
  have hx_i := hx i
  have hy_i := hy i
  have hx_scalar :
      T.lo.getScalar i ≤ x.getScalar i ∧ x.getScalar i ≤ T.hi.getScalar i := by
    simpa [Box.contains, Tensor.getScalar, Spec.get] using hx_i
  have hy_scalar :
      S.lo.getScalar i ≤ y.getScalar i ∧ y.getScalar i ≤ S.hi.getScalar i := by
    simpa [Box.contains, Tensor.getScalar, Spec.get] using hy_i
  change
    (boxSub T S).lo.getScalar i ≤ (Tensor.subSpec x y).getScalar i ∧
      (Tensor.subSpec x y).getScalar i ≤ (boxSub T S).hi.getScalar i
  simp only [boxSub, getScalar_sub]
  constructor <;> linarith [hx_scalar.1, hx_scalar.2, hy_scalar.1, hy_scalar.2]

/-- Project a `Box.contains` hypothesis to scalar inequalities at a single coordinate. -/
theorem boxContains_getScalar {n : Nat} {B : Box ℝ (.dim n .scalar)} {x : Tensor ℝ [n]}
    (h : Box.contains (α := ℝ) B x) (i : Fin n) :
    B.lo.getScalar i ≤ x.getScalar i ∧ x.getScalar i ≤ B.hi.getScalar i := by
  exact h i

/-! ## Distillation certificate for 2-layer MLPs -/

/--
Computable checker: returns `true` if IBP proves the student matches the teacher
up to `eps` (componentwise) on the given input box.
-/
noncomputable def checkEquivalenceTwoLayerMlp {inDim hidDim outDim : Nat}
    (teacher student : NN.MLTheory.CROWN.TwoLayerMLP ℝ inDim hidDim outDim)
    (xB : Box ℝ (.dim inDim .scalar))
    (eps : ℝ) : Bool :=
  let tB := NN.MLTheory.CROWN.boundIbp (α := ℝ) teacher xB
  let sB := NN.MLTheory.CROWN.boundIbp (α := ℝ) student xB
  checkBoxWithinAbs (n := outDim) (boxSub (n := outDim) tB sB) eps

/--
Soundness: if `checkEquivalenceTwoLayerMlp` returns `true`, then for all inputs `x` in `xB`,
the outputs are `eps`-close componentwise: `|T(x)_i - S(x)_i| ≤ eps`.
-/
theorem checkEquivalence_twoLayerMlp_sound {inDim hidDim outDim : Nat}
    (teacher student : NN.MLTheory.CROWN.TwoLayerMLP ℝ inDim hidDim outDim)
    (xB : Box ℝ (.dim inDim .scalar))
    (eps : ℝ)
    (hcheck : checkEquivalenceTwoLayerMlp (inDim := inDim) (hidDim := hidDim) (outDim := outDim)
      teacher student xB eps = true) :
    ∀ x : Tensor ℝ [inDim],
      Box.contains (α := ℝ) xB x →
      ∀ i : Fin outDim,
        |getScalar (Tensor.subSpec
          (NN.MLTheory.CROWN.forward (α := ℝ) teacher x)
          (NN.MLTheory.CROWN.forward (α := ℝ) student x)) i| ≤ eps := by
  intro x hx i
  -- Extract the checked predicate.
  have hwithin :
      boxWithinAbs (n := outDim)
        (boxSub (n := outDim)
          (NN.MLTheory.CROWN.boundIbp (α := ℝ) teacher xB)
          (NN.MLTheory.CROWN.boundIbp (α := ℝ) student xB))
        eps := by
    have hcheck' :
        checkBoxWithinAbs (n := outDim)
            (boxSub (n := outDim)
              (NN.MLTheory.CROWN.boundIbp (α := ℝ) teacher xB)
              (NN.MLTheory.CROWN.boundIbp (α := ℝ) student xB))
            eps = true := by
      simpa [checkEquivalenceTwoLayerMlp] using hcheck
    exact (checkBoxWithinAbs_spec (n := outDim)
      (B := boxSub (n := outDim)
        (NN.MLTheory.CROWN.boundIbp (α := ℝ) teacher xB)
        (NN.MLTheory.CROWN.boundIbp (α := ℝ) student xB))
      (eps := eps)).1 hcheck'

  -- Sound IBP enclosures for both networks at x.
  have ht :
      Box.contains (α := ℝ)
        (NN.MLTheory.CROWN.boundIbp (α := ℝ) teacher xB)
        (NN.MLTheory.CROWN.forward (α := ℝ) teacher x) :=
    NN.MLTheory.CROWN.Theorems.bound_ibp_sound (net := teacher) (xB := xB) (x := x) hx
  have hs :
      Box.contains (α := ℝ)
        (NN.MLTheory.CROWN.boundIbp (α := ℝ) student xB)
        (NN.MLTheory.CROWN.forward (α := ℝ) student x) :=
    NN.MLTheory.CROWN.Theorems.bound_ibp_sound (net := student) (xB := xB) (x := x) hx

  -- Therefore the difference lies in the difference box.
  have hdiff :
      Box.contains (α := ℝ)
        (boxSub (n := outDim)
          (NN.MLTheory.CROWN.boundIbp (α := ℝ) teacher xB)
          (NN.MLTheory.CROWN.boundIbp (α := ℝ) student xB))
        (Tensor.subSpec
          (NN.MLTheory.CROWN.forward (α := ℝ) teacher x)
          (NN.MLTheory.CROWN.forward (α := ℝ) student x)) :=
    boxSub_contains (n := outDim) (hx := ht) (hy := hs)

  -- Read off the per-component bounds and conclude abs ≤ eps.
  have hmem_i :=
    boxContains_getScalar (n := outDim) (B := boxSub (n := outDim)
      (NN.MLTheory.CROWN.boundIbp (α := ℝ) teacher xB)
      (NN.MLTheory.CROWN.boundIbp (α := ℝ) student xB))
      (x := Tensor.subSpec
        (NN.MLTheory.CROWN.forward (α := ℝ) teacher x)
        (NN.MLTheory.CROWN.forward (α := ℝ) student x))
      hdiff i

  have hlo : -eps ≤ (Tensor.subSpec
        (NN.MLTheory.CROWN.forward (α := ℝ) teacher x)
        (NN.MLTheory.CROWN.forward (α := ℝ) student x)).getScalar i :=
    le_trans (hwithin i).1 hmem_i.1

  have hhi : (Tensor.subSpec
        (NN.MLTheory.CROWN.forward (α := ℝ) teacher x)
        (NN.MLTheory.CROWN.forward (α := ℝ) student x)).getScalar i ≤ eps :=
    le_trans hmem_i.2 (hwithin i).2

  exact (abs_le).2 ⟨hlo, hhi⟩

end NN.MLTheory.CROWN.Distillation
