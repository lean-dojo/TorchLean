/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Models.Mlp

/-!
# Distillation / Equivalence certificates (MLP2)

This module adds a distillation-style certificate:

> prove that a Student network matches a Teacher network up to `ε`
> on an input box, i.e. `|T(x) - S(x)| ≤ ε` componentwise for all inputs `x` in the domain.

The implementation is deliberately simple and reuses TorchLean's existing, proved
IBP soundness theorem for 2-layer MLPs (`NN.MLTheory.CROWN.Theorems.bound_ibp_sound`).
-/

@[expose] public section


namespace NN.MLTheory.CROWN.Distillation

open _root_.Spec
open _root_.Spec.Tensor
open NN.MLTheory.CROWN

/-! ## Small vector helpers -/

@[simp] theorem vecGet_sub {n : Nat}
    (x y : Tensor ℝ (.dim n .scalar)) (i : Fin n) :
    (Tensor.subSpec x y).vecGet i = x.vecGet i - y.vecGet i := by
  cases x with
  | dim fx =>
    cases y with
    | dim fy =>
      cases hxi : fx i with
      | scalar xv =>
        cases hyi : fy i with
        | scalar yv =>
          simp [Tensor.vecGet, Tensor.subSpec, Spec.Tensor.subSpec, Spec.Tensor.map2Spec,
            Spec.get, Spec.getAtSpec, Tensor.toScalar, hxi, hyi]

/-! ## Interval arithmetic on output boxes -/

/-- Interval subtraction: `boxSub T S` encloses all pointwise differences `x - y` with `x ∈ T`, `y ∈
  S`. -/
def boxSub {n : Nat}
    (T S : Box ℝ (.dim n .scalar)) : Box ℝ (.dim n .scalar) :=
  { lo := Tensor.subSpec T.lo S.hi
    hi := Tensor.subSpec T.hi S.lo }

/-- Predicate asserting all coordinates of `B` lie in `[-eps, eps]`. -/
def boxWithinAbs {n : Nat} (B : Box ℝ (.dim n .scalar)) (eps : ℝ) : Prop :=
  ∀ i : Fin n, (-eps ≤ B.lo.vecGet i) ∧ (B.hi.vecGet i ≤ eps)

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
    {x y : Tensor ℝ (.dim n .scalar)}
    (hx : Box.contains (α := ℝ) T x)
    (hy : Box.contains (α := ℝ) S y) :
    Box.contains (α := ℝ) (boxSub (n := n) T S) (Tensor.subSpec x y) := by
  cases T with
  | mk Tlo Thi =>
    cases S with
    | mk Slo Shi =>
      cases x with
      | dim xf =>
        cases y with
        | dim yf =>
          cases Tlo with
          | dim TloF =>
            cases Thi with
            | dim ThiF =>
              cases Slo with
              | dim SloF =>
                cases Shi with
                | dim ShiF =>
                  -- Unfold `Box.contains` on vectors and work pointwise.
                  have hx' :
                      ∀ i : Fin n, Box.contains (α := ℝ) { lo := TloF i, hi := ThiF i } (xf i) := by
                    simpa [NN.MLTheory.CROWN.Box.contains] using hx
                  have hy' :
                      ∀ i : Fin n, Box.contains (α := ℝ) { lo := SloF i, hi := ShiF i } (yf i) := by
                    simpa [NN.MLTheory.CROWN.Box.contains] using hy
                  -- Goal is also pointwise.
                  refine (show ∀ i : Fin n, _ from ?_)
                  intro i
                  cases hTlo : TloF i with
                  | scalar tlo =>
                    cases hThi : ThiF i with
                    | scalar thi =>
                      cases hSlo : SloF i with
                      | scalar slo =>
                        cases hShi : ShiF i with
                        | scalar shi =>
                          cases hxf : xf i with
                          | scalar xv =>
                            cases hyf : yf i with
                            | scalar yv =>
                              have hx_i := hx' i
                              have hy_i := hy' i
                              -- Rewrite to expose scalar endpoints/values.
                              simp [hTlo, hThi, hSlo, hShi, hxf, hyf,
                                NN.MLTheory.CROWN.Box.contains] at hx_i hy_i
                              have hdiff : (tlo - shi ≤ xv - yv) ∧ (xv - yv ≤ thi - slo) := by
                                constructor <;> linarith [hx_i, hy_i]
                              simpa [boxSub, Tensor.subSpec, Spec.Tensor.subSpec,
                                Spec.Tensor.map2Spec,
                                NN.MLTheory.CROWN.Box.contains, hTlo, hThi, hSlo, hShi, hxf, hyf]
                                  using hdiff

/-- Project a `Box.contains` hypothesis to scalar inequalities at a single coordinate. -/
theorem boxContains_vecGet {n : Nat} {B : Box ℝ (.dim n .scalar)} {x : Tensor ℝ (.dim n .scalar)}
    (h : Box.contains (α := ℝ) B x) (i : Fin n) :
    B.lo.vecGet i ≤ x.vecGet i ∧ x.vecGet i ≤ B.hi.vecGet i := by
  cases B with
  | mk Blo Bhi =>
    cases x with
    | dim xf =>
      cases Blo with
      | dim BloF =>
        cases Bhi with
        | dim BhiF =>
          have h' : ∀ i : Fin n, Box.contains (α := ℝ) { lo := BloF i, hi := BhiF i } (xf i) := by
            simpa [NN.MLTheory.CROWN.Box.contains] using h
          cases hlo : BloF i with
          | scalar lo =>
            cases hhi : BhiF i with
            | scalar hi' =>
              cases hxv : xf i with
              | scalar xv =>
                have hi := h' i
                simp [NN.MLTheory.CROWN.Box.contains, hlo, hhi, hxv] at hi
                simpa [Tensor.vecGet, Spec.get, Spec.getAtSpec, hlo, hhi, hxv] using hi

/-! ## Distillation certificate for 2-layer MLPs -/

/--
Computable checker: returns `true` if IBP proves the student matches the teacher
up to `eps` (componentwise) on the given input box.
-/
noncomputable def checkEquivalenceMlp2 {inDim hidDim outDim : Nat}
    (teacher student : NN.MLTheory.CROWN.MLP2 ℝ inDim hidDim outDim)
    (xB : Box ℝ (.dim inDim .scalar))
    (eps : ℝ) : Bool :=
  let tB := NN.MLTheory.CROWN.boundIbp (α := ℝ) teacher xB
  let sB := NN.MLTheory.CROWN.boundIbp (α := ℝ) student xB
  checkBoxWithinAbs (n := outDim) (boxSub (n := outDim) tB sB) eps

/--
Soundness: if `checkEquivalence_mlp2` returns `true`, then for all inputs `x` in `xB`,
the outputs are `eps`-close componentwise: `|T(x)_i - S(x)_i| ≤ eps`.
-/
theorem checkEquivalence_mlp2_sound {inDim hidDim outDim : Nat}
    (teacher student : NN.MLTheory.CROWN.MLP2 ℝ inDim hidDim outDim)
    (xB : Box ℝ (.dim inDim .scalar))
    (eps : ℝ)
    (hcheck : checkEquivalenceMlp2 (inDim := inDim) (hidDim := hidDim) (outDim := outDim)
      teacher student xB eps = true) :
    ∀ x : Tensor ℝ (.dim inDim .scalar),
      Box.contains (α := ℝ) xB x →
      ∀ i : Fin outDim,
        |vecGet (Tensor.subSpec
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
      simpa [checkEquivalenceMlp2] using hcheck
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
    boxContains_vecGet (n := outDim) (B := boxSub (n := outDim)
      (NN.MLTheory.CROWN.boundIbp (α := ℝ) teacher xB)
      (NN.MLTheory.CROWN.boundIbp (α := ℝ) student xB))
      (x := Tensor.subSpec
        (NN.MLTheory.CROWN.forward (α := ℝ) teacher x)
        (NN.MLTheory.CROWN.forward (α := ℝ) student x))
      hdiff i

  have hlo : -eps ≤ (Tensor.subSpec
        (NN.MLTheory.CROWN.forward (α := ℝ) teacher x)
        (NN.MLTheory.CROWN.forward (α := ℝ) student x)).vecGet i :=
    le_trans (hwithin i).1 hmem_i.1

  have hhi : (Tensor.subSpec
        (NN.MLTheory.CROWN.forward (α := ℝ) teacher x)
        (NN.MLTheory.CROWN.forward (α := ℝ) student x)).vecGet i ≤ eps :=
    le_trans hmem_i.2 (hwithin i).2

  exact (abs_le).2 ⟨hlo, hhi⟩

end NN.MLTheory.CROWN.Distillation
