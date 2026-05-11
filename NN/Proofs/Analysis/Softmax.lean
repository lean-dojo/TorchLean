/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Algebra.BigOperators.Field
public import Mathlib.Algebra.Order.BigOperators.Group.Finset
public import Mathlib.Analysis.SpecialFunctions.Exp
public import NN.Proofs.Tensor.Basic
public import NN.Proofs.Utils.MathFunctions
public import NN.Spec.Layers.Activation

/-!
# Softmax analysis properties

This module proves theorem-level facts about TorchLean's spec-level softmax operators. The
definitions themselves live in `NN.Spec.Layers.Activation`; this file belongs under
`NN.Proofs.Analysis` because it imports real-analysis and finite-sum proof machinery to establish
properties of those definitions.

Current theorem surface:

- `sum_spec_softmax_vec_spec`: a nonempty vector softmax sums to `1`;
- `sum_spec_softmax_spec_row`: matrix softmax is rowwise, so each nonempty row sums to `1`;
- `sum_spec_softmax_spec_row_of_ne_zero`: the same row theorem with nonemptiness supplied as
  `nK ≠ 0`.

We intentionally state these over `ℝ`: positivity of `exp` and division by a positive denominator
are the mathematical facts that make the probabilistic interpretation precise.
-/

@[expose] public section

open scoped BigOperators

noncomputable section

namespace Proofs

open Spec
open Tensor
open Activation

/-! ## Scalar helpers

`softmaxVecSpec` is written over tensors, so even one coordinate has type `Tensor ℝ .scalar`.
Local helper definitions expose scalar coordinates to the proof without adding public API.
-/

/--
Eliminate a scalar tensor using the same matcher as `Activation.softmaxVecSpec`.

This keeps the proof aligned with the generated matcher that Lean creates for the spec definition,
which avoids fragile pattern-matching rewrites later in the file.
-/
private abbrev scalarElim {β : Sort _} (t : Tensor ℝ .scalar) (k : ℝ → β) : β :=
  Activation.softmaxVecSpec.match_1 (motive := fun _ => β) t k

@[simp] private theorem scalarElim_scalar {β : Sort _} (k : ℝ → β) (v : ℝ) :
    scalarElim (β := β) (Tensor.scalar v) k = k v := rfl

/-- Extract the real value from a scalar tensor for local proof steps. -/
private abbrev scalarVal (t : Tensor ℝ .scalar) : ℝ :=
  scalarElim (β := ℝ) t (fun v => v)

/-! ## Softmax sums -/
/--
`softmax_vec_spec` produces a vector whose entries sum to `1` (over `ℝ`).

This is the standard softmax identity:

`∑ᵢ softmax(x)ᵢ = 1`.

The input shape is `.dim (Nat.succ n) .scalar`, not `.dim n .scalar`, because the theorem needs a
nonempty denominator. The spec subtracts a row maximum before exponentiating for numerical
stability; the proof keeps that same `m` and then shows the denominator is a positive sum of
exponentials.
-/
theorem sum_spec_softmax_vec_spec {n : Nat}
    (t : Tensor ℝ (.dim (Nat.succ n) .scalar)) :
    Spec.Tensor.sumSpec (Activation.softmaxVecSpec (α := ℝ) (n := Nat.succ n) t) = 1 := by
  classical
  -- Rewrite the spec sum as a `Finset` sum over coordinates.
  rw [Spec.sum_spec_vec (v := Activation.softmaxVecSpec (α := ℝ) (n := Nat.succ n) t)]
  cases t with
  | dim f =>
      -- Coordinate view of the input vector.
      let x : Fin (Nat.succ n) → ℝ := fun j => scalarVal (f j)
      let first : ℝ := x ⟨0, Nat.succ_pos n⟩

      -- Spec-style `max` seed + fold (matches the definition of `softmax_vec_spec`).
      let m : ℝ :=
        (List.finRange (Nat.succ n)).foldl
          (fun acc j => scalarElim (β := ℝ) (f j) (fun v => max acc v))
          first

      -- Numerators `exp(x_j - m)` in two equivalent forms: a plain scalar form `a`,
      -- and an exact transcription of the spec-level tensor ops `aSpec`.
      let a : Fin (Nat.succ n) → ℝ := fun j => MathFunctions.exp (x j - m)
      let aSpec : Fin (Nat.succ n) → ℝ := fun j =>
        scalarVal
          (Spec.Tensor.mapSpec MathFunctions.exp
            (Spec.Tensor.map2Spec (fun x1 x2 => x1 - x2) (f j) (Tensor.scalar m)))

      have haSpec : aSpec = a := by
        funext j
        cases hj : f j with
        | scalar xj =>
            simp [aSpec, a, x, scalarVal, scalarElim, hj, Spec.Tensor.mapSpec, Spec.Tensor.map2Spec]

      -- Denominator as computed by the spec (fold over `List.finRange`).
      let denom0 : ℝ := (List.finRange (Nat.succ n)).foldl (fun acc j => acc + aSpec j) 0

      have hdenom0_sum_a : denom0 = ∑ j : Fin (Nat.succ n), a j := by
        have hfold : denom0 = ∑ j : Fin (Nat.succ n), aSpec j := by
          simp [denom0, Spec.finRange_foldl_add_eq_finset_sum]
        simp [hfold, haSpec]

      have hdenom0_pos : 0 < denom0 := by
        have huniv : (Finset.univ : Finset (Fin (Nat.succ n))).Nonempty :=
          Finset.univ_nonempty
        have hpos : 0 < ∑ j : Fin (Nat.succ n), a j := by
          refine Finset.sum_pos ?_ huniv
          intro j _
          -- `a j = exp(...)` is positive (over `ℝ`).
          simpa [a, mathfunc_exp_eq_rexp] using (Real.exp_pos (x j - m))
        simpa [hdenom0_sum_a] using hpos

      have hdenom0_ne : denom0 ≠ 0 := ne_of_gt hdenom0_pos

      -- Coordinate formula: each softmax entry is `aSpec i / denom0`.
      have hcoord : ∀ i : Fin (Nat.succ n),
          Spec.toVec (Activation.softmaxVecSpec (α := ℝ) (n := Nat.succ n) (Tensor.dim f)) i =
            aSpec i / denom0 := by
        intro i
        cases hi : f i with
        | scalar _xi =>
            simp [Activation.softmaxVecSpec, Spec.toVec, scalarVal, scalarElim, x, first, m,
              aSpec, denom0, hi,
              Spec.Tensor.subSpec, Spec.Tensor.expSpec, Spec.Tensor.divSpec,
              Spec.Tensor.mapSpec, Spec.Tensor.map2Spec, Spec.replicate]

      -- Sum the coordinate formula.
      have hsum :
          (∑ i : Fin (Nat.succ n),
            Spec.toVec (Activation.softmaxVecSpec (α := ℝ) (n := Nat.succ n) (Tensor.dim f)) i)
            =
          ∑ i : Fin (Nat.succ n), aSpec i / denom0 := by
        refine Finset.sum_congr rfl ?_
        intro i _
        simpa using hcoord i

      rw [hsum]

      -- Pull out the common denominator.
      have hsum_div :
          ∑ i : Fin (Nat.succ n), aSpec i / denom0 =
            (∑ i : Fin (Nat.succ n), aSpec i) / denom0 := by
        simpa using
          (Finset.sum_div (s := (Finset.univ : Finset (Fin (Nat.succ n))))
            (f := aSpec) (a := denom0)).symm

      have hsum_aSpec : (∑ i : Fin (Nat.succ n), aSpec i) = denom0 := by
        have hfold : denom0 = ∑ i : Fin (Nat.succ n), aSpec i := by
          simp [denom0, Spec.finRange_foldl_add_eq_finset_sum]
        exact hfold.symm

      calc
        ∑ i : Fin (Nat.succ n), aSpec i / denom0
            = (∑ i : Fin (Nat.succ n), aSpec i) / denom0 := hsum_div
        _ = denom0 / denom0 := by simp [hsum_aSpec]
        _ = 1 := by simp [div_self hdenom0_ne]

/-!
`softmaxSpec` on matrices is rowwise, so each row sums to `1`.

This is the attention-shaped theorem: for score matrices, the key axis is the last/vector axis, and
softmax is applied independently to every query row.
-/
theorem sum_spec_softmax_spec_row {nQ nK : Nat}
    (maskedScores : Tensor ℝ (.dim nQ (.dim (Nat.succ nK) .scalar))) (i : Fin nQ) :
    Spec.Tensor.sumSpec
        (Spec.get (Activation.softmaxSpec (α := ℝ)
          (s := .dim nQ (.dim (Nat.succ nK) .scalar)) maskedScores) i)
      = 1 := by
  cases maskedScores with
  | dim rows =>
      -- `softmax_spec` on a matrix is rowwise, and `get` picks a row.
      simpa [Activation.softmaxSpec] using (sum_spec_softmax_vec_spec (t := rows i))

/-!
Convenience row-sum theorem when the key dimension is written as an arbitrary `nK` plus a proof
`nK ≠ 0`.

Many model statements quantify over a natural key length `nK`; this wrapper converts that style
into the `Nat.succ _` shape required by `sum_spec_softmax_spec_row`.
-/
theorem sum_spec_softmax_spec_row_of_ne_zero {nQ nK : Nat} (hK : nK ≠ 0)
    (scores : Tensor ℝ (.dim nQ (.dim nK .scalar))) (i : Fin nQ) :
    Spec.Tensor.sumSpec
        (Spec.get (Activation.softmaxSpec (α := ℝ) (s := .dim nQ (.dim nK .scalar)) scores) i)
      = 1 := by
  cases nK with
  | zero =>
      cases (hK rfl)
  | succ nK' =>
      -- Reduce to the `Nat.succ _` specialization.
      simpa using (sum_spec_softmax_spec_row (nQ := nQ) (nK := nK') (maskedScores := scores) i)

end Proofs
