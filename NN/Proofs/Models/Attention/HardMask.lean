/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Analysis.Softmax
public import NN.Spec.Layers.Attention
public import Mathlib.Algebra.Order.Algebra
public import Mathlib.Analysis.SpecialFunctions.Pow.NNReal
public import Mathlib.Data.Sym.Sym2.Init
import Mathlib.Tactic.NormNum.GCD

/-!
# Hard masking with an all-true mask

`Spec.scaledDotProductAttention` normalizes scores through one of two code paths: plain axis
softmax when `ctx.mask = none`, and `Spec.hardMaskedSoftmaxSpec` when a Boolean mask is supplied.
This file proves that a mask allowing every position selects exactly the unmasked weights, so the
two branches agree wherever both apply. The statement is over `ℝ`, where the `max` used by the
stable softmax shift is the order maximum compared by the hard-mask row scan.
-/

@[expose] public section

namespace NN.Proofs.Models.Attention

open Spec TorchLean
open TorchLean.Tensor
open scoped BigOperators

noncomputable section

/-- Any scan that updates a recorded maximum by a strict comparison computes the fold of `max`. -/
private theorem foldl_of_step {n : Nat} (x : Fin n → ℝ) (f : Option ℝ → Fin n → Option ℝ)
    (hf : ∀ b i, f (some b) i = some (if x i > b then x i else b)) (l : List (Fin n)) (a : ℝ) :
    l.foldl f (some a) = some (l.foldl (fun acc i => max acc (x i)) a) := by
  induction l generalizing a with
  | nil => rfl
  | cons i l ih =>
      rw [List.foldl_cons, hf, ih, List.foldl_cons]
      have hstep : (if x i > a then x i else a) = max a (x i) := by
        rcases lt_or_ge a (x i) with h | h
        · rw [ite_eq_left h, max_eq_right h.le]
        · rw [ite_eq_right (not_lt.mpr h), max_eq_left h]
      rw [hstep]

/-- With every key allowed, the hard-mask row maximum is the stable softmax shift. -/
private theorem hardMaskedMax?_allTrue {n : Nat} (scores : Tensor ℝ [Nat.succ n]) :
    Spec.hardMaskedMax? scores (Tensor.dim fun _ => Tensor.scalar true) =
      some (Tensor.item (Activation.maxVecSpec scores)) := by
  unfold Spec.hardMaskedMax?
  simp only [TorchLean.Tensor.getScalar_dim_entry, Tensor.item_scalar, ite_true]
  rw [List.finRange_succ, List.foldl_cons]
  dsimp only
  refine (foldl_of_step (fun i => scores.getScalar i) _ (fun b i => rfl) _ _).trans ?_
  change _ = some ((List.finRange (Nat.succ n)).foldl
    (fun acc j => max acc (scores.getScalar j)) (scores.getScalar ⟨0, Nat.succ_pos n⟩))
  rw [List.finRange_succ, List.foldl_cons,
    show (⟨0, Nat.succ_pos n⟩ : Fin (Nat.succ n)) = 0 from rfl, max_self]

/-- Hard-masked softmax with every entry allowed is the stable softmax kernel. -/
theorem hardMaskedSoftmaxVecSpec_allTrue {n : Nat} (scores : Tensor ℝ [n]) :
    Spec.hardMaskedSoftmaxVecSpec scores (Tensor.dim fun _ => Tensor.scalar true) =
      Activation.softmaxVecSpec scores := by
  cases n with
  | zero =>
      apply TorchLean.Tensor.ext_vector
      intro i
      exact i.elim0
  | succ n =>
      have hnum :
          map2Spec
              (fun score allowed =>
                if allowed then
                  MathFunctions.exp (score - Tensor.item (Activation.maxVecSpec scores))
                else 0)
              scores (Tensor.dim fun _ => Tensor.scalar true) =
            Activation.maxShiftedExpVecSpec scores := by
        apply TorchLean.Tensor.ext_vector
        intro i
        rw [TorchLean.Tensor.getScalar_map2Spec, Proofs.getScalar_maxShiftedExpVecSpec]
        simp [Proofs.mathfunc_exp_eq_rexp]
      unfold Spec.hardMaskedSoftmaxVecSpec
      rw [hardMaskedMax?_allTrue]
      simp only []
      rw [hnum]
      rfl

/-- Hard-masked softmax with the all-true mask is the unmasked axis-`1` softmax. -/
theorem hardMaskedSoftmaxSpec_allTrueMask {nQ nK : Nat} (scores : Tensor ℝ [nQ, nK]) :
    Spec.hardMaskedSoftmaxSpec scores (Spec.allTrueMask nQ nK) =
      Activation.softmaxSpec (α := ℝ) (s := [nQ, nK]) 1 scores := by
  apply Spec.matrix_ext
  intro i j
  rw [Spec.get2_eq_getScalar_get, Spec.get2_eq_getScalar_get, Proofs.get_softmaxSpec_one]
  simp only [Spec.hardMaskedSoftmaxSpec, Spec.allTrueMask, Spec.get_dim,
    TorchLean.Tensor.unstack_dim]
  rw [hardMaskedSoftmaxVecSpec_allTrue]
  rfl

/-- Supplying the all-true mask to scaled dot-product attention is the same as supplying no mask. -/
theorem scaledDotProductAttention_allTrueMask {nQ nK dModel : Nat} {h1 : nQ ≠ 0} {h2 : nK ≠ 0}
    (ctx : Spec.AttentionContext ℝ nQ nK dModel h1 h2) :
    Spec.scaledDotProductAttention (h1 := h1) (h2 := h2)
        { ctx with mask := some (Spec.allTrueMask nQ nK) } =
      Spec.scaledDotProductAttention (h1 := h1) (h2 := h2) { ctx with mask := none } := by
  simp only [Spec.scaledDotProductAttention, hardMaskedSoftmaxSpec_allTrueMask]

/-- The all-true mask leaves the attention backward pass unchanged. -/
theorem scaledDotProductAttentionBackward_allTrueMask {nQ nK dModel : Nat}
    {h1 : nQ ≠ 0} {h2 : nK ≠ 0}
    (ctx : Spec.AttentionContext ℝ nQ nK dModel h1 h2) (dOut : Tensor ℝ [nQ, dModel]) :
    Spec.scaledDotProductAttentionBackward (h1 := h1) (h2 := h2)
        { ctx with mask := some (Spec.allTrueMask nQ nK) } dOut =
      Spec.scaledDotProductAttentionBackward (h1 := h1) (h2 := h2)
        { ctx with mask := none } dOut := by
  simp only [Spec.scaledDotProductAttentionBackward, hardMaskedSoftmaxSpec_allTrueMask]

/-- The all-true mask leaves the attention forward-mode derivative unchanged. -/
theorem scaledDotProductAttentionJvp_allTrueMask {nQ nK dModel : Nat}
    {h1 : nQ ≠ 0} {h2 : nK ≠ 0}
    (ctx : Spec.AttentionContext ℝ nQ nK dModel h1 h2)
    (dQ : Tensor ℝ [nQ, dModel]) (dK dV : Tensor ℝ [nK, dModel]) :
    Spec.scaledDotProductAttentionJvp (h1 := h1) (h2 := h2)
        { ctx with mask := some (Spec.allTrueMask nQ nK) } dQ dK dV =
      Spec.scaledDotProductAttentionJvp (h1 := h1) (h2 := h2)
        { ctx with mask := none } dQ dK dV := by
  simp only [Spec.scaledDotProductAttentionJvp, hardMaskedSoftmaxSpec_allTrueMask]

end

end NN.Proofs.Models.Attention
