/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Softmax
public import TorchLean.Tensor.Layers.Attention

/-!
# Hard masking with an all-true mask

`TorchLean.Attention.scaledDotProduct` normalizes the scores along one of two branches: the plain
axis softmax when `ctx.mask = none`, and `TorchLean.Softmax.hardMasked` when a Boolean mask is
supplied. A mask that allows every position gives the unmasked weights, so the two branches agree,
and so do the backward pass and the forward-mode derivative. The statements are over `ℝ`, where the
strict comparison in the hard-mask row scan finds the same maximum as the `max` of the stable
softmax shift.
-/

@[expose] public section

open FloatLib.Numerics (MathFunctions)

namespace TorchLean

open TorchLean.Tensor

noncomputable section

/-- Any scan that updates a recorded maximum by a strict comparison computes the fold of `max`. -/
private theorem Attention.foldl_of_step {n : Nat} (x : Fin n → ℝ) (f : Option ℝ → Fin n → Option ℝ)
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
private theorem Softmax.hardMaskedMax?_allTrue {n : Nat} (scores : Tensor ℝ [Nat.succ n]) :
    hardMaskedMax? scores (Tensor.dim fun _ => Tensor.scalar true) =
      some (Tensor.item (maxVec scores)) := by
  unfold hardMaskedMax?
  simp only [Tensor.getScalar_dim_entry, Tensor.item_scalar, ite_true]
  rw [List.finRange_succ, List.foldl_cons]
  dsimp only
  refine (Attention.foldl_of_step (fun i => scores.getScalar i) _ (fun b i => rfl) _ _).trans ?_
  rw [item_maxVec]
  rw [List.finRange_succ, List.foldl_cons,
    show (⟨0, Nat.succ_pos n⟩ : Fin (Nat.succ n)) = 0 from rfl, max_self]

/-- Hard-masked softmax with every entry allowed is the stable softmax kernel. -/
theorem Softmax.hardMaskedVec_allTrue {n : Nat} (scores : Tensor ℝ [n]) :
    hardMaskedVec scores (Tensor.dim fun _ => Tensor.scalar true) =
      vec scores := by
  cases n with
  | zero =>
      apply Tensor.ext_vector
      intro i
      exact i.elim0
  | succ n =>
      have hnum :
          zipWith
              (fun score allowed =>
                if allowed then
                  MathFunctions.exp (score - Tensor.item (maxVec scores))
                else 0)
              scores (Tensor.dim fun _ => Tensor.scalar true) =
            maxShiftedExpVec scores := by
        apply Tensor.ext_vector
        intro i
        rw [Tensor.getScalar_zipWith, getScalar_maxShiftedExpVec]
        simp [MathFunctions.exp_real]
      unfold hardMaskedVec
      rw [hardMaskedMax?_allTrue]
      dsimp only
      rw [hnum]
      rfl

/-- Hard-masked softmax with the all-true mask is the unmasked axis-`1` softmax. -/
theorem Softmax.hardMasked_allTrueMask {nQ nK : Nat} (scores : Tensor ℝ [nQ, nK]) :
    hardMasked scores (Attention.allTrueMask nQ nK) =
      Tensor.softmax (α := ℝ) (s := [nQ, nK]) scores ⟨1, Shape.AxisInBounds.proof⟩ := by
  apply Tensor.ext_get2
  intro i j
  rw [Tensor.get2_eq_getScalar_unstack, Tensor.get2_eq_getScalar_unstack, unstack_softmax_one]
  simp only [hardMasked, Attention.allTrueMask, Tensor.unstack_dim]
  rw [hardMaskedVec_allTrue]

/-- Supplying the all-true mask to scaled dot-product attention is the same as supplying no mask. -/
theorem Attention.scaledDotProduct_allTrueMask {nQ nK dModel : Nat} {h1 : nQ ≠ 0} {h2 : nK ≠ 0}
    (ctx : Inputs ℝ nQ nK dModel h1 h2) :
    scaledDotProduct (h1 := h1) (h2 := h2)
        { ctx with mask := some (allTrueMask nQ nK) } =
      scaledDotProduct (h1 := h1) (h2 := h2) { ctx with mask := none } := by
  simp only [scaledDotProduct, Softmax.hardMasked_allTrueMask]

/-- The all-true mask leaves the attention backward pass unchanged. -/
theorem Attention.scaledDotProductBackward_allTrueMask {nQ nK dModel : Nat}
    {h1 : nQ ≠ 0} {h2 : nK ≠ 0}
    (ctx : Inputs ℝ nQ nK dModel h1 h2) (dOut : Tensor ℝ [nQ, dModel]) :
    scaledDotProductBackward (h1 := h1) (h2 := h2)
        { ctx with mask := some (allTrueMask nQ nK) } dOut =
      scaledDotProductBackward (h1 := h1) (h2 := h2)
        { ctx with mask := none } dOut := by
  simp only [scaledDotProductBackward, Softmax.hardMasked_allTrueMask]

/-- The all-true mask leaves the attention forward-mode derivative unchanged. -/
theorem Attention.scaledDotProductJVP_allTrueMask {nQ nK dModel : Nat}
    {h1 : nQ ≠ 0} {h2 : nK ≠ 0}
    (ctx : Inputs ℝ nQ nK dModel h1 h2)
    (dQ : Tensor ℝ [nQ, dModel]) (dK dV : Tensor ℝ [nK, dModel]) :
    scaledDotProductJVP (h1 := h1) (h2 := h2)
        { ctx with mask := some (allTrueMask nQ nK) } dQ dK dV =
      scaledDotProductJVP (h1 := h1) (h2 := h2)
        { ctx with mask := none } dQ dK dV := by
  simp only [scaledDotProductJVP, Softmax.hardMasked_allTrueMask]

end

end TorchLean
