/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Movement.SpecialForms
public import TorchLean.Tensor.Rep.Segment
public import TorchLean.Tensor.ShapeChange -- shake: keep
public import TorchLean.Tensor.Context -- shake: keep

/-!
# Concatenating tensors

Two tensors are joined along an axis by `cat` of the two-member family
`Fin.cons a (Fin.cons b finZeroElim)`. `cat_two_apply` reads that join as `a` below the extent of
`a` along the axis and as `b`, shifted, above it, so `roundoff` analyzes it as its two branches.
`cat_cons_apply` is the same read for a join along the leading axis, at plain coordinate pairs.

For any family of `k + 1` blocks, `cat_succ_apply` reads the first block below its extent and the
`cat` of the other blocks, shifted, above it. With `cat_one_apply`, which reads a one-block `cat`
as its block, a family of a known number of blocks is analyzed one block at a time.
`Shape.set_append_rank` and `Shape.rank_lt_rank_append` place the joined axis after a leading
shape, for a join that is not at the outermost axis.
-/

@[expose] public section

namespace TorchLean.Tensor

variable {α : Type} [Storage α] {s : Shape}

/-- Replacing the extent `k` of the axis after `leading` by `j` gives
`leading ++ (.dim j suffix)`. -/
@[tensor_shape_simps]
theorem _root_.TorchLean.Shape.set_append_rank (leading suffix : Shape) (k j : Nat) :
    (leading ++ (.dim k suffix)).set leading.rank j =
      leading ++ Shape.dim j suffix := by
  simp [Shape.rank_eq_length]

/-- An index below `n` of a two-block axis lies in the first block. -/
private theorem finSigmaFinEquiv_symm_of_lt {n m : Nat} (v : Fin (∑ i, ![n, m] i))
    (h : v.val < n) : finSigmaFinEquiv.symm v = ⟨0, ⟨v.val, h⟩⟩ := by
  rw [Equiv.symm_apply_eq]
  apply Fin.ext
  rw [finSigmaFinEquiv_apply]
  simp

/-- An index at least `n` of a two-block axis lies in the second block, shifted by `n`. -/
private theorem finSigmaFinEquiv_symm_of_le {n m : Nat} (v : Fin (∑ i, ![n, m] i))
    (h : n ≤ v.val) :
    finSigmaFinEquiv.symm v = ⟨1, ⟨v.val - n, by
      have := v.isLt
      simp only [Fin.sum_univ_two, Matrix.cons_val_zero, Matrix.cons_val_one] at this
      show v.val - n < m
      omega⟩⟩ := by
  rw [Equiv.symm_apply_eq]
  apply Fin.ext
  rw [finSigmaFinEquiv_apply]
  simp only [Fin.isValue, Fin.val_one, Fin.sum_univ_one]
  show v.val = n + (v.val - n)
  omega

/-- An output coordinate of a two-block `cat` reads the first block below its extent along `dim`
and the second block, shifted by that extent, above it. -/
theorem cat_two_apply {n m : Nat} (dim : Fin s.rank) (a : Tensor α (s.set dim n))
    (b : Tensor α (s.set dim m)) (out : Shape)
    (hout : out = s.set dim (∑ i, ![n, m] i)) (c : out.Coord) :
    cat dim ![n, m] (Fin.cons a (Fin.cons b finZeroElim)) out hout c =
      let split := Shape.Coord.setEquiv dim.val (shape := s) (∑ i, ![n, m] i)
        (Shape.Coord.reindexEq hout c)
      if h : split.1.val < n then
        a ((Shape.Coord.setEquiv dim.val (shape := s) n).symm
          (⟨split.1.val, h⟩, split.2))
      else
        b ((Shape.Coord.setEquiv dim.val (shape := s) m).symm
          (⟨split.1.val - n, by
            have := split.1.isLt
            simp only [Fin.sum_univ_two, Matrix.cons_val_zero, Matrix.cons_val_one] at this
            omega⟩, split.2)) := by
  rw [cat_apply]
  simp only []
  split_ifs with h
  · rw [finSigmaFinEquiv_symm_of_lt _ h]
    rfl
  · rw [finSigmaFinEquiv_symm_of_le _ (Nat.le_of_not_lt h)]
    rfl

/-- An output coordinate of the join of `a` and `b` along their leading axis reads `a` below `n`
and `b`, shifted by `n`, above it. -/
theorem cat_cons_apply {n m : Nat} {t : Shape} (a : Tensor α (.dim n t))
    (b : Tensor α (.dim m t)) (c : Shape.Coord (.dim (n + m) t)) :
    (cat (s := .dim 0 t) ⟨0, by simp⟩ ![n, m] (Fin.cons a (Fin.cons b finZeroElim))
        (.dim (n + m) t)) c =
      if h : c.1.val < n then a (⟨c.1.val, h⟩, c.2)
      else b (⟨c.1.val - n, by have := c.1.isLt; omega⟩, c.2) := by
  have key := cat_two_apply (s := .dim 0 t) ⟨0, by simp⟩ a b (.dim (n + m) t) (by tensor_shape) c
  rw [key]
  simp only []
  have h₁ : (Shape.Coord.setEquiv 0 (shape := .dim 0 t) (∑ i, ![n, m] i)
      (Shape.Coord.reindexEq (by tensor_shape) c)).1.val = c.1.val :=
    (Shape.Coord.val_setEquiv_fst 0 _ _).trans (Shape.Coord.index_reindexEq _ c 0)
  have h₂ : ∀ k, Shape.Coord.index (Shape.Coord.setEquiv 0 (shape := .dim 0 t)
      (∑ i, ![n, m] i) (Shape.Coord.reindexEq (by tensor_shape) c)).2 k =
        Shape.Coord.index c.2 k := fun k => by
    have e := (Shape.Coord.index_setEquiv_snd 0 (shape := .dim 0 t) (∑ i, ![n, m] i)
      (Shape.Coord.reindexEq (by tensor_shape) c) k).trans (Shape.Coord.index_reindexEq _ c _)
    simp only [Nat.not_lt_zero, ite_false] at e
    exact e
  split_ifs with ha hb hb
  · exact congrArg a (Prod.ext (Fin.ext h₁) (Shape.Coord.ext_index fun k _ => h₂ k))
  · exact absurd (h₁ ▸ ha) hb
  · exact absurd (h₁.symm ▸ hb) ha
  · exact congrArg b (Prod.ext (Fin.ext (congrArg (· - n) h₁))
      (Shape.Coord.ext_index fun k _ => h₂ k))

/-- An index below the first size of a family of `k + 1` blocks lies in the first block. -/
private theorem finSigmaFinEquiv_symm_first {k : Nat} {n : Fin (k + 1) → Nat}
    (v : Fin (∑ i, n i)) (h : v.val < n 0) : finSigmaFinEquiv.symm v = ⟨0, ⟨v.val, h⟩⟩ := by
  rw [Equiv.symm_apply_eq]
  apply Fin.ext
  rw [finSigmaFinEquiv_apply]
  simp

/-- The flat index of a block after the first: the first size plus its flat index among the
other blocks. -/
private theorem finSigmaFinEquiv_mk_succ {k : Nat} {n : Fin (k + 1) → Nat} (j : Fin k)
    (b : Fin (n j.succ)) :
    (finSigmaFinEquiv (⟨j.succ, b⟩ : (i : Fin (k + 1)) × Fin (n i)) : Nat) =
      n 0 + (finSigmaFinEquiv (⟨j, b⟩ : (i : Fin k) × Fin (n i.succ)) : Nat) := by
  rw [finSigmaFinEquiv_apply, finSigmaFinEquiv_apply]
  show (∑ x : Fin (j.val + 1), n (Fin.castLE _ x)) + b.val =
    n 0 + ((∑ x : Fin j.val, n (Fin.castLE _ x).succ) + b.val)
  rw [Fin.sum_univ_succ]
  simp only [Fin.castLE_zero, Fin.castLE_succ]
  omega

/-- An index `v` at least the first size of a family of `k + 1` blocks lies in block `j.succ` at
offset `b`, where `(j, b)` locates `v - n 0` among the other `k` blocks. -/
private theorem finSigmaFinEquiv_symm_succ {k : Nat} {n : Fin (k + 1) → Nat}
    (v : Fin (∑ i, n i)) (h : n 0 ≤ v.val) (hv : v.val - n 0 < ∑ i : Fin k, n i.succ) :
    finSigmaFinEquiv.symm v =
      ⟨((finSigmaFinEquiv (n := fun i : Fin k => n i.succ)).symm ⟨v.val - n 0, hv⟩).1.succ,
        ((finSigmaFinEquiv (n := fun i : Fin k => n i.succ)).symm ⟨v.val - n 0, hv⟩).2⟩ := by
  rw [Equiv.symm_apply_eq]
  apply Fin.ext
  rw [finSigmaFinEquiv_mk_succ, Sigma.eta, Equiv.apply_symm_apply]
  simp only
  omega

/-- An output coordinate of a `cat` of `k + 1` blocks reads the first block below its extent along
`dim`, and above it the `cat` of the other blocks, at the index shifted by that extent. -/
theorem cat_succ_apply {k : Nat} (dim : Fin s.rank) (sizes : Fin (k + 1) → Nat)
    (xs : (i : Fin (k + 1)) → Tensor α (s.set dim (sizes i))) (out : Shape)
    (hout : out = s.set dim (∑ i, sizes i)) (c : out.Coord) :
    cat dim sizes xs out hout c =
      let split := Shape.Coord.setEquiv dim.val (shape := s) (∑ i, sizes i)
        (Shape.Coord.reindexEq hout c)
      if h : split.1.val < sizes 0 then
        xs 0 ((Shape.Coord.setEquiv dim.val (shape := s) (sizes 0)).symm
          (⟨split.1.val, h⟩, split.2))
      else
        cat dim (fun i : Fin k => sizes i.succ) (fun i => xs i.succ)
          (s.set dim (∑ i : Fin k, sizes i.succ)) rfl
          ((Shape.Coord.setEquiv dim.val (shape := s)
            (∑ i : Fin k, sizes i.succ)).symm
            (⟨split.1.val - sizes 0, by
              have := split.1.isLt
              have := Fin.sum_univ_succ sizes
              omega⟩, split.2)) := by
  rw [cat_apply]
  simp only []
  split_ifs with h
  · rw [finSigmaFinEquiv_symm_first _ h]
  · rw [cat_apply]
    rw [Shape.Coord.reindexEq_self, Equiv.apply_symm_apply]
    rw [finSigmaFinEquiv_symm_succ _ (Nat.le_of_not_lt h)]

/-- An output coordinate of a `cat` of one block reads that block. -/
theorem cat_one_apply (dim : Fin s.rank) (sizes : Fin 1 → Nat)
    (xs : (i : Fin 1) → Tensor α (s.set dim (sizes i))) (out : Shape)
    (hout : out = s.set dim (∑ i, sizes i)) (c : out.Coord) :
    cat dim sizes xs out hout c =
      let split := Shape.Coord.setEquiv dim.val (shape := s) (∑ i, sizes i)
        (Shape.Coord.reindexEq hout c)
      xs 0 ((Shape.Coord.setEquiv dim.val (shape := s) (sizes 0)).symm
        (⟨split.1.val, by have := split.1.isLt; simpa using this⟩, split.2)) := by
  rw [cat_apply]
  simp only []
  rw [finSigmaFinEquiv_symm_first]

end TorchLean.Tensor
