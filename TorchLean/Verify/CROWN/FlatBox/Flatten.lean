/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.CROWN.FlatBox.Enclosure
public import TorchLean.Tensor.Layers.Linear
public import TorchLean.Tensor.Sequence
public import TorchLean.Tensor.SomeTensor
import TorchLean.Tensor.Algebra
import TorchLean.Tensor.ShapeChange.Flatten
import TorchLean.Tensor.Rep.Reindex

/-!
# Flattening shaped values

`flatOfSome` flattens a shaped real value into the flat tensor the bound passes work on. The
lemmas say how flattening commutes with pointwise maps, dimension casts and maps over a leading
axis, and that a linear layer without input columns is its bias.
-/

@[expose] public section

open TorchLean.Shape (size_append)

namespace TorchLean.CROWN.Graph

open _root_.TorchLean.Tensor

noncomputable section

/-- Flatten a real tensor of any shape into a `FlatTensor ℝ` whose length is the size of its
shape. -/
def flatOfSome (t : SomeTensor ℝ) : FlatTensor ℝ :=
  { n := t.shape.size, v := Tensor.reshape t.tensor [_] (Shape.size_vector _).symm }

/-- Flattening commutes with pointwise binary operations. -/
theorem _root_.TorchLean.Tensor.reshape_vector_zipWith {s : Shape} (f : ℝ → ℝ → ℝ)
    (a b : Tensor ℝ s) :
    Tensor.reshape (Tensor.zipWith f a b) [_] (Shape.size_vector _).symm =
      Tensor.zipWith f (Tensor.reshape a [_] (Shape.size_vector _).symm)
        (Tensor.reshape b [_] (Shape.size_vector _).symm) := by
  unfold Tensor.reshape
  exact (Tensor.Rep.zipWith_reshape _ _ _ _).symm

/-- Flattening commutes with pointwise unary operations. -/
theorem _root_.TorchLean.Tensor.reshape_vector_map {s : Shape} (f : ℝ → ℝ) (a : Tensor ℝ s) :
    Tensor.reshape (Tensor.map f a) [_] (Shape.size_vector _).symm =
      Tensor.map f (Tensor.reshape a [_] (Shape.size_vector _).symm) := by
  unfold Tensor.reshape
  exact (Tensor.Rep.map_reshape _ _ _).symm

/-- Two flat values are equal when their lengths agree and the vectors agree up to the cast. -/
theorem flatTensor_eq_of_cast {n n' : Nat} (h : n = n') (v : Tensor ℝ [n]) (v' : Tensor ℝ [n'])
    (hv : castDimScalar (α := ℝ) h v = v') :
    ({ n := n, v := v } : FlatTensor ℝ) = { n := n', v := v' } := by
  cases h
  rw [castDimScalar_self] at hv
  rw [hv]

/-- Two dimension casts compose to one. -/
private theorem castDim_castDim {n n' n'' : Nat} (h : n = n') (h' : n' = n'')
    (t : Tensor ℝ [n]) :
    castDimScalar (α := ℝ) h' (castDimScalar (α := ℝ) h t) =
      castDimScalar (α := ℝ) (h.trans h') t := by
  cases h
  cases h'
  rfl

/-- Flattening a stack with one slice is that slice flattened. -/
private theorem flatten_dim_one {s : Shape} (values : Fin 1 → Tensor ℝ s)
    (h : (Shape.dim 1 s).size = s.size) :
    castDimScalar (α := ℝ) h
        (Tensor.reshape (Tensor.dim values) [_] (Shape.size_vector _).symm) =
      Tensor.reshape (values 0) [_] (Shape.size_vector _).symm := by
  ext i
  rw [getScalar_castDimScalar]
  have hIndex : (0 : Fin 1).val * s.size + i.val < (Shape.dim 1 s).size := by
    simp
  have hi : Fin.cast h.symm i = ⟨(0 : Fin 1).val * s.size + i.val, hIndex⟩ := by
    ext
    simp only [Fin.val_cast]
    simp
  rw [hi, Tensor.getScalar_reshape_dim values 0 i hIndex]

/-- If `leading` has size one, flattening `Tensor.mapLeading leading f x` gives `f` applied to the
flattened `x`. -/
theorem flatten_mapLeading_of_size_one {n m : Nat} (f : Tensor ℝ [n] → Tensor ℝ [m]) :
    ∀ (leading : Shape), leading.size = 1 → ∀ (x : Tensor ℝ (leading ++ [n]))
      (hm : (leading ++ [m]).size = m) (hn : (leading ++ [n]).size = n),
      castDimScalar (α := ℝ) hm
          (Tensor.reshape (Tensor.mapLeading leading f x) [_] (Shape.size_vector _).symm) =
        f (castDimScalar (α := ℝ) hn (Tensor.reshape x [_] (Shape.size_vector _).symm)) := by
  intro leading
  induction leading with
  | scalar =>
      intro _ x hm hn
      have hx :
          castDimScalar (α := ℝ) hn (Tensor.reshape x [_] (Shape.size_vector _).symm) = x := by
        ext i
        rw [getScalar_castDimScalar]
        erw [getScalar_reshape_vector (x := x) _ (by simp)]
      rw [hx]
      ext i
      rw [getScalar_castDimScalar]
      change getScalar (Tensor.reshape (f x) [_] (Shape.size_vector _).symm) _ = _
      rw [getScalar_reshape_vector (x := f x) _ (by simp)]
      rfl
  | dim d rest ih =>
      intro hsize x hm hn
      have hd : d = 1 := Nat.eq_one_of_mul_eq_one_right hsize
      have hrest : rest.size = 1 := Nat.eq_one_of_mul_eq_one_left hsize
      subst hd
      have hstack : ∀ {s : Shape} (y : Tensor ℝ (Shape.dim 1 s)),
          Tensor.reshape y [_] (Shape.size_vector _).symm =
            castDimScalar (α := ℝ) (by simp)
              (Tensor.reshape (Tensor.unstack y 0) [_] (Shape.size_vector _).symm) := by
        intro s y
        conv_lhs => rw [← Tensor.dim_unstack y]
        rw [← flatten_dim_one (Tensor.unstack y) (by simp), castDim_castDim]
        exact (castDimScalar_self (α := ℝ) _ _).symm
      change castDimScalar (α := ℝ) hm (Tensor.reshape
          (Tensor.dim fun i => Tensor.mapLeading rest f (Tensor.unstack x i)) [_]
          (Shape.size_vector _).symm) = _
      erw [hstack (Tensor.dim fun i => Tensor.mapLeading rest f (Tensor.unstack x i)), hstack x]
      rw [castDim_castDim, castDim_castDim, Tensor.unstack_dim]
      exact ih hrest (Tensor.unstack x 0) _ _

/-- Over any leading shape, flattening `Tensor.mapLeading leading (fun _ => c) x` gives
`Graph.linearBiasRows c leading.size`, the row `c` repeated at every leading index. -/
theorem flatten_mapLeading_const {n m : Nat} (c : Tensor ℝ [m]) :
    ∀ (leading : Shape) (x : Tensor ℝ (leading ++ [n]))
      (h : (leading ++ [m]).size = leading.size * m),
      castDimScalar (α := ℝ) h
          (Tensor.reshape (Tensor.mapLeading leading (fun _ => c) x) [_]
            (Shape.size_vector _).symm) =
        Graph.linearBiasRows c leading.size := by
  intro leading
  induction leading with
  | scalar =>
      intro x h
      ext k
      obtain ⟨kv, hkv⟩ := k
      rw [getScalar_castDimScalar]
      change getScalar (Tensor.reshape c [_] (Shape.size_vector _).symm) _ = _
      rw [getScalar_reshape_vector (x := c) _ (by simp)]
      simp only [Graph.linearBiasRows, Tensor.getScalar_ofFn]
      congr 1
      ext
      have hkm : kv < m := by simpa using hkv
      simp only [Fin.val_cast]
      simp [Fin.modNat, Nat.mod_eq_of_lt hkm]
  | dim d rest ih =>
      intro x h
      ext k
      obtain ⟨kv, hkv⟩ := k
      rw [getScalar_castDimScalar]
      have hS : (rest ++ [m]).size = rest.size * m := by
        rw [Shape.size_append]; simp
      have hk' : kv < d * (rest ++ [m]).size := by
        rw [hS, ← Nat.mul_assoc]
        simpa using hkv
      have hpos : 0 < (rest ++ [m]).size := by
        rcases Nat.eq_zero_or_pos (rest ++ [m]).size with h0 | h0
        · rw [h0, Nat.mul_zero] at hk'
          exact absurd hk' (Nat.not_lt_zero _)
        · exact h0
      have hq : kv / (rest ++ [m]).size < d := by
        rw [Nat.div_lt_iff_lt_mul hpos]
        exact hk'
      have hr : kv % (rest ++ [m]).size < (rest ++ [m]).size := Nat.mod_lt _ hpos
      have hIndex : (⟨kv / (rest ++ [m]).size, hq⟩ : Fin d).val * (rest ++ [m]).size +
          (⟨kv % (rest ++ [m]).size, hr⟩ : Fin (rest ++ [m]).size).val <
          (Shape.dim d (rest ++ [m])).size := by
        change kv / _ * _ + kv % _ < d * (rest ++ [m]).size
        rw [Nat.mul_comm (kv / _) _, Nat.div_add_mod]
        exact hk'
      have hk : Fin.cast h.symm ⟨kv, hkv⟩ = ⟨(⟨kv / (rest ++ [m]).size, hq⟩ : Fin d).val *
          (rest ++ [m]).size +
          (⟨kv % (rest ++ [m]).size, hr⟩ : Fin (rest ++ [m]).size).val, hIndex⟩ := by
        ext
        change kv = kv / _ * _ + kv % _
        rw [Nat.mul_comm (kv / _) _, Nat.div_add_mod]
      change getScalar (Tensor.reshape
          (Tensor.dim fun i => Tensor.mapLeading rest (fun _ => c) (Tensor.unstack x i)) [_]
          (Shape.size_vector _).symm) _ = _
      rw [hk, Tensor.getScalar_reshape_dim _ _ _ hIndex]
      have hrow := congrArg (fun t => getScalar t (Fin.cast hS ⟨_, hr⟩))
        (ih (Tensor.unstack x ⟨_, hq⟩) hS)
      simp only [getScalar_castDimScalar, Fin.cast_cast, Fin.cast_eq_self] at hrow
      rw [hrow]
      simp only [Graph.linearBiasRows, Tensor.getScalar_ofFn]
      congr 1
      ext
      simp only [Fin.modNat, Fin.val_cast]
      exact Nat.mod_mod_of_dvd _ (by rw [hS]; exact Dvd.intro_left _ rfl)

/-- Without input columns the affine map is its bias. -/
theorem _root_.TorchLean.Linear.forward_of_inDim_zero {m n : Nat} (hn : n = 0) (W : Tensor ℝ [m, n])
    (b : Tensor ℝ [m]) (x : Tensor ℝ [n]) :
    Linear.forward (α := ℝ) { weights := W, bias := b } x = b := by
  subst hn
  ext i
  unfold Linear.forward
  rw [congrFun (Tensor.getScalar_add _ _) _,
    Tensor.getScalar_contract_matVec_eq_sum]
  simp

end

end TorchLean.CROWN.Graph
