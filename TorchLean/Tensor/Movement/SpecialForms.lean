/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Data.Fin.VecNotation
public import TorchLean.Tensor.AdjacentSwap
public import TorchLean.Tensor.Movement
public import TorchLean.Tensor.Scatter

/-!
# Coordinate lemmas for `cat`, `permute` and `gather`

These lemmas read entries of the data-movement operators of `TorchLean.Tensor.Movement`
and compare coordinates one axis at a time through `Shape.Coord.index`.

* `eq_castShape_of_index` proves a tensor equal to a cast of another from their entries at
  coordinates with the same indices.
* `cat_apply_of_index` reads an entry of `cat` from the block that holds its index along the
  concatenated axis.
* `permute_permute` composes two permutations into one.
* `gather_select_apply` reads an index selection, the gather along one axis that batches every
  other axis with itself, at the selected index. `Shape.Coord.val_axisEquiv_fst` and
  `Shape.Coord.index_axisEquiv_symm` give the indices of the coordinates it splits and rebuilds.
* `castShape_gather` moves `gather` between equal output shapes.
* `appendEquiv_reindexEq_cast_fst` reads a cast coordinate back through
  `Shape.Coord.appendEquiv`.
-/

@[expose] public section

open TorchLean.Shape (rank_cons rank_eq_length)

namespace TorchLean

open Shape.Coord (index_appendEquiv_fst)

variable {α : Type} [Storage α]

/-! ## Comparing tensors through coordinates -/

/-- A tensor is a cast of another when each of its entries is the entry of the other at a
coordinate with the same indices. -/
theorem Tensor.eq_castShape_of_index {s t : Shape} {x : Tensor α t} {y : Tensor α s} (h : s = t)
    (hxy : ∀ (c : t.Coord) (c' : s.Coord),
      (∀ k, Shape.Coord.index c' k = Shape.Coord.index c k) → x c = y c') :
    x = castShape y h := by
  subst h
  exact Rep.ext fun c => hxy c c fun _ => rfl

/-! ## Blocks of a concatenation -/

private theorem Tensor.finSigmaFinEquiv_symm_eq {m : Nat} {n : Fin m → Nat} (p : Fin (∑ i, n i))
    (j : Fin m) (o : Fin (n j))
    (h : p.val = ∑ i : Fin j.val, n (Fin.castLE j.isLt.le i) + o.val) :
    finSigmaFinEquiv.symm p = ⟨j, o⟩ := by
  rw [Equiv.symm_apply_eq]
  apply Fin.ext
  rw [finSigmaFinEquiv_apply]
  exact h

/-- An entry of `cat` whose index along `dim` falls `o` past the start of block `j` reads block `j`
at offset `o`. -/
theorem Tensor.cat_apply_of_index {s : Shape} {k : Nat} (dim : Fin s.rank) (sizes : Fin k → Nat)
    (xs : (i : Fin k) → Tensor α (s.set dim (sizes i))) (out : Shape)
    (hout : out = s.set dim (∑ i, sizes i)) (c : out.Coord) (j : Fin k) (o : Fin (sizes j))
    (h : Shape.Coord.index c dim.val = ∑ i : Fin j.val, sizes (Fin.castLE j.isLt.le i) + o.val) :
    cat dim sizes xs out hout c =
      xs j ((Shape.Coord.setEquiv dim.val (shape := s) (sizes j)).symm
        (o, (Shape.Coord.setEquiv dim.val (shape := s) (∑ i, sizes i)
          (Shape.Coord.reindexEq hout c)).2)) := by
  rw [cat_apply]
  dsimp only
  have hpart : finSigmaFinEquiv.symm
      (Shape.Coord.setEquiv dim.val (shape := s) (∑ i, sizes i)
        (Shape.Coord.reindexEq hout c)).1 = ⟨j, o⟩ := by
    apply finSigmaFinEquiv_symm_eq
    rw [Shape.Coord.val_setEquiv_fst, Shape.Coord.index_reindexEq]
    exact h
  rw [hpart]

/-! ## Composing permutations -/

/-- Permuting twice is permuting once by the composite permutation. -/
theorem Tensor.permute_permute {s : Shape} (x : Tensor α s) (σ : Equiv.Perm (Fin s.rank))
    (τ : Equiv.Perm (Fin (s.permute σ).rank)) :
    permute (permute x σ (s.permute σ)) τ ((s.permute σ).permute τ) =
      permute x (((finCongr (s.rank_permute σ)).permCongr τ).trans σ) ((s.permute σ).permute τ)
        (s.permute_permute σ τ).symm := by
  refine Rep.ext fun c => ?_
  rw [permute_apply, permute_apply, permute_apply]
  congr 1
  apply Shape.Coord.ext_index
  intro k hk
  rw [Shape.Coord.index_ofFun, dite_eq_left hk, Shape.Coord.index_ofFun, dite_eq_left hk]
  simp only [Fin.val_cast, Shape.Coord.val_get, Shape.Coord.index_reindexEq,
    Shape.Coord.index_ofFun]
  rw [dite_eq_left (by simp)]
  rfl

/-! ## Index selection -/

/-- The index along `axis` is the first component of `Coord.axisEquiv`. -/
theorem Shape.Coord.val_axisEquiv_fst : (axis : Nat) → {shape : Shape} →
    [_h : Shape.AxisInBounds axis shape] → (c : shape.Coord) →
    (axisEquiv axis (shape := shape) c).1.val = Shape.Coord.index c axis
  | 0, _ :: _, _, _ => rfl
  | axis + 1, _ :: inner, h, c => by
      have : Shape.AxisInBounds axis inner :=
        ⟨by have := h.proof; simp only [Shape.rank_cons] at this; omega⟩
      exact val_axisEquiv_fst axis (shape := inner) c.2

/-- The coordinate that `Coord.axisEquiv` rebuilds from an index `i` along `axis` and a
coordinate `c` of the other axes has index `i` at `axis` and the indices of `c`, shifted past
`axis`, elsewhere. -/
theorem Shape.Coord.index_axisEquiv_symm {s : Shape} (axis : Nat) [Shape.AxisInBounds axis s]
    (i : Fin (s.axisSize axis)) (c : Shape.Coord (s.eraseIdx axis)) (k : Nat) :
    Shape.Coord.index ((axisEquiv axis (shape := s)).symm (i, c)) k =
      if k = axis then i.val
      else if k < axis then Shape.Coord.index c k else Shape.Coord.index c (k - 1) := by
  set c' := (axisEquiv axis (shape := s)).symm (i, c)
  have hc : axisEquiv axis (shape := s) c' = (i, c) :=
    Equiv.apply_symm_apply _ _
  have hi := val_axisEquiv_fst axis c'
  have hr := index_axisEquiv_snd axis c'
  rw [hc] at hi hr
  split_ifs with h₁ h₂
  · rw [h₁, ← hi]
  · rw [hr k, ite_eq_left h₂]
  · rw [hr (k - 1), ite_eq_right (by omega), Nat.sub_add_cancel (by omega)]

/-- An index selection, the gather along `dim` that batches every other axis with itself, reads
its input at the selected index of the axis. -/
theorem Tensor.gather_select_apply {s : Shape} (dim : Fin s.rank) {count : Nat}
    (x : Tensor α s) (indices : Tensor (Fin (s.axisSize dim)) [count])
    (hout : s.set dim count = s.set dim count ++
      s.eraseAxes (s.selectDims dim ++ (s.selectBatch dim count).map Prod.fst))
    (c : Shape.Coord (s.set dim count)) :
    gather x (s.selectDims dim) (s.selectBatch dim count) (selectIndices dim indices)
        (s.set dim count) (.select s dim count) hout c =
      let split := Shape.Coord.setEquiv dim (shape := s) count c
      x ((Shape.Coord.axisEquiv dim (shape := s)).symm
        (indices.getScalar split.1, split.2)) := by
  rw [gather_apply]
  simp only []
  set ct := (Shape.Coord.appendEquiv (s.set dim count)
    (s.eraseAxes (s.selectDims dim ++ (s.selectBatch dim count).map Prod.fst))
    (Shape.Coord.reindexEq hout c)).1 with hct
  have hc : ct = c := by
    refine Shape.Coord.ext_index fun k hk => ?_
    rw [hct, index_appendEquiv_fst _ _ _ k (by rwa [← Shape.rank_eq_length]),
      Shape.Coord.index_reindexEq]
  rw [hc]
  congr 1
  refine Shape.Coord.ext_index fun k hk => ?_
  rw [Shape.GatherValid.source, Shape.Coord.index_ofFun, dite_eq_left hk,
    Shape.Coord.index_axisEquiv_symm]
  by_cases hka : k = dim
  · subst hka
    rw [dite_eq_left (by simp [Shape.selectDims]), ite_eq_left rfl]
    simp only [selectIndices, Fin.val_cast, Rep.get_ofFn]
  · have hmem : (⟨k, hk⟩ : Fin s.rank) ∉ s.selectDims dim := by
      simp [Shape.selectDims, Fin.ext_iff, hka]
    rw [dite_eq_right hmem, ite_eq_right hka]
    split
    · rename_i p hp
      have hp1 : p.1 = ⟨k, hk⟩ := by simpa using List.find?_some hp
      have hp2 : p.2 = p.1.cast Shape.rank_set.symm := by
        obtain ⟨a, -, rfl⟩ := List.mem_map.mp (List.mem_of_find?_eq_some hp)
        rfl
      rw [Fin.val_cast, Shape.Coord.val_get, hp2, hp1, Fin.val_cast]
      simp only [Shape.Coord.index_setEquiv_snd]
      split_ifs <;> first | rfl | omega | (congr 1; omega)
    · rename_i hnone
      exfalso
      have hin : ((⟨k, hk⟩ : Fin s.rank), (⟨k, hk⟩ : Fin s.rank).cast Shape.rank_set.symm) ∈
          s.selectBatch dim count :=
        List.mem_map.mpr ⟨⟨k, hk⟩, Shape.mem_otherAxes.mpr hmem, rfl⟩
      simpa using List.find?_eq_none.mp hnone _ hin

/-- The output shape argument of `gather` only names its shape: two equal output shapes give the
same entries, through the cast between them. -/
theorem Tensor.castShape_gather {s t : Shape} (x : Tensor α s) (dims : List (Fin s.rank))
    (batchDims : List (Fin s.rank × Fin t.rank))
    (indices : Tensor (Einops.AxisTuple s.extent dims) t) {out out' : Shape}
    (h : Shape.GatherValid s t dims batchDims)
    (hout : out = t ++ s.eraseAxes (dims ++ batchDims.map Prod.fst))
    (hout' : out' = t ++ s.eraseAxes (dims ++ batchDims.map Prod.fst)) :
    castShape (gather x dims batchDims indices out h hout) (hout.trans hout'.symm) =
      gather x dims batchDims indices out' h hout' := by
  subst hout
  subst hout'
  rfl


/-- A coordinate `u` of `t`, cast to `t ++ e` along `h : t ++ e = t`, reindexed to `t ++ e'` and
split by `Shape.Coord.appendEquiv`, gives back `u` as its first part. -/
theorem Shape.Coord.appendEquiv_reindexEq_cast_fst {t e e' : Shape} (h : t ++ e = t) (u : t.Coord)
    (h' : t ++ e = t ++ e') :
    (Coord.appendEquiv t e'
      (Shape.Coord.reindexEq h' (_root_.cast (congrArg Shape.Coord h.symm) u))).1 = u := by
  refine Shape.Coord.ext_index fun k hk => ?_
  rw [index_appendEquiv_fst _ _ _ k (by rwa [← Shape.rank_eq_length]),
    Shape.Coord.index_reindexEq, Shape.Coord.index_cast h.symm]

end TorchLean
