/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Contract
public import TorchLean.Tensor.ShapeChange.Flatten
public import TorchLean.Tensor.Einops.Equivalence.Index
public import TorchLean.Tensor.Rep.Traversal.Sequence

/-!
# The flat layout of a matrix product

`IR.MatmulDims` is the geometry of a matrix product with broadcast batch axes and vector
operands, and `MatmulDims.matmulFlat` the row-major contraction over it. Over operands whose batch
prefixes broadcast to the product's leading shape the contraction is the registered batched product
`Tensor.batchMatmul` (`MatmulDims.matmulFlat_read`, `MatmulDims.matmulFlat_eq_batchMatmul`), through
`MatmulDims.batchIndex_linearize`.
-/

@[expose] public section

open TorchLean.Shape (rank rank_eq_length size)

namespace TorchLean

open Tensor Shape

namespace IR


/--
Geometry of a matrix product after broadcasting batch axes and promoting vectors.

A left vector is viewed as a one-row matrix and a right vector as a one-column matrix. Their
inserted axes are removed from the output, and the original operand shapes are kept for validation
and flat indexing.
-/
structure MatmulDims where
  /-- Broadcast batch shape of the output. -/
  leading : Shape
  /-- Row count of the left operand. -/
  rows : Nat
  /-- Contracted extent shared by both operands. -/
  inner : Nat
  /-- Column count of the right operand. -/
  cols : Nat
  /-- Original left operand shape. -/
  leftShape : Shape
  /-- Original right operand shape. -/
  rightShape : Shape
  /-- Batch axes of the left operand, before broadcasting. -/
  leftLeading : Shape := leading
  /-- Batch axes of the right operand, before broadcasting. -/
  rightLeading : Shape := leading
  /-- Whether the left operand is a vector, viewed as a one-row matrix. -/
  leftVector : Bool := false
  /-- Whether the right operand is a vector, viewed as a one-column matrix. -/
  rightVector : Bool := false

namespace MatmulDims

/-- Shape of the `matmul` result. -/
@[simp] def outShape (dims : MatmulDims) : Shape :=
  dims.leading ++
    ((if dims.leftVector then [] else [dims.rows]) ++
      (if dims.rightVector then [] else [dims.cols]))

/-- One axis of `batchIndex`'s fold over the reversed extents `srcRev` and `tgtRev`: the state is
the remaining target index, the source index so far and the source stride. -/
def batchStep (srcRev tgtRev : List Nat) (state : Nat × Nat × Nat) (axis : Nat) :
    Nat × Nat × Nat :=
  (state.1 / tgtRev.getD axis 1,
    state.2.1 + (if srcRev.getD axis 1 = 1 then 0 else state.1 % tgtRev.getD axis 1) * state.2.2,
    state.2.2 * srcRev.getD axis 1)

/-- Project a flat output batch coordinate to one operand's unbroadcast batch coordinate. -/
def batchIndex (source target : Shape) (index : Nat) : Nat :=
  ((List.range source.length).foldl (batchStep source.reverse target.reverse) (index, 0, 1)).2.1

/-- Left scalar used by one output coordinate and one contraction coordinate. -/
def leftIndex (dims : MatmulDims) (output inner : Nat) : Nat :=
  let batch := output / (dims.rows * dims.cols)
  let row := output % (dims.rows * dims.cols) / dims.cols
  batchIndex dims.leftLeading dims.leading batch * (dims.rows * dims.inner) +
    row * dims.inner + inner

/-- Right scalar used by one output coordinate and one contraction coordinate. -/
def rightIndex (dims : MatmulDims) (output inner : Nat) : Nat :=
  let batch := output / (dims.rows * dims.cols)
  let col := output % dims.cols
  batchIndex dims.rightLeading dims.leading batch * (dims.inner * dims.cols) +
    inner * dims.cols + col

end MatmulDims

end IR

namespace IR.MatmulDims

open Shape (Coord)

private theorem prod_take_succ (l : List Nat) (m : Nat) :
    (l.take (m + 1)).prod = (l.take m).prod * l.getD m 1 := by
  induction l generalizing m with
  | nil => simp
  | cons a l ih =>
      cases m with
      | zero => simp
      | succ m => simp [List.take_succ_cons, ih, Nat.mul_assoc]

/-- The fold over a source with one more outer axis is the fold over the rest, then one step. -/
private theorem fold_cons (tgtRev : List Nat) (state : Nat × Nat × Nat) (s₀ : Nat)
    (S : List Nat) :
    (List.range (s₀ :: S).length).foldl (batchStep (s₀ :: S).reverse tgtRev) state =
      batchStep (s₀ :: S).reverse tgtRev
        ((List.range S.length).foldl (batchStep S.reverse tgtRev) state) S.length := by
  rw [List.length_cons, List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil]
  congr 1
  apply List.foldl_ext
  intro st a ha
  rw [List.mem_range] at ha
  simp only [batchStep, List.reverse_cons]
  rw [List.getD_append _ _ _ _ (by simpa using ha)]

/-- After the fold over a source's axes, the remaining index is the input index divided by the
target extents read so far, and the stride is the source's size. -/
private theorem fold_state (tgtRev : List Nat) (index : Nat) : ∀ S : List Nat,
    ((List.range S.length).foldl (batchStep S.reverse tgtRev) (index, 0, 1)).1 =
        index / (tgtRev.take S.length).prod ∧
      ((List.range S.length).foldl (batchStep S.reverse tgtRev) (index, 0, 1)).2.2 = S.prod
  | [] => by simp
  | s₀ :: S => by
      obtain ⟨h1, h2⟩ := fold_state tgtRev index S
      rw [fold_cons]
      have hs : (S.reverse ++ [s₀]).getD S.length 1 = s₀ := by
        rw [List.getD_append_right _ _ _ _ (by simp)]
        simp
      simp only [batchStep, List.reverse_cons, hs, h1, h2, List.length_cons, prod_take_succ,
        Nat.div_div_eq_div_mul, List.prod_cons]
      exact ⟨trivial, Nat.mul_comm _ _⟩

/-- `batchIndex` over a source with an outer axis `s₀`: the index over the remaining axes plus
`s₀`'s digit of the target index, zero where `s₀` is one, times the remaining source size. -/
private theorem batchIndex_cons (s₀ : Nat) (S T : Shape) (index : Nat) :
    batchIndex (s₀ :: S) T index = batchIndex S T index +
      (if s₀ = 1 then 0 else
        index / (T.reverse.take S.length).prod % T.reverse.getD S.length 1) * S.prod := by
  unfold batchIndex
  show ((List.range (s₀ :: S).length).foldl (batchStep (s₀ :: S).reverse T.reverse)
    (index, 0, 1)).2.1 = _
  rw [fold_cons]
  obtain ⟨h1, h2⟩ := fold_state T.reverse index S
  have hs : (S.reverse ++ [s₀]).getD S.length 1 = s₀ := by
    rw [List.getD_append_right _ _ _ _ (by simp)]
    simp
  simp only [batchStep, List.reverse_cons, hs, h1, h2]

/-- The row-major index of a coordinate, divided by the extents of its `m` innermost axes, has the
index along the axis before them as its digit. -/
private theorem linearize_digit : ∀ (T : Shape) (c : T.Coord) (m : Nat), m < T.length →
    (Coord.linearize c).val / (T.reverse.take m).prod % T.reverse.getD m 1 =
      Shape.Coord.index c (T.length - 1 - m)
  | [], _, m, hm => absurd hm (Nat.not_lt_zero m)
  | t₀ :: T', (c₀, c'), m, hm => by
      have hlin := Coord.linearize_cons_val (s := T') c₀ c'
      have hx := (Coord.linearize c').isLt
      have hd := linearize_digit T' c' m
      simp only [Shape.size_eq_prod] at hlin hx
      rw [hlin]
      generalize (Coord.linearize c').val = x at hx hd ⊢
      have hQ : 0 < T'.prod := by omega
      rcases Nat.lt_or_ge m T'.length with hlt | hge
      · have htake : ((t₀ :: T').reverse.take m) = T'.reverse.take m := by
          rw [List.reverse_cons, List.take_append_of_le_length (by simp; omega)]
        have hget : (t₀ :: T').reverse.getD m 1 = T'.reverse.getD m 1 := by
          rw [List.reverse_cons, List.getD_append _ _ _ _ (by simp; omega)]
        rw [htake, hget]
        have hdvd : (T'.reverse.take m).prod * T'.reverse.getD m 1 ∣ T'.prod := by
          rw [← prod_take_succ, ← List.prod_reverse T']
          exact ⟨_, (List.prod_take_mul_prod_drop _ (m + 1)).symm⟩
        obtain ⟨R, hR⟩ := hdvd
        have hP : 0 < (T'.reverse.take m).prod := by
          rcases Nat.eq_zero_or_pos (T'.reverse.take m).prod with h0 | h0
          · rw [h0, Nat.zero_mul, Nat.zero_mul] at hR; omega
          · exact h0
        rw [hR, show (T'.reverse.take m).prod * T'.reverse.getD m 1 * R * c₀.val =
            (T'.reverse.getD m 1 * (R * c₀.val)) * (T'.reverse.take m).prod by ring,
          Nat.add_mul_div_right _ _ hP, Nat.add_mul_mod_self_left, hd hlt]
        have hidx : (t₀ :: T').length - 1 - m = (T'.length - 1 - m) + 1 := by
          simp only [List.length_cons]; omega
        rw [hidx, Shape.Coord.index_cons_succ]
      · have hm' : m = T'.length := by simp at hm; omega
        subst hm'
        have htake : ((t₀ :: T').reverse.take T'.length) = T'.reverse := by
          rw [List.reverse_cons, List.take_append_of_le_length (by simp)]
          simp
        have hget : (t₀ :: T').reverse.getD T'.length 1 = t₀ := by
          rw [List.reverse_cons, List.getD_append_right _ _ _ _ (by simp)]
          simp
        rw [htake, hget, List.prod_reverse, Nat.add_mul_div_left _ _ hQ, Nat.div_eq_of_lt hx,
          Nat.zero_add, Nat.mod_eq_of_lt c₀.isLt]
        have hidx : (t₀ :: T').length - 1 - T'.length = 0 := by simp
        rw [hidx, Shape.Coord.index_zero]

/-- A shape that broadcasts to `T` broadcasts there without its outer axis. -/
private theorem canBroadcastTo_tail {s₀ : Nat} {S : Shape} : ∀ {T : Shape},
    Shape.CanBroadcastTo (s₀ :: S) T → Shape.CanBroadcastTo S T
  | [], h => absurd h Shape.not_canBroadcastTo_dim_scalar
  | t₀ :: T, h => by
      by_cases hr : Shape.rank S = Shape.rank T
      · exact ((Shape.canBroadcastTo_dim_dim_of_rank_eq hr).mp h).2.expand_dims
      · exact (canBroadcastTo_tail ((Shape.canBroadcastTo_dim_dim_of_rank_ne hr).mp h)).expand_dims

/-- `batchIndex` reads a target index as the source coordinate of right-aligned broadcasting: each
source axis takes the index of the target axis it is placed at, and index zero where its extent
is one. -/
theorem batchIndex_linearize : ∀ (S : Shape) {T : Shape}, Shape.CanBroadcastTo S T →
    ∀ (c : T.Coord) (p : S.Coord),
    (∀ a < S.length, Shape.Coord.index p a =
      if S.getD a 1 = 1 then 0 else Shape.Coord.index c (a + (T.length - S.length))) →
    batchIndex S T (Coord.linearize c).val = (Coord.linearize p).val
  | [], _, _, _, p, _ => by
      cases p
      rfl
  | s₀ :: S', T, h, c, (p₀, p'), hp => by
      have hle := h.rank_le
      simp only [Shape.rank_eq_length, List.length_cons] at hle
      rw [batchIndex_cons, batchIndex_linearize S' (canBroadcastTo_tail h) c p' (fun a ha => by
        have := hp (a + 1) (by simp; omega)
        rw [Shape.Coord.index_cons_succ] at this
        rw [this]
        simp only [List.getD_cons_succ, List.length_cons]
        congr 2
        omega)]
      have hlin := Coord.linearize_cons_val (s := S') p₀ p'
      simp only [Shape.size_eq_prod] at hlin
      rw [hlin, linearize_digit T c S'.length (by omega)]
      have h0 := hp 0 (by simp)
      simp only [Shape.Coord.index_zero, List.getD_cons_zero, List.length_cons,
        Nat.zero_add] at h0
      rw [Nat.mul_comm, h0]
      have hidx : T.length - 1 - S'.length = T.length - (S'.length + 1) := by omega
      rw [hidx]

end IR.MatmulDims

/-- Row-major matrix contraction using the shared broadcast and vector-promotion layout. -/
def IR.MatmulDims.matmulFlat {α : Type} [Storage α] [Context α]
    (dims : MatmulDims) {leftDim rightDim : Nat}
    (left : Tensor α [leftDim]) (right : Tensor α [rightDim]) :
    Tensor α [dims.outShape.size] :=
  Tensor.ofFn fun output =>
    (List.range dims.inner).foldl (fun acc inner =>
      acc + getAtOrZero left [dims.leftIndex output.val inner] *
        getAtOrZero right [dims.rightIndex output.val inner]) 0

variable {α : Type} [Storage α] [Context α]

section
open Shape (Coord)

/-- The row-major index of a batch coordinate followed by a row and a column. -/
theorem Shape.Coord.linearize_matrix (L : Shape) {u v : Nat} (b : L.Coord) (i : Fin u) (t : Fin v) :
    (Coord.linearize ((Coord.appendEquiv L [u, v]).symm (b, i, t, PUnit.unit))).val =
      (Coord.linearize b).val * (u * v) + i.val * v + t.val := by
  rw [Coord.linearize_appendEquiv_symm_val,
    Coord.linearize_cons_val (s := [v]) i (t, PUnit.unit),
    Coord.linearize_cons_val (s := []) t PUnit.unit]
  simp only [Shape.size_eq_prod, List.prod_cons, List.prod_nil]
  change 0 + 1 * t.val + v * 1 * i.val + u * (v * 1) * (Coord.linearize b).val = _
  ring

/-- The batch, row and column of a row-major index over `batch ++ [u, v]`. -/
theorem IR.MatmulDims.matrix_indices (batch u v row col : Nat) (hrow : row < u) (hcol : col < v) :
    let output := batch * (u * v) + row * v + col
    output / (u * v) = batch ∧ output % (u * v) / v = row ∧ output % v = col := by
  have hsmall : row * v + col < u * v := by
    have h := Nat.mul_le_mul_right v (Nat.succ_le_of_lt hrow)
    nlinarith
  have hout : batch * (u * v) + row * v + col = (row * v + col) + (u * v) * batch := by ring
  dsimp only
  refine ⟨?_, ?_, ?_⟩
  · rw [hout, Nat.add_mul_div_left _ _ (Nat.zero_lt_of_lt hsmall), Nat.div_eq_of_lt hsmall,
      Nat.zero_add]
  · rw [hout, Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hsmall,
      show row * v + col = col + v * row by ring,
      Nat.add_mul_div_left _ _ (Nat.zero_lt_of_lt hcol), Nat.div_eq_of_lt hcol, Nat.zero_add]
  · rw [show batch * (u * v) + row * v + col = col + v * (row + u * batch) by ring,
      Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hcol]

/-- A flat read of a batched operand at its broadcast batch index, a row and a column is the entry
of the operand broadcast to the batch shape `L` at that batch coordinate, row and column. -/
theorem Tensor.flat_read_broadcast {L₁ L : Shape} (h : Shape.CanBroadcastTo L₁ L) {u v : Nat}
    (x : Tensor α (L₁ ++ [u, v])) (b : L.Coord) (i : Fin u) (t : Fin v) :
    getAtOrZero (x.reshape [_] (Shape.size_vector _).symm)
        [IR.MatmulDims.batchIndex L₁ L (Coord.linearize b).val * (u * v) +
          i.val * v + t.val] =
      Tensor.broadcast x _ (CanBroadcastTo.concat_right [u, v] h).dims
        (CanBroadcastTo.concat_right [u, v] h).broadcastDims
        ((Coord.appendEquiv L [u, v]).symm (b, i, t, PUnit.unit)) := by
  have hle := h.rank_le
  simp only [Shape.rank_eq_length] at hle
  let p : L₁.Coord := Shape.Coord.ofFun fun a =>
    if hi : L₁.extent a = 1 then ⟨0, by omega⟩
    else Fin.cast ((h.broadcastDims.extent_eq a).resolve_left hi).symm
      (Shape.Coord.get b (h.dims a))
  have hp : ∀ a < L₁.length, Shape.Coord.index p a =
      if L₁.getD a 1 = 1 then 0 else Shape.Coord.index b (a + (L.length - L₁.length)) := by
    intro a ha
    have ha' : a < L₁.rank := by rw [Shape.rank_eq_length]; exact ha
    have hext : L₁.extent ⟨a, ha'⟩ = L₁.getD a 1 := by
      simp [Shape.extent, List.getD_eq_getElem?_getD, ha]
    rw [Shape.Coord.index_ofFun, dite_eq_left ha', ← hext]
    split_ifs
    · rfl
    · rw [Fin.val_cast, Shape.Coord.val_get]
      simp [Shape.CanBroadcastTo.dims, Shape.rank_eq_length]
  rw [IR.MatmulDims.batchIndex_linearize L₁ h b p hp, ← Coord.linearize_matrix L₁ p i t]
  refine (getAtOrZero_reshape_toFin x _).trans ?_
  rw [Tensor.broadcast_apply]
  congr 1
  apply Shape.Coord.ext_index
  intro k hk
  have hkl : k < L₁.length + 2 := by
    simpa [Shape.rank_eq_length] using hk
  rw [Shape.Coord.index_ofFun, dite_eq_left hk, Coord.index_appendEquiv_symm]
  have hext : (L₁ ++ [u, v]).extent ⟨k, hk⟩ = (L₁ ++ [u, v]).getD k 1 := by
    simp [Shape.extent, List.getD_eq_getElem?_getD, hkl]
  have hdims : (((CanBroadcastTo.concat_right [u, v] h).dims ⟨k, hk⟩) :
      Nat) = k + (L.length - L₁.length) := by
    simp [Shape.CanBroadcastTo.dims, Shape.rank_eq_length]
  by_cases hk1 : k < L₁.length
  · rw [ite_eq_left hk1, hp k hk1]
    rw [List.getD_append _ _ _ _ hk1] at hext
    split_ifs with he h2 h2
    · rfl
    · simp_all
    · simp_all
    · rw [Fin.val_cast, Shape.Coord.val_get]
      rw [Coord.index_appendEquiv_symm,
        hdims, ite_eq_left (by omega)]
  · rw [ite_eq_right hk1]
    rw [List.getD_append_right _ _ _ _ (by omega)] at hext
    split_ifs with he
    · rw [hext] at he
      rcases Nat.lt_or_ge k (L₁.length + 1) with h2 | h2
      · have hj : k - L₁.length = 0 := by omega
        rw [hj] at he ⊢
        simp only [List.getD_cons_zero] at he
        have hi : Shape.Coord.index (s := [u, v]) (i, t, PUnit.unit) 0 = i.val := rfl
        rw [hi]
        have := i.isLt
        omega
      · have hj : k - L₁.length = 1 := by omega
        rw [hj] at he ⊢
        simp only [List.getD_cons_succ, List.getD_cons_zero] at he
        have ht : Shape.Coord.index (s := [u, v]) (i, t, PUnit.unit) 1 = t.val := rfl
        rw [ht]
        have := t.isLt
        omega
    · rw [Fin.val_cast, Shape.Coord.val_get]
      rw [Coord.index_appendEquiv_symm,
        hdims, ite_eq_right (by omega)]
      congr 1
      omega

/-- Over operands whose batch prefixes broadcast to `dims.leading`, and a result with as many
entries as the batched product, the flat contraction `matmulFlat` has, at the row-major index of
each output coordinate, the entry of the batched product of the broadcast operands. -/
theorem IR.MatmulDims.matmulFlat_read (dims : MatmulDims)
    (hA : Shape.CanBroadcastTo dims.leftLeading dims.leading)
    (hB : Shape.CanBroadcastTo dims.rightLeading dims.leading)
    (A : Tensor α (dims.leftLeading ++ [dims.rows, dims.inner]))
    (B : Tensor α (dims.rightLeading ++ [dims.inner, dims.cols]))
    (hsz : dims.outShape.size = (dims.leading ++ [dims.rows, dims.cols]).size)
    (c : (dims.leading ++ [dims.rows, dims.cols]).Coord) :
    getAtOrZero (matmulFlat dims (A.reshape [_] (Shape.size_vector _).symm)
        (B.reshape [_] (Shape.size_vector _).symm)) [(Coord.linearize c).val] =
      Tensor.batchMatmul hA hB A B c := by
  obtain ⟨⟨b, i, j, ⟨⟩⟩, hc⟩ :=
    (Coord.appendEquiv dims.leading [dims.rows, dims.cols]).symm.surjective
      c
  have hc' : c = ((Coord.appendEquiv dims.leading [dims.rows, dims.cols]).symm (b, i, j,
      PUnit.unit)) := by
    rw [hc]
  subst hc'
  have hlt : (Coord.linearize ((Coord.appendEquiv dims.leading [dims.rows, dims.cols]).symm
        (b, i, j, PUnit.unit))).val < dims.outShape.size := by
    rw [hsz]; exact (Coord.linearize _).isLt
  rw [show (Coord.linearize ((Coord.appendEquiv dims.leading [dims.rows, dims.cols]).symm
      (b, i, j, PUnit.unit))).val = (⟨_, hlt⟩ : Fin dims.outShape.size).val from rfl,
    getAtOrZero_eq_getScalar]
  simp only [matmulFlat, Tensor.getScalar_ofFn]
  rw [Coord.linearize_matrix]
  obtain ⟨hbatch, hrow, hcol⟩ := matrix_indices (Coord.linearize b).val dims.rows dims.cols i.val
    j.val i.isLt j.isLt
  simp only [MatmulDims.leftIndex, MatmulDims.rightIndex, hbatch, hrow,
    hcol]
  rw [← List.map_coe_finRange_eq_range, List.foldl_map]
  have hA' := fun t : Fin dims.inner => flat_read_broadcast hA A b i t
  have hB' := fun t : Fin dims.inner => flat_read_broadcast hB B b t j
  simp only [hA', hB']
  simp only [Tensor.batchMatmul]
  rw [Tensor.matmulCommonBatch_apply]
  simp only [Equiv.apply_symm_apply]

/-- A flat read of a transported tensor is the flat read of the tensor. -/
theorem Tensor.flat_read_cast {s t : Shape} (h : s = t) (x : Tensor α s) (index : Nat) :
    getAtOrZero ((h ▸ x : Tensor α t).reshape [_] (Shape.size_vector _).symm) [index] =
      getAtOrZero (x.reshape [_] (Shape.size_vector _).symm) [index] := by
  subst h
  rfl

/-- A flat vector over a shape equal to `L` whose entry at the row-major index of each coordinate
of `L` is the entry of `y` there, reshaped to that shape, is `y`. -/
theorem Tensor.reshape_eq_of_flat_read {out L : Shape} (ho : out = L) (v : Tensor α [out.size])
    (y : Tensor α L) (h : ∀ c : Coord L, getAtOrZero v [(Coord.linearize c).val] = y c) :
    v.reshape out (Shape.size_vector _) = Tensor.castShape y ho.symm := by
  have key : v = Tensor.reshape (Tensor.castShape y ho.symm) [_] (Shape.size_vector _).symm := by
    apply Tensor.ext_vector
    intro o
    rw [← getAtOrZero_eq_getScalar, ← getAtOrZero_eq_getScalar]
    have hsz : out.size = Shape.size L := by rw [ho]
    have ho' : o.val < Shape.size L := by
      have := o.isLt
      omega
    obtain ⟨c, hc⟩ : ∃ c : Coord L, (Coord.linearize c).val = o.val :=
      ⟨Coord.unlinearize ⟨o.val, ho'⟩, by simp⟩
    rw [← hc, h c, ← Coord.linearize_cast_val ho c]
    refine Eq.trans ?_ (getAtOrZero_reshape_toFin _ _).symm
    rw [Tensor.castShape_apply, cast_cast, cast_eq]
  rw [key, Tensor.reshape_reshape_self]

/-- A flat read of a reshaped tensor is the flat read of the tensor. -/
theorem Tensor.flat_read_reshape {s t : Shape} (x : Tensor α s) (h : s.size = t.size)
    (index : Nat) :
    getAtOrZero ((x.reshape t h).reshape [_] (Shape.size_vector _).symm) [index] =
      getAtOrZero (x.reshape [_] (Shape.size_vector _).symm) [index] := by
  have key : ∀ {m₁ m₂ : Nat} (h₁ : s.size = Shape.size [m₁]) (h₂ : s.size = Shape.size [m₂]),
      getAtOrZero (x.reshape [m₁] h₁) [index] = getAtOrZero (x.reshape [m₂] h₂) [index] := by
    intro m₁ m₂ h₁ h₂
    obtain rfl : m₁ = m₂ := by rw [Shape.size_vector] at h₁ h₂; omega
    rfl
  exact key (h.trans (Shape.size_vector _).symm) _

/-- A flat vector whose entry at the row-major index of each coordinate of `L` is the entry of `y`
there, reshaped to a shape with as many entries, is `y` reshaped to that shape. -/
theorem Tensor.reshape_eq_reshape_of_flat_read {out L : Shape} (hs : L.size = out.size)
    (v : Tensor α [out.size]) (y : Tensor α L)
    (h : ∀ c : Coord L, getAtOrZero v [(Coord.linearize c).val] = y c) :
    v.reshape out (Shape.size_vector _) = y.reshape out hs := by
  refine (reshape_eq_of_flat_read rfl v (y.reshape out hs) fun c => ?_).trans rfl
  have hidx : (Coord.linearize (Coord.unlinearize
      (Fin.cast hs.symm (Coord.linearize c)))).val = (Coord.linearize c).val := by
    simp
  rw [Tensor.reshape_apply, ← h, hidx]

/-- The flat contraction `matmulFlat`, over operands whose batch prefixes broadcast to
`dims.leading`, is the registered batched product `Tensor.batchMatmul`. -/
theorem IR.MatmulDims.matmulFlat_eq_batchMatmul (dims : MatmulDims)
    (hA : Shape.CanBroadcastTo dims.leftLeading dims.leading)
    (hB : Shape.CanBroadcastTo dims.rightLeading dims.leading)
    (A : Tensor α (dims.leftLeading ++ [dims.rows, dims.inner]))
    (B : Tensor α (dims.rightLeading ++ [dims.inner, dims.cols]))
    (ho : dims.outShape = dims.leading ++ [dims.rows, dims.cols]) :
    Tensor.reshape (matmulFlat dims (A.reshape [_] (Shape.size_vector _).symm)
        (B.reshape [_] (Shape.size_vector _).symm)) dims.outShape (Shape.size_vector _) =
      Tensor.castShape (Tensor.batchMatmul hA hB A B) ho.symm :=
  reshape_eq_of_flat_read ho _ _ (matmulFlat_read dims hA hB A B (congrArg Shape.size ho))

end

/-- The flat contraction `matmulFlat` reads its operands only through `getAtOrZero`. -/
theorem IR.MatmulDims.matmulFlat_congr (dims : MatmulDims) {n₁ n₂ m₁ m₂ : Nat}
    {a₁ : Tensor α [n₁]} {a₂ : Tensor α [n₂]} {b₁ : Tensor α [m₁]} {b₂ : Tensor α [m₂]}
    (ha : ∀ index, getAtOrZero a₁ [index] = getAtOrZero a₂ [index])
    (hb : ∀ index, getAtOrZero b₁ [index] = getAtOrZero b₂ [index]) :
    matmulFlat dims a₁ b₁ = matmulFlat dims a₂ b₂ := by
  simp only [matmulFlat, ha, hb]

end TorchLean
