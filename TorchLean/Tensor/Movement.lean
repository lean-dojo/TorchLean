/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Algebra.BigOperators.Fin
public import TorchLean.Tensor.Axes
public import TorchLean.Tensor.Stride
public import TorchLean.Tensor.Reduce
public import TorchLean.Tensor.Conversion
public import TorchLean.Tensor.Rep.Segment

/-!
# Data movement on any axes

These operators take the axes they act on as `Fin s.rank` arguments and work at every rank. Each
output entry of an operator in the table, of `sliceBackward` or of `padBackward` is an input entry
or a constant (the padding value of `pad`, the coordinates of `indices`, the zeros of
`sliceBackward`), so they do no arithmetic and add no rounding. Each operator in the table has a
read lemma `op_apply` that gives its value at an output coordinate.

| Operator | PyTorch | JAX |
| --- | --- | --- |
| `broadcast x t dims` | `expand`, `unsqueeze` | `broadcast_in_dim` |
| `permute x σ out` | `permute` | `transpose` |
| `reshape x t` | `reshape` | `reshape` |
| `slice x dims starts stops steps` | `x[a:b:c]`, `narrow` | `slice` |
| `pad x low high interior value` | `F.pad` (constant) | `pad` |
| `cat dim sizes xs out` | `cat` | `concatenate` |
| `flip x dims` | `flip` | `rev` |
| `indices s dim` | `arange`, broadcast | `iota` |
| `gather x dims batchDims indices out` | `gather`, `index_select`, `embedding` | `gather` |

`sliceBackward`, the pullback of `slice`, writes each upstream entry back where the slice read it
and zero elsewhere. `padBackward`, the pullback of `pad` to its input, reads the upstream entry at
`padPosition`, where `pad` placed the input entry (`pad_padPosition`). `reduceBroadcast dims h`,
the pullback of `broadcast`, is the one operator here that adds. It sums over each axis of `t`
that no axis of `s` is placed at and over each repeated axis of extent one, one zero-seeded sum
per axis, by recursion over the axes of `t`. PyTorch's right-aligned placement of
`cb : CanBroadcastTo s t` is `cb.dims`, and at `s = t` the reduction returns its argument
(`reduceBroadcast_self`).

`reshape` is `Tensor.reshape`, and `reshape_apply` reads it. `permute`, `cat` and `gather` take
their output shape `out` as an argument with a proof that it is the shape they compute, which the
`tensor_shape` tactic supplies by default. Compiled code computes `permute` through
`permuteFast_eq`, one pass over the flat output that reads each entry at its strided flat
index (`weightedIndex_permute`). `cat` is the family concatenation `Tensor.Rep.concatenateAxes`,
with the axes before `dim` leading and the axes after it trailing.

The `index_*` lemmas give the indices of the coordinates that `Coord.setEquiv`,
`Coord.axisEquiv` and `Shape.Coord.appendEquiv` build. `cat_apply` compares
coordinates through them, as do the lemmas of `TorchLean.Tensor.Movement.SpecialForms`.

`Shape.adjacentSwap s depth` is the permutation exchanging the axes `depth` and `depth + 1`, so
`permute x (Shape.adjacentSwap [m, n] 0) [n, m]` is the transpose of a matrix. An index selection
is `gather` along one axis that batches every other axis with itself (`Shape.selectDims`,
`Shape.selectBatch`, `selectIndices`), and a scatter-add along one axis scatters over every axis
at the index tuples of `selectSources`.

Each operator is a hand transcription of the PyTorch and JAX semantics in the table. The capture
tests `test_data_movement` and `test_constant_index_gather_and_scatter_match_eager_words` in
`scripts/checks/test_ir_capture.py` compare `pad`, `flip`, `broadcast`, `slice`, `cat`, `permute`,
`reshape` and `gather` with eager PyTorch, and `test_data_movement_and_contraction` compares
`slice`, `pad`, `flip`, `cat` and `broadcast` with eager JAX. An imported index outside its axis
stops the run, as torch raises (`test_runtime_index_outside_its_axis_fails_execution`).
-/

@[expose] public section

open TorchLean.Shape.Coord (linearize unlinearize)
open TorchLean.Shape (rank rank_cons)
open TorchLean.Shape (rank_eq_length)

namespace TorchLean.Shape

/-- A vector of extent `n` broadcasts along the first axis of the shape `n :: s`. -/
theorem broadcastDims_head (n : Nat) (s : Shape) :
    BroadcastDims [n] (n :: s) fun _ => ⟨0, by simp [Shape.rank_eq_length]⟩ where
  strictMono i j hij := absurd hij (by
    have := j.isLt
    simp only [Shape.rank_eq_length, List.length_cons, List.length_nil] at this
    rw [Fin.lt_def]
    omega)
  extent_eq i := by
    match i with
    | ⟨0, _⟩ => exact Or.inr rfl
    | ⟨k + 1, hk⟩ => exact absurd hk (by simp [Shape.rank_eq_length])

/-- A vector of extent `n` broadcasts along the second axis of the shape `[m, n]`. -/
theorem broadcastDims_second (m n : Nat) :
    BroadcastDims [n] [m, n] fun _ => ⟨1, by simp [Shape.rank_eq_length]⟩ where
  strictMono i j hij := absurd hij (by
    have := j.isLt
    simp only [Shape.rank_eq_length, List.length_cons, List.length_nil] at this
    rw [Fin.lt_def]
    omega)
  extent_eq i := by
    match i with
    | ⟨0, _⟩ => exact Or.inr rfl
    | ⟨k + 1, hk⟩ => exact absurd hk (by simp [Shape.rank_eq_length])

/-! ## Swapping adjacent axes -/

/-- The position that swapping positions `depth` and `depth + 1` sends `k` to. -/
private def swapIndex (depth k : Nat) : Nat :=
  if k = depth then depth + 1 else if k = depth + 1 then depth else k

/-- Swapping adjacent extents at `depth` moves extent `swapIndex depth k` to position `k`. -/
private theorem getElem?_swapAdjacentAtDepth : (depth : Nat) → (s : Shape) →
    depth + 1 < s.length → (k : Nat) →
    (s.swapAdjacentAtDepth depth)[k]? = s[swapIndex depth k]?
  | 0, a :: b :: rest, _, k => by
      rcases k with _ | _ | k <;> simp [Shape.swapAdjacentAtDepth, swapIndex]
  | depth + 1, a :: rest, h, k => by
      have hRest : depth + 1 < rest.length := by simp at h; omega
      cases k with
      | zero => simp [Shape.swapAdjacentAtDepth, swapIndex]
      | succ k =>
          simp only [Shape.swapAdjacentAtDepth, List.getElem?_cons_succ]
          rw [getElem?_swapAdjacentAtDepth depth rest hRest k]
          unfold swapIndex
          split_ifs <;> first | omega | simp_all

/-- Swapping adjacent extents is permuting by the transposition of the two positions. -/
theorem permute_swap (s : Shape) (depth : Nat) (h : depth + 1 < s.rank) :
    s.permute (Equiv.swap ⟨depth, by omega⟩ ⟨depth + 1, h⟩) = s.swapAdjacentAtDepth depth := by
  apply List.ext_getElem?
  intro k
  rw [getElem?_swapAdjacentAtDepth depth s (by rwa [← Shape.rank_eq_length])]
  simp only [permute, List.getElem?_ofFn]
  by_cases hk : k < s.rank
  · have hk' : swapIndex depth k < s.length := by
      rw [← Shape.rank_eq_length]; unfold swapIndex; split_ifs <;> omega
    rw [List.getElem?_eq_getElem hk', dite_eq_left hk]
    simp only [Option.some.injEq, extent, Equiv.swap_apply_def, Fin.ext_iff]
    unfold swapIndex
    split_ifs <;> simp_all
  · rw [dite_eq_right hk, eq_comm, List.getElem?_eq_none]
    rw [← Shape.rank_eq_length]; unfold swapIndex; split_ifs <;> omega

/-- With fewer than two axes left at `depth`, swapping adjacent extents changes nothing. -/
theorem swapAdjacentAtDepth_of_rank_le (depth : Nat) (s : Shape)
    (h : s.rank ≤ depth + 1) : s.swapAdjacentAtDepth depth = s := by
  induction depth generalizing s with
  | zero =>
      match s, h with
      | [], _ => rfl
      | [_], _ => rfl
      | _ :: _ :: _, h => exact absurd h (by simp)
  | succ depth ih =>
      match s, h with
      | [], _ => rfl
      | _ :: rest, h =>
          simp only [swapAdjacentAtDepth]
          rw [ih rest (by simp only [Shape.rank_cons] at h; omega)]

/-- The permutation exchanging axes `depth` and `depth + 1` of `s`, and the identity when `s` has
fewer than `depth + 2` axes. It is irreducible, so that `tensor_shape` proves the output shape of a
`permute` by it with `permute_adjacentSwap`. -/
@[irreducible] def adjacentSwap (s : Shape) (depth : Nat) : Equiv.Perm (Fin s.rank) :=
  if h : depth + 1 < s.rank then Equiv.swap ⟨depth, by omega⟩ ⟨depth + 1, h⟩ else Equiv.refl _

/-- Permuting by the adjacent swap at `depth` swaps the extents at `depth` and `depth + 1`. -/
@[tensor_shape_simps] theorem permute_adjacentSwap (s : Shape) (depth : Nat) :
    s.permute (s.adjacentSwap depth) = s.swapAdjacentAtDepth depth := by
  unfold adjacentSwap
  split_ifs with h
  · exact s.permute_swap depth h
  · rw [permute_refl, swapAdjacentAtDepth_of_rank_le depth s (by omega)]

attribute [tensor_shape_simps] swapAdjacentAtDepth_involutive swapAdjacentAtDepth_append_rank

end TorchLean.Shape

namespace TorchLean.Tensor

open Shape (Coord)

variable {α : Type} [Storage α] {s : Shape}

/-! ## Broadcasting -/

/-- Broadcast `x` to the shape `t`, placing axis `i` of `x` at axis `dims i` of `t`. The result
is constant along the other axes of `t`, and an axis of extent one is repeated along its target
axis. -/
def broadcast (x : Tensor α s) (t : Shape) (dims : Fin s.rank → Fin t.rank)
    (h : Shape.BroadcastDims s t dims) : Tensor α t :=
  Rep.pull (fun c => Shape.Coord.ofFun fun i =>
    if hi : s.extent i = 1 then ⟨0, by omega⟩
    else Fin.cast ((h.extent_eq i).resolve_left hi).symm (Shape.Coord.get c (dims i))) x

/-- An output coordinate of `broadcast` reads `x` at its indices along the axes `dims i`, and at
index zero along each axis of extent one. -/
@[simp] theorem broadcast_apply (x : Tensor α s) (t : Shape) (dims : Fin s.rank → Fin t.rank)
    (h : Shape.BroadcastDims s t dims) (c : t.Coord) :
    broadcast x t dims h c = x (Shape.Coord.ofFun fun i =>
      if hi : s.extent i = 1 then ⟨0, by omega⟩
      else Fin.cast ((h.extent_eq i).resolve_left hi).symm (Shape.Coord.get c (dims i))) :=
  Rep.pull_apply _ x c

/-- When no axis of extent one has to be repeated, `broadcast` reads `x` at the indices along the
axes `dims i`. -/
theorem broadcast_apply_of_extent_eq {t : Shape} (x : Tensor α s) (dims : Fin s.rank → Fin t.rank)
    (h : Shape.BroadcastDims s t dims) (hext : ∀ i, s.extent i = t.extent (dims i)) (c : t.Coord) :
    broadcast x t dims h c =
      x (Shape.Coord.ofFun fun i => Fin.cast (hext i).symm (Shape.Coord.get c (dims i))) := by
  rw [broadcast_apply]
  congr 1
  apply Shape.Coord.ext_val
  intro i
  simp only [Shape.Coord.get_ofFun]
  split_ifs with hi
  · have hlt := (Shape.Coord.get c (dims i)).isLt
    have hone : t.extent (dims i) = 1 := (hext i).symm.trans hi
    show 0 = (Shape.Coord.get c (dims i)).val
    omega
  · rfl

/-- `Shape.CanBroadcastTo` in the list form used by the internal representation: after padding the
source with leading ones to the target rank, extents agree pointwise or the source extent is one. -/
theorem _root_.TorchLean.Shape.CanBroadcastTo.forall₂_toList {s₁ s₂ : Shape}
    (h : Shape.CanBroadcastTo s₁ s₂) :
    List.Forall₂ (fun a b => a = b ∨ a = 1)
      (Shape.padLeft (s₂.rank - s₁.rank) s₁) s₂ := by
  induction s₂ generalizing s₁ with
  | scalar =>
      cases s₁ with
      | scalar => exact List.Forall₂.nil
      | dim _ _ => exact absurd h Shape.not_canBroadcastTo_dim_scalar
  | dim n t ih =>
      cases s₁ with
      | scalar =>
          have hk : (Shape.dim n t).rank - Shape.rank Shape.scalar =
              (t.rank - Shape.rank Shape.scalar) + 1 := by
            simp
          rw [hk]
          exact List.Forall₂.cons (Or.inr rfl) (ih (Shape.canBroadcastTo_scalar_dim.mp h))
      | dim m s =>
          by_cases hRank : s.rank = t.rank
          · obtain ⟨hHead, hTail⟩ := (Shape.canBroadcastTo_dim_dim_of_rank_eq hRank).mp h
            have hk : (Shape.dim n t).rank - (Shape.dim m s).rank = 0 := by
              simp [hRank]
            have hk' : t.rank - s.rank = 0 := by grind
            have hTailList := ih hTail
            rw [hk'] at hTailList
            rw [hk]
            exact List.Forall₂.cons hHead hTailList
          · have hTail := (Shape.canBroadcastTo_dim_dim_of_rank_ne hRank).mp h
            have hle := hTail.rank_le
            have hk : (Shape.dim n t).rank - (Shape.dim m s).rank =
                (t.rank - (Shape.dim m s).rank) + 1 := by
              simp only [Shape.rank_cons] at hle ⊢
              grind
            rw [hk]
            exact List.Forall₂.cons (Or.inr rfl) (ih hTail)

/-- The right-aligned placement of `CanBroadcastTo s t`, as in PyTorch's `broadcast_to`: axis `i`
of `s` goes to axis `i + (t.rank - s.rank)` of `t`. -/
def _root_.TorchLean.Shape.CanBroadcastTo.dims {s t : Shape} (h : Shape.CanBroadcastTo s t) :
    Fin s.rank → Fin t.rank :=
  fun i => ⟨i.val + (t.rank - s.rank), by have := h.rank_le; omega⟩

/-- Right-aligned broadcasting places axis `i` of `s` at axis `i + (t.rank - s.rank)` of `t`, and
each extent of `s` is one or the extent there. -/
theorem _root_.TorchLean.Shape.CanBroadcastTo.broadcastDims {s t : Shape}
    (h : Shape.CanBroadcastTo s t) :
    Shape.BroadcastDims s t h.dims where
  strictMono a b hab := by
    simp only [Fin.lt_def, Shape.CanBroadcastTo.dims] at hab ⊢
    omega
  extent_eq i := by
    have hle := h.rank_le
    have hpair := List.forall₂_iff_get.mp h.forall₂_toList
    have hi : i.val + (t.rank - s.rank) < (Shape.padLeft (t.rank - s.rank) s).length := by
      simp only [← Shape.rank_eq_length, Shape.rank_padLeft]; omega
    have hpad := hpair.2 (i.val + (t.rank - s.rank)) hi
      (by rw [← hpair.1]; exact hi)
    simp only [List.get_eq_getElem, Shape.padLeft] at hpad
    rw [List.getElem_append_right (by simp)] at hpad
    simp only [List.length_replicate, Nat.add_sub_cancel] at hpad
    rcases hpad with hpad | hpad
    · exact Or.inr hpad
    · exact Or.inl hpad

/-- Broadcasting a tensor to its own shape returns it. -/
@[simp] theorem broadcast_self (x : Tensor α s) (h : Shape.CanBroadcastTo s s) :
    broadcast x s h.dims h.broadcastDims = x := by
  apply Rep.ext
  intro c
  rw [broadcast_apply]
  congr 1
  apply Shape.Coord.ext_val
  intro i
  simp only [Shape.Coord.get_ofFun]
  split_ifs with hi
  · have := (Shape.Coord.get c i).isLt
    show 0 = (Shape.Coord.get c i).val
    omega
  · show (Shape.Coord.get c (h.dims i)).val = (Shape.Coord.get c i).val
    rw [Shape.Coord.val_get, Shape.Coord.val_get]
    simp [Shape.CanBroadcastTo.dims]

/-- The pullback of `broadcast x t dims h`, by recursion over the axes of `t`. An outermost axis of
`t` with no axis of `s` placed at it is summed away, and one with an axis of the same extent placed
at it passes through slice by slice. One with an axis of extent one placed at it, at a different
extent, is summed and kept as a singleton. Every sum adds the slices in order from zero, even at
extent one or zero. Over a commutative additive monoid it sums each fiber of the coordinate map of
`broadcast`, and over the reals it is the adjoint of `broadcast`
(`reduceBroadcast_apply` and `inner_broadcast` in
`TorchLean.Tensor.Movement.BroadcastAdjoint`). -/
def reduceBroadcast [Add α] [Zero α] :
    {s t : Shape} → (dims : Fin s.rank → Fin t.rank) → Shape.BroadcastDims s t dims →
      Tensor α t → Tensor α s
  | .scalar, .scalar, _, _, g => g
  | .dim _ _, .scalar, dims, _, _ => (dims ⟨0, Nat.succ_pos _⟩).elim0
  | .scalar, .dim _ t, _, _, g =>
      reduceBroadcast (fun i => i.elim0) (Shape.BroadcastDims.scalar t)
        (castShape (reduce .sum g [⟨0, Nat.succ_pos _⟩] false) (Shape.reduce_singleton _ _))
  | .dim m _, .dim n _, dims, h, g =>
      if placed : (dims ⟨0, Nat.succ_pos _⟩).val = 0 then
        if hExtent : m = n then
          Tensor.dim fun i => reduceBroadcast _ h.tail (Tensor.unstack g (Fin.cast hExtent i))
        else
          let summed := reduceBroadcast _ h.tail
            (castShape (reduce .sum g [⟨0, Nat.succ_pos _⟩] false) (Shape.reduce_singleton _ _))
          Tensor.dim fun _ => summed
      else
        reduceBroadcast _ (h.shift (h.avoids_of_head placed))
          (castShape (reduce .sum g [⟨0, Nat.succ_pos _⟩] false) (Shape.reduce_singleton _ _))

/-- A map of scalars that preserves zero and addition commutes with `reduceBroadcast`. -/
theorem map_reduceBroadcast {β : Type} [Add α] [Zero α] [Storage β] [Add β] [Zero β] (f : α → β)
    (zero : f 0 = 0) (add : ∀ a b, f (a + b) = f a + f b) {t : Shape}
    (dims : Fin s.rank → Fin t.rank) (h : Shape.BroadcastDims s t dims) (g : Tensor α t) :
    map f (reduceBroadcast dims h g) = reduceBroadcast dims h (map f g) := by
  have columns {n : Nat} {inner : Shape} (t : Tensor α (.dim n inner)) :
      map f (castShape (target := inner) (reduce .sum t [⟨0, Nat.succ_pos _⟩] false)
          (Shape.reduce_singleton _ _)) =
        castShape (target := inner) (reduce .sum (map f t) [⟨0, Nat.succ_pos _⟩] false)
          (Shape.reduce_singleton _ _) := by
    rw [← map_reduce_sum f zero add]
    exact map_castShape f _ _
  induction dims, h, g using reduceBroadcast.induct with
  | case1 => rfl
  | case2 _ _ _ dims => exact (dims ⟨0, Nat.succ_pos _⟩).elim0
  | case3 _ _ _ _ _ ih => rw [reduceBroadcast, reduceBroadcast, ih, columns]
  | case4 _ _ _ _ _ _ placed ih =>
      rw [reduceBroadcast, reduceBroadcast, dite_eq_left placed, dite_eq_left placed,
        dite_eq_left rfl, dite_eq_left rfl, map_dim]
      simp only [unstack_map, ih]
  | case5 _ _ _ _ hExtent _ _ _ placed ih =>
      rw [reduceBroadcast, reduceBroadcast, dite_eq_left placed, dite_eq_left placed,
        dite_eq_right hExtent, dite_eq_right hExtent, map_dim, ih, columns]
  | case6 _ _ _ _ _ _ _ unplaced ih =>
      rw [reduceBroadcast, reduceBroadcast, dite_eq_right unplaced, dite_eq_right unplaced, ih,
        columns]

/-- A property of scalars that holds at zero and is closed under addition holds at every entry
of `reduceBroadcast` of a tensor whose entries all have it. -/
theorem reduceBroadcast_induction [Add α] [Zero α] (p : α → Prop)
    (hom : ∀ a b, p a → p b → p (a + b)) (unit : p 0) {t : Shape}
    (dims : Fin s.rank → Fin t.rank) (h : Shape.BroadcastDims s t dims) (g : Tensor α t)
    (base : ∀ c, p (g c)) (c : s.Coord) : p (reduceBroadcast dims h g c) := by
  let _ : Add {a // p a} := ⟨fun a b => ⟨a + b, hom _ _ a.2 b.2⟩⟩
  let _ : Zero {a // p a} := ⟨⟨0, unit⟩⟩
  let lifted : Tensor {a // p a} t := Rep.ofFn fun c => ⟨g c, base c⟩
  have values : map Subtype.val lifted = g := Rep.ext fun c => by simp [lifted]
  have reduced := map_reduceBroadcast Subtype.val rfl (fun _ _ => rfl) dims h lifted
  rw [values] at reduced
  rw [← reduced, Tensor.map_apply]
  exact (reduceBroadcast dims h lifted c).2

/-- `reduceBroadcast` at a placement of `s` into itself that keeps each axis in place returns its
argument. -/
private theorem reduceBroadcast_eq_self [Add α] [Zero α] :
    {s : Shape} → (dims : Fin s.rank → Fin s.rank) → (h : Shape.BroadcastDims s s dims) →
      (∀ i, (dims i).val = i.val) → (g : Tensor α s) → reduceBroadcast dims h g = g
  | .scalar, _, _, _, _ => rfl
  | .dim _ inner, dims, h, hd, g => by
      have placed : (dims ⟨0, Nat.succ_pos _⟩).val = 0 := hd _
      rw [reduceBroadcast, dite_eq_left placed, dite_eq_left rfl]
      apply Rep.ext
      rintro ⟨outer, coordinate⟩
      rw [Tensor.dim_apply]
      change (reduceBroadcast _ h.tail (Tensor.unstack g outer)) coordinate = g (outer, coordinate)
      rw [reduceBroadcast_eq_self _ h.tail (fun j => by
        have e := hd j.succ
        simp only [Shape.BroadcastDims.shiftDims, Fin.val_succ, Shape.rank_cons] at e ⊢; omega)
        (Tensor.unstack g outer)]
      exact Tensor.unstack_apply g outer coordinate

/-- `reduceBroadcast` at the placement of `s` into itself returns its argument. -/
theorem reduceBroadcast_self [Add α] [Zero α] (g : Tensor α s) (h : Shape.CanBroadcastTo s s) :
    reduceBroadcast h.dims h.broadcastDims g = g :=
  reduceBroadcast_eq_self _ _ (fun i => show i.val + (s.rank - s.rank) = i.val by omega) g

/-! ## Permutation -/

/-- Permute the axes of `x` into the shape `out`: axis `i` of the result is axis `σ i` of `x`, as in
PyTorch's `x.permute(σ 0, ..., σ (r - 1))`. -/
def permute (x : Tensor α s) (σ : Equiv.Perm (Fin s.rank)) (out : Shape)
    (hout : out = s.permute σ := by tensor_shape) : Tensor α out :=
  Rep.pull (fun c => Shape.Coord.ofFun fun j =>
    Fin.cast (s.extent_permute_symm σ j)
      (Shape.Coord.get (Shape.Coord.reindexEq hout c) ((σ.symm j).cast (s.rank_permute σ).symm))) x

/-- An output coordinate of `permute x σ` reads `x` with index `c (σ⁻¹ j)` along axis `j`. -/
@[simp] theorem permute_apply (x : Tensor α s) (σ : Equiv.Perm (Fin s.rank)) (out : Shape)
    (hout : out = s.permute σ) (c : out.Coord) :
    permute x σ out hout c = x (Shape.Coord.ofFun fun j =>
      Fin.cast (s.extent_permute_symm σ j)
        (Shape.Coord.get (Shape.Coord.reindexEq hout c)
          ((σ.symm j).cast (s.rank_permute σ).symm))) :=
  Rep.pull_apply _ x c

/-- Permuting `permute x σ out hout` by `s.unpermute σ hout` back into `s` returns `x`. -/
theorem permute_unpermute (x : Tensor α s) (σ : Equiv.Perm (Fin s.rank)) {out : Shape}
    (hout : out = s.permute σ) :
    permute (permute x σ out hout) (s.unpermute σ hout) s (s.permute_unpermute σ hout).symm =
      x := by
  subst hout
  refine Rep.ext fun c => ?_
  rw [permute_apply, permute_apply]
  congr 1
  apply Shape.Coord.ext_index
  intro k hk
  rw [Shape.Coord.index_ofFun, dite_eq_left hk]
  simp only [Fin.val_cast, Shape.Coord.val_get, Shape.Coord.index_reindexEq,
    Shape.Coord.index_ofFun]
  simp [Shape.unpermute, Equiv.permCongr_def]

/-- Map an output coordinate back to the input coordinate of an adjacent-axis swap.

At depth zero, an output coordinate `(j, i, rest)` reads `(i, j, rest)`. At greater
depths, the preceding coordinates stay fixed. If there are fewer than two axes
left to exchange, the coordinate stays unchanged, as in `Shape.swapAdjacentAtDepth`. -/
def _root_.TorchLean.Shape.Coord.swapAdjacentAtDepth (shape : Shape) (depth : Nat) :
    (shape.swapAdjacentAtDepth depth).Coord → shape.Coord :=
  match depth, shape with
  | 0, .dim _ (.dim _ _) =>
      fun coordinate => (coordinate.2.1, coordinate.1, coordinate.2.2)
  | depth + 1, .dim _ rest =>
      fun coordinate =>
        (coordinate.1, swapAdjacentAtDepth rest depth coordinate.2)
  | depth, .scalar => by
      cases depth <;> exact id
  | 0, .dim _ .scalar => id

/-- The source coordinate of an adjacent-axis swap swaps the indices at `depth`. -/
theorem _root_.TorchLean.Shape.Coord.toList_swapAdjacentAtDepth (s : Shape) (depth : Nat)
    (c : (s.swapAdjacentAtDepth depth).Coord) :
    Shape.Coord.toList s (Coord.swapAdjacentAtDepth s depth c) =
      Shape.swapAdjacentAtDepth (Shape.Coord.toList _ c) depth := by
  induction depth generalizing s with
  | zero =>
      match s with
      | .scalar => rfl
      | .dim _ .scalar => rfl
      | .dim _ (.dim _ _) => rfl
  | succ depth ih =>
      match s with
      | .scalar => rfl
      | .dim _ s => exact congrArg (c.1.val :: ·) (ih s c.2)

/-- The source coordinate of an adjacent swap reads the swapped position. -/
private theorem index_swapAdjacentAtDepth : (depth : Nat) → (s : Shape) →
    depth + 1 < s.rank → (c : (s.swapAdjacentAtDepth depth).Coord) → (k : Nat) →
    Shape.Coord.index (Coord.swapAdjacentAtDepth s depth c) k =
      Shape.Coord.index c (Shape.swapIndex depth k)
  | 0, _ :: _ :: _, _, c, k => by
      rcases k with _ | _ | k <;> rfl
  | depth + 1, _ :: rest, h, c, k => by
      have hRest : depth + 1 < Shape.rank rest := by simp only [Shape.rank_cons] at h; omega
      cases k with
      | zero => rfl
      | succ k =>
          have := index_swapAdjacentAtDepth depth rest hRest c.2 k
          change Shape.Coord.index (Coord.swapAdjacentAtDepth rest depth c.2) k = _
          rw [this]
          unfold Shape.swapIndex
          split_ifs <;> first | omega | rfl

/-- With fewer than two axes left at `depth`, the source coordinate of an adjacent swap keeps every
index. -/
private theorem index_swapAdjacentAtDepth_of_rank_le : (depth : Nat) → (s : Shape) →
    s.rank ≤ depth + 1 → (c : (s.swapAdjacentAtDepth depth).Coord) → (k : Nat) →
    Shape.Coord.index (Coord.swapAdjacentAtDepth s depth c) k = Shape.Coord.index c k
  | 0, [], _, _, _ => rfl
  | 0, [_], _, _, _ => rfl
  | 0, _ :: _ :: _, h, _, _ => absurd h (by simp)
  | depth + 1, [], _, _, _ => rfl
  | depth + 1, _ :: rest, h, c, k => by
      cases k with
      | zero => rfl
      | succ k =>
          change Shape.Coord.index (Coord.swapAdjacentAtDepth rest depth c.2) k = _
          exact index_swapAdjacentAtDepth_of_rank_le depth rest
            (by simp only [Shape.rank_cons] at h; omega) c.2 k

/-- An output coordinate of `permute x (s.adjacentSwap depth)` reads `x` at the coordinate with the
indices along `depth` and `depth + 1` exchanged. -/
theorem permute_adjacentSwap_apply (x : Tensor α s) (depth : Nat) (out : Shape)
    (hout : out = s.permute (s.adjacentSwap depth)) (c : out.Coord) :
    permute x (s.adjacentSwap depth) out hout c =
      x (Coord.swapAdjacentAtDepth s depth
        (cast (congrArg Shape.Coord (hout.trans (s.permute_adjacentSwap depth))) c)) := by
  rw [permute_apply]
  congr 1
  apply Shape.Coord.ext_index
  intro k hk
  rw [Shape.Coord.index_ofFun, dite_eq_left hk, Fin.val_cast, Shape.Coord.val_get,
    Fin.val_cast, Shape.Coord.index_reindexEq]
  by_cases h : depth + 1 < s.rank
  · rw [index_swapAdjacentAtDepth depth s h,
      Shape.Coord.index_cast (hout.trans (s.permute_adjacentSwap depth))]
    simp only [Shape.adjacentSwap, dite_eq_left h, Equiv.symm_swap, Equiv.swap_apply_def]
    unfold Shape.swapIndex
    split_ifs <;> simp_all [Fin.ext_iff]
  · rw [index_swapAdjacentAtDepth_of_rank_le depth s (by omega),
      Shape.Coord.index_cast (hout.trans (s.permute_adjacentSwap depth))]
    simp only [Shape.adjacentSwap, dite_eq_right h]
    rfl

/-- An output coordinate of `permute x (s.adjacentSwap depth)` into `s.swapAdjacentAtDepth depth`
reads `x` at the coordinate with the indices along `depth` and `depth + 1` exchanged. -/
theorem permute_swapAdjacentAtDepth_apply (x : Tensor α s) (depth : Nat)
    (hout : s.swapAdjacentAtDepth depth = s.permute (s.adjacentSwap depth))
    (c : (s.swapAdjacentAtDepth depth).Coord) :
    permute x (s.adjacentSwap depth) (s.swapAdjacentAtDepth depth) hout c =
      x (Coord.swapAdjacentAtDepth s depth c) := by
  rw [permute_adjacentSwap_apply, cast_eq]

/-- Exchanging the two leading axes reads the coordinate `(i, j, r)` at `(j, i, r)`. -/
theorem permute_adjacentSwap_zero_apply {m n : Nat} {t : Shape} (x : Tensor α (m :: n :: t))
    (hout : n :: m :: t = Shape.permute (m :: n :: t) (Shape.adjacentSwap (m :: n :: t) 0))
    (c : Shape.Coord (n :: m :: t)) :
    permute x (Shape.adjacentSwap (m :: n :: t) 0) (n :: m :: t) hout c =
      x (c.2.1, c.1, c.2.2) := by
  rw [permute_adjacentSwap_apply, cast_eq]
  rfl

/-- With fewer than two axes left at `depth`, an adjacent swap returns its input. -/
theorem permute_adjacentSwap_of_rank_le (tensor : Tensor α s) (depth : Nat)
    (h : s.rank ≤ depth + 1)
    (hout : s = s.permute (s.adjacentSwap depth)) :
    permute tensor (s.adjacentSwap depth) s hout = tensor := by
  apply Rep.ext
  intro c
  rw [permute_adjacentSwap_apply]
  congr 1
  apply Shape.Coord.ext_index
  intro k _
  rw [index_swapAdjacentAtDepth_of_rank_le depth s h,
    Shape.Coord.index_cast (hout.trans (s.permute_adjacentSwap depth))]

/-- The flat index that `permute x σ` reads for output entry `o`: the sum over the output axes `k`
of the index of `o` along `k` times the stride of axis `σ k` of `s`. -/
theorem weightedIndex_permute (σ : Equiv.Perm (Fin s.rank)) (out : Shape)
    (hout : out = s.permute σ) (o : Fin (Shape.size out)) :
    Shape.weightedIndex (List.ofFn fun k : Fin out.rank =>
        (out.stride k, s.stride (σ (k.cast ((congrArg Shape.rank hout).trans
          (s.rank_permute σ)))))) o.val =
      (Coord.linearize (Shape.Coord.ofFun fun j =>
        Fin.cast (s.extent_permute_symm σ j)
          (Shape.Coord.get (Shape.Coord.reindexEq hout (Coord.unlinearize o))
            ((σ.symm j).cast (s.rank_permute σ).symm)) : s.Coord)).val := by
  subst hout
  rw [Shape.weightedIndex_ofFn, Shape.Coord.linearize_val_eq_sum]
  simp only [Shape.Coord.get_ofFun, Fin.val_cast, Shape.Coord.reindexEq_self]
  refine Fintype.sum_equiv ((finCongr (s.rank_permute σ)).trans σ) _ _ fun k => ?_
  have hk : (σ.symm (σ (finCongr (s.rank_permute σ) k))).cast (s.rank_permute σ).symm = k := by
    ext
    simp
  rw [Equiv.trans_apply, hk]
  rfl

/-- `permute` as one pass over the flat output: entry `o` reads the input entry at the flat index
that `weightedIndex_permute` gives, without building coordinates. -/
def permuteFast {s : Shape} (x : Tensor α s) (σ : Equiv.Perm (Fin s.rank))
    (out : Shape) (hout : out = s.permute σ) : Tensor α out :=
  let axes := List.ofFn fun k : Fin out.rank =>
    (out.stride k, s.stride (σ (k.cast ((congrArg Shape.rank hout).trans (s.rank_permute σ)))))
  Rep.pullFlat (fun o => ⟨Shape.weightedIndex axes o.val, by
    rw [weightedIndex_permute σ out hout o]
    exact (Coord.linearize _).isLt⟩) x

/-- Compiled code computes `permute` with `permuteFast`. -/
@[csimp] theorem permuteFast_eq : @permute = @permuteFast := by
  funext α _ s x σ out hout
  exact (Rep.pullFlat_eq_pull _ _ (fun o => Fin.ext (weightedIndex_permute σ out hout o))
    x).symm

/-! ## Reshaping -/

/-- An output coordinate of `reshape` reads `x` at the coordinate with the same row-major index. -/
@[simp] theorem reshape_apply (x : Tensor α s) (t : Shape) (h : s.size = t.size) (c : t.Coord) :
    reshape x t h c = x (Coord.unlinearize (Fin.cast h.symm (Coord.linearize c))) :=
  Rep.reshape_apply _ x c

/-! ## Slicing -/

/-- Slice `x` along the axes `dims` at once: along `dims[k]` keep the indices
`starts[k], starts[k] + steps[k], ...` below `stops[k]`, as in `x[a:b:c]`. The other axes are
kept whole. Torch 2.8.0 clamps a stop past the end, so `torch.arange(5)[1:9]` has 4 entries, and
`SliceValid` refuses that stop here. -/
def slice (x : Tensor α s) (dims : List (Fin s.rank)) (starts stops steps : List Nat)
    (h : Shape.SliceValid s dims starts stops steps) :
    Tensor α (s.slice dims starts stops steps) :=
  Rep.pull (fun c => Shape.Coord.ofFun fun d =>
    ⟨Shape.valueAt dims starts 0 d + Shape.valueAt dims steps 1 d *
        (Shape.Coord.get c (d.cast (s.rank_slice dims starts stops steps).symm)).val,
      h.add_mul_lt d _⟩) x

/-- An output coordinate of `slice` reads `x` with index `start + step * i` along each axis, where
`i` is its own index there and `start` and `step` are the listed ones, or `0` and `1` on an
unlisted axis. -/
@[simp] theorem slice_apply (x : Tensor α s) (dims : List (Fin s.rank))
    (starts stops steps : List Nat) (h : Shape.SliceValid s dims starts stops steps)
    (c : (s.slice dims starts stops steps).Coord) :
    slice x dims starts stops steps h c = x (Shape.Coord.ofFun fun d =>
      ⟨Shape.valueAt dims starts 0 d + Shape.valueAt dims steps 1 d *
          (Shape.Coord.get c (d.cast (s.rank_slice dims starts stops steps).symm)).val,
        h.add_mul_lt d _⟩) :=
  Rep.pull_apply _ x c

/-- The pullback of `slice x dims starts stops steps` at `g`: an entry read by the slice receives
the upstream entry that read it, and every other entry zero. -/
def sliceBackward [Zero α] (dims : List (Fin s.rank)) (starts stops steps : List Nat)
    (g : Tensor α (s.slice dims starts stops steps)) : Tensor α s :=
  Rep.ofFn fun c =>
    if hc : ∀ d : Fin (s.slice dims starts stops steps).rank,
        let e := d.cast (s.rank_slice dims starts stops steps)
        Shape.valueAt dims starts 0 e ≤ (Shape.Coord.get c e).val ∧
        ((Shape.Coord.get c e).val - Shape.valueAt dims starts 0 e) %
          Shape.valueAt dims steps 1 e = 0 ∧
        ((Shape.Coord.get c e).val - Shape.valueAt dims starts 0 e) /
          Shape.valueAt dims steps 1 e < (s.slice dims starts stops steps).extent d then
      g (Shape.Coord.ofFun fun d =>
        ⟨((Shape.Coord.get c (d.cast (s.rank_slice dims starts stops steps))).val -
            Shape.valueAt dims starts 0 (d.cast (s.rank_slice dims starts stops steps))) /
          Shape.valueAt dims steps 1 (d.cast (s.rank_slice dims starts stops steps)),
          (hc d).2.2⟩)
    else 0

/-! ## Padding -/

/-- Pad `x` with `value`: along axis `d`, add `low d` entries before, `high d` after and
`interior d` between consecutive entries, as JAX's `pad` does with nonnegative padding (cropping
is `slice`). -/
def pad (x : Tensor α s) (low high interior : Fin s.rank → Nat) (value : α) :
    Tensor α (s.pad low high interior) :=
  Rep.ofFn fun c =>
    if h : ∀ d, low d ≤ (Shape.Coord.get c (d.cast (s.rank_pad low high interior).symm)).val ∧
        ((Shape.Coord.get c (d.cast (s.rank_pad low high interior).symm)).val - low d) %
          (interior d + 1) = 0 ∧
        ((Shape.Coord.get c (d.cast (s.rank_pad low high interior).symm)).val - low d) /
          (interior d + 1) < s.extent d then
      x (Shape.Coord.ofFun fun d =>
        ⟨((Shape.Coord.get c (d.cast (s.rank_pad low high interior).symm)).val - low d) /
          (interior d + 1), (h d).2.2⟩)
    else value

/-- An output coordinate of `pad` reads `x` where every index `o` is `low + (interior + 1) * i`
for an input index `i`, and is `value` elsewhere. -/
theorem pad_apply (x : Tensor α s) (low high interior : Fin s.rank → Nat) (value : α)
    (c : (s.pad low high interior).Coord) :
    pad x low high interior value c =
      if h : ∀ d, low d ≤ (Shape.Coord.get c (d.cast (s.rank_pad low high interior).symm)).val ∧
          ((Shape.Coord.get c (d.cast (s.rank_pad low high interior).symm)).val - low d) %
            (interior d + 1) = 0 ∧
          ((Shape.Coord.get c (d.cast (s.rank_pad low high interior).symm)).val - low d) /
            (interior d + 1) < s.extent d then
        x (Shape.Coord.ofFun fun d =>
          ⟨((Shape.Coord.get c (d.cast (s.rank_pad low high interior).symm)).val - low d) /
            (interior d + 1), (h d).2.2⟩)
      else value :=
  Rep.get_ofFn _ c

/-- An input index `i` of an axis of extent `n` lands at `low + (interior + 1) * i`, inside the
padded extent. -/
theorem pad_index_lt {low high interior n i : Nat} (h : i < n) :
    low + (interior + 1) * i < low + high + (n + (n - 1) * interior) := by
  have hm : (interior + 1) * i ≤ (interior + 1) * (n - 1) := Nat.mul_le_mul_left _ (by omega)
  have he : (interior + 1) * (n - 1) = (n - 1) + (n - 1) * interior := by
    rw [Nat.add_mul, Nat.one_mul, Nat.mul_comm, Nat.add_comm]
  omega

/-- The coordinate of `pad x low high interior value` that holds the entry of `x` at `c`: index
`low d + (interior d + 1) * i` along each axis `d`. -/
def padPosition (low high interior : Fin s.rank → Nat) (c : s.Coord) :
    (s.pad low high interior).Coord :=
  Shape.Coord.ofFun fun d =>
    ⟨low (d.cast (s.rank_pad low high interior)) +
        (interior (d.cast (s.rank_pad low high interior)) + 1) *
          (Shape.Coord.get c (d.cast (s.rank_pad low high interior))).val, by
      rw [Shape.extent_pad]
      exact pad_index_lt (Shape.Coord.get c _).isLt⟩

/-- The output of `pad` at `padPosition low high interior c` is the entry of `x` at `c`. -/
theorem pad_padPosition (x : Tensor α s) (low high interior : Fin s.rank → Nat) (value : α)
    (c : s.Coord) : pad x low high interior value (padPosition low high interior c) = x c := by
  rw [pad_apply]
  have key : ∀ d : Fin s.rank,
      (Shape.Coord.get (padPosition low high interior c)
        (d.cast (s.rank_pad low high interior).symm)).val =
        low d + (interior d + 1) * (Shape.Coord.get c d).val := by
    intro d
    simp [padPosition, Shape.Coord.get_ofFun]
  have hcond : ∀ d : Fin s.rank, low d ≤ (Shape.Coord.get (padPosition low high interior c)
        (d.cast (s.rank_pad low high interior).symm)).val ∧
      ((Shape.Coord.get (padPosition low high interior c)
        (d.cast (s.rank_pad low high interior).symm)).val - low d) % (interior d + 1) = 0 ∧
      ((Shape.Coord.get (padPosition low high interior c)
        (d.cast (s.rank_pad low high interior).symm)).val - low d) / (interior d + 1) <
        s.extent d := by
    intro d
    rw [key d, Nat.add_sub_cancel_left, Nat.mul_mod_right,
      Nat.mul_div_cancel_left _ (Nat.succ_pos _)]
    exact ⟨Nat.le_add_right _ _, rfl, (Shape.Coord.get c d).isLt⟩
  split
  · congr 1
    apply Shape.Coord.ext_val
    intro d
    simp only [Shape.Coord.get_ofFun]
    rw [key d, Nat.add_sub_cancel_left, Nat.mul_div_cancel_left _ (Nat.succ_pos _)]
  · exact absurd hcond ‹_›

/-- The pullback of `pad x low high interior value` to `x` at `g`: each entry reads `g` where `pad`
placed the input entry (`padPosition`). -/
def padBackward (low high interior : Fin s.rank → Nat) (g : Tensor α (s.pad low high interior)) :
    Tensor α s :=
  Rep.ofFn fun c => g (padPosition low high interior c)

/-- Entry `c` of `padBackward low high interior g` is `g` at `padPosition low high interior c`. -/
@[simp] theorem padBackward_apply (low high interior : Fin s.rank → Nat)
    (g : Tensor α (s.pad low high interior)) (c : s.Coord) :
    padBackward low high interior g c = g (padPosition low high interior c) :=
  Rep.get_ofFn _ c

/-! ## Indices through the coordinate equivalences -/

/-- A shape cast reads the tensor at the coordinate with the same indices. -/
theorem castShape_apply_reindex {t : Shape} (x : Tensor α s) (h : s = t) (c : t.Coord) :
    castShape x h c = x (Shape.Coord.reindexEq h.symm c) := by
  subst h
  rw [Shape.Coord.reindexEq_self]
  rfl

/-- Shifting a comparison past an outer axis. -/
private theorem ite_add_one (k a : Nat) :
    (if k + 1 < a + 1 then k + 1 else k + 1 + 1) = (if k < a then k else k + 1) + 1 := by
  split_ifs <;> omega

/-- The index along `axis` is the first component of `Shape.Coord.setEquiv`. -/
theorem _root_.TorchLean.Shape.Coord.val_setEquiv_fst : (axis : Nat) → {shape : Shape} →
    [_h : Shape.AxisInBounds axis shape] → (count : Nat) →
    (c : Shape.Coord (shape.set axis count)) →
    (Coord.setEquiv axis (shape := shape) count c).1.val =
      Shape.Coord.index c axis
  | 0, _ :: _, _, _, _ => rfl
  | axis + 1, _ :: inner, h, count, c => by
      have : Shape.AxisInBounds axis inner :=
        ⟨by have := h.proof; simp only [Shape.rank_cons] at this; omega⟩
      exact val_setEquiv_fst axis (shape := inner) count c.2

/-- The other indices, in order, are the second component of
`Shape.Coord.setEquiv`. -/
theorem _root_.TorchLean.Shape.Coord.index_setEquiv_snd : (axis : Nat) → {shape : Shape} →
    [_h : Shape.AxisInBounds axis shape] → (count : Nat) →
    (c : Shape.Coord (shape.set axis count)) → (k : Nat) →
    Shape.Coord.index (Coord.setEquiv axis (shape := shape) count c).2 k =
      Shape.Coord.index c (if k < axis then k else k + 1)
  | 0, _ :: _, _, _, _, _ => rfl
  | axis + 1, _ :: inner, h, count, c, k => by
      have : Shape.AxisInBounds axis inner :=
        ⟨by have := h.proof; simp only [Shape.rank_cons] at this; omega⟩
      cases k with
      | zero =>
          rw [ite_eq_left (Nat.succ_pos axis)]
          rfl
      | succ k =>
          rw [ite_add_one]
          exact index_setEquiv_snd axis (shape := inner) count c.2 k

/-- The coordinate `(Shape.Coord.setEquiv axis count).symm (j, r)` has index `j`
along `axis` and the indices of `r` along the other axes, in order. -/
theorem _root_.TorchLean.Shape.Coord.index_setEquiv_symm (axis : Nat) {shape : Shape}
    [Shape.AxisInBounds axis shape] (count : Nat) (j : Fin count)
    (r : Shape.Coord (shape.eraseIdx axis)) (k : Nat) :
    Shape.Coord.index ((Coord.setEquiv axis (shape := shape) count).symm
        (j, r)) k =
      if k < axis then Shape.Coord.index r k
      else if k = axis then j.val else Shape.Coord.index r (k - 1) := by
  set c := (Coord.setEquiv axis (shape := shape) count).symm (j, r)
  have hc : Coord.setEquiv axis (shape := shape) count c = (j, r) :=
    Equiv.apply_symm_apply _ _
  have hj := Coord.val_setEquiv_fst axis count c
  have hr := Coord.index_setEquiv_snd axis count c
  rw [hc] at hj hr
  split_ifs with h₁ h₂
  · rw [hr k, ite_eq_left h₁]
  · rw [h₂, ← hj]
  · rw [hr (k - 1), ite_eq_right (by omega), Nat.sub_add_cancel (by omega)]

/-- The leading part of `Shape.Coord.appendEquiv` keeps the leading indices. -/
theorem _root_.TorchLean.Shape.Coord.index_appendEquiv_fst :
      (l t : Shape) → (c : (l ++ t).Coord) → (k : Nat) →
    k < l.length →
    Shape.Coord.index (Coord.appendEquiv l t c).1 k = Shape.Coord.index c k
  | [], _, _, _, h => absurd h (Nat.not_lt_zero _)
  | _ :: l, t, c, k, h => by
      cases k with
      | zero => rfl
      | succ k =>
          exact index_appendEquiv_fst l t c.2 k (by simp only [List.length_cons] at h; omega)

/-- The trailing part of `Shape.Coord.appendEquiv` keeps the trailing indices, shifted. -/
theorem _root_.TorchLean.Shape.Coord.index_appendEquiv_snd :
      (l t : Shape) → (c : (l ++ t).Coord) → (k : Nat) →
    Shape.Coord.index (Coord.appendEquiv l t c).2 k =
      Shape.Coord.index c (k + l.length)
  | [], _, _, _ => rfl
  | _ :: l, t, c, k => by
      rw [List.length_cons, ← Nat.add_assoc]
      exact index_appendEquiv_snd l t c.2 k

/-- The coordinate `(Shape.Coord.appendEquiv l t).symm (cl, ct)` has the indices of `cl` and
then those of `ct`. -/
theorem _root_.TorchLean.Shape.Coord.index_appendEquiv_symm (l t : Shape) (cl : l.Coord)
    (ct : t.Coord) (k : Nat) :
    Shape.Coord.index ((Coord.appendEquiv l t).symm (cl, ct)) k =
      if k < l.length then Shape.Coord.index cl k else Shape.Coord.index ct (k - l.length) := by
  set c := (Coord.appendEquiv l t).symm (cl, ct)
  have hc : Coord.appendEquiv l t c = (cl, ct) := Equiv.apply_symm_apply _ _
  split_ifs with h
  · have := Coord.index_appendEquiv_fst l t c k h
    rw [hc] at this
    exact this.symm
  · have := Coord.index_appendEquiv_snd l t c (k - l.length)
    rw [hc, Nat.sub_add_cancel (by omega)] at this
    exact this.symm

/-- The other indices, in order, are the second component of `Shape.Coord.axisEquiv`. -/
theorem _root_.TorchLean.Shape.Coord.index_axisEquiv_snd : (axis : Nat) → {shape : Shape} →
    [_h : Shape.AxisInBounds axis shape] → (c : shape.Coord) → (k : Nat) →
    Shape.Coord.index (Coord.axisEquiv axis (shape := shape) c).2 k =
      Shape.Coord.index c (if k < axis then k else k + 1)
  | 0, _ :: _, _, _, _ => rfl
  | axis + 1, _ :: inner, h, c, k => by
      have : Shape.AxisInBounds axis inner :=
        ⟨by have := h.proof; simp only [Shape.rank_cons] at this; omega⟩
      cases k with
      | zero =>
          rw [ite_eq_left (Nat.succ_pos axis)]
          rfl
      | succ k =>
          rw [ite_add_one]
          exact index_axisEquiv_snd axis (shape := inner) c.2 k

/-! ## Concatenation -/

/-- An entry of `Rep.concatenateAxes` whose index along the joined axis falls `o` past the
start of block `j` reads block `j` at offset `o`. -/
private theorem concatenateAxes_apply_of_index (leading trailing : Shape) (lengths : List Nat)
    (xs : (j : Fin lengths.length) → Tensor α (leading ++ lengths.get j :: trailing))
    (c : (leading ++ lengths.sum :: trailing).Coord) (j : Fin lengths.length)
    (o : Fin (lengths.get j))
    (h : Shape.Coord.index c leading.length =
      ∑ i : Fin j.val, lengths.get (Fin.castLE j.isLt.le i) + o) :
    Rep.concatenateAxes leading trailing lengths xs c =
      let parts := Coord.appendEquiv leading (lengths.sum :: trailing) c
      xs j ((Coord.appendEquiv leading (lengths.get j :: trailing)).symm
        (parts.1, (o, parts.2.2))) := by
  set parts := Coord.appendEquiv leading (lengths.sum :: trailing) c
  have hj : (Coord.segmentEquiv lengths).symm parts.2.1 = ⟨j, o⟩ := by
    rw [Equiv.symm_apply_eq]
    apply Fin.ext
    change _ = (finSigmaFinEquiv (⟨j, o⟩ : (i : Fin lengths.length) × Fin (lengths.get i))).val
    rw [finSigmaFinEquiv_apply, ← h]
    simpa using Coord.index_appendEquiv_snd leading (lengths.sum :: trailing) c 0
  have hseg : (Coord.concatEquiv leading trailing lengths).symm c =
      ⟨((Coord.segmentEquiv lengths).symm parts.2.1).1,
        (Coord.appendEquiv leading
          (lengths.get ((Coord.segmentEquiv lengths).symm parts.2.1).1 :: trailing)).symm
          (parts.1, ((Coord.segmentEquiv lengths).symm parts.2.1).2, parts.2.2)⟩ :=
    rfl
  rw [Rep.concatenateAxes, Rep.get_ofFn]
  dsimp only
  rw [hseg]
  generalize (Coord.segmentEquiv lengths).symm parts.2.1 = q at hj ⊢
  subst hj
  rfl

/-- Concatenate the tensors `xs i` along `dim`, in the order of `i`, into the shape `out`, by
`Rep.concatenateAxes` with the axes of `s` before `dim` leading and those after it
trailing. Tensor `xs i` has extent `sizes i` along `dim` and the extents of `s` elsewhere, so the
extent of `s` along `dim` is ignored. -/
def cat {k : Nat} (dim : Fin s.rank) (sizes : Fin k → Nat)
    (xs : (i : Fin k) → Tensor α (s.set dim (sizes i))) (out : Shape)
    (hout : out = s.set dim (∑ i, sizes i) := by tensor_shape) : Tensor α out :=
  castShape (Rep.concatenateAxes (s.take dim) (s.drop (dim + 1)) (List.ofFn sizes)
      fun i => castShape (xs (i.cast List.length_ofFn)) (by
        rw [List.get_ofFn]
        exact List.set_eq_take_cons_drop _ (dim.isLt.trans_eq s.rank_eq_length)))
    (by rw [hout, List.sum_ofFn,
      List.set_eq_take_cons_drop _ (dim.isLt.trans_eq s.rank_eq_length)])

/-- An output coordinate of `cat` reads the tensor whose block contains its index along `dim`, at
the offset into that block, and at the same indices along the other axes. -/
theorem cat_apply {k : Nat} (dim : Fin s.rank) (sizes : Fin k → Nat)
    (xs : (i : Fin k) → Tensor α (s.set dim (sizes i))) (out : Shape)
    (hout : out = s.set dim (∑ i, sizes i)) (c : out.Coord) :
    cat dim sizes xs out hout c =
      let split := Coord.setEquiv dim.val (shape := s) (∑ i, sizes i)
        (Shape.Coord.reindexEq hout c)
      let part := finSigmaFinEquiv.symm split.1
      xs part.1 ((Coord.setEquiv dim.val (shape := s) (sizes part.1)).symm
        (part.2, split.2)) := by
  have hdim : dim.val < s.length := dim.isLt.trans_eq s.rank_eq_length
  have hlead : (s.take dim).length = dim := by simp [hdim.le]
  dsimp only
  rcases hp : finSigmaFinEquiv.symm (Coord.setEquiv dim.val (shape := s)
    (∑ i, sizes i) (Shape.Coord.reindexEq hout c)).1 with ⟨j, o⟩
  have hidx : Shape.Coord.index c dim = ∑ i : Fin j.val, sizes (Fin.castLE j.isLt.le i) + o := by
    have e := congrArg (fun p => (finSigmaFinEquiv p).val) hp
    simp only [Equiv.apply_symm_apply, finSigmaFinEquiv_apply] at e
    rw [← e, Coord.val_setEquiv_fst, Shape.Coord.index_reindexEq]
  rw [cat, castShape_apply_reindex, concatenateAxes_apply_of_index _ _ _ _ _
    (j.cast List.length_ofFn.symm) (o.cast (by simp)) (by
      rw [Shape.Coord.index_reindexEq, hlead, hidx]
      simp only [List.get_ofFn]
      rfl)]
  dsimp only
  rw [castShape_apply_reindex]
  show xs j _ = _
  congr 1
  apply Shape.Coord.ext_index
  intro m _
  simp only [Shape.Coord.index_reindexEq, Coord.index_setEquiv_symm,
    Coord.index_setEquiv_snd]
  rw [Coord.index_appendEquiv_symm, hlead]
  by_cases h1 : m < dim.val
  · simp only [h1, ite_true]
    rw [Coord.index_appendEquiv_fst _ _ _ _ (by omega), Shape.Coord.index_reindexEq]
  · by_cases h2 : m = dim.val
    · subst h2
      simp
    · have h3 : ¬ m - 1 < dim.val := by omega
      have hm : m - dim.val = (m - dim.val - 1) + 1 := by omega
      simp only [h1, h2, h3, ite_false]
      rw [hm, Shape.Coord.index_cons_succ, ← Shape.Coord.index_succ, Coord.index_appendEquiv_snd,
        Shape.Coord.index_reindexEq, hlead]
      congr 1
      omega

-- The output shape of a concatenation: the extents of a two-block family add, and a replaced axis
-- of a literal shape is read through the `List.set` equations.
attribute [tensor_shape_simps] Fin.sum_univ_two Matrix.cons_val_zero Matrix.cons_val_one
  Matrix.head_cons List.set_nil List.set_cons_zero List.set_cons_succ

/-! ## Reversal -/

/-- Reverse `x` along every axis in `dims`. An axis listed more than once is reversed once. Torch
2.8.0's `torch.flip(x, (0, 0))` raises "dim 0 appears multiple times in the list of dims". -/
def flip (x : Tensor α s) (dims : List (Fin s.rank)) : Tensor α s :=
  Rep.pull (fun c => Shape.Coord.ofFun fun d =>
    if d ∈ dims then (Shape.Coord.get c d).rev else Shape.Coord.get c d) x

/-- An output coordinate of `flip` reads `x` with index `n - 1 - i` along each listed axis of
extent `n` and the same index elsewhere. -/
@[simp] theorem flip_apply (x : Tensor α s) (dims : List (Fin s.rank)) (c : s.Coord) :
    flip x dims c = x (Shape.Coord.ofFun fun d =>
      if d ∈ dims then (Shape.Coord.get c d).rev else Shape.Coord.get c d) :=
  Rep.pull_apply _ x c

/-! ## Coordinates -/

/-- The tensor of shape `s` holding at each coordinate its index along `dim` (JAX's `iota`). -/
def indices (s : Shape) (dim : Fin s.rank) : Tensor (Fin (s.extent dim)) s :=
  Rep.ofFn fun c => Shape.Coord.get c dim

/-- `indices s dim` holds at `c` the index of `c` along `dim`. -/
@[simp] theorem indices_apply (s : Shape) (dim : Fin s.rank) (c : s.Coord) :
    indices s dim c = Shape.Coord.get c dim :=
  Rep.get_ofFn _ c

/-! ## Gathering -/

/-- Gather entries of `x` at the positions in `indices`. Each entry of `indices` gives an index
along every axis in `dims`. A batch pair `(a, b)` reads axis `a` of `x` at the index of the
index-tensor coordinate along its axis `b`. The result, of shape `out`, has the axes of `indices`
followed by the remaining axes of `x`, which are kept whole. Each index is a `Fin` of the extent of
its axis, so the index that torch 2.8.0's `torch.gather` rejects at run time ("index 3 is out of
bounds for dimension 0 with size 3") is a type error here. -/
def gather (x : Tensor α s) (dims : List (Fin s.rank)) {t : Shape}
    (batchDims : List (Fin s.rank × Fin t.rank))
    (indices : Tensor (Einops.AxisTuple s.extent dims) t) (out : Shape)
    (h : Shape.GatherValid s t dims batchDims)
    (hout : out = t ++ s.eraseAxes (dims ++ batchDims.map Prod.fst) := by tensor_shape) :
    Tensor α out :=
  Rep.pull (fun c =>
    let parts := Coord.appendEquiv t (s.eraseAxes (dims ++ batchDims.map Prod.fst))
      (Shape.Coord.reindexEq hout c)
    h.source (indices parts.1) parts.1 parts.2) x

/-- An output coordinate `(ct, cr)` of `gather` reads `x` at the index `indices ct` along the
indexed axes, at `ct` along the batch axes and at `cr` along the remaining axes. -/
@[simp] theorem gather_apply (x : Tensor α s) (dims : List (Fin s.rank)) {t : Shape}
    (batchDims : List (Fin s.rank × Fin t.rank))
    (indices : Tensor (Einops.AxisTuple s.extent dims) t) (out : Shape)
    (h : Shape.GatherValid s t dims batchDims)
    (hout : out = t ++ s.eraseAxes (dims ++ batchDims.map Prod.fst)) (c : out.Coord) :
    gather x dims batchDims indices out h hout c =
      let parts := Coord.appendEquiv t (s.eraseAxes (dims ++ batchDims.map Prod.fst))
        (Shape.Coord.reindexEq hout c)
      x (h.source (indices parts.1) parts.1 parts.2) :=
  Rep.pull_apply _ x c

/-- The index tensor of an index selection along `dim`: each output coordinate holds the entry of
`indices` at its index along `dim`. -/
def selectIndices (dim : Fin s.rank) {count : Nat}
    (indices : Tensor (Fin (s.axisSize dim)) [count]) :
    Tensor (Einops.AxisTuple s.extent (s.selectDims dim)) (s.set dim count) :=
  Rep.ofFn fun c i =>
    Fin.cast ((Shape.axisSize_eq_extent s dim).trans
        (congrArg s.extent (s.get_selectDims dim i).symm))
      (indices.getScalar (Coord.setEquiv dim (shape := s) count c).1)

/-- The coordinates a scatter-add along `dim` writes to, as index tuples over every axis: at each
coordinate of `s.set dim count`, the entry of `indices` at its index along `dim` and its
indices along the other axes. -/
def selectSources (dim : Fin s.rank) {count : Nat}
    (indices : Tensor (Fin (s.axisSize dim)) [count]) :
    Tensor (Einops.AxisTuple s.extent (List.finRange s.rank)) (s.set dim count) :=
  Rep.ofFn fun u k =>
    Shape.Coord.get ((Coord.axisEquiv dim (shape := s)).symm
      (indices.getScalar (Coord.setEquiv dim (shape := s) count u).1,
        (Coord.setEquiv dim (shape := s) count u).2))
      ((List.finRange s.rank).get k)

end TorchLean.Tensor
