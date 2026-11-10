/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.NN.Softmax
public import TorchLean.NN.Norm.MatrixEntries
public import TorchLean.Tensor.Operations
public import TorchLean.Tensor.Reduce.Real
public import TorchLean.Tensor.Op.Table.Rule
public import TorchLean.Verify.RuntimeApprox.OnlineSoftmax.Real
public import TorchLean.Verify.RuntimeApprox.TransformerBlock

/-!
# Transformer block operations as op-table forwards

At the reals, `layerNorm`, `softmax`, `attention`, `heads` and `block` of
`TorchLean.Verify.RuntimeApprox.TransformerBlock` are op-table forwards, read entry by entry. They
are `Op.layerNorm`, `Op.softmax` on the last axis, `Op.attention`, and for `block` their composition
with `Op.linear`, `Op.reshape`, `Op.permute` and `Op.add`, the heads batched in one `Op.attention`.
The attention statements hold for both schedules, without a mask and with a mask whose allowed
entries in a row are the keys of that row.
-/

@[expose] public section

open TorchLean.Shape (rank size)

namespace TorchLean.TransformerBlock

open Tensor

private theorem sqrt_max_zero (a : ℝ) : √(max a 0) = √a := by
  rcases le_total a 0 with h | h
  · rw [max_eq_right h, Real.sqrt_zero, Real.sqrt_eq_zero_of_nonpos h]
  · rw [max_eq_left h]

/-- The forward of `Op.layerNorm` on a matrix with positive extents is `layerNorm` of each row,
with the weight, bias and epsilon of the op. -/
theorem get2_layerNorm_forward {n d : ℕ} (hn : n > 0) (hd : d > 0) (hlead : n = Shape.size [n])
    (hx : [n, d] = [n, d]) (x : Tensor ℝ [n, d]) (γ β : Tensor ℝ [d])
    (ε : Tensor ℝ []) (t : Fin n) (j : Fin d) :
    get2 (((Op.layerNorm hn hd hlead hx).rule? (α := ℝ) |>.get rfl).forward
        (x, γ, β, ε, ())).1 t j =
      layerNorm (fun i => Tensor.getScalar γ i) (fun i => Tensor.getScalar β i) ε.item
        (fun i => get2 x t i) j := by
  change get2 (Tensor.layerNorm x [⟨1, Nat.one_lt_two⟩] γ β ε.item) t j = _
  rw [Normalization.get2_layerNorm hd]
  simp only [layerNorm, sqrt_max_zero]
  ring

/-- The forward of `Op.softmax` on the last axis of a matrix is `softmax` over all keys of each
row. -/
theorem get2_softmax_forward {n d : ℕ} (x : Tensor ℝ [n, d]) (t : Fin n) (j : Fin d) :
    get2 (((Op.softmax (s := [n, d]) ⟨1, Nat.one_lt_two⟩).rule? (α := ℝ) |>.get rfl).forward
        (x, ())).1 t j =
      softmax Finset.univ (fun i => get2 x t i) j := by
  change get2 (Tensor.alongAxis ⟨1, Nat.one_lt_two⟩
    (fun x => Softmax.innermost x) x) t j = _
  rw [Tensor.alongAxis_of_last _ _ x rfl, get2_eq_getScalar_unstack]
  rw [Softmax.unstack_innermost_matrix, Softmax.getScalar_vec_eq_exp_div]
  rfl

private theorem batchSlice_unit {α : Type} [Storage α] {s : Shape}
    (x : Tensor α s) (b : Shape.Coord []) : x.batchSlice b = x :=
  Tensor.Rep.ext fun q => Tensor.Rep.get_ofFn _ q

private theorem scaledMatmul_apply {n m d : ℕ} (A : Tensor ℝ [n, d]) (B : Tensor ℝ [d, m])
    (s : ℝ) (t : Fin n) (j : Fin m) :
    Attention.scaledMatmul (batch := []) (Shape.CanBroadcastTo.refl [])
      (Shape.CanBroadcastTo.refl []) (m := n) (n := d) (p := m) A B s (t, j, ()) =
      (∑ i, get2 A t i * get2 B i j) * s := by
  rw [Attention.scaledMatmul, Attention.scaledProducts_apply]
  simp only [Tensor.broadcast_self, batchSlice_unit]
  rw [Attention.scaledProductsSlice_one Numeric.QuotientArithmetic.supported_real]
  simp only [Bool.false_eq_true, ↓reduceIte, Tensor.scale]
  erw [Tensor.map_apply]
  change Tensor.matmul A B (t, j, ()) * s = _
  rw [← get2_eq_apply, get2_matmul_eq_sum]

private theorem get2_permute_swap {m d : ℕ} (k : Tensor ℝ [m, d])
    (hout : [d, m] = Shape.permute [m, d]
      (Shape.adjacentSwap ([m, d]) (Shape.rank [])))
    (i : Fin d) (j : Fin m) :
    get2 (Tensor.permute k (Shape.adjacentSwap [m, d] (Shape.rank []))
        [d, m] hout) i j = get2 k j i := by
  rw [get2_eq_apply, get2_eq_apply]
  exact Tensor.permute_adjacentSwap_zero_apply k hout (i, j, ())

private theorem attentionValues_apply_unit {n m w : ℕ} (scores : Tensor ℝ [n, m])
    (v : Tensor ℝ [m, w]) (mask : Option (Tensor Bool [n, m])) (t : Fin n) (c : Fin w)
    (S : Fin m → ℝ) (hS : ∀ j, scores (t, j, ()) = S j) :
    Attention.values (batch := [])
        (Attention.weights (batch := []) scores mask) v none (t, c, ()) =
      (∑ j, if mask.elim true (fun mask => get2 mask t j) then
          Real.exp (S j) * get2 v j c else 0) /
        ∑ j, if mask.elim true (fun mask => get2 mask t j) then Real.exp (S j) else 0 := by
  have h := OnlineSoftmax.attentionValues_apply_real
    (batch := []) scores v none mask () t c
  simp only [batchSlice_unit, Option.elim_none, hS] at h
  simp only [get2_eq_apply]
  exact h

private theorem attention_twoPass_apply {n m d w : ℕ} (q : Tensor ℝ [n, d])
    (k : Tensor ℝ [m, d]) (v : Tensor ℝ [m, w]) (s : ℝ) (mask : Option (Tensor Bool [n, m]))
    (t : Fin n) (c : Fin w) :
    get2 (Tensor.attention (batch := []) q k v s mask none) t c =
      attention s (Finset.univ.filter fun j => mask.elim true (fun mask => get2 mask t j) = true)
        (fun i => get2 q t i) (fun j i => get2 k j i) (fun j c => get2 v j c) c := by
  rw [get2_eq_apply]
  dsimp only [Tensor.attention]
  rw [attentionValues_apply_unit _ v mask t c (fun j => (∑ i, get2 q t i * get2 k j i) * s) ?hS]
  · simp only [attention, softmax, Finset.sum_filter, Finset.sum_div]
    refine Finset.sum_congr rfl fun j _ => ?_
    split_ifs <;> simp only [mul_comm s, div_mul_eq_mul_div, zero_div]
  · intro j
    refine (scaledMatmul_apply q _ s t j).trans ?_
    congr 1
    refine Finset.sum_congr rfl fun i _ => ?_
    congr 1
    exact get2_permute_swap k _ i j

private theorem attention_apply {n m d w : ℕ} (q : Tensor ℝ [n, d]) (k : Tensor ℝ [m, d])
    (v : Tensor ℝ [m, w]) (s : ℝ) (mask : Option (Tensor Bool [n, m]))
    (schedule : Softmax.Schedule) (t : Fin n) (c : Fin w) :
    get2 (Tensor.attention (batch := []) q k v s mask none schedule) t c =
      attention s (Finset.univ.filter fun j => mask.elim true (fun mask => get2 mask t j) = true)
        (fun i => get2 q t i) (fun j i => get2 k j i) (fun j c => get2 v j c) c := by
  cases schedule with
  | twoPass => exact attention_twoPass_apply q k v s mask t c
  | online blocks =>
    rw [Tensor.attention_online_eq (batch := [])]
    exact attention_twoPass_apply q k v s mask t c

/-- The forward of `Op.attention` without a mask, for every schedule, is `attention` over all keys
with the frozen scale, at each row and column. -/
theorem get2_attention_forward {n m d w : ℕ} (schedule : Softmax.Schedule)
    (q : Tensor ℝ [n, d]) (k : Tensor ℝ [m, d]) (v : Tensor ℝ [m, w]) (τ : Tensor ℝ [])
    (t : Fin n) (c : Fin w) :
    get2 (((Op.attention (batch := []) (rows := n) (keys := m) (depth := d) (width := w)
        none schedule).rule? (α := ℝ) |>.get rfl).forward (q, k, v, τ, ())).1 t c =
      attention τ.item Finset.univ (fun i => get2 q t i) (fun j i => get2 k j i)
        (fun j c => get2 v j c) c := by
  change get2 (Tensor.attention (batch := []) q k v τ.item none none schedule) t c = _
  rw [attention_apply]
  simp only [Option.elim_none, Finset.filter_true]

/-- The forward of `Op.attention` with a mask, for every schedule, is `attention` over the keys
the mask allows in the row. A row without an allowed key gives zero on both sides. -/
theorem get2_attention_forward_mask {n m d w : ℕ} (mask : Tensor Bool [n, m])
    (schedule : Softmax.Schedule) (q : Tensor ℝ [n, d]) (k : Tensor ℝ [m, d])
    (v : Tensor ℝ [m, w]) (τ : Tensor ℝ []) (t : Fin n) (c : Fin w) :
    get2 (((Op.attention (batch := []) (rows := n) (keys := m) (depth := d) (width := w)
        (some mask) schedule).rule? (α := ℝ) |>.get rfl).forward (q, k, v, τ, ())).1 t c =
      attention τ.item (Finset.univ.filter fun j => get2 mask t j = true) (fun i => get2 q t i)
        (fun j i => get2 k j i) (fun j c => get2 v j c) c :=
  attention_apply q k v τ.item (some mask) schedule t c

/-- Head `z` of `heads` over all keys is the forward of `Op.attention` without a mask on the
slices `q z`, `k z` and `v z` of head `z`, for every schedule. -/
theorem heads_eq_attention_forward {n h e : ℕ} (schedule : Softmax.Schedule)
    (q k v : Fin h → Tensor ℝ [n, e]) (τ : Tensor ℝ []) (t : Fin n) (z : Fin h) (c : Fin e) :
    heads τ.item (fun _ => Finset.univ) (fun t p => get2 (q p.1) t p.2)
        (fun j p => get2 (k p.1) j p.2) (fun j p => get2 (v p.1) j p.2) t (z, c) =
      get2 (((Op.attention (batch := []) (rows := n) (keys := n) (depth := e) (width := e)
        none schedule).rule? (α := ℝ) |>.get rfl).forward (q z, k z, v z, τ, ())).1 t c :=
  (get2_attention_forward schedule (q z) (k z) (v z) τ t c).symm

/-- Head `z` of `heads`, token `t` attending over the keys the mask allows in row `t`, is the
forward of `Op.attention` with that mask on the slices of head `z`, for every schedule. -/
theorem heads_eq_attention_forward_mask {n h e : ℕ} (mask : Tensor Bool [n, n])
    (schedule : Softmax.Schedule) (q k v : Fin h → Tensor ℝ [n, e]) (τ : Tensor ℝ [])
    (t : Fin n) (z : Fin h) (c : Fin e) :
    heads τ.item (fun t => Finset.univ.filter fun j => get2 mask t j = true)
        (fun t p => get2 (q p.1) t p.2) (fun j p => get2 (k p.1) j p.2)
        (fun j p => get2 (v p.1) j p.2) t (z, c) =
      get2 (((Op.attention (batch := []) (rows := n) (keys := n) (depth := e) (width := e)
        (some mask) schedule).rule? (α := ℝ) |>.get rfl).forward (q z, k z, v z, τ, ())).1 t c :=
  (get2_attention_forward_mask mask schedule (q z) (k z) (v z) τ t c).symm

private theorem batchSlice_heads {α : Type} [Storage α] {h : ℕ} {s : Shape}
    (x : Tensor α (h :: s)) (b : Shape.Coord [h]) :
    Tensor.batchSlice (batch := [h]) (s := s) x b = x.unstack b.1 :=
  (Tensor.batchSlice_mk (batch := []) (s := s) x b.1 b.2).trans (batchSlice_unit _ _)

private theorem batchSlice_heads_apply {α : Type} [Storage α] {h : ℕ} {s : Shape}
    (x : Tensor α (h :: s)) (b : Shape.Coord [h]) (q : s.Coord) :
    Tensor.batchSlice (batch := [h]) (s := s) x b q = x (b.1, q) := by
  rw [batchSlice_heads]
  exact Tensor.unstack_apply x b.1 q

private theorem scaledMatmul_apply_heads {h n m d : ℕ} (A : Tensor ℝ [h, n, d])
    (B : Tensor ℝ [h, d, m]) (s : ℝ) (z : Fin h) (t : Fin n) (j : Fin m) :
    Attention.scaledMatmul (batch := [h]) (Shape.CanBroadcastTo.refl [h])
      (Shape.CanBroadcastTo.refl [h]) (m := n) (n := d) (p := m) A B s (z, t, j, ()) =
      (∑ i, A (z, t, i, ()) * B (z, i, j, ())) * s := by
  rw [Attention.scaledMatmul, Attention.scaledProducts_apply]
  simp only [Tensor.broadcast_self, batchSlice_heads]
  rw [Attention.scaledProductsSlice_one Numeric.QuotientArithmetic.supported_real]
  simp only [Bool.false_eq_true, ↓reduceIte, Tensor.scale]
  erw [Tensor.map_apply]
  change Tensor.matmul (A.unstack z) (B.unstack z) (t, j, ()) * s = _
  rw [← get2_eq_apply, get2_matmul_eq_sum]
  simp only [get2_eq_apply, Tensor.unstack_apply]

private theorem permute_swap_heads_apply {h m d : ℕ} (k : Tensor ℝ [h, m, d])
    (hout : [h, d, m] = Shape.permute [h, m, d]
      (Shape.adjacentSwap ([h, m, d]) (Shape.rank [h])))
    (z : Fin h) (i : Fin d) (j : Fin m) :
    Tensor.permute k (Shape.adjacentSwap [h, m, d] (Shape.rank [h]))
        [h, d, m] hout (z, i, j, ()) = k (z, j, i, ()) := by
  rw [Tensor.permute_adjacentSwap_apply]
  rfl

private theorem attentionValues_apply_heads {h n m w : ℕ} (scores : Tensor ℝ [h, n, m])
    (v : Tensor ℝ [h, m, w]) (mask : Option (Tensor Bool [n, m])) (z : Fin h) (t : Fin n)
    (c : Fin w) (S : Fin m → ℝ) (hS : ∀ j, scores (z, t, j, ()) = S j) :
    Attention.values (batch := [h])
        (Attention.weights (batch := [h]) scores mask) v none (z, t, c, ()) =
      (∑ j, if mask.elim true (fun mask => get2 mask t j) then
          Real.exp (S j) * v (z, j, c, ()) else 0) /
        ∑ j, if mask.elim true (fun mask => get2 mask t j) then Real.exp (S j) else 0 := by
  have h := OnlineSoftmax.attentionValues_apply_real
    (batch := [h]) scores v none mask (z, ()) t c
  simp only [batchSlice_heads_apply, Option.elim_none, hS] at h
  simp only [get2_eq_apply]
  exact h

private theorem attention_twoPass_apply_heads {h n m d w : ℕ}
    (q : Tensor ℝ [h, n, d]) (k : Tensor ℝ [h, m, d])
    (v : Tensor ℝ [h, m, w]) (s : ℝ)
    (mask : Option (Tensor Bool [n, m])) (z : Fin h) (t : Fin n) (c : Fin w) :
    (Tensor.attention (batch := [h]) q k v s mask none) (z, t, c, ()) =
      attention s (Finset.univ.filter fun j => mask.elim true (fun mask => get2 mask t j) = true)
        (fun i => q (z, t, i, ())) (fun j i => k (z, j, i, ())) (fun j c => v (z, j, c, ())) c := by
  dsimp only [Tensor.attention]
  rw [attentionValues_apply_heads _ v mask z t c
    (fun j => (∑ i, q (z, t, i, ()) * k (z, j, i, ())) * s) ?hS]
  · simp only [attention, softmax, Finset.sum_filter, Finset.sum_div]
    refine Finset.sum_congr rfl fun j _ => ?_
    split_ifs <;> simp only [mul_comm s, div_mul_eq_mul_div, zero_div]
  · intro j
    refine (scaledMatmul_apply_heads q _ s z t j).trans ?_
    congr 1
    refine Finset.sum_congr rfl fun i _ => ?_
    congr 1
    exact permute_swap_heads_apply k _ z i j

private theorem attention_apply_heads {h n m d w : ℕ} (q : Tensor ℝ [h, n, d])
    (k : Tensor ℝ [h, m, d]) (v : Tensor ℝ [h, m, w]) (s : ℝ)
    (mask : Option (Tensor Bool [n, m])) (schedule : Softmax.Schedule) (z : Fin h) (t : Fin n)
    (c : Fin w) :
    Tensor.attention (batch := [h]) q k v s mask none schedule (z, t, c, ()) =
      attention s (Finset.univ.filter fun j => mask.elim true (fun mask => get2 mask t j) = true)
        (fun i => q (z, t, i, ())) (fun j i => k (z, j, i, ())) (fun j c => v (z, j, c, ())) c := by
  cases schedule with
  | twoPass => exact attention_twoPass_apply_heads q k v s mask z t c
  | online blocks =>
    rw [Tensor.attention_online_eq (batch := [h])]
    exact attention_twoPass_apply_heads q k v s mask z t c

private theorem attention_forward_apply_heads {h n m d w : ℕ} (schedule : Softmax.Schedule)
    (q : Tensor ℝ [h, n, d]) (k : Tensor ℝ [h, m, d])
    (v : Tensor ℝ [h, m, w]) (τ : Tensor ℝ []) (z : Fin h) (t : Fin n)
    (c : Fin w) :
    (((Op.attention (batch := [h]) (rows := n) (keys := m) (depth := d) (width := w)
        none schedule).rule? (α := ℝ) |>.get rfl).forward (q, k, v, τ, ())).1 (z, t, c, ()) =
      attention τ.item Finset.univ (fun i => q (z, t, i, ())) (fun j i => k (z, j, i, ()))
        (fun j c => v (z, j, c, ())) c := by
  change Tensor.attention (batch := [h]) q k v τ.item none none schedule (z, t, c, ()) = _
  rw [attention_apply_heads]
  simp only [Option.elim_none, Finset.filter_true]

private theorem attention_forward_apply_heads_mask {h n m d w : ℕ} (mask : Tensor Bool [n, m])
    (schedule : Softmax.Schedule) (q : Tensor ℝ [h, n, d])
    (k : Tensor ℝ [h, m, d]) (v : Tensor ℝ [h, m, w])
    (τ : Tensor ℝ []) (z : Fin h) (t : Fin n) (c : Fin w) :
    (((Op.attention (batch := [h]) (rows := n) (keys := m) (depth := d) (width := w)
        (some mask) schedule).rule? (α := ℝ) |>.get rfl).forward (q, k, v, τ, ())).1 (z, t, c, ()) =
      attention τ.item (Finset.univ.filter fun j => get2 mask t j = true)
        (fun i => q (z, t, i, ())) (fun j i => k (z, j, i, ())) (fun j c => v (z, j, c, ())) c :=
  attention_apply_heads q k v τ.item (some mask) schedule z t c

private theorem get2_add_forward {n d : ℕ} (x y : Tensor ℝ [n, d]) (t : Fin n) (i : Fin d) :
    get2 (((Op.add (s := [n, d])).rule? (α := ℝ) |>.get rfl).forward (x, y, ())).1 t i =
      get2 x t i + get2 y t i := by
  change get2 (x + y) t i = _
  rw [get2_eq_apply, get2_eq_apply, get2_eq_apply]
  exact Tensor.add_apply x y (t, i, ())

private theorem get2_linear_forward {n d m : ℕ} (W : Tensor ℝ [m, d]) (b : Tensor ℝ [m])
    (x : Tensor ℝ [n, d]) (t : Fin n) (c : Fin m) :
    get2 (((Op.linear (lead := [n]) (inDim := d) (outDim := m)).rule? (α := ℝ) |>.get rfl).forward
        (W, b, x, ())).1 t c =
      Tensor.getScalar b c + ∑ j, get2 x t j * get2 W c j := by
  change Tensor.getScalar (Tensor.unstack
    (Tensor.mapLeading [n] (Linear.forward { weights := W, bias := b }) x) t) c = _
  rw [Tensor.unstack_mapLeading, Linear.forward, Tensor.getScalar_add]
  dsimp only
  rw [Tensor.getScalar_contract_matVec_eq_sum, add_comm]
  congr 1
  exact Finset.sum_congr rfl fun k _ => mul_comm _ _

private theorem reshape_apply_of_val {s t : Shape} (y : Tensor ℝ s) (h : s.size = t.size)
    (i : t.Coord) (c : s.Coord)
    (hv : (Shape.Coord.linearize i).val = (Shape.Coord.linearize c).val) :
    Tensor.reshape y t h i = y c := by
  have e : Tensor.reshape y t h i = y (Shape.Coord.unlinearize
      (finCongr h.symm (Shape.Coord.linearize i))) :=
    Tensor.Rep.reshape_apply _ y i
  rw [e]
  congr 1
  apply Shape.Coord.linearize_injective
  rw [Shape.Coord.linearize_unlinearize]
  exact Fin.ext (by simp [hv])

private theorem linearize_heads {n h e : ℕ} (t : Fin n) (z : Fin h) (c : Fin e) :
    (Shape.Coord.linearize (s := [n, h, e]) (t, z, c, ())).val =
      (Shape.Coord.linearize (s := [n, h * e]) (t, finProdFinEquiv (z, c), ())).val := by
  erw [Shape.Coord.linearize_cons_val, Shape.Coord.linearize_cons_val,
    Shape.Coord.linearize_cons_val, Shape.Coord.linearize_cons_val,
    Shape.Coord.linearize_cons_val]
  simp only [finProdFinEquiv_apply_val, Shape.size_cons,
    Shape.size_nil]
  ring

private theorem split_forward_apply {n h e : ℕ} (y : Tensor ℝ [n, h * e])
    (hsplit : Shape.size [n, h * e] = Shape.size [n, h, e])
    (hto : [h, n, e] = Shape.permute [n, h, e] (Shape.adjacentSwap [n, h, e] 0))
    (z : Fin h) (t : Fin n) (c : Fin e) :
    (((Op.permute (Shape.adjacentSwap [n, h, e] 0) [h, n, e] hto).rule? (α := ℝ)
        |>.get rfl).forward ((((Op.reshape hsplit).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1, ())).1 (z, t, c, ()) =
      get2 y t (finProdFinEquiv (z, c)) := by
  change Tensor.permute (Tensor.reshape y [n, h, e] hsplit) (Shape.adjacentSwap [n, h, e] 0)
    [h, n, e] hto (z, t, c, ()) = _
  refine (Tensor.permute_adjacentSwap_zero_apply (Tensor.reshape y [n, h, e] hsplit) hto
    (z, t, c, ())).trans ?_
  rw [get2_eq_apply]
  exact reshape_apply_of_val _ hsplit _ _ (linearize_heads t z c)

private theorem sum_merge_forward {n h e : ℕ} (o : Tensor ℝ [h, n, e])
    (hfrom : [n, h, e] = Shape.permute [h, n, e]
      (Shape.adjacentSwap [h, n, e] 0))
    (hmerge : Shape.size [n, h, e] = Shape.size [n, h * e])
    (t : Fin n) (w : Fin (h * e) → ℝ) :
    ∑ p, get2 (((Op.reshape hmerge).rule? (α := ℝ) |>.get rfl).forward
        ((((Op.permute (Shape.adjacentSwap [h, n, e] 0) [n, h, e] hfrom).rule?
          (α := ℝ) |>.get rfl).forward (o, ())).1, ())).1 t p * w p =
      ∑ zc : Fin h × Fin e, o (zc.1, t, zc.2, ()) * w (finProdFinEquiv zc) := by
  rw [← Equiv.sum_comp finProdFinEquiv]
  refine Finset.sum_congr rfl fun zc _ => ?_
  congr 1
  change get2 (Tensor.reshape (Tensor.permute o (Shape.adjacentSwap [h, n, e] 0)
    [n, h, e] hfrom) [n, h * e] hmerge) t (finProdFinEquiv zc) = _
  rw [get2_eq_apply]
  refine (reshape_apply_of_val _ hmerge _ (t, zc.1, zc.2, ())
    (linearize_heads t zc.1 zc.2).symm).trans ?_
  exact Tensor.permute_adjacentSwap_zero_apply (m := h) (n := n) (t := [e]) o hfrom
    (t, zc.1, zc.2, ())

/-- With positive extents and an op `σ` acting entrywise as `act`, `block` at token `t`, entry `i`
is a composition of op forwards. It runs `Op.layerNorm`, `Op.linear`, `Op.reshape` and `Op.permute`
to the head axis, `Op.attention` without a mask for every schedule, the way back, and `Op.add` for
the residuals. The parameters are read from the tensors, head `z` and column `c` at index
`finProdFinEquiv (z, c)`. -/
theorem block_eq_forward {n d h e f : ℕ} (hn : n > 0) (hd : d > 0) (hlead : n = Shape.size [n])
    (hx : [n, d] = [n, d])
    (hsplit : Shape.size [n, h * e] = Shape.size [n, h, e])
    (hto : [h, n, e] = Shape.permute [n, h, e] (Shape.adjacentSwap [n, h, e] 0))
    (hfrom : [n, h, e] = Shape.permute [h, n, e]
      (Shape.adjacentSwap [h, n, e] 0))
    (hmerge : Shape.size [n, h, e] = Shape.size [n, h * e])
    (act : ℝ → ℝ) (σ : Op [.value [n, f]] [.value [n, f]]) (hσ : (σ.rule? (α := ℝ)).isSome)
    (hact : ∀ y t c, get2 ((σ.rule?.get hσ).forward (y, ())).1 t c = act (get2 y t c))
    (schedule : Softmax.Schedule) (x : Tensor ℝ [n, d]) (γ1 β1 γ2 β2 bo b2 : Tensor ℝ [d])
    (ε τ : Tensor ℝ []) (Wq Wk Wv : Tensor ℝ [h * e, d]) (bq bk bv : Tensor ℝ [h * e])
    (Wo : Tensor ℝ [d, h * e]) (W1 : Tensor ℝ [f, d]) (b1 : Tensor ℝ [f])
    (W2 : Tensor ℝ [d, f]) (t : Fin n) (i : Fin d) :
    let norm : Tensor ℝ [n, d] → Tensor ℝ [d] → Tensor ℝ [d] → Tensor ℝ [n, d] := fun y γ β =>
      (((Op.layerNorm hn hd hlead hx).rule? (α := ℝ) |>.get rfl).forward (y, γ, β, ε, ())).1
    let split : Tensor ℝ [h * e, d] → Tensor ℝ [h * e] → Tensor ℝ [h, n, e] :=
      fun W b =>
        (((Op.permute (Shape.adjacentSwap [n, h, e] 0) [h, n, e] hto).rule?
          (α := ℝ) |>.get rfl).forward ((((Op.reshape hsplit).rule? (α := ℝ) |>.get rfl).forward
            ((((Op.linear (lead := [n]) (inDim := d) (outDim := h * e)).rule? (α := ℝ)
              |>.get rfl).forward (W, b, norm x γ1 β1, ())).1, ())).1, ())).1
    let o : Tensor ℝ [h, n, e] :=
      (((Op.attention (batch := [h]) (rows := n) (keys := n) (depth := e) (width := e)
        none schedule).rule? (α := ℝ) |>.get rfl).forward
          (split Wq bq, split Wk bk, split Wv bv, τ, ())).1
    let m : Tensor ℝ [n, h * e] :=
      (((Op.reshape hmerge).rule? (α := ℝ) |>.get rfl).forward
        ((((Op.permute (Shape.adjacentSwap [h, n, e] 0) [n, h, e] hfrom).rule?
          (α := ℝ) |>.get rfl).forward (o, ())).1, ())).1
    let s : Tensor ℝ [n, d] := (((Op.add (s := [n, d])).rule? (α := ℝ) |>.get rfl).forward
      (x, (((Op.linear (lead := [n]) (inDim := h * e) (outDim := d)).rule? (α := ℝ)
        |>.get rfl).forward (Wo, bo, m, ())).1, ())).1
    let g : Tensor ℝ [n, f] := ((σ.rule?.get hσ).forward
      ((((Op.linear (lead := [n]) (inDim := d) (outDim := f)).rule? (α := ℝ) |>.get rfl).forward
        (W1, b1, norm s γ2 β2, ())).1, ())).1
    block act
        { γ1 := Tensor.getScalar γ1, β1 := Tensor.getScalar β1
          wq := fun j p => get2 Wq (finProdFinEquiv p) j
          bq := fun p => Tensor.getScalar bq (finProdFinEquiv p)
          wk := fun j p => get2 Wk (finProdFinEquiv p) j
          bk := fun p => Tensor.getScalar bk (finProdFinEquiv p)
          wv := fun j p => get2 Wv (finProdFinEquiv p) j
          bv := fun p => Tensor.getScalar bv (finProdFinEquiv p)
          wo := fun p j => get2 Wo j (finProdFinEquiv p), bo := Tensor.getScalar bo
          γ2 := Tensor.getScalar γ2, β2 := Tensor.getScalar β2
          w1 := fun j c => get2 W1 c j, b1 := Tensor.getScalar b1
          w2 := fun c j => get2 W2 j c, b2 := Tensor.getScalar b2 }
        ε.item τ.item (fun _ => Finset.univ) (fun t j => get2 x t j) t i =
      get2 (((Op.add (s := [n, d])).rule? (α := ℝ) |>.get rfl).forward
        (s, (((Op.linear (lead := [n]) (inDim := f) (outDim := d)).rule? (α := ℝ)
          |>.get rfl).forward (W2, b2, g, ())).1, ())).1 t i := by
  intro norm split o m s g
  have hA : ∀ (y : Tensor ℝ [n, d]) (γ β : Tensor ℝ [d]) t j, get2 (norm y γ β) t j =
      layerNorm (Tensor.getScalar γ) (Tensor.getScalar β) ε.item (fun i => get2 y t i) j :=
    fun y γ β t j => get2_layerNorm_forward hn hd hlead hx y γ β ε t j
  have hQ : ∀ (W : Tensor ℝ [h * e, d]) (b : Tensor ℝ [h * e]) (z : Fin h) (t : Fin n)
      (c : Fin e), split W b (z, t, c, ()) = Tensor.getScalar b (finProdFinEquiv (z, c)) +
        ∑ j, get2 (norm x γ1 β1) t j * get2 W (finProdFinEquiv (z, c)) j :=
    fun W b z t c => (split_forward_apply _ hsplit hto z t c).trans (get2_linear_forward W b _ t _)
  have ho : ∀ z t c, o (z, t, c, ()) = attention τ.item Finset.univ
      (fun i => split Wq bq (z, t, i, ())) (fun j i => split Wk bk (z, j, i, ()))
      (fun j c => split Wv bv (z, j, c, ())) c :=
    fun z t c => attention_forward_apply_heads schedule _ _ _ τ z t c
  have hS : ∀ t i, get2 s t i = get2 x t i + (Tensor.getScalar bo i +
      ∑ zc : Fin h × Fin e, o (zc.1, t, zc.2, ()) * get2 Wo i (finProdFinEquiv zc)) :=
    fun t i => (get2_add_forward x _ t i).trans (congrArg (get2 x t i + ·)
      ((get2_linear_forward Wo bo m t i).trans (congrArg (Tensor.getScalar bo i + ·)
        (sum_merge_forward o hfrom hmerge t fun p => get2 Wo i p))))
  have hG : ∀ t c, get2 g t c =
      act (Tensor.getScalar b1 c + ∑ j, get2 (norm s γ2 β2) t j * get2 W1 c j) :=
    fun t c => (hact _ t c).trans (congrArg act (get2_linear_forward W1 b1 _ t c))
  refine Eq.trans ?_ ((get2_add_forward s _ t i).trans
    (congrArg (get2 s t i + ·) (get2_linear_forward W2 b2 g t i))).symm
  simp only [block, heads, hS, hG, hA, ho, hQ]

/-- `block_eq_forward` with a mask: token `t` attends over the keys the mask allows in row `t`, and
`Op.attention` takes that mask. A row without an allowed key gives zero attention on both sides. -/
theorem block_eq_forward_mask {n d h e f : ℕ} (hn : n > 0) (hd : d > 0) (hlead : n = Shape.size [n])
    (hx : [n, d] = [n, d])
    (hsplit : Shape.size [n, h * e] = Shape.size [n, h, e])
    (hto : [h, n, e] = Shape.permute [n, h, e] (Shape.adjacentSwap [n, h, e] 0))
    (hfrom : [n, h, e] = Shape.permute [h, n, e]
      (Shape.adjacentSwap [h, n, e] 0))
    (hmerge : Shape.size [n, h, e] = Shape.size [n, h * e])
    (mask : Tensor Bool [n, n]) (act : ℝ → ℝ) (σ : Op [.value [n, f]] [.value [n, f]])
    (hσ : (σ.rule? (α := ℝ)).isSome)
    (hact : ∀ y t c, get2 ((σ.rule?.get hσ).forward (y, ())).1 t c = act (get2 y t c))
    (schedule : Softmax.Schedule) (x : Tensor ℝ [n, d]) (γ1 β1 γ2 β2 bo b2 : Tensor ℝ [d])
    (ε τ : Tensor ℝ []) (Wq Wk Wv : Tensor ℝ [h * e, d]) (bq bk bv : Tensor ℝ [h * e])
    (Wo : Tensor ℝ [d, h * e]) (W1 : Tensor ℝ [f, d]) (b1 : Tensor ℝ [f])
    (W2 : Tensor ℝ [d, f]) (t : Fin n) (i : Fin d) :
    let norm : Tensor ℝ [n, d] → Tensor ℝ [d] → Tensor ℝ [d] → Tensor ℝ [n, d] := fun y γ β =>
      (((Op.layerNorm hn hd hlead hx).rule? (α := ℝ) |>.get rfl).forward (y, γ, β, ε, ())).1
    let split : Tensor ℝ [h * e, d] → Tensor ℝ [h * e] → Tensor ℝ [h, n, e] :=
      fun W b =>
        (((Op.permute (Shape.adjacentSwap [n, h, e] 0) [h, n, e] hto).rule?
          (α := ℝ) |>.get rfl).forward ((((Op.reshape hsplit).rule? (α := ℝ) |>.get rfl).forward
            ((((Op.linear (lead := [n]) (inDim := d) (outDim := h * e)).rule? (α := ℝ)
              |>.get rfl).forward (W, b, norm x γ1 β1, ())).1, ())).1, ())).1
    let o : Tensor ℝ [h, n, e] :=
      (((Op.attention (batch := [h]) (rows := n) (keys := n) (depth := e) (width := e)
        (some mask) schedule).rule? (α := ℝ) |>.get rfl).forward
          (split Wq bq, split Wk bk, split Wv bv, τ, ())).1
    let m : Tensor ℝ [n, h * e] :=
      (((Op.reshape hmerge).rule? (α := ℝ) |>.get rfl).forward
        ((((Op.permute (Shape.adjacentSwap [h, n, e] 0) [n, h, e] hfrom).rule?
          (α := ℝ) |>.get rfl).forward (o, ())).1, ())).1
    let s : Tensor ℝ [n, d] := (((Op.add (s := [n, d])).rule? (α := ℝ) |>.get rfl).forward
      (x, (((Op.linear (lead := [n]) (inDim := h * e) (outDim := d)).rule? (α := ℝ)
        |>.get rfl).forward (Wo, bo, m, ())).1, ())).1
    let g : Tensor ℝ [n, f] := ((σ.rule?.get hσ).forward
      ((((Op.linear (lead := [n]) (inDim := d) (outDim := f)).rule? (α := ℝ) |>.get rfl).forward
        (W1, b1, norm s γ2 β2, ())).1, ())).1
    block act
        { γ1 := Tensor.getScalar γ1, β1 := Tensor.getScalar β1
          wq := fun j p => get2 Wq (finProdFinEquiv p) j
          bq := fun p => Tensor.getScalar bq (finProdFinEquiv p)
          wk := fun j p => get2 Wk (finProdFinEquiv p) j
          bk := fun p => Tensor.getScalar bk (finProdFinEquiv p)
          wv := fun j p => get2 Wv (finProdFinEquiv p) j
          bv := fun p => Tensor.getScalar bv (finProdFinEquiv p)
          wo := fun p j => get2 Wo j (finProdFinEquiv p), bo := Tensor.getScalar bo
          γ2 := Tensor.getScalar γ2, β2 := Tensor.getScalar β2
          w1 := fun j c => get2 W1 c j, b1 := Tensor.getScalar b1
          w2 := fun c j => get2 W2 j c, b2 := Tensor.getScalar b2 }
        ε.item τ.item (fun t => Finset.univ.filter fun j => get2 mask t j = true)
        (fun t j => get2 x t j) t i =
      get2 (((Op.add (s := [n, d])).rule? (α := ℝ) |>.get rfl).forward
        (s, (((Op.linear (lead := [n]) (inDim := f) (outDim := d)).rule? (α := ℝ)
          |>.get rfl).forward (W2, b2, g, ())).1, ())).1 t i := by
  intro norm split o m s g
  have hA : ∀ (y : Tensor ℝ [n, d]) (γ β : Tensor ℝ [d]) t j, get2 (norm y γ β) t j =
      layerNorm (Tensor.getScalar γ) (Tensor.getScalar β) ε.item (fun i => get2 y t i) j :=
    fun y γ β t j => get2_layerNorm_forward hn hd hlead hx y γ β ε t j
  have hQ : ∀ (W : Tensor ℝ [h * e, d]) (b : Tensor ℝ [h * e]) (z : Fin h) (t : Fin n)
      (c : Fin e), split W b (z, t, c, ()) = Tensor.getScalar b (finProdFinEquiv (z, c)) +
        ∑ j, get2 (norm x γ1 β1) t j * get2 W (finProdFinEquiv (z, c)) j :=
    fun W b z t c => (split_forward_apply _ hsplit hto z t c).trans (get2_linear_forward W b _ t _)
  have ho : ∀ z t c, o (z, t, c, ()) =
      attention τ.item (Finset.univ.filter fun j => get2 mask t j = true)
        (fun i => split Wq bq (z, t, i, ())) (fun j i => split Wk bk (z, j, i, ()))
        (fun j c => split Wv bv (z, j, c, ())) c :=
    fun z t c => attention_forward_apply_heads_mask mask schedule _ _ _ τ z t c
  have hS : ∀ t i, get2 s t i = get2 x t i + (Tensor.getScalar bo i +
      ∑ zc : Fin h × Fin e, o (zc.1, t, zc.2, ()) * get2 Wo i (finProdFinEquiv zc)) :=
    fun t i => (get2_add_forward x _ t i).trans (congrArg (get2 x t i + ·)
      ((get2_linear_forward Wo bo m t i).trans (congrArg (Tensor.getScalar bo i + ·)
        (sum_merge_forward o hfrom hmerge t fun p => get2 Wo i p))))
  have hG : ∀ t c, get2 g t c =
      act (Tensor.getScalar b1 c + ∑ j, get2 (norm s γ2 β2) t j * get2 W1 c j) :=
    fun t c => (hact _ t c).trans (congrArg act (get2_linear_forward W1 b1 _ t c))
  refine Eq.trans ?_ ((get2_add_forward s _ t i).trans
    (congrArg (get2 s t i + ·) (get2_linear_forward W2 b2 g t i))).symm
  simp only [block, heads, hS, hG, hA, ho, hQ]

end TorchLean.TransformerBlock
