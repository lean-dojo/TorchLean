/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Window.PoolLeading
public import TorchLean.Tensor.Layers.Conv
public import TorchLean.Tensor.Layers.Pooling

/-!
# Channels-first convolution and spatial pooling as general window operators

`contract`, the dense contraction of `forward`, is `TorchLean.Tensor.conv` of the input with a
batch axis of one in PyTorch's layout, with symmetric padding and no dilation, read at the output
coordinate with the same indices after the batch index (`contract_eq_conv`). Both read the same
window cells in the same order, with the same zero times the kernel entry at a padding hole, one
through runtime index lists and the other through typed coordinates.
`groupedContract_eq_conv` proves the same for the grouped, dilated contraction with asymmetric
padding of `grouped`, with `groups` feature groups and the weights in PyTorch's packed
layout `(outC, inC / groups, k...)` (`packedKernel`). Over a batch axis of any extent,
`TorchLean.Tensor.conv` reads only slice `b` of the input at an output coordinate with batch index
`b` (`TorchLean.Tensor.conv_channelsFirst_unstack`). So `grouped` at every index of a batch
axis is `TorchLean.Tensor.conv` over that axis plus the bias broadcast along the output channels,
which is the forward value of `Op.conv` (`mapLeading_grouped_eq_conv`).

`TorchLean.Tensor.pool (.avg true)` over every axis of a spatial tensor with unit dilation,
symmetric padding and floor mode divides by the number of kernel cells, since every window cell
lies in the padded input (`poolCount_true_finRange`). `pool_finRange_spatial` gives the output
shape of such a pool as `Pool.outSpatial`, and `pool_tail_channelsFirst` gives it for the spatial
axes of a channels-first tensor, the channel axis kept.

Each gradient of `forward` is its general convolution counterpart in the same layout, again with
a batch axis of one. `inputDeriv` is `TorchLean.Tensor.convBackwardInput` of the kernel and
the cotangent (`inputDeriv_eq_convBackwardInput`), `kernelDeriv` is
`TorchLean.Tensor.convBackwardWeight` of the input and the cotangent
(`kernelDeriv_eq_convBackwardWeight`), and `biasDeriv` sums the cotangent with
`TorchLean.Tensor.reduce .sum` over every axis but the output channels
(`biasDeriv_eq_reduce`). `conv_channelsFirst` is the shape equation behind the casts of the
cotangent, and `conv_channelsFirst_dilated` the one behind the cast of the forward, both for a
batch axis of any extent.
-/

@[expose] public section

open TorchLean.Conv TorchLean.Tensor TorchLean.Shape TorchLean.Pool TorchLean.ConvTranspose
open TorchLean.Shape (rank rank_cons rank_eq_length)

namespace TorchLean.Shape

/-- On a list of axes mapped by an injective function, the value at an image axis is the value at
the axis. -/
theorem valueAt_map {r q : Nat} (f : Fin r → Fin q) (hf : Function.Injective f)
    (dims : List (Fin r)) (values : List Nat) (default : Nat) (a : Fin r) :
    valueAt (dims.map f) values default (f a) = valueAt dims values default a := by
  induction dims generalizing values with
  | nil => rfl
  | cons d dims ih =>
      cases values with
      | nil => rfl
      | cons v values =>
          simp only [List.map_cons, valueAt, hf.eq_iff]
          split_ifs <;> simp [ih]

/-- Listing every axis, the value at axis `a` is entry `a` of the values. -/
theorem valueAt_finRange {n : Nat} (values : List Nat) (h : values.length = n) (default : Nat)
    (a : Fin n) : valueAt (List.finRange n) values default a = values[a.val]'(h ▸ a.isLt) := by
  induction n generalizing values with
  | zero => exact a.elim0
  | succ n ih =>
      cases values with
      | nil => simp at h
      | cons v values =>
          rw [List.finRange_succ]
          cases a using Fin.cases with
          | zero => simp [valueAt]
          | succ b =>
              simp only [valueAt, Fin.succ_ne_zero, ite_false]
              rw [valueAt_map Fin.succ (Fin.succ_injective n)]
              simpa using ih values (by simpa using h) b

/-- A dilated extent with unit dilation is the extent. -/
theorem dilatedKernelExtent_one (n : Nat) : Shape.dilatedKernelExtent n 1 = n := by
  unfold Shape.dilatedKernelExtent
  split_ifs with h
  · omega
  · omega

end TorchLean.Shape

namespace TorchLean

/-- Entry `i` of the row-major list of a vector is its scalar at `i`. -/
theorem Tensor.getElem_data_toList {α : Type} [Storage α] {d : Nat} (t : Tensor α [d]) (i : Fin d) :
    t.data.toList[i.val]'(by simp [Rep.data_size, Shape.size]) =
      t.getScalar i := by
  rw [Array.getElem_toList]
  have hj : i.val < Shape.size [d] := by simp [Shape.size]
  rw [Rep.data_getFlat t ⟨i.val, hj⟩, Rep.getFlat_eq_get, getScalar_eq_apply]
  congr 1
  apply Prod.ext
  · apply Fin.ext
    simp [Coord.unlinearizeFast_eq, Coord.unlinearizeFast]
  · rfl

/-- Entry `a` of the row-major array of a vector is its scalar at `a`. -/
theorem Tensor.getElem_data {α : Type} [Storage α] {d : Nat} (t : Tensor α [d]) (a : Nat)
    (h : a < t.data.size) :
    t.data[a] =
      t.getScalar ⟨a, by simpa [Rep.data_size, Shape.size] using h⟩ := by
  rw [← Array.getElem_toList]
  exact getElem_data_toList t ⟨a, _⟩

/-- The row-major list of a vector has its length. -/
@[simp] theorem Tensor.length_data_toList {α : Type} [Storage α] {d : Nat} (t : Tensor α [d]) :
    t.data.toList.length = d := by
  simp [Rep.data_size, Shape.size]

/-- Pooling every axis of a spatial tensor with unit dilation, symmetric padding and floor mode has
the spatial output shape of `Pool.outSpatial`. -/
theorem Shape.pool_finRange_spatial {d : Nat} (inSpatial kernel stride padding : Tensor Nat [d]) :
    Shape.pool inSpatial.data.toList (List.finRange (Shape.rank inSpatial.data.toList))
        kernel.data.toList stride.data.toList padding.data.toList padding.data.toList
        (List.replicate d 1) false =
      (Pool.outSpatial inSpatial kernel stride padding).data.toList := by
  have hrank : Shape.rank inSpatial.data.toList = d := by
    simp [Shape.rank_eq_length]
  apply List.ext_getElem
  · simp [Shape.pool, Shape.rank_eq_length]
  · intro a h₁ h₂
    have ha : a < d := by simpa using h₂
    simp only [Shape.pool, List.getElem_ofFn]
    repeat rw [Shape.valueAt_finRange _ (by simp [Shape.rank_eq_length])]
    simp only [List.getElem_replicate, Shape.extent]
    rw [Tensor.getElem_data_toList (Pool.outSpatial inSpatial kernel stride padding)
      ⟨a, ha⟩]
    simp [Pool.outSpatial, Tensor.getElem_data]

/-- Convolving an input with a batch axis of `n` in PyTorch's layout, with any kernel input-feature
extent, gives the batch axis, the output channels and the spatial shape of
`outSpatialDilated`. -/
theorem Shape.conv_channelsFirst_dilated {n d inC outC I : Nat}
    (inSpatial kernel stride dilation paddingBefore paddingAfter : Tensor Nat [d]) :
    Shape.conv (n :: inC :: inSpatial.data.toList) (outC :: I :: kernel.data.toList)
        (Shape.ConvDims.channelsFirst (by simp))
        (fun j => stride.getScalar (Fin.cast (by simp) j))
        (fun j => paddingBefore.getScalar (Fin.cast (by simp) j))
        (fun j => paddingAfter.getScalar (Fin.cast (by simp) j)) (fun _ => 1)
        (fun j => dilation.getScalar (Fin.cast (by simp) j)) 1 =
      n :: outC :: (outSpatialDilated inSpatial kernel stride dilation paddingBefore
        paddingAfter).data.toList := by
  apply List.ext_getElem
  · simp [Shape.conv, Shape.rank_eq_length]
  · intro j h₁ h₂
    simp only [Shape.conv, List.getElem_ofFn, Shape.ConvDims.channelsFirst, finCongr_symm,
      finCongr_apply_mk]
    match j with
    | 0 =>
        simp only [Shape.convExtent, Shape.extent, Nat.div_one]
        rfl
    | 1 =>
        simp only [Shape.convExtent, Shape.extent]
        rfl
    | j + 2 =>
        have hj : j < d := by simp at h₂; omega
        simp only [Shape.convExtent, Shape.extent, List.getElem_cons_succ, Fin.cast_mk,
          Shape.dilatedKernelExtent_one]
        rw [Tensor.getElem_data_toList (outSpatialDilated inSpatial kernel stride
          dilation paddingBefore paddingAfter) ⟨j, hj⟩]
        simp only [outSpatialDilated, Tensor.getScalar_ofFn]
        congr 1
        · exact Tensor.getElem_data_toList inSpatial ⟨j, hj⟩
        · exact Tensor.getElem_data_toList kernel ⟨j, hj⟩

/-- Convolving an input with a batch axis of `n` in PyTorch's layout, with unit dilation and equal
padding on both sides, gives the batch axis, the output channels, and the spatial shape of
`outSpatialDilated`. -/
theorem Shape.conv_channelsFirst {n d inC outC : Nat}
    (inSpatial kernel stride padding : Tensor Nat [d]) :
    Shape.conv (n :: inC :: inSpatial.data.toList) (outC :: inC :: kernel.data.toList)
        (Shape.ConvDims.channelsFirst (by simp))
        (fun j => stride.getScalar (Fin.cast (by simp) j))
        (fun j => padding.getScalar (Fin.cast (by simp) j))
        (fun j => padding.getScalar (Fin.cast (by simp) j)) (fun _ => 1) (fun _ => 1) 1 =
      n :: outC :: (outSpatialDilated inSpatial kernel stride
          (Tensor.ones [d]) padding padding).data.toList := by
  have h := conv_channelsFirst_dilated (n := n) (inC := inC) (outC := outC) (I := inC) inSpatial
    kernel stride (Tensor.ones [d]) padding padding
  simpa only [Tensor.getScalar_full] using h

/-- A transposed convolution in PyTorch's layout, with the kernel's two feature axes exchanged,
symmetric padding, no output padding and no dilation, keeps the batch axis `n` of its input. Its
other axes are the output channels and the spatial shape of `ConvTranspose.outSpatial`. -/
theorem Shape.convTranspose_channelsFirst {n d inC outC : Nat}
    (inSpatial kernel stride padding : Tensor Nat [d]) :
    Shape.convTranspose (n :: inC :: inSpatial.data.toList) (inC :: outC :: kernel.data.toList)
        (Shape.ConvDims.channelsFirst (by simp)).swapKernel
        (fun j => stride.getScalar (Fin.cast (by simp) j))
        (fun j => padding.getScalar (Fin.cast (by simp) j))
        (fun j => padding.getScalar (Fin.cast (by simp) j)) (fun _ => 0) (fun _ => 1) 1 =
      n :: outC :: (ConvTranspose.outSpatial inSpatial kernel stride padding).data.toList := by
  apply List.ext_getElem
  · simp [Shape.convTranspose, Shape.rank_eq_length]
  · intro j h₁ h₂
    simp only [Shape.convTranspose, List.getElem_ofFn, Shape.ConvDims.swapKernel,
      Shape.ConvDims.channelsFirst, finCongr_symm, finCongr_apply_mk]
    match j with
    | 0 =>
        simp only [Shape.convTransposeExtent, Shape.extent]
        rfl
    | 1 =>
        simp only [Shape.convTransposeExtent, Shape.extent, Nat.one_mul]
        rfl
    | j + 2 =>
        have hj : j < d := by simp at h₂; omega
        simp only [Shape.convTransposeExtent, Shape.extent, List.getElem_cons_succ, Fin.cast_mk]
        rw [Tensor.getElem_data_toList (ConvTranspose.outSpatial inSpatial kernel stride
          padding) ⟨j, hj⟩]
        simp only [ConvTranspose.outSpatial, Tensor.getScalar_ofFn]
        congr 1
        · exact Tensor.getElem_data_toList inSpatial ⟨j, hj⟩
        · rw [Equiv.trans_apply, Equiv.swap_apply_of_ne_of_ne
            (Fin.ne_of_val_ne (show j + 2 ≠ 0 by omega))
            (Fin.ne_of_val_ne (show j + 2 ≠ 1 by omega))]
          exact Tensor.getElem_data_toList kernel ⟨j, hj⟩

namespace Conv

/-- The window positions `out * stride + window` of `mkInputIdx?`, axis by axis. -/
private def windowPositions (o w s : List Nat) : List Nat :=
  List.zipWith (· + ·) (List.zipWith (· * ·) o s) w

/-- `mkInputIdx?` finds an input index list exactly when no window position lies in the low
padding, and the list is then the positions less the padding. -/
private theorem mkInputIdx?_eq {o w s p : List Nat} (hw : w.length = o.length)
    (hs : s.length = o.length) (hp : p.length = o.length) :
    mkInputIdx? o w s p =
      if (List.zipWith (fun a b => decide (a ≤ b)) p (windowPositions o w s)).all id then
        some (List.zipWith (· - ·) (windowPositions o w s) p)
      else none := by
  induction o generalizing w s p with
  | nil =>
      cases w <;> cases s <;> cases p <;> simp_all [mkInputIdx?, windowPositions]
  | cons oo os ih =>
      cases w with
      | nil => simp at hw
      | cons ww ws =>
          cases s with
          | nil => simp at hs
          | cons ss sss =>
              cases p with
              | nil => simp at hp
              | cons pp ps =>
                  simp only [List.length_cons, Nat.add_right_cancel_iff] at hw hs hp
                  simp only [mkInputIdx?, windowPositions, List.zipWith_cons_cons, List.all_cons]
                  rw [ih hw hs hp]
                  simp only [windowPositions]
                  by_cases hq : oo * ss + ww < pp
                  · simp [hq, Nat.not_le.mpr hq]
                  · simp only [hq, dite_false, Nat.not_lt.mp hq, decide_true, id, Bool.true_and]
                    by_cases hall : (List.zipWith (fun a b => decide (a ≤ b)) ps
                        (List.zipWith (· + ·) (List.zipWith (· * ·) os sss) ws)).all id = true
                    · simp [hall]
                    · simp [hall]

/-- The dilated index map of the grouped contraction is the dense one at the dilated kernel
offsets. -/
private theorem mkDilatedInputIdx?_eq {o k s dil p : List Nat} (hd : dil.length = k.length) :
    mkDilatedInputIdx? o k s dil p = mkInputIdx? o (List.zipWith (· * ·) k dil) s p := by
  induction o generalizing k s dil p with
  | nil =>
      cases k <;> cases dil <;> cases s <;> cases p <;> simp_all [mkDilatedInputIdx?, mkInputIdx?]
  | cons oo os ih =>
      cases k with
      | nil => cases dil <;> simp_all [mkDilatedInputIdx?, mkInputIdx?]
      | cons kk ks =>
          cases dil with
          | nil => simp at hd
          | cons dd ds =>
              simp only [List.length_cons, Nat.add_right_cancel_iff] at hd
              cases s <;> cases p <;> simp [mkDilatedInputIdx?, mkInputIdx?, ih hd]

/-- `mkTransposeInputIdx?` lands an input position on an output position exactly when every axis
has a positive stride, the window cell is at most the position plus the padding, and the difference
is a multiple of the stride. In that case the output index is the difference divided by the
stride. -/
private theorem mkTransposeInputIdx?_eq {o k s p : List Nat} (hk : k.length = o.length)
    (hs : s.length = o.length) (hp : p.length = o.length) :
    mkTransposeInputIdx? o k s p =
      if (s.all fun a => decide (a ≠ 0)) &&
          (List.zipWith (fun a b => decide (a ≤ b)) k (List.zipWith (· + ·) o p)).all id &&
          (List.zipWith (fun r t => decide (r % t = 0))
            (List.zipWith (· - ·) (List.zipWith (· + ·) o p) k) s).all id then
        some (List.zipWith (· / ·) (List.zipWith (· - ·) (List.zipWith (· + ·) o p) k) s)
      else none := by
  induction o generalizing k s p with
  | nil => cases k <;> cases s <;> cases p <;> simp_all [mkTransposeInputIdx?]
  | cons oo os ih =>
      cases k with
      | nil => simp at hk
      | cons kk ks =>
          cases s with
          | nil => simp at hs
          | cons ss sss =>
              cases p with
              | nil => simp at hp
              | cons pp ps =>
                  simp only [List.length_cons, Nat.add_right_cancel_iff] at hk hs hp
                  simp only [mkTransposeInputIdx?, List.zipWith_cons_cons, List.all_cons]
                  rw [ih hk hs hp]
                  by_cases h0 : ss = 0
                  · simp [h0]
                  · by_cases hq : oo + pp < kk
                    · simp [h0, hq, Nat.not_le.mpr hq]
                    · by_cases hr : (oo + pp - kk) % ss = 0
                      · simp only [h0, hq, hr, ite_false, dite_false, dite_true, decide_false,
                          decide_true, Bool.not_false, Bool.true_and, Nat.not_lt.mp hq, id,
                          decide_not]
                        split_ifs with hall <;> simp_all
                      · simp [h0, hq, hr, Nat.not_lt.mp hq]

end Conv

section Instances

variable {α : Type} [Storage α] [Context α]

/-- Two folds over the coordinates of equal shapes keep a relation that their steps keep at
coordinates with the same indices. -/
private theorem Shape.foldl_finRange_rel {β γ : Type} {t u : Shape} (h : t = u) (R : β → γ → Prop)
    (f : β → Fin (Shape.size t) → β)
    (g : γ → Fin (Shape.size u) → γ) (b : β) (b' : γ) (hb : R b b')
    (hfg : ∀ a a' (i : Fin (Shape.size t))
      (i' : Fin (Shape.size u)),
      (∀ j, Shape.Coord.index (Coord.unlinearize i) j =
        Shape.Coord.index (Coord.unlinearize i') j) →
      R a a' → R (f a i) (g a' i')) :
    R ((List.finRange (Shape.size t)).foldl f b)
      ((List.finRange (Shape.size u)).foldl g b') := by
  subst h
  exact List.foldl_rel hb fun i _ a a' hR => hfg a a' i i (fun _ => rfl) hR

/-- A window cell of a floor-mode pooling with unit dilation and symmetric padding lies in the
padded input. -/
private theorem Shape.position_lt_of_poolExtent {n kernel stride padding out cell : Nat}
    (hout : out < Shape.poolExtent n kernel stride padding padding 1 false)
    (hcell : cell < kernel) :
    out * stride + cell * 1 < n + padding + padding := by
  have hext : Shape.poolExtent n kernel stride padding padding 1 false =
      if (n = 0 || kernel = 0 || stride = 0 || padding > kernel / 2 || padding > kernel / 2) then 0
      else Shape.slidingWindowOutDimDilated n kernel stride 1 padding padding := rfl
  rw [hext] at hout
  split_ifs at hout with h₁
  · omega
  · simp only [Bool.or_eq_true, decide_eq_true_eq, not_or] at h₁
    unfold Shape.slidingWindowOutDimDilated Shape.dilatedKernelExtent at hout
    have hk : kernel ≠ 0 := h₁.1.1.1.2
    simp only [hk, ite_false, one_mul, Nat.sub_add_cancel (Nat.one_le_iff_ne_zero.mpr hk)] at hout
    split_ifs at hout with h₃
    · omega
    · simp only [Bool.or_eq_true, decide_eq_true_eq, not_or] at h₃
      have hdiv := Nat.div_mul_le_self (n + padding + padding - kernel) stride
      have hle : out * stride ≤ (n + padding + padding - kernel) / stride * stride :=
        Nat.mul_le_mul_right _ (by omega)
      omega


/-- Every window cell of a floor-mode pool over every axis of a spatial tensor, with unit dilation
and symmetric padding, lies in the padded input, so `poolCount true` counts the whole window. -/
theorem Tensor.poolCount_true_finRange {d : Nat}
    {inSpatial kernel stride padding : Tensor Nat [d]}
    (c : (Shape.pool inSpatial.data.toList (List.finRange (Shape.rank inSpatial.data.toList))
      kernel.data.toList stride.data.toList padding.data.toList padding.data.toList
      (List.replicate d 1) false).Coord) :
    Tensor.poolCount true c =
      Shape.size (Shape.poolWindow inSpatial.data.toList
        (List.finRange (Shape.rank inSpatial.data.toList)) kernel.data.toList) := by
  unfold Tensor.poolCount
  simp only [ite_true]
  rw [(List.countP_eq_length).mpr, List.length_finRange]
  intro i _
  rw [decide_eq_true_eq]
  intro a
  have hrank : Shape.rank inSpatial.data.toList = d := by simp [Shape.rank_eq_length]
  have hrank' : Shape.rank inSpatial.data.toList = d := hrank
  have ha : a.val < d := by have := a.isLt; omega
  have hvalue : ∀ (l : List Nat) (hl : l.length = d) (default : Nat),
      Shape.valueAt (List.finRange (Shape.rank inSpatial.data.toList)) l default a =
        l[a.val]'(by omega) :=
    fun l hl default => Shape.valueAt_finRange l (by rw [hl, hrank]) default a
  have hout := Shape.Coord.index_lt_ofFn _ c a
  have hcell := Shape.Coord.index_lt_ofFn _
    (Coord.unlinearize i :
      Shape.Coord (Shape.poolWindow inSpatial.data.toList
        (List.finRange (Shape.rank inSpatial.data.toList)) kernel.data.toList)) a
  rw [hvalue (List.replicate d 1) (by simp)] at hout
  simp only [List.getElem_replicate] at hout
  have := position_lt_of_poolExtent hout hcell
  rw [Shape.poolPosition, hvalue (List.replicate d 1) (by simp)]
  simp only [List.getElem_replicate] at this ⊢
  exact this


/-! ## Convolution -/

/-- A runtime index list outside the shape reads zero. -/
private theorem Tensor.getAtOrZero_eq_zero_of_not_lt {shape : Shape} (x : Tensor α shape)
    (l : List Nat) (j : Nat) (hj : j < l.length) (hjs : j < shape.length)
    (hout : shape[j] ≤ l[j]) : getAtOrZero x l = 0 := by
  induction shape generalizing l j with
  | scalar => simp at hjs
  | dim n rest ih =>
      cases l with
      | nil => simp at hj
      | cons i is =>
          rw [getAtOrZero_dim_cons]
          split_ifs with hi
          · cases j with
            | zero => simp at hout; omega
            | succ j =>
                exact ih _ is j (by simpa using hj) (by simpa using hjs) (by simpa using hout)
          · rfl

/-- A runtime index list that lists a coordinate reads the entry at it. -/
private theorem Tensor.getAtOrZero_eq_of_toList {shape : Shape} (x : Tensor α shape)
    (C : shape.Coord) {l : List Nat} (h : Shape.Coord.toList shape C = l) :
    getAtOrZero x l = x C := by
  subst h
  exact getAtOrZero_coord_toList x C rfl

/-- A channels-first convolution has the input channels, then the kernel, as its window. -/
private theorem Shape.convWindow_channelsFirst {n c o i : Nat} {spatial kernel : List Nat}
    (h : kernel.length = spatial.length) :
    Shape.convWindow (o :: i :: kernel) (Shape.ConvDims.channelsFirst (n := n) (c := c) h) =
      i :: kernel := by
  apply List.ext_getElem
  · simp [Shape.convWindow, h]
  · intro j h₁ h₂
    simp only [Shape.convWindow, List.getElem_ofFn, Shape.ConvDims.channelsFirst, finCongr_apply,
      Shape.extent]
    rfl

/-- A shape `s` broadcasts to `n :: s`, its axis `i` going to axis `i + 1`. -/
theorem _root_.TorchLean.Shape.BroadcastDims.succ (n : Nat) (s : Shape) :
    Shape.BroadcastDims s (n :: s) Fin.succ where
  strictMono := Fin.strictMono_succ
  extent_eq _ := Or.inr rfl

/-- A fold over the coordinates of a shape is the fold over its flat indices. -/
private theorem Shape.foldl_coordinates_eq_finRange {β : Type} (s : Shape) (f : β → s.Coord → β)
    (b : β) :
    (Tensor.Rep.coordinates s).foldl f b =
      (List.finRange (Shape.size s)).foldl
        (fun acc i => f acc (Coord.unlinearize i)) b := by
  rw [Tensor.Rep.foldl_eq_fin_foldl, Fin.foldl_eq_finRange_foldl]
  simp [Tensor.Rep.coordinates, Tensor.Rep.ofFn]

/-- `contract`, the dense contraction of `forward`, is `TorchLean.Tensor.conv` of the input
with a batch axis of one, in PyTorch's layout, with symmetric padding and no dilation. Both sides
are read at output coordinates that agree after the batch index. -/
theorem Conv.contract_eq_conv {d inC outC : Nat} {kernel stride padding inSpatial : Tensor Nat [d]}
    (weights : Tensor α (outC :: inC :: kernel.data.toList))
    (input : Tensor α (inC :: inSpatial.data.toList))
    (c : Shape.Coord (outC :: (outSpatialDilated inSpatial kernel stride
          (Tensor.ones [d]) padding padding).data.toList))
    (c' : (Shape.conv (1 :: inC :: inSpatial.data.toList) (outC :: inC :: kernel.data.toList)
      (Shape.ConvDims.channelsFirst (by simp))
      (fun j => stride.getScalar (Fin.cast (by simp) j))
      (fun j => padding.getScalar (Fin.cast (by simp) j))
      (fun j => padding.getScalar (Fin.cast (by simp) j)) (fun _ => 1) (fun _ => 1) 1).Coord)
    (hc : ∀ j, Shape.Coord.index c' (j + 1) = Shape.Coord.index c j) :
    contract (stride := stride) (padding := padding) weights input c =
      Tensor.conv
        (Tensor.broadcast input (1 :: inC :: inSpatial.data.toList) Fin.succ
          (Shape.BroadcastDims.succ 1 _))
        weights (Shape.ConvDims.channelsFirst (by simp))
        (fun j => stride.getScalar (Fin.cast (by simp) j))
        (fun j => padding.getScalar (Fin.cast (by simp) j))
        (fun j => padding.getScalar (Fin.cast (by simp) j)) (fun _ => 1) (fun _ => 1) 1 1 c' := by
  obtain ⟨oc, rest⟩ := c
  rw [Tensor.conv_apply]
  simp only [contract, contractWith, Tensor.generate,
    Tensor.dim_apply, Tensor.Rep.get_ofFn]
  generalize ho : Shape.Coord.toList (outSpatialDilated inSpatial kernel stride
      (Tensor.ones [d]) padding padding).data.toList
    rest = outIdx
  have hflat : ∀ (F : α → Shape.Coord (inC :: kernel.data.toList) → α),
      (List.finRange inC).foldl (fun acc lc =>
          (Tensor.Rep.coordinates kernel.data.toList).foldl
            (fun acc k => F acc (lc, k)) acc) 0 =
        (List.finRange (Shape.size (inC :: kernel.data.toList))).foldl
          (fun acc i => F acc (Coord.unlinearize i)) 0 := by
    intro F
    rw [← foldl_coordinates_eq_finRange, Tensor.Rep.foldl_coordinates_cons]
  simp only [foldlIndices_eq_foldl]
  refine Eq.trans (hflat (fun acc ck => acc +
    (match mkInputIdx? outIdx (Shape.Coord.toList kernel.data.toList ck.2) stride.data.toList
        padding.data.toList with
      | none => 0
      | some inputIndex => getAtOrZero input (ck.1.val :: inputIndex)) *
      getAtOrZero weights (oc.val :: ck.1.val :: Shape.Coord.toList kernel.data.toList ck.2))) ?_
  refine foldl_finRange_rel (convWindow_channelsFirst _).symm Eq _ _ 0 0 rfl ?_
  intro acc acc' i i' hidx hacc
  subst hacc
  generalize hk : (Coord.unlinearize i :
    Shape.Coord (inC :: kernel.data.toList)) = k at hidx ⊢
  generalize hkw : Coord.unlinearize i' = kw at hidx ⊢
  congr 2
  · have hoLen : outIdx.length = d := by
      rw [← ho, Shape.Coord.length_toList]
      simp
    have hwLen : (Shape.Coord.toList kernel.data.toList k.2).length = d := by
      rw [Shape.Coord.length_toList]
      simp
    have hsLen : stride.data.toList.length = d := by simp
    have hpLen : padding.data.toList.length = d := by simp
    have hoAt : ∀ j (hj : j < d), outIdx[j]'(by omega) = Shape.Coord.index c' (j + 2) := by
      intro j hj
      subst ho
      rw [Shape.Coord.getElem_toList]
      exact (hc (j + 1)).symm
    have hwAt : ∀ j (hj : j < d), (Shape.Coord.toList kernel.data.toList k.2)[j]'(by omega) =
        Shape.Coord.index kw (j + 1) := by
      intro j hj
      rw [Shape.Coord.getElem_toList]
      exact hidx (j + 1)
    have hc1 : Shape.Coord.index c' 1 = oc.val := by
      have := hc 0
      simpa using this
    have hspLen : inSpatial.data.toList.length = d := by simp
    -- The list entries of the window positions and the padding at axis `j`.
    have hq : ∀ j (hj : j < d), (windowPositions outIdx (Shape.Coord.toList kernel.data.toList k.2)
        stride.data.toList)[j]'(by simp [windowPositions, hoLen, hwLen, hsLen]; omega) =
          Shape.Coord.index c' (j + 2) * stride.getScalar ⟨j, hj⟩ +
            Shape.Coord.index kw (j + 1) := by
      intro j hj
      simp only [windowPositions, List.getElem_zipWith]
      rw [hoAt j hj, hwAt j hj, Tensor.getElem_data_toList stride ⟨j, hj⟩]
    have hpad : ∀ j (hj : j < d), padding.data.toList[j]'(by omega) = padding.getScalar ⟨j, hj⟩ :=
      fun j hj => Tensor.getElem_data_toList padding ⟨j, hj⟩
    have hsp : ∀ j (hj : j < d), inSpatial.data.toList[j]'(by omega) =
        inSpatial.getScalar ⟨j, hj⟩ := fun j hj =>
      Tensor.getElem_data_toList inSpatial ⟨j, hj⟩
    unfold Tensor.convInput
    rw [mkInputIdx?_eq (by rw [hwLen, hoLen]) (by rw [hsLen, hoLen]) (by rw [hpLen, hoLen])]
    have hlenL : (windowPositions outIdx (Shape.Coord.toList kernel.data.toList k.2)
        stride.data.toList).length = d := by
      simp [windowPositions, hoLen, hwLen, hsLen]
    have hlow : (List.zipWith (fun a b => decide (a ≤ b)) padding.data.toList
        (windowPositions outIdx (Shape.Coord.toList kernel.data.toList k.2)
          stride.data.toList)).all id = true ↔
        ∀ j (hj : j < d), padding.getScalar ⟨j, hj⟩ ≤
          Shape.Coord.index c' (j + 2) * stride.getScalar ⟨j, hj⟩ +
            Shape.Coord.index kw (j + 1) := by
      rw [List.all_eq_true]
      constructor
      · intro h j hj
        have hj' : j < (List.zipWith (fun a b => decide (a ≤ b)) padding.data.toList
            (windowPositions outIdx (Shape.Coord.toList kernel.data.toList k.2)
              stride.data.toList)).length := by simp [hlenL, hpLen, hj]
        have := h _ (List.getElem_mem hj')
        simpa [List.getElem_zipWith, hq j hj, hpad j hj] using this
      · intro h x hx
        obtain ⟨j, hj, rfl⟩ := List.getElem_of_mem hx
        have hj' : j < d := by simpa [hlenL, hpLen] using hj
        simpa [List.getElem_zipWith, hq j hj', hpad j hj'] using h j hj'
    have hc0 : Shape.Coord.index c' 0 = 0 := by
      have := Shape.index_conv_lt (Shape.ConvDims.channelsFirst (n := 1) (c := inC) (o := outC)
        (i := inC) (spatial := inSpatial.data.toList) (kernel := kernel.data.toList) (by simp)) c'
        ⟨0, by omega⟩
      simp only [Shape.convExtent, Shape.ConvDims.channelsFirst, finCongr_apply_mk, Shape.extent,
        List.getElem_cons_zero, Nat.div_one] at this
      exact Nat.lt_one_iff.mp this
    have hoc : oc.val / (outC / 1) = 0 := by
      rw [Nat.div_one]
      exact Nat.div_eq_of_lt oc.isLt
    -- The condition of `convInput` along each position.
    have hcond : ∀ p : Fin (inSpatial.data.toList.length + 2),
        (Shape.ConvDims.channelsFirst (n := 1) (c := inC) (o := outC) (i := inC)
          (spatial := inSpatial.data.toList) (kernel := kernel.data.toList) (by simp)).inputAligned
          (fun j => stride.getScalar (Fin.cast (by simp) j))
          (fun j => padding.getScalar (Fin.cast (by simp) j))
          (fun j => padding.getScalar (Fin.cast (by simp) j)) (fun _ => 1) (fun _ => 1) 1 c' kw p =
            true ∧
        (Shape.ConvDims.channelsFirst (n := 1) (c := inC) (o := outC) (i := inC)
          (spatial := inSpatial.data.toList) (kernel := kernel.data.toList) (by simp)).inputIndex
          (fun j => stride.getScalar (Fin.cast (by simp) j))
          (fun j => padding.getScalar (Fin.cast (by simp) j))
          (fun j => padding.getScalar (Fin.cast (by simp) j)) (fun _ => 1) (fun _ => 1) 1 1 c' kw
            p <
          Shape.extent (1 :: inC :: inSpatial.data.toList)
            ((Shape.ConvDims.channelsFirst (n := 1) (c := inC) (o := outC) (i := inC)
              (spatial := inSpatial.data.toList) (kernel := kernel.data.toList) (by simp)).input
                p) ↔
        match p with
        | ⟨0, _⟩ => True
        | ⟨1, _⟩ => True
        | ⟨j + 2, h⟩ =>
            padding.getScalar ⟨j, by simp at h; omega⟩ ≤
                Shape.Coord.index c' (j + 2) * stride.getScalar ⟨j, by simp at h; omega⟩ +
                  Shape.Coord.index kw (j + 1) ∧
              Shape.Coord.index c' (j + 2) * stride.getScalar ⟨j, by simp at h; omega⟩ +
                  Shape.Coord.index kw (j + 1) - padding.getScalar ⟨j, by simp at h; omega⟩ <
                inSpatial.getScalar ⟨j, by simp at h; omega⟩ := by
      intro p
      match p with
      | ⟨0, _⟩ =>
          simp only [Shape.ConvDims.inputAligned, Shape.ConvDims.inputIndex,
            Shape.ConvDims.inputOffset, Shape.ConvDims.channelsFirst, finCongr_apply_mk,
            Shape.extent, List.getElem_cons_zero, iff_true]
          show True ∧ Shape.Coord.index c' 1 / (outC / 1) * (1 / 1) + Shape.Coord.index c' 0 < 1
          rw [hc1, hoc, hc0]
          exact ⟨trivial, Nat.zero_lt_one⟩
      | ⟨1, _⟩ =>
          simp only [Shape.ConvDims.inputAligned, Shape.ConvDims.inputIndex,
            Shape.ConvDims.inputOffset, Shape.ConvDims.channelsFirst, finCongr_apply_mk,
            Shape.extent, List.getElem_cons_zero, List.getElem_cons_succ, true_and, iff_true]
          show Shape.Coord.index c' 1 / (outC / 1) * inC + Shape.Coord.index kw 0 < inC
          rw [hc1, hoc, zero_mul, zero_add, ← hidx 0, Shape.Coord.index_zero]
          exact k.1.isLt
      | ⟨j + 2, h⟩ =>
          simp only [Shape.ConvDims.inputAligned, Shape.ConvDims.inputIndex,
            Shape.ConvDims.inputOffset, Shape.ConvDims.channelsFirst, finCongr_apply_mk,
            Fin.cast_mk, Bool.and_eq_true, decide_eq_true_eq, Nat.mod_one, Nat.div_one,
            and_true, Shape.extent, List.getElem_cons_succ]
          have hj : j < d := by simp at h; omega
          show decide (padding.getScalar ⟨j, hj⟩ ≤ Shape.Coord.index c' (j + 2) *
                stride.getScalar ⟨j, hj⟩ + Shape.Coord.index kw (j + 1) * 1) = true ∧
              Shape.Coord.index c' (j + 2) * stride.getScalar ⟨j, hj⟩ +
                  Shape.Coord.index kw (j + 1) * 1 - padding.getScalar ⟨j, hj⟩ <
                inSpatial.data.toList[j]'(by omega) ↔ _
          rw [Nat.mul_one, decide_eq_true_eq, hsp j hj]
          exact Iff.rfl
    have hLlen : (List.zipWith (· - ·) (windowPositions outIdx
        (Shape.Coord.toList kernel.data.toList k.2) stride.data.toList)
          padding.data.toList).length = d := by simp [hlenL, hpLen]
    have hLat : ∀ j (hj : j < d), (List.zipWith (· - ·) (windowPositions outIdx
        (Shape.Coord.toList kernel.data.toList k.2) stride.data.toList)
          padding.data.toList)[j]'(by omega) =
        Shape.Coord.index c' (j + 2) * stride.getScalar ⟨j, hj⟩ + Shape.Coord.index kw (j + 1) -
          padding.getScalar ⟨j, hj⟩ := by
      intro j hj
      rw [List.getElem_zipWith, hq j hj, hpad j hj]
    by_cases hP : ∀ j (hj : j < d), padding.getScalar ⟨j, hj⟩ ≤
        Shape.Coord.index c' (j + 2) * stride.getScalar ⟨j, hj⟩ + Shape.Coord.index kw (j + 1)
    · rw [ite_eq_left (hlow.mpr hP)]
      dsimp only
      by_cases hQ : ∀ j (hj : j < d), Shape.Coord.index c' (j + 2) * stride.getScalar ⟨j, hj⟩ +
          Shape.Coord.index kw (j + 1) - padding.getScalar ⟨j, hj⟩ < inSpatial.getScalar ⟨j, hj⟩
      · have hC := fun p => (hcond p).mpr (by
          match p with
          | ⟨0, _⟩ => trivial
          | ⟨1, _⟩ => trivial
          | ⟨j + 2, h⟩ => exact ⟨hP j _, hQ j _⟩)
        rw [dite_eq_left hC]
        simp only [Tensor.broadcast, Tensor.Rep.pull_apply]
        refine getAtOrZero_eq_of_toList input _ ?_
        apply List.ext_getElem
        · simp [Shape.Coord.length_toList, hLlen]
        · intro j h₁ h₂
          have hjr : j < Shape.rank (inC :: inSpatial.data.toList) := by
            simp [Shape.Coord.length_toList, Shape.rank_eq_length] at h₁ ⊢
            omega
          rw [Shape.Coord.getElem_toList, Shape.Coord.index_ofFun, dite_eq_left hjr]
          split_ifs with hi
          · simp only [Shape.extent] at hi
            rcases j with _ | j
            · simp only [List.getElem_cons_zero] at hi ⊢
              have := k.1.isLt
              omega
            · simp only [List.getElem_cons_succ] at hi ⊢
              have hj : j < d := by simp [Shape.rank_eq_length] at hjr; omega
              rw [hLat j hj]
              have := hQ j hj
              rw [← hsp j hj] at this
              omega
          · rw [Fin.val_cast]
            erw [Shape.Coord.val_get]
            rw [Shape.Coord.ofPositions, Shape.Coord.index_ofFun,
              dite_eq_left (by simp [Shape.rank_eq_length] at hjr ⊢; omega)]
            simp only [Fin.val_succ, Shape.ConvDims.channelsFirst, finCongr_symm,
              finCongr_apply_mk]
            rcases j with _ | j
            · simp only [List.getElem_cons_zero]
              show Shape.Coord.index c' 1 / (outC / 1) * inC + Shape.Coord.index kw 0 = k.1.val
              rw [hc1, hoc, zero_mul, zero_add, ← hidx 0, Shape.Coord.index_zero]
            · have hj : j < d := by simp [Shape.rank_eq_length] at hjr; omega
              simp only [List.getElem_cons_succ]
              rw [hLat j hj]
              show (Shape.Coord.index c' (j + 2) * stride.getScalar ⟨j, hj⟩ +
                  Shape.Coord.index kw (j + 1) * 1 - padding.getScalar ⟨j, hj⟩) / 1 = _
              rw [Nat.mul_one, Nat.div_one]
      · rw [dite_eq_right]
        swap
        · intro h
          exact hQ fun j hj => ((hcond ⟨j + 2, by omega⟩).mp (h ⟨j + 2, by omega⟩)).2
        obtain ⟨j, hj, hjq⟩ : ∃ j, ∃ hj : j < d,
            ¬ (Shape.Coord.index c' (j + 2) * stride.getScalar ⟨j, hj⟩ +
              Shape.Coord.index kw (j + 1) - padding.getScalar ⟨j, hj⟩ <
                inSpatial.getScalar ⟨j, hj⟩) := by
          simpa only [not_forall] using hQ
        refine getAtOrZero_eq_zero_of_not_lt input _ (j + 1) (by simp [hLlen]; omega)
          (by simp; omega) ?_
        simp only [List.getElem_cons_succ, hLat j hj, hsp j hj]
        omega
    · rw [ite_eq_right (mt hlow.mp hP)]
      rw [dite_eq_right]
      intro h
      exact hP fun j hj => ((hcond ⟨j + 2, by omega⟩).mp (h ⟨j + 2, by omega⟩)).1
  · have hlist : (oc.val :: k.1.val :: Shape.Coord.toList kernel.data.toList k.2) =
        Shape.Coord.toList (outC :: inC :: kernel.data.toList) (oc, k) := rfl
    rw [hlist, getAtOrZero_coord_toList weights _ rfl]
    congr 1
    apply Shape.Coord.ext_index
    intro j hj
    simp only [Shape.ConvDims.kernelCoord, Shape.Coord.ofPositions, Shape.Coord.index_ofFun,
      dite_eq_left hj, Shape.ConvDims.channelsFirst, finCongr_symm]
    rcases j with _ | q
    · simp only [Shape.ConvDims.outFeature,
        finCongr_apply, Fin.val_cast]
      have := hc 0
      simp only [Shape.Coord.index_zero] at this
      exact this.symm
    · exact hidx q

/-- In a grouped convolution whose groups divide the output channels, the input channel that
output channel `oc` reads at channel `j` of its group is an input channel. -/
theorem Conv.packedChannel_lt {groups inC outC oc j : Nat} (h : groups ∣ outC) (hoc : oc < outC)
    (hj : j < inC / groups) : oc / (outC / groups) * (inC / groups) + j < inC := by
  obtain ⟨m, rfl⟩ := h
  rcases Nat.eq_zero_or_pos groups with hg | hg
  · subst hg
    simp at hj
  have hm : 0 < m := Nat.pos_of_ne_zero fun hm => by simp [hm] at hoc
  rw [Nat.mul_div_cancel_left m hg]
  have h1 : oc / m < groups := (Nat.div_lt_iff_lt_mul hm).2 hoc
  have h2 : (oc / m + 1) * (inC / groups) ≤ groups * (inC / groups) :=
    Nat.mul_le_mul_right _ h1
  have h3 : groups * (inC / groups) ≤ inC := by
    rw [Nat.mul_comm]
    exact Nat.div_mul_le_self inC groups
  rw [Nat.succ_mul] at h2
  omega

/-- Grouped convolution weights in PyTorch's packed layout `(outC, inC / groups, k...)`, from dense
weights by selecting the diagonal group blocks: output channel `oc` keeps the input channels of its
group `oc / (outC / groups)`, and the entries outside those blocks are ignored.
`TorchLean.Tensor.conv` with `groups` feature groups reads this layout. -/
def Conv.packedKernel {α : Type} [Storage α] {d inC outC : Nat} {kernel : Tensor Nat [d]}
    (groups : Nat) (h : groups ∣ outC)
    (weights : Tensor α (outC :: inC :: kernel.data.toList)) :
    Tensor α (outC :: inC / groups :: kernel.data.toList) :=
  Tensor.Rep.pull (fun c => (c.1,
    ⟨c.1.val / (outC / groups) * (inC / groups) + c.2.1.val,
      packedChannel_lt h c.1.isLt c.2.1.isLt⟩, c.2.2)) weights

/-- `Conv.groupedContract`, the grouped, dilated contraction of `grouped` with asymmetric
padding, is `TorchLean.Tensor.conv` with `groups` feature groups of the input with a batch axis of
one and the weights in PyTorch's packed layout (`packedKernel`). Both sides are read at output
coordinates that agree after the batch index. They read the same window cells in the same order,
with the same zero times the kernel entry at a padding hole or an input channel past the last. -/
theorem Conv.groupedContract_eq_conv {d inC outC : Nat}
    {kernel stride dilation paddingBefore paddingAfter inSpatial : Tensor Nat [d]} (groups : Nat)
    (hdvd : groups ∣ outC) (weights : Tensor α (outC :: inC :: kernel.data.toList))
    (input : Tensor α (inC :: inSpatial.data.toList))
    (c : Shape.Coord (outC :: (outSpatialDilated inSpatial kernel stride dilation
      paddingBefore paddingAfter).data.toList))
    (c' : (Shape.conv (1 :: inC :: inSpatial.data.toList) (outC :: inC / groups ::
      kernel.data.toList) (Shape.ConvDims.channelsFirst (by simp))
      (fun j => stride.getScalar (Fin.cast (by simp) j))
      (fun j => paddingBefore.getScalar (Fin.cast (by simp) j))
      (fun j => paddingAfter.getScalar (Fin.cast (by simp) j)) (fun _ => 1)
      (fun j => dilation.getScalar (Fin.cast (by simp) j)) 1).Coord)
    (hc : ∀ j, Shape.Coord.index c' (j + 1) = Shape.Coord.index c j) :
    groupedContract (stride := stride) (dilation := dilation) (paddingBefore := paddingBefore)
        (paddingAfter := paddingAfter) groups weights input c =
      Tensor.conv
        (Tensor.broadcast input (1 :: inC :: inSpatial.data.toList) Fin.succ
          (Shape.BroadcastDims.succ 1 _))
        (packedKernel groups hdvd weights) (Shape.ConvDims.channelsFirst (by simp))
        (fun j => stride.getScalar (Fin.cast (by simp) j))
        (fun j => paddingBefore.getScalar (Fin.cast (by simp) j))
        (fun j => paddingAfter.getScalar (Fin.cast (by simp) j)) (fun _ => 1)
        (fun j => dilation.getScalar (Fin.cast (by simp) j)) groups 1 c' := by
  obtain ⟨oc, rest⟩ := c
  rw [Tensor.conv_apply]
  simp only [groupedContract, contractWith, Tensor.generate,
    Tensor.dim_apply, Tensor.Rep.get_ofFn]
  generalize ho : Shape.Coord.toList (outSpatialDilated inSpatial kernel stride dilation
    paddingBefore paddingAfter).data.toList rest = outIdx
  generalize hch : oc.val / (outC / groups) * (inC / groups) = base
  have hflat : ∀ (F : α → Shape.Coord (inC / groups :: kernel.data.toList) → α),
      (List.finRange (inC / groups)).foldl (fun acc lc =>
          (Tensor.Rep.coordinates kernel.data.toList).foldl
            (fun acc k => F acc (lc, k)) acc) 0 =
        (List.finRange (Shape.size
          (inC / groups :: kernel.data.toList))).foldl
          (fun acc i => F acc (Coord.unlinearize i)) 0 := by
    intro F
    rw [← foldl_coordinates_eq_finRange, Tensor.Rep.foldl_coordinates_cons]
  simp only [foldlIndices_eq_foldl]
  refine Eq.trans (hflat (fun acc ck => acc +
    (match mkDilatedInputIdx? outIdx (Shape.Coord.toList kernel.data.toList ck.2)
        stride.data.toList dilation.data.toList paddingBefore.data.toList with
      | none => 0
      | some inputIndex => getAtOrZero input ((base + ck.1.val) :: inputIndex)) *
      getAtOrZero weights
        (oc.val :: (base + ck.1.val) :: Shape.Coord.toList kernel.data.toList ck.2))) ?_
  refine foldl_finRange_rel (convWindow_channelsFirst _).symm Eq _ _ 0 0 rfl ?_
  intro acc acc' i i' hidx hacc
  subst hacc
  generalize hk : (Coord.unlinearize i :
    Shape.Coord (inC / groups :: kernel.data.toList)) = k at hidx ⊢
  generalize hkw : Coord.unlinearize i' = kw at hidx ⊢
  congr 2
  · have hoLen : outIdx.length = d := by
      rw [← ho, Shape.Coord.length_toList]
      simp
    have hkLen : (Shape.Coord.toList kernel.data.toList k.2).length = d := by
      rw [Shape.Coord.length_toList]
      simp
    have hdLen : dilation.data.toList.length = d := by simp
    have hwLen : (List.zipWith (· * ·) (Shape.Coord.toList kernel.data.toList k.2)
        dilation.data.toList).length = d := by simp [hkLen, hdLen]
    have hsLen : stride.data.toList.length = d := by simp
    have hpLen : paddingBefore.data.toList.length = d := by simp
    have hoAt : ∀ j (hj : j < d), outIdx[j]'(by omega) = Shape.Coord.index c' (j + 2) := by
      intro j hj
      subst ho
      rw [Shape.Coord.getElem_toList]
      exact (hc (j + 1)).symm
    have hc1 : Shape.Coord.index c' 1 = oc.val := by
      have := hc 0
      simpa using this
    have hspLen : inSpatial.data.toList.length = d := by simp
    -- The list entries of the window positions and the low padding at axis `j`.
    have hq : ∀ j (hj : j < d), (windowPositions outIdx (List.zipWith (· * ·)
        (Shape.Coord.toList kernel.data.toList k.2) dilation.data.toList)
        stride.data.toList)[j]'(by simp [windowPositions, hoLen, hwLen, hsLen]; omega) =
          Shape.Coord.index c' (j + 2) * stride.getScalar ⟨j, hj⟩ +
            Shape.Coord.index kw (j + 1) * dilation.getScalar ⟨j, hj⟩ := by
      intro j hj
      simp only [windowPositions, List.getElem_zipWith]
      rw [hoAt j hj, Shape.Coord.getElem_toList, ← Shape.Coord.index_succ, hidx (j + 1),
        Tensor.getElem_data_toList stride ⟨j, hj⟩,
        Tensor.getElem_data_toList dilation ⟨j, hj⟩]
    have hpad : ∀ j (hj : j < d), paddingBefore.data.toList[j]'(by omega) =
        paddingBefore.getScalar ⟨j, hj⟩ :=
      fun j hj => Tensor.getElem_data_toList paddingBefore ⟨j, hj⟩
    have hsp : ∀ j (hj : j < d), inSpatial.data.toList[j]'(by omega) =
        inSpatial.getScalar ⟨j, hj⟩ := fun j hj =>
      Tensor.getElem_data_toList inSpatial ⟨j, hj⟩
    unfold Tensor.convInput
    rw [mkDilatedInputIdx?_eq (by rw [hdLen, hkLen]),
      mkInputIdx?_eq (by rw [hwLen, hoLen]) (by rw [hsLen, hoLen]) (by rw [hpLen, hoLen])]
    have hlenL : (windowPositions outIdx (List.zipWith (· * ·)
        (Shape.Coord.toList kernel.data.toList k.2) dilation.data.toList)
        stride.data.toList).length = d := by
      simp [windowPositions, hoLen, hwLen, hsLen]
    have hlow : (List.zipWith (fun a b => decide (a ≤ b)) paddingBefore.data.toList
        (windowPositions outIdx (List.zipWith (· * ·)
          (Shape.Coord.toList kernel.data.toList k.2) dilation.data.toList)
          stride.data.toList)).all id = true ↔
        ∀ j (hj : j < d), paddingBefore.getScalar ⟨j, hj⟩ ≤
          Shape.Coord.index c' (j + 2) * stride.getScalar ⟨j, hj⟩ +
            Shape.Coord.index kw (j + 1) * dilation.getScalar ⟨j, hj⟩ := by
      rw [List.all_eq_true]
      constructor
      · intro h j hj
        have hj' : j < (List.zipWith (fun a b => decide (a ≤ b)) paddingBefore.data.toList
            (windowPositions outIdx (List.zipWith (· * ·)
              (Shape.Coord.toList kernel.data.toList k.2) dilation.data.toList)
              stride.data.toList)).length := by simp [hlenL, hpLen, hj]
        have := h _ (List.getElem_mem hj')
        simpa [List.getElem_zipWith, hq j hj, hpad j hj] using this
      · intro h x hx
        obtain ⟨j, hj, rfl⟩ := List.getElem_of_mem hx
        have hj' : j < d := by simpa [hlenL, hpLen] using hj
        simpa [List.getElem_zipWith, hq j hj', hpad j hj'] using h j hj'
    have hc0 : Shape.Coord.index c' 0 = 0 := by
      have := Shape.index_conv_lt (Shape.ConvDims.channelsFirst (n := 1) (c := inC) (o := outC)
        (i := inC / groups) (spatial := inSpatial.data.toList) (kernel := kernel.data.toList)
        (by simp)) c' ⟨0, by omega⟩
      simp only [Shape.convExtent, Shape.ConvDims.channelsFirst, finCongr_apply_mk, Shape.extent,
        List.getElem_cons_zero, Nat.div_one] at this
      exact Nat.lt_one_iff.mp this
    have hoc : oc.val / (outC / 1) = 0 := by
      rw [Nat.div_one]
      exact Nat.div_eq_of_lt oc.isLt
    have hk0 : Shape.Coord.index kw 0 = k.1.val := by rw [← hidx 0, Shape.Coord.index_zero]
    -- The condition of `convInput` along each position.
    have hcond : ∀ p : Fin (inSpatial.data.toList.length + 2),
        (Shape.ConvDims.channelsFirst (n := 1) (c := inC) (o := outC) (i := inC / groups)
          (spatial := inSpatial.data.toList) (kernel := kernel.data.toList) (by simp)).inputAligned
          (fun j => stride.getScalar (Fin.cast (by simp) j))
          (fun j => paddingBefore.getScalar (Fin.cast (by simp) j))
          (fun j => paddingAfter.getScalar (Fin.cast (by simp) j)) (fun _ => 1)
          (fun j => dilation.getScalar (Fin.cast (by simp) j)) 1 c' kw p = true ∧
        (Shape.ConvDims.channelsFirst (n := 1) (c := inC) (o := outC) (i := inC / groups)
          (spatial := inSpatial.data.toList) (kernel := kernel.data.toList) (by simp)).inputIndex
          (fun j => stride.getScalar (Fin.cast (by simp) j))
          (fun j => paddingBefore.getScalar (Fin.cast (by simp) j))
          (fun j => paddingAfter.getScalar (Fin.cast (by simp) j)) (fun _ => 1)
          (fun j => dilation.getScalar (Fin.cast (by simp) j)) groups 1 c' kw p <
          Shape.extent (1 :: inC :: inSpatial.data.toList)
            ((Shape.ConvDims.channelsFirst (n := 1) (c := inC) (o := outC) (i := inC / groups)
              (spatial := inSpatial.data.toList) (kernel := kernel.data.toList) (by simp)).input
                p) ↔
        match p with
        | ⟨0, _⟩ => True
        | ⟨1, _⟩ => base + k.1.val < inC
        | ⟨j + 2, h⟩ =>
            paddingBefore.getScalar ⟨j, by simp at h; omega⟩ ≤
                Shape.Coord.index c' (j + 2) * stride.getScalar ⟨j, by simp at h; omega⟩ +
                  Shape.Coord.index kw (j + 1) * dilation.getScalar ⟨j, by simp at h; omega⟩ ∧
              Shape.Coord.index c' (j + 2) * stride.getScalar ⟨j, by simp at h; omega⟩ +
                  Shape.Coord.index kw (j + 1) * dilation.getScalar ⟨j, by simp at h; omega⟩ -
                  paddingBefore.getScalar ⟨j, by simp at h; omega⟩ <
                inSpatial.getScalar ⟨j, by simp at h; omega⟩ := by
      intro p
      match p with
      | ⟨0, _⟩ =>
          simp only [Shape.ConvDims.inputAligned, Shape.ConvDims.inputIndex,
            Shape.ConvDims.inputOffset, Shape.ConvDims.channelsFirst, finCongr_apply_mk,
            Shape.extent, List.getElem_cons_zero, iff_true]
          show True ∧ Shape.Coord.index c' 1 / (outC / 1) * (1 / 1) + Shape.Coord.index c' 0 < 1
          rw [hc1, hoc, hc0]
          exact ⟨trivial, Nat.zero_lt_one⟩
      | ⟨1, _⟩ =>
          simp only [Shape.ConvDims.inputAligned, Shape.ConvDims.inputIndex,
            Shape.ConvDims.inputOffset, Shape.ConvDims.channelsFirst, finCongr_apply_mk,
            Shape.extent, List.getElem_cons_zero, List.getElem_cons_succ, true_and]
          show Shape.Coord.index c' 1 / (outC / groups) * (inC / groups) +
            Shape.Coord.index kw 0 < inC ↔ _
          rw [hc1, hch, hk0]
      | ⟨j + 2, h⟩ =>
          simp only [Shape.ConvDims.inputAligned, Shape.ConvDims.inputIndex,
            Shape.ConvDims.inputOffset, Shape.ConvDims.channelsFirst, finCongr_apply_mk,
            Fin.cast_mk, Bool.and_eq_true, decide_eq_true_eq, Nat.mod_one, Nat.div_one,
            and_true, Shape.extent, List.getElem_cons_succ]
          have hj : j < d := by simp at h; omega
          show decide (paddingBefore.getScalar ⟨j, hj⟩ ≤ Shape.Coord.index c' (j + 2) *
                stride.getScalar ⟨j, hj⟩ + Shape.Coord.index kw (j + 1) *
                  dilation.getScalar ⟨j, hj⟩) = true ∧
              Shape.Coord.index c' (j + 2) * stride.getScalar ⟨j, hj⟩ +
                  Shape.Coord.index kw (j + 1) * dilation.getScalar ⟨j, hj⟩ -
                  paddingBefore.getScalar ⟨j, hj⟩ <
                inSpatial.data.toList[j]'(by omega) ↔ _
          rw [decide_eq_true_eq, hsp j hj]
          exact Iff.rfl
    have hLlen : (List.zipWith (· - ·) (windowPositions outIdx (List.zipWith (· * ·)
        (Shape.Coord.toList kernel.data.toList k.2) dilation.data.toList) stride.data.toList)
          paddingBefore.data.toList).length = d := by simp [hlenL, hpLen]
    have hLat : ∀ j (hj : j < d), (List.zipWith (· - ·) (windowPositions outIdx
        (List.zipWith (· * ·) (Shape.Coord.toList kernel.data.toList k.2) dilation.data.toList)
          stride.data.toList) paddingBefore.data.toList)[j]'(by omega) =
        Shape.Coord.index c' (j + 2) * stride.getScalar ⟨j, hj⟩ +
          Shape.Coord.index kw (j + 1) * dilation.getScalar ⟨j, hj⟩ -
          paddingBefore.getScalar ⟨j, hj⟩ := by
      intro j hj
      rw [List.getElem_zipWith, hq j hj, hpad j hj]
    by_cases hP : ∀ j (hj : j < d), paddingBefore.getScalar ⟨j, hj⟩ ≤
        Shape.Coord.index c' (j + 2) * stride.getScalar ⟨j, hj⟩ +
          Shape.Coord.index kw (j + 1) * dilation.getScalar ⟨j, hj⟩
    · rw [ite_eq_left (hlow.mpr hP)]
      dsimp only
      by_cases hQ : ∀ j (hj : j < d), Shape.Coord.index c' (j + 2) * stride.getScalar ⟨j, hj⟩ +
          Shape.Coord.index kw (j + 1) * dilation.getScalar ⟨j, hj⟩ -
            paddingBefore.getScalar ⟨j, hj⟩ < inSpatial.getScalar ⟨j, hj⟩
      · by_cases hCh : base + k.1.val < inC
        · have hC := fun p => (hcond p).mpr (by
            match p with
            | ⟨0, _⟩ => trivial
            | ⟨1, _⟩ => exact hCh
            | ⟨j + 2, h⟩ => exact ⟨hP j _, hQ j _⟩)
          rw [dite_eq_left hC]
          simp only [Tensor.broadcast, Tensor.Rep.pull_apply]
          refine getAtOrZero_eq_of_toList input _ ?_
          apply List.ext_getElem
          · simp [Shape.Coord.length_toList, hLlen]
          · intro j h₁ h₂
            have hjr : j < Shape.rank (inC :: inSpatial.data.toList) := by
              simp [Shape.Coord.length_toList, Shape.rank_eq_length] at h₁ ⊢
              omega
            rw [Shape.Coord.getElem_toList, Shape.Coord.index_ofFun, dite_eq_left hjr]
            split_ifs with hi
            · simp only [Shape.extent] at hi
              rcases j with _ | j
              · simp only [List.getElem_cons_zero] at hi ⊢
                omega
              · simp only [List.getElem_cons_succ] at hi ⊢
                have hj : j < d := by simp [Shape.rank_eq_length] at hjr; omega
                rw [hLat j hj]
                have := hQ j hj
                rw [← hsp j hj] at this
                omega
            · rw [Fin.val_cast]
              erw [Shape.Coord.val_get]
              rw [Shape.Coord.ofPositions, Shape.Coord.index_ofFun,
                dite_eq_left (by simp [Shape.rank_eq_length] at hjr ⊢; omega)]
              simp only [Fin.val_succ, Shape.ConvDims.channelsFirst, finCongr_symm,
                finCongr_apply_mk]
              rcases j with _ | j
              · simp only [List.getElem_cons_zero]
                show Shape.Coord.index c' 1 / (outC / groups) * (inC / groups) +
                  Shape.Coord.index kw 0 = base + k.1.val
                rw [hc1, hch, hk0]
              · have hj : j < d := by simp [Shape.rank_eq_length] at hjr; omega
                simp only [List.getElem_cons_succ]
                rw [hLat j hj]
                show (Shape.Coord.index c' (j + 2) * stride.getScalar ⟨j, hj⟩ +
                    Shape.Coord.index kw (j + 1) * dilation.getScalar ⟨j, hj⟩ -
                    paddingBefore.getScalar ⟨j, hj⟩) / 1 = _
                rw [Nat.div_one]
        · rw [dite_eq_right]
          swap
          · intro h
            exact hCh ((hcond ⟨1, by omega⟩).mp (h ⟨1, by omega⟩))
          refine getAtOrZero_eq_zero_of_not_lt input _ 0 (by simp) (by simp) ?_
          simp only [List.getElem_cons_zero]
          omega
      · rw [dite_eq_right]
        swap
        · intro h
          exact hQ fun j hj => ((hcond ⟨j + 2, by omega⟩).mp (h ⟨j + 2, by omega⟩)).2
        obtain ⟨j, hj, hjq⟩ : ∃ j, ∃ hj : j < d,
            ¬ (Shape.Coord.index c' (j + 2) * stride.getScalar ⟨j, hj⟩ +
              Shape.Coord.index kw (j + 1) * dilation.getScalar ⟨j, hj⟩ -
                paddingBefore.getScalar ⟨j, hj⟩ < inSpatial.getScalar ⟨j, hj⟩) := by
          simpa only [not_forall] using hQ
        refine getAtOrZero_eq_zero_of_not_lt input _ (j + 1) (by simp [hLlen]; omega)
          (by simp; omega) ?_
        simp only [List.getElem_cons_succ, hLat j hj, hsp j hj]
        omega
    · rw [ite_eq_right (mt hlow.mp hP)]
      rw [dite_eq_right]
      intro h
      exact hP fun j hj => ((hcond ⟨j + 2, by omega⟩).mp (h ⟨j + 2, by omega⟩)).1
  · refine Eq.trans ?_ (congrArg (packedKernel groups hdvd weights)
      (?_ : ((oc, k) : Shape.Coord (outC :: inC / groups :: kernel.data.toList)) = _))
    · unfold packedKernel
      rw [Tensor.Rep.pull_apply]
      refine getAtOrZero_eq_of_toList weights _ ?_
      rw [← hch]
      rfl
    · refine Shape.Coord.ext_index (s := outC :: inC / groups :: kernel.data.toList)
        fun j hj => ?_
      simp only [Shape.ConvDims.kernelCoord, Shape.Coord.ofPositions, Shape.Coord.index_ofFun,
        dite_eq_left hj, Shape.ConvDims.channelsFirst, finCongr_symm]
      rcases j with _ | q
      · simp only [Shape.ConvDims.outFeature, finCongr_apply, Fin.val_cast]
        have := hc 0
        simp only [Shape.Coord.index_zero] at this
        exact this.symm
      · exact hidx q

/-- At an output coordinate with batch index `b`, a channels-first convolution with input dilation
one and one batch group reads slice `b` of the input alone. Its entry there is the entry of the
convolution of that slice with a batch axis of one, at the coordinate with the same other
indices. -/
theorem _root_.TorchLean.Tensor.conv_channelsFirst_unstack {N C O I : Nat} {sp ks : List Nat}
    (h : ks.length = sp.length) (x : Tensor α (N :: C :: sp)) (w : Tensor α (O :: I :: ks))
    (stride padLow padHigh dilation : Fin sp.length → Nat) (groups : Nat) (b : Fin N)
    (o : (Shape.conv (N :: C :: sp) (O :: I :: ks) (Shape.ConvDims.channelsFirst h) stride padLow
      padHigh (fun _ => 1) dilation 1).Coord)
    (o' : (Shape.conv (1 :: C :: sp) (O :: I :: ks) (Shape.ConvDims.channelsFirst h) stride padLow
      padHigh (fun _ => 1) dilation 1).Coord)
    (hb : Shape.Coord.index o 0 = b.val) (hb' : Shape.Coord.index o' 0 = 0)
    (ho : ∀ j, Shape.Coord.index o (j + 1) = Shape.Coord.index o' (j + 1)) :
    Tensor.conv x w (Shape.ConvDims.channelsFirst h) stride padLow padHigh (fun _ => 1)
        dilation groups 1 o =
      Tensor.conv
        (Tensor.broadcast (Tensor.unstack x b) (1 :: C :: sp) Fin.succ
          (Shape.BroadcastDims.succ 1 _)) w
        (Shape.ConvDims.channelsFirst h) stride padLow padHigh (fun _ => 1) dilation groups 1
        o' := by
  rw [Tensor.conv_apply, Tensor.conv_apply]
  have hf : (Shape.ConvDims.channelsFirst (n := N) (c := C) (o := O) (i := I) h).outFeature o =
      (Shape.ConvDims.channelsFirst (n := 1) (c := C) (o := O) (i := I) h).outFeature o' :=
    Fin.ext (ho 0)
  rw [hf]
  congr 1
  funext acc i
  congr 2
  have hKK : ∀ m, Shape.Coord.index
      (Coord.unlinearize (s := Shape.convWindow (O :: I :: ks)
        (Shape.ConvDims.channelsFirst (n := N) (c := C) (o := O) (i := I) h)) i) m =
        Shape.Coord.index
          (Coord.unlinearize (s := Shape.convWindow (O :: I :: ks)
            (Shape.ConvDims.channelsFirst (n := 1) (c := C) (o := O) (i := I) h)) i) m :=
    fun _ => rfl
  generalize Coord.unlinearize (s := Shape.convWindow (O :: I :: ks)
    (Shape.ConvDims.channelsFirst (n := N) (c := C) (o := O) (i := I) h)) i = K at hKK ⊢
  generalize Coord.unlinearize (s := Shape.convWindow (O :: I :: ks)
    (Shape.ConvDims.channelsFirst (n := 1) (c := C) (o := O) (i := I) h)) i = K' at hKK ⊢
  have hoc : Shape.Coord.index o 1 / (O / 1) = 0 := by
    have := Shape.index_conv_lt
      (Shape.ConvDims.channelsFirst (n := N) (c := C) (o := O) (i := I) h) o ⟨1, by omega⟩
    simp only [Shape.convExtent, Shape.ConvDims.channelsFirst, finCongr_apply_mk, Shape.extent,
      List.getElem_cons_zero] at this
    rw [Nat.div_one]
    exact Nat.div_eq_of_lt this
  have hoc' : Shape.Coord.index o' 1 / (O / 1) = 0 := by rw [← ho 0]; exact hoc
  unfold Tensor.convInput
  have hin : ∀ (n : Nat) (p : Fin (sp.length + 2)),
      (Shape.ConvDims.channelsFirst (n := n) (c := C) (o := O) (i := I) h).input p =
        Fin.cast (by simp [Shape.rank_eq_length]) p := fun _ _ => rfl
  have hker : ∀ (n : Nat) (p : Fin (sp.length + 2)),
      (Shape.ConvDims.channelsFirst (n := n) (c := C) (o := O) (i := I) h).kernel p =
        Fin.cast (by simp [Shape.rank_eq_length, h]) p := fun _ _ => rfl
  have hcond : ∀ p : Fin (sp.length + 2),
      ((Shape.ConvDims.channelsFirst (n := N) (c := C) (o := O) (i := I) h).inputAligned stride
          padLow padHigh (fun _ => 1) dilation 1 o K p = true ∧
        (Shape.ConvDims.channelsFirst (n := N) (c := C) (o := O) (i := I) h).inputIndex stride
            padLow padHigh (fun _ => 1) dilation groups 1 o K p <
          Shape.extent (N :: C :: sp)
            ((Shape.ConvDims.channelsFirst (n := N) (c := C) (o := O) (i := I) h).input p)) ↔
      ((Shape.ConvDims.channelsFirst (n := 1) (c := C) (o := O) (i := I) h).inputAligned stride
          padLow padHigh (fun _ => 1) dilation 1 o' K' p = true ∧
        (Shape.ConvDims.channelsFirst (n := 1) (c := C) (o := O) (i := I) h).inputIndex stride
            padLow padHigh (fun _ => 1) dilation groups 1 o' K' p <
          Shape.extent (1 :: C :: sp)
            ((Shape.ConvDims.channelsFirst (n := 1) (c := C) (o := O) (i := I) h).input p)) := by
    intro p
    match p with
    | ⟨0, _⟩ =>
        simp only [Shape.ConvDims.inputAligned, Shape.ConvDims.inputIndex,
          Shape.ConvDims.inputOffset, hin, hker, Fin.cast_mk, Shape.extent,
          List.getElem_cons_zero, true_and]
        rw [hoc, hoc', hb, hb']
        simp [b.isLt]
    | ⟨1, _⟩ =>
        simp only [Shape.ConvDims.inputAligned, Shape.ConvDims.inputIndex,
          Shape.ConvDims.inputOffset, hin, hker, Fin.cast_mk, Shape.extent,
          List.getElem_cons_zero, List.getElem_cons_succ, true_and]
        rw [ho 0, hKK 0]
    | ⟨j + 2, _⟩ =>
        simp only [Shape.ConvDims.inputAligned, Shape.ConvDims.inputIndex,
          Shape.ConvDims.inputOffset, hin, Fin.cast_mk, Shape.extent, List.getElem_cons_succ]
        rw [ho (j + 1), hKK (j + 1)]
  have hidx0 : (Shape.ConvDims.channelsFirst (n := N) (c := C) (o := O) (i := I) h).inputIndex
      stride padLow padHigh (fun _ => 1) dilation groups 1 o K ⟨0, by omega⟩ = b.val := by
    simp only [Shape.ConvDims.inputIndex, Shape.ConvDims.inputOffset, hin, hker, Fin.cast_mk,
      Shape.extent, List.getElem_cons_zero]
    rw [hoc, hb]
    simp
  have hidx : ∀ p : Fin (sp.length + 2), p.val ≠ 0 →
      (Shape.ConvDims.channelsFirst (n := N) (c := C) (o := O) (i := I) h).inputIndex
          stride padLow padHigh (fun _ => 1) dilation groups 1 o K p =
        (Shape.ConvDims.channelsFirst (n := 1) (c := C) (o := O) (i := I) h).inputIndex
          stride padLow padHigh (fun _ => 1) dilation groups 1 o' K' p := by
    intro p hp
    match p with
    | ⟨0, _⟩ => exact absurd rfl hp
    | ⟨1, _⟩ =>
        simp only [Shape.ConvDims.inputIndex, Shape.ConvDims.inputOffset, hin, hker, Fin.cast_mk,
          Shape.extent]
        rw [ho 0, hKK 0]
    | ⟨j + 2, _⟩ =>
        simp only [Shape.ConvDims.inputIndex, Shape.ConvDims.inputOffset, hin, Fin.cast_mk]
        rw [ho (j + 1), hKK (j + 1)]
  by_cases hy : ∀ p,
      (Shape.ConvDims.channelsFirst (n := 1) (c := C) (o := O) (i := I) h).inputAligned
          stride padLow padHigh (fun _ => 1) dilation 1 o' K' p = true ∧
        (Shape.ConvDims.channelsFirst (n := 1) (c := C) (o := O) (i := I) h).inputIndex stride
            padLow padHigh (fun _ => 1) dilation groups 1 o' K' p <
          Shape.extent (1 :: C :: sp)
            ((Shape.ConvDims.channelsFirst (n := 1) (c := C) (o := O) (i := I) h).input p)
  · have hx := fun p => (hcond p).2 (hy p)
    rw [dite_eq_left hx, dite_eq_left hy]
    refine Eq.trans ?_ (Tensor.broadcast_apply (Tensor.unstack x b)
      (1 :: C :: sp) Fin.succ (Shape.BroadcastDims.succ 1 _) _).symm
    refine Eq.trans ?_ (Tensor.unstack_apply x b _).symm
    congr 1
    apply Shape.Coord.ext_index
    intro m hm
    have hm' : m < sp.length + 2 := by
      simp only [Shape.rank_eq_length, List.length_cons] at hm
      omega
    have hpos := Shape.Coord.index_ofPositions
      (Shape.ConvDims.channelsFirst (n := N) (c := C) (o := O) (i := I) h).input
      (fun p => ⟨(Shape.ConvDims.channelsFirst (n := N) (c := C) (o := O) (i := I) h).inputIndex
        stride padLow padHigh (fun _ => 1) dilation groups 1 o K p, (hx p).2⟩) ⟨m, hm'⟩
    simp only [hin, Fin.cast_mk] at hpos
    rw [hpos]
    cases m with
    | zero => exact hidx0
    | succ m =>
        have hm1 : m < Shape.rank (C :: sp) := by
          simp only [Shape.rank_eq_length, List.length_cons] at hm ⊢
          omega
        rw [Shape.Coord.index_cons_succ, Shape.Coord.index_ofFun, dite_eq_left hm1]
        rw [hidx ⟨m + 1, hm'⟩ (by simp)]
        have hpos' := Shape.Coord.index_ofPositions
          (Shape.ConvDims.channelsFirst (n := 1) (c := C) (o := O) (i := I) h).input
          (fun p => ⟨(Shape.ConvDims.channelsFirst (n := 1) (c := C) (o := O) (i := I)
            h).inputIndex stride padLow padHigh (fun _ => 1) dilation groups 1 o' K' p,
              (hy p).2⟩) ⟨m + 1, hm'⟩
        simp only [hin, Fin.cast_mk] at hpos'
        split_ifs with hext
        · have := (hy ⟨m + 1, hm'⟩).2
          simp only [hin, Fin.cast_mk, Shape.extent, List.getElem_cons_succ] at this hext
          simp only
          omega
        · rw [Fin.val_cast]
          exact hpos'.symm.trans (Shape.Coord.val_get (s := 1 :: C :: sp) _
            ⟨m + 1, by simp only [Shape.rank_eq_length, List.length_cons]; omega⟩).symm
  · have hx : ¬ ∀ p, _ := fun hx => hy fun p => (hcond p).1 (hx p)
    rw [dite_eq_right hx, dite_eq_right hy]

/-- `grouped` at every index of a batch axis is `TorchLean.Tensor.conv` over that axis
with `groups` feature groups and the weights in PyTorch's packed layout (`packedKernel`), plus the
bias broadcast along the output channels: `Op.conv`'s forward value, cast to the batch, output
channels and `outSpatialDilated`. -/
theorem Conv.mapLeading_grouped_eq_conv {N d inC outC : Nat}
    {kernel stride dilation paddingBefore paddingAfter inSpatial : Tensor Nat [d]}
    (h : kernel.data.toList.length = inSpatial.data.toList.length) (groups : Nat)
    (hdvd : groups ∣ outC) (weights : Tensor α (outC :: inC :: kernel.data.toList))
    (bias : Tensor α [outC]) (x : Tensor α (N :: inC :: inSpatial.data.toList)) :
    Tensor.mapLeading [N] (grouped (stride := stride) (dilation := dilation)
        (paddingBefore := paddingBefore) (paddingAfter := paddingAfter) groups weights bias) x =
      Tensor.castShape
        (Tensor.conv x (packedKernel groups hdvd weights)
            (Shape.ConvDims.channelsFirst h)
            (fun j => stride.getScalar (Fin.cast (by simp) j))
            (fun j => paddingBefore.getScalar (Fin.cast (by simp) j))
            (fun j => paddingAfter.getScalar (Fin.cast (by simp) j)) (fun _ => 1)
            (fun j => dilation.getScalar (Fin.cast (by simp) j)) groups 1 +
          Tensor.broadcast bias
            (Shape.conv (N :: inC :: inSpatial.data.toList)
              (outC :: inC / groups :: kernel.data.toList) (Shape.ConvDims.channelsFirst h)
              (fun j => stride.getScalar (Fin.cast (by simp) j))
              (fun j => paddingBefore.getScalar (Fin.cast (by simp) j))
              (fun j => paddingAfter.getScalar (Fin.cast (by simp) j)) (fun _ => 1)
              (fun j => dilation.getScalar (Fin.cast (by simp) j)) 1)
            (fun _ => (Shape.ConvDims.channelsFirst (n := N) (c := inC) (o := outC)
              (i := inC / groups) h).output
              (fun j => stride.getScalar (Fin.cast (by simp) j))
              (fun j => paddingBefore.getScalar (Fin.cast (by simp) j))
              (fun j => paddingAfter.getScalar (Fin.cast (by simp) j)) (fun _ => 1)
              (fun j => dilation.getScalar (Fin.cast (by simp) j)) 1 ⟨1, by omega⟩)
            ((Shape.ConvDims.channelsFirst (n := N) (c := inC) (o := outC) (i := inC / groups)
              h).broadcastDims_conv
              (fun j => stride.getScalar (Fin.cast (by simp) j))
              (fun j => paddingBefore.getScalar (Fin.cast (by simp) j))
              (fun j => paddingAfter.getScalar (Fin.cast (by simp) j)) (fun _ => 1)
              (fun j => dilation.getScalar (Fin.cast (by simp) j)) 1))
        (conv_channelsFirst_dilated inSpatial kernel stride dilation paddingBefore
          paddingAfter) := by
  apply Tensor.Rep.ext
  rintro ⟨b, c⟩
  rw [Tensor.castShape_apply, Tensor.add_apply]
  simp only [Tensor.mapLeading, Tensor.dim_apply]
  simp only [grouped, Tensor.add_apply]
  have e := (conv_channelsFirst_dilated (n := N) (inC := inC) (outC := outC) (I := inC / groups)
    inSpatial kernel stride dilation paddingBefore paddingAfter).symm
  congr 1
  · have e1 := (conv_channelsFirst_dilated (n := 1) (inC := inC) (outC := outC)
      (I := inC / groups) inSpatial kernel stride dilation paddingBefore paddingAfter).symm
    rw [groupedContract_eq_conv groups hdvd weights (Tensor.unstack x b) c
      (_root_.cast (congrArg Shape.Coord e1) (⟨0, Nat.one_pos⟩, c))
      (fun j => Shape.Coord.index_cast e1 (⟨0, Nat.one_pos⟩, c) (j + 1))]
    symm
    exact Tensor.conv_channelsFirst_unstack h x (packedKernel groups hdvd weights)
      _ _ _ _ groups b _ _ (Shape.Coord.index_cast e (b, c) 0)
      (Shape.Coord.index_cast e1 (⟨0, Nat.one_pos⟩, c) 0)
      (fun j => (Shape.Coord.index_cast e (b, c) (j + 1)).trans
        (Shape.Coord.index_cast e1 (⟨0, Nat.one_pos⟩, c) (j + 1)).symm)
  · simp only [Conv.biasBroadcastWith,
      Tensor.dim_apply, Tensor.generate,
      Tensor.Rep.get_ofFn]
    have hbias : getAtOrZero bias [c.1.val] = bias (c.1, PUnit.unit) := by
      rw [getAtOrZero_cons_val, getAtOrZero_scalar_nil]
      exact Tensor.unstack_apply bias c.1 PUnit.unit
    rw [hbias]
    erw [Tensor.broadcast_apply]
    congr 1
    refine Shape.Coord.ext_index (s := [outC]) fun k hk => ?_
    have hk0 : k = 0 := by
      simp only [Shape.rank_eq_length, List.length_cons, List.length_nil] at hk
      omega
    subst hk0
    rw [Shape.Coord.index_ofFun, dite_eq_left hk]
    split_ifs with hone
    · simp only [Shape.extent, List.getElem_cons_zero] at hone
      have := c.1.isLt
      show c.1.val = 0
      omega
    · rw [Fin.val_cast, Shape.Coord.val_get]
      exact (Shape.Coord.index_cast e (b, c) 1).symm

/-- For a channels-first convolution with one group, the window of the input gradient is the output
channels, then the kernel. -/
private theorem Shape.convBackwardInputWindow_channelsFirst {n c o i : Nat}
    {spatial kernel : List Nat} (h : kernel.length = spatial.length) :
    Shape.convBackwardInputWindow (o :: i :: kernel)
      (Shape.ConvDims.channelsFirst (n := n) (c := c) h) 1 = o :: kernel := by
  apply List.ext_getElem
  · simp [Shape.convBackwardInputWindow, h]
  · intro j h₁ h₂
    match j with
    | 0 => simp [Shape.convBackwardInputWindow, Shape.ConvDims.channelsFirst, Shape.extent]; rfl
    | j + 1 =>
        simp only [Shape.convBackwardInputWindow, List.getElem_cons_succ, List.getElem_ofFn,
          Shape.ConvDims.channelsFirst, finCongr_apply, Shape.extent]
        rfl

/-- `inputDeriv`, the input gradient of `forward`, is `TorchLean.Tensor.convBackwardInput`
of the kernel and the cotangent with a batch axis of one, in PyTorch's layout, with symmetric
padding and no dilation. Both sides are read at input coordinates that agree after the batch
index. -/
theorem Conv.inputDeriv_eq_convBackwardInput {d inC outC : Nat}
    {kernel stride padding inSpatial : Tensor Nat [d]}
    (layer : Conv d inC outC kernel stride padding α)
    (input : Tensor α (inC :: inSpatial.data.toList))
    (gradOutput : Tensor α
      (outC :: (outSpatialDilated inSpatial kernel stride
          (Tensor.ones [d]) padding padding).data.toList))
    (c : Shape.Coord (inC :: inSpatial.data.toList))
    (c' : Shape.Coord (1 :: inC :: inSpatial.data.toList))
    (hc : ∀ j, Shape.Coord.index c' (j + 1) = Shape.Coord.index c j) :
    inputDeriv layer input gradOutput c =
      Tensor.convBackwardInput layer.kernel (Shape.ConvDims.channelsFirst (by simp))
        (fun j => stride.getScalar (Fin.cast (by simp) j))
        (fun j => padding.getScalar (Fin.cast (by simp) j))
        (fun j => padding.getScalar (Fin.cast (by simp) j)) (fun _ => 1) 1
        (Tensor.castShape
          (Tensor.broadcast gradOutput
            (1 :: outC :: (outSpatialDilated inSpatial kernel stride
                (Tensor.ones [d]) padding padding).data.toList) Fin.succ
            (Shape.BroadcastDims.succ 1 _))
          (conv_channelsFirst inSpatial kernel stride padding).symm) c' := by
  obtain ⟨ic, rest⟩ := c
  rw [Tensor.convBackwardInput]
  simp only [inputDeriv, Tensor.generate,
    Tensor.dim_apply, Tensor.Rep.get_ofFn,
    Conv.foldlIndices_eq_foldl]
  generalize hi : Shape.Coord.toList inSpatial.data.toList rest = inIdx
  have hflat : ∀ (F : α → Shape.Coord (outC :: kernel.data.toList) → α),
      (List.finRange outC).foldl (fun acc oc =>
          (Tensor.Rep.coordinates kernel.data.toList).foldl
            (fun acc k => F acc (oc, k)) acc) 0 =
        (List.finRange (Shape.size (outC :: kernel.data.toList))).foldl
          (fun acc i => F acc (Coord.unlinearize i)) 0 := by
    intro F
    rw [← foldl_coordinates_eq_finRange, Tensor.Rep.foldl_coordinates_cons]
  refine Eq.trans (hflat (fun acc ck => acc +
    match Conv.mkTransposeInputIdx? inIdx (Shape.Coord.toList kernel.data.toList ck.2)
        stride.data.toList padding.data.toList with
    | none => 0
    | some outIdx => getAtOrZero gradOutput (ck.1.val :: outIdx) *
        getAtOrZero layer.kernel
          (ck.1.val :: ic.val :: Shape.Coord.toList kernel.data.toList ck.2)))
    ?_
  refine foldl_finRange_rel (convBackwardInputWindow_channelsFirst _).symm Eq _ _ 0 0 rfl ?_
  intro acc acc' i i' hidx hacc
  subst hacc
  generalize hk : (Coord.unlinearize i :
    Shape.Coord (outC :: kernel.data.toList)) = k at hidx ⊢
  generalize hkw : Coord.unlinearize i' = kw at hidx ⊢
  have hiLen : inIdx.length = d := by
    rw [← hi, Shape.Coord.length_toList]
    simp
  have hkLen : (Shape.Coord.toList kernel.data.toList k.2).length = d := by
    rw [Shape.Coord.length_toList]
    simp
  have hsLen : stride.data.toList.length = d := by simp
  have hpLen : padding.data.toList.length = d := by simp
  have hiAt : ∀ j (hj : j < d), inIdx[j]'(by omega) = Shape.Coord.index c' (j + 2) := by
    intro j hj
    subst hi
    rw [Shape.Coord.getElem_toList]
    exact (hc (j + 1)).symm
  have hkAt : ∀ j (hj : j < d), (Shape.Coord.toList kernel.data.toList k.2)[j]'(by omega) =
      Shape.Coord.index kw (j + 1) := by
    intro j hj
    rw [Shape.Coord.getElem_toList]
    exact hidx (j + 1)
  have hc1 : Shape.Coord.index c' 1 = ic.val := by
    have := hc 0
    simpa using this
  have hc0 : Shape.Coord.index c' 0 = 0 := by
    have := c'.1.isLt
    simp only [Shape.Coord.index_zero]
    omega
  have hk0 : Shape.Coord.index kw 0 = k.1.val := by
    rw [← hidx 0, Shape.Coord.index_zero]
  have hsAt : ∀ j (hj : j < d), stride.data.toList[j]'(by omega) = stride.getScalar ⟨j, hj⟩ :=
    fun j hj => Tensor.getElem_data_toList stride ⟨j, hj⟩
  have hpAt : ∀ j (hj : j < d), padding.data.toList[j]'(by omega) = padding.getScalar ⟨j, hj⟩ :=
    fun j hj => Tensor.getElem_data_toList padding ⟨j, hj⟩
  rw [mkTransposeInputIdx?_eq (by rw [hkLen, hiLen]) (by rw [hsLen, hiLen])
    (by rw [hpLen, hiLen])]
  -- The alignment test on both sides is one condition on the natural numbers.
  have hA : ((stride.data.toList.all fun a => decide (a ≠ 0)) &&
        (List.zipWith (fun a b => decide (a ≤ b)) (Shape.Coord.toList kernel.data.toList k.2)
          (List.zipWith (· + ·) inIdx padding.data.toList)).all id &&
        (List.zipWith (fun r t => decide (r % t = 0))
          (List.zipWith (· - ·) (List.zipWith (· + ·) inIdx padding.data.toList)
            (Shape.Coord.toList kernel.data.toList k.2)) stride.data.toList).all id) = true ↔
      ∀ j (hj : j < d), stride.getScalar ⟨j, hj⟩ ≠ 0 ∧
        Shape.Coord.index kw (j + 1) ≤ Shape.Coord.index c' (j + 2) + padding.getScalar ⟨j, hj⟩ ∧
        (Shape.Coord.index c' (j + 2) + padding.getScalar ⟨j, hj⟩ - Shape.Coord.index kw (j + 1)) %
          stride.getScalar ⟨j, hj⟩ = 0 := by
    simp only [Bool.and_eq_true, List.all_eq_true]
    constructor
    · rintro ⟨⟨h₁, h₂⟩, h₃⟩ j hj
      refine ⟨?_, ?_, ?_⟩
      · have := h₁ _ (List.getElem_mem (show j < stride.data.toList.length by omega))
        simpa [hsAt j hj] using this
      · have hj' : j < (List.zipWith (fun a b => decide (a ≤ b))
            (Shape.Coord.toList kernel.data.toList k.2)
            (List.zipWith (· + ·) inIdx padding.data.toList)).length := by
          simp [hkLen, hiLen, hpLen, hj]
        have := h₂ _ (List.getElem_mem hj')
        simpa [List.getElem_zipWith, hkAt j hj, hiAt j hj, hpAt j hj] using this
      · have hj' : j < (List.zipWith (fun r t => decide (r % t = 0))
            (List.zipWith (· - ·) (List.zipWith (· + ·) inIdx padding.data.toList)
              (Shape.Coord.toList kernel.data.toList k.2)) stride.data.toList).length := by
          simp [hkLen, hiLen, hpLen, hsLen, hj]
        have := h₃ _ (List.getElem_mem hj')
        simpa [List.getElem_zipWith, hkAt j hj, hiAt j hj, hpAt j hj, hsAt j hj] using this
    · intro h
      refine ⟨⟨?_, ?_⟩, ?_⟩
      · intro x hx
        obtain ⟨j, hj, rfl⟩ := List.getElem_of_mem hx
        have hj' : j < d := by omega
        simpa [hsAt j hj'] using (h j hj').1
      · intro x hx
        obtain ⟨j, hj, rfl⟩ := List.getElem_of_mem hx
        have hj' : j < d := by simpa [hkLen, hiLen, hpLen] using hj
        simpa [List.getElem_zipWith, hkAt j hj', hiAt j hj', hpAt j hj'] using (h j hj').2.1
      · intro x hx
        obtain ⟨j, hj, rfl⟩ := List.getElem_of_mem hx
        have hj' : j < d := by simpa [hkLen, hiLen, hpLen, hsLen] using hj
        simpa [List.getElem_zipWith, hkAt j hj', hiAt j hj', hpAt j hj', hsAt j hj'] using
          (h j hj').2.2
  by_cases hN : ∀ j (hj : j < d), stride.getScalar ⟨j, hj⟩ ≠ 0 ∧
      Shape.Coord.index kw (j + 1) ≤ Shape.Coord.index c' (j + 2) + padding.getScalar ⟨j, hj⟩ ∧
      (Shape.Coord.index c' (j + 2) + padding.getScalar ⟨j, hj⟩ - Shape.Coord.index kw (j + 1)) %
        stride.getScalar ⟨j, hj⟩ = 0
  · rw [ite_eq_left (hA.mpr hN), ite_eq_left]
    swap
    · intro j
      have hj : j.val < d := by have := j.isLt; simp at this; omega
      have := hN j.val hj
      simp only [Nat.mul_one]
      exact this
    dsimp only
    congr 2
    · have hk1 : kw.1.val = k.1.val := hk0
      have hic : ic.val / inC = 0 := Nat.div_eq_of_lt ic.isLt
      have hLlen : (List.zipWith (· / ·) (List.zipWith (· - ·)
          (List.zipWith (· + ·) inIdx padding.data.toList)
          (Shape.Coord.toList kernel.data.toList k.2)) stride.data.toList).length = d := by
        simp [hiLen, hpLen, hkLen, hsLen]
      have hLat : ∀ j (hj : j < d), (List.zipWith (· / ·) (List.zipWith (· - ·)
          (List.zipWith (· + ·) inIdx padding.data.toList)
          (Shape.Coord.toList kernel.data.toList k.2)) stride.data.toList)[j]'(by omega) =
          (Shape.Coord.index c' (j + 2) + padding.getScalar ⟨j, hj⟩ -
            Shape.Coord.index kw (j + 1)) / stride.getScalar ⟨j, hj⟩ := by
        intro j hj
        simp only [List.getElem_zipWith, hiAt j hj, hpAt j hj, hkAt j hj, hsAt j hj]
      have hout : ∀ j (hj : j < d),
          (outSpatialDilated inSpatial kernel stride
              (Tensor.ones [d]) padding padding).data.toList[j]'(by simp; omega) =
            (Shape.conv (1 :: inC :: inSpatial.data.toList) (outC :: inC :: kernel.data.toList)
              (Shape.ConvDims.channelsFirst (by simp))
              (fun j => stride.getScalar (Fin.cast (by simp) j))
              (fun j => padding.getScalar (Fin.cast (by simp) j))
              (fun j => padding.getScalar (Fin.cast (by simp) j)) (fun _ => 1) (fun _ => 1)
              1)[j + 2]'(by simp [Shape.conv, Shape.rank_eq_length]; omega) := by
        intro j hj
        simp only [conv_channelsFirst, List.getElem_cons_succ]
      unfold Tensor.getOrZero
      split
      · rename_i hin
        rw [Tensor.castShape_apply]
        simp only [Tensor.broadcast, Tensor.Rep.pull_apply]
        refine getAtOrZero_eq_of_toList gradOutput _ ?_
        apply List.ext_getElem
        · simp [Shape.Coord.length_toList, hLlen]
        · intro j h₁ h₂
          have hjr : j < Shape.rank
              (outC :: (outSpatialDilated inSpatial kernel stride
                  (Tensor.ones [d]) padding padding).data.toList) := by
            simp [Shape.Coord.length_toList, Shape.rank_eq_length] at h₁ ⊢
            omega
          rw [Shape.Coord.getElem_toList, Shape.Coord.index_ofFun, dite_eq_left hjr]
          split_ifs with hone
          · simp only [Shape.extent] at hone
            match j with
            | 0 =>
                simp only [List.getElem_cons_zero] at hone ⊢
                have := k.1.isLt
                omega
            | j + 1 =>
                have hj : j < d := by simp [Shape.rank_eq_length] at hjr; omega
                simp only [List.getElem_cons_succ] at hone ⊢
                rw [hLat j hj]
                have hb := hin ⟨j + 2, by simp; omega⟩
                have hb' : (Shape.Coord.index c' (j + 2) + padding.getScalar ⟨j, hj⟩ -
                    Shape.Coord.index kw (j + 1) * 1) / stride.getScalar ⟨j, hj⟩ <
                    (outSpatialDilated inSpatial kernel stride
                        (Tensor.ones [d]) padding padding).data.toList[j]'(by
                      simp; omega) := by
                  rw [hout j hj]
                  exact hb
                rw [Nat.mul_one] at hb'
                exact (Nat.lt_one_iff.mp (lt_of_lt_of_eq hb' hone)).symm
          · rw [Fin.val_cast]
            erw [Shape.Coord.val_get]
            rw [Shape.Coord.index_cast (conv_channelsFirst inSpatial kernel stride padding),
              Shape.Coord.ofPositions, Shape.Coord.index_ofFun,
              dite_eq_left (by simp [Shape.conv, Shape.rank_eq_length] at hjr ⊢; omega)]
            simp only [Shape.ConvDims.output, Shape.ConvDims.channelsFirst,
              finCongr_apply_mk]
            match j with
            | 0 =>
                simp only [List.getElem_cons_zero]
                show Shape.Coord.index c' 1 / inC * (outC / 1) + kw.1.val = k.1.val
                rw [hc1, hic, zero_mul, zero_add, hk1]
            | j + 1 =>
                have hj : j < d := by simp [Shape.rank_eq_length] at hjr; omega
                simp only [List.getElem_cons_succ]
                rw [hLat j hj]
                show (Shape.Coord.index c' (j + 2) + padding.getScalar ⟨j, hj⟩ -
                    Shape.Coord.index kw (j + 1) * 1) / stride.getScalar ⟨j, hj⟩ = _
                rw [Nat.mul_one]
      · rename_i hin
        obtain ⟨p, hp⟩ := not_forall.mp hin
        match p with
        | ⟨0, _⟩ =>
            exfalso
            apply hp
            rw [Shape.extent_conv_output]
            show Shape.Coord.index c' 0 < 1 / 1
            rw [hc0]
            omega
        | ⟨1, _⟩ =>
            exfalso
            apply hp
            rw [Shape.extent_conv_output]
            show Shape.Coord.index c' 1 / inC * (outC / 1) + kw.1.val < outC
            rw [hc1, hic, zero_mul, zero_add, hk1]
            exact k.1.isLt
        | ⟨j + 2, h⟩ =>
            have hj : j < d := by simp at h; omega
            refine getAtOrZero_eq_zero_of_not_lt gradOutput _ (j + 1) (by simp [hLlen]; omega)
              (by simp; omega) ?_
            simp only [List.getElem_cons_succ]
            rw [hLat j hj, hout j hj]
            have hb : ¬ ((Shape.Coord.index c' (j + 2) + padding.getScalar ⟨j, hj⟩ -
                Shape.Coord.index kw (j + 1) * 1) / stride.getScalar ⟨j, hj⟩ <
                (Shape.conv (1 :: inC :: inSpatial.data.toList) (outC :: inC :: kernel.data.toList)
                  (Shape.ConvDims.channelsFirst (by simp))
                  (fun j => stride.getScalar (Fin.cast (by simp) j))
                  (fun j => padding.getScalar (Fin.cast (by simp) j))
                  (fun j => padding.getScalar (Fin.cast (by simp) j)) (fun _ => 1) (fun _ => 1)
                  1)[j + 2]'(by simp [Shape.conv, Shape.rank_eq_length]; omega)) := hp
            rw [Nat.mul_one] at hb
            omega
    · have hk1 : kw.1.val = k.1.val := hk0
      have hic : ic.val / inC = 0 := Nat.div_eq_of_lt ic.isLt
      have hicm : ic.val % inC = ic.val := Nat.mod_eq_of_lt ic.isLt
      unfold Tensor.getOrZero
      split
      · rename_i hin
        refine getAtOrZero_eq_of_toList layer.kernel _ ?_
        apply List.ext_getElem
        · simp [Shape.Coord.length_toList, hkLen]
        · intro j h₁ h₂
          have hjr : j < Shape.rank (outC :: inC :: kernel.data.toList) := by
            simp [Shape.Coord.length_toList, Shape.rank_eq_length] at h₁ ⊢
            omega
          rw [Shape.Coord.getElem_toList, Shape.Coord.ofPositions, Shape.Coord.index_ofFun,
            dite_eq_left hjr]
          simp only [Shape.ConvDims.channelsFirst, finCongr_symm, finCongr_apply_mk]
          match j with
          | 0 =>
              show Shape.Coord.index c' 1 / inC * (outC / 1) + kw.1.val = k.1.val
              rw [hc1, hic, zero_mul, zero_add, hk1]
          | 1 =>
              show Shape.Coord.index c' 1 % inC = ic.val
              rw [hc1, hicm]
          | j + 2 =>
              have hj : j < d := by simp [Shape.rank_eq_length] at hjr; omega
              show Shape.Coord.index kw (j + 1) = (Shape.Coord.toList kernel.data.toList k.2)[j]
              rw [hkAt j hj]
      · rename_i hin
        exfalso
        apply hin
        intro p
        match p with
        | ⟨0, _⟩ =>
            show Shape.Coord.index c' 1 / inC * (outC / 1) + kw.1.val < outC
            rw [hc1, hic, zero_mul, zero_add, hk1]
            exact k.1.isLt
        | ⟨1, _⟩ =>
            show Shape.Coord.index c' 1 % inC < inC
            rw [hc1, hicm]
            exact ic.isLt
        | ⟨j + 2, h⟩ =>
            have hj : j < d := by simp at h; omega
            show Shape.Coord.index kw (j + 1) < kernel.data.toList[j]'(by simp; omega)
            rw [← hkAt j hj, Shape.Coord.getElem_toList]
            have := Shape.Coord.index_lt k.2 (k := j) (by simp [Shape.rank_eq_length]; omega)
            exact this
  · rw [ite_eq_right (mt hA.mp hN), ite_eq_right]
    intro h
    apply hN
    intro j hj
    have := h ⟨j, by simp; omega⟩
    simp only [Nat.mul_one] at this
    exact this


/-- For a channels-first convolution of a batch of one, the window of the weight gradient is the
batch axis, then the output's spatial shape. -/
private theorem Shape.convBackwardWeightWindow_channelsFirst {d inC outC : Nat}
    (inSpatial kernel stride padding : Tensor Nat [d]) :
    Shape.convBackwardWeightWindow (1 :: inC :: inSpatial.data.toList)
        (outC :: inC :: kernel.data.toList) (Shape.ConvDims.channelsFirst (by simp))
        (fun j => stride.getScalar (Fin.cast (by simp) j))
        (fun j => padding.getScalar (Fin.cast (by simp) j))
        (fun j => padding.getScalar (Fin.cast (by simp) j)) (fun _ => 1) =
      1 :: (outSpatialDilated inSpatial kernel stride
          (Tensor.ones [d]) padding padding).data.toList := by
  apply List.ext_getElem
  · simp [Shape.convBackwardWeightWindow]
  · intro j h₁ h₂
    match j with
    | 0 =>
        simp only [Shape.convBackwardWeightWindow, List.getElem_cons_zero, Shape.convExtent,
          Shape.ConvDims.channelsFirst, Shape.extent, Nat.div_one]
        rfl
    | j + 1 =>
        have hj : j < d := by simp at h₂; omega
        simp only [Shape.convBackwardWeightWindow, List.getElem_cons_succ, List.getElem_ofFn,
          Shape.convExtent, Shape.extent, Shape.ConvDims.channelsFirst, finCongr_apply,
          Fin.val_cast, Fin.succ_mk, Shape.dilatedKernelExtent_one]
        rw [Tensor.getElem_data_toList
          (outSpatialDilated inSpatial kernel stride (Tensor.ones [d]) padding padding)
          ⟨j, hj⟩]
        simp only [outSpatialDilated, Tensor.getScalar_ofFn, Tensor.getScalar_full]
        congr 1
        · exact Tensor.getElem_data_toList inSpatial ⟨j, hj⟩
        · exact Tensor.getElem_data_toList kernel ⟨j, hj⟩

/-- `kernelDeriv`, the kernel gradient of `forward`, is
`TorchLean.Tensor.convBackwardWeight` of the input and the cotangent, each with a batch axis of
one, in PyTorch's layout, with symmetric padding and no dilation. -/
theorem Conv.kernelDeriv_eq_convBackwardWeight {d inC outC : Nat}
    {kernel stride padding inSpatial : Tensor Nat [d]}
    (layer : Conv d inC outC kernel stride padding α)
    (input : Tensor α (inC :: inSpatial.data.toList))
    (gradOutput : Tensor α
      (outC :: (outSpatialDilated inSpatial kernel stride
          (Tensor.ones [d]) padding padding).data.toList)) :
    kernelDeriv layer input gradOutput =
      Tensor.convBackwardWeight
        (Tensor.broadcast input (1 :: inC :: inSpatial.data.toList) Fin.succ
          (Shape.BroadcastDims.succ 1 _))
        (Shape.ConvDims.channelsFirst (by simp))
        (fun j => stride.getScalar (Fin.cast (by simp) j))
        (fun j => padding.getScalar (Fin.cast (by simp) j))
        (fun j => padding.getScalar (Fin.cast (by simp) j)) (fun _ => 1) 1
        (Tensor.castShape
          (Tensor.broadcast gradOutput
            (1 :: outC :: (outSpatialDilated inSpatial kernel stride
                (Tensor.ones [d]) padding padding).data.toList) Fin.succ
            (Shape.BroadcastDims.succ 1 _))
          (conv_channelsFirst inSpatial kernel stride padding).symm) := by
  apply Tensor.Rep.ext
  rintro ⟨oc, ic, kk⟩
  rw [Tensor.convBackwardWeight]
  simp only [kernelDeriv, Tensor.generate,
    Tensor.dim_apply, Tensor.Rep.get_ofFn,
    Conv.foldlIndices_eq_foldl]
  have hflat : ∀ F : α → Shape.Coord (outSpatialDilated inSpatial kernel stride
      (Tensor.ones [d]) padding padding).data.toList →
      α, (Tensor.Rep.coordinates
        (outSpatialDilated inSpatial kernel stride
            (Tensor.ones [d]) padding padding).data.toList).foldl F 0 =
      (List.finRange (Shape.size
        (1 :: (outSpatialDilated inSpatial kernel stride
            (Tensor.ones [d]) padding padding).data.toList))).foldl
        (fun acc i => F acc (Coord.unlinearize i :
          Shape.Coord (1 :: (outSpatialDilated inSpatial kernel stride
              (Tensor.ones [d]) padding padding).data.toList)).2)
        0 := by
    intro F
    calc (Tensor.Rep.coordinates
          (outSpatialDilated inSpatial kernel stride
              (Tensor.ones [d]) padding padding).data.toList).foldl F 0 =
        (Tensor.Rep.coordinates
          (1 :: (outSpatialDilated inSpatial kernel stride
              (Tensor.ones [d]) padding padding).data.toList)).foldl
          (fun acc c => F acc c.2) 0 := by
          rw [Tensor.Rep.foldl_coordinates_cons,
            show List.finRange 1 = [0] from rfl]
          rfl
      _ = _ := foldl_coordinates_eq_finRange _ _ _
  refine Eq.trans (hflat (fun acc o => acc +
    (match Conv.mkInputIdx?
        (Shape.Coord.toList (outSpatialDilated inSpatial kernel stride
            (Tensor.ones [d]) padding padding).data.toList o)
        (Shape.Coord.toList kernel.data.toList kk) stride.data.toList padding.data.toList with
      | none => 0
      | some inIdx => getAtOrZero input (ic.val :: inIdx)) *
      getAtOrZero gradOutput (oc.val ::
        Shape.Coord.toList (outSpatialDilated inSpatial kernel stride
            (Tensor.ones [d]) padding padding).data.toList o))) ?_
  refine foldl_finRange_rel
    (convBackwardWeightWindow_channelsFirst inSpatial kernel stride padding).symm Eq _ _ 0 0 rfl ?_
  intro acc acc' i i' hidx hacc
  subst hacc
  generalize hck : (Coord.unlinearize i :
    Shape.Coord (1 :: (outSpatialDilated inSpatial kernel stride
        (Tensor.ones [d]) padding padding).data.toList)) = ck at hidx ⊢
  generalize hkw : Coord.unlinearize i' = ow at hidx ⊢
  have hoLen : (Shape.Coord.toList (outSpatialDilated inSpatial kernel stride
      (Tensor.ones [d]) padding padding).data.toList
      ck.2).length = d := by
    rw [Shape.Coord.length_toList]
    simp
  have hkLen : (Shape.Coord.toList kernel.data.toList kk).length = d := by
    rw [Shape.Coord.length_toList]
    simp
  have hsLen : stride.data.toList.length = d := by simp
  have hpLen : padding.data.toList.length = d := by simp
  have hoAt : ∀ j (hj : j < d), (Shape.Coord.toList
      (outSpatialDilated inSpatial kernel stride
          (Tensor.ones [d]) padding padding).data.toList ck.2)[j]'(by omega) =
        Shape.Coord.index ow (j + 1) := by
    intro j hj
    rw [Shape.Coord.getElem_toList]
    exact hidx (j + 1)
  have hkAt : ∀ j (hj : j < d), (Shape.Coord.toList kernel.data.toList kk)[j]'(by omega) =
      Shape.Coord.index kk j := fun j hj => by rw [Shape.Coord.getElem_toList]
  have how0 : ow.1.val = 0 := by
    have h0 := hidx 0
    have := ck.1.isLt
    simp only [Shape.Coord.index_zero] at h0
    have : Shape.Coord.index ow 0 = ow.1.val := rfl
    omega
  have hsAt : ∀ j (hj : j < d), stride.data.toList[j]'(by omega) = stride.getScalar ⟨j, hj⟩ :=
    fun j hj => Tensor.getElem_data_toList stride ⟨j, hj⟩
  have hpAt : ∀ j (hj : j < d), padding.data.toList[j]'(by omega) = padding.getScalar ⟨j, hj⟩ :=
    fun j hj => Tensor.getElem_data_toList padding ⟨j, hj⟩
  have hspAt : ∀ j (hj : j < d), inSpatial.data.toList[j]'(by simp; omega) =
      inSpatial.getScalar ⟨j, hj⟩ :=
    fun j hj => Tensor.getElem_data_toList inSpatial ⟨j, hj⟩
  have hoc : oc.val / (outC / 1) = 0 := by
    rw [Nat.div_one]
    exact Nat.div_eq_of_lt oc.isLt
  congr 2
  · have hlenP : (windowPositions (Shape.Coord.toList
        (outSpatialDilated inSpatial kernel stride
            (Tensor.ones [d]) padding padding).data.toList ck.2)
        (Shape.Coord.toList kernel.data.toList kk) stride.data.toList).length = d := by
      simp [windowPositions, hoLen, hkLen, hsLen]
    have hpos : ∀ j (hj : j < d), (windowPositions (Shape.Coord.toList
        (outSpatialDilated inSpatial kernel stride
            (Tensor.ones [d]) padding padding).data.toList ck.2)
        (Shape.Coord.toList kernel.data.toList kk) stride.data.toList)[j]'(by omega) =
          Shape.Coord.index ow (j + 1) * stride.getScalar ⟨j, hj⟩ + Shape.Coord.index kk j := by
      intro j hj
      simp only [windowPositions, List.getElem_zipWith]
      rw [hoAt j hj, hkAt j hj, hsAt j hj]
    rw [mkInputIdx?_eq (by rw [hkLen, hoLen]) (by rw [hsLen, hoLen]) (by rw [hpLen, hoLen])]
    have hlow : (List.zipWith (fun a b => decide (a ≤ b)) padding.data.toList
        (windowPositions (Shape.Coord.toList
          (outSpatialDilated inSpatial kernel stride
              (Tensor.ones [d]) padding padding).data.toList ck.2)
          (Shape.Coord.toList kernel.data.toList kk) stride.data.toList)).all id = true ↔
        ∀ j (hj : j < d), padding.getScalar ⟨j, hj⟩ ≤
          Shape.Coord.index ow (j + 1) * stride.getScalar ⟨j, hj⟩ + Shape.Coord.index kk j := by
      rw [List.all_eq_true]
      constructor
      · intro h j hj
        have hj' : j < (List.zipWith (fun a b => decide (a ≤ b)) padding.data.toList
            (windowPositions (Shape.Coord.toList
              (outSpatialDilated inSpatial kernel stride
                  (Tensor.ones [d]) padding padding).data.toList ck.2)
              (Shape.Coord.toList kernel.data.toList kk) stride.data.toList)).length := by
          simp [hlenP, hpLen, hj]
        have := h _ (List.getElem_mem hj')
        simpa [List.getElem_zipWith, hpos j hj, hpAt j hj] using this
      · intro h x hx
        obtain ⟨j, hj, rfl⟩ := List.getElem_of_mem hx
        have hj' : j < d := by simpa [hlenP, hpLen] using hj
        simpa [List.getElem_zipWith, hpos j hj', hpAt j hj'] using h j hj'
    have hLat : ∀ j (hj : j < d), (List.zipWith (· - ·) (windowPositions (Shape.Coord.toList
        (outSpatialDilated inSpatial kernel stride
            (Tensor.ones [d]) padding padding).data.toList ck.2)
        (Shape.Coord.toList kernel.data.toList kk) stride.data.toList)
          padding.data.toList)[j]'(by simp [hlenP, hpLen]; omega) =
        Shape.Coord.index ow (j + 1) * stride.getScalar ⟨j, hj⟩ + Shape.Coord.index kk j -
          padding.getScalar ⟨j, hj⟩ := by
      intro j hj
      rw [List.getElem_zipWith, hpos j hj, hpAt j hj]
    have hLlen : (List.zipWith (· - ·) (windowPositions (Shape.Coord.toList
        (outSpatialDilated inSpatial kernel stride
            (Tensor.ones [d]) padding padding).data.toList ck.2)
        (Shape.Coord.toList kernel.data.toList kk) stride.data.toList)
          padding.data.toList).length = d := by simp [hlenP, hpLen]
    by_cases hN : ∀ j (hj : j < d), padding.getScalar ⟨j, hj⟩ ≤
        Shape.Coord.index ow (j + 1) * stride.getScalar ⟨j, hj⟩ + Shape.Coord.index kk j
    · rw [ite_eq_left (hlow.mpr hN), ite_eq_left]
      swap
      · intro j
        have hj : j.val < d := by have := j.isLt; simp at this; omega
        show padding.getScalar ⟨j.val, hj⟩ ≤ Shape.Coord.index ow (j.val + 1) *
          stride.getScalar ⟨j.val, hj⟩ + Shape.Coord.index kk j.val * 1
        rw [Nat.mul_one]
        exact hN j.val hj
      dsimp only
      unfold Tensor.getOrZero
      split
      · rename_i hin
        simp only [Tensor.broadcast, Tensor.Rep.pull_apply]
        refine getAtOrZero_eq_of_toList input _ ?_
        apply List.ext_getElem
        · simp [Shape.Coord.length_toList, hLlen]
        · intro j h₁ h₂
          have hjr : j < Shape.rank (inC :: inSpatial.data.toList) := by
            simp [Shape.Coord.length_toList, Shape.rank_eq_length] at h₁ ⊢
            omega
          rw [Shape.Coord.getElem_toList, Shape.Coord.index_ofFun, dite_eq_left hjr]
          split_ifs with hone
          · simp only [Shape.extent] at hone
            match j with
            | 0 =>
                simp only [List.getElem_cons_zero] at hone ⊢
                have := ic.isLt
                omega
            | j + 1 =>
                have hj : j < d := by simp [Shape.rank_eq_length] at hjr; omega
                simp only [List.getElem_cons_succ] at hone ⊢
                rw [hLat j hj]
                have hb := hin ⟨j + 2, by simp; omega⟩
                have hb' : Shape.Coord.index ow (j + 1) * stride.getScalar ⟨j, hj⟩ +
                    Shape.Coord.index kk j * 1 - padding.getScalar ⟨j, hj⟩ <
                    inSpatial.data.toList[j]'(by simp; omega) := hb
                rw [Nat.mul_one] at hb'
                exact (Nat.lt_one_iff.mp (lt_of_lt_of_eq hb' hone)).symm
          · rw [Fin.val_cast]
            erw [Shape.Coord.val_get]
            rw [Shape.Coord.ofPositions, Shape.Coord.index_ofFun,
              dite_eq_left (by simp [Shape.rank_eq_length] at hjr ⊢; omega)]
            simp only [Shape.ConvDims.channelsFirst, finCongr_symm, finCongr_apply_mk]
            match j with
            | 0 =>
                simp only [List.getElem_cons_zero]
                show Shape.Coord.index (s := outC :: inC :: kernel.data.toList) (oc, ic, kk) 0 /
                    (outC / 1) * inC + ic.val = ic.val
                show oc.val / (outC / 1) * inC + ic.val = ic.val
                rw [hoc, zero_mul, zero_add]
            | j + 1 =>
                have hj : j < d := by simp [Shape.rank_eq_length] at hjr; omega
                simp only [List.getElem_cons_succ]
                rw [hLat j hj]
                show Shape.Coord.index ow (j + 1) * stride.getScalar ⟨j, hj⟩ +
                    Shape.Coord.index kk j * 1 - padding.getScalar ⟨j, hj⟩ = _
                rw [Nat.mul_one]
      · rename_i hin
        obtain ⟨p, hp⟩ := not_forall.mp hin
        match p with
        | ⟨0, _⟩ =>
            exfalso
            apply hp
            show ow.1.val < 1
            rw [how0]
            omega
        | ⟨1, _⟩ =>
            exfalso
            apply hp
            show oc.val / (outC / 1) * inC + ic.val < inC
            rw [hoc, zero_mul, zero_add]
            exact ic.isLt
        | ⟨j + 2, h⟩ =>
            have hj : j < d := by simp at h; omega
            refine getAtOrZero_eq_zero_of_not_lt input _ (j + 1) (by simp [hLlen]; omega)
              (by simp; omega) ?_
            simp only [List.getElem_cons_succ]
            rw [hLat j hj]
            have hb : ¬ (Shape.Coord.index ow (j + 1) * stride.getScalar ⟨j, hj⟩ +
                Shape.Coord.index kk j * 1 - padding.getScalar ⟨j, hj⟩ <
                inSpatial.data.toList[j]'(by simp; omega)) := hp
            rw [Nat.mul_one] at hb
            omega
    · rw [ite_eq_right (mt hlow.mp hN), ite_eq_right]
      intro h
      apply hN
      intro j hj
      have := h ⟨j, by simp; omega⟩
      have h' : padding.getScalar ⟨j, hj⟩ ≤ Shape.Coord.index ow (j + 1) *
          stride.getScalar ⟨j, hj⟩ + Shape.Coord.index kk j * 1 := this
      rw [Nat.mul_one] at h'
      exact h'
  · -- The cotangent is read at an output position of the window, always in range.
    unfold Tensor.getOrZero
    split
    · rename_i hin
      rw [Tensor.castShape_apply]
      simp only [Tensor.broadcast, Tensor.Rep.pull_apply]
      refine getAtOrZero_eq_of_toList gradOutput _ ?_
      apply List.ext_getElem
      · simp [Shape.Coord.length_toList, hoLen]
      · intro j h₁ h₂
        have hjr : j < Shape.rank
            (outC :: (outSpatialDilated inSpatial kernel stride
                (Tensor.ones [d]) padding padding).data.toList) := by
          simp [Shape.Coord.length_toList, Shape.rank_eq_length] at h₁ ⊢
          omega
        rw [Shape.Coord.getElem_toList, Shape.Coord.index_ofFun, dite_eq_left hjr]
        split_ifs with hone
        · match j with
          | 0 =>
              simp only [Shape.extent, List.getElem_cons_zero] at hone ⊢
              have := oc.isLt
              omega
          | j + 1 =>
              have hj : j < d := by simp [Shape.rank_eq_length] at hjr; omega
              simp only [Shape.extent, List.getElem_cons_succ] at hone ⊢
              rw [hoAt j hj]
              have := Shape.Coord.index_lt ck.2 (k := j) (by simp [Shape.rank_eq_length]; omega)
              have hidxj := hidx (j + 1)
              simp only [Shape.extent] at this
              have h' : Shape.Coord.index ck (j + 1) = Shape.Coord.index ck.2 j := rfl
              rw [← hidxj, h']
              omega
        · rw [Fin.val_cast]
          erw [Shape.Coord.val_get]
          rw [Shape.Coord.index_cast (conv_channelsFirst inSpatial kernel stride padding),
            Shape.Coord.ofPositions, Shape.Coord.index_ofFun,
            dite_eq_left (by simp [Shape.conv, Shape.rank_eq_length] at hjr ⊢; omega)]
          simp only [Shape.ConvDims.output, Shape.ConvDims.channelsFirst, finCongr_apply_mk]
          match j with
          | 0 => rfl
          | j + 1 =>
              have hj : j < d := by simp [Shape.rank_eq_length] at hjr; omega
              simp only [List.getElem_cons_succ]
              rw [hoAt j hj]
              rfl
    · rename_i hin
      exfalso
      apply hin
      intro p
      rw [Shape.extent_conv_output]
      match p with
      | ⟨0, _⟩ =>
          show ow.1.val < 1 / 1
          rw [how0]
          omega
      | ⟨1, _⟩ =>
          show Shape.Coord.index (s := outC :: inC :: kernel.data.toList) (oc, ic, kk) 0 < outC
          exact oc.isLt
      | ⟨j + 2, h⟩ =>
          have := Shape.Coord.index_lt ow (k := j + 1)
            (by simp [Shape.convBackwardWeightWindow, Shape.rank_eq_length]; simp at h; omega)
          simp only [Shape.convBackwardWeightWindow, Shape.extent, List.getElem_cons_succ,
            List.getElem_ofFn] at this
          exact this

/-- Summing a tensor with a batch axis of one over every axis but the second, read at an entry
`oc`, adds the entries `(0, oc, r)` in row-major order of `r`. -/
private theorem Conv.reduce_otherAxes_apply {S : Shape} {outC : Nat} {sp : Shape}
    (e : S = 1 :: outC :: sp) (x : Tensor α (1 :: outC :: sp)) (f : Fin S.rank)
    (hf : f.val = 1) (h : S.reduce (S.otherAxes [f]) false = [outC]) (oc : Fin outC) :
    Tensor.castShape (Tensor.reduce .sum (Tensor.castShape x e.symm)
        (S.otherAxes [f]) false) h (oc, ()) =
      (Tensor.Rep.coordinates sp).foldl
        (fun acc r => acc + x (⟨0, Nat.one_pos⟩, oc, r)) 0 := by
  subst e
  obtain ⟨f, hfr⟩ := f
  subst hf
  rw [Tensor.castShape_self, Tensor.castShape_apply, Tensor.reduce_sum_apply]
  have hsel : Shape.selectAxes (1 :: outC :: sp)
      (Shape.otherAxes (1 :: outC :: sp) [⟨1, hfr⟩]) = 1 :: sp := by
    rw [Shape.selectAxes, Shape.sortedAxes_otherAxes]
    exact (Shape.eraseAxes_singleton _ _).trans rfl
  have hflat : (Tensor.Rep.coordinates sp).foldl
      (fun acc r => acc + x (⟨0, Nat.one_pos⟩, oc, r)) 0 =
      (List.finRange (Shape.size (1 :: sp))).foldl
        (fun acc i => acc + x (⟨0, Nat.one_pos⟩, oc,
          (Coord.unlinearize i : Shape.Coord (1 :: sp)).2)) 0 := by
    calc (Tensor.Rep.coordinates sp).foldl
          (fun acc r => acc + x (⟨0, Nat.one_pos⟩, oc, r)) 0 =
        (Tensor.Rep.coordinates (1 :: sp)).foldl
          (fun acc c => acc + x (⟨0, Nat.one_pos⟩, oc, c.2)) 0 := by
          rw [Tensor.Rep.foldl_coordinates_cons,
            show List.finRange 1 = [0] from rfl]
          rfl
      _ = _ := foldl_coordinates_eq_finRange _ _ _
  rw [hflat]
  refine foldl_finRange_rel hsel Eq _ _ 0 0 rfl ?_
  intro acc acc' i i' hidx hacc
  subst hacc
  congr 2
  apply Shape.Coord.ext_index
  intro k hk
  rw [Shape.index_reduceSource, dite_eq_left hk, Shape.sortedAxes_otherAxes]
  match k, hk with
  | 0, hk =>
      have hm : (⟨0, hk⟩ : Fin (Shape.rank (1 :: outC :: sp))) ∈
          Shape.otherAxes (1 :: outC :: sp) [⟨1, hfr⟩] :=
        Shape.mem_otherAxes.mpr fun h =>
          Nat.zero_ne_one (congrArg Fin.val (List.mem_singleton.mp h))
      rw [ite_eq_left hm,
        Shape.idxOf_otherAxes_singleton _ fun h => Nat.zero_ne_one (congrArg Fin.val h)]
      simp only [Fin.mk_lt_mk, Nat.zero_lt_one, ↓reduceIte]
      rw [hidx 0, Shape.Coord.index_zero]
      show _ = 0
      have := (Coord.unlinearize i' : Shape.Coord (1 :: sp)).1.isLt
      omega
  | 1, hk =>
      have hm : (⟨1, hk⟩ : Fin (Shape.rank (1 :: outC :: sp))) ∉
          Shape.otherAxes (1 :: outC :: sp) [⟨1, hfr⟩] :=
        fun h => Shape.mem_otherAxes.mp h (List.mem_singleton.mpr rfl)
      have hother : Shape.otherAxes (1 :: outC :: sp)
          (Shape.otherAxes (1 :: outC :: sp) [⟨1, hfr⟩]) = [⟨1, hfr⟩] := by
        have hsorted : Shape.otherAxes (1 :: outC :: sp)
            (Shape.otherAxes (1 :: outC :: sp) [⟨1, hfr⟩]) =
              Shape.sortedAxes (1 :: outC :: sp) [⟨1, hfr⟩] :=
          List.filter_congr fun a _ => by simp only [Shape.mem_otherAxes, not_not]
        rw [hsorted, Shape.sortedAxes_singleton]
      rw [ite_eq_right hm, Shape.keptIndex, Bool.cond_false, hother, List.idxOf_cons_self]
      exact Shape.Coord.index_cast h.symm (oc, ()) 0
  | k + 2, hk =>
      have hm : (⟨k + 2, hk⟩ : Fin (Shape.rank (1 :: outC :: sp))) ∈
          Shape.otherAxes (1 :: outC :: sp) [⟨1, hfr⟩] :=
        Shape.mem_otherAxes.mpr fun h =>
          absurd (congrArg Fin.val (List.mem_singleton.mp h)) (by show k + 2 ≠ 1; omega)
      rw [ite_eq_left hm, Shape.idxOf_otherAxes_singleton _ fun h =>
        absurd (congrArg Fin.val h) (by show k + 2 ≠ 1; omega)]
      simp only [Fin.mk_lt_mk]
      rw [ite_eq_right (by omega), show k + 2 - 1 = k + 1 by omega, hidx (k + 1),
        Shape.Coord.index_succ]
      rfl

/-- `biasDeriv`, the bias gradient of `forward`, sums the cotangent, with a batch axis of
one and cast to any equal shape `S`, over every axis of `S` but the second. -/
theorem Conv.biasDeriv_eq_reduce {d inC outC : Nat}
    {kernel stride padding inSpatial : Tensor Nat [d]}
    (layer : Conv d inC outC kernel stride padding α)
    (input : Tensor α (inC :: inSpatial.data.toList))
    (gradOutput : Tensor α
      (outC :: (outSpatialDilated inSpatial kernel stride
          (Tensor.ones [d]) padding padding).data.toList))
    {S : Shape} (e : S = 1 :: outC :: (outSpatialDilated inSpatial kernel stride
        (Tensor.ones [d]) padding padding).data.toList)
    (f : Fin S.rank) (hf : f.val = 1) (h : Shape.reduce S (Shape.otherAxes S [f]) false = [outC]) :
    biasDeriv layer input gradOutput =
      Tensor.castShape (Tensor.reduce .sum
        (Tensor.castShape (Tensor.broadcast gradOutput
            (1 :: outC :: (outSpatialDilated inSpatial kernel stride
                (Tensor.ones [d]) padding padding).data.toList) Fin.succ
            (Shape.BroadcastDims.succ 1 _)) e.symm)
        (Shape.otherAxes S [f]) false) h := by
  apply Tensor.Rep.ext
  rintro ⟨oc, u⟩
  cases u
  rw [reduce_otherAxes_apply e _ f hf h oc]
  simp only [biasDeriv, Tensor.dim_apply,
    Conv.foldlIndices_eq_foldl, Tensor.scalar_apply]
  congr 1
  funext acc r
  congr 1
  simp only [Tensor.broadcast, Tensor.Rep.pull_apply]
  refine getAtOrZero_eq_of_toList gradOutput _ ?_
  apply List.ext_getElem
  · simp [Shape.Coord.length_toList]
  · intro j h₁ h₂
    have hjr : j < Shape.rank
        (outC :: (outSpatialDilated inSpatial kernel stride
            (Tensor.ones [d]) padding padding).data.toList) := by
      simp [Shape.Coord.length_toList, Shape.rank_eq_length] at h₁ ⊢
      omega
    rw [Shape.Coord.getElem_toList, Shape.Coord.index_ofFun, dite_eq_left hjr]
    split_ifs with hone
    · match j with
      | 0 =>
          simp only [Shape.extent, List.getElem_cons_zero] at hone ⊢
          have := oc.isLt
          omega
      | j + 1 =>
          simp only [Shape.extent, List.getElem_cons_succ] at hone ⊢
          rw [Shape.Coord.getElem_toList]
          have := Shape.Coord.index_lt r (k := j) (by simp [Shape.rank_eq_length] at hjr ⊢; omega)
          simp only [Shape.extent] at this
          omega
    · rw [Fin.val_cast]
      erw [Shape.Coord.val_get]
      match j with
      | 0 => rfl
      | j + 1 =>
          simp only [List.getElem_cons_succ]
          rw [Shape.Coord.getElem_toList]
          rfl

end Instances

/-- Pooling the spatial axes of a channels-first tensor with unit dilation, symmetric padding and
floor mode keeps the channel axis and has the spatial output shape of `Pool.outSpatial`. -/
theorem Shape.pool_tail_channelsFirst {d : Nat} (C : Nat)
    (inSpatial kernel stride padding : Tensor Nat [d]) :
    Shape.pool (C :: inSpatial.data.toList)
        (List.finRange (Shape.rank (C :: inSpatial.data.toList))).tail
        kernel.data.toList stride.data.toList padding.data.toList padding.data.toList
        (List.replicate d 1) false =
      C :: (Pool.outSpatial inSpatial kernel stride padding).data.toList := by
  have hd : (List.finRange (Shape.rank (C :: inSpatial.data.toList))).tail.map Fin.val =
      (List.finRange (Shape.rank inSpatial.data.toList)).map fun a => a.val + 1 := by
    show (List.finRange (Shape.rank inSpatial.data.toList + 1)).tail.map Fin.val = _
    rw [List.finRange_succ]
    simp [Function.comp_def]
  rw [Shape.pool_cons hd, pool_finRange_spatial]

end TorchLean
