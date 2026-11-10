/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.CROWN.Conv.Enclosure
public import TorchLean.Verify.CROWN.FlatBox.Matmul
public import TorchLean.Tensor.Layers.Conv.Window

/-!
# The interval convolution on flat boxes, from packed weights

A grouped convolution stores its weights in PyTorch's packed layout `(outC, inC / groups, k...)`
(`Conv.packedKernel`). At `d = 2` this is the `weight` shape of `torch.nn.Conv2d` (PyTorch 2.8
docs), `(out_channels, in_channels/groups, kernel_size[0], kernel_size[1])`. The Python check
`test_convolutions_execute_with_their_payload` of `PyTorchCaptureTests`
(`scripts/checks/test_ir_capture.py`) compares Lean's outputs for grouped `nn.Conv2d` and
`nn.Conv1d` layers with torch's, on integer weights and inputs, so every summation order is exact.

The interval transfer `ibpConv` reads the dense layout `(outC, inC, k...)`, and only at the input
channels of each output channel's group. `denseKernel` spreads packed weights over the dense
layout with zeros outside the groups, `packedKernel` recovers them (`packedKernel_denseKernel`),
and the transfer depends on its weights only through their packed layout (`ibpConv_congr_packed`).

`Graph.convBox?` runs `ibpConv` on the dense layout of packed weights over a flat box. At a
lawful scalar, `CROWN.ibpConv_sound` encloses the grouped convolution with the kernel and bias
read through `toReal`, and `Directed.convBox?_encloses` gives the same enclosure for `convBox?`.
Packing commutes with `Tensor.map` (`packedKernel_map`).
-/

public section

namespace TorchLean

open Conv Tensor

/-- The dense layout `(outC, inC, k...)` of grouped convolution weights given in PyTorch's packed
layout `(outC, inC / groups, k...)`. The group of output channel `oc` starts at input channel
`base = oc / (outC / groups) * (inC / groups)` and has `inC / groups` channels. At an input channel
`ic` of that group the entry is the packed entry at `ic - base`, and it is zero elsewhere. -/
def CROWN.denseKernel {α : Type} [Storage α] [Zero α] {d inC outC : Nat} {kernel : Tensor Nat [d]}
    (groups : Nat) (K : Tensor α (outC :: inC / groups :: kernel.data.toList)) :
    Tensor α (outC :: inC :: kernel.data.toList) :=
  Tensor.Rep.ofFn fun c =>
    if h : c.1.val / (outC / groups) * (inC / groups) ≤ c.2.1.val ∧
        c.2.1.val - c.1.val / (outC / groups) * (inC / groups) < inC / groups then
      K (c.1, ⟨_, h.2⟩, c.2.2)
    else 0

/-- Packing the dense layout of packed weights gives them back. -/
theorem CROWN.packedKernel_denseKernel {α : Type} [Storage α] [Zero α] {d inC outC : Nat}
    {kernel : Tensor Nat [d]} (groups : Nat) (h : groups ∣ outC)
    (K : Tensor α (outC :: inC / groups :: kernel.data.toList)) :
    packedKernel groups h (denseKernel groups K) = K := by
  apply Tensor.Rep.ext
  rintro ⟨oc, lc, c⟩
  simp only [packedKernel, denseKernel, Tensor.Rep.pull_apply,
    Tensor.Rep.get_ofFn]
  rw [dite_eq_left ⟨Nat.le_add_right _ _, by simp⟩]
  congr
  simp

/-- Packing weights commutes with a map of their entries. -/
theorem Conv.packedKernel_map {α β : Type} [Storage α] [Storage β] {d inC outC : Nat}
    {kernel : Tensor Nat [d]} (groups : Nat) (h : groups ∣ outC) (f : α → β)
    (D : Tensor α (outC :: inC :: kernel.data.toList)) :
    packedKernel groups h (Tensor.map f D) = Tensor.map f (packedKernel groups h D) :=
  (Tensor.Rep.map_pull f _ D).symm

/-- The packed weights at output channel `oc`, group-local channel `lc` and index list `l` are the
dense weights at `oc`, input channel `oc / (outC / groups) * (inC / groups) + lc` and `l`. -/
theorem Conv.getAtOrZero_packedKernel {α : Type} [Storage α] [Zero α] {d inC outC : Nat}
    {kernel : Tensor Nat [d]} (groups : Nat) (h : groups ∣ outC)
    (D : Tensor α (outC :: inC :: kernel.data.toList)) (oc : Fin outC)
    (lc : Fin (inC / groups)) (l : List Nat) :
    getAtOrZero (packedKernel groups h D) (oc.val :: lc.val :: l) =
      getAtOrZero D (oc.val :: (oc.val / (outC / groups) * (inC / groups) + lc.val) :: l) := by
  have hlt := packedChannel_lt h oc.isLt lc.isLt
  simp only [getAtOrZero_dim_cons, oc.isLt, lc.isLt, hlt, dite_true]
  congr 1
  apply Tensor.Rep.ext
  intro c
  simp [packedKernel]

variable {α : Type} [Storage α] [Context α] [CROWN.BoundOps α]

/-- `ibpConv` reads its weights only through their packed layout: two layers with the same packed
weights and bias have the same interval convolution. -/
theorem CROWN.ibpConv_congr_packed {d inC outC : Nat}
    {kernel stride padding inSpatial : Tensor Nat [d]}
    {layer layer' : Conv d inC outC kernel stride padding α} (groups : Nat) (h : groups ∣ outC)
    (hk : packedKernel groups h layer.kernel = packedKernel groups h layer'.kernel)
    (hb : layer.bias = layer'.bias) (dilation paddingAfter : Tensor Nat [d]) (lead : Shape)
    (xB : Box α (lead ++ (inC :: Tensor.to inSpatial (List Nat)))) :
    ibpConv layer dilation paddingAfter groups lead xB =
      ibpConv layer' dilation paddingAfter groups lead xB := by
  have hread : ∀ (oc : Fin outC) (lc : Fin (inC / groups)) (l : List Nat),
      getAtOrZero layer.kernel (oc.val :: (oc.val / (outC / groups) * (inC / groups) + lc.val) ::
        l) =
      getAtOrZero layer'.kernel (oc.val :: (oc.val / (outC / groups) * (inC / groups) + lc.val) ::
        l) := by
    intro oc lc l
    rw [← getAtOrZero_packedKernel groups h, ← getAtOrZero_packedKernel groups h, hk]
  induction lead with
  | scalar => simp only [ibpConv, ibpConv.mapRows, hread, hb]
  | dim n rest ih =>
      simp only [ibpConv] at ih ⊢
      simp only [ibpConv.mapRows, ih]

namespace CROWN

/-- Interval box of a grouped convolution with packed weights `K` and bias `b`, mapped over the
leading axes `lead` of a flat box: the box is read at shape `lead` followed by
`(inC, inSpatial...)`, `ibpConv` runs on the dense layout `denseKernel groups K`, and its box is
flattened. Returns `none` unless `groups` divides `outC` and the box has the input's size. -/
@[expose] def Graph.convBox? {d inC outC : Nat}
    (kernel stride padLow padHigh dilation inSpatial : Tensor Nat [d]) (groups : Nat) (lead : Shape)
    (K : Tensor α (outC :: inC / groups :: kernel.data.toList)) (b : Tensor α [outC])
    (X : FlatBox α) : Option (FlatBox α) :=
  if h : groups ∣ outC ∧
      X.dim = (lead ++ (inC :: Tensor.to inSpatial (List Nat))).size then
    let yBox := ibpConv (inSpatial := inSpatial)
      ({ kernel := denseKernel groups K, bias := b } : Conv d inC outC kernel stride padLow α)
      dilation padHigh groups lead
      { lo := ibpUnflatten X.dim X.lo h.2, hi := ibpUnflatten X.dim X.hi h.2 }
    some { dim := _, lo := Tensor.reshape yBox.lo [_] (Shape.size_vector _).symm
           hi := Tensor.reshape yBox.hi [_] (Shape.size_vector _).symm }
  else none

/-- If the box has the input's size, `convBox?` on the packing of dense weights `D` returns the
flattened `ibpConv` box of `D` itself. -/
theorem Graph.convBox?_packedKernel {d inC outC : Nat}
    {kernel stride padLow padHigh dilation inSpatial : Tensor Nat [d]} {groups : Nat} {lead : Shape}
    (hg : groups ∣ outC) (D : Tensor α (outC :: inC :: kernel.data.toList))
    (b : Tensor α [outC]) (X : FlatBox α)
    (h : X.dim = (lead ++ (inC :: Tensor.to inSpatial (List Nat))).size) :
    convBox? kernel stride padLow padHigh dilation inSpatial groups lead (packedKernel groups hg D)
        b X =
      let yBox := ibpConv (inSpatial := inSpatial)
        ({ kernel := D, bias := b } : Conv d inC outC kernel stride padLow α)
        dilation padHigh groups lead
        { lo := ibpUnflatten X.dim X.lo h, hi := ibpUnflatten X.dim X.hi h }
      some { dim := _, lo := Tensor.reshape yBox.lo [_] (Shape.size_vector _).symm
             hi := Tensor.reshape yBox.hi [_] (Shape.size_vector _).symm } := by
  unfold convBox?
  rw [dite_eq_left ⟨hg, h⟩]
  rw [ibpConv_congr_packed
    (layer := { kernel := denseKernel groups (packedKernel groups hg D), bias := b })
    (layer' := { kernel := D, bias := b }) groups hg
    (packedKernel_denseKernel groups hg (packedKernel groups hg D)) rfl]

/-- `convBox?` depends on its geometry only through the entries of each vector. -/
theorem Graph.convBox?_congr {d d' inC outC : Nat} (hd : d = d')
    {kernel stride padLow padHigh dilation inSpatial : Tensor Nat [d]}
    {kernel' stride' padLow' padHigh' dilation' inSpatial' : Tensor Nat [d']}
    (hk : kernel.data.toList = kernel'.data.toList) (hs : stride.data.toList = stride'.data.toList)
    (hl : padLow.data.toList = padLow'.data.toList)
    (hh : padHigh.data.toList = padHigh'.data.toList)
    (hdil : dilation.data.toList = dilation'.data.toList)
    (hsp : inSpatial.data.toList = inSpatial'.data.toList) (groups : Nat) (lead : Shape)
    (K : Tensor α (outC :: inC / groups :: kernel.data.toList)) (b : Tensor α [outC])
    (X : FlatBox α) :
    convBox? kernel stride padLow padHigh dilation inSpatial groups lead K b X =
      convBox? kernel' stride' padLow' padHigh' dilation' inSpatial' groups lead
        (Tensor.castShape K (by rw [hk]) :
          Tensor α (outC :: inC / groups :: kernel'.data.toList)) b X := by
  subst hd
  obtain rfl := Tensor.Rep.data_injective (Array.toList_inj.mp hk)
  obtain rfl := Tensor.Rep.data_injective (Array.toList_inj.mp hs)
  obtain rfl := Tensor.Rep.data_injective (Array.toList_inj.mp hl)
  obtain rfl := Tensor.Rep.data_injective (Array.toList_inj.mp hh)
  obtain rfl := Tensor.Rep.data_injective (Array.toList_inj.mp hdil)
  obtain rfl := Tensor.Rep.data_injective (Array.toList_inj.mp hsp)
  rfl

namespace Directed

open Graph (intervalMul_encloses)
open LawfulBoundOps (toReal_minimum toReal_maximum)

noncomputable section

open Graph (ibpUnflatten convBox?)

variable [LawfulBoundOps α]

local notation "value" => LawfulBoundOps.toReal (α := α)

/-- Read each endpoint of a shaped box as a real number through `toReal`. -/
@[expose] def realBox {s : Shape} (box : Box α s) : Box ℝ s :=
  ⟨Tensor.map value box.lo, Tensor.map value box.hi⟩

/-- Reading endpoints as reals commutes with `getAtOrZero`, including at out-of-range indices,
which read zero. -/
theorem getAtOrZero_map_value {s : Shape} (tensor : Tensor α s) (indices : List Nat) :
    getAtOrZero (Tensor.map value tensor) indices = value (getAtOrZero tensor indices) := by
  induction s generalizing indices with
  | scalar =>
    cases indices <;> simp [LawfulBoundOps.toReal_zero]
  | dim n rest ih =>
    cases indices with
    | nil => simp [LawfulBoundOps.toReal_zero]
    | cons i indices =>
      by_cases hi : i < n
      · simpa only [getAtOrZero_dim_cons, hi, ↓reduceDIte, Tensor.unstack_map] using
          ih (tensor.unstack ⟨i, hi⟩) indices
      · simp [hi, LawfulBoundOps.toReal_zero]

/-- Interpretation commutes with reading a coordinate. -/
theorem map_value_apply {dims : List Nat}
    (tensor : Tensor α dims) (index : Shape.Coord dims) :
    (Tensor.map value tensor) index = value (tensor index) :=
  Tensor.map_apply value tensor index

private theorem weight_mul_encloses {lo hi weight : α} {x : ℝ}
    (hl : value lo ≤ x) (hu : x ≤ value hi) :
    value (Context.minimum (BoundOps.mulDown weight lo) (BoundOps.mulDown weight hi)) ≤
        x * value weight ∧
      x * value weight ≤
        value (Context.maximum (BoundOps.mulUp weight lo) (BoundOps.mulUp weight hi)) := by
  have h := intervalMul_encloses (al := weight) (au := weight)
    (le_refl (value weight)) (le_refl (value weight)) hl hu
  simp only [Graph.intervalMul, toReal_minimum, toReal_maximum, min_self, max_self] at h
  rw [toReal_minimum, toReal_maximum, mul_comm x]
  exact h

/-- If the real reading of `box` contains `input`, then for any dilation, padding and number of
groups the real reading of the `ibpConv` box contains the grouped convolution of `input`, mapped
over `leading`, with the kernel and bias read through `toReal`. -/
theorem _root_.TorchLean.CROWN.ibpConv_sound
    {d inC outC : Nat} {kernel stride padding inSpatial : Tensor Nat [d]}
    (layer : Conv d inC outC kernel stride padding α)
    (dilation paddingAfter : Tensor Nat [d]) (groups : Nat) (leading : Shape)
    (box : Box α (leading ++ (inC :: inSpatial.data.toList)))
    (input : Tensor ℝ (leading ++ (inC :: inSpatial.data.toList)))
    (h : Box.Contains (realBox box) input) :
    Box.Contains (realBox (ibpConv layer dilation paddingAfter groups leading box))
      (Tensor.mapLeading leading
        (grouped (inSpatial := inSpatial) (stride := stride)
          (dilation := dilation) (paddingBefore := padding) (paddingAfter := paddingAfter)
          groups (Tensor.map value layer.kernel) (Tensor.map value layer.bias)) input) := by
  induction leading with
  | scalar =>
    apply (Box.contains_iff_apply _ _).mpr
    intro index
    rcases index with ⟨outChannel, outIndex⟩
    simp only [realBox, map_value_apply, ibpConv, ibpConv.mapRows,
      Tensor.mapLeading, grouped, groupedContract, contractWith,
      biasBroadcastWith, Tensor.add_apply,
      Tensor.dim_apply, Tensor.generate, Tensor.Rep.get_ofFn,
      Bool.false_eq_true, ↓reduceIte, getAtOrZero_map_value]
    constructor
    · apply (LawfulBoundOps.addDown_le _ _).trans
      apply add_le_add_left
      apply List.foldl_rel (r := fun a b => value a ≤ b)
        (by simp [LawfulBoundOps.toReal_zero])
      intro localChannel _ lo exactValue hacc
      apply foldlIndices_rel (fun a b => value a ≤ b) _ _ hacc
      intro lo exactValue kernelIndex hacc
      cases hindex : mkDilatedInputIdx? (Shape.Coord.toList
          (outSpatialDilated inSpatial kernel stride dilation padding paddingAfter).data.toList
          outIndex) kernelIndex stride.data.toList dilation.data.toList padding.data.toList with
      | none => simpa [hindex] using hacc
      | some inputIndex =>
        have hx := contains_getAtOrZero h
          (((outChannel.val / (outC / groups)) * (inC / groups) +
            localChannel.val) :: inputIndex)
        simp only [realBox, getAtOrZero_map_value] at hx
        simpa only [hindex, getAtOrZero_map_value] using
          (LawfulBoundOps.addDown_le _ _).trans
            (add_le_add hacc (weight_mul_encloses
              (weight := getAtOrZero layer.kernel
                (outChannel.val ::
                  ((outChannel.val / (outC / groups)) * (inC / groups) +
                    localChannel.val) :: kernelIndex)) hx.1 hx.2).1)
    · apply le_trans _ (LawfulBoundOps.le_addUp _ _)
      apply add_le_add_left
      apply List.foldl_rel (r := fun a b => a ≤ value b)
        (by simp [LawfulBoundOps.toReal_zero])
      intro localChannel _ exactValue hi hacc
      apply foldlIndices_rel (fun a b => a ≤ value b) _ _ hacc
      intro exactValue hi kernelIndex hacc
      cases hindex : mkDilatedInputIdx? (Shape.Coord.toList
          (outSpatialDilated inSpatial kernel stride dilation padding paddingAfter).data.toList
          outIndex) kernelIndex stride.data.toList dilation.data.toList padding.data.toList with
      | none => simpa [hindex] using hacc
      | some inputIndex =>
        have hx := contains_getAtOrZero h
          (((outChannel.val / (outC / groups)) * (inC / groups) +
            localChannel.val) :: inputIndex)
        simp only [realBox, getAtOrZero_map_value] at hx
        simpa only [hindex, getAtOrZero_map_value] using
          (add_le_add hacc (weight_mul_encloses
            (weight := getAtOrZero layer.kernel
              (outChannel.val ::
                ((outChannel.val / (outC / groups)) * (inC / groups) +
                  localChannel.val) :: kernelIndex)) hx.1 hx.2).2).trans
            (LawfulBoundOps.le_addUp _ _)
  | dim n rest ih =>
    intro head
    simpa only [List.append_eq, realBox, ibpConv, ibpConv.mapRows, Tensor.mapLeading,
      Tensor.unstack_map, Tensor.unstack_dim] using
      ih ⟨box.lo.unstack head, box.hi.unstack head⟩ (input.unstack head) (by
        simpa only [realBox, Tensor.unstack_map] using h head)

/-- Interpreting endpoints commutes with flattening a shaped box. -/
theorem realBox_flattenBox {s : Shape} (box : Box α s) :
    realBox (flattenBox box) = flattenBox (realBox box) := by
  simp only [realBox, flattenBox, Tensor.reshape, Tensor.Rep.map_reshape]

/-- A shaped enclosure gives the corresponding row enclosure after flattening. -/
theorem rowEncloses_flatten {s : Shape} {box : Box α s} {input : Tensor ℝ s}
    (h : Box.Contains (realBox box) input) :
    RowEncloses
      ⟨s.size, Tensor.reshape box.lo [_] (Shape.size_vector _).symm,
          Tensor.reshape box.hi [_] (Shape.size_vector _).symm⟩ s.size
      (fun i => getAtOrZero (Tensor.reshape input [_] (Shape.size_vector _).symm) [i]) := by
  have hf := (Box.contains_flattenBox_iff (realBox box) input).mpr h
  rw [← realBox_flattenBox] at hf
  rw [rowEncloses_iff]
  intro i
  simpa only [realBox, flattenBox, Box.Contains, Tensor.unstack_map, Tensor.getScalar,
    Tensor.item_map, Tensor.getAtOrZero_eq_getScalar] using hf i

/-- If a flat box of width `s.size` encloses `f`, then the real reading of the box read back at
shape `s` contains `f` reshaped to `s`. -/
theorem realBox_contains_ibpUnflatten {s : Shape} {box : FlatBox α} {n : Nat}
    {f : Nat → ℝ} (hdim : box.dim = s.size)
    (hn : n = s.size) (h : RowEncloses box n f) :
    Box.Contains
      (realBox
        ⟨ibpUnflatten box.dim box.lo hdim, ibpUnflatten box.dim box.hi hdim⟩)
      (Tensor.reshape (pointVector s.size f) s (Shape.size_vector _)) := by
  rcases box with ⟨d, lo, hi⟩
  change d = s.size at hdim
  subst d n
  apply (Box.contains_flattenBox_iff _ _).mp
  rw [← realBox_flattenBox]
  simp only [flattenBox, ibpUnflatten, castShape_self, reshape_reshape_self]
  intro i
  simp only [realBox, Box.Contains, Tensor.unstack_map, Tensor.item_map]
  change value (lo.getScalar i) ≤ (pointVector s.size f).getScalar i ∧
    (pointVector s.size f).getScalar i ≤ value (hi.getScalar i)
  simpa only [pointVector, Tensor.getAtOrZero_eq_getScalar, Tensor.getScalar_ofFn] using h.2 i

/-- At a lawful scalar, if `convBox?` returns `B` on a box `X` that encloses the flattening of
`x`, then `B` encloses the flattened grouped convolution of `x` (mapped over `lead`). The
convolution uses the dense layout of `K` and the bias `b`, both read through `toReal`. -/
theorem convBox?_encloses {d inC outC : Nat}
    {kernel stride padLow padHigh dilation inSpatial : Tensor Nat [d]} {groups : Nat} {lead : Shape}
    {K : Tensor α (outC :: inC / groups :: kernel.data.toList)} {b : Tensor α [outC]}
    {X B : FlatBox α}
    (hB : convBox? kernel stride padLow padHigh dilation inSpatial groups lead K b X = some B)
    (x : Tensor ℝ (lead ++ (inC :: Tensor.to inSpatial (List Nat))))
    (hX : RowEncloses X (lead ++ (inC :: Tensor.to inSpatial (List Nat))).size
      fun i => getAtOrZero (Tensor.reshape x [_] (Shape.size_vector _).symm) [i]) :
    RowEncloses B (lead ++ (outC :: Tensor.to
        (outSpatialDilated inSpatial kernel stride dilation padLow padHigh) (List Nat))).size
      fun i => getAtOrZero (Tensor.reshape (Tensor.mapLeading lead
        (grouped (inSpatial := inSpatial) (stride := stride) (dilation := dilation)
          (paddingBefore := padLow) (paddingAfter := padHigh) groups
          (Tensor.map value (denseKernel groups K)) (Tensor.map value b)) x)
        [_] (Shape.size_vector _).symm) [i] := by
  unfold convBox? at hB
  obtain ⟨h, hB⟩ := Option.dite_none_right_eq_some.mp hB
  obtain rfl := Option.some.inj hB
  have hx := realBox_contains_ibpUnflatten h.2 rfl hX
  rw [pointVector_getAtOrZero, reshape_reshape_self] at hx
  simpa only [Tensor.to_list_eq_data] using rowEncloses_flatten (ibpConv_sound
    ({ kernel := denseKernel groups K, bias := b } : Conv d inC outC kernel stride padLow α)
    dilation padHigh groups lead _ x hx)

end

end Directed

end CROWN

end TorchLean
