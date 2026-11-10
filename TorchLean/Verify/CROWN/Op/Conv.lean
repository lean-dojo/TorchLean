/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.CROWN.Op
public import TorchLean.Tensor.Op.Table.Conv
public import TorchLean.Verify.CROWN.FlatBox.Flatten

/-!
# The convolution rule on operator graphs

In PyTorch's layout, the flattened forward value of `Op.conv` on packed weights is the flattened
`grouped` of any dense weights that pack to them, over the leading axes its batch axis
stands for (`flat_conv`). `convBox?_of_convOpBox?` reads a box of `convOpBox?` as the box
`Graph.convBox?` gives at the operator's geometry, where the flat forward value of `Op.conv` is the
flattened `grouped`.
-/

@[expose] public section

open TorchLean.Shape (size size_append size_cons)

namespace TorchLean.CROWN.Op

open Conv
open Graph

noncomputable section

/-- Dimension numbers in PyTorch's layout are `ConvDims.channelsFirst`. -/
theorem channelsFirst_eq {N C O I : Nat} {sp ks : List Nat} (h : ks.length = sp.length)
    {dims : Shape.ConvDims (N :: C :: sp) (O :: I :: ks) sp.length} (hcf : ChannelsFirst dims) :
    dims = Shape.ConvDims.channelsFirst h := by
  obtain ⟨di, dk⟩ := dims
  simp only [Shape.ConvDims.channelsFirst, Shape.ConvDims.mk.injEq]
  constructor
  · ext p
    exact (hcf p).1.trans rfl
  · ext p
    exact (hcf p).2.trans rfl

/-- A vector rebuilt from the entries of another along an equality of lengths has its entries. -/
theorem _root_.TorchLean.Tensor.toList_ofFn_getScalar {r d : Nat} (e : r = d) (t : Tensor Nat [d]) :
    (Tensor.ofFn fun j : Fin r => t.getScalar (Fin.cast e j)).data.toList = t.data.toList := by
  subst e
  simp

/-- Flat values of tensors with as many entries and the same row-major data are equal. -/
theorem flatOfSome_eq_of_data {s t : Shape} {x : Tensor ℝ s} {y : Tensor ℝ t}
    (h : s.size = t.size) (hd : Tensor.to x (Array ℝ) = Tensor.to y (Array ℝ)) :
    flatOfSome ⟨s, x⟩ = flatOfSome ⟨t, y⟩ := by
  unfold flatOfSome
  refine flatTensor_eq_of_cast h _ _ (Tensor.Rep.data_injective ?_)
  have hc := Tensor.to_array_castShape (Tensor.reshape x [_] (Shape.size_vector _).symm)
    (congrArg (fun k => Shape.dim k Shape.scalar) h)
  simp only [Tensor.to_array_eq_data] at hc hd
  simp only [castDimScalar] at hc ⊢
  rw [hc]
  exact hd

/-- Reshaping keeps the flat value. -/
theorem flatOfSome_reshape {s t : Shape} (x : Tensor ℝ s) (h : s.size = t.size) :
    flatOfSome ⟨t, Tensor.reshape x t h⟩ = flatOfSome ⟨s, x⟩ :=
  flatOfSome_eq_of_data h.symm (Tensor.to_array_reshape x h)

/-- A shape cast keeps the flat value. -/
theorem flatOfSome_castShape {s t : Shape} (x : Tensor ℝ s) (h : s = t) :
    flatOfSome ⟨t, Tensor.castShape x h⟩ = flatOfSome ⟨s, x⟩ := by
  subst h
  rfl

/-- The flattened forward value of `Op.conv` in PyTorch's layout, on weights
`packedKernel groups hg D` and an input `xt` whose batch axis stands for `lead`, is the flattened
`grouped` with the dense weights `D` over `lead`. -/
theorem flat_conv {N d inC outC : Nat}
    {kernel stride padLow padHigh dilation inSpatial : Tensor Nat [d]}
    (h : kernel.data.toList.length = inSpatial.data.toList.length) {groups : Nat}
    (hg : groups ∣ outC) {lead x y : Shape} (hN : N = lead.size)
    (h₁ : x.size = Shape.size (N :: inC :: inSpatial.data.toList))
    (h₂ : Shape.size (Shape.conv (N :: inC :: inSpatial.data.toList) (outC :: inC / groups ::
        kernel.data.toList) (Shape.ConvDims.channelsFirst h)
        (fun j => stride.getScalar (Fin.cast (by simp) j))
        (fun j => padLow.getScalar (Fin.cast (by simp) j))
        (fun j => padHigh.getScalar (Fin.cast (by simp) j)) (fun _ => 1)
        (fun j => dilation.getScalar (Fin.cast (by simp) j)) 1) = y.size)
    (h₃ : x.size = (lead ++ (inC :: Tensor.to inSpatial (List Nat))).size)
    (D : Tensor ℝ (outC :: inC :: kernel.data.toList)) (b : Tensor ℝ [outC])
    (xt : Tensor ℝ x) :
    flatOfSome ⟨y, Tensor.reshape
      (Tensor.conv (Tensor.reshape xt _ h₁) (packedKernel groups hg D)
          (Shape.ConvDims.channelsFirst h) (fun j => stride.getScalar (Fin.cast (by simp) j))
          (fun j => padLow.getScalar (Fin.cast (by simp) j))
          (fun j => padHigh.getScalar (Fin.cast (by simp) j)) (fun _ => 1)
          (fun j => dilation.getScalar (Fin.cast (by simp) j)) groups 1 +
        Tensor.broadcast b
          (Shape.conv (N :: inC :: inSpatial.data.toList)
            (outC :: inC / groups :: kernel.data.toList) (Shape.ConvDims.channelsFirst h)
            (fun j => stride.getScalar (Fin.cast (by simp) j))
            (fun j => padLow.getScalar (Fin.cast (by simp) j))
            (fun j => padHigh.getScalar (Fin.cast (by simp) j)) (fun _ => 1)
            (fun j => dilation.getScalar (Fin.cast (by simp) j)) 1)
          (fun _ => (Shape.ConvDims.channelsFirst (n := N) (c := inC) (o := outC)
            (i := inC / groups) h).output
            (fun j => stride.getScalar (Fin.cast (by simp) j))
            (fun j => padLow.getScalar (Fin.cast (by simp) j))
            (fun j => padHigh.getScalar (Fin.cast (by simp) j)) (fun _ => 1)
            (fun j => dilation.getScalar (Fin.cast (by simp) j)) 1 ⟨1, by omega⟩)
          ((Shape.ConvDims.channelsFirst (n := N) (c := inC) (o := outC) (i := inC / groups)
            h).broadcastDims_conv
            (fun j => stride.getScalar (Fin.cast (by simp) j))
            (fun j => padLow.getScalar (Fin.cast (by simp) j))
            (fun j => padHigh.getScalar (Fin.cast (by simp) j)) (fun _ => 1)
            (fun j => dilation.getScalar (Fin.cast (by simp) j)) 1)) _ h₂⟩ =
      flatOfSome ⟨_, Tensor.mapLeading lead
        (grouped (inSpatial := inSpatial) (stride := stride) (dilation := dilation)
          (paddingBefore := padLow) (paddingAfter := padHigh) groups D b)
        (Tensor.reshape xt _ h₃)⟩ := by
  subst hN
  have hA : (lead ++ (inC :: Tensor.to inSpatial (List Nat))).size =
      Shape.size (lead.size :: (inC :: Tensor.to inSpatial (List Nat))) := by
    simp only [Shape.size_append, Shape.size_cons]
  have hB : Shape.size (lead.size :: (outC :: Tensor.to
        (outSpatialDilated inSpatial kernel stride dilation padLow padHigh) (List Nat))) =
      (lead ++ (outC :: Tensor.to
        (outSpatialDilated inSpatial kernel stride dilation padLow padHigh)
          (List Nat))).size := by
    simp only [Shape.size_append, Shape.size_cons]
  rw [Tensor.mapLeading_eq_reshape lead _ _ hA hB]
  rw [Conv.mapLeading_grouped_eq_conv h groups hg D b]
  rw [flatOfSome_reshape]
  erw [flatOfSome_reshape, flatOfSome_castShape]
  rfl

/-- A box of `convOpBox?` is the box `Graph.convBox?` gives at the operator's geometry. At that
geometry, for any map `f` to the reals, the flat forward value of `Op.conv` on the images of the
kernel and bias is the flattened `grouped` on the images of the dense weights and bias. -/
theorem convBox?_of_convOpBox? {β : Type} [Storage β] [Context β] [CROWN.BoundOps β]
    (f : β → ℝ) {s k : Shape} {r : Nat} {dims : Shape.ConvDims s k r}
    {stride padLow padHigh dilation : Fin r → Nat} {groups : Nat} {lead x y : Shape}
    {hlead : s.extent (dims.input ⟨0, by omega⟩) = lead.size}
    {hx : x = s.expandAxis (dims.input ⟨0, by omega⟩) lead}
    {hy : y = (s.conv k dims stride padLow padHigh (fun _ => 1) dilation 1).expandAxis
      (dims.output stride padLow padHigh (fun _ => 1) dilation 1 ⟨0, by omega⟩) lead}
    {K : Tensor β k} {b : Tensor β [k.extent (dims.kernel ⟨0, by omega⟩)]} {X B : FlatBox β}
    (hB : convOpBox? dims stride padLow padHigh dilation groups lead K b X = some B)
    (xt : Tensor ℝ x) {rule : Op.Rule ℝ _ _}
    (hs : (Op.conv dims stride padLow padHigh dilation groups hlead hx hy).rule? (α := ℝ) =
      some rule) :
    ∃ (d inC outC : Nat) (kernel stride' padLow' padHigh' dilation' inSpatial : Tensor Nat [d])
      (K' : Tensor β (outC :: inC / groups :: kernel.data.toList))
      (b' : Tensor β [outC])
      (h : x.size = (lead ++ (inC :: Tensor.to inSpatial (List Nat))).size),
      convBox? kernel stride' padLow' padHigh' dilation' inSpatial groups lead K' b' X = some B ∧
        flatOfSome ⟨y, (rule.forward (Tensor.map f K, Tensor.map f b, xt, ())).1⟩ =
          flatOfSome ⟨_, Tensor.mapLeading lead
            (grouped (inSpatial := inSpatial) (stride := stride') (dilation := dilation')
              (paddingBefore := padLow') (paddingAfter := padHigh') groups
              (Tensor.map f (denseKernel groups K')) (Tensor.map f b'))
            (Tensor.reshape xt _ h)⟩ := by
  rw [Op.rule?_conv] at hs
  cases hs
  unfold convOpBox? at hB
  split at hB
  · rename_i N C sp O I ks dims K b
    obtain ⟨⟨hcf, hI, hks, hsp⟩, hB⟩ := Option.dite_none_right_eq_some.mp hB
    subst hI
    obtain ⟨d, T, hT⟩ : ∃ d, ∃ T : Tensor Nat [d], T.data.toList = sp :=
      ⟨_, vecOf sp rfl, vecOf_toList sp rfl⟩
    subst hT hsp
    obtain ⟨KT, hKT⟩ : ∃ KT : Tensor Nat [d], KT.data.toList = ks :=
      ⟨vecOf ks (by simpa using hks), vecOf_toList _ _⟩
    subst hKT
    have hlen : KT.data.toList.length = T.data.toList.length := by simp
    obtain rfl := channelsFirst_eq hlen hcf
    have hd : T.data.toList.length = d := by simp
    obtain ⟨sT, rfl⟩ : ∃ sT : Tensor Nat [d], stride = fun j => sT.getScalar (Fin.cast hd j) :=
      ⟨Tensor.ofFn fun j => stride (Fin.cast hd.symm j), by funext j; simp⟩
    obtain ⟨lT, rfl⟩ : ∃ lT : Tensor Nat [d], padLow = fun j => lT.getScalar (Fin.cast hd j) :=
      ⟨Tensor.ofFn fun j => padLow (Fin.cast hd.symm j), by funext j; simp⟩
    obtain ⟨hT, rfl⟩ : ∃ hT : Tensor Nat [d], padHigh = fun j => hT.getScalar (Fin.cast hd j) :=
      ⟨Tensor.ofFn fun j => padHigh (Fin.cast hd.symm j), by funext j; simp⟩
    obtain ⟨dT, rfl⟩ : ∃ dT : Tensor Nat [d], dilation = fun j => dT.getScalar (Fin.cast hd j) :=
      ⟨Tensor.ofFn fun j => dilation (Fin.cast hd.symm j), by funext j; simp⟩
    have hN : N = lead.size := hlead
    subst hN hx hy
    rw [convBox?_congr hd (vecOf_toList _ _) (Tensor.toList_ofFn_getScalar hd sT)
      (Tensor.toList_ofFn_getScalar hd lT) (Tensor.toList_ofFn_getScalar hd hT)
        (Tensor.toList_ofFn_getScalar hd dT)
      (vecOf_toList _ _), Tensor.castShape_trans] at hB
    have hg : groups ∣ O := by
      have hB' := hB
      unfold convBox? at hB'
      exact (Option.dite_none_right_eq_some.mp hB').1.1
    have hxs : Shape.expandAxis (lead.size :: C :: T.data.toList)
        ((Shape.ConvDims.channelsFirst (n := lead.size) (c := C) (o := O) (i := C / groups)
          hlen).input ⟨0, by omega⟩) lead =
        lead ++ (C :: Tensor.to T (List Nat)) := by
      rfl
    have h₃ := congrArg Shape.size hxs
    refine ⟨d, C, O, KT, sT, lT, hT, dT, T, _, _, h₃, hB, ?_⟩
    refine Eq.trans ?_ (flat_conv hlen hg rfl (Shape.size_eq_of_expandAxis hlead rfl)
      (Shape.size_conv_expandAxis hlead rfl) h₃ (Tensor.map f (denseKernel groups K))
      (Tensor.map f b) xt)
    rw [packedKernel_map, packedKernel_denseKernel]
    rfl
  · cases hB

end

end TorchLean.CROWN.Op
