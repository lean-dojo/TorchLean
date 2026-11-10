/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.CROWN.Conv
public import TorchLean.NN.Conv.Index
public import TorchLean.Tensor.ShapeChange.Flatten
public import TorchLean.Tensor.Context.Real
public import TorchLean.Tensor.Operations
public import TorchLean.Tensor.Contract

/-!
# Real convolution and its affine representation

The coefficient matrix uses the same grouped channel and dilated spatial relation as the convolution
specification. The equalities in this module concern real arithmetic and assert nothing about
agreement between floating-point reduction orders. The main one,
`convLinearMatrix_spec`, says that `convLinearMatrix` and `convBiasBroadcast`
compute the flattened grouped convolution over any leading batch shape.
-/

public section

open TorchLean.Shape.Coord (equivFin equivFin_apply linearize)
open TorchLean.Shape (size)
open TorchLean.Shape.Coord (linearize_cons unlinearize unlinearize_linearize)
open TorchLean.Shape (size_cons)

namespace TorchLean.CROWN

open Tensor Conv
open scoped BigOperators

section

open Shape (Coord)

private theorem sum_channel_indicator (count start channel : Nat) (value : ℝ) :
    (∑ channelIndex : Fin count, if start + channelIndex.val = channel then value else 0) =
      if start ≤ channel ∧ channel < start + count then value else 0 := by
  by_cases hRange : start ≤ channel ∧ channel < start + count
  · rw [ite_eq_left hRange]
    let channelIndex : Fin count := ⟨channel - start, by omega⟩
    rw [Finset.sum_eq_single channelIndex]
    · have hIndex : start + channelIndex.val = channel := by dsimp [channelIndex]; omega
      simp [hIndex]
    · intro other _ hne
      have hIndex : start + other.val ≠ channel := by
        intro h
        apply hne
        apply Fin.ext
        dsimp [channelIndex]
        omega
      simp [hIndex]
    · simp
  · rw [ite_eq_right hRange]
    apply Finset.sum_eq_zero
    intro channelIndex _
    have hIndex : start + channelIndex.val ≠ channel := by
      intro h
      exact hRange ⟨by omega, by have := channelIndex.isLt; omega⟩
    simp [hIndex]

private theorem channel_lookup_sum {inC : Nat} (s : Shape)
    (input : Tensor ℝ (.dim inC s)) (count start : Nat)
    (indices : List Nat) (weight : Nat → ℝ) :
    (∑ channelIndex : Fin count,
      getAtOrZero input ((start + channelIndex.val) :: indices) *
        weight (start + channelIndex.val)) =
      ∑ c : (Shape.dim inC s).Coord, input c *
        (if start ≤ c.1.val ∧ c.1.val < start + count then
          if indices = Shape.Coord.toList s c.2 then weight c.1.val else 0
        else 0) := by
  simp_rw [getAtOrZero_eq_sum_indicator (Shape.dim inC s) input, Finset.sum_mul]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  rintro ⟨head, tail⟩ _
  calc
    _ = ∑ channelIndex : Fin count, input (head, tail) *
        (if start + channelIndex.val = head.val then
          if indices = Shape.Coord.toList s tail then weight head.val else 0
        else 0) := by
      apply Finset.sum_congr rfl
      intro channelIndex _
      by_cases hChannel : start + channelIndex.val = head.val
      · simp [Shape.Coord.toList, hChannel]
      · simp [Shape.Coord.toList, hChannel]
    _ = input (head, tail) *
        (∑ channelIndex : Fin count, if start + channelIndex.val = head.val then
          (if indices = Shape.Coord.toList s tail then weight head.val else 0)
        else 0) := (Finset.mul_sum _ _ _).symm
    _ = _ := by rw [sum_channel_indicator]

private theorem _root_.TorchLean.Conv.contractWith_apply
    {d inC outC : Nat} {kernel inSpatial outSpatial : Tensor Nat [d]}
    (count : Nat) (inputChannel : Fin outC → Fin count → Nat)
    (inputIndex? : List Nat → List Nat → Option (List Nat))
    (weights : Tensor ℝ (outC :: inC :: kernel.data.toList))
    (input : Tensor ℝ (inC :: inSpatial.data.toList))
    (outChannel : Fin outC) (outIndex : Shape.Coord outSpatial.data.toList) :
    (contractWith count inputChannel inputIndex? weights input) (outChannel, outIndex) =
      ∑ channelIndex : Fin count, ∑ kernelIndex : Shape.Coord kernel.data.toList,
        (match inputIndex? (Shape.Coord.toList _ outIndex)
            (Shape.Coord.toList _ kernelIndex) with
        | none => 0
        | some indices => getAtOrZero input (inputChannel outChannel channelIndex :: indices)) *
          getAtOrZero weights (outChannel.val :: inputChannel outChannel channelIndex ::
            Shape.Coord.toList _ kernelIndex) := by
  simp only [contractWith, Tensor.dim_apply,
    Tensor.generate, Tensor.Rep.get_ofFn]
  simp_rw [foldlIndices_add kernel.data.toList]
  exact ((Fin.sum_univ_def _).trans (List.sum_eq_foldl.trans List.foldl_map)).symm

private theorem _root_.TorchLean.Conv.contractWith_eq_coefficients
    {d inC outC : Nat} {kernel inSpatial outSpatial : Tensor Nat [d]}
    (count : Nat) (start : Fin outC → Nat)
    (inputIndex? : List Nat → List Nat → Option (List Nat))
    (weights : Tensor ℝ (outC :: inC :: kernel.data.toList))
    (input : Tensor ℝ (inC :: inSpatial.data.toList))
    (outChannel : Fin outC) (outIndex : Shape.Coord outSpatial.data.toList) :
    (contractWith count (fun channel i => start channel + i.val) inputIndex?
        weights input) (outChannel, outIndex) =
      ∑ c : Shape.Coord (inC :: inSpatial.data.toList), input c *
        (if start outChannel ≤ c.1.val ∧ c.1.val < start outChannel + count then
          ∑ k : Shape.Coord kernel.data.toList,
            if inputIndex? (Shape.Coord.toList _ outIndex) (Shape.Coord.toList _ k) =
                some (Shape.Coord.toList _ c.2) then
              getAtOrZero weights (outChannel.val :: c.1.val :: Shape.Coord.toList _ k)
            else 0
        else 0) := by
  rw [contractWith_apply, Finset.sum_comm]
  calc
    _ = ∑ k : Shape.Coord kernel.data.toList,
        ∑ c : Shape.Coord (inC :: inSpatial.data.toList), input c *
          (if start outChannel ≤ c.1.val ∧ c.1.val < start outChannel + count then
            if inputIndex? (Shape.Coord.toList _ outIndex) (Shape.Coord.toList _ k) =
                some (Shape.Coord.toList _ c.2) then
              getAtOrZero weights (outChannel.val :: c.1.val :: Shape.Coord.toList _ k)
            else 0
          else 0) := by
      apply Finset.sum_congr rfl
      intro k _
      cases hIndex : inputIndex? (Shape.Coord.toList _ outIndex)
          (Shape.Coord.toList _ k) with
      | none => simp
      | some indices =>
          simpa only [Option.some.injEq] using
            channel_lookup_sum inSpatial.data.toList input count
              (start outChannel) indices
              (fun channel => getAtOrZero weights
                (outChannel.val :: channel :: Shape.Coord.toList _ k))
    _ = _ := by
      rw [Finset.sum_comm]
      apply Finset.sum_congr rfl
      intro c _
      by_cases hRange : start outChannel ≤ c.1.val ∧
          c.1.val < start outChannel + count
      · simp only [ite_eq_left hRange, Finset.mul_sum]
      · simp only [ite_eq_right hRange, mul_zero, Finset.sum_const_zero]

private theorem foldlIndices_filter (s : Shape) (init : ℝ)
    (p : List Nat → Prop) [DecidablePred p] (weight : List Nat → ℝ) :
    foldlIndices s init
        (fun acc indices => if p indices then acc + weight indices else acc) =
      init + ∑ k : s.Coord, if p (Shape.Coord.toList s k) then
        weight (Shape.Coord.toList s k) else 0 := by
  simpa only [add_ite, add_zero] using
    foldlIndices_add s init (fun indices => if p indices then weight indices else 0)

/-- Positive dilation makes the spatial input-index relation injective in the kernel index. -/
theorem _root_.TorchLean.Conv.mkDilatedInputIdx?_kernel_inj
    {outIdx left right stride dilation padding inputIdx : List Nat}
    (hdil : ∀ d ∈ dilation, 0 < d)
    (hl : mkDilatedInputIdx? outIdx left stride dilation padding = some inputIdx)
    (hr : mkDilatedInputIdx? outIdx right stride dilation padding = some inputIdx) :
    left = right := by
  induction outIdx generalizing left right stride dilation padding inputIdx with
  | nil =>
      cases left <;> cases right <;> cases stride <;> cases dilation <;> cases padding <;>
        simp_all [mkDilatedInputIdx?]
  | cons o os ih =>
      cases left with
      | nil => simp [mkDilatedInputIdx?] at hl
      | cons l ls =>
          cases right with
          | nil => simp [mkDilatedInputIdx?] at hr
          | cons r rs =>
              cases stride with
              | nil => simp [mkDilatedInputIdx?] at hl
              | cons s ss =>
                  cases dilation with
                  | nil => simp [mkDilatedInputIdx?] at hl
                  | cons d ds =>
                      cases padding with
                      | nil => simp [mkDilatedInputIdx?] at hl
                      | cons p ps =>
                          simp only [mkDilatedInputIdx?] at hl hr
                          split at hl
                          · contradiction
                          · rename_i hleft
                            split at hr
                            · contradiction
                            · rename_i hright
                              cases hlt : mkDilatedInputIdx? os ls ss ds ps with
                              | none => simp [hlt] at hl
                              | some il =>
                                  cases hrt : mkDilatedInputIdx? os rs ss ds ps with
                                  | none => simp [hrt] at hr
                                  | some ir =>
                                      simp only [hlt, Option.some.injEq] at hl
                                      simp only [hrt, Option.some.injEq] at hr
                                      have hc := List.cons.inj (hl.trans hr.symm)
                                      have hd : 0 < d := hdil d (by simp)
                                      have hp : l * d = r * d := by omega
                                      have hhead := Nat.eq_of_mul_eq_mul_right hd hp
                                      have htail := ih (fun d hm => hdil d (by simp [hm])) hlt
                                        (hc.2 ▸ hrt)
                                      exact congrArg₂ List.cons hhead htail

/-- At `ℝ` the weight read under positive dilation and the sum under zero dilation are both the sum
of the weights of the kernel coordinates that reach the input position. -/
theorem convKernelCoefficient_eq_sum
    (kernel outIdx inputIdx stride dilation padding : List Nat) (weight : List Nat → ℝ) :
    convKernelCoefficient kernel outIdx inputIdx stride dilation padding weight =
      ∑ k : Shape.Coord kernel,
        if mkDilatedInputIdx? outIdx (Shape.Coord.toList _ k) stride dilation padding =
            some inputIdx then
          weight (Shape.Coord.toList _ k)
        else 0 := by
  unfold convKernelCoefficient
  split
  · rename_i hdil
    have hpos : ∀ d ∈ dilation, 0 < d := by simpa using List.all_eq_true.mp hdil
    dsimp only
    cases hfind : (List.finRange (Shape.size kernel)).find? (fun k =>
        decide (mkDilatedInputIdx? outIdx
          (Shape.Coord.toList _ (Coord.unlinearize k)) stride dilation padding =
            some inputIdx)) with
    | none =>
        simp only
        symm
        apply Finset.sum_eq_zero
        intro k _
        have hn := List.find?_eq_none.mp hfind (Coord.linearize k) (List.mem_finRange _)
        have hne : mkDilatedInputIdx? outIdx (Shape.Coord.toList _ k) stride dilation padding ≠
            some inputIdx := by simpa using hn
        simp [hne]
    | some found =>
        simp only
        have hfound : mkDilatedInputIdx? outIdx
            (Shape.Coord.toList _ (Coord.unlinearize found)) stride dilation padding =
              some inputIdx := by
          simpa using List.find?_some hfind
        symm
        rw [Finset.sum_eq_single (Coord.unlinearize found)]
        · simp [hfound]
        · intro k _ hk
          have hne : mkDilatedInputIdx? outIdx (Shape.Coord.toList _ k) stride dilation padding ≠
              some inputIdx := by
            intro h
            apply hk
            have := congrArg (Shape.Coord.ofList? kernel)
              (mkDilatedInputIdx?_kernel_inj hpos h hfound)
            simpa only [Shape.Coord.ofList?_toList, Option.some.injEq] using this
          simp [hne]
        · simp
  · simpa using foldlIndices_filter kernel 0
      (fun k => mkDilatedInputIdx? outIdx k stride dilation padding = some inputIdx) weight

/-- For positive dilation, selecting a stored coefficient commutes with any scalar map that
preserves zero. -/
theorem convKernelCoefficient_map {α β : Type} [Storage α] [Context α]
    [Storage β] [Context β] (f : α → β) (hf : f 0 = 0)
    (kernel outIdx inputIdx stride dilation padding : List Nat) (weight : List Nat → α)
    (hdil : ∀ d ∈ dilation, 0 < d) :
    convKernelCoefficient kernel outIdx inputIdx stride dilation padding (fun k => f (weight k)) =
      f (convKernelCoefficient kernel outIdx inputIdx stride dilation padding weight) := by
  have hpos : dilation.all (fun d => decide (0 < d)) = true := by
    simpa using hdil
  simp only [convKernelCoefficient, hpos, ↓reduceIte]
  split <;> simp [hf]

private theorem convLinearMatrix_scalar_apply
    {d inC outC : Nat} {kernel stride padding inSpatial : Tensor Nat [d]}
    (layer : Conv d inC outC kernel stride padding ℝ)
    (dilation paddingAfter : Tensor Nat [d]) (groups : Nat)
    (outChannel : Fin outC)
    (outIndex : Shape.Coord (outSpatialDilated inSpatial kernel stride dilation padding
        paddingAfter).data.toList)
    (inChannel : Fin inC) (inIndex : Shape.Coord inSpatial.data.toList) :
    get2 (convLinearMatrix layer dilation paddingAfter groups .scalar)
        (Coord.linearize (outChannel, outIndex))
        (Coord.linearize (inChannel, inIndex)) =
      let start := (outChannel.val / (outC / groups)) * (inC / groups)
      if start ≤ inChannel.val ∧ inChannel.val < start + inC / groups then
        ∑ k : Shape.Coord kernel.data.toList,
          if mkDilatedInputIdx? (Shape.Coord.toList _ outIndex) (Shape.Coord.toList _ k)
              stride.data.toList dilation.data.toList padding.data.toList =
                some (Shape.Coord.toList _ inIndex) then
            getAtOrZero layer.kernel
              (outChannel.val :: inChannel.val :: Shape.Coord.toList _ k)
          else 0
      else 0 := by
  let start := (outChannel.val / (outC / groups)) * (inC / groups)
  by_cases hRange : start ≤ inChannel.val ∧ inChannel.val < start + inC / groups
  · have hLower : ¬inChannel.val < start := by omega
    have hUpper : ¬start + inC / groups ≤ inChannel.val := by omega
    dsimp only [start] at hRange hLower hUpper
    -- Here and below, simp leaves out `linearize_cons` and `size_cons`. They would rewrite the
    -- flat indices and their `Fin` types before `get2_eq_apply` and `unlinearize_linearize` match.
    simp [convLinearMatrix, get2_eq_apply, Shape.Coord.toList,
      hLower, hUpper, hRange, convKernelCoefficient_eq_sum, -Coord.linearize_cons, -Shape.size_cons]
  · have hOutside : inChannel.val < start ∨ start + inC / groups ≤ inChannel.val := by
      omega
    rcases hOutside with hLower | hUpper
    · dsimp only [start] at hRange hLower
      simp [convLinearMatrix, get2_eq_apply, Shape.Coord.toList,
        hLower, hRange, -Coord.linearize_cons, -Shape.size_cons]
    · dsimp only [start] at hRange hUpper
      simp [convLinearMatrix, get2_eq_apply, Shape.Coord.toList,
        hUpper, hRange, -Coord.linearize_cons, -Shape.size_cons]

private theorem convBiasBroadcast_scalar_apply
    {d outC : Nat} {outSpatial : Tensor Nat [d]} (bias : Tensor ℝ [outC])
    (outChannel : Fin outC) (outIndex : Shape.Coord outSpatial.data.toList) :
    getScalar (convBiasBroadcast (outSpatial := outSpatial) bias .scalar)
        (Coord.linearize (outChannel, outIndex)) =
      biasBroadcastWith outSpatial bias (outChannel, outIndex) := by
  simp [convBiasBroadcast, biasBroadcastWith, getScalar_eq_apply,
    Tensor.generate, Tensor.item, Shape.Coord.toList, -Coord.linearize_cons, -Shape.size_cons]

private theorem _root_.TorchLean.Conv.grouped_affine
    {d inC outC : Nat} {kernel stride padding inSpatial : Tensor Nat [d]}
    (layer : Conv d inC outC kernel stride padding ℝ)
    (dilation paddingAfter : Tensor Nat [d]) (groups : Nat)
    (input : Tensor ℝ (inC :: inSpatial.data.toList))
    (outIndex : Shape.Coord (outC ::
      (outSpatialDilated inSpatial kernel stride dilation padding
        paddingAfter).data.toList)) :
    grouped (stride := stride) (dilation := dilation) (paddingBefore := padding)
        (paddingAfter := paddingAfter) groups layer.kernel layer.bias input outIndex =
      (∑ inIndex : Shape.Coord (inC :: inSpatial.data.toList),
        get2 (convLinearMatrix layer dilation paddingAfter groups .scalar)
          (Coord.linearize outIndex) (Coord.linearize inIndex) * input inIndex) +
      getScalar (convBiasBroadcast (outSpatial :=
        outSpatialDilated inSpatial kernel stride dilation padding paddingAfter)
          layer.bias .scalar) (Coord.linearize outIndex) := by
  rcases outIndex with ⟨outChannel, outSpatialIndex⟩
  simp only [grouped, add_def,
    Tensor.zipWith_apply]
  congr 1
  · unfold groupedContract
    rw [contractWith_eq_coefficients]
    apply Finset.sum_congr rfl
    rintro ⟨inChannel, inIndex⟩ _
    rw [convLinearMatrix_scalar_apply]
    exact mul_comm _ _
  · exact (convBiasBroadcast_scalar_apply layer.bias outChannel outSpatialIndex).symm

private def liftCoefficient {inShape outShape : Shape}
    (coefficient : outShape.Coord → inShape.Coord → ℝ) :
    (leading : Shape) → (leading ++ outShape).Coord →
      (leading ++ inShape).Coord → ℝ
  | .scalar, outIndex, inIndex => coefficient outIndex inIndex
  | .dim _ rest, (outHead, outTail), (inHead, inTail) =>
      if outHead = inHead then liftCoefficient coefficient rest outTail inTail else 0

private def liftBias {outShape : Shape} (bias : outShape.Coord → ℝ) :
    (leading : Shape) → (leading ++ outShape).Coord → ℝ
  | .scalar, outIndex => bias outIndex
  | .dim _ rest, (_, outTail) => liftBias bias rest outTail

private theorem mapLeading_affine {inShape outShape : Shape}
    (f : Tensor ℝ inShape → Tensor ℝ outShape)
    (coefficient : outShape.Coord → inShape.Coord → ℝ)
    (bias : outShape.Coord → ℝ)
    (hAffine : ∀ input outIndex,
      f input outIndex =
        (∑ inIndex : inShape.Coord, coefficient outIndex inIndex * input inIndex) +
          bias outIndex)
    (leading : Shape) (input : Tensor ℝ (leading ++ inShape))
    (outIndex : (leading ++ outShape).Coord) :
    Tensor.mapLeading leading f input outIndex =
      (∑ inIndex : (leading ++ inShape).Coord,
        liftCoefficient coefficient leading outIndex inIndex * input inIndex) +
          liftBias bias leading outIndex := by
  induction leading with
  | scalar => exact hAffine input outIndex
  | dim n rest ih =>
      rcases outIndex with ⟨head, tail⟩
      simp only [Tensor.mapLeading, Tensor.dim_apply,
        liftBias]
      rw [ih]
      congr 1
      erw [Fintype.sum_prod_type]
      symm
      rw [Finset.sum_eq_single head]
      · simp [liftCoefficient]
      · intro other _ hne
        have hOther : head ≠ other := Ne.symm hne
        simp [liftCoefficient, hOther]
      · simp

private theorem convLinearMatrix_leading
    {d inC outC : Nat} {kernel stride padding inSpatial : Tensor Nat [d]}
    (layer : Conv d inC outC kernel stride padding ℝ)
    (dilation paddingAfter : Tensor Nat [d]) (groups : Nat) (leading : Shape)
    (outIndex : (leading ++ (outC ::
      (outSpatialDilated inSpatial kernel stride dilation padding
        paddingAfter).data.toList)).Coord)
    (inIndex : (leading ++ (inC :: inSpatial.data.toList)).Coord) :
    get2 (convLinearMatrix layer dilation paddingAfter groups leading)
        (Coord.linearize outIndex) (Coord.linearize inIndex) =
      liftCoefficient
        (fun outCoord inCoord =>
          get2 (convLinearMatrix layer dilation paddingAfter groups .scalar)
            (Coord.linearize outCoord) (Coord.linearize inCoord))
        leading outIndex inIndex := by
  induction leading with
  | scalar => rfl
  | dim n rest ih =>
      rcases outIndex with ⟨outHead, outTail⟩
      rcases inIndex with ⟨inHead, inTail⟩
      by_cases hHead : outHead = inHead
      · subst inHead
        rw [liftCoefficient, ite_eq_left rfl, ← ih]
        simp [convLinearMatrix, get2_eq_apply,
          Tensor.dim_apply, Coord.unlinearize_linearize,
          Shape.Coord.toList, -Coord.linearize_cons, -Shape.size_cons]
      · rw [liftCoefficient, ite_eq_right hHead]
        have hValue : outHead.val ≠ inHead.val := fun h => hHead (Fin.ext h)
        simp [convLinearMatrix, get2_eq_apply,
          Tensor.dim_apply, Coord.unlinearize_linearize,
          Shape.Coord.toList, hValue, -Coord.linearize_cons, -Shape.size_cons]

private theorem convBiasBroadcast_leading
    {d outC : Nat} {outSpatial : Tensor Nat [d]} (bias : Tensor ℝ [outC])
    (leading : Shape)
    (outIndex : (leading ++ (outC :: outSpatial.data.toList)).Coord) :
    getScalar (convBiasBroadcast (outSpatial := outSpatial) bias leading)
        (Coord.linearize outIndex) =
      liftBias (fun outCoord =>
        getScalar (convBiasBroadcast (outSpatial := outSpatial) bias .scalar)
          (Coord.linearize outCoord)) leading outIndex := by
  induction leading with
  | scalar => rfl
  | dim n rest ih =>
      rcases outIndex with ⟨head, tail⟩
      rw [liftBias, ← ih]
      simp [convBiasBroadcast, getScalar_eq_apply,
        Tensor.dim_apply, Coord.unlinearize_linearize,
        Shape.Coord.toList, -Coord.linearize_cons, -Shape.size_cons]

private theorem matVec_flatten_apply {s : Shape} {m : Nat}
    (matrix : Tensor ℝ [m, s.size]) (input : Tensor ℝ s) (row : Fin m) :
    getScalar
        (Tensor.contract matrix (Tensor.reshape input [_] (Shape.size_vector _).symm) _ _ [m]
            .matVec) row =
      ∑ c : s.Coord, get2 matrix row (Coord.linearize c) * input c := by
  rw [Tensor.getScalar_contract_matVec_eq_sum, ← (Coord.equivFin s).sum_comp]
  simp only [Coord.equivFin_apply, Tensor.getScalar_reshape_toFin]

/--
The CROWN convolution matrix and broadcast bias compute the flattened real convolution.

The statement covers grouped channels, dilation, asymmetric zero padding and any leading batch
shape. It is an equality of the total mathematical definitions, with no hypothesis on the
configuration, which graph execution validates separately. It makes no floating-point
reduction-order claim.
-/
theorem convLinearMatrix_spec
    {d inC outC : Nat} {kernel stride padding inSpatial : Tensor Nat [d]}
    (layer : Conv d inC outC kernel stride padding ℝ)
    (dilation paddingAfter : Tensor Nat [d]) (groups : Nat) (leading : Shape)
    (input : Tensor ℝ (leading ++ (inC :: inSpatial.data.toList))) :
    (Tensor.contract (convLinearMatrix layer dilation paddingAfter groups leading)
          (Tensor.reshape input [_] (Shape.size_vector _).symm) _ _ [_] .matVec) +
        (convBiasBroadcast (outSpatial :=
          outSpatialDilated inSpatial kernel stride dilation padding paddingAfter)
          layer.bias leading) =
      Tensor.reshape (Tensor.mapLeading leading
        (grouped (stride := stride) (dilation := dilation) (paddingBefore := padding)
          (paddingAfter := paddingAfter) groups layer.kernel layer.bias) input) [_]
              (Shape.size_vector _).symm := by
  apply Tensor.ext_vector
  intro row
  let outShape := leading ++ (outC ::
    (outSpatialDilated inSpatial kernel stride dilation padding paddingAfter).data.toList)
  obtain ⟨outIndex, rfl⟩ := (Coord.equivFin outShape).surjective row
  change getScalar _ (Coord.linearize outIndex) =
    getScalar _ (Coord.linearize outIndex)
  simp only [getScalar_add]
  -- The matrix and the reshaped input spell the flattened input extent differently.
  erw [matVec_flatten_apply]
  simp only [Tensor.getScalar_reshape_toFin]
  rw [mapLeading_affine _ _ _ (grouped_affine layer dilation paddingAfter groups)]
  congr 1
  · apply Finset.sum_congr rfl
    intro inIndex _
    rw [convLinearMatrix_leading]
    rfl
  · exact convBiasBroadcast_leading layer.bias leading outIndex

end

end TorchLean.CROWN
