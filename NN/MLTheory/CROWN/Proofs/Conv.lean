/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Operators.Conv
public import NN.Proofs.Autograd.Tape.Ops.Conv.Index
public import NN.Proofs.Tensor.Basic.Core
public import NN.Spec.Core.Context.Real
public import NN.Tensor.Operations

/-!
# Real convolution and its affine representation

The coefficient matrix uses the same grouped channel and dilated spatial relation as the
convolution specification. The equalities in this module concern real arithmetic; they do not
assert agreement between different floating-point reduction orders.
-/

public section

namespace NN.MLTheory.CROWN

open Spec TorchLean TorchLean.Tensor
open Spec.Conv.Internal
open scoped BigOperators

namespace ConvProof

private def coordMultiIndexEquiv : (s : Shape) → s.Coord ≃ MultiIndex s.toList
  | .scalar => Equiv.refl _
  | .dim n rest =>
      Equiv.prodCongr (Equiv.refl (Fin n)) (coordMultiIndexEquiv rest)

private theorem coordMultiIndex_toList (s : Shape) (c : s.Coord) :
    (coordMultiIndexEquiv s c).toList = Shape.Coord.toList s c := by
  induction s with
  | scalar => rfl
  | dim n rest ih =>
      rcases c with ⟨head, tail⟩
      simp [coordMultiIndexEquiv, MultiIndex.toList, Shape.Coord.toList, ih]

private theorem coordMultiIndex_get (s : Shape) (x : Tensor ℝ s) (c : s.Coord) :
    (coordMultiIndexEquiv s c).get x = x c := by
  induction s with
  | scalar =>
      cases c
      rfl
  | dim n rest ih =>
      rcases c with ⟨head, tail⟩
      change (coordMultiIndexEquiv rest tail).get (x.unstack head) = x (head, tail)
      rw [ih]
      exact TorchLean.Tensor.Internal.Rep.unstack_apply x head tail

private theorem getAtOrZero_eq_coord_sum (s : Shape) (x : Tensor ℝ s)
    (indices : List Nat) :
    getAtOrZero x indices =
      ∑ c : s.Coord, if indices = Shape.Coord.toList s c then x c else 0 := by
  rw [getAtOrZero_eq_sum_indicator s.toList x indices,
    ← (coordMultiIndexEquiv s).sum_comp]
  simp only [coordMultiIndex_toList, coordMultiIndex_get]

private theorem foldlIndices_eq_coord_sum (s : Shape) (init : ℝ) (f : List Nat → ℝ) :
    foldlIndices s.toList init (fun acc indices => acc + f indices) =
      init + ∑ c : s.Coord, f (Shape.Coord.toList s c) := by
  rw [foldlIndices_add, ← (coordMultiIndexEquiv s).sum_comp]
  simp only [coordMultiIndex_toList]

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
  simp_rw [getAtOrZero_eq_coord_sum (.dim inC s) input, Finset.sum_mul]
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

private theorem convCoreWith_apply
    {d inC outC : Nat} {kernel inSpatial outSpatial : Tensor Nat [d]}
    (count : Nat) (inputChannel : Fin outC → Fin count → Nat)
    (inputIndex? : List Nat → List Nat → Option (List Nat))
    (weights : Tensor ℝ (Shape.ofList (outC :: inC :: kernel.data.toList)))
    (input : Tensor ℝ (Shape.ofList (inC :: inSpatial.data.toList)))
    (outChannel : Fin outC) (outIndex : (Shape.ofList outSpatial.data.toList).Coord) :
    convCoreWith count inputChannel inputIndex? weights input (outChannel, outIndex) =
      ∑ channelIndex : Fin count, ∑ kernelIndex : (Shape.ofList kernel.data.toList).Coord,
        (match inputIndex? (Shape.Coord.toList _ outIndex)
            (Shape.Coord.toList _ kernelIndex) with
        | none => 0
        | some indices => getAtOrZero input (inputChannel outChannel channelIndex :: indices)) *
          getAtOrZero weights (outChannel.val :: inputChannel outChannel channelIndex ::
            Shape.Coord.toList _ kernelIndex) := by
  simp only [convCoreWith, Tensor.dim, TorchLean.Tensor.Internal.Rep.stack_apply,
    Tensor.generate, TorchLean.Tensor.Internal.Rep.get_ofFn]
  simp_rw [foldlIndices_eq_coord_sum (Shape.ofList kernel.data.toList)]
  exact List.finRange_foldl_add_eq_finset_sum _

private theorem convCoreWith_eq_coefficients
    {d inC outC : Nat} {kernel inSpatial outSpatial : Tensor Nat [d]}
    (count : Nat) (start : Fin outC → Nat)
    (inputIndex? : List Nat → List Nat → Option (List Nat))
    (weights : Tensor ℝ (Shape.ofList (outC :: inC :: kernel.data.toList)))
    (input : Tensor ℝ (Shape.ofList (inC :: inSpatial.data.toList)))
    (outChannel : Fin outC) (outIndex : (Shape.ofList outSpatial.data.toList).Coord) :
    convCoreWith count (fun channel i => start channel + i.val) inputIndex?
        weights input (outChannel, outIndex) =
      ∑ c : (Shape.ofList (inC :: inSpatial.data.toList)).Coord, input c *
        (if start outChannel ≤ c.1.val ∧ c.1.val < start outChannel + count then
          ∑ k : (Shape.ofList kernel.data.toList).Coord,
            if inputIndex? (Shape.Coord.toList _ outIndex) (Shape.Coord.toList _ k) =
                some (Shape.Coord.toList _ c.2) then
              getAtOrZero weights (outChannel.val :: c.1.val :: Shape.Coord.toList _ k)
            else 0
        else 0) := by
  rw [convCoreWith_apply, Finset.sum_comm]
  calc
    _ = ∑ k : (Shape.ofList kernel.data.toList).Coord,
        ∑ c : (Shape.ofList (inC :: inSpatial.data.toList)).Coord, input c *
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
            channel_lookup_sum (Shape.ofList inSpatial.data.toList) input count
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
    foldlIndices s.toList init
        (fun acc indices => if p indices then acc + weight indices else acc) =
      init + ∑ k : s.Coord, if p (Shape.Coord.toList s k) then
        weight (Shape.Coord.toList s k) else 0 := by
  simpa only [add_ite, add_zero] using
    foldlIndices_eq_coord_sum s init (fun indices => if p indices then weight indices else 0)

private theorem convLinearMatrix_scalar_apply
    {d inC outC : Nat} {kernel stride padding inSpatial : Tensor Nat [d]}
    (layer : ConvSpec d inC outC kernel stride padding ℝ)
    (dilation paddingAfter : Tensor Nat [d]) (groups : Nat)
    (outChannel : Fin outC)
    (outIndex : (Shape.ofList
      (convOutSpatialDilated inSpatial kernel stride dilation padding
        paddingAfter).data.toList).Coord)
    (inChannel : Fin inC) (inIndex : (Shape.ofList inSpatial.data.toList).Coord) :
    get2 (convLinearMatrix layer dilation paddingAfter groups .scalar)
        (Shape.Coord.linearize (outChannel, outIndex))
        (Shape.Coord.linearize (inChannel, inIndex)) =
      let start := (outChannel.val / (outC / groups)) * (inC / groups)
      if start ≤ inChannel.val ∧ inChannel.val < start + inC / groups then
        ∑ k : (Shape.ofList kernel.data.toList).Coord,
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
    simp [convLinearMatrix, get2_eq_apply, Tensor.dim, Shape.Coord.toList, Shape.rank,
      hLower, hUpper, hRange, foldlIndices_filter]
  · have hOutside : inChannel.val < start ∨ start + inC / groups ≤ inChannel.val := by
      omega
    rcases hOutside with hLower | hUpper
    · dsimp only [start] at hRange hLower
      simp [convLinearMatrix, get2_eq_apply, Tensor.dim, Shape.Coord.toList, Shape.rank,
        hLower, hRange]
    · dsimp only [start] at hRange hUpper
      simp [convLinearMatrix, get2_eq_apply, Tensor.dim, Shape.Coord.toList, Shape.rank,
        hUpper, hRange]

private theorem convBiasBroadcast_scalar_apply
    {d outC : Nat} {outSpatial : Tensor Nat [d]} (bias : Tensor ℝ [outC])
    (outChannel : Fin outC) (outIndex : (Shape.ofList outSpatial.data.toList).Coord) :
    getScalar (convBiasBroadcast (outSpatial := outSpatial) bias .scalar)
        (Shape.Coord.linearize (outChannel, outIndex)) =
      convBiasBroadcastWith outSpatial bias (outChannel, outIndex) := by
  simp [convBiasBroadcast, convBiasBroadcastWith, getScalar_eq_apply, Tensor.dim,
    Tensor.generate, Tensor.item, Shape.Coord.toList, Shape.rank]

private theorem groupedConvSpec_affine
    {d inC outC : Nat} {kernel stride padding inSpatial : Tensor Nat [d]}
    (layer : ConvSpec d inC outC kernel stride padding ℝ)
    (dilation paddingAfter : Tensor Nat [d]) (groups : Nat)
    (input : Tensor ℝ (Shape.ofList (inC :: inSpatial.data.toList)))
    (outIndex : (Shape.ofList (outC ::
      (convOutSpatialDilated inSpatial kernel stride dilation padding
        paddingAfter).data.toList)).Coord) :
    groupedConvSpec (stride := stride) (dilation := dilation) (paddingBefore := padding)
        (paddingAfter := paddingAfter) groups layer.kernel layer.bias input outIndex =
      (∑ inIndex : (Shape.ofList (inC :: inSpatial.data.toList)).Coord,
        get2 (convLinearMatrix layer dilation paddingAfter groups .scalar)
          (Shape.Coord.linearize outIndex) (Shape.Coord.linearize inIndex) * input inIndex) +
      getScalar (convBiasBroadcast (outSpatial :=
        convOutSpatialDilated inSpatial kernel stride dilation padding paddingAfter)
          layer.bias .scalar) (Shape.Coord.linearize outIndex) := by
  rcases outIndex with ⟨outChannel, outSpatialIndex⟩
  simp only [groupedConvSpec, addSpec, map2Spec,
    TorchLean.Tensor.Internal.Rep.zipWith_apply]
  congr 1
  · unfold groupedConvCoreSpec
    rw [convCoreWith_eq_coefficients]
    apply Finset.sum_congr rfl
    rintro ⟨inChannel, inIndex⟩ _
    rw [convLinearMatrix_scalar_apply]
    exact mul_comm _ _
  · exact (convBiasBroadcast_scalar_apply layer.bias outChannel outSpatialIndex).symm

private def liftCoefficient {inShape outShape : Shape}
    (coefficient : outShape.Coord → inShape.Coord → ℝ) :
    (leading : Shape) → (leading.concat outShape).Coord →
      (leading.concat inShape).Coord → ℝ
  | .scalar, outIndex, inIndex => coefficient outIndex inIndex
  | .dim _ rest, (outHead, outTail), (inHead, inTail) =>
      if outHead = inHead then liftCoefficient coefficient rest outTail inTail else 0

private def liftBias {outShape : Shape} (bias : outShape.Coord → ℝ) :
    (leading : Shape) → (leading.concat outShape).Coord → ℝ
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
    (leading : Shape) (input : Tensor ℝ (leading.concat inShape))
    (outIndex : (leading.concat outShape).Coord) :
    Tensor.mapLeading leading f input outIndex =
      (∑ inIndex : (leading.concat inShape).Coord,
        liftCoefficient coefficient leading outIndex inIndex * input inIndex) +
          liftBias bias leading outIndex := by
  induction leading with
  | scalar => exact hAffine input outIndex
  | dim n rest ih =>
      rcases outIndex with ⟨head, tail⟩
      simp only [Tensor.mapLeading, Tensor.dim, TorchLean.Tensor.Internal.Rep.stack_apply,
        liftBias]
      rw [ih]
      congr 1
      rw [Fintype.sum_prod_type]
      symm
      rw [Finset.sum_eq_single head]
      · simp [liftCoefficient, Tensor.unstack]
      · intro other _ hne
        have hOther : head ≠ other := Ne.symm hne
        simp [liftCoefficient, hOther]
      · simp

private theorem convLinearMatrix_leading
    {d inC outC : Nat} {kernel stride padding inSpatial : Tensor Nat [d]}
    (layer : ConvSpec d inC outC kernel stride padding ℝ)
    (dilation paddingAfter : Tensor Nat [d]) (groups : Nat) (leading : Shape)
    (outIndex : (leading.concat (Shape.ofList (outC ::
      (convOutSpatialDilated inSpatial kernel stride dilation padding
        paddingAfter).data.toList))).Coord)
    (inIndex : (leading.concat (Shape.ofList (inC :: inSpatial.data.toList))).Coord) :
    get2 (convLinearMatrix layer dilation paddingAfter groups leading)
        (Shape.Coord.linearize outIndex) (Shape.Coord.linearize inIndex) =
      liftCoefficient
        (fun outCoord inCoord =>
          get2 (convLinearMatrix layer dilation paddingAfter groups .scalar)
            (Shape.Coord.linearize outCoord) (Shape.Coord.linearize inCoord))
        leading outIndex inIndex := by
  induction leading with
  | scalar => rfl
  | dim n rest ih =>
      rcases outIndex with ⟨outHead, outTail⟩
      rcases inIndex with ⟨inHead, inTail⟩
      by_cases hHead : outHead = inHead
      · subst inHead
        rw [liftCoefficient, ite_eq_left rfl, ← ih]
        simp [convLinearMatrix, get2_eq_apply, Tensor.dim,
          TorchLean.Tensor.Internal.Rep.stack_apply, Shape.Coord.unlinearize_linearize,
          Shape.Coord.toList, Shape.rank]
      · rw [liftCoefficient, ite_eq_right hHead]
        have hValue : outHead.val ≠ inHead.val := fun h => hHead (Fin.ext h)
        simp [convLinearMatrix, get2_eq_apply, Tensor.dim,
          TorchLean.Tensor.Internal.Rep.stack_apply, Shape.Coord.unlinearize_linearize,
          Shape.Coord.toList, Shape.rank, hValue]

private theorem convBiasBroadcast_leading
    {d outC : Nat} {outSpatial : Tensor Nat [d]} (bias : Tensor ℝ [outC])
    (leading : Shape)
    (outIndex : (leading.concat (Shape.ofList (outC :: outSpatial.data.toList))).Coord) :
    getScalar (convBiasBroadcast (outSpatial := outSpatial) bias leading)
        (Shape.Coord.linearize outIndex) =
      liftBias (fun outCoord =>
        getScalar (convBiasBroadcast (outSpatial := outSpatial) bias .scalar)
          (Shape.Coord.linearize outCoord)) leading outIndex := by
  induction leading with
  | scalar => rfl
  | dim n rest ih =>
      rcases outIndex with ⟨head, tail⟩
      rw [liftBias, ← ih]
      simp [convBiasBroadcast, getScalar_eq_apply, Tensor.dim,
        TorchLean.Tensor.Internal.Rep.stack_apply, Shape.Coord.unlinearize_linearize,
        Shape.Coord.toList, Shape.rank]

private theorem matVec_flatten_apply {s : Shape} {m : Nat}
    (matrix : Tensor ℝ [m, s.size]) (input : Tensor ℝ s) (row : Fin m) :
    getScalar (matVecMulSpec matrix (flattenSpec input)) row =
      ∑ c : s.Coord, get2 matrix row (Shape.Coord.linearize c) * input c := by
  rw [Proofs.TensorAlgebra.getScalar_mat_vec_mul_spec, ← (Shape.Coord.equivFin s).sum_comp]
  simp only [Shape.Coord.equivFin_apply, Spec.getScalar_flattenSpec_linearize]

/--
The CROWN convolution matrix and broadcast bias compute the flattened real convolution.

The statement includes grouped channels, dilation, asymmetric zero padding, and any leading batch
shape. It is an equality of the total mathematical definitions; graph execution additionally
validates the convolution configuration. It makes no floating-point reduction-order claim.
-/
theorem conv_linear_matrix_add_bias_eq_grouped_conv
    {d inC outC : Nat} {kernel stride padding inSpatial : Tensor Nat [d]}
    (layer : ConvSpec d inC outC kernel stride padding ℝ)
    (dilation paddingAfter : Tensor Nat [d]) (groups : Nat) (leading : Shape)
    (input : Tensor ℝ (leading.concat (Shape.ofList (inC :: inSpatial.data.toList)))) :
    addSpec
        (matVecMulSpec (convLinearMatrix layer dilation paddingAfter groups leading)
          (flattenSpec input))
        (convBiasBroadcast (outSpatial :=
          convOutSpatialDilated inSpatial kernel stride dilation padding paddingAfter)
          layer.bias leading) =
      flattenSpec (Tensor.mapLeading leading
        (groupedConvSpec (stride := stride) (dilation := dilation) (paddingBefore := padding)
          (paddingAfter := paddingAfter) groups layer.kernel layer.bias) input) := by
  apply Tensor.ext_vector
  intro row
  let outShape := leading.concat (Shape.ofList (outC ::
    (convOutSpatialDilated inSpatial kernel stride dilation padding paddingAfter).data.toList))
  obtain ⟨outIndex, rfl⟩ := (Shape.Coord.equivFin outShape).surjective row
  change getScalar _ (Shape.Coord.linearize outIndex) =
    getScalar _ (Shape.Coord.linearize outIndex)
  simp only [getScalar_add_spec, matVec_flatten_apply, Spec.getScalar_flattenSpec_linearize]
  rw [mapLeading_affine _ _ _ (groupedConvSpec_affine layer dilation paddingAfter groups)]
  congr 1
  · apply Finset.sum_congr rfl
    intro inIndex _
    rw [convLinearMatrix_leading]
    rfl
  · exact convBiasBroadcast_leading layer.bias leading outIndex

end ConvProof

end NN.MLTheory.CROWN
