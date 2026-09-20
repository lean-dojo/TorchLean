/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Tape.Ops.Conv.Index
public import NN.Tensor.Conversion
public import Mathlib.Analysis.Calculus.FDeriv.Add
public import Mathlib.Analysis.Calculus.FDeriv.Bilinear
public import Mathlib.Analysis.InnerProductSpace.PiL2
public import NN.Spec.Core.Context.Real

/-!
# Derivative of General Convolution

This file proves the derivative and reverse-mode formulas for channels-first convolution at an
arbitrary spatial rank.  All sums use bounded multi-indices; no axis count is fixed in a theorem.
-/

@[expose] public section

namespace Proofs.Autograd.Conv

open scoped BigOperators
open Spec TorchLean
open TorchLean.Tensor
open Spec.Conv.Internal
open Proofs.TensorAlgebra

noncomputable section

/-- Read a channel and bounded spatial coordinate from a channels-first tensor. -/
def channelGet {α : Type} [TorchLean.Storage α] {channels : Nat} {dims : List Nat}
    (x : Tensor α (Shape.ofList (channels :: dims)))
    (channel : Fin channels) (i : MultiIndex dims) : α :=
  MultiIndex.get x (channel, i)

/-- Read two leading channels followed by a bounded spatial coordinate. -/
def channelPairGet {α : Type} [TorchLean.Storage α] {outer inner : Nat} {dims : List Nat}
    (x : Tensor α (Shape.ofList (outer :: inner :: dims)))
    (i : Fin outer) (j : Fin inner) (k : MultiIndex dims) : α :=
  MultiIndex.get x (i, (j, k))

/-- Split a channels-first tensor dot product into channel and spatial sums. -/
theorem dot_eq_sum_channel {α : Type} [TorchLean.Storage α] [CommSemiring α]
    (channels : Nat) (dims : List Nat)
    (x y : Tensor α (Shape.ofList (channels :: dims))) :
    TensorAlgebra.dot x y =
      ∑ channel : Fin channels, ∑ i : MultiIndex dims,
        channelGet x channel i * channelGet y channel i := by
  rw [dot_eq_sum_get, MultiIndex.sum_cons]
  rfl

/-- Expand a channels-first lookup into the unique bounded spatial coordinate that it names. -/
theorem getAtOrZero_channel_eq_sum_indicator {α : Type} [TorchLean.Storage α] [AddCommMonoid α]
    {channels : Nat} {dims : List Nat}
    (x : Tensor α (Shape.ofList (channels :: dims)))
    (channel : Fin channels) (indices : List Nat) :
    getAtOrZero x (channel.val :: indices) =
      ∑ i : MultiIndex dims,
        if indices = i.toList then channelGet x channel i else 0 := by
  rw [show x = Tensor.dim (Tensor.unstack x) from (Tensor.dim_unstack x).symm]
  rw [getAtOrZero_dim_cons]
  simp only [channel.isLt, ↓reduceDIte]
  rw [getAtOrZero_eq_sum_indicator]
  apply Finset.sum_congr rfl
  intro i _
  simp only [channelGet, MultiIndex.get_dim]

/-- The scalar coefficient connecting one input coordinate to one output coordinate. -/
def convCoefficient
    {d inC outC : Nat}
    {kernel stride padding inSpatial : TorchLean.Tensor Nat [d]}
    (weights : Tensor ℝ (Shape.ofList (outC :: inC :: (Tensor.to kernel (List Nat)))))
    (outCh : Fin outC)
    (outIdx : MultiIndex (Tensor.to (convOutSpatial inSpatial kernel stride padding) (List Nat)))
    (inCh : Fin inC) (inIdx : MultiIndex (Tensor.to inSpatial (List Nat))) : ℝ :=
  ∑ kIdx : MultiIndex (Tensor.to kernel (List Nat)),
    if mkInputIdx? outIdx.toList kIdx.toList (Tensor.to stride (List Nat))
        (Tensor.to padding (List Nat)) =
        some inIdx.toList then
      channelPairGet weights outCh inCh kIdx
    else
      0

/-- One coordinate of the generic convolution contraction, written as finite sums. -/
theorem channelGet_convCoreSpec
    {d inC outC : Nat}
    {kernel stride padding inSpatial : TorchLean.Tensor Nat [d]}
    (weights : Tensor ℝ (Shape.ofList (outC :: inC :: (Tensor.to kernel (List Nat)))))
    (input : Tensor ℝ (Shape.ofList (inC :: (Tensor.to inSpatial (List Nat)))))
    (outCh : Fin outC)
    (outIdx : MultiIndex (Tensor.to (convOutSpatial inSpatial kernel stride padding) (List Nat))) :
    channelGet (convCoreSpec weights input) outCh outIdx =
      ∑ inCh : Fin inC, ∑ kIdx : MultiIndex (Tensor.to kernel (List Nat)),
        (match mkInputIdx? outIdx.toList kIdx.toList (Tensor.to stride (List Nat))
            (Tensor.to padding (List Nat)) with
          | none => 0
          | some inIdx => getAtOrZero input (inCh.val :: inIdx)) *
        channelPairGet weights outCh inCh kIdx := by
  simp only [channelGet, convCoreSpec, Conv.Internal.convCoreWith]
  rw [MultiIndex.get_dim, MultiIndex.get_generate]
  simp_rw [foldlIndices_add]
  rw [List.finRange_foldl_add_eq_finset_sum]
  apply Finset.sum_congr rfl
  intro inCh _
  apply Finset.sum_congr rfl
  intro kIdx _
  have hWeight := getAtOrZero_toList
    (dims := outC :: inC :: (Tensor.to kernel (List Nat))) weights (outCh, (inCh, kIdx))
  have hWeight' :
      getAtOrZero weights (outCh.val :: inCh.val :: kIdx.toList) =
        MultiIndex.get weights (outCh, (inCh, kIdx)) := by
    simpa only [MultiIndex.toList] using hWeight
  change
    (match mkInputIdx? outIdx.toList kIdx.toList (Tensor.to stride (List Nat))
        (Tensor.to padding (List Nat)) with
      | none => 0
      | some inIdx => getAtOrZero input (inCh.val :: inIdx)) *
        getAtOrZero weights (outCh.val :: inCh.val :: kIdx.toList) =
      (match mkInputIdx? outIdx.toList kIdx.toList (Tensor.to stride (List Nat))
          (Tensor.to padding (List Nat)) with
        | none => 0
        | some inIdx => getAtOrZero input (inCh.val :: inIdx)) *
        MultiIndex.get weights (outCh, (inCh, kIdx))
  rw [hWeight']

/-- Convolution is the matrix represented by `convCoefficient` at every spatial rank. -/
theorem channelGet_convCoreSpec_eq_coefficients
    {d inC outC : Nat}
    {kernel stride padding inSpatial : TorchLean.Tensor Nat [d]}
    (weights : Tensor ℝ (Shape.ofList (outC :: inC :: (Tensor.to kernel (List Nat)))))
    (input : Tensor ℝ (Shape.ofList (inC :: (Tensor.to inSpatial (List Nat)))))
    (outCh : Fin outC)
    (outIdx : MultiIndex (Tensor.to (convOutSpatial inSpatial kernel stride padding) (List Nat))) :
    channelGet (convCoreSpec weights input) outCh outIdx =
      ∑ inCh : Fin inC, ∑ inIdx : MultiIndex (Tensor.to inSpatial (List Nat)),
        channelGet input inCh inIdx *
          convCoefficient weights outCh outIdx inCh inIdx := by
  rw [channelGet_convCoreSpec]
  apply Finset.sum_congr rfl
  intro inCh _
  unfold convCoefficient
  calc
    (∑ kIdx : MultiIndex (Tensor.to kernel (List Nat)),
        (match mkInputIdx? outIdx.toList kIdx.toList (Tensor.to stride (List Nat))
            (Tensor.to padding (List Nat)) with
          | none => 0
          | some inputIdx => getAtOrZero input (inCh.val :: inputIdx)) *
          channelPairGet weights outCh inCh kIdx) =
      ∑ kIdx : MultiIndex (Tensor.to kernel (List Nat)),
        ∑ inIdx : MultiIndex (Tensor.to inSpatial (List Nat)),
          channelGet input inCh inIdx *
            (if mkInputIdx? outIdx.toList kIdx.toList (Tensor.to stride (List Nat))
                (Tensor.to padding (List Nat)) =
                some inIdx.toList then
              channelPairGet weights outCh inCh kIdx
            else 0) := by
        apply Finset.sum_congr rfl
        intro kIdx _
        cases hInput : mkInputIdx? outIdx.toList kIdx.toList (Tensor.to stride (List Nat))
            (Tensor.to padding (List Nat)) with
        | none => simp
        | some inputIdx =>
            change getAtOrZero input (inCh.val :: inputIdx) *
                channelPairGet weights outCh inCh kIdx =
              ∑ inIdx : MultiIndex (Tensor.to inSpatial (List Nat)),
                channelGet input inCh inIdx *
                  (if some inputIdx = some inIdx.toList then
                    channelPairGet weights outCh inCh kIdx
                  else 0)
            rw [getAtOrZero_channel_eq_sum_indicator input inCh inputIdx]
            rw [Finset.sum_mul]
            apply Finset.sum_congr rfl
            intro inIdx _
            by_cases hEq : inputIdx = inIdx.toList <;> simp [hEq]
    _ =
      ∑ inIdx : MultiIndex (Tensor.to inSpatial (List Nat)),
        ∑ kIdx : MultiIndex (Tensor.to kernel (List Nat)),
          channelGet input inCh inIdx *
            (if mkInputIdx? outIdx.toList kIdx.toList (Tensor.to stride (List Nat))
                (Tensor.to padding (List Nat)) =
                some inIdx.toList then
              channelPairGet weights outCh inCh kIdx
            else 0) := by
        rw [Finset.sum_comm]
    _ =
      ∑ inIdx : MultiIndex (Tensor.to inSpatial (List Nat)),
        channelGet input inCh inIdx *
          ∑ kIdx : MultiIndex (Tensor.to kernel (List Nat)),
            if mkInputIdx? outIdx.toList kIdx.toList (Tensor.to stride (List Nat))
                (Tensor.to padding (List Nat)) =
                some inIdx.toList then
              channelPairGet weights outCh inCh kIdx
            else 0 := by
        apply Finset.sum_congr rfl
        intro inIdx _
        rw [Finset.mul_sum]

/-- One kernel-gradient coordinate is the contraction of input and output cotangent. -/
theorem channelPairGet_convKernelDerivSpec
    {d inC outC : Nat}
    {kernel stride padding inSpatial : TorchLean.Tensor Nat [d]}
    (layer : ConvSpec d inC outC kernel stride padding ℝ)
    (input : Tensor ℝ (Shape.ofList (inC :: (Tensor.to inSpatial (List Nat)))))
    (gradOutput : Tensor ℝ
      (Shape.ofList
        (outC :: (Tensor.to (convOutSpatial inSpatial kernel stride padding) (List Nat)))))
    (outCh : Fin outC) (inCh : Fin inC) (kIdx : MultiIndex (Tensor.to kernel (List Nat))) :
    channelPairGet (convKernelDerivSpec layer input gradOutput) outCh inCh kIdx =
      ∑ outIdx : MultiIndex (Tensor.to (convOutSpatial inSpatial kernel stride padding) (List Nat)),
        (match mkInputIdx? outIdx.toList kIdx.toList (Tensor.to stride (List Nat))
            (Tensor.to padding (List Nat)) with
          | none => 0
          | some inIdx => getAtOrZero input (inCh.val :: inIdx)) *
        channelGet gradOutput outCh outIdx := by
  simp only [channelPairGet, convKernelDerivSpec]
  simp only [MultiIndex.get, Tensor.unstack_dim]
  rw [MultiIndex.get_generate, foldlIndices_add]
  simp only [zero_add]
  apply Finset.sum_congr rfl
  intro outIdx _
  have hGrad := getAtOrZero_toList
    (dims := outC :: (Tensor.to (convOutSpatial inSpatial kernel stride padding) (List Nat)))
    gradOutput (outCh, outIdx)
  have hGrad' :
      getAtOrZero gradOutput (outCh.val :: outIdx.toList) =
        MultiIndex.get gradOutput (outCh, outIdx) := by
    simpa only [MultiIndex.toList] using hGrad
  change
    (match mkInputIdx? outIdx.toList kIdx.toList (Tensor.to stride (List Nat))
        (Tensor.to padding (List Nat)) with
      | none => 0
      | some inputIdx => getAtOrZero input (inCh.val :: inputIdx)) *
        getAtOrZero gradOutput (outCh.val :: outIdx.toList) =
      (match mkInputIdx? outIdx.toList kIdx.toList (Tensor.to stride (List Nat))
          (Tensor.to padding (List Nat)) with
        | none => 0
        | some inputIdx => getAtOrZero input (inCh.val :: inputIdx)) *
        MultiIndex.get gradOutput (outCh, outIdx)
  rw [hGrad']

/-- One bias-gradient coordinate is the spatial sum of the output cotangent. -/
theorem channelGet_convBiasDerivSpec
    {d inC outC : Nat}
    {kernel stride padding inSpatial : TorchLean.Tensor Nat [d]}
    (layer : ConvSpec d inC outC kernel stride padding ℝ)
    (input : Tensor ℝ (Shape.ofList (inC :: (Tensor.to inSpatial (List Nat)))))
    (gradOutput : Tensor ℝ
      (Shape.ofList
        (outC :: (Tensor.to (convOutSpatial inSpatial kernel stride padding) (List Nat)))))
    (outCh : Fin outC) :
    channelGet (dims := []) (convBiasDerivSpec layer input gradOutput) outCh PUnit.unit =
      ∑ outIdx : MultiIndex (Tensor.to (convOutSpatial inSpatial kernel stride padding) (List Nat)),
        channelGet gradOutput outCh outIdx := by
  simp only [channelGet, convBiasDerivSpec]
  rw [MultiIndex.get_vector_eq_getScalar, TorchLean.Tensor.getScalar_dim]
  rw [foldlIndices_add]
  simp only [zero_add]
  apply Finset.sum_congr rfl
  intro outIdx _
  have hGrad := getAtOrZero_toList
    (dims := outC :: (Tensor.to (convOutSpatial inSpatial kernel stride padding) (List Nat)))
    gradOutput (outCh, outIdx)
  change getAtOrZero gradOutput (outCh.val :: outIdx.toList) =
    MultiIndex.get gradOutput (outCh, outIdx)
  simpa only [MultiIndex.toList] using hGrad

/-- A bias-gradient coordinate in the ordinary rank-one tensor view. -/
theorem getScalar_convBiasDerivSpec
    {d inC outC : Nat}
    {kernel stride padding inSpatial : TorchLean.Tensor Nat [d]}
    (layer : ConvSpec d inC outC kernel stride padding ℝ)
    (input : Tensor ℝ (Shape.ofList (inC :: (Tensor.to inSpatial (List Nat)))))
    (gradOutput : Tensor ℝ
      (Shape.ofList
        (outC :: (Tensor.to (convOutSpatial inSpatial kernel stride padding) (List Nat)))))
    (outCh : Fin outC) :
    TorchLean.Tensor.getScalar (convBiasDerivSpec layer input gradOutput) outCh =
      ∑ outIdx : MultiIndex (Tensor.to (convOutSpatial inSpatial kernel stride padding) (List Nat)),
        channelGet gradOutput outCh outIdx := by
  simp only [convBiasDerivSpec, TorchLean.Tensor.getScalar_dim]
  rw [foldlIndices_add]
  simp only [zero_add]
  apply Finset.sum_congr rfl
  intro outIdx _
  change getAtOrZero gradOutput (outCh.val :: outIdx.toList) =
    MultiIndex.get (dims := outC ::
      (Tensor.to (convOutSpatial inSpatial kernel stride padding) (List Nat)))
      gradOutput (outCh, outIdx)
  have hGrad := getAtOrZero_toList
    (dims := outC :: (Tensor.to (convOutSpatial inSpatial kernel stride padding) (List Nat)))
    gradOutput (outCh, outIdx)
  convert hGrad using 1
  rfl

/-- Broadcasting a bias reads the same channel value at every spatial coordinate. -/
theorem channelGet_convBiasBroadcastSpec
    {d outC : Nat}
    {kernel stride padding inSpatial : TorchLean.Tensor Nat [d]}
    (bias : Tensor ℝ [outC])
    (outCh : Fin outC)
    (outIdx : MultiIndex (Tensor.to (convOutSpatial inSpatial kernel stride padding) (List Nat))) :
    channelGet (convBiasBroadcastSpec (kernel := kernel) (stride := stride)
      (padding := padding) (inSpatial := inSpatial) bias) outCh outIdx =
        TorchLean.Tensor.getScalar bias outCh := by
  simp only [channelGet, convBiasBroadcastSpec]
  change MultiIndex.get (dims := outC ::
      (Tensor.to (convOutSpatial inSpatial kernel stride padding) (List Nat)))
    (Tensor.dim fun outCh =>
      TorchLean.Tensor.generate
        (Tensor.to (convOutSpatial inSpatial kernel stride padding) (List Nat))
        fun _ => getAtOrZero bias [outCh.val]) (outCh, outIdx) = _
  rw [MultiIndex.get_dim, MultiIndex.get_generate]
  change getAtOrZero bias [outCh.val] = (bias.unstack outCh).item
  simp

/-- One input-gradient coordinate is the transpose-index convolution used by the runtime. -/
theorem channelGet_convInputDerivSpec
    {d inC outC : Nat}
    {kernel stride padding inSpatial : TorchLean.Tensor Nat [d]}
    (layer : ConvSpec d inC outC kernel stride padding ℝ)
    (input : Tensor ℝ (Shape.ofList (inC :: (Tensor.to inSpatial (List Nat)))))
    (gradOutput : Tensor ℝ
      (Shape.ofList
        (outC :: (Tensor.to (convOutSpatial inSpatial kernel stride padding) (List Nat)))))
    (inCh : Fin inC) (inIdx : MultiIndex (Tensor.to inSpatial (List Nat))) :
    channelGet (convInputDerivSpec layer input gradOutput) inCh inIdx =
      ∑ outCh : Fin outC, ∑ kIdx : MultiIndex (Tensor.to kernel (List Nat)),
        (match mkTransposeInputIdx? inIdx.toList kIdx.toList (Tensor.to stride (List Nat))
            (Tensor.to padding (List Nat)) with
          | none => 0
          | some outIdx =>
              getAtOrZero gradOutput (outCh.val :: outIdx) *
                channelPairGet layer.kernel outCh inCh kIdx) := by
  simp only [channelGet, convInputDerivSpec]
  rw [MultiIndex.get_dim, MultiIndex.get_generate]
  simp_rw [foldlIndices_add]
  rw [List.finRange_foldl_add_eq_finset_sum]
  apply Finset.sum_congr rfl
  intro outCh _
  apply Finset.sum_congr rfl
  intro kIdx _
  have hWeight := getAtOrZero_toList
    (dims := outC :: inC :: (Tensor.to kernel (List Nat))) layer.kernel (outCh, (inCh, kIdx))
  have hWeight' :
      getAtOrZero layer.kernel (outCh.val :: inCh.val :: kIdx.toList) =
        MultiIndex.get layer.kernel (outCh, (inCh, kIdx)) := by
    simpa only [MultiIndex.toList] using hWeight
  split <;> simp_all [channelPairGet]

/-- The implemented input gradient is multiplication by the transposed coefficient matrix. -/
theorem channelGet_convInputDerivSpec_eq_coefficients
    {d inC outC : Nat}
    {kernel stride padding inSpatial : TorchLean.Tensor Nat [d]}
    (layer : ConvSpec d inC outC kernel stride padding ℝ)
    (input : Tensor ℝ (Shape.ofList (inC :: (Tensor.to inSpatial (List Nat)))))
    (gradOutput : Tensor ℝ
      (Shape.ofList
        (outC :: (Tensor.to (convOutSpatial inSpatial kernel stride padding) (List Nat)))))
    (hStride : PositiveStrides (Tensor.to stride (List Nat)))
    (inCh : Fin inC) (inIdx : MultiIndex (Tensor.to inSpatial (List Nat))) :
    channelGet (convInputDerivSpec layer input gradOutput) inCh inIdx =
      ∑ outCh : Fin outC,
        ∑ outIdx : MultiIndex
            (Tensor.to (convOutSpatial inSpatial kernel stride padding) (List Nat)),
          convCoefficient layer.kernel outCh outIdx inCh inIdx *
            channelGet gradOutput outCh outIdx := by
  rw [channelGet_convInputDerivSpec]
  apply Finset.sum_congr rfl
  intro outCh _
  unfold convCoefficient
  simp_rw [Finset.sum_mul]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro kIdx _
  cases hTranspose : mkTransposeInputIdx? inIdx.toList kIdx.toList (Tensor.to stride (List Nat))
      (Tensor.to padding (List Nat)) with
  | none =>
      simp only
      symm
      apply Finset.sum_eq_zero
      intro outIdx _
      have hForward :
          mkInputIdx? outIdx.toList kIdx.toList (Tensor.to stride (List Nat))
              (Tensor.to padding (List Nat)) ≠
            some inIdx.toList := by
        intro h
        have := mkTransposeInputIdx?_of_mkInputIdx?_eq_some hStride h
        rw [hTranspose] at this
        contradiction
      have hForwardData :
          mkInputIdx? outIdx.toList kIdx.toList stride.data.toList padding.data.toList ≠
            some inIdx.toList := by
        simpa only [Tensor.to_list_eq_data] using hForward
      simp [hForwardData]
  | some outputIdx =>
      change getAtOrZero gradOutput (outCh.val :: outputIdx) *
          channelPairGet layer.kernel outCh inCh kIdx =
        ∑ outIdx : MultiIndex
            (Tensor.to (convOutSpatial inSpatial kernel stride padding) (List Nat)),
          (if mkInputIdx? outIdx.toList kIdx.toList (Tensor.to stride (List Nat))
              (Tensor.to padding (List Nat)) =
              some inIdx.toList then
            channelPairGet layer.kernel outCh inCh kIdx
          else 0) * channelGet gradOutput outCh outIdx
      rw [getAtOrZero_channel_eq_sum_indicator gradOutput outCh outputIdx]
      rw [Finset.sum_mul]
      apply Finset.sum_congr rfl
      intro outIdx _
      have hRelation := mkInputIdx?_eq_some_iff
        (outIdx := outIdx.toList) (kIdx := kIdx.toList)
        (stride := (Tensor.to stride (List Nat))) (padding := (Tensor.to padding (List Nat)))
        (inIdx := inIdx.toList) hStride
      have hEq :
          mkInputIdx? outIdx.toList kIdx.toList (Tensor.to stride (List Nat))
              (Tensor.to padding (List Nat)) =
              some inIdx.toList ↔ outputIdx = outIdx.toList := by
        rw [hRelation, hTranspose]
        simp
      have hEqData :
          mkInputIdx? outIdx.toList kIdx.toList stride.data.toList padding.data.toList =
              some inIdx.toList ↔ outputIdx = outIdx.toList := by
        simpa only [Tensor.to_list_eq_data] using hEq
      by_cases hOutput : outputIdx = outIdx.toList
      · simp [hOutput, hEqData.mpr hOutput]
        ring
      · have hForward := fun h => hOutput (hEq.mp h)
        rw [ite_eq_right hForward]
        simp [hOutput]

/-! ## Adjoint identities -/

/-- The forward input map and implemented input gradient are adjoint at every spatial rank. -/
theorem convCoreSpec_convInputDerivSpec_adjoint
    {d inC outC : Nat}
    {kernel stride padding inSpatial : TorchLean.Tensor Nat [d]}
    (layer : ConvSpec d inC outC kernel stride padding ℝ)
    (input : Tensor ℝ (Shape.ofList (inC :: (Tensor.to inSpatial (List Nat)))))
    (deltaInput : Tensor ℝ (Shape.ofList (inC :: (Tensor.to inSpatial (List Nat)))))
    (gradOutput : Tensor ℝ
      (Shape.ofList
        (outC :: (Tensor.to (convOutSpatial inSpatial kernel stride padding) (List Nat)))))
    (hStride : PositiveStrides (Tensor.to stride (List Nat))) :
    TensorAlgebra.dot (convCoreSpec layer.kernel deltaInput) gradOutput =
      TensorAlgebra.dot deltaInput (convInputDerivSpec layer input gradOutput) := by
  classical
  rw [dot_eq_sum_channel, dot_eq_sum_channel]
  simp_rw [channelGet_convCoreSpec_eq_coefficients]
  simp_rw [channelGet_convInputDerivSpec_eq_coefficients layer input gradOutput hStride]
  simp_rw [Finset.sum_mul, Finset.mul_sum]
  calc
    (∑ outCh,
        ∑ outIdx,
          ∑ inCh,
            ∑ inIdx,
              channelGet deltaInput inCh inIdx *
                convCoefficient layer.kernel outCh outIdx inCh inIdx *
                channelGet gradOutput outCh outIdx) =
      ∑ outCh,
        ∑ inCh,
          ∑ outIdx,
            ∑ inIdx,
              channelGet deltaInput inCh inIdx *
                convCoefficient layer.kernel outCh outIdx inCh inIdx *
                channelGet gradOutput outCh outIdx := by
        apply Finset.sum_congr rfl
        intro outCh _
        rw [Finset.sum_comm]
    _ =
      ∑ inCh,
        ∑ outCh,
          ∑ outIdx,
            ∑ inIdx,
              channelGet deltaInput inCh inIdx *
                convCoefficient layer.kernel outCh outIdx inCh inIdx *
                channelGet gradOutput outCh outIdx := by
        rw [Finset.sum_comm]
    _ =
      ∑ inCh,
        ∑ inIdx,
          ∑ outCh,
            ∑ outIdx,
              channelGet deltaInput inCh inIdx *
                (convCoefficient layer.kernel outCh outIdx inCh inIdx *
                  channelGet gradOutput outCh outIdx) := by
        apply Finset.sum_congr rfl
        intro inCh _
        calc
          (∑ outCh, ∑ outIdx, ∑ inIdx,
              channelGet deltaInput inCh inIdx *
                convCoefficient layer.kernel outCh outIdx inCh inIdx *
                channelGet gradOutput outCh outIdx) =
            ∑ outCh, ∑ inIdx, ∑ outIdx,
              channelGet deltaInput inCh inIdx *
                convCoefficient layer.kernel outCh outIdx inCh inIdx *
                channelGet gradOutput outCh outIdx := by
              apply Finset.sum_congr rfl
              intro outCh _
              rw [Finset.sum_comm]
          _ =
            ∑ inIdx, ∑ outCh, ∑ outIdx,
              channelGet deltaInput inCh inIdx *
                convCoefficient layer.kernel outCh outIdx inCh inIdx *
                channelGet gradOutput outCh outIdx := by
              rw [Finset.sum_comm]
          _ = _ := by
              apply Finset.sum_congr rfl
              intro inIdx _
              apply Finset.sum_congr rfl
              intro outCh _
              apply Finset.sum_congr rfl
              intro outIdx _
              ring

/-- The kernel contraction and implemented kernel gradient are adjoint. -/
theorem convCoreSpec_convKernelDerivSpec_adjoint
    {d inC outC : Nat}
    {kernel stride padding inSpatial : TorchLean.Tensor Nat [d]}
    (layer : ConvSpec d inC outC kernel stride padding ℝ)
    (input : Tensor ℝ (Shape.ofList (inC :: (Tensor.to inSpatial (List Nat)))))
    (deltaKernel : Tensor ℝ (Shape.ofList (outC :: inC :: (Tensor.to kernel (List Nat)))))
    (gradOutput : Tensor ℝ
      (Shape.ofList
        (outC :: (Tensor.to (convOutSpatial inSpatial kernel stride padding) (List Nat))))) :
    TensorAlgebra.dot (convCoreSpec deltaKernel input) gradOutput =
      TensorAlgebra.dot deltaKernel (convKernelDerivSpec layer input gradOutput) := by
  classical
  rw [dot_eq_sum_get, dot_eq_sum_get]
  rw [MultiIndex.sum_cons outC
    (convOutSpatial inSpatial kernel stride padding).data.toList]
  rw [MultiIndex.sum_cons outC (inC :: (Tensor.to kernel (List Nat)))]
  simp_rw [MultiIndex.sum_cons inC (Tensor.to kernel (List Nat))]
  change
    (∑ outCh : Fin outC,
      ∑ outIdx : MultiIndex (Tensor.to (convOutSpatial inSpatial kernel stride padding) (List Nat)),
        channelGet (convCoreSpec deltaKernel input) outCh outIdx *
          channelGet gradOutput outCh outIdx) =
    ∑ outCh : Fin outC, ∑ inCh : Fin inC, ∑ kIdx : MultiIndex (Tensor.to kernel (List Nat)),
      channelPairGet deltaKernel outCh inCh kIdx *
        channelPairGet (convKernelDerivSpec layer input gradOutput) outCh inCh kIdx
  simp_rw [channelGet_convCoreSpec, channelPairGet_convKernelDerivSpec]
  apply Finset.sum_congr rfl
  intro outCh _
  simp_rw [Finset.sum_mul]
  calc
    (∑ outIdx,
        ∑ inCh,
          ∑ kIdx,
            (match mkInputIdx? outIdx.toList kIdx.toList (Tensor.to stride (List Nat))
                (Tensor.to padding (List Nat)) with
              | none => 0
              | some inputIdx => getAtOrZero input (inCh.val :: inputIdx)) *
              channelPairGet deltaKernel outCh inCh kIdx *
              channelGet gradOutput outCh outIdx) =
      ∑ inCh,
        ∑ outIdx,
          ∑ kIdx,
            (match mkInputIdx? outIdx.toList kIdx.toList (Tensor.to stride (List Nat))
                (Tensor.to padding (List Nat)) with
              | none => 0
              | some inputIdx => getAtOrZero input (inCh.val :: inputIdx)) *
              channelPairGet deltaKernel outCh inCh kIdx *
              channelGet gradOutput outCh outIdx := by
        rw [Finset.sum_comm]
    _ =
      ∑ inCh,
        ∑ kIdx,
          ∑ outIdx,
            (match mkInputIdx? outIdx.toList kIdx.toList (Tensor.to stride (List Nat))
                (Tensor.to padding (List Nat)) with
              | none => 0
              | some inputIdx => getAtOrZero input (inCh.val :: inputIdx)) *
              channelPairGet deltaKernel outCh inCh kIdx *
              channelGet gradOutput outCh outIdx := by
        apply Finset.sum_congr rfl
        intro inCh _
        rw [Finset.sum_comm]
    _ =
      ∑ inCh,
        ∑ kIdx,
          channelPairGet deltaKernel outCh inCh kIdx *
            ∑ outIdx,
              (match mkInputIdx? outIdx.toList kIdx.toList (Tensor.to stride (List Nat))
                  (Tensor.to padding (List Nat)) with
                | none => 0
                | some inputIdx => getAtOrZero input (inCh.val :: inputIdx)) *
                channelGet gradOutput outCh outIdx := by
        apply Finset.sum_congr rfl
        intro inCh _
        apply Finset.sum_congr rfl
        intro kIdx _
        rw [Finset.mul_sum]
        apply Finset.sum_congr rfl
        intro outIdx _
        ring

/-- Bias broadcasting and spatial reduction are adjoint. -/
theorem convBiasBroadcastSpec_convBiasDerivSpec_adjoint
    {d inC outC : Nat}
    {kernel stride padding inSpatial : TorchLean.Tensor Nat [d]}
    (layer : ConvSpec d inC outC kernel stride padding ℝ)
    (input : Tensor ℝ (Shape.ofList (inC :: (Tensor.to inSpatial (List Nat)))))
    (deltaBias : Tensor ℝ [outC])
    (gradOutput : Tensor ℝ
      (Shape.ofList
        (outC :: (Tensor.to (convOutSpatial inSpatial kernel stride padding) (List Nat))))) :
    TensorAlgebra.dot
      (convBiasBroadcastSpec (kernel := kernel) (stride := stride) (padding := padding)
        (inSpatial := inSpatial) deltaBias) gradOutput =
      TensorAlgebra.dot deltaBias (convBiasDerivSpec layer input gradOutput) := by
  classical
  rw [dot_eq_sum_channel]
  rw [TensorAlgebra.dot_vec_eq_sum]
  apply Finset.sum_congr rfl
  intro outCh _
  rw [getScalar_convBiasDerivSpec, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro outIdx _
  rw [channelGet_convBiasBroadcastSpec]

/-! ## Fréchet derivative -/

/-- Euclidean coordinates indexed by a tensor shape list. -/
abbrev CoordVec (dims : List Nat) := EuclideanSpace ℝ (MultiIndex dims)

/-- Build shape-indexed Euclidean coordinates from a coordinate function. -/
def coordVecOfFun {dims : List Nat} (f : MultiIndex dims → ℝ) : CoordVec dims :=
  (EuclideanSpace.equiv (𝕜 := ℝ) (ι := MultiIndex dims)).symm f

/-- Coordinates of `coordVecOfFun f` are the values of `f`. -/
@[simp]
theorem coordVecOfFun_apply {dims : List Nat} (f : MultiIndex dims → ℝ)
    (i : MultiIndex dims) : coordVecOfFun f i = f i := by
  simp [coordVecOfFun, EuclideanSpace.equiv]

/-- Vectorize a tensor using bounded multi-indices rather than flattened natural indices. -/
def tensorToCoordVec {dims : List Nat} (x : Tensor ℝ (Shape.ofList dims)) : CoordVec dims :=
  coordVecOfFun fun i ↦ i.get x

/-- Coordinate `i` of a vectorized tensor is the tensor entry at `i`.

Convolution is indexed by bounded multi-indices rather than one flat `Fin`, because the stride and
padding arithmetic is stated per axis; keeping the vectorization multi-indexed means no flattening
appears in any of the derivative proofs. -/
@[simp]
theorem tensorToCoordVec_apply {dims : List Nat} (x : Tensor ℝ (Shape.ofList dims))
    (i : MultiIndex dims) : tensorToCoordVec x i = i.get x := by
  simp [tensorToCoordVec]

/-- The kernel/input contraction in Euclidean coordinates. -/
def convCoreVec
    {d inC outC : Nat}
    {kernel stride padding inSpatial : TorchLean.Tensor Nat [d]}
    (weights : CoordVec (outC :: inC :: (Tensor.to kernel (List Nat))))
    (input : CoordVec (inC :: (Tensor.to inSpatial (List Nat)))) :
    CoordVec (outC :: (Tensor.to (convOutSpatial inSpatial kernel stride padding) (List Nat))) :=
  coordVecOfFun fun outCoord ↦
    ∑ inCh : Fin inC, ∑ inIdx : MultiIndex (Tensor.to inSpatial (List Nat)),
      input (inCh, inIdx) *
        ∑ kIdx : MultiIndex (Tensor.to kernel (List Nat)),
          if mkInputIdx? outCoord.2.toList kIdx.toList (Tensor.to stride (List Nat))
              (Tensor.to padding (List Nat)) =
              some inIdx.toList then
            weights (outCoord.1, (inCh, kIdx))
          else
            0

/-- Unfolds one output coordinate of the contraction into its double sum. -/
@[simp]
theorem convCoreVec_apply
    {d inC outC : Nat}
    {kernel stride padding inSpatial : TorchLean.Tensor Nat [d]}
    (weights : CoordVec (outC :: inC :: (Tensor.to kernel (List Nat))))
    (input : CoordVec (inC :: (Tensor.to inSpatial (List Nat))))
    (outCoord : MultiIndex
      (outC :: (Tensor.to (convOutSpatial inSpatial kernel stride padding) (List Nat)))) :
    convCoreVec (kernel := kernel) (stride := stride) (padding := padding) weights input outCoord =
      ∑ inCh : Fin inC, ∑ inIdx : MultiIndex (Tensor.to inSpatial (List Nat)),
        input (inCh, inIdx) *
          ∑ kIdx : MultiIndex (Tensor.to kernel (List Nat)),
            if mkInputIdx? outCoord.2.toList kIdx.toList (Tensor.to stride (List Nat))
                (Tensor.to padding (List Nat)) =
                some inIdx.toList then
              weights (outCoord.1, (inCh, kIdx))
            else
              0 := by
  simp [convCoreVec]

/-- Tensor convolution and its Euclidean-coordinate contraction agree at every spatial rank. -/
theorem tensorToCoordVec_convCoreSpec
    {d inC outC : Nat}
    {kernel stride padding inSpatial : TorchLean.Tensor Nat [d]}
    (weights : Tensor ℝ (Shape.ofList (outC :: inC :: (Tensor.to kernel (List Nat)))))
    (input : Tensor ℝ (Shape.ofList (inC :: (Tensor.to inSpatial (List Nat))))) :
    tensorToCoordVec (convCoreSpec weights input) =
      convCoreVec (kernel := kernel) (stride := stride) (padding := padding)
        (tensorToCoordVec weights) (tensorToCoordVec input) := by
  ext outCoord
  rcases outCoord with ⟨outCh, outIdx⟩
  rw [tensorToCoordVec_apply]
  change channelGet (convCoreSpec weights input) outCh outIdx = _
  rw [channelGet_convCoreSpec_eq_coefficients]
  simp only [convCoreVec_apply, tensorToCoordVec_apply, channelGet, channelPairGet,
    convCoefficient]

/-- Euclidean kernel coordinates for a rank-general convolution. -/
abbrev ConvKernelCoords {d : Nat} (outC inC : Nat) (kernel : TorchLean.Tensor Nat [d]) :=
  CoordVec (outC :: inC :: (Tensor.to kernel (List Nat)))

/-- Euclidean input coordinates for a rank-general convolution. -/
abbrev ConvInputCoords {d : Nat} (inC : Nat) (inSpatial : TorchLean.Tensor Nat [d]) :=
  CoordVec (inC :: (Tensor.to inSpatial (List Nat)))

/-- Euclidean output coordinates for a rank-general convolution. -/
abbrev ConvOutputCoords {d : Nat} (outC : Nat)
    (inSpatial kernel stride padding : TorchLean.Tensor Nat [d]) :=
  CoordVec (outC :: (Tensor.to (convOutSpatial inSpatial kernel stride padding) (List Nat)))

/-- The convolution contraction is additive in its input coordinates. -/
theorem convCoreVec_add_input
    {d inC outC : Nat}
    {kernel stride padding inSpatial : TorchLean.Tensor Nat [d]}
    (weights : ConvKernelCoords outC inC kernel)
    (x y : ConvInputCoords inC inSpatial) :
    convCoreVec (kernel := kernel) (stride := stride) (padding := padding) weights (x + y) =
      convCoreVec (kernel := kernel) (stride := stride) (padding := padding) weights x +
        convCoreVec (kernel := kernel) (stride := stride) (padding := padding) weights y := by
  ext outCoord
  simp [convCoreVec, Finset.sum_add_distrib, add_mul]

/-- The convolution contraction respects scalar multiplication in its input coordinates. -/
theorem convCoreVec_smul_input
    {d inC outC : Nat}
    {kernel stride padding inSpatial : TorchLean.Tensor Nat [d]}
    (weights : ConvKernelCoords outC inC kernel) (c : ℝ)
    (input : ConvInputCoords inC inSpatial) :
    convCoreVec (kernel := kernel) (stride := stride) (padding := padding) weights (c • input) =
      c • convCoreVec (kernel := kernel) (stride := stride) (padding := padding) weights input := by
  ext outCoord
  simp only [convCoreVec_apply, PiLp.smul_apply, smul_eq_mul]
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro inCh _
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro inIdx _
  ring

/-- The convolution contraction is additive in its kernel coordinates. -/
theorem convCoreVec_add_kernel
    {d inC outC : Nat}
    {kernel stride padding inSpatial : TorchLean.Tensor Nat [d]}
    (w z : ConvKernelCoords outC inC kernel)
    (input : ConvInputCoords inC inSpatial) :
    convCoreVec (kernel := kernel) (stride := stride) (padding := padding) (w + z) input =
      convCoreVec (kernel := kernel) (stride := stride) (padding := padding) w input +
        convCoreVec (kernel := kernel) (stride := stride) (padding := padding) z input := by
  ext outCoord
  simp only [convCoreVec_apply, PiLp.add_apply]
  rw [← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro inCh _
  rw [← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro inIdx _
  rw [← mul_add, ← Finset.sum_add_distrib]
  apply congrArg (input (inCh, inIdx) * ·)
  apply Finset.sum_congr rfl
  intro kIdx _
  by_cases h : mkInputIdx? outCoord.2.toList kIdx.toList
      stride.data.toList padding.data.toList = some inIdx.toList <;> simp [h]

/-- The convolution contraction respects scalar multiplication in its kernel coordinates. -/
theorem convCoreVec_smul_kernel
    {d inC outC : Nat}
    {kernel stride padding inSpatial : TorchLean.Tensor Nat [d]}
    (c : ℝ) (weights : ConvKernelCoords outC inC kernel)
    (input : ConvInputCoords inC inSpatial) :
    convCoreVec (kernel := kernel) (stride := stride) (padding := padding) (c • weights) input =
      c • convCoreVec (kernel := kernel) (stride := stride) (padding := padding) weights input := by
  ext outCoord
  simp only [convCoreVec_apply, PiLp.smul_apply, smul_eq_mul]
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro inCh _
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro inIdx _
  have hsum :
      (∑ kIdx : MultiIndex (Tensor.to kernel (List Nat)),
          if mkInputIdx? outCoord.2.toList kIdx.toList
              (Tensor.to stride (List Nat)) (Tensor.to padding (List Nat)) = some inIdx.toList then
            c * weights (outCoord.1, (inCh, kIdx))
          else
            0) =
        c * ∑ kIdx : MultiIndex (Tensor.to kernel (List Nat)),
          if mkInputIdx? outCoord.2.toList kIdx.toList
              (Tensor.to stride (List Nat)) (Tensor.to padding (List Nat)) = some inIdx.toList then
            weights (outCoord.1, (inCh, kIdx))
          else
            0 := by
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro kIdx _
    by_cases h : mkInputIdx? outCoord.2.toList kIdx.toList
        stride.data.toList padding.data.toList = some inIdx.toList <;> simp [h]
  rw [hsum]
  ring

/-- Continuous bilinear form of rank-general convolution. -/
def convCoreBilin
    {d inC outC : Nat}
    {kernel stride padding inSpatial : TorchLean.Tensor Nat [d]} :
    ConvKernelCoords outC inC kernel →L[ℝ]
      ConvInputCoords inC inSpatial →L[ℝ]
        ConvOutputCoords outC inSpatial kernel stride padding := by
  classical
  let inner : ConvKernelCoords outC inC kernel →
      ConvInputCoords inC inSpatial →ₗ[ℝ]
        ConvOutputCoords outC inSpatial kernel stride padding :=
    fun weights ↦
      { toFun := fun input ↦
          convCoreVec (kernel := kernel) (stride := stride) (padding := padding) weights input
        map_add' := convCoreVec_add_input weights
        map_smul' := convCoreVec_smul_input weights }
  let innerContinuous (weights : ConvKernelCoords outC inC kernel) :
      ConvInputCoords inC inSpatial →L[ℝ]
        ConvOutputCoords outC inSpatial kernel stride padding :=
    ⟨inner weights, LinearMap.continuous_of_finiteDimensional (f := inner weights)⟩
  let outer : ConvKernelCoords outC inC kernel →ₗ[ℝ]
      ConvInputCoords inC inSpatial →L[ℝ]
        ConvOutputCoords outC inSpatial kernel stride padding :=
    { toFun := innerContinuous
      map_add' := by
        intro w z
        apply ContinuousLinearMap.ext
        intro input
        exact convCoreVec_add_kernel w z input
      map_smul' := by
        intro c w
        apply ContinuousLinearMap.ext
        intro input
        exact convCoreVec_smul_kernel c w input }
  exact ⟨outer, LinearMap.continuous_of_finiteDimensional (f := outer)⟩

/-- The bundled bilinear map computes the same contraction as `convCoreVec`.

Bundling matters: once convolution is a continuous bilinear map, its derivative in each argument
comes
from Mathlib rather than from a hand-written difference quotient. -/
@[simp]
theorem convCoreBilin_apply
    {d inC outC : Nat}
    {kernel stride padding inSpatial : TorchLean.Tensor Nat [d]}
    (weights : CoordVec (outC :: inC :: (Tensor.to kernel (List Nat))))
    (input : CoordVec (inC :: (Tensor.to inSpatial (List Nat)))) :
    convCoreBilin (kernel := kernel) (stride := stride) (padding := padding) weights input =
      convCoreVec (kernel := kernel) (stride := stride) (padding := padding) weights input := by
  rfl

/-- Continuous linear bias broadcast in Euclidean coordinates. -/
def convBiasBroadcastCLM
    {d outC : Nat}
    {kernel stride padding inSpatial : TorchLean.Tensor Nat [d]} :
    CoordVec [outC] →L[ℝ]
      CoordVec
        (outC :: (Tensor.to (convOutSpatial inSpatial kernel stride padding) (List Nat))) := by
  let linear : CoordVec [outC] →ₗ[ℝ]
      CoordVec (outC :: (Tensor.to (convOutSpatial inSpatial kernel stride padding) (List Nat))) :=
    { toFun := fun bias ↦ coordVecOfFun fun outCoord ↦ bias (outCoord.1, PUnit.unit)
      map_add' := by intro x y; ext i; simp
      map_smul' := by intro c x; ext i; simp [smul_eq_mul] }
  exact ⟨linear, LinearMap.continuous_of_finiteDimensional (f := linear)⟩

/-- Bias broadcast copies the channel entry to every spatial position of that channel. -/
@[simp]
theorem convBiasBroadcastCLM_apply
    {d outC : Nat}
    {kernel stride padding inSpatial : TorchLean.Tensor Nat [d]}
    (bias : CoordVec [outC])
    (outCoord : MultiIndex
      (outC :: (Tensor.to (convOutSpatial inSpatial kernel stride padding) (List Nat)))) :
    convBiasBroadcastCLM (kernel := kernel) (stride := stride) (padding := padding) bias outCoord =
      bias (outCoord.1, PUnit.unit) := by
  simp [convBiasBroadcastCLM]

/-- Coordinate conversion preserves pointwise tensor addition. -/
theorem tensorToCoordVec_addSpec
    {dims : List Nat} (x y : Tensor ℝ (Shape.ofList dims)) :
    tensorToCoordVec (addSpec x y) = tensorToCoordVec x + tensorToCoordVec y := by
  ext i
  rw [tensorToCoordVec_apply, MultiIndex.get_addSpec]
  rfl

/-- Tensor bias broadcasting agrees with the corresponding coordinate map. -/
theorem tensorToCoordVec_convBiasBroadcastSpec
    {d outC : Nat}
    {kernel stride padding inSpatial : TorchLean.Tensor Nat [d]}
    (bias : Tensor ℝ [outC]) :
    tensorToCoordVec (convBiasBroadcastSpec
        (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial) bias) =
      convBiasBroadcastCLM (kernel := kernel) (stride := stride) (padding := padding)
        (tensorToCoordVec bias) := by
  ext outCoord
  rcases outCoord with ⟨outCh, outIdx⟩
  rw [tensorToCoordVec_apply]
  change channelGet (convBiasBroadcastSpec bias) outCh outIdx = _
  rw [channelGet_convBiasBroadcastSpec]
  rw [convBiasBroadcastCLM_apply]
  rw [tensorToCoordVec_apply]
  exact (MultiIndex.get_vector_eq_getScalar bias outCh).symm

/-- Coordinate form of a complete convolution layer, including bias. -/
def convForwardVec
    {d inC outC : Nat}
    {kernel stride padding inSpatial : TorchLean.Tensor Nat [d]}
    (state : (CoordVec (outC :: inC :: (Tensor.to kernel (List Nat))) × CoordVec [outC]) ×
      CoordVec (inC :: (Tensor.to inSpatial (List Nat)))) :
    CoordVec (outC :: (Tensor.to (convOutSpatial inSpatial kernel stride padding) (List Nat))) :=
  convCoreBilin (kernel := kernel) (stride := stride) (padding := padding)
      state.1.1 state.2 +
    convBiasBroadcastCLM (kernel := kernel) (stride := stride) (padding := padding) state.1.2

/-- Tensor-level `convSpec` agrees with `convForwardVec`. -/
theorem tensorToCoordVec_convSpec
    {d inC outC : Nat}
    {kernel stride padding inSpatial : TorchLean.Tensor Nat [d]}
    (layer : ConvSpec d inC outC kernel stride padding ℝ)
    (input : Tensor ℝ (Shape.ofList (inC :: (Tensor.to inSpatial (List Nat))))) :
    tensorToCoordVec (convSpec layer input) =
      convForwardVec (kernel := kernel) (stride := stride) (padding := padding)
        ((tensorToCoordVec layer.kernel, tensorToCoordVec layer.bias), tensorToCoordVec input) := by
  rw [show convSpec layer input =
    addSpec (convCoreSpec layer.kernel input) (convBiasBroadcastSpec layer.bias) by rfl]
  rw [tensorToCoordVec_addSpec, tensorToCoordVec_convCoreSpec,
    tensorToCoordVec_convBiasBroadcastSpec]
  rfl

/-- Euclidean state space of one rank-general convolution application. -/
abbrev ConvState
    {d : Nat} (inC outC : Nat) (kernel inSpatial : TorchLean.Tensor Nat [d]) :=
  (CoordVec (outC :: inC :: (Tensor.to kernel (List Nat))) × CoordVec [outC]) ×
    CoordVec (inC :: (Tensor.to inSpatial (List Nat)))

/-- Projection of the kernel coordinates from a convolution state. -/
def convWeightProjection
    {d inC outC : Nat} {kernel inSpatial : TorchLean.Tensor Nat [d]} :
    ConvState inC outC kernel inSpatial →L[ℝ]
        CoordVec (outC :: inC :: (Tensor.to kernel (List Nat))) :=
  (ContinuousLinearMap.fst ℝ _ _).comp (ContinuousLinearMap.fst ℝ _ _)

/-- Projection of the bias coordinates from a convolution state. -/
def convBiasProjection
    {d inC outC : Nat} {kernel inSpatial : TorchLean.Tensor Nat [d]} :
    ConvState inC outC kernel inSpatial →L[ℝ] CoordVec [outC] :=
  (ContinuousLinearMap.snd ℝ _ _).comp (ContinuousLinearMap.fst ℝ _ _)

/-- Projection of the input coordinates from a convolution state. -/
def convInputProjection
    {d inC outC : Nat} {kernel inSpatial : TorchLean.Tensor Nat [d]} :
    ConvState inC outC kernel inSpatial →L[ℝ]
        CoordVec (inC :: (Tensor.to inSpatial (List Nat))) :=
  ContinuousLinearMap.snd ℝ _ _

/-- Product-rule derivative of a complete convolution state. -/
def convDerivative
    {d inC outC : Nat}
    {kernel stride padding inSpatial : TorchLean.Tensor Nat [d]}
    (state : ConvState inC outC kernel inSpatial) :
    ConvState inC outC kernel inSpatial →L[ℝ]
        CoordVec
          (outC :: (Tensor.to (convOutSpatial inSpatial kernel stride padding) (List Nat))) :=
  (convCoreBilin (kernel := kernel) (stride := stride) (padding := padding)).precompR
      (ConvState inC outC kernel inSpatial) state.1.1
      (convInputProjection (inC := inC) (outC := outC) (kernel := kernel)
        (inSpatial := inSpatial))
    + (convCoreBilin (kernel := kernel) (stride := stride) (padding := padding)).precompL
      (ConvState inC outC kernel inSpatial)
      (convWeightProjection (inC := inC) (outC := outC) (kernel := kernel)
        (inSpatial := inSpatial)) state.2
    + (convBiasBroadcastCLM (kernel := kernel) (stride := stride) (padding := padding)).comp
      (convBiasProjection (inC := inC) (outC := outC) (kernel := kernel)
        (inSpatial := inSpatial))

/-- The exact derivative of a rank-general convolution is its kernel/input product rule plus
bias. -/
theorem hasFDerivAt_convForwardVec
    {d inC outC : Nat}
    {kernel stride padding inSpatial : TorchLean.Tensor Nat [d]}
    (state : ConvState inC outC kernel inSpatial) :
    HasFDerivAt
      (convForwardVec (kernel := kernel) (stride := stride) (padding := padding))
      (convDerivative (kernel := kernel) (stride := stride) (padding := padding) state)
      state := by
  let weightProjection :=
    convWeightProjection (inC := inC) (outC := outC) (kernel := kernel)
      (inSpatial := inSpatial)
  let inputProjection :=
    convInputProjection (inC := inC) (outC := outC) (kernel := kernel)
      (inSpatial := inSpatial)
  let biasProjection :=
    convBiasProjection (inC := inC) (outC := outC) (kernel := kernel)
      (inSpatial := inSpatial)
  let core := convCoreBilin (inC := inC) (outC := outC)
    (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
  have hCore : HasFDerivAt
      (fun s : ConvState inC outC kernel inSpatial ↦ core (weightProjection s) (inputProjection s))
      (core.precompR (ConvState inC outC kernel inSpatial) state.1.1 inputProjection +
        core.precompL (ConvState inC outC kernel inSpatial) weightProjection state.2)
      state :=
    core.hasFDerivAt_of_bilinear weightProjection.hasFDerivAt inputProjection.hasFDerivAt
  let biasMap : ConvState inC outC kernel inSpatial →L[ℝ]
      CoordVec (outC :: (Tensor.to (convOutSpatial inSpatial kernel stride padding) (List Nat))) :=
    (convBiasBroadcastCLM (kernel := kernel) (stride := stride) (padding := padding)
      (inSpatial := inSpatial)).comp biasProjection
  have hBias : HasFDerivAt (fun s : ConvState inC outC kernel inSpatial ↦ biasMap s)
      biasMap state := biasMap.hasFDerivAt
  have hSum := hCore.add hBias
  change HasFDerivAt
    ((fun s : ConvState inC outC kernel inSpatial ↦
        core (weightProjection s) (inputProjection s)) + fun s ↦ biasMap s)
    (core.precompR (ConvState inC outC kernel inSpatial) state.1.1 inputProjection +
      core.precompL (ConvState inC outC kernel inSpatial) weightProjection state.2 + biasMap)
    state
  exact hSum

/-- Applying the analytic derivative gives the tensor-level convolution JVP. -/
theorem tensorToCoordVec_convJvpSpec
    {d inC outC : Nat}
    {kernel stride padding inSpatial : TorchLean.Tensor Nat [d]}
    (layer tangentLayer : ConvSpec d inC outC kernel stride padding ℝ)
    (input tangentInput : Tensor ℝ (Shape.ofList (inC :: (Tensor.to inSpatial (List Nat))))) :
    tensorToCoordVec (convJvpSpec layer tangentLayer input tangentInput) =
      convDerivative (kernel := kernel) (stride := stride) (padding := padding)
        ((tensorToCoordVec layer.kernel, tensorToCoordVec layer.bias), tensorToCoordVec input)
        ((tensorToCoordVec tangentLayer.kernel, tensorToCoordVec tangentLayer.bias),
          tensorToCoordVec tangentInput) := by
  rw [show convJvpSpec layer tangentLayer input tangentInput =
    addSpec
      (addSpec (convCoreSpec tangentLayer.kernel input)
        (convCoreSpec layer.kernel tangentInput))
      (convBiasBroadcastSpec tangentLayer.bias) by rfl]
  rw [tensorToCoordVec_addSpec, tensorToCoordVec_addSpec]
  rw [tensorToCoordVec_convCoreSpec, tensorToCoordVec_convCoreSpec,
    tensorToCoordVec_convBiasBroadcastSpec]
  simp [convDerivative, convWeightProjection, convInputProjection, convBiasProjection]
  rw [add_comm]
  rw [convCoreBilin_apply, convCoreBilin_apply]

/-- The implemented convolution backward pass is the adjoint of the exact JVP.

`convBackwardSpec` returns a `ConvGradients` record, so the equation below reads one inner product
per gradient, each paired with the matching piece of the input tangent. An earlier version returned
a bare triple and had to project the components out by position, which is the same proposition and
considerably harder to check against a sentence describing it. -/
theorem convJvpSpec_convBackwardSpec_adjoint
    {d inC outC : Nat}
    {kernel stride padding inSpatial : TorchLean.Tensor Nat [d]}
    (layer tangentLayer : ConvSpec d inC outC kernel stride padding ℝ)
    (input tangentInput : Tensor ℝ (Shape.ofList (inC :: (Tensor.to inSpatial (List Nat)))))
    (gradOutput : Tensor ℝ
      (Shape.ofList
        (outC :: (Tensor.to (convOutSpatial inSpatial kernel stride padding) (List Nat)))))
    (hStride : PositiveStrides (Tensor.to stride (List Nat))) :
    TensorAlgebra.dot (convJvpSpec layer tangentLayer input tangentInput) gradOutput =
      (let gradients := convBackwardSpec layer input gradOutput
      TensorAlgebra.dot tangentLayer.kernel gradients.kernelGradient +
        TensorAlgebra.dot tangentLayer.bias gradients.biasGradient +
        TensorAlgebra.dot tangentInput gradients.inputGradient) := by
  rw [show convJvpSpec layer tangentLayer input tangentInput =
    addSpec
      (addSpec (convCoreSpec tangentLayer.kernel input)
        (convCoreSpec layer.kernel tangentInput))
      (convBiasBroadcastSpec tangentLayer.bias) by rfl]
  rw [TensorAlgebra.dot_add_left, TensorAlgebra.dot_add_left]
  rw [convCoreSpec_convKernelDerivSpec_adjoint layer input tangentLayer.kernel gradOutput]
  rw [convCoreSpec_convInputDerivSpec_adjoint layer input tangentInput gradOutput hStride]
  rw [convBiasBroadcastSpec_convBiasDerivSpec_adjoint layer input tangentLayer.bias gradOutput]
  simp only [convBackwardSpec]
  ring

end
end Proofs.Autograd.Conv
