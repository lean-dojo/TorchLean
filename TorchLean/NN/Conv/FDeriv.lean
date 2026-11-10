/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Conv.Index
public import TorchLean.Tensor.Euclidean
public import TorchLean.Tensor.Conversion
public import Mathlib.Analysis.Calculus.FDeriv.Add
public import Mathlib.Analysis.Calculus.FDeriv.Bilinear
public import TorchLean.Tensor.Context.Real

/-!
# Derivative of General Convolution

Derivative and reverse-mode formulas for dense channels-first convolution at an arbitrary spatial
rank. Every sum runs over bounded multi-indices, so no theorem fixes the number of axes. The
convolution contraction is bilinear in kernel and input, and the Fréchet derivative of the full
layer is the product rule plus the bias term. The input-gradient adjoint identity needs positive
strides, which make the forward and transpose index relations agree. The theorems are about the
exact-real specification and do not cover the native LibTorch kernels.
-/

@[expose] public section

namespace TorchLean.Conv

open scoped BigOperators
open Tensor
open Rep (dot)

noncomputable section

/-- Read a channel and bounded spatial coordinate from a channels-first tensor. -/
def channelGet {α : Type} [Storage α] {channels : Nat} {dims : List Nat}
    (x : Tensor α (channels :: dims))
    (channel : Fin channels) (i : Shape.Coord dims) : α :=
  x (channel, i)

/-- Read the entry at two leading channel indices and a bounded spatial coordinate, as in a kernel
laid out `outC × inC × kernel`. -/
def channelPairGet {α : Type} [Storage α] {outer inner : Nat} {dims : List Nat}
    (x : Tensor α (outer :: inner :: dims))
    (i : Fin outer) (j : Fin inner) (k : Shape.Coord dims) : α :=
  x (i, (j, k))

/-- Split a channels-first tensor dot product into channel and spatial sums. -/
theorem dot_eq_sum_channel {α : Type} [Storage α] [CommSemiring α]
    (channels : Nat) (dims : List Nat)
    (x y : Tensor α (channels :: dims)) :
    Rep.dot x y =
      ∑ channel : Fin channels, ∑ i : Shape.Coord dims,
        channelGet x channel i * channelGet y channel i := by
  rw [Rep.dot, Fintype.sum_prod_type]
  rfl

/-- A channels-first lookup at `channel :: indices` is an indicator sum over spatial coordinates: it
reads the coordinate whose index list is `indices`, and is zero when no coordinate has that list. -/
theorem getAtOrZero_channel_eq_sum {α : Type} [Storage α] [AddCommMonoid α]
    {channels : Nat} {dims : List Nat}
    (x : Tensor α (channels :: dims))
    (channel : Fin channels) (indices : List Nat) :
    getAtOrZero x (channel.val :: indices) =
      ∑ i : Shape.Coord dims,
        if indices = i.toList then channelGet x channel i else 0 := by
  rw [show x = Tensor.dim (Tensor.unstack x) from (Tensor.dim_unstack x).symm]
  simp only [getAtOrZero_dim_cons, channel.isLt, ↓reduceDIte, Tensor.unstack_dim]
  rw [getAtOrZero_eq_sum_indicator]
  apply Finset.sum_congr rfl
  intro i _
  simp only [channelGet, Tensor.dim_apply]

/-- The coefficient of input entry `(inCh, inIdx)` in output entry `(outCh, outIdx)`: the sum of the
kernel weights at every offset `kIdx` whose forward index map sends `outIdx` to `inIdx`. -/
def coeff
    {d inC outC : Nat}
    {kernel stride padding inSpatial : Tensor Nat [d]}
    (weights : Tensor ℝ (outC :: inC :: (Tensor.to kernel (List Nat))))
    (outCh : Fin outC)
    (outIdx : Shape.Coord
      (Tensor.to (outSpatialDilated inSpatial kernel stride
          (Tensor.ones [d]) padding padding) (List Nat)))
    (inCh : Fin inC) (inIdx : Shape.Coord (Tensor.to inSpatial (List Nat))) : ℝ :=
  ∑ kIdx : Shape.Coord (Tensor.to kernel (List Nat)),
    if mkInputIdx? outIdx.toList kIdx.toList (Tensor.to stride (List Nat))
        (Tensor.to padding (List Nat)) =
        some inIdx.toList then
      channelPairGet weights outCh inCh kIdx
    else
      0

/-- One output coordinate of the convolution contraction as a finite sum over input channels and
kernel offsets, with padded positions reading zero. -/
theorem channelGet_contract
    {d inC outC : Nat}
    {kernel stride padding inSpatial : Tensor Nat [d]}
    (weights : Tensor ℝ (outC :: inC :: (Tensor.to kernel (List Nat))))
    (input : Tensor ℝ (inC :: (Tensor.to inSpatial (List Nat))))
    (outCh : Fin outC)
    (outIdx : Shape.Coord
      (Tensor.to (outSpatialDilated inSpatial kernel stride
          (Tensor.ones [d]) padding padding) (List Nat))) :
    channelGet (contract weights input) outCh outIdx =
      ∑ inCh : Fin inC, ∑ kIdx : Shape.Coord (Tensor.to kernel (List Nat)),
        (match mkInputIdx? outIdx.toList kIdx.toList (Tensor.to stride (List Nat))
            (Tensor.to padding (List Nat)) with
          | none => 0
          | some inIdx => getAtOrZero input (inCh.val :: inIdx)) *
        channelPairGet weights outCh inCh kIdx := by
  simp only [channelGet, contract, Conv.contractWith]
  rw [Tensor.dim_apply, Tensor.generate, Tensor.Rep.get_ofFn]
  simp_rw [foldlIndices_add]
  rw [List.finRange_foldl_add_eq_add_sum, zero_add]
  apply Finset.sum_congr rfl
  intro inCh _
  apply Finset.sum_congr rfl
  intro kIdx _
  have hWeight' :
      getAtOrZero weights (outCh.val :: inCh.val :: kIdx.toList) =
        weights (outCh, (inCh, kIdx)) :=
    getAtOrZero_toList _ weights (outCh, (inCh, kIdx))
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
        weights (outCh, (inCh, kIdx))
  rw [hWeight']

/-- Convolution is the matrix represented by `coeff` at every spatial rank. -/
theorem channelGet_contract_eq_coeff
    {d inC outC : Nat}
    {kernel stride padding inSpatial : Tensor Nat [d]}
    (weights : Tensor ℝ (outC :: inC :: (Tensor.to kernel (List Nat))))
    (input : Tensor ℝ (inC :: (Tensor.to inSpatial (List Nat))))
    (outCh : Fin outC)
    (outIdx : Shape.Coord
      (Tensor.to (outSpatialDilated inSpatial kernel stride
          (Tensor.ones [d]) padding padding) (List Nat))) :
    channelGet (contract weights input) outCh outIdx =
      ∑ inCh : Fin inC, ∑ inIdx : Shape.Coord (Tensor.to inSpatial (List Nat)),
        channelGet input inCh inIdx *
          coeff weights outCh outIdx inCh inIdx := by
  rw [channelGet_contract]
  apply Finset.sum_congr rfl
  intro inCh _
  unfold coeff
  calc
    (∑ kIdx : Shape.Coord (Tensor.to kernel (List Nat)),
        (match mkInputIdx? outIdx.toList kIdx.toList (Tensor.to stride (List Nat))
            (Tensor.to padding (List Nat)) with
          | none => 0
          | some inputIdx => getAtOrZero input (inCh.val :: inputIdx)) *
          channelPairGet weights outCh inCh kIdx) =
      ∑ kIdx : Shape.Coord (Tensor.to kernel (List Nat)),
        ∑ inIdx : Shape.Coord (Tensor.to inSpatial (List Nat)),
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
              ∑ inIdx : Shape.Coord (Tensor.to inSpatial (List Nat)),
                channelGet input inCh inIdx *
                  (if some inputIdx = some inIdx.toList then
                    channelPairGet weights outCh inCh kIdx
                  else 0)
            rw [getAtOrZero_channel_eq_sum input inCh inputIdx]
            rw [Finset.sum_mul]
            apply Finset.sum_congr rfl
            intro inIdx _
            by_cases hEq : inputIdx = inIdx.toList <;> simp [hEq]
    _ =
      ∑ inIdx : Shape.Coord (Tensor.to inSpatial (List Nat)),
        ∑ kIdx : Shape.Coord (Tensor.to kernel (List Nat)),
          channelGet input inCh inIdx *
            (if mkInputIdx? outIdx.toList kIdx.toList (Tensor.to stride (List Nat))
                (Tensor.to padding (List Nat)) =
                some inIdx.toList then
              channelPairGet weights outCh inCh kIdx
            else 0) := by
        rw [Finset.sum_comm]
    _ =
      ∑ inIdx : Shape.Coord (Tensor.to inSpatial (List Nat)),
        channelGet input inCh inIdx *
          ∑ kIdx : Shape.Coord (Tensor.to kernel (List Nat)),
            if mkInputIdx? outIdx.toList kIdx.toList (Tensor.to stride (List Nat))
                (Tensor.to padding (List Nat)) =
                some inIdx.toList then
              channelPairGet weights outCh inCh kIdx
            else 0 := by
        apply Finset.sum_congr rfl
        intro inIdx _
        rw [Finset.mul_sum]

/-- One kernel-gradient coordinate is the contraction of input and output cotangent. -/
theorem channelPairGet_kernelDeriv
    {d inC outC : Nat}
    {kernel stride padding inSpatial : Tensor Nat [d]}
    (layer : Conv d inC outC kernel stride padding ℝ)
    (input : Tensor ℝ (inC :: (Tensor.to inSpatial (List Nat))))
    (gradOutput : Tensor ℝ
      (outC :: (Tensor.to (outSpatialDilated inSpatial kernel stride
            (Tensor.ones [d]) padding padding) (List Nat))))
    (outCh : Fin outC) (inCh : Fin inC) (kIdx : Shape.Coord (Tensor.to kernel (List Nat))) :
    channelPairGet (kernelDeriv layer input gradOutput) outCh inCh kIdx =
      ∑ outIdx : Shape.Coord
          (Tensor.to (outSpatialDilated inSpatial kernel stride
              (Tensor.ones [d]) padding padding) (List Nat)),
        (match mkInputIdx? outIdx.toList kIdx.toList (Tensor.to stride (List Nat))
            (Tensor.to padding (List Nat)) with
          | none => 0
          | some inIdx => getAtOrZero input (inCh.val :: inIdx)) *
        channelGet gradOutput outCh outIdx := by
  simp only [channelPairGet, kernelDeriv]
  simp only [Tensor.dim_apply]
  rw [Tensor.generate, Tensor.Rep.get_ofFn, foldlIndices_add]
  simp only [zero_add]
  apply Finset.sum_congr rfl
  intro outIdx _
  have hGrad' :
      getAtOrZero gradOutput (outCh.val :: outIdx.toList) =
        gradOutput (outCh, outIdx) :=
    getAtOrZero_toList _ gradOutput (outCh, outIdx)
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
        gradOutput (outCh, outIdx)
  rw [hGrad']

/-- One bias-gradient coordinate is the spatial sum of the output cotangent. -/
theorem getScalar_biasDeriv
    {d inC outC : Nat}
    {kernel stride padding inSpatial : Tensor Nat [d]}
    (layer : Conv d inC outC kernel stride padding ℝ)
    (input : Tensor ℝ (inC :: (Tensor.to inSpatial (List Nat))))
    (gradOutput : Tensor ℝ
      (outC :: (Tensor.to (outSpatialDilated inSpatial kernel stride
            (Tensor.ones [d]) padding padding) (List Nat))))
    (outCh : Fin outC) :
    Tensor.getScalar (biasDeriv layer input gradOutput) outCh =
      ∑ outIdx : Shape.Coord
          (Tensor.to (outSpatialDilated inSpatial kernel stride
              (Tensor.ones [d]) padding padding) (List Nat)),
        channelGet gradOutput outCh outIdx := by
  simp only [biasDeriv, Tensor.getScalar_dim]
  rw [foldlIndices_add]
  simp only [zero_add]
  apply Finset.sum_congr rfl
  intro outIdx _
  change getAtOrZero gradOutput (outCh.val :: outIdx.toList) =
    gradOutput (outCh, outIdx)
  exact getAtOrZero_toList _ gradOutput (outCh, outIdx)

/-- Broadcasting a bias reads the same channel value at every spatial coordinate. -/
theorem channelGet_biasBroadcast
    {d outC : Nat}
    {kernel stride padding inSpatial : Tensor Nat [d]}
    (bias : Tensor ℝ [outC])
    (outCh : Fin outC)
    (outIdx : Shape.Coord
      (Tensor.to (outSpatialDilated inSpatial kernel stride
          (Tensor.ones [d]) padding padding) (List Nat))) :
    channelGet (biasBroadcastWith (outSpatialDilated inSpatial kernel stride
      (Tensor.ones [d]) padding padding) bias) outCh outIdx =
        Tensor.getScalar bias outCh := by
  simp only [channelGet]
  change (Tensor.dim fun outCh =>
      Tensor.generate
        (Tensor.to (outSpatialDilated inSpatial kernel stride
            (Tensor.ones [d]) padding padding) (List Nat))
        fun _ => getAtOrZero bias [outCh.val]) (outCh, outIdx) = _
  rw [Tensor.dim_apply, Tensor.generate, Tensor.Rep.get_ofFn]
  change getAtOrZero bias [outCh.val] = (bias.unstack outCh).item
  simp

/-- One input-gradient coordinate is the specification's transpose-index contraction. -/
theorem channelGet_inputDeriv
    {d inC outC : Nat}
    {kernel stride padding inSpatial : Tensor Nat [d]}
    (layer : Conv d inC outC kernel stride padding ℝ)
    (input : Tensor ℝ (inC :: (Tensor.to inSpatial (List Nat))))
    (gradOutput : Tensor ℝ
      (outC :: (Tensor.to (outSpatialDilated inSpatial kernel stride
            (Tensor.ones [d]) padding padding) (List Nat))))
    (inCh : Fin inC) (inIdx : Shape.Coord (Tensor.to inSpatial (List Nat))) :
    channelGet (inputDeriv layer input gradOutput) inCh inIdx =
      ∑ outCh : Fin outC, ∑ kIdx : Shape.Coord (Tensor.to kernel (List Nat)),
        (match mkTransposeInputIdx? inIdx.toList kIdx.toList (Tensor.to stride (List Nat))
            (Tensor.to padding (List Nat)) with
          | none => 0
          | some outIdx =>
              getAtOrZero gradOutput (outCh.val :: outIdx) *
                channelPairGet layer.kernel outCh inCh kIdx) := by
  simp only [channelGet, inputDeriv]
  rw [Tensor.dim_apply, Tensor.generate, Tensor.Rep.get_ofFn]
  simp_rw [foldlIndices_add]
  rw [List.finRange_foldl_add_eq_add_sum, zero_add]
  apply Finset.sum_congr rfl
  intro outCh _
  apply Finset.sum_congr rfl
  intro kIdx _
  have hWeight' :
      getAtOrZero layer.kernel (outCh.val :: inCh.val :: kIdx.toList) =
        layer.kernel (outCh, (inCh, kIdx)) :=
    getAtOrZero_toList _ layer.kernel (outCh, (inCh, kIdx))
  split <;> simp_all [channelPairGet]

/-- For positive strides, the specification's input gradient multiplies the output cotangent by the
transposed coefficient matrix. -/
theorem channelGet_inputDeriv_eq_coeff
    {d inC outC : Nat}
    {kernel stride padding inSpatial : Tensor Nat [d]}
    (layer : Conv d inC outC kernel stride padding ℝ)
    (input : Tensor ℝ (inC :: (Tensor.to inSpatial (List Nat))))
    (gradOutput : Tensor ℝ
      (outC :: (Tensor.to (outSpatialDilated inSpatial kernel stride
            (Tensor.ones [d]) padding padding) (List Nat))))
    (hStride : PositiveStrides (Tensor.to stride (List Nat)))
    (inCh : Fin inC) (inIdx : Shape.Coord (Tensor.to inSpatial (List Nat))) :
    channelGet (inputDeriv layer input gradOutput) inCh inIdx =
      ∑ outCh : Fin outC,
        ∑ outIdx : Shape.Coord
            (Tensor.to (outSpatialDilated inSpatial kernel stride
                (Tensor.ones [d]) padding padding) (List Nat)),
          coeff layer.kernel outCh outIdx inCh inIdx *
            channelGet gradOutput outCh outIdx := by
  rw [channelGet_inputDeriv]
  apply Finset.sum_congr rfl
  intro outCh _
  unfold coeff
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
        have := mkTransposeInputIdx?_of_forward hStride h
        rw [hTranspose] at this
        contradiction
      have hForwardData := hForward
      simp only [Tensor.to_list_eq_data] at hForwardData
      simp [hForwardData]
  | some outputIdx =>
      change getAtOrZero gradOutput (outCh.val :: outputIdx) *
          channelPairGet layer.kernel outCh inCh kIdx =
        ∑ outIdx : Shape.Coord
            (Tensor.to (outSpatialDilated inSpatial kernel stride
                (Tensor.ones [d]) padding padding) (List Nat)),
          (if mkInputIdx? outIdx.toList kIdx.toList (Tensor.to stride (List Nat))
              (Tensor.to padding (List Nat)) =
              some inIdx.toList then
            channelPairGet layer.kernel outCh inCh kIdx
          else 0) * channelGet gradOutput outCh outIdx
      rw [getAtOrZero_channel_eq_sum gradOutput outCh outputIdx]
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
      have hEqData := hEq
      simp only [Tensor.to_list_eq_data] at hEqData
      by_cases hOutput : outputIdx = outIdx.toList
      · simp [hOutput, hEqData.mpr hOutput]
        ring
      · have hForward := fun h => hOutput (hEq.mp h)
        rw [ite_eq_right hForward]
        simp only [Tensor.to_list_eq_data] at hOutput
        simp [hOutput]

/-! ## Adjoint identities -/

/-- For positive strides, the forward map in the input and the input-gradient specification are
adjoint at every spatial rank. -/
theorem contract_inputDeriv_adjoint
    {d inC outC : Nat}
    {kernel stride padding inSpatial : Tensor Nat [d]}
    (layer : Conv d inC outC kernel stride padding ℝ)
    (input : Tensor ℝ (inC :: (Tensor.to inSpatial (List Nat))))
    (deltaInput : Tensor ℝ (inC :: (Tensor.to inSpatial (List Nat))))
    (gradOutput : Tensor ℝ
      (outC :: (Tensor.to (outSpatialDilated inSpatial kernel stride
            (Tensor.ones [d]) padding padding) (List Nat))))
    (hStride : PositiveStrides (Tensor.to stride (List Nat))) :
    Rep.dot (contract layer.kernel deltaInput) gradOutput =
      Rep.dot deltaInput (inputDeriv layer input gradOutput) := by
  classical
  rw [dot_eq_sum_channel, dot_eq_sum_channel]
  simp_rw [channelGet_contract_eq_coeff]
  simp_rw [channelGet_inputDeriv_eq_coeff layer input gradOutput hStride]
  simp_rw [Finset.sum_mul, Finset.mul_sum]
  calc
    (∑ outCh,
        ∑ outIdx,
          ∑ inCh,
            ∑ inIdx,
              channelGet deltaInput inCh inIdx *
                coeff layer.kernel outCh outIdx inCh inIdx *
                channelGet gradOutput outCh outIdx) =
      ∑ outCh,
        ∑ inCh,
          ∑ outIdx,
            ∑ inIdx,
              channelGet deltaInput inCh inIdx *
                coeff layer.kernel outCh outIdx inCh inIdx *
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
                coeff layer.kernel outCh outIdx inCh inIdx *
                channelGet gradOutput outCh outIdx := by
        rw [Finset.sum_comm]
    _ =
      ∑ inCh,
        ∑ inIdx,
          ∑ outCh,
            ∑ outIdx,
              channelGet deltaInput inCh inIdx *
                (coeff layer.kernel outCh outIdx inCh inIdx *
                  channelGet gradOutput outCh outIdx) := by
        apply Finset.sum_congr rfl
        intro inCh _
        calc
          (∑ outCh, ∑ outIdx, ∑ inIdx,
              channelGet deltaInput inCh inIdx *
                coeff layer.kernel outCh outIdx inCh inIdx *
                channelGet gradOutput outCh outIdx) =
            ∑ outCh, ∑ inIdx, ∑ outIdx,
              channelGet deltaInput inCh inIdx *
                coeff layer.kernel outCh outIdx inCh inIdx *
                channelGet gradOutput outCh outIdx := by
              apply Finset.sum_congr rfl
              intro outCh _
              rw [Finset.sum_comm]
          _ =
            ∑ inIdx, ∑ outCh, ∑ outIdx,
              channelGet deltaInput inCh inIdx *
                coeff layer.kernel outCh outIdx inCh inIdx *
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

/-- The kernel contraction and kernel-gradient specification are adjoint. -/
theorem contract_kernelDeriv_adjoint
    {d inC outC : Nat}
    {kernel stride padding inSpatial : Tensor Nat [d]}
    (layer : Conv d inC outC kernel stride padding ℝ)
    (input : Tensor ℝ (inC :: (Tensor.to inSpatial (List Nat))))
    (deltaKernel : Tensor ℝ (outC :: inC :: (Tensor.to kernel (List Nat))))
    (gradOutput : Tensor ℝ
      (outC :: (Tensor.to (outSpatialDilated inSpatial kernel stride
            (Tensor.ones [d]) padding padding) (List Nat)))) :
    Rep.dot (contract deltaKernel input) gradOutput =
      Rep.dot deltaKernel (kernelDeriv layer input gradOutput) := by
  classical
  rw [Rep.dot, Rep.dot]
  rw [Fintype.sum_prod_type (α₁ := Fin outC)
    (α₂ := Shape.Coord (outSpatialDilated inSpatial kernel stride
        (Tensor.ones [d]) padding padding).data.toList)]
  rw [Fintype.sum_prod_type (α₁ := Fin outC)
    (α₂ := Shape.Coord (inC :: (Tensor.to kernel (List Nat))))]
  simp_rw [Fintype.sum_prod_type (α₁ := Fin inC)
    (α₂ := Shape.Coord (Tensor.to kernel (List Nat)))]
  change
    (∑ outCh : Fin outC,
      ∑ outIdx : Shape.Coord
          (Tensor.to (outSpatialDilated inSpatial kernel stride
              (Tensor.ones [d]) padding padding) (List Nat)),
        channelGet (contract deltaKernel input) outCh outIdx *
          channelGet gradOutput outCh outIdx) =
    ∑ outCh : Fin outC, ∑ inCh : Fin inC, ∑ kIdx : Shape.Coord (Tensor.to kernel (List Nat)),
      channelPairGet deltaKernel outCh inCh kIdx *
        channelPairGet (kernelDeriv layer input gradOutput) outCh inCh kIdx
  simp_rw [channelGet_contract, channelPairGet_kernelDeriv]
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
theorem biasBroadcast_adjoint
    {d inC outC : Nat}
    {kernel stride padding inSpatial : Tensor Nat [d]}
    (layer : Conv d inC outC kernel stride padding ℝ)
    (input : Tensor ℝ (inC :: (Tensor.to inSpatial (List Nat))))
    (deltaBias : Tensor ℝ [outC])
    (gradOutput : Tensor ℝ
      (outC :: (Tensor.to (outSpatialDilated inSpatial kernel stride
            (Tensor.ones [d]) padding padding) (List Nat)))) :
    Rep.dot
      (biasBroadcastWith (outSpatialDilated inSpatial kernel stride
        (Tensor.ones [d]) padding padding) deltaBias) gradOutput =
      Rep.dot deltaBias (biasDeriv layer input gradOutput) := by
  classical
  rw [dot_eq_sum_channel]
  rw [Tensor.dot_vec_eq_sum]
  apply Finset.sum_congr rfl
  intro outCh _
  rw [getScalar_biasDeriv, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro outIdx _
  rw [channelGet_biasBroadcast]

/-! ## Fréchet derivative -/

/-- The convolution contraction is additive in its input. -/
theorem contract_add_input
    {d inC outC : Nat}
    {kernel stride padding inSpatial : Tensor Nat [d]}
    (weights : Tensor ℝ (outC :: inC :: (Tensor.to kernel (List Nat))))
    (x y : Tensor ℝ (inC :: (Tensor.to inSpatial (List Nat)))) :
    contract (stride := stride) (padding := padding) weights (x + y) =
      contract (stride := stride) (padding := padding) weights x +
        contract (stride := stride) (padding := padding) weights y := by
  apply Tensor.Rep.ext
  rintro ⟨outCh, outIdx⟩
  rw [Tensor.add_apply]
  change channelGet _ outCh outIdx = channelGet _ outCh outIdx + channelGet _ outCh outIdx
  simp only [channelGet_contract_eq_coeff]
  simp only [channelGet, Tensor.add_apply, add_mul, Finset.sum_add_distrib]

/-- The convolution contraction respects scalar multiplication in its input. -/
theorem contract_smul_input
    {d inC outC : Nat}
    {kernel stride padding inSpatial : Tensor Nat [d]}
    (c : ℝ) (weights : Tensor ℝ (outC :: inC :: (Tensor.to kernel (List Nat))))
    (input : Tensor ℝ (inC :: (Tensor.to inSpatial (List Nat)))) :
    contract (stride := stride) (padding := padding) weights (c • input) =
      c • contract (stride := stride) (padding := padding) weights input := by
  apply Tensor.Rep.ext
  rintro ⟨outCh, outIdx⟩
  rw [Tensor.Rep.smul_apply, smul_eq_mul]
  change channelGet _ outCh outIdx = c * channelGet _ outCh outIdx
  simp only [channelGet_contract_eq_coeff, Finset.mul_sum]
  simp only [channelGet, Tensor.Rep.smul_apply, smul_eq_mul, mul_assoc]

/-- The convolution contraction is additive in its kernel. -/
theorem contract_add_kernel
    {d inC outC : Nat}
    {kernel stride padding inSpatial : Tensor Nat [d]}
    (w z : Tensor ℝ (outC :: inC :: (Tensor.to kernel (List Nat))))
    (input : Tensor ℝ (inC :: (Tensor.to inSpatial (List Nat)))) :
    contract (stride := stride) (padding := padding) (w + z) input =
      contract (stride := stride) (padding := padding) w input +
        contract (stride := stride) (padding := padding) z input := by
  apply Tensor.Rep.ext
  rintro ⟨outCh, outIdx⟩
  rw [Tensor.add_apply]
  change channelGet _ outCh outIdx = channelGet _ outCh outIdx + channelGet _ outCh outIdx
  simp only [channelGet_contract]
  simp only [channelPairGet, Tensor.add_apply, mul_add, Finset.sum_add_distrib]

/-- The convolution contraction respects scalar multiplication in its kernel. -/
theorem contract_smul_kernel
    {d inC outC : Nat}
    {kernel stride padding inSpatial : Tensor Nat [d]}
    (c : ℝ) (weights : Tensor ℝ (outC :: inC :: (Tensor.to kernel (List Nat))))
    (input : Tensor ℝ (inC :: (Tensor.to inSpatial (List Nat)))) :
    contract (stride := stride) (padding := padding) (c • weights) input =
      c • contract (stride := stride) (padding := padding) weights input := by
  apply Tensor.Rep.ext
  rintro ⟨outCh, outIdx⟩
  rw [Tensor.Rep.smul_apply, smul_eq_mul]
  change channelGet _ outCh outIdx = c * channelGet _ outCh outIdx
  simp only [channelGet_contract, Finset.mul_sum]
  simp only [channelPairGet, Tensor.Rep.smul_apply, smul_eq_mul, mul_left_comm _ c]

/-- Rank-general convolution as a continuous bilinear map of kernel and input. -/
def contractBilin
    {d inC outC : Nat}
    {kernel stride padding inSpatial : Tensor Nat [d]} :
    Tensor ℝ (outC :: inC :: (Tensor.to kernel (List Nat))) →L[ℝ]
      Tensor ℝ (inC :: (Tensor.to inSpatial (List Nat))) →L[ℝ]
        Tensor ℝ (outC :: (Tensor.to (outSpatialDilated inSpatial kernel stride
              (Tensor.ones [d]) padding padding) (List Nat))) :=
  LinearMap.toContinuousLinearMap <| LinearMap.toContinuousLinearMap.toLinearMap ∘ₗ
    LinearMap.mk₂ ℝ (contract (stride := stride) (padding := padding))
      contract_add_kernel contract_smul_kernel
      contract_add_input contract_smul_input

/-- The bundled bilinear map computes the same contraction as `Conv.contract`.

Once convolution is a continuous bilinear map, mathlib's bilinear rule (`hasFDerivAt_of_bilinear`)
gives its derivative in each argument, and no difference-quotient argument is needed. -/
@[simp]
theorem contractBilin_apply
    {d inC outC : Nat}
    {kernel stride padding inSpatial : Tensor Nat [d]}
    (weights : Tensor ℝ (outC :: inC :: (Tensor.to kernel (List Nat))))
    (input : Tensor ℝ (inC :: (Tensor.to inSpatial (List Nat)))) :
    contractBilin (kernel := kernel) (stride := stride) (padding := padding) weights input =
      contract (stride := stride) (padding := padding) weights input := by
  rfl

/-- Continuous linear bias broadcast. -/
def biasBroadcastCLM
    {d outC : Nat}
    {kernel stride padding inSpatial : Tensor Nat [d]} :
    Tensor ℝ [outC] →L[ℝ]
      Tensor ℝ (outC :: (Tensor.to (outSpatialDilated inSpatial kernel stride
            (Tensor.ones [d]) padding padding) (List Nat))) :=
  LinearMap.toContinuousLinearMap
    { toFun := biasBroadcastWith (outSpatialDilated inSpatial kernel stride
        (Tensor.ones [d]) padding padding)
      map_add' := by
        intro x y
        apply Tensor.Rep.ext
        rintro ⟨outCh, outIdx⟩
        rw [Tensor.add_apply]
        change channelGet _ outCh outIdx = channelGet _ outCh outIdx + channelGet _ outCh outIdx
        simp only [channelGet_biasBroadcast, Tensor.getScalar_eq_apply, Tensor.add_apply]
      map_smul' := by
        intro c x
        apply Tensor.Rep.ext
        rintro ⟨outCh, outIdx⟩
        rw [Tensor.Rep.smul_apply, RingHom.id_apply]
        change channelGet _ outCh outIdx = c • channelGet _ outCh outIdx
        simp only [channelGet_biasBroadcast, Tensor.getScalar_eq_apply,
          Tensor.Rep.smul_apply] }

/-- The bundled bias broadcast computes `biasBroadcastWith` over the output spatial shape. -/
@[simp]
theorem biasBroadcastCLM_apply
    {d outC : Nat}
    {kernel stride padding inSpatial : Tensor Nat [d]}
    (bias : Tensor ℝ [outC]) :
    biasBroadcastCLM (kernel := kernel) (stride := stride) (padding := padding)
        (inSpatial := inSpatial) bias =
      biasBroadcastWith (outSpatialDilated inSpatial kernel stride
        (Tensor.ones [d]) padding padding) bias := by
  rfl

/-- The space of a convolution's arguments: kernel and bias, paired with the input. -/
abbrev State
    {d : Nat} (inC outC : Nat) (kernel inSpatial : Tensor Nat [d]) :=
  (Tensor ℝ (outC :: inC :: (Tensor.to kernel (List Nat))) × Tensor ℝ [outC]) ×
    Tensor ℝ (inC :: (Tensor.to inSpatial (List Nat)))

/-- Projection of the kernel from a convolution state. -/
def weightProjection
    {d inC outC : Nat} {kernel inSpatial : Tensor Nat [d]} :
    State inC outC kernel inSpatial →L[ℝ]
        Tensor ℝ (outC :: inC :: (Tensor.to kernel (List Nat))) :=
  (ContinuousLinearMap.fst ℝ _ _).comp (ContinuousLinearMap.fst ℝ _ _)

/-- Projection of the bias from a convolution state. -/
def biasProjection
    {d inC outC : Nat} {kernel inSpatial : Tensor Nat [d]} :
    State inC outC kernel inSpatial →L[ℝ] Tensor ℝ [outC] :=
  (ContinuousLinearMap.snd ℝ _ _).comp (ContinuousLinearMap.fst ℝ _ _)

/-- Projection of the input from a convolution state. -/
def inputProjection
    {d inC outC : Nat} {kernel inSpatial : Tensor Nat [d]} :
    State inC outC kernel inSpatial →L[ℝ]
        Tensor ℝ (inC :: (Tensor.to inSpatial (List Nat))) :=
  ContinuousLinearMap.snd ℝ _ _

/-- The derivative of `Conv.forward` in kernel, bias and input at `state`, by the product rule:
the kernel applied to the input tangent, plus the kernel tangent applied to the input, plus the
broadcast bias tangent. -/
def derivative
    {d inC outC : Nat}
    {kernel stride padding inSpatial : Tensor Nat [d]}
    (state : State inC outC kernel inSpatial) :
    State inC outC kernel inSpatial →L[ℝ]
        Tensor ℝ (outC :: (Tensor.to (outSpatialDilated inSpatial kernel stride
              (Tensor.ones [d]) padding padding) (List Nat))) :=
  (contractBilin (kernel := kernel) (stride := stride) (padding := padding)).precompR
      (State inC outC kernel inSpatial) state.1.1
      (inputProjection (inC := inC) (outC := outC) (kernel := kernel)
        (inSpatial := inSpatial))
    + (contractBilin (kernel := kernel) (stride := stride) (padding := padding)).precompL
      (State inC outC kernel inSpatial)
      (weightProjection (inC := inC) (outC := outC) (kernel := kernel)
        (inSpatial := inSpatial)) state.2
    + (biasBroadcastCLM (kernel := kernel) (stride := stride) (padding := padding)).comp
      (biasProjection (inC := inC) (outC := outC) (kernel := kernel)
        (inSpatial := inSpatial))

/-- The exact derivative of a rank-general convolution is its kernel/input product rule plus
bias. -/
theorem hasFDerivAt_forward
    {d inC outC : Nat}
    {kernel stride padding inSpatial : Tensor Nat [d]}
    (state : State inC outC kernel inSpatial) :
    HasFDerivAt
      (fun s : State inC outC kernel inSpatial ↦
        forward (stride := stride) (padding := padding) ⟨s.1.1, s.1.2⟩ s.2)
      (derivative (kernel := kernel) (stride := stride) (padding := padding) state)
      state := by
  let weightProjection :=
    Conv.weightProjection (inC := inC) (outC := outC) (kernel := kernel)
      (inSpatial := inSpatial)
  let inputProjection :=
    Conv.inputProjection (inC := inC) (outC := outC) (kernel := kernel)
      (inSpatial := inSpatial)
  let biasProjection :=
    Conv.biasProjection (inC := inC) (outC := outC) (kernel := kernel)
      (inSpatial := inSpatial)
  let core := contractBilin (inC := inC) (outC := outC)
    (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
  have hCore : HasFDerivAt
      (fun s : State inC outC kernel inSpatial ↦ core (weightProjection s) (inputProjection s))
      (core.precompR (State inC outC kernel inSpatial) state.1.1 inputProjection +
        core.precompL (State inC outC kernel inSpatial) weightProjection state.2)
      state :=
    core.hasFDerivAt_of_bilinear weightProjection.hasFDerivAt inputProjection.hasFDerivAt
  let biasMap : State inC outC kernel inSpatial →L[ℝ]
      Tensor ℝ (outC :: (Tensor.to (outSpatialDilated inSpatial kernel stride
            (Tensor.ones [d]) padding padding) (List Nat))) :=
    (biasBroadcastCLM (kernel := kernel) (stride := stride) (padding := padding)
      (inSpatial := inSpatial)).comp biasProjection
  have hBias : HasFDerivAt (fun s : State inC outC kernel inSpatial ↦ biasMap s)
      biasMap state := biasMap.hasFDerivAt
  have hSum := hCore.add hBias
  change HasFDerivAt
    ((fun s : State inC outC kernel inSpatial ↦
        core (weightProjection s) (inputProjection s)) + fun s ↦ biasMap s)
    (core.precompR (State inC outC kernel inSpatial) state.1.1 inputProjection +
      core.precompL (State inC outC kernel inSpatial) weightProjection state.2 + biasMap)
    state
  exact hSum

/-- Applying the analytic derivative gives the tensor-level convolution JVP. -/
theorem jvp_eq_derivative
    {d inC outC : Nat}
    {kernel stride padding inSpatial : Tensor Nat [d]}
    (layer tangentLayer : Conv d inC outC kernel stride padding ℝ)
    (input tangentInput : Tensor ℝ (inC :: (Tensor.to inSpatial (List Nat)))) :
    jvp layer tangentLayer input tangentInput =
      derivative (kernel := kernel) (stride := stride) (padding := padding)
        ((layer.kernel, layer.bias), input)
        ((tangentLayer.kernel, tangentLayer.bias), tangentInput) := by
  rw [show jvp layer tangentLayer input tangentInput =
    (contract tangentLayer.kernel input +
        contract layer.kernel tangentInput) +
      biasBroadcastWith (outSpatialDilated inSpatial kernel stride
        (Tensor.ones [d]) padding padding) tangentLayer.bias by rfl]
  rw [add_comm (contract tangentLayer.kernel input)]
  rfl

/-- For positive strides, the convolution backward specification is the adjoint of the JVP: pairing
the JVP with `gradOutput` gives the sum of the `Gradients` fields of `Conv.backward` (kernel,
bias and input gradient), each paired with its tangent.

Only the input gradient needs the stride hypothesis, for its transpose index relation. -/
theorem jvp_backward_adjoint
    {d inC outC : Nat}
    {kernel stride padding inSpatial : Tensor Nat [d]}
    (layer tangentLayer : Conv d inC outC kernel stride padding ℝ)
    (input tangentInput : Tensor ℝ (inC :: (Tensor.to inSpatial (List Nat))))
    (gradOutput : Tensor ℝ
      (outC :: (Tensor.to (outSpatialDilated inSpatial kernel stride
            (Tensor.ones [d]) padding padding) (List Nat))))
    (hStride : PositiveStrides (Tensor.to stride (List Nat))) :
    Rep.dot (jvp layer tangentLayer input tangentInput) gradOutput =
      (let gradients := backward layer input gradOutput
      Rep.dot tangentLayer.kernel gradients.kernelGradient +
        Rep.dot tangentLayer.bias gradients.biasGradient +
        Rep.dot tangentInput gradients.inputGradient) := by
  rw [show jvp layer tangentLayer input tangentInput =
    (contract tangentLayer.kernel input +
        contract layer.kernel tangentInput) +
      biasBroadcastWith (outSpatialDilated inSpatial kernel stride
        (Tensor.ones [d]) padding padding) tangentLayer.bias by rfl]
  rw [Tensor.dot_add_left, Tensor.dot_add_left]
  rw [contract_kernelDeriv_adjoint layer input tangentLayer.kernel gradOutput]
  rw [contract_inputDeriv_adjoint layer input tangentInput gradOutput hStride]
  rw [biasBroadcast_adjoint layer input tangentLayer.bias gradOutput]
  simp only [backward]
  ring

end
end TorchLean.Conv
