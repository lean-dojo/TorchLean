/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Conv.Index
public import TorchLean.Tensor.Conversion
public import TorchLean.Verify.RuntimeApprox.NF.Scalar

/-!
# Rounded Convolution

Forward- and reverse-mode error bounds for channels-first convolution at arbitrary spatial rank.
The bounds follow the ordered folds in `Conv.forward` and `Conv.backward` and round
products and sums separately, so rounded addition is never reassociated. LibTorch's convolution
kernels are outside the statements. Inputs and weights are tensors, and lists only describe the
coordinate traversal inside the proofs. A replacement implementation must keep that order.
-/

@[expose] public section

namespace TorchLean.NF

open Conv Tensor
open FloatLib FloatLib.Numerics FloatLib.Floats.Formats
open Flocq

noncomputable section

variable {β : Radix} {fexp : ℤ → ℤ} [ValidExp fexp]
variable {rnd : ℝ → ℤ} [ValidRndToNearest rnd]

local notation "R" => NF β fexp rnd

/-! ## Ordered rounded sums -/

/-- Error budget after one rounded addition. -/
def accumulationError (acc term : R) (accError termError : ℝ) : ℝ :=
  accError + termError +
    ulp β fexp
      (NF.toReal acc +
        NF.toReal term) / 2

/-- Error budget after one rounded multiplication. -/
def productError (x y : R) (xError yError : ℝ) : ℝ :=
  let xValue := NF.toReal x
  let yValue := NF.toReal y
  (abs xValue + xError) * yError + (abs yValue + yError) * xError +
    ulp β fexp (xValue * yValue) / 2

/-- Fold rounded terms with their absolute-error budgets, in the supplied index order. -/
def foldErrorState {ι : Type} (indices : List ι)
    (term : ι → R) (termError : ι → ℝ) (initial : R × ℝ) : R × ℝ :=
  indices.foldl (fun state index =>
    let value := term index
    (state.1 + value,
      accumulationError (β := β) (fexp := fexp) (rnd := rnd)
        state.1 value state.2 (termError index))) initial

/-- Error budget for a rounded sum beginning at zero. -/
def foldError {ι : Type} (indices : List ι)
    (term : ι → R) (termError : ι → ℝ) : ℝ :=
  (foldErrorState (β := β) (fexp := fexp) (rnd := rnd)
    indices term termError (0, 0)).2

private theorem approx_foldErrorState {ι : Type} (indices : List ι)
    (ideal : ι → ℝ) (rounded : ι → R) (termError : ι → ℝ) :
    ∀ (idealAcc : ℝ) (state : R × ℝ),
      abs (NF.toReal state.1 - idealAcc) ≤ state.2 →
      (∀ index ∈ indices,
        abs (NF.toReal (rounded index) -
          ideal index) ≤ termError index) →
      abs
          (NF.toReal
              (indices.foldl (fun acc index => acc + rounded index) state.1) -
            indices.foldl (fun acc index => acc + ideal index) idealAcc) ≤
        (foldErrorState (β := β) (fexp := fexp) (rnd := rnd)
          indices rounded termError state).2 := by
  intro idealAcc state hAcc hTerm
  induction indices generalizing idealAcc state with
  | nil => simpa [foldErrorState] using hAcc
  | cons head tail ih =>
      have hHead := hTerm head (by simp)
      have hNext :
          abs
              (NF.toReal
                  (state.1 + rounded head) -
                (idealAcc + ideal head)) ≤
            accumulationError (β := β) (fexp := fexp) (rnd := rnd)
              state.1 (rounded head) state.2 (termError head) := by
        simpa [accumulationError, add_assoc, add_comm, add_left_comm] using
          (approx_add (β := β) (fexp := fexp) (rnd := rnd)
            (x := idealAcc) (y := ideal head) (xR := state.1) (yR := rounded head)
            (epsx := state.2) (epsy := termError head) hAcc hHead)
      simpa [foldErrorState] using
        ih (idealAcc + ideal head)
          (state.1 + rounded head,
            accumulationError (β := β) (fexp := fexp) (rnd := rnd)
              state.1 (rounded head) state.2 (termError head))
          hNext (by
            intro index hIndex
            exact hTerm index (by simp [hIndex]))

/-- If each rounded term is within `termError` of its ideal value, the rounded left-to-right sum
from zero is within `foldError` of the ideal sum. -/
theorem approx_fold {ι : Type} (indices : List ι)
    (ideal : ι → ℝ) (rounded : ι → R) (termError : ι → ℝ)
    (hTerm : ∀ index ∈ indices,
      abs (NF.toReal (rounded index) -
        ideal index) ≤ termError index) :
    abs
        (NF.toReal
            (indices.foldl (fun acc index => acc + rounded index) 0) -
          indices.foldl (fun acc index => acc + ideal index) 0) ≤
      foldError (β := β) (fexp := fexp) (rnd := rnd)
        indices rounded termError := by
  have hZero :
      |NF.toReal (0 : R) - 0| ≤ (0 : ℝ) := by
    simp [toReal_zero (β := β) (fexp := fexp) (rnd := rnd)]
  simpa [foldError, foldErrorState] using
    approx_foldErrorState (β := β) (fexp := fexp) (rnd := rnd)
      indices ideal rounded termError 0 (0, 0) hZero hTerm

/-- Error budget for a rounded dot product whose two factors are approximated pointwise. -/
def productFoldError {ι : Type} (indices : List ι)
    (left right : ι → R) (leftError rightError : ι → ℝ) : ℝ :=
  foldError (β := β) (fexp := fexp) (rnd := rnd) indices
    (fun index => left index * right index)
    (fun index => productError (β := β) (fexp := fexp) (rnd := rnd)
      (left index) (right index) (leftError index) (rightError index))

/-- If both factors are approximated pointwise, the rounded dot product summed in index order is
within `productFoldError` of the ideal one. -/
theorem approx_product_fold {ι : Type} (indices : List ι)
    (leftIdeal rightIdeal : ι → ℝ) (leftRounded rightRounded : ι → R)
    (leftError rightError : ι → ℝ)
    (hLeft : ∀ index ∈ indices,
      abs (NF.toReal (leftRounded index) -
        leftIdeal index) ≤ leftError index)
    (hRight : ∀ index ∈ indices,
      abs (NF.toReal (rightRounded index) -
        rightIdeal index) ≤ rightError index) :
    abs
        (NF.toReal
            (indices.foldl
              (fun acc index => acc + leftRounded index * rightRounded index) 0) -
          indices.foldl
            (fun acc index => acc + leftIdeal index * rightIdeal index) 0) ≤
      productFoldError (β := β) (fexp := fexp) (rnd := rnd)
        indices leftRounded rightRounded leftError rightError := by
  apply approx_fold (β := β) (fexp := fexp) (rnd := rnd)
  intro index hIndex
  simpa [productError] using
    (approx_mul (β := β) (fexp := fexp) (rnd := rnd)
      (x := leftIdeal index) (y := rightIdeal index)
      (xR := leftRounded index) (yR := rightRounded index)
      (epsx := leftError index) (epsy := rightError index)
      (hLeft index hIndex) (hRight index hIndex))

/-! ## Arbitrary-rank index traversal -/

/-- All coordinates of a list-shaped tensor in row-major order, the order in which the nested spec
loop visits them. -/
def enumerateIndices : List Nat → List (List Nat)
  | [] => [[]]
  | n :: dims =>
      (List.finRange n).flatMap fun index =>
        (enumerateIndices dims).map fun tail => index.val :: tail

/-- The nested index loop equals a single fold over the flattened index list.

Flattening keeps the nested loop's order, so the rounded fold performs the same additions in the
same sequence. -/
theorem foldlIndices_eq_foldl {α : Type} (dims : List Nat)
    (initial : α) (step : α → List Nat → α) :
    foldlIndices dims initial step =
      (enumerateIndices dims).foldl step initial := by
  induction dims generalizing initial step with
  | nil => rfl
  | cons n dims ih =>
      simp only [foldlIndices, enumerateIndices, List.foldl_flatMap]
      apply List.foldl_ext
      intro state index _
      rw [ih]
      simp [List.foldl_map]

/-- Same flattening one level up, for the channel loop wrapped around the spatial loop. -/
theorem foldChannelsIndices_eq_foldl {α : Type} (channels : Nat) (dims : List Nat)
    (initial : α) (step : α → Fin channels → List Nat → α) :
    (List.finRange channels).foldl
        (fun state channel => foldlIndices dims state (step · channel ·)) initial =
      ((List.finRange channels).product (enumerateIndices dims)).foldl
        (fun state index => step state index.1 index.2) initial := by
  simp only [List.product, List.foldl_flatMap]
  apply List.foldl_ext
  intro state channel _
  rw [foldlIndices_eq_foldl]
  simp [List.foldl_map]

/-! ## Tensor coordinates -/

private theorem get_conv_contract
    {α : Type} [Context α]
    {d inC outC : Nat} {kernel stride padding inSpatial : Tensor Nat [d]}
    (weights : Tensor α (outC :: inC :: (Tensor.to kernel (List Nat))))
    (input : Tensor α (inC :: (Tensor.to inSpatial (List Nat))))
    (outChannel : Fin outC)
    (outIndex :
      Shape.Coord (Tensor.to (outSpatialDilated inSpatial kernel stride
          (Tensor.ones [d]) padding padding) (List Nat))) :
    (Conv.contract (stride := stride) (padding := padding) weights input) (outChannel, outIndex) =
      (List.finRange inC).foldl (fun acc inChannel =>
        foldlIndices (Tensor.to kernel (List Nat)) acc (fun acc kernelIndex =>
          acc +
            (match mkInputIdx? outIndex.toList kernelIndex (Tensor.to stride (List Nat))
                (Tensor.to padding (List Nat)) with
              | none => 0
              | some inputIndex => getAtOrZero input (inChannel.val :: inputIndex)) *
            getAtOrZero weights (outChannel.val :: inChannel.val :: kernelIndex))) 0 := by
  simp only [Conv.contract, Conv.contractWith, Tensor.dim_apply,
    Tensor.generate, Tensor.Rep.get_ofFn]
  rfl

private theorem get_conv_biasBroadcast
    {α : Type} [Context α]
    {d outC : Nat} {kernel stride padding inSpatial : Tensor Nat [d]}
    (bias : Tensor α [outC]) (outChannel : Fin outC)
    (outIndex :
      Shape.Coord (Tensor.to (outSpatialDilated inSpatial kernel stride
          (Tensor.ones [d]) padding padding) (List Nat))) :
    (biasBroadcastWith (outSpatialDilated inSpatial kernel stride
          (Tensor.ones [d]) padding padding) bias) (outChannel, outIndex) =
      getAtOrZero bias [outChannel.val] := by
  simp only [Conv.biasBroadcastWith,
    Tensor.dim_apply, Tensor.generate, Tensor.Rep.get_ofFn]

/-- `Tensor.getAtOrZero` preserves a uniform tensor approximation at every index, including
out-of-range indices where both sides read zero. -/
theorem approx_getAtOrZero {shape : Shape} {ideal : Tensor ℝ shape}
    {rounded : Tensor R shape} {error : ℝ}
    (h : ApproxTensor
      (toSpec := NF.toReal) ideal rounded error)
    (index : List Nat) :
    abs (NF.toReal
        (getAtOrZero rounded index) - getAtOrZero ideal index) ≤ error := by
  induction shape generalizing index with
  | scalar =>
      cases index with
      | nil =>
          rw [← Tensor.scalar_item ideal, ← Tensor.scalar_item rounded] at h
          simpa [getAtOrZero] using (approxTensor_scalar_iff.mp h)
      | cons _ _ =>
          simpa [getAtOrZero,
            toReal_zero (β := β) (fexp := fexp) (rnd := rnd)] using
              ApproxTensor.eps_nonneg h
  | dim n shape ih =>
      cases index with
      | nil =>
          simpa [getAtOrZero,
            toReal_zero (β := β) (fexp := fexp) (rnd := rnd)] using
              ApproxTensor.eps_nonneg h
      | cons i tail =>
          by_cases hi : i < n
          · simpa [getAtOrZero, hi] using
              ih (ApproxTensor.unstack h ⟨i, hi⟩) tail
          · simpa [getAtOrZero, hi,
              toReal_zero (β := β) (fexp := fexp) (rnd := rnd)] using
                ApproxTensor.eps_nonneg h

/-! ## Forward convolution -/

/-- Input element for an output position and `(channel, offset)` pair, or zero outside the input. -/
def convInputValue
    {α : Type} [Context α]
    {d inC : Nat} {kernel stride padding inSpatial : Tensor Nat [d]}
    (input : Tensor α (inC :: (Tensor.to inSpatial (List Nat))))
    (outIndex : Shape.Coord
      (Tensor.to (outSpatialDilated inSpatial kernel stride
          (Tensor.ones [d]) padding padding) (List Nat)))
    (index : Fin inC × List Nat) : α :=
  match mkInputIdx? outIndex.toList index.2 (Tensor.to stride (List Nat))
      (Tensor.to padding (List Nat)) with
  | none => 0
  | some inputIndex => getAtOrZero input (index.1.val :: inputIndex)

/-- Weight used by one `(input channel, kernel offset)` pair at the selected output channel. -/
def convWeightValue
    {α : Type} [Context α]
    {d inC outC : Nat} {kernel : Tensor Nat [d]}
    (weights : Tensor α (outC :: inC :: (Tensor.to kernel (List Nat))))
    (outChannel : Fin outC) (index : Fin inC × List Nat) : α :=
  getAtOrZero weights (outChannel.val :: index.1.val :: index.2)

/--
Absolute-error budget for one coordinate of a rounded arbitrary-rank convolution.

The budget follows the spec's multiplication and accumulation order, then accounts for the final
bias addition. It is conservative, because padded reads get the uniform input-error budget although
their exact and rounded values are both zero.
-/
def convPointError
    {d inC outC : Nat} {kernel stride padding inSpatial : Tensor Nat [d]}
    (layer : Conv d inC outC kernel stride padding R)
    (input : Tensor R (inC :: (Tensor.to inSpatial (List Nat))))
    (weightError biasError inputError : ℝ)
    (outChannel : Fin outC)
    (outIndex :
      Shape.Coord
        (Tensor.to (outSpatialDilated inSpatial kernel stride
            (Tensor.ones [d]) padding padding) (List Nat))) : ℝ :=
  let indices := (List.finRange inC).product (enumerateIndices (Tensor.to kernel (List Nat)))
  let inputValue := convInputValue input outIndex
  let weightValue := convWeightValue layer.kernel outChannel
  let sumValue := indices.foldl
    (fun acc index => acc + inputValue index * weightValue index) 0
  let sumError := productFoldError (β := β) (fexp := fexp) (rnd := rnd)
    indices inputValue weightValue (fun _ => inputError) (fun _ => weightError)
  accumulationError (β := β) (fexp := fexp) (rnd := rnd)
    sumValue (getAtOrZero layer.bias [outChannel.val]) sumError biasError

/-- Given uniform bounds on the kernel, bias and input, each output coordinate of the rounded
convolution is within `convPointError` of the ideal one. -/
theorem approx_conv_forward
    {d inC outC : Nat} {kernel stride padding inSpatial : Tensor Nat [d]}
    {idealLayer : Conv d inC outC kernel stride padding ℝ}
    {roundedLayer : Conv d inC outC kernel stride padding R}
    {idealInput : Tensor ℝ (inC :: (Tensor.to inSpatial (List Nat)))}
    {roundedInput : Tensor R (inC :: (Tensor.to inSpatial (List Nat)))}
    {weightError biasError inputError : ℝ}
    (hWeight : ApproxTensor
      (toSpec := NF.toReal)
      idealLayer.kernel roundedLayer.kernel weightError)
    (hBias : ApproxTensor
      (toSpec := NF.toReal)
      idealLayer.bias roundedLayer.bias biasError)
    (hInput : ApproxTensor
      (toSpec := NF.toReal)
      idealInput roundedInput inputError)
    (outChannel : Fin outC)
    (outIndex :
      Shape.Coord (Tensor.to (outSpatialDilated inSpatial kernel stride
          (Tensor.ones [d]) padding padding) (List Nat))) :
    abs
        (NF.toReal
            ((forward roundedLayer roundedInput) (outChannel, outIndex)) -
          (forward idealLayer idealInput) (outChannel, outIndex)) ≤
      convPointError (β := β) (fexp := fexp) (rnd := rnd)
        roundedLayer roundedInput weightError biasError inputError outChannel outIndex := by
  let indices := (List.finRange inC).product (enumerateIndices (Tensor.to kernel (List Nat)))
  let inputIdeal := convInputValue idealInput outIndex
  let inputRounded := convInputValue roundedInput outIndex
  let weightIdeal := convWeightValue idealLayer.kernel outChannel
  let weightRounded := convWeightValue roundedLayer.kernel outChannel
  have hInputValue : ∀ index ∈ indices,
      abs (NF.toReal (inputRounded index) -
        inputIdeal index) ≤ inputError := by
    intro index _
    simp only [inputRounded, inputIdeal, convInputValue]
    split
    · simpa [toReal_zero (β := β) (fexp := fexp) (rnd := rnd)] using
        ApproxTensor.eps_nonneg hInput
    · exact approx_getAtOrZero (β := β) (fexp := fexp) (rnd := rnd) hInput _
  have hWeightValue : ∀ index ∈ indices,
      abs (NF.toReal (weightRounded index) -
        weightIdeal index) ≤ weightError := by
    intro index _
    exact approx_getAtOrZero (β := β) (fexp := fexp) (rnd := rnd) hWeight _
  have hSum := approx_product_fold (β := β) (fexp := fexp) (rnd := rnd)
    indices inputIdeal weightIdeal inputRounded weightRounded
    (fun _ => inputError) (fun _ => weightError) hInputValue hWeightValue
  have hBiasValue := approx_getAtOrZero
    (β := β) (fexp := fexp) (rnd := rnd) hBias [outChannel.val]
  have hFinal := approx_add (β := β) (fexp := fexp) (rnd := rnd)
    (x := indices.foldl (fun acc index => acc + inputIdeal index * weightIdeal index) 0)
    (y := getAtOrZero idealLayer.bias [outChannel.val])
    (xR := indices.foldl
      (fun acc index => acc + inputRounded index * weightRounded index) 0)
    (yR := getAtOrZero roundedLayer.bias [outChannel.val])
    (epsx := productFoldError (β := β) (fexp := fexp) (rnd := rnd)
      indices inputRounded weightRounded (fun _ => inputError) (fun _ => weightError))
    (epsy := biasError) hSum hBiasValue
  have hRounded :
      (forward roundedLayer roundedInput) (outChannel, outIndex) =
        indices.foldl
            (fun acc index => acc + inputRounded index * weightRounded index) 0 +
          getAtOrZero roundedLayer.bias [outChannel.val] := by
    rw [forward, Tensor.add_apply,
      get_conv_contract, get_conv_biasBroadcast,
      foldChannelsIndices_eq_foldl]
    rfl
  have hIdeal :
      (forward idealLayer idealInput) (outChannel, outIndex) =
        indices.foldl
            (fun acc index => acc + inputIdeal index * weightIdeal index) 0 +
          getAtOrZero idealLayer.bias [outChannel.val] := by
    rw [forward, Tensor.add_apply,
      get_conv_contract, get_conv_biasBroadcast,
      foldChannelsIndices_eq_foldl]
    rfl
  rw [hRounded, hIdeal]
  simpa [convPointError, indices, inputRounded, weightRounded,
    accumulationError] using hFinal

/-! ## Backward convolution -/

private theorem get_conv_kernelDeriv
    {α : Type} [Context α]
    {d inC outC : Nat} {kernel stride padding inSpatial : Tensor Nat [d]}
    (layer : Conv d inC outC kernel stride padding α)
    (input : Tensor α (inC :: (Tensor.to inSpatial (List Nat))))
    (gradOutput : Tensor α
      (outC :: (Tensor.to (outSpatialDilated inSpatial kernel stride
            (Tensor.ones [d]) padding padding) (List Nat))))
    (outChannel : Fin outC) (inChannel : Fin inC)
    (kernelIndex : Shape.Coord (Tensor.to kernel (List Nat))) :
    (kernelDeriv layer input gradOutput) (outChannel, (inChannel, kernelIndex)) =
      foldlIndices (Tensor.to (outSpatialDilated inSpatial kernel stride
          (Tensor.ones [d]) padding padding) (List Nat)) 0
        (fun acc outIndex =>
          acc +
            (match mkInputIdx? outIndex kernelIndex.toList (Tensor.to stride (List Nat))
                (Tensor.to padding (List Nat)) with
              | none => 0
              | some inputIndex => getAtOrZero input (inChannel.val :: inputIndex)) *
            getAtOrZero gradOutput (outChannel.val :: outIndex)) := by
  simp only [kernelDeriv, Tensor.dim_apply]
  rw [Tensor.generate, Tensor.Rep.get_ofFn]
  rfl

/--
The input value entering one term of the kernel-gradient accumulation, zero at padded positions.
-/
def kernelGradientInputValue
    {α : Type} [Context α]
    {d inC : Nat} {kernel inSpatial : Tensor Nat [d]}
    (stride padding : Tensor Nat [d])
    (input : Tensor α (inC :: (Tensor.to inSpatial (List Nat))))
    (inChannel : Fin inC) (kernelIndex : Shape.Coord (Tensor.to kernel (List Nat)))
    (outIndex : List Nat) : α :=
  match mkInputIdx? outIndex kernelIndex.toList (Tensor.to stride (List Nat))
      (Tensor.to padding (List Nat)) with
  | none => 0
  | some inputIndex => getAtOrZero input (inChannel.val :: inputIndex)

/-- The output-gradient value entering that same term. -/
def kernelGradientOutputValue
    {α : Type} [Context α]
    {d outC : Nat} {kernel stride padding inSpatial : Tensor Nat [d]}
    (gradOutput : Tensor α
      (outC :: (Tensor.to (outSpatialDilated inSpatial kernel stride
            (Tensor.ones [d]) padding padding) (List Nat))))
    (outChannel : Fin outC) (outIndex : List Nat) : α :=
  getAtOrZero gradOutput (outChannel.val :: outIndex)

/-- Error budget for one coordinate of the rounded convolution kernel gradient. -/
def convKernelGradPointError
    {d inC outC : Nat} {kernel stride padding inSpatial : Tensor Nat [d]}
    (input : Tensor R (inC :: (Tensor.to inSpatial (List Nat))))
    (gradOutput : Tensor R
      (outC :: (Tensor.to (outSpatialDilated inSpatial kernel stride
            (Tensor.ones [d]) padding padding) (List Nat))))
    (inputError gradOutputError : ℝ)
    (outChannel : Fin outC) (inChannel : Fin inC)
    (kernelIndex : Shape.Coord (Tensor.to kernel (List Nat))) : ℝ :=
  let indices :=
    enumerateIndices (Tensor.to (outSpatialDilated inSpatial kernel stride
        (Tensor.ones [d]) padding padding) (List Nat))
  productFoldError (β := β) (fexp := fexp) (rnd := rnd) indices
    (kernelGradientInputValue stride padding input inChannel kernelIndex)
    (kernelGradientOutputValue gradOutput outChannel)
    (fun _ => inputError) (fun _ => gradOutputError)

/-- Given uniform bounds on the input and the output gradient, each coordinate of the rounded kernel
gradient is within `convKernelGradPointError` of the ideal one. -/
theorem approx_conv_kernelDeriv
    {d inC outC : Nat} {kernel stride padding inSpatial : Tensor Nat [d]}
    {idealLayer : Conv d inC outC kernel stride padding ℝ}
    {roundedLayer : Conv d inC outC kernel stride padding R}
    {idealInput : Tensor ℝ (inC :: (Tensor.to inSpatial (List Nat)))}
    {roundedInput : Tensor R (inC :: (Tensor.to inSpatial (List Nat)))}
    {idealGradOutput : Tensor ℝ
      (outC :: (Tensor.to (outSpatialDilated inSpatial kernel stride
            (Tensor.ones [d]) padding padding) (List Nat)))}
    {roundedGradOutput : Tensor R
      (outC :: (Tensor.to (outSpatialDilated inSpatial kernel stride
            (Tensor.ones [d]) padding padding) (List Nat)))}
    {inputError gradOutputError : ℝ}
    (hInput : ApproxTensor
      (toSpec := NF.toReal)
      idealInput roundedInput inputError)
    (hGradOutput : ApproxTensor
      (toSpec := NF.toReal)
      idealGradOutput roundedGradOutput gradOutputError)
    (outChannel : Fin outC) (inChannel : Fin inC)
    (kernelIndex : Shape.Coord (Tensor.to kernel (List Nat))) :
    abs
        (NF.toReal
            ((kernelDeriv roundedLayer roundedInput roundedGradOutput)
              (outChannel, (inChannel, kernelIndex))) -
          (kernelDeriv idealLayer idealInput idealGradOutput)
            (outChannel, (inChannel, kernelIndex))) ≤
      convKernelGradPointError
        (β := β) (fexp := fexp) (rnd := rnd)
        roundedInput roundedGradOutput inputError gradOutputError
        outChannel inChannel kernelIndex := by
  let indices :=
    enumerateIndices (Tensor.to (outSpatialDilated inSpatial kernel stride
        (Tensor.ones [d]) padding padding) (List Nat))
  let inputIdeal := kernelGradientInputValue stride padding idealInput inChannel kernelIndex
  let inputRounded := kernelGradientInputValue stride padding roundedInput inChannel kernelIndex
  let gradIdeal := kernelGradientOutputValue idealGradOutput outChannel
  let gradRounded := kernelGradientOutputValue roundedGradOutput outChannel
  have hInputValue : ∀ index ∈ indices,
      abs (NF.toReal (inputRounded index) -
        inputIdeal index) ≤ inputError := by
    intro index _
    simp only [inputRounded, inputIdeal, kernelGradientInputValue]
    split
    · simpa [toReal_zero (β := β) (fexp := fexp) (rnd := rnd)] using
        ApproxTensor.eps_nonneg hInput
    · exact approx_getAtOrZero (β := β) (fexp := fexp) (rnd := rnd) hInput _
  have hGradValue : ∀ index ∈ indices,
      abs (NF.toReal (gradRounded index) -
        gradIdeal index) ≤ gradOutputError := by
    intro index _
    exact approx_getAtOrZero (β := β) (fexp := fexp) (rnd := rnd) hGradOutput _
  have hBound := approx_product_fold (β := β) (fexp := fexp) (rnd := rnd)
    indices inputIdeal gradIdeal inputRounded gradRounded
    (fun _ => inputError) (fun _ => gradOutputError) hInputValue hGradValue
  have hRounded :
      (kernelDeriv roundedLayer roundedInput roundedGradOutput)
        (outChannel, (inChannel, kernelIndex)) =
        indices.foldl
          (fun acc index => acc + inputRounded index * gradRounded index) 0 := by
    rw [get_conv_kernelDeriv, foldlIndices_eq_foldl]
    rfl
  have hIdeal :
      (kernelDeriv idealLayer idealInput idealGradOutput)
        (outChannel, (inChannel, kernelIndex)) =
        indices.foldl
          (fun acc index => acc + inputIdeal index * gradIdeal index) 0 := by
    rw [get_conv_kernelDeriv, foldlIndices_eq_foldl]
    rfl
  rw [hRounded, hIdeal]
  simpa only [convKernelGradPointError, indices] using hBound

private theorem get_conv_biasDeriv
    {α : Type} [Context α]
    {d inC outC : Nat} {kernel stride padding inSpatial : Tensor Nat [d]}
    (layer : Conv d inC outC kernel stride padding α)
    (input : Tensor α (inC :: (Tensor.to inSpatial (List Nat))))
    (gradOutput : Tensor α
      (outC :: (Tensor.to (outSpatialDilated inSpatial kernel stride
            (Tensor.ones [d]) padding padding) (List Nat))))
    (outChannel : Fin outC) :
    (biasDeriv layer input gradOutput) (outChannel, PUnit.unit) =
      foldlIndices (Tensor.to (outSpatialDilated inSpatial kernel stride
          (Tensor.ones [d]) padding padding) (List Nat)) 0
        (fun acc outIndex =>
          acc + getAtOrZero gradOutput (outChannel.val :: outIndex)) := by
  simp [biasDeriv, Tensor.dim_apply]

/-- Error budget for one coordinate of the rounded convolution bias gradient. -/
def convBiasGradPointError
    {d outC : Nat} {kernel stride padding inSpatial : Tensor Nat [d]}
    (gradOutput : Tensor R
      (outC :: (Tensor.to (outSpatialDilated inSpatial kernel stride
            (Tensor.ones [d]) padding padding) (List Nat))))
    (gradOutputError : ℝ) (outChannel : Fin outC) : ℝ :=
  let indices :=
    enumerateIndices (Tensor.to (outSpatialDilated inSpatial kernel stride
        (Tensor.ones [d]) padding padding) (List Nat))
  foldError (β := β) (fexp := fexp) (rnd := rnd) indices
    (fun index => getAtOrZero gradOutput (outChannel.val :: index))
    (fun _ => gradOutputError)

/-- Given a uniform bound on the output gradient, each coordinate of the rounded bias gradient is
within `convBiasGradPointError` of the ideal one. -/
theorem approx_conv_biasDeriv
    {d inC outC : Nat} {kernel stride padding inSpatial : Tensor Nat [d]}
    {idealLayer : Conv d inC outC kernel stride padding ℝ}
    {roundedLayer : Conv d inC outC kernel stride padding R}
    {idealInput : Tensor ℝ (inC :: (Tensor.to inSpatial (List Nat)))}
    {roundedInput : Tensor R (inC :: (Tensor.to inSpatial (List Nat)))}
    {idealGradOutput : Tensor ℝ
      (outC :: (Tensor.to (outSpatialDilated inSpatial kernel stride
            (Tensor.ones [d]) padding padding) (List Nat)))}
    {roundedGradOutput : Tensor R
      (outC :: (Tensor.to (outSpatialDilated inSpatial kernel stride
            (Tensor.ones [d]) padding padding) (List Nat)))}
    {gradOutputError : ℝ}
    (hGradOutput : ApproxTensor
      (toSpec := NF.toReal)
      idealGradOutput roundedGradOutput gradOutputError)
    (outChannel : Fin outC) :
    abs
        (NF.toReal
            ((biasDeriv roundedLayer roundedInput roundedGradOutput)
              (outChannel, PUnit.unit)) -
          (biasDeriv idealLayer idealInput idealGradOutput) (outChannel, PUnit.unit)) ≤
      convBiasGradPointError
        (β := β) (fexp := fexp) (rnd := rnd)
        roundedGradOutput gradOutputError outChannel := by
  let indices :=
    enumerateIndices (Tensor.to (outSpatialDilated inSpatial kernel stride
        (Tensor.ones [d]) padding padding) (List Nat))
  let idealValue := fun index => getAtOrZero idealGradOutput (outChannel.val :: index)
  let roundedValue := fun index => getAtOrZero roundedGradOutput (outChannel.val :: index)
  have hValue : ∀ index ∈ indices,
      abs (NF.toReal (roundedValue index) -
        idealValue index) ≤ gradOutputError := by
    intro index _
    exact approx_getAtOrZero (β := β) (fexp := fexp) (rnd := rnd) hGradOutput _
  have hBound := approx_fold (β := β) (fexp := fexp) (rnd := rnd)
    indices idealValue roundedValue (fun _ => gradOutputError) hValue
  have hRounded :
      (biasDeriv roundedLayer roundedInput roundedGradOutput) (outChannel, PUnit.unit) =
        indices.foldl (fun acc index => acc + roundedValue index) 0 := by
    rw [get_conv_biasDeriv, foldlIndices_eq_foldl]
  have hIdeal :
      (biasDeriv idealLayer idealInput idealGradOutput) (outChannel, PUnit.unit) =
        indices.foldl (fun acc index => acc + idealValue index) 0 := by
    rw [get_conv_biasDeriv, foldlIndices_eq_foldl]
  rw [hRounded, hIdeal]
  simpa only [convBiasGradPointError, indices] using hBound

private theorem get_conv_inputDeriv
    {α : Type} [Context α]
    {d inC outC : Nat} {kernel stride padding inSpatial : Tensor Nat [d]}
    (layer : Conv d inC outC kernel stride padding α)
    (input : Tensor α (inC :: (Tensor.to inSpatial (List Nat))))
    (gradOutput : Tensor α
      (outC :: (Tensor.to (outSpatialDilated inSpatial kernel stride
            (Tensor.ones [d]) padding padding) (List Nat))))
    (inChannel : Fin inC) (inputIndex : Shape.Coord (Tensor.to inSpatial (List Nat))) :
    (inputDeriv layer input gradOutput) (inChannel, inputIndex) =
      (List.finRange outC).foldl (fun acc outChannel =>
        foldlIndices (Tensor.to kernel (List Nat)) acc (fun acc kernelIndex =>
          acc +
            (match mkTransposeInputIdx? inputIndex.toList kernelIndex
                (Tensor.to stride (List Nat)) (Tensor.to padding (List Nat)) with
              | none => 0
              | some outIndex =>
                  getAtOrZero gradOutput (outChannel.val :: outIndex) *
                    getAtOrZero layer.kernel
                      (outChannel.val :: inChannel.val :: kernelIndex)))) 0 := by
  simp only [inputDeriv, Tensor.dim_apply]
  rw [Tensor.generate, Tensor.Rep.get_ofFn]
  rfl

/-- Input-gradient summand: an output gradient times its kernel weight, or zero when the transposed
index has no corresponding output position. -/
def inputGradientTermValue
    {α : Type} [Context α]
    {d inC outC : Nat} {kernel stride padding inSpatial : Tensor Nat [d]}
    (layer : Conv d inC outC kernel stride padding α)
    (gradOutput : Tensor α
      (outC :: (Tensor.to (outSpatialDilated inSpatial kernel stride
            (Tensor.ones [d]) padding padding) (List Nat))))
    (inChannel : Fin inC) (inputIndex : Shape.Coord (Tensor.to inSpatial (List Nat)))
    (index : Fin outC × List Nat) : α :=
  match mkTransposeInputIdx? inputIndex.toList index.2 (Tensor.to stride (List Nat))
      (Tensor.to padding (List Nat)) with
  | none => 0
  | some outIndex =>
      getAtOrZero gradOutput (index.1.val :: outIndex) *
        getAtOrZero layer.kernel (index.1.val :: inChannel.val :: index.2)

/-- Error budget for one coordinate of the rounded convolution input gradient. -/
def convInputGradPointError
    {d inC outC : Nat} {kernel stride padding inSpatial : Tensor Nat [d]}
    (layer : Conv d inC outC kernel stride padding R)
    (gradOutput : Tensor R
      (outC :: (Tensor.to (outSpatialDilated inSpatial kernel stride
            (Tensor.ones [d]) padding padding) (List Nat))))
    (weightError gradOutputError : ℝ)
    (inChannel : Fin inC) (inputIndex : Shape.Coord (Tensor.to inSpatial (List Nat))) : ℝ :=
  let indices := (List.finRange outC).product (enumerateIndices (Tensor.to kernel (List Nat)))
  foldError (β := β) (fexp := fexp) (rnd := rnd) indices
    (inputGradientTermValue layer gradOutput inChannel inputIndex)
    (fun index =>
      match mkTransposeInputIdx? inputIndex.toList index.2
          (Tensor.to stride (List Nat)) (Tensor.to padding (List Nat)) with
      | none => 0
      | some outIndex =>
          productError (β := β) (fexp := fexp) (rnd := rnd)
            (getAtOrZero gradOutput (index.1.val :: outIndex))
            (getAtOrZero layer.kernel (index.1.val :: inChannel.val :: index.2))
            gradOutputError weightError)

/-- Given uniform bounds on the kernel and the output gradient, each coordinate of the rounded input
gradient is within `convInputGradPointError` of the ideal one. -/
theorem approx_conv_inputDeriv
    {d inC outC : Nat} {kernel stride padding inSpatial : Tensor Nat [d]}
    {idealLayer : Conv d inC outC kernel stride padding ℝ}
    {roundedLayer : Conv d inC outC kernel stride padding R}
    {idealInput : Tensor ℝ (inC :: (Tensor.to inSpatial (List Nat)))}
    {roundedInput : Tensor R (inC :: (Tensor.to inSpatial (List Nat)))}
    {idealGradOutput : Tensor ℝ
      (outC :: (Tensor.to (outSpatialDilated inSpatial kernel stride
            (Tensor.ones [d]) padding padding) (List Nat)))}
    {roundedGradOutput : Tensor R
      (outC :: (Tensor.to (outSpatialDilated inSpatial kernel stride
            (Tensor.ones [d]) padding padding) (List Nat)))}
    {weightError gradOutputError : ℝ}
    (hWeight : ApproxTensor
      (toSpec := NF.toReal)
      idealLayer.kernel roundedLayer.kernel weightError)
    (hGradOutput : ApproxTensor
      (toSpec := NF.toReal)
      idealGradOutput roundedGradOutput gradOutputError)
    (inChannel : Fin inC) (inputIndex : Shape.Coord (Tensor.to inSpatial (List Nat))) :
    abs
        (NF.toReal
            ((inputDeriv roundedLayer roundedInput roundedGradOutput)
              (inChannel, inputIndex)) -
          (inputDeriv idealLayer idealInput idealGradOutput) (inChannel, inputIndex)) ≤
      convInputGradPointError
        (β := β) (fexp := fexp) (rnd := rnd)
        roundedLayer roundedGradOutput weightError gradOutputError
        inChannel inputIndex := by
  let indices := (List.finRange outC).product (enumerateIndices (Tensor.to kernel (List Nat)))
  let idealTerm := inputGradientTermValue idealLayer idealGradOutput inChannel inputIndex
  let roundedTerm := inputGradientTermValue roundedLayer roundedGradOutput inChannel inputIndex
  let termError := fun (index : Fin outC × List Nat) =>
    match mkTransposeInputIdx? inputIndex.toList index.2 (Tensor.to stride (List Nat))
        (Tensor.to padding (List Nat)) with
    | none => 0
    | some outIndex =>
        productError (β := β) (fexp := fexp) (rnd := rnd)
          (getAtOrZero roundedGradOutput (index.1.val :: outIndex))
          (getAtOrZero roundedLayer.kernel
            (index.1.val :: inChannel.val :: index.2))
          gradOutputError weightError
  have hTerm : ∀ index ∈ indices,
      abs (NF.toReal (roundedTerm index) -
        idealTerm index) ≤ termError index := by
    intro index _
    cases hIndex : mkTransposeInputIdx? inputIndex.toList index.2
        (Tensor.to stride (List Nat)) (Tensor.to padding (List Nat)) with
    | none =>
        have hIndexData := hIndex
        simp only [Tensor.to_list_eq_data] at hIndexData
        simp [roundedTerm, idealTerm, termError, inputGradientTermValue, hIndexData,
          toReal_zero (β := β) (fexp := fexp) (rnd := rnd)]
    | some outIndex =>
        have hIndexData := hIndex
        simp only [Tensor.to_list_eq_data] at hIndexData
        have hGrad := approx_getAtOrZero
          (β := β) (fexp := fexp) (rnd := rnd) hGradOutput
          (index.1.val :: outIndex)
        have hKernel := approx_getAtOrZero
          (β := β) (fexp := fexp) (rnd := rnd) hWeight
          (index.1.val :: inChannel.val :: index.2)
        simpa [roundedTerm, idealTerm, termError, inputGradientTermValue,
          hIndex, hIndexData, productError] using
          (approx_mul (β := β) (fexp := fexp) (rnd := rnd) hGrad hKernel)
  have hBound := approx_fold (β := β) (fexp := fexp) (rnd := rnd)
    indices idealTerm roundedTerm termError hTerm
  have hRounded :
      (inputDeriv roundedLayer roundedInput roundedGradOutput) (inChannel, inputIndex) =
        indices.foldl
          (fun acc index => acc + roundedTerm index) 0 := by
    rw [get_conv_inputDeriv, foldChannelsIndices_eq_foldl]
    rfl
  have hIdeal :
      (inputDeriv idealLayer idealInput idealGradOutput) (inChannel, inputIndex) =
        indices.foldl
          (fun acc index => acc + idealTerm index) 0 := by
    rw [get_conv_inputDeriv, foldChannelsIndices_eq_foldl]
    rfl
  rw [hRounded, hIdeal]
  change
    abs
        (NF.toReal
            (indices.foldl (fun acc index => acc + roundedTerm index) 0) -
          indices.foldl (fun acc index => acc + idealTerm index) 0) ≤
      foldError (β := β) (fexp := fexp) (rnd := rnd)
        indices roundedTerm termError
  exact hBound

end
end TorchLean.NF
