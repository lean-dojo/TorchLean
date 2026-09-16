/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Tape.Ops.Conv.Index
public import NN.Proofs.RuntimeApprox.NF.FoldLemmas
public import NN.Tensor.Conversion
public import NN.Proofs.RuntimeApprox.NF.Ops.Scalar

/-!
# Rounded Convolution

Forward- and reverse-mode error bounds for channels-first convolution at arbitrary spatial rank.
The bounds replay the exact ordered folds in `Spec.convSpec` and `Spec.convBackwardSpec`; no
associativity of rounded addition is assumed.
-/

@[expose] public section

namespace Proofs.RuntimeApprox.NFBackend

open Spec TorchLean
open TorchLean.Tensor
open Spec.Conv.Internal
open FloatLib FloatLib.Numerics FloatLib.Floats.Formats
open Flocq

noncomputable section

variable {beta : Radix} {fexp : ℤ → ℤ} [ValidExp fexp]
variable {rnd : ℝ → ℤ} [ValidRndToNearest rnd]

local notation "R" => NF beta fexp rnd

/-! ## Ordered rounded sums -/

/-- Error budget after one rounded addition. -/
def accumulationError (acc term : R) (accError termError : ℝ) : ℝ :=
  accError + termError +
    ulp beta fexp
      (toSpec (β := beta) (fexp := fexp) (rnd := rnd) acc +
        toSpec (β := beta) (fexp := fexp) (rnd := rnd) term) / 2

/-- Error budget after one rounded multiplication. -/
def productError (x y : R) (xError yError : ℝ) : ℝ :=
  let xValue := toSpec (β := beta) (fexp := fexp) (rnd := rnd) x
  let yValue := toSpec (β := beta) (fexp := fexp) (rnd := rnd) y
  (abs xValue + xError) * yError + (abs yValue + yError) * xError +
    ulp beta fexp (xValue * yValue) / 2

/-- Replay a rounded sum while carrying its absolute-error budget. -/
def foldErrorState {iota : Type} (indices : List iota)
    (term : iota → R) (termError : iota → ℝ) (initial : R × ℝ) : R × ℝ :=
  indices.foldl (fun state index =>
    let value := term index
    (state.1 + value,
      accumulationError (beta := beta) (fexp := fexp) (rnd := rnd)
        state.1 value state.2 (termError index))) initial

/-- Error budget for a rounded sum beginning at zero. -/
def foldError {iota : Type} (indices : List iota)
    (term : iota → R) (termError : iota → ℝ) : ℝ :=
  (foldErrorState (beta := beta) (fexp := fexp) (rnd := rnd)
    indices term termError (0, 0)).2

private theorem approx_fold_state {iota : Type} (indices : List iota)
    (ideal : iota → ℝ) (rounded : iota → R) (termError : iota → ℝ) :
    ∀ (idealAcc : ℝ) (state : R × ℝ),
      abs (toSpec (β := beta) (fexp := fexp) (rnd := rnd) state.1 - idealAcc) ≤ state.2 →
      (∀ index ∈ indices,
        abs (toSpec (β := beta) (fexp := fexp) (rnd := rnd) (rounded index) -
          ideal index) ≤ termError index) →
      abs
          (toSpec (β := beta) (fexp := fexp) (rnd := rnd)
              (indices.foldl (fun acc index => acc + rounded index) state.1) -
            indices.foldl (fun acc index => acc + ideal index) idealAcc) ≤
        (foldErrorState (beta := beta) (fexp := fexp) (rnd := rnd)
          indices rounded termError state).2 := by
  intro idealAcc state hAcc hTerm
  induction indices generalizing idealAcc state with
  | nil => simpa [foldErrorState] using hAcc
  | cons head tail ih =>
      have hHead := hTerm head (by simp)
      have hNext :
          abs
              (toSpec (β := beta) (fexp := fexp) (rnd := rnd)
                  (state.1 + rounded head) -
                (idealAcc + ideal head)) ≤
            accumulationError (beta := beta) (fexp := fexp) (rnd := rnd)
              state.1 (rounded head) state.2 (termError head) := by
        simpa [accumulationError, add_assoc, add_comm, add_left_comm] using
          (approx_add_nf (β := beta) (fexp := fexp) (rnd := rnd)
            (x := idealAcc) (y := ideal head) (xR := state.1) (yR := rounded head)
            (epsx := state.2) (epsy := termError head) hAcc hHead)
      simpa [foldErrorState] using
        ih (idealAcc + ideal head)
          (state.1 + rounded head,
            accumulationError (beta := beta) (fexp := fexp) (rnd := rnd)
              state.1 (rounded head) state.2 (termError head))
          hNext (by
            intro index hIndex
            exact hTerm index (by simp [hIndex]))

/-- A fold of individually bounded rounded terms is bounded by `foldError`. -/
theorem approx_fold {iota : Type} (indices : List iota)
    (ideal : iota → ℝ) (rounded : iota → R) (termError : iota → ℝ)
    (hTerm : ∀ index ∈ indices,
      abs (toSpec (β := beta) (fexp := fexp) (rnd := rnd) (rounded index) -
        ideal index) ≤ termError index) :
    abs
        (toSpec (β := beta) (fexp := fexp) (rnd := rnd)
            (indices.foldl (fun acc index => acc + rounded index) 0) -
          indices.foldl (fun acc index => acc + ideal index) 0) ≤
      foldError (beta := beta) (fexp := fexp) (rnd := rnd)
        indices rounded termError := by
  have hZero :
      abs (toSpec (β := beta) (fexp := fexp) (rnd := rnd) (0 : R) - 0) ≤ (0 : ℝ) := by
    simp [toSpec_zero (β := beta) (fexp := fexp) (rnd := rnd)]
  simpa [foldError, foldErrorState] using
    approx_fold_state (beta := beta) (fexp := fexp) (rnd := rnd)
      indices ideal rounded termError 0 (0, 0) hZero hTerm

/-- Replay bound for a dot product whose two arguments are already approximated pointwise. -/
def productFoldError {iota : Type} (indices : List iota)
    (left right : iota → R) (leftError rightError : iota → ℝ) : ℝ :=
  foldError (beta := beta) (fexp := fexp) (rnd := rnd) indices
    (fun index => left index * right index)
    (fun index => productError (beta := beta) (fexp := fexp) (rnd := rnd)
      (left index) (right index) (leftError index) (rightError index))

/-- An ordered rounded dot product is enclosed by `productFoldError`. -/
theorem approx_product_fold {iota : Type} (indices : List iota)
    (leftIdeal rightIdeal : iota → ℝ) (leftRounded rightRounded : iota → R)
    (leftError rightError : iota → ℝ)
    (hLeft : ∀ index ∈ indices,
      abs (toSpec (β := beta) (fexp := fexp) (rnd := rnd) (leftRounded index) -
        leftIdeal index) ≤ leftError index)
    (hRight : ∀ index ∈ indices,
      abs (toSpec (β := beta) (fexp := fexp) (rnd := rnd) (rightRounded index) -
        rightIdeal index) ≤ rightError index) :
    abs
        (toSpec (β := beta) (fexp := fexp) (rnd := rnd)
            (indices.foldl
              (fun acc index => acc + leftRounded index * rightRounded index) 0) -
          indices.foldl
            (fun acc index => acc + leftIdeal index * rightIdeal index) 0) ≤
      productFoldError (beta := beta) (fexp := fexp) (rnd := rnd)
        indices leftRounded rightRounded leftError rightError := by
  apply approx_fold (beta := beta) (fexp := fexp) (rnd := rnd)
  intro index hIndex
  simpa [productError] using
    (approx_mul_nf (β := beta) (fexp := fexp) (rnd := rnd)
      (x := leftIdeal index) (y := rightIdeal index)
      (xR := leftRounded index) (yR := rightRounded index)
      (epsx := leftError index) (epsy := rightError index)
      (hLeft index hIndex) (hRight index hIndex))

/-! ## Arbitrary-rank index traversal -/

/-- Runtime order of all bounded coordinates of a list-shaped tensor. -/
def enumerateIndices : List Nat → List (List Nat)
  | [] => [[]]
  | n :: dims =>
      (List.finRange n).flatMap fun index =>
        (enumerateIndices dims).map fun tail => index.val :: tail

/-- The nested index loop equals a single fold over the flattened index list.

This is the workhorse of the file. The implementation loops over dimensions recursively, whereas an
error bound is much easier to state as one sum over coordinates; turning one into the other once,
here,
keeps every later bound free of nested inductions. -/
theorem foldlIndices_eq_enumerateIndices {a : Type} (dims : List Nat)
    (initial : a) (step : a → List Nat → a) :
    foldlIndices dims initial step =
      (enumerateIndices dims).foldl step initial := by
  induction dims generalizing initial step with
  | nil => rfl
  | cons n dims ih =>
      simp only [foldlIndices, enumerateIndices, foldl_flatMap]
      apply foldl_congr
      intro state index
      rw [ih]
      simp [List.foldl_map]

/-- Flatten a channel loop followed by an arbitrary spatial index loop. -/
def enumerateChannelIndices (channels : Nat) (dims : List Nat) :
    List (Fin channels × List Nat) :=
  (List.finRange channels).flatMap fun channel =>
    (enumerateIndices dims).map fun index => (channel, index)

/-- Same flattening one level up, for the channel loop wrapped around the spatial loop. -/
theorem foldChannelsIndices_eq_foldl {a : Type} (channels : Nat) (dims : List Nat)
    (initial : a) (step : a → Fin channels → List Nat → a) :
    (List.finRange channels).foldl
        (fun state channel => foldlIndices dims state (step · channel ·)) initial =
      (enumerateChannelIndices channels dims).foldl
        (fun state index => step state index.1 index.2) initial := by
  simp only [enumerateChannelIndices, foldl_flatMap]
  apply foldl_congr
  intro state channel
  rw [foldlIndices_eq_enumerateIndices]
  simp [List.foldl_map]

/-! ## Tensor coordinates -/

private theorem get_convCoreSpec
    {alpha : Type} [Context alpha]
    {d inC outC : Nat} {kernel stride padding inSpatial : TorchLean.Tensor Nat [d]}
    (weights : Tensor alpha (Shape.ofList (outC :: inC :: (Tensor.to kernel (List Nat)))))
    (input : Tensor alpha (Shape.ofList (inC :: (Tensor.to inSpatial (List Nat)))))
    (outChannel : Fin outC)
    (outIndex :
      MultiIndex (Tensor.to (convOutSpatial inSpatial kernel stride padding) (List Nat))) :
    MultiIndex.get (convCoreSpec (stride := stride) (padding := padding) weights input)
        (outChannel, outIndex) =
      (List.finRange inC).foldl (fun acc inChannel =>
        foldlIndices (Tensor.to kernel (List Nat)) acc (fun acc kernelIndex =>
          acc +
            (match mkInputIdx? outIndex.toList kernelIndex (Tensor.to stride (List Nat))
                (Tensor.to padding (List Nat)) with
              | none => 0
              | some inputIndex => getAtOrZero input (inChannel.val :: inputIndex)) *
            getAtOrZero weights (outChannel.val :: inChannel.val :: kernelIndex))) 0 := by
  simp only [convCoreSpec, Conv.Internal.convCoreWith, MultiIndex.get_dim,
    MultiIndex.get_generate]
  rfl

private theorem get_convBiasBroadcastSpec
    {alpha : Type} [Context alpha]
    {d outC : Nat} {kernel stride padding inSpatial : TorchLean.Tensor Nat [d]}
    (bias : Tensor alpha [outC]) (outChannel : Fin outC)
    (outIndex :
      MultiIndex (Tensor.to (convOutSpatial inSpatial kernel stride padding) (List Nat))) :
    MultiIndex.get
        (convBiasBroadcastSpec (kernel := kernel) (stride := stride)
          (padding := padding) (inSpatial := inSpatial) bias)
        (outChannel, outIndex) = getAtOrZero bias [outChannel.val] := by
  simp only [convBiasBroadcastSpec, Conv.Internal.convBiasBroadcastWith,
    MultiIndex.get_dim, MultiIndex.get_generate]

/-- Total tensor lookup preserves a uniform tensor approximation. -/
theorem approx_getAtOrZero {shape : Shape} {ideal : Tensor ℝ shape}
    {rounded : Tensor R shape} {error : ℝ}
    (h : approxTensor
      (toSpec := toSpec (β := beta) (fexp := fexp) (rnd := rnd)) ideal rounded error)
    (index : List Nat) :
    abs (toSpec (β := beta) (fexp := fexp) (rnd := rnd)
        (getAtOrZero rounded index) - getAtOrZero ideal index) ≤ error := by
  induction shape generalizing index with
  | scalar =>
      cases index with
      | nil =>
          rw [← Tensor.scalar_item ideal, ← Tensor.scalar_item rounded] at h
          simpa [getAtOrZero] using (approxTensor_scalar_iff.mp h)
      | cons _ _ =>
          simpa [getAtOrZero,
            toSpec_zero (β := beta) (fexp := fexp) (rnd := rnd)] using
              approxTensor_eps_nonneg h
  | dim n shape ih =>
      cases index with
      | nil =>
          simpa [getAtOrZero,
            toSpec_zero (β := beta) (fexp := fexp) (rnd := rnd)] using
              approxTensor_eps_nonneg h
      | cons i tail =>
          by_cases hi : i < n
          · simpa [getAtOrZero, hi] using
              ih (approxTensor_dim_get h ⟨i, hi⟩) tail
          · simpa [getAtOrZero, hi,
              toSpec_zero (β := beta) (fexp := fexp) (rnd := rnd)] using
                approxTensor_eps_nonneg h

/-! ## Forward convolution -/

/-- The input element a convolution reads for one output position and one `(channel, offset)` pair,
or zero when the strided, padded index falls outside the input.

Naming the read explicitly is what lets the approximation proof treat padding as an exact zero
rather than as another rounded value. -/
def convolutionInputValue
    {alpha : Type} [Context alpha]
    {d inC : Nat} {kernel stride padding inSpatial : TorchLean.Tensor Nat [d]}
    (input : Tensor alpha (Shape.ofList (inC :: (Tensor.to inSpatial (List Nat)))))
    (outIndex : MultiIndex (Tensor.to (convOutSpatial inSpatial kernel stride padding) (List Nat)))
    (index : Fin inC × List Nat) : alpha :=
  match mkInputIdx? outIndex.toList index.2 (Tensor.to stride (List Nat))
      (Tensor.to padding (List Nat)) with
  | none => 0
  | some inputIndex => getAtOrZero input (index.1.val :: inputIndex)

/-- The weight a given input coordinate is multiplied by, as the implementation reads it.

Named so the error bound can talk about the summands of the accumulation in the same order the code
produces them, which is what makes the bound tight rather than merely valid. -/
def convolutionWeightValue
    {alpha : Type} [Context alpha]
    {d inC outC : Nat} {kernel : TorchLean.Tensor Nat [d]}
    (weights : Tensor alpha (Shape.ofList (outC :: inC :: (Tensor.to kernel (List Nat)))))
    (outChannel : Fin outC) (index : Fin inC × List Nat) : alpha :=
  getAtOrZero weights (outChannel.val :: index.1.val :: index.2)

/--
Absolute-error budget for one coordinate of a rounded arbitrary-rank convolution.

The budget follows the implementation's multiplication and accumulation order, then accounts for
the final bias addition.
-/
def convolutionPointError
    {d inC outC : Nat} {kernel stride padding inSpatial : TorchLean.Tensor Nat [d]}
    (layer : ConvSpec d inC outC kernel stride padding R)
    (input : Tensor R (Shape.ofList (inC :: (Tensor.to inSpatial (List Nat)))))
    (weightError biasError inputError : ℝ)
    (outChannel : Fin outC)
    (outIndex :
      MultiIndex (Tensor.to (convOutSpatial inSpatial kernel stride padding) (List Nat))) : ℝ :=
  let indices := enumerateChannelIndices inC (Tensor.to kernel (List Nat))
  let inputValue := convolutionInputValue input outIndex
  let weightValue := convolutionWeightValue layer.kernel outChannel
  let sumValue := indices.foldl
    (fun acc index => acc + inputValue index * weightValue index) 0
  let sumError := productFoldError (beta := beta) (fexp := fexp) (rnd := rnd)
    indices inputValue weightValue (fun _ => inputError) (fun _ => weightError)
  accumulationError (beta := beta) (fexp := fexp) (rnd := rnd)
    sumValue (getAtOrZero layer.bias [outChannel.val]) sumError biasError

/-- One coordinate of rounded convolution is enclosed by `convolutionPointError`. -/
theorem approx_convSpec_coordinate
    {d inC outC : Nat} {kernel stride padding inSpatial : TorchLean.Tensor Nat [d]}
    {idealLayer : ConvSpec d inC outC kernel stride padding ℝ}
    {roundedLayer : ConvSpec d inC outC kernel stride padding R}
    {idealInput : Tensor ℝ (Shape.ofList (inC :: (Tensor.to inSpatial (List Nat))))}
    {roundedInput : Tensor R (Shape.ofList (inC :: (Tensor.to inSpatial (List Nat))))}
    {weightError biasError inputError : ℝ}
    (hWeight : approxTensor
      (toSpec := toSpec (β := beta) (fexp := fexp) (rnd := rnd))
      idealLayer.kernel roundedLayer.kernel weightError)
    (hBias : approxTensor
      (toSpec := toSpec (β := beta) (fexp := fexp) (rnd := rnd))
      idealLayer.bias roundedLayer.bias biasError)
    (hInput : approxTensor
      (toSpec := toSpec (β := beta) (fexp := fexp) (rnd := rnd))
      idealInput roundedInput inputError)
    (outChannel : Fin outC)
    (outIndex :
      MultiIndex (Tensor.to (convOutSpatial inSpatial kernel stride padding) (List Nat))) :
    abs
        (toSpec (β := beta) (fexp := fexp) (rnd := rnd)
            (MultiIndex.get (convSpec roundedLayer roundedInput) (outChannel, outIndex)) -
          MultiIndex.get (convSpec idealLayer idealInput) (outChannel, outIndex)) ≤
      convolutionPointError (beta := beta) (fexp := fexp) (rnd := rnd)
        roundedLayer roundedInput weightError biasError inputError outChannel outIndex := by
  let indices := enumerateChannelIndices inC (Tensor.to kernel (List Nat))
  let inputIdeal := convolutionInputValue idealInput outIndex
  let inputRounded := convolutionInputValue roundedInput outIndex
  let weightIdeal := convolutionWeightValue idealLayer.kernel outChannel
  let weightRounded := convolutionWeightValue roundedLayer.kernel outChannel
  have hInputValue : ∀ index ∈ indices,
      abs (toSpec (β := beta) (fexp := fexp) (rnd := rnd) (inputRounded index) -
        inputIdeal index) ≤ inputError := by
    intro index _
    simp only [inputRounded, inputIdeal, convolutionInputValue]
    split
    · simpa [toSpec_zero (β := beta) (fexp := fexp) (rnd := rnd)] using
        approxTensor_eps_nonneg hInput
    · exact approx_getAtOrZero (beta := beta) (fexp := fexp) (rnd := rnd) hInput _
  have hWeightValue : ∀ index ∈ indices,
      abs (toSpec (β := beta) (fexp := fexp) (rnd := rnd) (weightRounded index) -
        weightIdeal index) ≤ weightError := by
    intro index _
    exact approx_getAtOrZero (beta := beta) (fexp := fexp) (rnd := rnd) hWeight _
  have hSum := approx_product_fold (beta := beta) (fexp := fexp) (rnd := rnd)
    indices inputIdeal weightIdeal inputRounded weightRounded
    (fun _ => inputError) (fun _ => weightError) hInputValue hWeightValue
  have hBiasValue := approx_getAtOrZero
    (beta := beta) (fexp := fexp) (rnd := rnd) hBias [outChannel.val]
  have hFinal := approx_add_nf (β := beta) (fexp := fexp) (rnd := rnd)
    (x := indices.foldl (fun acc index => acc + inputIdeal index * weightIdeal index) 0)
    (y := getAtOrZero idealLayer.bias [outChannel.val])
    (xR := indices.foldl
      (fun acc index => acc + inputRounded index * weightRounded index) 0)
    (yR := getAtOrZero roundedLayer.bias [outChannel.val])
    (epsx := productFoldError (beta := beta) (fexp := fexp) (rnd := rnd)
      indices inputRounded weightRounded (fun _ => inputError) (fun _ => weightError))
    (epsy := biasError) hSum hBiasValue
  have hRounded :
      MultiIndex.get (convSpec roundedLayer roundedInput) (outChannel, outIndex) =
        indices.foldl
            (fun acc index => acc + inputRounded index * weightRounded index) 0 +
          getAtOrZero roundedLayer.bias [outChannel.val] := by
    rw [convSpec, MultiIndex.get_addSpec,
      get_convCoreSpec, get_convBiasBroadcastSpec,
      foldChannelsIndices_eq_foldl]
    rfl
  have hIdeal :
      MultiIndex.get (convSpec idealLayer idealInput) (outChannel, outIndex) =
        indices.foldl
            (fun acc index => acc + inputIdeal index * weightIdeal index) 0 +
          getAtOrZero idealLayer.bias [outChannel.val] := by
    rw [convSpec, MultiIndex.get_addSpec,
      get_convCoreSpec, get_convBiasBroadcastSpec,
      foldChannelsIndices_eq_foldl]
    rfl
  rw [hRounded, hIdeal]
  simpa [convolutionPointError, indices, inputRounded, weightRounded,
    accumulationError] using hFinal

/-! ## Backward convolution -/

private theorem get_convKernelDerivSpec
    {alpha : Type} [Context alpha]
    {d inC outC : Nat} {kernel stride padding inSpatial : TorchLean.Tensor Nat [d]}
    (layer : ConvSpec d inC outC kernel stride padding alpha)
    (input : Tensor alpha (Shape.ofList (inC :: (Tensor.to inSpatial (List Nat)))))
    (gradOutput : Tensor alpha
      (Shape.ofList
        (outC :: (Tensor.to (convOutSpatial inSpatial kernel stride padding) (List Nat)))))
    (outChannel : Fin outC) (inChannel : Fin inC)
    (kernelIndex : MultiIndex (Tensor.to kernel (List Nat))) :
    MultiIndex.get (convKernelDerivSpec layer input gradOutput)
        (outChannel, (inChannel, kernelIndex)) =
      foldlIndices (Tensor.to (convOutSpatial inSpatial kernel stride padding) (List Nat)) 0
        (fun acc outIndex =>
          acc +
            (match mkInputIdx? outIndex kernelIndex.toList (Tensor.to stride (List Nat))
                (Tensor.to padding (List Nat)) with
              | none => 0
              | some inputIndex => getAtOrZero input (inChannel.val :: inputIndex)) *
            getAtOrZero gradOutput (outChannel.val :: outIndex)) := by
  simp only [convKernelDerivSpec, MultiIndex.get, Tensor.unstack_dim]
  rw [MultiIndex.get_generate]
  rfl

/--
The input value entering one term of the kernel-gradient accumulation, zero outside the padding.
-/
def kernelGradientInputValue
    {alpha : Type} [Context alpha]
    {d inC : Nat} {kernel inSpatial : TorchLean.Tensor Nat [d]}
    (stride padding : TorchLean.Tensor Nat [d])
    (input : Tensor alpha (Shape.ofList (inC :: (Tensor.to inSpatial (List Nat)))))
    (inChannel : Fin inC) (kernelIndex : MultiIndex (Tensor.to kernel (List Nat)))
    (outIndex : List Nat) : alpha :=
  match mkInputIdx? outIndex kernelIndex.toList (Tensor.to stride (List Nat))
      (Tensor.to padding (List Nat)) with
  | none => 0
  | some inputIndex => getAtOrZero input (inChannel.val :: inputIndex)

/-- The output-gradient value entering that same term. -/
def kernelGradientOutputValue
    {alpha : Type} [Context alpha]
    {d outC : Nat} {kernel stride padding inSpatial : TorchLean.Tensor Nat [d]}
    (gradOutput : Tensor alpha
      (Shape.ofList
        (outC :: (Tensor.to (convOutSpatial inSpatial kernel stride padding) (List Nat)))))
    (outChannel : Fin outC) (outIndex : List Nat) : alpha :=
  getAtOrZero gradOutput (outChannel.val :: outIndex)

/-- Error budget for one coordinate of the rounded convolution kernel gradient. -/
def convolutionKernelGradientPointError
    {d inC outC : Nat} {kernel stride padding inSpatial : TorchLean.Tensor Nat [d]}
    (input : Tensor R (Shape.ofList (inC :: (Tensor.to inSpatial (List Nat)))))
    (gradOutput : Tensor R
      (Shape.ofList
        (outC :: (Tensor.to (convOutSpatial inSpatial kernel stride padding) (List Nat)))))
    (inputError gradOutputError : ℝ)
    (outChannel : Fin outC) (inChannel : Fin inC)
    (kernelIndex : MultiIndex (Tensor.to kernel (List Nat))) : ℝ :=
  let indices :=
    enumerateIndices (Tensor.to (convOutSpatial inSpatial kernel stride padding) (List Nat))
  productFoldError (beta := beta) (fexp := fexp) (rnd := rnd) indices
    (kernelGradientInputValue stride padding input inChannel kernelIndex)
    (kernelGradientOutputValue gradOutput outChannel)
    (fun _ => inputError) (fun _ => gradOutputError)

/-- Every coordinate of the rounded kernel gradient encloses its ideal real value. -/
theorem approx_convKernelDerivSpec_coordinate
    {d inC outC : Nat} {kernel stride padding inSpatial : TorchLean.Tensor Nat [d]}
    {idealLayer : ConvSpec d inC outC kernel stride padding ℝ}
    {roundedLayer : ConvSpec d inC outC kernel stride padding R}
    {idealInput : Tensor ℝ (Shape.ofList (inC :: (Tensor.to inSpatial (List Nat))))}
    {roundedInput : Tensor R (Shape.ofList (inC :: (Tensor.to inSpatial (List Nat))))}
    {idealGradOutput : Tensor ℝ
      (Shape.ofList
        (outC :: (Tensor.to (convOutSpatial inSpatial kernel stride padding) (List Nat))))}
    {roundedGradOutput : Tensor R
      (Shape.ofList
        (outC :: (Tensor.to (convOutSpatial inSpatial kernel stride padding) (List Nat))))}
    {inputError gradOutputError : ℝ}
    (hInput : approxTensor
      (toSpec := toSpec (β := beta) (fexp := fexp) (rnd := rnd))
      idealInput roundedInput inputError)
    (hGradOutput : approxTensor
      (toSpec := toSpec (β := beta) (fexp := fexp) (rnd := rnd))
      idealGradOutput roundedGradOutput gradOutputError)
    (outChannel : Fin outC) (inChannel : Fin inC)
    (kernelIndex : MultiIndex (Tensor.to kernel (List Nat))) :
    abs
        (toSpec (β := beta) (fexp := fexp) (rnd := rnd)
            (MultiIndex.get
              (convKernelDerivSpec roundedLayer roundedInput roundedGradOutput)
              (outChannel, (inChannel, kernelIndex))) -
          MultiIndex.get
            (convKernelDerivSpec idealLayer idealInput idealGradOutput)
            (outChannel, (inChannel, kernelIndex))) ≤
      convolutionKernelGradientPointError
        (beta := beta) (fexp := fexp) (rnd := rnd)
        roundedInput roundedGradOutput inputError gradOutputError
        outChannel inChannel kernelIndex := by
  let indices :=
    enumerateIndices (Tensor.to (convOutSpatial inSpatial kernel stride padding) (List Nat))
  let inputIdeal := kernelGradientInputValue stride padding idealInput inChannel kernelIndex
  let inputRounded := kernelGradientInputValue stride padding roundedInput inChannel kernelIndex
  let gradIdeal := kernelGradientOutputValue idealGradOutput outChannel
  let gradRounded := kernelGradientOutputValue roundedGradOutput outChannel
  have hInputValue : ∀ index ∈ indices,
      abs (toSpec (β := beta) (fexp := fexp) (rnd := rnd) (inputRounded index) -
        inputIdeal index) ≤ inputError := by
    intro index _
    simp only [inputRounded, inputIdeal, kernelGradientInputValue]
    split
    · simpa [toSpec_zero (β := beta) (fexp := fexp) (rnd := rnd)] using
        approxTensor_eps_nonneg hInput
    · exact approx_getAtOrZero (beta := beta) (fexp := fexp) (rnd := rnd) hInput _
  have hGradValue : ∀ index ∈ indices,
      abs (toSpec (β := beta) (fexp := fexp) (rnd := rnd) (gradRounded index) -
        gradIdeal index) ≤ gradOutputError := by
    intro index _
    exact approx_getAtOrZero (beta := beta) (fexp := fexp) (rnd := rnd) hGradOutput _
  have hBound := approx_product_fold (beta := beta) (fexp := fexp) (rnd := rnd)
    indices inputIdeal gradIdeal inputRounded gradRounded
    (fun _ => inputError) (fun _ => gradOutputError) hInputValue hGradValue
  have hRounded :
      MultiIndex.get
          (convKernelDerivSpec roundedLayer roundedInput roundedGradOutput)
          (outChannel, (inChannel, kernelIndex)) =
        indices.foldl
          (fun acc index => acc + inputRounded index * gradRounded index) 0 := by
    rw [get_convKernelDerivSpec, foldlIndices_eq_enumerateIndices]
    rfl
  have hIdeal :
      MultiIndex.get
          (convKernelDerivSpec idealLayer idealInput idealGradOutput)
          (outChannel, (inChannel, kernelIndex)) =
        indices.foldl
          (fun acc index => acc + inputIdeal index * gradIdeal index) 0 := by
    rw [get_convKernelDerivSpec, foldlIndices_eq_enumerateIndices]
    rfl
  rw [hRounded, hIdeal]
  simpa only [convolutionKernelGradientPointError, indices] using hBound

private theorem get_convBiasDerivSpec
    {alpha : Type} [Context alpha]
    {d inC outC : Nat} {kernel stride padding inSpatial : TorchLean.Tensor Nat [d]}
    (layer : ConvSpec d inC outC kernel stride padding alpha)
    (input : Tensor alpha (Shape.ofList (inC :: (Tensor.to inSpatial (List Nat)))))
    (gradOutput : Tensor alpha
      (Shape.ofList
        (outC :: (Tensor.to (convOutSpatial inSpatial kernel stride padding) (List Nat)))))
    (outChannel : Fin outC) :
    MultiIndex.get (dims := [outC]) (convBiasDerivSpec layer input gradOutput)
        (outChannel, PUnit.unit) =
      foldlIndices (Tensor.to (convOutSpatial inSpatial kernel stride padding) (List Nat)) 0
        (fun acc outIndex =>
          acc + getAtOrZero gradOutput (outChannel.val :: outIndex)) := by
  simp [convBiasDerivSpec, MultiIndex.get]

/-- Error budget for one coordinate of the rounded convolution bias gradient. -/
def convolutionBiasGradientPointError
    {d outC : Nat} {kernel stride padding inSpatial : TorchLean.Tensor Nat [d]}
    (gradOutput : Tensor R
      (Shape.ofList
        (outC :: (Tensor.to (convOutSpatial inSpatial kernel stride padding) (List Nat)))))
    (gradOutputError : ℝ) (outChannel : Fin outC) : ℝ :=
  let indices :=
    enumerateIndices (Tensor.to (convOutSpatial inSpatial kernel stride padding) (List Nat))
  foldError (beta := beta) (fexp := fexp) (rnd := rnd) indices
    (fun index => getAtOrZero gradOutput (outChannel.val :: index))
    (fun _ => gradOutputError)

/-- Every coordinate of the rounded bias gradient encloses its ideal real value. -/
theorem approx_convBiasDerivSpec_coordinate
    {d inC outC : Nat} {kernel stride padding inSpatial : TorchLean.Tensor Nat [d]}
    {idealLayer : ConvSpec d inC outC kernel stride padding ℝ}
    {roundedLayer : ConvSpec d inC outC kernel stride padding R}
    {idealInput : Tensor ℝ (Shape.ofList (inC :: (Tensor.to inSpatial (List Nat))))}
    {roundedInput : Tensor R (Shape.ofList (inC :: (Tensor.to inSpatial (List Nat))))}
    {idealGradOutput : Tensor ℝ
      (Shape.ofList
        (outC :: (Tensor.to (convOutSpatial inSpatial kernel stride padding) (List Nat))))}
    {roundedGradOutput : Tensor R
      (Shape.ofList
        (outC :: (Tensor.to (convOutSpatial inSpatial kernel stride padding) (List Nat))))}
    {gradOutputError : ℝ}
    (hGradOutput : approxTensor
      (toSpec := toSpec (β := beta) (fexp := fexp) (rnd := rnd))
      idealGradOutput roundedGradOutput gradOutputError)
    (outChannel : Fin outC) :
    abs
        (toSpec (β := beta) (fexp := fexp) (rnd := rnd)
            (MultiIndex.get (dims := [outC])
              (convBiasDerivSpec roundedLayer roundedInput roundedGradOutput)
              (outChannel, PUnit.unit)) -
          MultiIndex.get (dims := [outC])
            (convBiasDerivSpec idealLayer idealInput idealGradOutput)
            (outChannel, PUnit.unit)) <=
      convolutionBiasGradientPointError
        (beta := beta) (fexp := fexp) (rnd := rnd)
        roundedGradOutput gradOutputError outChannel := by
  let indices :=
    enumerateIndices (Tensor.to (convOutSpatial inSpatial kernel stride padding) (List Nat))
  let idealValue := fun index => getAtOrZero idealGradOutput (outChannel.val :: index)
  let roundedValue := fun index => getAtOrZero roundedGradOutput (outChannel.val :: index)
  have hValue : forall index, index ∈ indices ->
      abs (toSpec (β := beta) (fexp := fexp) (rnd := rnd) (roundedValue index) -
        idealValue index) <= gradOutputError := by
    intro index _
    exact approx_getAtOrZero (beta := beta) (fexp := fexp) (rnd := rnd) hGradOutput _
  have hBound := approx_fold (beta := beta) (fexp := fexp) (rnd := rnd)
    indices idealValue roundedValue (fun _ => gradOutputError) hValue
  have hRounded :
      MultiIndex.get (dims := [outC])
          (convBiasDerivSpec roundedLayer roundedInput roundedGradOutput)
          (outChannel, PUnit.unit) =
        indices.foldl (fun acc index => acc + roundedValue index) 0 := by
    rw [get_convBiasDerivSpec, foldlIndices_eq_enumerateIndices]
  have hIdeal :
      MultiIndex.get (dims := [outC])
          (convBiasDerivSpec idealLayer idealInput idealGradOutput)
          (outChannel, PUnit.unit) =
        indices.foldl (fun acc index => acc + idealValue index) 0 := by
    rw [get_convBiasDerivSpec, foldlIndices_eq_enumerateIndices]
  rw [hRounded, hIdeal]
  simpa only [convolutionBiasGradientPointError, indices] using hBound

private theorem get_convInputDerivSpec
    {alpha : Type} [Context alpha]
    {d inC outC : Nat} {kernel stride padding inSpatial : TorchLean.Tensor Nat [d]}
    (layer : ConvSpec d inC outC kernel stride padding alpha)
    (input : Tensor alpha (Shape.ofList (inC :: (Tensor.to inSpatial (List Nat)))))
    (gradOutput : Tensor alpha
      (Shape.ofList
        (outC :: (Tensor.to (convOutSpatial inSpatial kernel stride padding) (List Nat)))))
    (inChannel : Fin inC) (inputIndex : MultiIndex (Tensor.to inSpatial (List Nat))) :
    MultiIndex.get (convInputDerivSpec layer input gradOutput)
        (inChannel, inputIndex) =
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
  simp only [convInputDerivSpec, MultiIndex.get_dim]
  rw [MultiIndex.get_generate]
  rfl

/-- One summand of the input-gradient accumulation: an output gradient times the kernel weight that
connected them, or zero when the transposed index falls outside the input. -/
def inputGradientTermValue
    {alpha : Type} [Context alpha]
    {d inC outC : Nat} {kernel stride padding inSpatial : TorchLean.Tensor Nat [d]}
    (layer : ConvSpec d inC outC kernel stride padding alpha)
    (gradOutput : Tensor alpha
      (Shape.ofList
        (outC :: (Tensor.to (convOutSpatial inSpatial kernel stride padding) (List Nat)))))
    (inChannel : Fin inC) (inputIndex : MultiIndex (Tensor.to inSpatial (List Nat)))
    (index : Fin outC × List Nat) : alpha :=
  match mkTransposeInputIdx? inputIndex.toList index.2 (Tensor.to stride (List Nat))
      (Tensor.to padding (List Nat)) with
  | none => 0
  | some outIndex =>
      getAtOrZero gradOutput (index.1.val :: outIndex) *
        getAtOrZero layer.kernel (index.1.val :: inChannel.val :: index.2)

/-- Error budget for one coordinate of the rounded convolution input gradient. -/
def convolutionInputGradientPointError
    {d inC outC : Nat} {kernel stride padding inSpatial : TorchLean.Tensor Nat [d]}
    (layer : ConvSpec d inC outC kernel stride padding R)
    (gradOutput : Tensor R
      (Shape.ofList
        (outC :: (Tensor.to (convOutSpatial inSpatial kernel stride padding) (List Nat)))))
    (weightError gradOutputError : ℝ)
    (inChannel : Fin inC) (inputIndex : MultiIndex (Tensor.to inSpatial (List Nat))) : ℝ :=
  let indices := enumerateChannelIndices outC (Tensor.to kernel (List Nat))
  foldError (beta := beta) (fexp := fexp) (rnd := rnd) indices
    (inputGradientTermValue layer gradOutput inChannel inputIndex)
    (fun index =>
      match mkTransposeInputIdx? inputIndex.toList index.2
          (Tensor.to stride (List Nat)) (Tensor.to padding (List Nat)) with
      | none => 0
      | some outIndex =>
          productError (beta := beta) (fexp := fexp) (rnd := rnd)
            (getAtOrZero gradOutput (index.1.val :: outIndex))
            (getAtOrZero layer.kernel (index.1.val :: inChannel.val :: index.2))
            gradOutputError weightError)

/-- Every coordinate of the rounded input gradient encloses its ideal real value. -/
theorem approx_convInputDerivSpec_coordinate
    {d inC outC : Nat} {kernel stride padding inSpatial : TorchLean.Tensor Nat [d]}
    {idealLayer : ConvSpec d inC outC kernel stride padding ℝ}
    {roundedLayer : ConvSpec d inC outC kernel stride padding R}
    {idealInput : Tensor ℝ (Shape.ofList (inC :: (Tensor.to inSpatial (List Nat))))}
    {roundedInput : Tensor R (Shape.ofList (inC :: (Tensor.to inSpatial (List Nat))))}
    {idealGradOutput : Tensor ℝ
      (Shape.ofList
        (outC :: (Tensor.to (convOutSpatial inSpatial kernel stride padding) (List Nat))))}
    {roundedGradOutput : Tensor R
      (Shape.ofList
        (outC :: (Tensor.to (convOutSpatial inSpatial kernel stride padding) (List Nat))))}
    {weightError gradOutputError : ℝ}
    (hWeight : approxTensor
      (toSpec := toSpec (β := beta) (fexp := fexp) (rnd := rnd))
      idealLayer.kernel roundedLayer.kernel weightError)
    (hGradOutput : approxTensor
      (toSpec := toSpec (β := beta) (fexp := fexp) (rnd := rnd))
      idealGradOutput roundedGradOutput gradOutputError)
    (inChannel : Fin inC) (inputIndex : MultiIndex (Tensor.to inSpatial (List Nat))) :
    abs
        (toSpec (β := beta) (fexp := fexp) (rnd := rnd)
            (MultiIndex.get
              (convInputDerivSpec roundedLayer roundedInput roundedGradOutput)
              (inChannel, inputIndex)) -
          MultiIndex.get
            (convInputDerivSpec idealLayer idealInput idealGradOutput)
            (inChannel, inputIndex)) ≤
      convolutionInputGradientPointError
        (beta := beta) (fexp := fexp) (rnd := rnd)
        roundedLayer roundedGradOutput weightError gradOutputError
        inChannel inputIndex := by
  let indices := enumerateChannelIndices outC (Tensor.to kernel (List Nat))
  let idealTerm := inputGradientTermValue idealLayer idealGradOutput inChannel inputIndex
  let roundedTerm := inputGradientTermValue roundedLayer roundedGradOutput inChannel inputIndex
  let termError := fun (index : Fin outC × List Nat) =>
    match mkTransposeInputIdx? inputIndex.toList index.2 (Tensor.to stride (List Nat))
        (Tensor.to padding (List Nat)) with
    | none => 0
    | some outIndex =>
        productError (beta := beta) (fexp := fexp) (rnd := rnd)
          (getAtOrZero roundedGradOutput (index.1.val :: outIndex))
          (getAtOrZero roundedLayer.kernel
            (index.1.val :: inChannel.val :: index.2))
          gradOutputError weightError
  have hTerm : ∀ index ∈ indices,
      abs (toSpec (β := beta) (fexp := fexp) (rnd := rnd) (roundedTerm index) -
        idealTerm index) ≤ termError index := by
    intro index _
    cases hIndex : mkTransposeInputIdx? inputIndex.toList index.2
        (Tensor.to stride (List Nat)) (Tensor.to padding (List Nat)) with
    | none =>
        have hIndexData :
            mkTransposeInputIdx? inputIndex.toList index.2 stride.data.toList
                padding.data.toList = none := by
          simpa only [Tensor.to_list_eq_data] using hIndex
        simp [roundedTerm, idealTerm, termError, inputGradientTermValue, hIndexData,
          toSpec_zero (β := beta) (fexp := fexp) (rnd := rnd)]
    | some outIndex =>
        have hIndexData :
            mkTransposeInputIdx? inputIndex.toList index.2 stride.data.toList
                padding.data.toList = some outIndex := by
          simpa only [Tensor.to_list_eq_data] using hIndex
        have hGrad := approx_getAtOrZero
          (beta := beta) (fexp := fexp) (rnd := rnd) hGradOutput
          (index.1.val :: outIndex)
        have hKernel := approx_getAtOrZero
          (beta := beta) (fexp := fexp) (rnd := rnd) hWeight
          (index.1.val :: inChannel.val :: index.2)
        simpa [roundedTerm, idealTerm, termError, inputGradientTermValue,
          hIndex, hIndexData, productError] using
          (approx_mul_nf (β := beta) (fexp := fexp) (rnd := rnd) hGrad hKernel)
  have hBound := approx_fold (beta := beta) (fexp := fexp) (rnd := rnd)
    indices idealTerm roundedTerm termError hTerm
  have hRounded :
      MultiIndex.get
          (convInputDerivSpec roundedLayer roundedInput roundedGradOutput)
          (inChannel, inputIndex) =
        indices.foldl
          (fun acc index => acc + roundedTerm index) 0 := by
    rw [get_convInputDerivSpec, foldChannelsIndices_eq_foldl]
    rfl
  have hIdeal :
      MultiIndex.get
          (convInputDerivSpec idealLayer idealInput idealGradOutput)
          (inChannel, inputIndex) =
        indices.foldl
          (fun acc index => acc + idealTerm index) 0 := by
    rw [get_convInputDerivSpec, foldChannelsIndices_eq_foldl]
    rfl
  rw [hRounded, hIdeal]
  change
    abs
        (toSpec (β := beta) (fexp := fexp) (rnd := rnd)
            (indices.foldl (fun acc index => acc + roundedTerm index) 0) -
          indices.foldl (fun acc index => acc + idealTerm index) 0) ≤
      foldError (beta := beta) (fexp := fexp) (rnd := rnd)
        indices roundedTerm termError
  exact hBound

end
end Proofs.RuntimeApprox.NFBackend
