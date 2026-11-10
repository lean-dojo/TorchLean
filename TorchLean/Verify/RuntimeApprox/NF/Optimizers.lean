/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.RuntimeApprox.NF.ShapeOps
public import TorchLean.Verify.RuntimeApprox.OptimizerContract

/-!
# Rounded Optimizer Steps for `NF`

Instances of `OptimizerContract` for the `NF` rounded-real model.
Exact and rounded updates both use the public optimizer equations, with shared elementwise error
transformers. The real-valued error traces are noncomputable, and native buffer management and
LibTorch execution lie outside these contracts.

SGD and momentum SGD come first, and they compose with the generic
`OptimizerContract.runApprox_of_stepStreamApprox` theorem over arbitrary finite gradient streams and
arbitrary tensor ranks. AdamW builds on the positive-division and square-root rules and lives in the
same module, so all optimizer numerical contracts share one public home.

The Adam recurrence follows Kingma and Ba, *Adam: A Method for Stochastic Optimization*, ICLR 2015
(https://arxiv.org/abs/1412.6980). The decoupled decay term follows Loshchilov and Hutter,
*Decoupled Weight Decay Regularization*, ICLR 2019 (https://arxiv.org/abs/1711.05101).
-/

@[expose] public section

namespace TorchLean.NF

open TorchLean.Tensor
open FloatLib FloatLib.Numerics FloatLib.Floats.Formats
open Flocq
open OptimizerContract
open Robustness (linfNorm)

noncomputable section

variable {β : Radix} {fexp : ℤ → ℤ} [ValidExp fexp]
variable {rnd : ℝ → ℤ} [ValidRndToNearest rnd]

local notation "R" => NF β fexp rnd

-- NF has no upper exponent bound, so the guards `x - x == 0` hold and the usual schedule applies.
omit [ValidRndToNearest rnd] in
private theorem sub_self_beq (x : R) : (x - x == 0) = true := by
  have h : x - x = 0 := by
    change NF.ofReal (x.val - x.val) = NF.ofReal 0
    rw [sub_self]
  rw [h]
  exact beq_self_eq_true _

private theorem weightedSquare_eq {α : Type} [Storage α] [Context α]
    (finite : ∀ x : α, (x - x == 0) = true) {s : Shape}
    (values : Tensor α s) (weight : α) :
    Optimizer.weightedSquare values weight = Tensor.scale (Tensor.square values) weight := by
  apply Tensor.Rep.ext
  intro coordinate
  simp [Optimizer.weightedSquare, Tensor.scale, Tensor.square, finite]

private theorem correctMoment_eq {α : Type} [Storage α] [Context α]
    (finite : ∀ x : α, (x - x == 0) = true) {s : Shape}
    (moment : Tensor α s) (denominator : α) :
    Optimizer.correctMoment moment denominator = Tensor.scale moment (1 / denominator) := by
  simp [Optimizer.correctMoment, finite]

private theorem correctedRMS_eq {α : Type} [Storage α] [Context α]
    (finite : ∀ x : α, (x - x == 0) = true) {s : Shape}
    (moment : Tensor α s) (denominator : α) :
    Optimizer.correctedRMS moment denominator =
      Tensor.sqrt (Tensor.scale moment (1 / denominator)) .clampNegative := by
  apply Tensor.Rep.ext
  intro coordinate
  simp [Optimizer.correctedRMS, correctMoment_eq finite, Tensor.sqrt,
    finite]

private theorem normalizedUpdate_eq {α : Type} [Storage α] [Context α]
    (unsupported : (inferInstance : Numeric.QuotientArithmetic α).supported = false)
    {s : Shape} (lr epsilon : α) (moment root : Tensor α s) :
    Optimizer.normalizedUpdate lr epsilon moment root =
      Tensor.scale (moment / (root + Tensor.full s epsilon)) lr := by
  apply Tensor.Rep.ext
  intro coordinate
  simp [Optimizer.normalizedUpdate, unsupported, Tensor.scale, div_def, add_def]

/-! ## SGD -/

/-- Exact/runtime relation for SGD state: the runtime learning rate is within `error` of the exact
one. -/
def SGD.StateApprox {s : Shape} (stateS : SGD.State ℝ s)
    (stateR : SGD.State R s) (error : ℝ) : Prop :=
  abs (NF.toReal stateR.lr -
    stateS.lr) ≤ error

/-- Parameter error after one SGD update, expressed using the rounded-real tensors. -/
def SGD.stepError {s : Shape} (lrError parameterError gradientError : ℝ)
    (runtimeState : SGD.State R s)
    (runtimeParameters runtimeGradients : Tensor R s) :
    StepError (fun _ => ℝ) s :=
  let scaledGradientError := linfNorm
    (scaleApproxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
      gradientError lrError runtimeState.lr runtimeGradients)
  let scaledRuntimeGradient := Tensor.scale runtimeGradients runtimeState.lr
  { optimizerStateError := lrError
    parameterError := linfNorm
      (subBoundTensor (β := β) (fexp := fexp)
        parameterError scaledGradientError runtimeParameters scaledRuntimeGradient) }

/-- Numerical refinement contract for TorchLean's plain SGD update. -/
def SGD.contract : OptimizerContract R (NF.toReal) where
  name := "SGD"
  ExactState := SGD.State ℝ
  RuntimeState := SGD.State R
  StateError := fun _ => ℝ
  StepAssumptions := fun _ => Unit
  stateApprox := StateApprox (β := β) (fexp := fexp) (rnd := rnd)
  assumptionsHold := fun _ _ _ _ _ _ _ _ _ _ => True
  updateExact := SGD.update
  updateRuntime := SGD.update
  nextError := fun lrError parameterError gradientError state parameters gradients _ =>
    stepError (β := β) (fexp := fexp) (rnd := rnd)
      lrError parameterError gradientError state parameters gradients
  stateErrorReport := fun lrError => #[("learning rate", lrError)]
  assumptionReport := fun _ => #[]
  update_sound := by
    intro s exactState runtimeState lrError
      exactParameters runtimeParameters parameterError
      exactGradients runtimeGradients gradientError
      _assumptions hstate hparams hgrads _assumptionsHold
    have hscaled := ApproxTensor.scale_of_approx
      (β := β) (fexp := fexp) (rnd := rnd)
      exactState.lr runtimeState.lr hgrads hstate
    have hnext := ApproxTensor.sub
      (β := β) (fexp := fexp) (rnd := rnd) hparams hscaled
    constructor
    · exact hstate
    · simpa [stepError, SGD.update] using hnext

/-- One TorchLean SGD parameter update refines its exact-real counterpart.

The statement is the parameter component of `contract.update_sound`. SGD carries the learning
rate over unchanged, so the state component of that contract is `hstate` itself. -/
theorem SGD.approxTensor_update {s : Shape}
    {stateS : SGD.State ℝ s} {stateR : SGD.State R s}
    {lrError : ℝ}
    {exactParameters : Tensor ℝ s} {runtimeParameters : Tensor R s} {parameterError : ℝ}
    {exactGradients : Tensor ℝ s} {runtimeGradients : Tensor R s} {gradientError : ℝ}
    (hstate : StateApprox (β := β) (fexp := fexp) (rnd := rnd)
      stateS stateR lrError)
    (hparams : ApproxTensor (α := R)
      (toSpec := NF.toReal)
      exactParameters runtimeParameters parameterError)
    (hgrads : ApproxTensor (α := R)
      (toSpec := NF.toReal)
      exactGradients runtimeGradients gradientError) :
    ApproxTensor (α := R) (toSpec := NF.toReal)
      (SGD.update stateS exactParameters exactGradients).parameters
      (SGD.update stateR runtimeParameters runtimeGradients).parameters
      (stepError (β := β) (fexp := fexp) (rnd := rnd)
        lrError parameterError gradientError
        stateR runtimeParameters runtimeGradients).parameterError :=
  ((contract (β := β) (fexp := fexp) (rnd := rnd)).update_sound
    stateS stateR lrError exactParameters runtimeParameters parameterError
    exactGradients runtimeGradients gradientError () hstate hparams hgrads trivial).2

/-! ## Momentum SGD -/

/-- Error budgets for momentum SGD's scalar hyperparameters and momentum buffer. -/
structure MomentumSGD.StateError (s : Shape) where
  /-- Learning-rate error. -/
  lr : ℝ
  /-- Momentum-coefficient error. -/
  momentum : ℝ
  /-- Infinity-norm error in the stored momentum buffer. -/
  momentumBuffer : ℝ

/-- Exact/runtime relation for momentum SGD state. -/
def MomentumSGD.StateApprox {s : Shape} (stateS : MomentumSGD.State ℝ s)
    (stateR : MomentumSGD.State R s) (error : StateError s) : Prop :=
  abs (NF.toReal stateR.lr -
    stateS.lr) ≤ error.lr ∧
  abs (NF.toReal stateR.momentum - stateS.momentum) ≤
    error.momentum ∧
  ApproxTensor (α := R) (toSpec := NF.toReal)
    stateS.momentumBuffer stateR.momentumBuffer error.momentumBuffer

/-- State and parameter bounds for one momentum-SGD update. -/
def MomentumSGD.stepError {s : Shape} (stateError : StateError s)
    (parameterError gradientError : ℝ) (runtimeState : MomentumSGD.State R s)
    (runtimeParameters runtimeGradients : Tensor R s) : StepError StateError s :=
  let scaledRuntimeBuffer := Tensor.scale runtimeState.momentumBuffer runtimeState.momentum
  let scaledBufferError := linfNorm
    (scaleApproxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
      stateError.momentumBuffer stateError.momentum
      runtimeState.momentum runtimeState.momentumBuffer)
  let nextRuntimeBuffer := Optimizer.updateMomentumBuffer
    runtimeState.momentumBuffer runtimeState.momentum runtimeGradients
  let nextBufferError := linfNorm
    (addBoundTensor (β := β) (fexp := fexp)
      scaledBufferError gradientError scaledRuntimeBuffer runtimeGradients)
  let scaledUpdateError := linfNorm
    (scaleApproxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
      nextBufferError stateError.lr
      runtimeState.lr nextRuntimeBuffer)
  { optimizerStateError := { stateError with momentumBuffer := nextBufferError }
    parameterError := linfNorm
      (subBoundTensor (β := β) (fexp := fexp)
        parameterError scaledUpdateError runtimeParameters
        (Tensor.scale nextRuntimeBuffer runtimeState.lr)) }

/-- Numerical refinement contract for momentum SGD with arbitrary-rank parameter tensors. -/
def MomentumSGD.contract :
    OptimizerContract R (NF.toReal) where
  name := "Momentum SGD"
  ExactState := MomentumSGD.State ℝ
  RuntimeState := MomentumSGD.State R
  StateError := StateError
  StepAssumptions := fun _ => Unit
  stateApprox := StateApprox (β := β) (fexp := fexp) (rnd := rnd)
  assumptionsHold := fun _ _ _ _ _ _ _ _ _ _ => True
  updateExact := MomentumSGD.update
  updateRuntime := MomentumSGD.update
  nextError := fun stateError parameterError gradientError state parameters gradients _ =>
    stepError (β := β) (fexp := fexp) (rnd := rnd)
      stateError parameterError gradientError state parameters gradients
  stateErrorReport := fun error =>
    #[("learning rate", error.lr), ("momentum", error.momentum),
      ("momentum buffer", error.momentumBuffer)]
  assumptionReport := fun _ => #[]
  update_sound := by
    intro s stateS stateR stateError paramsS paramsR paramsError gradsS gradsR gradsError
      _assumptions hstate hparams hgrads _assumptionsHold
    rcases hstate with ⟨hlr, hmomentum, hbuf⟩
    have hscaledBuf := ApproxTensor.scale_of_approx
      (β := β) (fexp := fexp) (rnd := rnd)
      stateS.momentum stateR.momentum hbuf hmomentum
    have hnewBuf := ApproxTensor.add
      (β := β) (fexp := fexp) (rnd := rnd) hscaledBuf hgrads
    have hscaledUpdate := ApproxTensor.scale_of_approx
      (β := β) (fexp := fexp) (rnd := rnd)
      stateS.lr stateR.lr hnewBuf hlr
    have hnextParams := ApproxTensor.sub
      (β := β) (fexp := fexp) (rnd := rnd) hparams hscaledUpdate
    constructor
    · exact ⟨hlr, hmomentum, by
        simpa [stepError, MomentumSGD.update,
          Optimizer.updateMomentumBuffer] using hnewBuf⟩
    · simpa [stepError, MomentumSGD.update,
        Optimizer.updateMomentumBuffer] using hnextParams

/-- One public momentum-SGD update refines its exact-real counterpart in both the optimizer state
and the parameters.

The corollary names the one-step statement of `contract.update_sound`, the contract
field that finite runs and graph-level composition use. -/
theorem MomentumSGD.approxTensor_update {s : Shape}
    {stateS : MomentumSGD.State ℝ s}
    {stateR : MomentumSGD.State R s}
    {stateError : StateError s}
    {paramsS : Tensor ℝ s} {paramsR : Tensor R s} {paramsError : ℝ}
    {gradsS : Tensor ℝ s} {gradsR : Tensor R s} {gradsError : ℝ}
    (hstate : StateApprox (β := β) (fexp := fexp) (rnd := rnd)
      stateS stateR stateError)
    (hparams : ApproxTensor (α := R)
      (toSpec := NF.toReal)
      paramsS paramsR paramsError)
    (hgrads : ApproxTensor (α := R)
      (toSpec := NF.toReal)
      gradsS gradsR gradsError) :
    let nextError := stepError (β := β) (fexp := fexp) (rnd := rnd)
      stateError paramsError gradsError stateR paramsR gradsR
    StateApprox (β := β) (fexp := fexp) (rnd := rnd)
        (MomentumSGD.update stateS paramsS gradsS).optimizerState
        (MomentumSGD.update stateR paramsR gradsR).optimizerState
        nextError.optimizerStateError ∧
      ApproxTensor (α := R) (toSpec := NF.toReal)
        (MomentumSGD.update stateS paramsS gradsS).parameters
        (MomentumSGD.update stateR paramsR gradsR).parameters
        nextError.parameterError := by
  exact contract.update_sound
    stateS stateR stateError paramsS paramsR paramsError gradsS gradsR gradsError ()
    hstate hparams hgrads trivial

/-! ## AdamW -/

/-- Error budgets relating exact and rounded AdamW state. -/
structure AdamW.StateError (s : Shape) where
  /-- Error in the stored learning rate. -/
  lr : ℝ
  /-- Error in the first-moment decay coefficient. -/
  beta1 : ℝ
  /-- Error in the second-moment decay coefficient. -/
  beta2 : ℝ
  /-- Error in the denominator stabilizer. -/
  epsilon : ℝ
  /-- Error in the decoupled weight-decay coefficient. -/
  weightDecay : ℝ
  /-- Infinity-norm error in the first-moment tensor. -/
  firstMoment : ℝ
  /-- Infinity-norm error in the second-moment tensor. -/
  secondMoment : ℝ

/-- Exact/runtime relation for the persistent AdamW state. -/
def AdamW.StateApprox {s : Shape} (stateS : AdamW.State ℝ s)
    (stateR : AdamW.State R s) (error : StateError s) : Prop :=
  abs (NF.toReal stateR.lr -
    stateS.lr) ≤ error.lr ∧
  abs (NF.toReal stateR.beta1 - stateS.beta1) ≤ error.beta1 ∧
  abs (NF.toReal stateR.beta2 - stateS.beta2) ≤ error.beta2 ∧
  abs (NF.toReal stateR.epsilon - stateS.epsilon) ≤
    error.epsilon ∧
  abs (NF.toReal stateR.weightDecay -
    stateS.weightDecay) ≤ error.weightDecay ∧
  ApproxTensor (α := R) (toSpec := NF.toReal)
    stateS.firstMoment stateR.firstMoment error.firstMoment ∧
  ApproxTensor (α := R) (toSpec := NF.toReal)
    stateS.secondMoment stateR.secondMoment error.secondMoment ∧
  stateS.stepCount = stateR.stepCount

/-- Errors for scalar expressions derived inside one AdamW step.

They are kept separate from persistent state errors because subtraction, powers, reciprocal, and
the product `lr * weightDecay` each round in the runtime scalar model. -/
structure AdamW.DerivedErrors where
  /-- Error in the rounded scalar expression `1 - beta1`. -/
  oneMinusBeta1 : ℝ
  /-- Error in the rounded scalar expression `1 - beta2`. -/
  oneMinusBeta2 : ℝ
  /-- Error in the reciprocal first-moment bias correction. -/
  firstMomentBiasInverse : ℝ
  /-- Error in the reciprocal second-moment bias correction. -/
  secondMomentBiasInverse : ℝ
  /-- Error in the rounded product `lr * weightDecay`. -/
  decayScale : ℝ

/-- Composed errors for the intermediate tensors in one AdamW step. -/
structure AdamW.ErrorTrace where
  /-- Error after squaring the gradient. -/
  squaredGradient : ℝ
  /-- Error after updating the first moment. -/
  firstMoment : ℝ
  /-- Error after updating the second moment. -/
  secondMoment : ℝ
  /-- Error after first-moment bias correction. -/
  correctedFirstMoment : ℝ
  /-- Error after second-moment bias correction. -/
  correctedSecondMoment : ℝ
  /-- Error after square root of the corrected second moment. -/
  standardDeviation : ℝ
  /-- Error after adding epsilon to the square-root denominator. -/
  denominator : ℝ
  /-- Error in the first moment divided by the epsilon-shifted RMS. -/
  normalizedMoment : ℝ
  /-- Error in the Adam update before subtraction from parameters. -/
  adaptiveUpdate : ℝ
  /-- Error in the decoupled weight-decay update. -/
  decayUpdate : ℝ
  /-- Error after applying decoupled weight decay. -/
  decayedParameters : ℝ
  /-- Final parameter error after the full AdamW step. -/
  parameterError : ℝ

/-- AdamW's one-step error trace from rounded-real tensors and scalar subexpression budgets.
Each tensor contributes one infinity-norm bound, regardless of rank. The caller supplies the
scalar budgets and proves the domain premises.
-/
def AdamW.traceStep {s : Shape} (stateError : StateError s)
    (derivedErrors : DerivedErrors)
    (parameterError gradientError minimumDenominator : ℝ)
    (runtimeState : AdamW.State R s)
    (runtimeParameters runtimeGradients : Tensor R s) :
    ErrorTrace :=
  let nextStepCount := runtimeState.stepCount + 1
  let oneMinusBeta1 := 1 - runtimeState.beta1
  let oneMinusBeta2 := 1 - runtimeState.beta2
  let firstMomentLeft := Tensor.scale runtimeState.firstMoment runtimeState.beta1
  let firstMomentLeftError := linfNorm
    (scaleApproxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
      stateError.firstMoment stateError.beta1
      runtimeState.beta1 runtimeState.firstMoment)
  let firstMomentRight := Tensor.scale runtimeGradients oneMinusBeta1
  let firstMomentRightError := linfNorm
    (scaleApproxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
      gradientError derivedErrors.oneMinusBeta1 oneMinusBeta1 runtimeGradients)
  let firstMoment := firstMomentLeft + firstMomentRight
  let firstMomentError := linfNorm
    (addBoundTensor (β := β) (fexp := fexp)
      firstMomentLeftError firstMomentRightError firstMomentLeft firstMomentRight)
  let squaredGradients := Tensor.square runtimeGradients
  let squaredGradientError := linfNorm
    (mulBoundTensor (β := β) (fexp := fexp)
      gradientError gradientError runtimeGradients runtimeGradients)
  let secondMomentLeft := Tensor.scale runtimeState.secondMoment runtimeState.beta2
  let secondMomentLeftError := linfNorm
    (scaleApproxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
      stateError.secondMoment stateError.beta2
      runtimeState.beta2 runtimeState.secondMoment)
  let secondMomentRight := Tensor.scale squaredGradients oneMinusBeta2
  let secondMomentRightError := linfNorm
    (scaleApproxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
      squaredGradientError derivedErrors.oneMinusBeta2 oneMinusBeta2 squaredGradients)
  let secondMoment := secondMomentLeft + secondMomentRight
  let secondMomentError := linfNorm
    (addBoundTensor (β := β) (fexp := fexp)
      secondMomentLeftError secondMomentRightError secondMomentLeft secondMomentRight)
  let firstMomentBiasInverse :=
    1 / (1 - npowRec nextStepCount runtimeState.beta1)
  let secondMomentBiasInverse :=
    1 / (1 - npowRec nextStepCount runtimeState.beta2)
  let correctedFirstMoment := Tensor.scale firstMoment firstMomentBiasInverse
  let correctedFirstMomentError := linfNorm
    (scaleApproxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
      firstMomentError derivedErrors.firstMomentBiasInverse
      firstMomentBiasInverse firstMoment)
  let correctedSecondMoment := Tensor.scale secondMoment secondMomentBiasInverse
  let correctedSecondMomentError := linfNorm
    (scaleApproxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
      secondMomentError derivedErrors.secondMomentBiasInverse
      secondMomentBiasInverse secondMoment)
  let standardDeviation := Tensor.sqrt correctedSecondMoment .clampNegative
  let standardDeviationError := linfNorm
    (sqrtBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
      correctedSecondMomentError correctedSecondMoment)
  let epsilon := Tensor.full s runtimeState.epsilon
  let denominator := standardDeviation + epsilon
  let denominatorError := linfNorm
    (addBoundTensor (β := β) (fexp := fexp)
      standardDeviationError stateError.epsilon standardDeviation epsilon)
  let normalizedMoment := correctedFirstMoment / denominator
  let normalizedMomentError := linfNorm
    (divPosBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
      minimumDenominator correctedFirstMomentError
      denominatorError correctedFirstMoment denominator)
  let adaptiveUpdate := Tensor.scale normalizedMoment runtimeState.lr
  let adaptiveUpdateError := linfNorm
    (scaleApproxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
      normalizedMomentError stateError.lr runtimeState.lr normalizedMoment)
  let decayScale := runtimeState.lr * runtimeState.weightDecay
  let decayUpdate := Tensor.scale runtimeParameters decayScale
  let decayUpdateError := linfNorm
    (scaleApproxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
      parameterError derivedErrors.decayScale decayScale runtimeParameters)
  let decayedParameters := runtimeParameters - decayUpdate
  let decayedParametersError := linfNorm
    (subBoundTensor (β := β) (fexp := fexp)
      parameterError decayUpdateError runtimeParameters decayUpdate)
  let nextParameterError := linfNorm
    (subBoundTensor (β := β) (fexp := fexp)
      decayedParametersError adaptiveUpdateError decayedParameters adaptiveUpdate)
  { squaredGradient := squaredGradientError
    firstMoment := firstMomentError
    secondMoment := secondMomentError
    correctedFirstMoment := correctedFirstMomentError
    correctedSecondMoment := correctedSecondMomentError
    standardDeviation := standardDeviationError
    denominator := denominatorError
    normalizedMoment := normalizedMomentError
    adaptiveUpdate := adaptiveUpdateError
    decayUpdate := decayUpdateError
    decayedParameters := decayedParametersError
    parameterError := nextParameterError }

/-- One AdamW update is numerically sound with an epsilon-controlled denominator margin.

The derived-scalar hypotheses account for rounding in `1-β`, bias correction, and decoupled decay.
`η` is a pointwise lower bound on the exact denominator `sqrt(max v̂ 0) + epsilon`, and at a
zero-gradient initial step epsilon alone supplies it. The square-root bound admits zero inputs,
and the denominator-error margin accounts for rounding in both the square root and epsilon before
justifying division.
-/
theorem AdamW.approxTensor_update {s : Shape}
    {stateS : AdamW.State ℝ s} {stateR : AdamW.State R s}
    {stateError : StateError s} {derivedErrors : DerivedErrors}
    {paramsS : Tensor ℝ s} {paramsR : Tensor R s} {paramsError : ℝ}
    {gradsS : Tensor ℝ s} {gradsR : Tensor R s} {gradsError η : ℝ}
    (hstate : StateApprox (β := β) (fexp := fexp) (rnd := rnd)
      stateS stateR stateError)
    (hparams : ApproxTensor (α := R)
      (toSpec := NF.toReal) paramsS paramsR paramsError)
    (hgrads : ApproxTensor (α := R)
      (toSpec := NF.toReal) gradsS gradsR gradsError)
    (honeMinus1 : _root_.abs (NF.toReal (1 - stateR.beta1) -
      (1 - stateS.beta1)) ≤ derivedErrors.oneMinusBeta1)
    (honeMinus2 : _root_.abs (NF.toReal (1 - stateR.beta2) -
      (1 - stateS.beta2)) ≤ derivedErrors.oneMinusBeta2)
    (hbias1 : _root_.abs (NF.toReal
      (1 / (1 - npowRec (stateR.stepCount + 1) stateR.beta1)) -
      (1 / (1 - npowRec (stateS.stepCount + 1) stateS.beta1))) ≤
        derivedErrors.firstMomentBiasInverse)
    (hbias2 : _root_.abs (NF.toReal
      (1 / (1 - npowRec (stateR.stepCount + 1) stateR.beta2)) -
      (1 / (1 - npowRec (stateS.stepCount + 1) stateS.beta2))) ≤
        derivedErrors.secondMomentBiasInverse)
    (hdecay : abs (NF.toReal
      (stateR.lr * stateR.weightDecay) -
      stateS.lr * stateS.weightDecay) ≤ derivedErrors.decayScale)
    (hDenominatorLower :
      let nextStepCount := stateS.stepCount + 1
      let moment2 := Tensor.scale stateS.secondMoment stateS.beta2 +
        Tensor.scale (Tensor.square gradsS) (1 - stateS.beta2)
      let corrected :=
        Tensor.scale moment2 (1 / (1 - npowRec nextStepCount stateS.beta2))
      Tensor.Forall (fun z : ℝ => η ≤ z)
        (Tensor.sqrt corrected .clampNegative + Tensor.full s stateS.epsilon))
    (hDenominatorMargin :
      (traceStep (β := β) (fexp := fexp) (rnd := rnd)
        stateError derivedErrors paramsError gradsError η stateR paramsR gradsR
      ).denominator < η) :
    let trace := traceStep (β := β) (fexp := fexp) (rnd := rnd)
      stateError derivedErrors paramsError gradsError η stateR paramsR gradsR
    StateApprox (β := β) (fexp := fexp) (rnd := rnd)
        (AdamW.update stateS paramsS gradsS).optimizerState
        (AdamW.update stateR paramsR gradsR).optimizerState
        { stateError with
          firstMoment := trace.firstMoment
          secondMoment := trace.secondMoment } ∧
      ApproxTensor (α := R) (toSpec := NF.toReal)
        (AdamW.update stateS paramsS gradsS).parameters
        (AdamW.update stateR paramsR gradsR).parameters
        trace.parameterError := by
  dsimp only
  have finiteReal (x : ℝ) : (x - x == 0) = true := by simp
  have finiteNF (x : R) : (x - x == 0) = true := sub_self_beq x
  have nfReplay : (inferInstance : Numeric.QuotientArithmetic R).supported = false := rfl
  rcases hstate with ⟨hlr, hbeta1, hbeta2, hepsilon, hweightDecay, hm, hv, ht⟩
  let mS := Tensor.scale stateS.firstMoment stateS.beta1 +
    Tensor.scale gradsS (1 - stateS.beta1)
  let mR := Tensor.scale stateR.firstMoment stateR.beta1 +
    Tensor.scale gradsR (1 - stateR.beta1)
  let vS := Tensor.scale stateS.secondMoment stateS.beta2 +
    Tensor.scale (Tensor.square gradsS) (1 - stateS.beta2)
  let vR := Tensor.scale stateR.secondMoment stateR.beta2 +
    Tensor.scale (Tensor.square gradsR) (1 - stateR.beta2)
  let bias1S := 1 / (1 - npowRec (stateS.stepCount + 1) stateS.beta1)
  let bias1R := 1 / (1 - npowRec (stateR.stepCount + 1) stateR.beta1)
  let bias2S := 1 / (1 - npowRec (stateS.stepCount + 1) stateS.beta2)
  let bias2R := 1 / (1 - npowRec (stateR.stepCount + 1) stateR.beta2)
  let mHatS := Tensor.scale mS bias1S
  let mHatR := Tensor.scale mR bias1R
  let vHatS := Tensor.scale vS bias2S
  let vHatR := Tensor.scale vR bias2R
  let trace := traceStep (β := β) (fexp := fexp) (rnd := rnd)
    stateError derivedErrors paramsError gradsError η stateR paramsR gradsR
  have hmLeft := ApproxTensor.scale_of_approx
    (β := β) (fexp := fexp) (rnd := rnd) stateS.beta1 stateR.beta1 hm hbeta1
  have hmRight := ApproxTensor.scale_of_approx
    (β := β) (fexp := fexp) (rnd := rnd)
    (1 - stateS.beta1) (1 - stateR.beta1) hgrads honeMinus1
  have hm' := ApproxTensor.add (β := β) (fexp := fexp) (rnd := rnd) hmLeft hmRight
  have hsq := ApproxTensor.square (β := β) (fexp := fexp) (rnd := rnd) hgrads
  have hvLeft := ApproxTensor.scale_of_approx
    (β := β) (fexp := fexp) (rnd := rnd) stateS.beta2 stateR.beta2 hv hbeta2
  have hvRight := ApproxTensor.scale_of_approx
    (β := β) (fexp := fexp) (rnd := rnd)
    (1 - stateS.beta2) (1 - stateR.beta2) hsq honeMinus2
  have hv' := ApproxTensor.add (β := β) (fexp := fexp) (rnd := rnd) hvLeft hvRight
  have hmHat := ApproxTensor.scale_of_approx
    (β := β) (fexp := fexp) (rnd := rnd) bias1S bias1R hm'
      (by simpa [bias1S, bias1R] using hbias1)
  have hvHat := ApproxTensor.scale_of_approx
    (β := β) (fexp := fexp) (rnd := rnd) bias2S bias2R hv'
      (by simpa [bias2S, bias2R] using hbias2)
  have hsqrt := ApproxTensor.sqrt
    (β := β) (fexp := fexp) (rnd := rnd) hvHat
  have hepsilonFill := ApproxTensor.full
    (β := β) (fexp := fexp) (rnd := rnd) hepsilon (s := s)
  have hdenominator := ApproxTensor.add
    (β := β) (fexp := fexp) (rnd := rnd) hsqrt hepsilonFill
  have hdenominatorLower : Tensor.Forall (fun z : ℝ => η ≤ z)
      (Tensor.sqrt vHatS .clampNegative + Tensor.full s stateS.epsilon) := by
    simpa [vHatS, vS, bias2S] using hDenominatorLower
  have hnormalized := ApproxTensor.div_of_pos_lb
    (β := β) (fexp := fexp) (rnd := rnd) η
    hmHat hdenominator hdenominatorLower hDenominatorMargin
  have hadamUpdate := ApproxTensor.scale_of_approx
    (β := β) (fexp := fexp) (rnd := rnd) stateS.lr stateR.lr hnormalized hlr
  have hdecayUpdate := ApproxTensor.scale_of_approx
    (β := β) (fexp := fexp) (rnd := rnd)
    (stateS.lr * stateS.weightDecay)
    (stateR.lr * stateR.weightDecay) hparams hdecay
  have hdecayed := ApproxTensor.sub
    (β := β) (fexp := fexp) (rnd := rnd) hparams hdecayUpdate
  have hnext := ApproxTensor.sub
    (β := β) (fexp := fexp) (rnd := rnd) hdecayed hadamUpdate
  constructor
  · refine ⟨hlr, hbeta1, hbeta2, hepsilon, hweightDecay, ?_, ?_, ?_⟩
    · simpa [AdamW.update, Adam.update, trace, traceStep, mS, mR] using hm'
    · simpa [AdamW.update, Adam.update, trace, traceStep, vS, vR,
        weightedSquare_eq finiteReal, weightedSquare_eq finiteNF] using hv'
    · simpa [AdamW.update, Adam.update] using ht
  · simpa [trace, traceStep, AdamW.update, Adam.update,
      mS, mR, vS, vR, mHatS, mHatR,
      vHatS, vHatR, bias1S, bias1R, bias2S, bias2R,
      weightedSquare_eq finiteReal, weightedSquare_eq finiteNF,
      correctMoment_eq finiteReal, correctMoment_eq finiteNF,
      correctedRMS_eq finiteReal, correctedRMS_eq finiteNF,
      normalizedUpdate_eq Numeric.QuotientArithmetic.supported_real,
      normalizedUpdate_eq nfReplay] using hnext

/-! ## AdamW contract instance -/

/-- Numerical assumptions and positivity margin for one AdamW update. -/
structure AdamW.StepAssumptions where
  /-- Bounds for rounded scalar subexpressions used by bias correction and decay. -/
  derivedErrors : DerivedErrors
  /-- Positive lower bound on `sqrt(max v̂ 0) + epsilon`, supplied by epsilon when `v̂ = 0`. -/
  minimumDenominator : ℝ

/-- Complete validity predicate for one AdamW contract application. -/
def AdamW.AssumptionsHold {s : Shape}
    (stateS : AdamW.State ℝ s) (stateR : AdamW.State R s)
    (stateError : StateError s)
    (_paramsS : Tensor ℝ s) (paramsR : Tensor R s) (paramsError : ℝ)
    (gradsS : Tensor ℝ s) (gradsR : Tensor R s) (gradsError : ℝ)
    (assumptions : StepAssumptions) : Prop :=
  _root_.abs (NF.toReal (1 - stateR.beta1) -
      (1 - stateS.beta1)) ≤ assumptions.derivedErrors.oneMinusBeta1 ∧
  _root_.abs (NF.toReal (1 - stateR.beta2) -
      (1 - stateS.beta2)) ≤ assumptions.derivedErrors.oneMinusBeta2 ∧
  _root_.abs (NF.toReal
      (1 / (1 - npowRec (stateR.stepCount + 1) stateR.beta1)) -
      (1 / (1 - npowRec (stateS.stepCount + 1) stateS.beta1))) ≤
    assumptions.derivedErrors.firstMomentBiasInverse ∧
  _root_.abs (NF.toReal
      (1 / (1 - npowRec (stateR.stepCount + 1) stateR.beta2)) -
      (1 / (1 - npowRec (stateS.stepCount + 1) stateS.beta2))) ≤
    assumptions.derivedErrors.secondMomentBiasInverse ∧
  abs (NF.toReal
      (stateR.lr * stateR.weightDecay) -
      stateS.lr * stateS.weightDecay) ≤
    assumptions.derivedErrors.decayScale ∧
  (let nextStepCount := stateS.stepCount + 1
   let moment2 := Tensor.scale stateS.secondMoment stateS.beta2 +
     Tensor.scale (Tensor.square gradsS) (1 - stateS.beta2)
   let corrected := Tensor.scale moment2 (1 / (1 - npowRec nextStepCount stateS.beta2))
   Tensor.Forall (fun z : ℝ => assumptions.minimumDenominator ≤ z)
     (Tensor.sqrt corrected .clampNegative + Tensor.full s stateS.epsilon)) ∧
  (traceStep (β := β) (fexp := fexp) (rnd := rnd)
      stateError assumptions.derivedErrors paramsError gradsError
      assumptions.minimumDenominator stateR paramsR gradsR).denominator <
    assumptions.minimumDenominator

/-- State and parameter error object produced by one AdamW contract step. -/
def AdamW.stepError {s : Shape} (stateError : StateError s)
    (paramsError gradsError : ℝ) (stateR : AdamW.State R s)
    (paramsR gradsR : Tensor R s) (assumptions : StepAssumptions) :
    StepError StateError s :=
  let trace := traceStep (β := β) (fexp := fexp) (rnd := rnd)
    stateError assumptions.derivedErrors paramsError gradsError
    assumptions.minimumDenominator stateR paramsR gradsR
  { optimizerStateError :=
      { stateError with
        firstMoment := trace.firstMoment
        secondMoment := trace.secondMoment }
    parameterError := trace.parameterError }

/-- AdamW instance of the generic numerical optimizer contract.

Its step assumptions are proof data that leave the update itself unchanged, so
`runApprox_of_stepStreamApprox` composes AdamW over finite runs in the same way as SGD and
momentum SGD.
-/
def AdamW.contract : OptimizerContract R
    (NF.toReal) where
  name := "AdamW"
  ExactState := AdamW.State ℝ
  RuntimeState := AdamW.State R
  StateError := StateError
  StepAssumptions := fun _ => StepAssumptions
  stateApprox := StateApprox (β := β) (fexp := fexp) (rnd := rnd)
  assumptionsHold := AssumptionsHold (β := β) (fexp := fexp) (rnd := rnd)
  updateExact := AdamW.update
  updateRuntime := AdamW.update
  nextError := fun stateError parameterError gradientError state parameters gradients assumptions =>
    stepError (β := β) (fexp := fexp) (rnd := rnd)
      stateError parameterError gradientError state parameters gradients assumptions
  stateErrorReport := fun error =>
    #[("learning rate", error.lr), ("beta1", error.beta1), ("beta2", error.beta2),
      ("epsilon", error.epsilon), ("weight decay", error.weightDecay),
      ("first moment", error.firstMoment), ("second moment", error.secondMoment)]
  assumptionReport := fun assumptions =>
    #[("minimum denominator", assumptions.minimumDenominator),
      ("1 - beta1", assumptions.derivedErrors.oneMinusBeta1),
      ("1 - beta2", assumptions.derivedErrors.oneMinusBeta2),
      ("first bias inverse", assumptions.derivedErrors.firstMomentBiasInverse),
      ("second bias inverse", assumptions.derivedErrors.secondMomentBiasInverse),
      ("decay scale", assumptions.derivedErrors.decayScale)]
  update_sound := by
    intro s stateS stateR stateError paramsS paramsR paramsError gradsS gradsR gradsError
      assumptions hstate hparams hgrads hvalid
    rcases hvalid with
      ⟨honeMinus1, honeMinus2, hbias1, hbias2, hdecay, hDenominatorLower, hDenominatorMargin⟩
    simpa [stepError] using
      (approxTensor_update (β := β) (fexp := fexp) (rnd := rnd)
        (derivedErrors := assumptions.derivedErrors)
        hstate hparams hgrads honeMinus1 honeMinus2 hbias1 hbias2 hdecay hDenominatorLower
        hDenominatorMargin)

end
end TorchLean.NF
