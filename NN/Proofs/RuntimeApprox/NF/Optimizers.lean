/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.RuntimeApprox.NF.ShapeOps
public import NN.Proofs.RuntimeApprox.Optimizer

/-!
# Rounded Optimizer Steps for `NF`

Concrete instances of `RuntimeApprox.Optimizer.NumericalStepContract` for TorchLean's rounded
`NF` runtime. These proofs use the same tensor equations as the public optimizers and the shared
elementwise error transformers; there is no second optimizer implementation in the proof layer.

The first contracts cover SGD and momentum SGD. They already compose with the generic
`NumericalStepContract.run_approx` theorem over arbitrary finite gradient streams and arbitrary
tensor ranks. Adaptive optimizers build on the positive-division and square-root rules and are kept
in this module so all optimizer numerical contracts share one public home.

The Adam recurrence follows Kingma and Ba, *Adam: A Method for Stochastic Optimization*, ICLR 2015
(https://arxiv.org/abs/1412.6980). The decoupled decay term follows Loshchilov and Hutter,
*Decoupled Weight Decay Regularization*, ICLR 2019 (https://arxiv.org/abs/1711.05101).
-/

@[expose] public section

namespace Proofs.RuntimeApprox.NFBackend.Optimizer

open Spec TorchLean
open TorchLean TorchLean.Tensor
open FloatLib FloatLib.Numerics FloatLib.Floats.Formats
open Flocq
open Proofs.RuntimeApprox.Optimizer

noncomputable section

variable {β : Radix} {fexp : ℤ → ℤ} [ValidExp fexp]
variable {rnd : ℝ → ℤ} [ValidRndToNearest rnd]

local notation "R" => NF β fexp rnd

/-! ## SGD -/

/-- Error in the runtime learning-rate scalar stored by SGD. -/
abbrev SGDStateError := ℝ

/-- Exact/runtime relation for SGD state. -/
def sgdStateApprox {s : Shape} (stateS : Optim.SGD.State ℝ s)
    (stateR : Optim.SGD.State R s) (error : SGDStateError) : Prop :=
  abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) stateR.learningRate -
    stateS.learningRate) ≤ error

/-- Parameter error after one SGD update, computed from the actual runtime tensors. -/
def sgdStepError {s : Shape} (learningRateError parameterError gradientError : ℝ)
    (runtimeState : Optim.SGD.State R s)
    (runtimeParameters runtimeGradients : Tensor R s) :
    StepError (fun _ => SGDStateError) s :=
  let scaledGradientError := linfNorm
    (scaleApproxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
      gradientError learningRateError runtimeState.learningRate runtimeGradients)
  let scaledRuntimeGradient := scaleSpec runtimeGradients runtimeState.learningRate
  { optimizerStateError := learningRateError
    parameterError := linfNorm
      (subBoundTensor (β := β) (fexp := fexp)
        parameterError scaledGradientError runtimeParameters scaledRuntimeGradient) }

/-- Numerical refinement contract for TorchLean's plain SGD update. -/
def sgdContract : NumericalStepContract R (toSpec (β := β) (fexp := fexp) (rnd := rnd)) where
  name := "SGD"
  ExactState := Optim.SGD.State ℝ
  RuntimeState := Optim.SGD.State R
  StateError := fun _ => SGDStateError
  StepAssumptions := fun _ => Unit
  stateApprox := sgdStateApprox (β := β) (fexp := fexp) (rnd := rnd)
  assumptionsHold := fun _ _ _ _ _ _ _ _ _ _ => True
  updateExact := Optim.SGD.update
  updateRuntime := Optim.SGD.update
  nextError := fun learningRateError parameterError gradientError state parameters gradients _ =>
    sgdStepError (β := β) (fexp := fexp) (rnd := rnd)
      learningRateError parameterError gradientError state parameters gradients
  stateErrorReport := fun learningRateError => #[("learning rate", learningRateError)]
  assumptionReport := fun _ => #[]
  updateApprox := by
    intro s exactState runtimeState learningRateError
      exactParameters runtimeParameters parameterError
      exactGradients runtimeGradients gradientError
      _assumptions stateApprox parametersApprox gradientsApprox _assumptionsHold
    have scaledGradientApprox := approxTensor_scale_spec_of_approx
      (β := β) (fexp := fexp) (rnd := rnd)
      exactState.learningRate runtimeState.learningRate gradientsApprox stateApprox
    have nextParametersApprox := approxTensor_sub_spec
      (β := β) (fexp := fexp) (rnd := rnd) parametersApprox scaledGradientApprox
    constructor
    · exact stateApprox
    · simpa [sgdStepError, Optim.SGD.update] using nextParametersApprox

/-- One actual TorchLean SGD parameter update refines its exact-real counterpart. -/
theorem approxTensor_sgd_update {s : Shape}
    {stateS : Optim.SGD.State ℝ s} {stateR : Optim.SGD.State R s}
    {learningRateError : ℝ}
    {exactParameters : Tensor ℝ s} {runtimeParameters : Tensor R s} {parameterError : ℝ}
    {exactGradients : Tensor ℝ s} {runtimeGradients : Tensor R s} {gradientError : ℝ}
    (hstate : sgdStateApprox (β := β) (fexp := fexp) (rnd := rnd)
      stateS stateR learningRateError)
    (hparams : approxTensor (α := R)
      (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
      exactParameters runtimeParameters parameterError)
    (hgrads : approxTensor (α := R)
      (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
      exactGradients runtimeGradients gradientError) :
    approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (Optim.SGD.update stateS exactParameters exactGradients).parameters
      (Optim.SGD.update stateR runtimeParameters runtimeGradients).parameters
      (sgdStepError (β := β) (fexp := fexp) (rnd := rnd)
        learningRateError parameterError gradientError
        stateR runtimeParameters runtimeGradients).parameterError := by
  have hscaled := approxTensor_scale_spec_of_approx
    (β := β) (fexp := fexp) (rnd := rnd)
    stateS.learningRate stateR.learningRate hgrads hstate
  have hnext := approxTensor_sub_spec
    (β := β) (fexp := fexp) (rnd := rnd) hparams hscaled
  simpa [sgdStepError, Optim.SGD.update] using hnext

/-! ## Momentum SGD -/

/-- Error budgets for momentum SGD's scalar hyperparameters and momentum buffer. -/
structure MomentumSGDStateError (s : Shape) where
  /-- Learning-rate error. -/
  learningRate : ℝ
  /-- Momentum-coefficient error. -/
  momentum : ℝ
  /-- Infinity-norm error in the stored momentum buffer. -/
  momentumBuffer : ℝ

/-- Exact/runtime relation for momentum SGD state. -/
def momentumSGDStateApprox {s : Shape} (stateS : Optim.MomentumSGD.State ℝ s)
    (stateR : Optim.MomentumSGD.State R s) (error : MomentumSGDStateError s) : Prop :=
  abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) stateR.learningRate -
    stateS.learningRate) ≤ error.learningRate ∧
  abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) stateR.momentum - stateS.momentum) ≤
    error.momentum ∧
  approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
    stateS.momentumBuffer stateR.momentumBuffer error.momentumBuffer

/-- State and parameter bounds for one momentum-SGD update. -/
def momentumSGDStepError {s : Shape} (stateError : MomentumSGDStateError s)
    (parameterError gradientError : ℝ) (runtimeState : Optim.MomentumSGD.State R s)
    (runtimeParameters runtimeGradients : Tensor R s) : StepError MomentumSGDStateError s :=
  let scaledRuntimeBuffer := scaleSpec runtimeState.momentumBuffer runtimeState.momentum
  let scaledBufferError := linfNorm
    (scaleApproxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
      stateError.momentumBuffer stateError.momentum
      runtimeState.momentum runtimeState.momentumBuffer)
  let nextRuntimeBuffer := Optim.updateMomentumBuffer
    runtimeState.momentumBuffer runtimeState.momentum runtimeGradients
  let nextBufferError := linfNorm
    (addBoundTensor (β := β) (fexp := fexp)
      scaledBufferError gradientError scaledRuntimeBuffer runtimeGradients)
  let scaledUpdateError := linfNorm
    (scaleApproxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
      nextBufferError stateError.learningRate
      runtimeState.learningRate nextRuntimeBuffer)
  { optimizerStateError := { stateError with momentumBuffer := nextBufferError }
    parameterError := linfNorm
      (subBoundTensor (β := β) (fexp := fexp)
        parameterError scaledUpdateError runtimeParameters
        (scaleSpec nextRuntimeBuffer runtimeState.learningRate)) }

/-- Numerical refinement contract for momentum SGD with arbitrary-rank parameter tensors. -/
def momentumSGDContract :
    NumericalStepContract R (toSpec (β := β) (fexp := fexp) (rnd := rnd)) where
  name := "Momentum SGD"
  ExactState := Optim.MomentumSGD.State ℝ
  RuntimeState := Optim.MomentumSGD.State R
  StateError := MomentumSGDStateError
  StepAssumptions := fun _ => Unit
  stateApprox := momentumSGDStateApprox (β := β) (fexp := fexp) (rnd := rnd)
  assumptionsHold := fun _ _ _ _ _ _ _ _ _ _ => True
  updateExact := Optim.MomentumSGD.update
  updateRuntime := Optim.MomentumSGD.update
  nextError := fun stateError parameterError gradientError state parameters gradients _ =>
    momentumSGDStepError (β := β) (fexp := fexp) (rnd := rnd)
      stateError parameterError gradientError state parameters gradients
  stateErrorReport := fun error =>
    #[("learning rate", error.learningRate), ("momentum", error.momentum),
      ("momentum buffer", error.momentumBuffer)]
  assumptionReport := fun _ => #[]
  updateApprox := by
    intro s stateS stateR stateError paramsS paramsR paramsError gradsS gradsR gradsError
      _assumptions hstate hparams hgrads _assumptionsHold
    rcases hstate with ⟨hlr, hmomentum, hbuf⟩
    have hscaledBuf := approxTensor_scale_spec_of_approx
      (β := β) (fexp := fexp) (rnd := rnd)
      stateS.momentum stateR.momentum hbuf hmomentum
    have hnewBuf := approxTensor_add_spec
      (β := β) (fexp := fexp) (rnd := rnd) hscaledBuf hgrads
    have hscaledUpdate := approxTensor_scale_spec_of_approx
      (β := β) (fexp := fexp) (rnd := rnd)
      stateS.learningRate stateR.learningRate hnewBuf hlr
    have hnextParams := approxTensor_sub_spec
      (β := β) (fexp := fexp) (rnd := rnd) hparams hscaledUpdate
    constructor
    · exact ⟨hlr, hmomentum, by
        simpa [momentumSGDStepError, Optim.MomentumSGD.update,
          Optim.updateMomentumBuffer] using hnewBuf⟩
    · simpa [momentumSGDStepError, Optim.MomentumSGD.update,
        Optim.updateMomentumBuffer] using hnextParams

/-- One public momentum-SGD update refines its exact-real counterpart.

This named corollary exposes the useful one-step statement without duplicating its proof; the
generic `momentumSGDContract.updateApprox` field remains the source used for finite runs and
graph-level composition. -/
theorem approxTensor_momentumSGD_update {s : Shape}
    {stateS : Optim.MomentumSGD.State ℝ s}
    {stateR : Optim.MomentumSGD.State R s}
    {stateError : MomentumSGDStateError s}
    {paramsS : Tensor ℝ s} {paramsR : Tensor R s} {paramsError : ℝ}
    {gradsS : Tensor ℝ s} {gradsR : Tensor R s} {gradsError : ℝ}
    (hstate : momentumSGDStateApprox (β := β) (fexp := fexp) (rnd := rnd)
      stateS stateR stateError)
    (hparams : approxTensor (α := R)
      (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
      paramsS paramsR paramsError)
    (hgrads : approxTensor (α := R)
      (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
      gradsS gradsR gradsError) :
    let nextError := momentumSGDStepError (β := β) (fexp := fexp) (rnd := rnd)
      stateError paramsError gradsError stateR paramsR gradsR
    momentumSGDStateApprox (β := β) (fexp := fexp) (rnd := rnd)
        (Optim.MomentumSGD.update stateS paramsS gradsS).optimizerState
        (Optim.MomentumSGD.update stateR paramsR gradsR).optimizerState
        nextError.optimizerStateError ∧
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (Optim.MomentumSGD.update stateS paramsS gradsS).parameters
        (Optim.MomentumSGD.update stateR paramsR gradsR).parameters
        nextError.parameterError := by
  exact momentumSGDContract.updateApprox
    stateS stateR stateError paramsS paramsR paramsError gradsS gradsR gradsError ()
    hstate hparams hgrads trivial

/-! ## AdamW -/

/-- Error budgets relating exact and rounded AdamW state. -/
structure AdamWStateError (s : Shape) where
  /-- Error in the stored learning rate. -/
  learningRate : ℝ
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
def adamWStateApprox {s : Shape} (stateS : Optim.AdamW.State ℝ s)
    (stateR : Optim.AdamW.State R s) (error : AdamWStateError s) : Prop :=
  abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) stateR.learningRate -
    stateS.learningRate) ≤ error.learningRate ∧
  abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) stateR.beta1 - stateS.beta1) ≤ error.beta1 ∧
  abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) stateR.beta2 - stateS.beta2) ≤ error.beta2 ∧
  abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) stateR.epsilon - stateS.epsilon) ≤
    error.epsilon ∧
  abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) stateR.weightDecay -
    stateS.weightDecay) ≤ error.weightDecay ∧
  approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
    stateS.firstMoment stateR.firstMoment error.firstMoment ∧
  approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
    stateS.secondMoment stateR.secondMoment error.secondMoment ∧
  stateS.stepCount = stateR.stepCount

/-- Errors for scalar expressions derived inside one AdamW step.

They are kept separate from persistent state errors because subtraction, powers, reciprocal, and
the product `lr * weightDecay` each round in the runtime scalar model. -/
structure AdamWDerivedErrors where
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
structure AdamWStepErrorTrace where
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
  /-- Error in the elementwise adaptive learning rate. -/
  adaptiveLearningRate : ℝ
  /-- Error in the Adam update before subtraction from parameters. -/
  adaptiveUpdate : ℝ
  /-- Error in the decoupled weight-decay update. -/
  decayUpdate : ℝ
  /-- Error after applying decoupled weight decay. -/
  decayedParameters : ℝ
  /-- Final parameter error after the full AdamW step. -/
  parameterError : ℝ

/-- Compute AdamW's complete one-step error trace from runtime values and scalar subexpression
budgets. The reduction to one infinity-norm number per tensor keeps the trace independent of
rank. -/
def adamWStepErrorTrace {s : Shape} (stateError : AdamWStateError s)
    (derivedErrors : AdamWDerivedErrors)
    (parameterError gradientError minimumSecondMoment : ℝ)
    (runtimeState : Optim.AdamW.State R s)
    (runtimeParameters runtimeGradients : Tensor R s) :
    AdamWStepErrorTrace :=
  let nextStepCount := runtimeState.stepCount + 1
  let oneMinusBeta1 := 1 - runtimeState.beta1
  let oneMinusBeta2 := 1 - runtimeState.beta2
  let firstMomentLeft := scaleSpec runtimeState.firstMoment runtimeState.beta1
  let firstMomentLeftError := linfNorm
    (scaleApproxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
      stateError.firstMoment stateError.beta1
      runtimeState.beta1 runtimeState.firstMoment)
  let firstMomentRight := scaleSpec runtimeGradients oneMinusBeta1
  let firstMomentRightError := linfNorm
    (scaleApproxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
      gradientError derivedErrors.oneMinusBeta1 oneMinusBeta1 runtimeGradients)
  let firstMoment := addSpec firstMomentLeft firstMomentRight
  let firstMomentError := linfNorm
    (addBoundTensor (β := β) (fexp := fexp)
      firstMomentLeftError firstMomentRightError firstMomentLeft firstMomentRight)
  let squaredGradients := squareSpec runtimeGradients
  let squaredGradientError := linfNorm
    (mulBoundTensor (β := β) (fexp := fexp)
      gradientError gradientError runtimeGradients runtimeGradients)
  let secondMomentLeft := scaleSpec runtimeState.secondMoment runtimeState.beta2
  let secondMomentLeftError := linfNorm
    (scaleApproxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
      stateError.secondMoment stateError.beta2
      runtimeState.beta2 runtimeState.secondMoment)
  let secondMomentRight := scaleSpec squaredGradients oneMinusBeta2
  let secondMomentRightError := linfNorm
    (scaleApproxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
      squaredGradientError derivedErrors.oneMinusBeta2 oneMinusBeta2 squaredGradients)
  let secondMoment := addSpec secondMomentLeft secondMomentRight
  let secondMomentError := linfNorm
    (addBoundTensor (β := β) (fexp := fexp)
      secondMomentLeftError secondMomentRightError secondMomentLeft secondMomentRight)
  let firstMomentBiasInverse :=
    1 / (1 - Optim.scalarPowNat runtimeState.beta1 nextStepCount)
  let secondMomentBiasInverse :=
    1 / (1 - Optim.scalarPowNat runtimeState.beta2 nextStepCount)
  let correctedFirstMoment := scaleSpec firstMoment firstMomentBiasInverse
  let correctedFirstMomentError := linfNorm
    (scaleApproxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
      firstMomentError derivedErrors.firstMomentBiasInverse
      firstMomentBiasInverse firstMoment)
  let correctedSecondMoment := scaleSpec secondMoment secondMomentBiasInverse
  let correctedSecondMomentError := linfNorm
    (scaleApproxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
      secondMomentError derivedErrors.secondMomentBiasInverse
      secondMomentBiasInverse secondMoment)
  let standardDeviation := sqrtSpec correctedSecondMoment
  let standardDeviationError := linfNorm
    (sqrtPosBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
      minimumSecondMoment correctedSecondMomentError correctedSecondMoment)
  let epsilon := Tensor.full s runtimeState.epsilon
  let denominator := addSpec standardDeviation epsilon
  let denominatorError := linfNorm
    (addBoundTensor (β := β) (fexp := fexp)
      standardDeviationError stateError.epsilon standardDeviation epsilon)
  let learningRate := Tensor.full s runtimeState.learningRate
  let adaptiveLearningRate := divSpec learningRate denominator
  let adaptiveLearningRateError := linfNorm
    (divPosBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
      (Real.sqrt minimumSecondMoment) stateError.learningRate
      denominatorError learningRate denominator)
  let adaptiveUpdate := mulSpec adaptiveLearningRate correctedFirstMoment
  let adaptiveUpdateError := linfNorm
    (mulBoundTensor (β := β) (fexp := fexp)
      adaptiveLearningRateError correctedFirstMomentError
      adaptiveLearningRate correctedFirstMoment)
  let decayScale := runtimeState.learningRate * runtimeState.weightDecay
  let decayUpdate := scaleSpec runtimeParameters decayScale
  let decayUpdateError := linfNorm
    (scaleApproxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
      parameterError derivedErrors.decayScale decayScale runtimeParameters)
  let decayedParameters := subSpec runtimeParameters decayUpdate
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
    adaptiveLearningRate := adaptiveLearningRateError
    adaptiveUpdate := adaptiveUpdateError
    decayUpdate := decayUpdateError
    decayedParameters := decayedParametersError
    parameterError := nextParameterError }

/-- One AdamW update is numerically sound on a certified positive second-moment domain.

The hypotheses for the derived scalar expressions expose rounding in `1-β`, bias correction, and
the decoupled decay coefficient. `η` keeps `sqrt(vHat)` away from its singular derivative at zero;
the two margin hypotheses ensure the rounded second moment and final denominator remain positive.
-/
theorem approxTensor_adamW_update {s : Shape}
    {stateS : Optim.AdamW.State ℝ s} {stateR : Optim.AdamW.State R s}
    {stateError : AdamWStateError s} {derivedErrors : AdamWDerivedErrors}
    {paramsS : Tensor ℝ s} {paramsR : Tensor R s} {paramsError : ℝ}
    {gradsS : Tensor ℝ s} {gradsR : Tensor R s} {gradsError η : ℝ}
    (hstate : adamWStateApprox (β := β) (fexp := fexp) (rnd := rnd)
      stateS stateR stateError)
    (hparams : approxTensor (α := R)
      (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) paramsS paramsR paramsError)
    (hgrads : approxTensor (α := R)
      (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) gradsS gradsR gradsError)
    (honeMinus1 : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (1 - stateR.beta1) -
      (1 - stateS.beta1)) ≤ derivedErrors.oneMinusBeta1)
    (honeMinus2 : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (1 - stateR.beta2) -
      (1 - stateS.beta2)) ≤ derivedErrors.oneMinusBeta2)
    (hbias1 : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd)
      (1 / (1 - Optim.scalarPowNat stateR.beta1 (stateR.stepCount + 1))) -
      (1 / (1 - Optim.scalarPowNat stateS.beta1 (stateS.stepCount + 1)))) ≤
        derivedErrors.firstMomentBiasInverse)
    (hbias2 : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd)
      (1 / (1 - Optim.scalarPowNat stateR.beta2 (stateR.stepCount + 1))) -
      (1 / (1 - Optim.scalarPowNat stateS.beta2 (stateS.stepCount + 1)))) ≤
        derivedErrors.secondMomentBiasInverse)
    (hdecay : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd)
      (stateR.learningRate * stateR.weightDecay) -
      stateS.learningRate * stateS.weightDecay) ≤ derivedErrors.decayScale)
    (hη : 0 < η)
    (hEpsilon : 0 ≤ stateS.epsilon)
    (hMoment2Hat :
      let nextStepCount := stateS.stepCount + 1
      let moment2 := addSpec (scaleSpec stateS.secondMoment stateS.beta2)
        (scaleSpec (squareSpec gradsS) (1 - stateS.beta2))
      Tensor.Forall (fun z : ℝ => η ≤ z)
        (scaleSpec moment2
          (1 / (1 - Optim.scalarPowNat stateS.beta2 nextStepCount))))
    (hMoment2Margin :
      (adamWStepErrorTrace (β := β) (fexp := fexp) (rnd := rnd)
        stateError derivedErrors paramsError gradsError η stateR paramsR gradsR
      ).correctedSecondMoment < η)
    (hDenominatorMargin :
      (adamWStepErrorTrace (β := β) (fexp := fexp) (rnd := rnd)
        stateError derivedErrors paramsError gradsError η stateR paramsR gradsR
      ).denominator < Real.sqrt η) :
    let trace := adamWStepErrorTrace (β := β) (fexp := fexp) (rnd := rnd)
      stateError derivedErrors paramsError gradsError η stateR paramsR gradsR
    adamWStateApprox (β := β) (fexp := fexp) (rnd := rnd)
        (Optim.AdamW.update stateS paramsS gradsS).optimizerState
        (Optim.AdamW.update stateR paramsR gradsR).optimizerState
        { stateError with
          firstMoment := trace.firstMoment
          secondMoment := trace.secondMoment } ∧
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (Optim.AdamW.update stateS paramsS gradsS).parameters
        (Optim.AdamW.update stateR paramsR gradsR).parameters
        trace.parameterError := by
  dsimp only
  rcases hstate with ⟨hlr, hbeta1, hbeta2, hepsilon, hweightDecay, hm, hv, ht⟩
  let mS := addSpec (scaleSpec stateS.firstMoment stateS.beta1)
    (scaleSpec gradsS (1 - stateS.beta1))
  let mR := addSpec (scaleSpec stateR.firstMoment stateR.beta1)
    (scaleSpec gradsR (1 - stateR.beta1))
  let vS := addSpec (scaleSpec stateS.secondMoment stateS.beta2)
    (scaleSpec (squareSpec gradsS) (1 - stateS.beta2))
  let vR := addSpec (scaleSpec stateR.secondMoment stateR.beta2)
    (scaleSpec (squareSpec gradsR) (1 - stateR.beta2))
  let bias1S := 1 / (1 - Optim.scalarPowNat stateS.beta1 (stateS.stepCount + 1))
  let bias1R := 1 / (1 - Optim.scalarPowNat stateR.beta1 (stateR.stepCount + 1))
  let bias2S := 1 / (1 - Optim.scalarPowNat stateS.beta2 (stateS.stepCount + 1))
  let bias2R := 1 / (1 - Optim.scalarPowNat stateR.beta2 (stateR.stepCount + 1))
  let mHatS := scaleSpec mS bias1S
  let mHatR := scaleSpec mR bias1R
  let vHatS := scaleSpec vS bias2S
  let vHatR := scaleSpec vR bias2R
  let trace := adamWStepErrorTrace (β := β) (fexp := fexp) (rnd := rnd)
    stateError derivedErrors paramsError gradsError η stateR paramsR gradsR
  have hmLeft := approxTensor_scale_spec_of_approx
    (β := β) (fexp := fexp) (rnd := rnd) stateS.beta1 stateR.beta1 hm hbeta1
  have hmRight := approxTensor_scale_spec_of_approx
    (β := β) (fexp := fexp) (rnd := rnd)
    (1 - stateS.beta1) (1 - stateR.beta1) hgrads honeMinus1
  have hm' := approxTensor_add_spec (β := β) (fexp := fexp) (rnd := rnd) hmLeft hmRight
  have hsq := approxTensor_square_spec (β := β) (fexp := fexp) (rnd := rnd) hgrads
  have hvLeft := approxTensor_scale_spec_of_approx
    (β := β) (fexp := fexp) (rnd := rnd) stateS.beta2 stateR.beta2 hv hbeta2
  have hvRight := approxTensor_scale_spec_of_approx
    (β := β) (fexp := fexp) (rnd := rnd)
    (1 - stateS.beta2) (1 - stateR.beta2) hsq honeMinus2
  have hv' := approxTensor_add_spec (β := β) (fexp := fexp) (rnd := rnd) hvLeft hvRight
  have hmHat := approxTensor_scale_spec_of_approx
    (β := β) (fexp := fexp) (rnd := rnd) bias1S bias1R hm'
      (by simpa [bias1S, bias1R] using hbias1)
  have hvHat := approxTensor_scale_spec_of_approx
    (β := β) (fexp := fexp) (rnd := rnd) bias2S bias2R hv'
      (by simpa [bias2S, bias2R] using hbias2)
  have hsqrt := approxTensor_sqrt_spec_of_pos_lb
    (β := β) (fexp := fexp) (rnd := rnd) η hη hvHat
      (by simpa [vHatS, vS, bias2S] using hMoment2Hat)
      hMoment2Margin
  have hepsilonFill := approxTensor_full_const
    (β := β) (fexp := fexp) (rnd := rnd) hepsilon (s := s)
  have hdenominator := approxTensor_add_spec
    (β := β) (fexp := fexp) (rnd := rnd) hsqrt hepsilonFill
  have hstdLower : Tensor.Forall (fun z : ℝ => Real.sqrt η ≤ z) (sqrtSpec vHatS) := by
    apply Tensor.forall_mapSpec (by simpa [vHatS, vS, bias2S] using hMoment2Hat)
    intro z hz
    change Real.sqrt η ≤ Real.sqrt (max z 0)
    exact Real.sqrt_le_sqrt (le_trans hz (le_max_left z 0))
  have hepsilonLower : Tensor.Forall (fun z : ℝ => 0 ≤ z) (Tensor.full s stateS.epsilon) :=
    Tensor.forall_full hEpsilon
  have hdenominatorLower : Tensor.Forall (fun z : ℝ => Real.sqrt η ≤ z)
      (addSpec (sqrtSpec vHatS) (Tensor.full s stateS.epsilon)) := by
    apply Tensor.forall_map2Spec hstdLower hepsilonLower
    intro a b ha hb
    linarith
  have hlrFill := approxTensor_full_const
    (β := β) (fexp := fexp) (rnd := rnd) hlr (s := s)
  have hadaptive := approxTensor_div_spec_of_pos_lb
    (β := β) (fexp := fexp) (rnd := rnd) (Real.sqrt η)
    hlrFill hdenominator hdenominatorLower hDenominatorMargin
  have hadamUpdate := approxTensor_mul_spec
    (β := β) (fexp := fexp) (rnd := rnd) hadaptive hmHat
  have hdecayUpdate := approxTensor_scale_spec_of_approx
    (β := β) (fexp := fexp) (rnd := rnd)
    (stateS.learningRate * stateS.weightDecay)
    (stateR.learningRate * stateR.weightDecay) hparams hdecay
  have hdecayed := approxTensor_sub_spec
    (β := β) (fexp := fexp) (rnd := rnd) hparams hdecayUpdate
  have hnext := approxTensor_sub_spec
    (β := β) (fexp := fexp) (rnd := rnd) hdecayed hadamUpdate
  constructor
  · refine ⟨hlr, hbeta1, hbeta2, hepsilon, hweightDecay, ?_, ?_, ?_⟩
    · simpa [Optim.AdamW.update, trace, adamWStepErrorTrace, mS, mR] using hm'
    · simpa [Optim.AdamW.update, trace, adamWStepErrorTrace, vS, vR] using hv'
    · simpa [Optim.AdamW.update] using ht
  · simpa [trace, adamWStepErrorTrace, Optim.AdamW.update,
      Optim.adaptiveLearningRate, mS, mR, vS, vR, mHatS, mHatR,
      vHatS, vHatR, bias1S, bias1R, bias2S, bias2R] using hnext

/-! ## AdamW contract instance -/

/-- Numerical assumptions and positivity margin for one AdamW update. -/
structure AdamWStepAssumptions where
  /-- Bounds for rounded scalar subexpressions used by bias correction and decay. -/
  derivedErrors : AdamWDerivedErrors
  /-- Strict lower bound on the exact bias-corrected second moment. -/
  minimumSecondMoment : ℝ

/-- Complete validity predicate for one AdamW contract application. -/
def adamWAssumptionsHold {s : Shape}
    (stateS : Optim.AdamW.State ℝ s) (stateR : Optim.AdamW.State R s)
    (stateError : AdamWStateError s)
    (_paramsS : Tensor ℝ s) (paramsR : Tensor R s) (paramsError : ℝ)
    (gradsS : Tensor ℝ s) (gradsR : Tensor R s) (gradsError : ℝ)
    (assumptions : AdamWStepAssumptions) : Prop :=
  abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (1 - stateR.beta1) -
      (1 - stateS.beta1)) ≤ assumptions.derivedErrors.oneMinusBeta1 ∧
  abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (1 - stateR.beta2) -
      (1 - stateS.beta2)) ≤ assumptions.derivedErrors.oneMinusBeta2 ∧
  abs (toSpec (β := β) (fexp := fexp) (rnd := rnd)
      (1 / (1 - Optim.scalarPowNat stateR.beta1 (stateR.stepCount + 1))) -
      (1 / (1 - Optim.scalarPowNat stateS.beta1 (stateS.stepCount + 1)))) ≤
    assumptions.derivedErrors.firstMomentBiasInverse ∧
  abs (toSpec (β := β) (fexp := fexp) (rnd := rnd)
      (1 / (1 - Optim.scalarPowNat stateR.beta2 (stateR.stepCount + 1))) -
      (1 / (1 - Optim.scalarPowNat stateS.beta2 (stateS.stepCount + 1)))) ≤
    assumptions.derivedErrors.secondMomentBiasInverse ∧
  abs (toSpec (β := β) (fexp := fexp) (rnd := rnd)
      (stateR.learningRate * stateR.weightDecay) -
      stateS.learningRate * stateS.weightDecay) ≤
    assumptions.derivedErrors.decayScale ∧
  0 < assumptions.minimumSecondMoment ∧
  0 ≤ stateS.epsilon ∧
  (let nextStepCount := stateS.stepCount + 1
   let moment2 := addSpec (scaleSpec stateS.secondMoment stateS.beta2)
     (scaleSpec (squareSpec gradsS) (1 - stateS.beta2))
   Tensor.Forall (fun z : ℝ => assumptions.minimumSecondMoment ≤ z)
     (scaleSpec moment2
       (1 / (1 - Optim.scalarPowNat stateS.beta2 nextStepCount)))) ∧
  (adamWStepErrorTrace (β := β) (fexp := fexp) (rnd := rnd)
      stateError assumptions.derivedErrors paramsError gradsError
      assumptions.minimumSecondMoment stateR paramsR gradsR).correctedSecondMoment <
    assumptions.minimumSecondMoment ∧
  (adamWStepErrorTrace (β := β) (fexp := fexp) (rnd := rnd)
      stateError assumptions.derivedErrors paramsError gradsError
      assumptions.minimumSecondMoment stateR paramsR gradsR).denominator <
    Real.sqrt assumptions.minimumSecondMoment

/-- State and parameter error object produced by one AdamW contract step. -/
def adamWStepError {s : Shape} (stateError : AdamWStateError s)
    (paramsError gradsError : ℝ) (stateR : Optim.AdamW.State R s)
    (paramsR gradsR : Tensor R s) (assumptions : AdamWStepAssumptions) :
    StepError AdamWStateError s :=
  let trace := adamWStepErrorTrace (β := β) (fexp := fexp) (rnd := rnd)
    stateError assumptions.derivedErrors paramsError gradsError
    assumptions.minimumSecondMoment stateR paramsR gradsR
  { optimizerStateError :=
      { stateError with
        firstMoment := trace.firstMoment
        secondMoment := trace.secondMoment }
    parameterError := trace.parameterError }

/-- AdamW instance of the generic numerical optimizer contract.

Its assumptions are proof data, not a second execution framework.
`NumericalStepContract.run_approx` therefore composes AdamW over finite runs exactly as it does SGD
and momentum SGD.
-/
def adamWContract : NumericalStepContract R
    (toSpec (β := β) (fexp := fexp) (rnd := rnd)) where
  name := "AdamW"
  ExactState := Optim.AdamW.State ℝ
  RuntimeState := Optim.AdamW.State R
  StateError := AdamWStateError
  StepAssumptions := fun _ => AdamWStepAssumptions
  stateApprox := adamWStateApprox (β := β) (fexp := fexp) (rnd := rnd)
  assumptionsHold := adamWAssumptionsHold (β := β) (fexp := fexp) (rnd := rnd)
  updateExact := Optim.AdamW.update
  updateRuntime := Optim.AdamW.update
  nextError := fun stateError parameterError gradientError state parameters gradients assumptions =>
    adamWStepError (β := β) (fexp := fexp) (rnd := rnd)
      stateError parameterError gradientError state parameters gradients assumptions
  stateErrorReport := fun error =>
    #[("learning rate", error.learningRate), ("beta1", error.beta1), ("beta2", error.beta2),
      ("epsilon", error.epsilon), ("weight decay", error.weightDecay),
      ("first moment", error.firstMoment), ("second moment", error.secondMoment)]
  assumptionReport := fun assumptions =>
    #[("minimum corrected second moment", assumptions.minimumSecondMoment),
      ("1 - beta1", assumptions.derivedErrors.oneMinusBeta1),
      ("1 - beta2", assumptions.derivedErrors.oneMinusBeta2),
      ("first bias inverse", assumptions.derivedErrors.firstMomentBiasInverse),
      ("second bias inverse", assumptions.derivedErrors.secondMomentBiasInverse),
      ("decay scale", assumptions.derivedErrors.decayScale)]
  updateApprox := by
    intro s stateS stateR stateError paramsS paramsR paramsError gradsS gradsR gradsError
      assumptions
      hstate hparams hgrads hvalid
    rcases hvalid with
      ⟨honeMinus1, honeMinus2, hbias1, hbias2, hdecay, hEta, hEpsilon,
        hMoment2Hat, hMoment2Margin, hDenominatorMargin⟩
    simpa [adamWStepError] using
      (approxTensor_adamW_update (β := β) (fexp := fexp) (rnd := rnd)
        (derivedErrors := assumptions.derivedErrors)
        hstate hparams hgrads honeMinus1 honeMinus2 hbias1 hbias2 hdecay hEta hEpsilon
        hMoment2Hat hMoment2Margin hDenominatorMargin)

end
end Proofs.RuntimeApprox.NFBackend.Optimizer
