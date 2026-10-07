/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.RuntimeApprox.NF.ShapeOps
public import NN.Spec.Layers.Normalization.Core

/-!
# Rounded Normalization Certificates

This module bounds the shared normalization core in the rounded-real `NF` model.
The proof is rank-generic: reductions such as a row mean or variance are certified separately and
then supplied as inputs here. The resulting theorem covers the numerical part shared by LayerNorm,
RMSNorm, BatchNorm, GroupNorm, and related affine normalizations.

If the exact stabilized variance is at least `η > 0`, the square-root stage is controlled by
`1 / sqrt η`. The division theorem requires square-root error strictly smaller than `sqrt η`,
so the rounded denominator cannot cross zero. This is a conditional mathematical bound, not
an executable certificate checker or a native normalization-kernel proof.

References:

* J. L. Ba, J. R. Kiros, G. E. Hinton, *Layer Normalization* (2016),
  https://arxiv.org/abs/1607.06450.
* B. Zhang, R. Sennrich, *Root Mean Square Layer Normalization* (2019),
  https://arxiv.org/abs/1910.07467.
* N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed. (2002), Chapters 2-3.
-/

@[expose] public section

namespace Proofs
namespace RuntimeApprox

open Spec TorchLean
open TorchLean TorchLean.Tensor
open NN.MLTheory.Robustness.Spec

noncomputable section

namespace NFBackend

open FloatLib FloatLib.Numerics FloatLib.Floats.Formats
open Flocq

variable {β : Radix} {fexp : ℤ → ℤ} [ValidExp fexp]
variable {rnd : ℝ → ℤ} [ValidRndToNearest rnd]

local notation "R" => NF β fexp rnd

/-- Stage-by-stage infinity-norm budget for an affine normalization.

Records the errors from centering, variance stabilization, square root, division, and affine
scaling. Its real-valued fields are mathematical quantities, not an executable report.
-/
structure NormalizationErrorTrace where
  centeredError : ℝ
  stabilizedError : ℝ
  stdError : ℝ
  normalizedError : ℝ
  scaledError : ℝ
  outputError : ℝ

/-- Compositional error trace for `Spec.normalizeCore` in the rounded-real model. -/
def normalizeCoreErrorTrace
    {s sMean sVar sGamma sBeta : Shape}
    (cbMean : Shape.CanBroadcastTo sMean s)
    (cbVar : Shape.CanBroadcastTo sVar s)
    (cbGamma : Shape.CanBroadcastTo sGamma s)
    (cbBeta : Shape.CanBroadcastTo sBeta s)
    (epsilonR : R)
    (xR : Tensor R s)
    (meanR : Tensor R sMean)
    (varianceR : Tensor R sVar)
    (gammaR : Tensor R sGamma)
    (betaR : Tensor R sBeta)
    (epsX epsMean epsVariance epsGamma epsBeta epsEpsilon η : ℝ) :
    NormalizationErrorTrace :=
  let meanBroadcastR := broadcastTo cbMean meanR
  let varianceBroadcastR := broadcastTo cbVar varianceR
  let gammaBroadcastR := broadcastTo cbGamma gammaR
  let betaBroadcastR := broadcastTo cbBeta betaR
  let epsilonFillR := Tensor.full s epsilonR
  let centeredR := subSpec xR meanBroadcastR
  let stabilizedR := addSpec varianceBroadcastR epsilonFillR
  let centeredError :=
    linfNorm (subBoundTensor (β := β) (fexp := fexp) epsX epsMean xR meanBroadcastR)
  let stabilizedError :=
    linfNorm
      (addBoundTensor (β := β) (fexp := fexp) epsVariance epsEpsilon
        varianceBroadcastR epsilonFillR)
  let stdR := sqrtSpec stabilizedR
  let stdError :=
    linfNorm
      (sqrtPosBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
        η stabilizedError stabilizedR)
  let normalizedR := divSpec centeredR stdR
  let normalizedError :=
    linfNorm
      (divPosBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
        (Real.sqrt η) centeredError stdError centeredR stdR)
  let scaledR := mulSpec normalizedR gammaBroadcastR
  let scaledError :=
    linfNorm
      (mulBoundTensor (β := β) (fexp := fexp)
        normalizedError epsGamma normalizedR gammaBroadcastR)
  let outputError :=
    linfNorm
      (addBoundTensor (β := β) (fexp := fexp)
        scaledError epsBeta scaledR betaBroadcastR)
  { centeredError, stabilizedError, stdError, normalizedError, scaledError, outputError }

/-- The rounded-real affine-normalization core approximates its exact real semantics.

The five input tensor hypotheses can themselves come from reductions or earlier graph nodes. The
epsilon scalar is treated like any other rounded constant. The exact stabilized variance must have
the pointwise lower bound `η`. The theorem retains two strict error-margin premises; its proof
uses the square-root margin for division, but does not use the stabilized-variance error margin
because `sqrtSpec` clamps its input. Both are propositions about the noncomputable error trace.
-/
theorem approxTensor_normalizeCore
    {s sMean sVar sGamma sBeta : Shape}
    (cbMean : Shape.CanBroadcastTo sMean s)
    (cbVar : Shape.CanBroadcastTo sVar s)
    (cbGamma : Shape.CanBroadcastTo sGamma s)
    (cbBeta : Shape.CanBroadcastTo sBeta s)
    (epsilonS : ℝ) (epsilonR : R)
    {xS : SpecTensor s} {xR : Tensor R s}
    {meanS : SpecTensor sMean} {meanR : Tensor R sMean}
    {varianceS : SpecTensor sVar} {varianceR : Tensor R sVar}
    {gammaS : SpecTensor sGamma} {gammaR : Tensor R sGamma}
    {betaS : SpecTensor sBeta} {betaR : Tensor R sBeta}
    {epsX epsMean epsVariance epsGamma epsBeta epsEpsilon η : ℝ}
    (hη : 0 < η)
    (hx : approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
      xS xR epsX)
    (hmean : approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
      meanS meanR epsMean)
    (hvariance : approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
      varianceS varianceR epsVariance)
    (hgamma : approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
      gammaS gammaR epsGamma)
    (hbeta : approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
      betaS betaR epsBeta)
    (hepsilon :
      abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) epsilonR - epsilonS) ≤ epsEpsilon)
    (hstabilized :
      Tensor.Forall (fun z : ℝ => η ≤ z)
        (addSpec (broadcastTo cbVar varianceS) (Tensor.full s epsilonS))) :
    let trace := normalizeCoreErrorTrace (β := β) (fexp := fexp) (rnd := rnd)
      cbMean cbVar cbGamma cbBeta epsilonR xR meanR varianceR gammaR betaR
      epsX epsMean epsVariance epsGamma epsBeta epsEpsilon η
    trace.stabilizedError < η →
    trace.stdError < Real.sqrt η →
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (normalizeCore s sMean sVar sGamma sBeta epsilonS
          xS meanS varianceS gammaS betaS cbMean cbVar cbGamma cbBeta)
        (normalizeCore s sMean sVar sGamma sBeta epsilonR
          xR meanR varianceR gammaR betaR cbMean cbVar cbGamma cbBeta)
        trace.outputError := by
  dsimp only
  intro _hstabilizedMargin hstdMargin
  let meanBroadcastS := broadcastTo cbMean meanS
  let meanBroadcastR := broadcastTo cbMean meanR
  let varianceBroadcastS := broadcastTo cbVar varianceS
  let varianceBroadcastR := broadcastTo cbVar varianceR
  let gammaBroadcastS := broadcastTo cbGamma gammaS
  let gammaBroadcastR := broadcastTo cbGamma gammaR
  let betaBroadcastS := broadcastTo cbBeta betaS
  let betaBroadcastR := broadcastTo cbBeta betaR
  let epsilonFillS := Tensor.full s epsilonS
  let epsilonFillR := Tensor.full s epsilonR
  let centeredS := subSpec xS meanBroadcastS
  let centeredR := subSpec xR meanBroadcastR
  let stabilizedS := addSpec varianceBroadcastS epsilonFillS
  let stabilizedR := addSpec varianceBroadcastR epsilonFillR
  let centeredError :=
    linfNorm (subBoundTensor (β := β) (fexp := fexp) epsX epsMean xR meanBroadcastR)
  let stabilizedError :=
    linfNorm
      (addBoundTensor (β := β) (fexp := fexp) epsVariance epsEpsilon
        varianceBroadcastR epsilonFillR)
  let stdS := sqrtSpec stabilizedS
  let stdR := sqrtSpec stabilizedR
  let stdError :=
    linfNorm
      (sqrtPosBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
        η stabilizedError stabilizedR)
  let normalizedS := divSpec centeredS stdS
  let normalizedR := divSpec centeredR stdR
  let normalizedError :=
    linfNorm
      (divPosBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
        (Real.sqrt η) centeredError stdError centeredR stdR)
  let scaledS := mulSpec normalizedS gammaBroadcastS
  let scaledR := mulSpec normalizedR gammaBroadcastR
  let scaledError :=
    linfNorm
      (mulBoundTensor (β := β) (fexp := fexp)
        normalizedError epsGamma normalizedR gammaBroadcastR)

  have hmeanBroadcast := approxTensor_broadcastTo
    (β := β) (fexp := fexp) (rnd := rnd) cbMean hmean
  have hvarianceBroadcast := approxTensor_broadcastTo
    (β := β) (fexp := fexp) (rnd := rnd) cbVar hvariance
  have hgammaBroadcast := approxTensor_broadcastTo
    (β := β) (fexp := fexp) (rnd := rnd) cbGamma hgamma
  have hbetaBroadcast := approxTensor_broadcastTo
    (β := β) (fexp := fexp) (rnd := rnd) cbBeta hbeta
  have hepsilonFill := approxTensor_full_const
    (β := β) (fexp := fexp) (rnd := rnd) hepsilon (s := s)
  have hcentered := approxTensor_sub_spec
    (β := β) (fexp := fexp) (rnd := rnd) hx hmeanBroadcast
  have hstabilizedApprox := approxTensor_add_spec
    (β := β) (fexp := fexp) (rnd := rnd) hvarianceBroadcast hepsilonFill
  have hstd := approxTensor_sqrt_spec_of_pos_lb
    (β := β) (fexp := fexp) (rnd := rnd) η hη
    hstabilizedApprox hstabilized
  have hstdLower : Tensor.Forall (fun z : ℝ => Real.sqrt η ≤ z) stdS := by
    apply Tensor.forall_map hstabilized
    intro z hz
    change Real.sqrt η ≤ Real.sqrt (max z 0)
    exact Real.sqrt_le_sqrt (le_trans hz (le_max_left z 0))
  have hnormalized := approxTensor_div_spec_of_pos_lb
    (β := β) (fexp := fexp) (rnd := rnd) (Real.sqrt η)
    hcentered hstd hstdLower hstdMargin
  have hscaled := approxTensor_mul_spec
    (β := β) (fexp := fexp) (rnd := rnd) hnormalized hgammaBroadcast
  have houtput := approxTensor_add_spec
    (β := β) (fexp := fexp) (rnd := rnd) hscaled hbetaBroadcast
  simpa [normalizeCore, normalizeCoreErrorTrace, meanBroadcastS, meanBroadcastR,
    varianceBroadcastS, varianceBroadcastR, gammaBroadcastS, gammaBroadcastR,
    betaBroadcastS, betaBroadcastR, epsilonFillS, epsilonFillR, centeredS, centeredR,
    stabilizedS, stabilizedR, centeredError, stabilizedError, stdS, stdR, stdError,
    normalizedS, normalizedR, normalizedError, scaledS, scaledR, scaledError] using houtput

end NFBackend

end
end RuntimeApprox
end Proofs
