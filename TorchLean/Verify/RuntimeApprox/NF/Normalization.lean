/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.RuntimeApprox.NF.ShapeOps
public import TorchLean.Tensor.Layers.Normalization.LayerNorm

/-!
# Rounded Normalization Certificates

This module bounds `TorchLean.Normalization.normalize` in the rounded-real `NF` model when it
normalizes the input less a broadcast mean with a constant `epsilon` as the stabilizer, as
`Tensor.batchNorm` does with running statistics. The proof is rank-generic: the mean, the variance
and the affine parameters are certified separately and then supplied as inputs here.

If the exact stabilized variance is at least `η > 0`, the square-root stage is controlled by
`1 / sqrt η`. The division theorem requires square-root error strictly smaller than `sqrt η`, so
the rounded denominator cannot cross zero. The result is a conditional mathematical bound, and
executable certificate checking and native normalization kernels lie outside it.

References:

* J. L. Ba, J. R. Kiros, G. E. Hinton, *Layer Normalization* (2016),
  https://arxiv.org/abs/1607.06450.
* B. Zhang, R. Sennrich, *Root Mean Square Layer Normalization* (2019),
  https://arxiv.org/abs/1910.07467.
* N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed. (2002), Chapters 2-3.
-/

@[expose] public section

namespace TorchLean

open Normalization
open TorchLean.Tensor
open Robustness

noncomputable section

section

open FloatLib FloatLib.Numerics FloatLib.Floats.Formats
open Flocq
open TorchLean.NF

variable {β : Radix} {fexp : ℤ → ℤ} [ValidExp fexp]
variable {rnd : ℝ → ℤ} [ValidRndToNearest rnd]

local notation "R" => NF β fexp rnd

/-- Stage-by-stage infinity-norm budget for an affine normalization.

The fields record the errors after centering, variance stabilization, square root, division,
scaling, and the final shift, as real numbers used in proofs.
-/
structure NF.NormalizationErrorTrace where
  /-- The error after centering. -/
  centeredError : ℝ
  /-- The error after variance stabilization. -/
  stabilizedError : ℝ
  /-- The error after the square root. -/
  stdError : ℝ
  /-- The error after the division. -/
  normalizedError : ℝ
  /-- The error after scaling. -/
  scaledError : ℝ
  /-- The error after the final shift. -/
  outputError : ℝ

/-- Compositional error trace of `Normalization.normalize` in the rounded-real model, applied to
`xR` less the broadcast `meanR` with the stabilizer `epsilonR` at every entry. -/
def NF.normalizeErrorTrace
    {s sMean sVar sGamma sBeta : Shape}
    {meanDims : Fin sMean.rank → Fin s.rank} (hMean : sMean.BroadcastDims s meanDims)
    {varDims : Fin sVar.rank → Fin s.rank} (hVar : sVar.BroadcastDims s varDims)
    {gammaDims : Fin sGamma.rank → Fin s.rank} (hGamma : sGamma.BroadcastDims s gammaDims)
    {betaDims : Fin sBeta.rank → Fin s.rank} (hBeta : sBeta.BroadcastDims s betaDims)
    (epsilonR : R)
    (xR : Tensor R s)
    (meanR : Tensor R sMean)
    (varianceR : Tensor R sVar)
    (gammaR : Tensor R sGamma)
    (betaR : Tensor R sBeta)
    (epsX epsMean epsVariance epsGamma epsBeta epsEpsilon η : ℝ) :
    NormalizationErrorTrace :=
  let meanBroadcastR := Tensor.broadcast meanR _ meanDims hMean
  let varianceBroadcastR := Tensor.broadcast varianceR _ varDims hVar
  let gammaBroadcastR := Tensor.broadcast gammaR _ gammaDims hGamma
  let betaBroadcastR := Tensor.broadcast betaR _ betaDims hBeta
  let epsilonFillR := Tensor.full s epsilonR
  let centeredR := xR - meanBroadcastR
  let stabilizedR := varianceBroadcastR + epsilonFillR
  let centeredError :=
    linfNorm (subBoundTensor (β := β) (fexp := fexp) epsX epsMean xR meanBroadcastR)
  let stabilizedError :=
    linfNorm
      (addBoundTensor (β := β) (fexp := fexp) epsVariance epsEpsilon
        varianceBroadcastR epsilonFillR)
  let stdR := Tensor.sqrt stabilizedR .clampNegative
  let stdError :=
    linfNorm
      (sqrtPosBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
        η stabilizedError stabilizedR)
  let normalizedR := centeredR / stdR
  let normalizedError :=
    linfNorm
      (divPosBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
        (Real.sqrt η) centeredError stdError centeredR stdR)
  let scaledR := normalizedR * gammaBroadcastR
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

The five input tensor hypotheses can come from reductions or earlier graph nodes, and the epsilon
scalar is treated like any other rounded constant. The exact stabilized variance must have the
pointwise lower bound `η`. The theorem keeps two strict error-margin premises about the
noncomputable error trace. Its proof uses the square-root margin for division and leaves the
stabilized-variance margin unused, because `Tensor.sqrt _ .clampNegative` clamps its input.
-/
theorem ApproxTensor.normalize
    {s sMean sVar sGamma sBeta : Shape}
    {meanDims : Fin sMean.rank → Fin s.rank} (hMean : sMean.BroadcastDims s meanDims)
    {varDims : Fin sVar.rank → Fin s.rank} (hVar : sVar.BroadcastDims s varDims)
    {gammaDims : Fin sGamma.rank → Fin s.rank} (hGamma : sGamma.BroadcastDims s gammaDims)
    {betaDims : Fin sBeta.rank → Fin s.rank} (hBeta : sBeta.BroadcastDims s betaDims)
    (epsilonS : ℝ) (epsilonR : R)
    {xS : Tensor ℝ s} {xR : Tensor R s}
    {meanS : Tensor ℝ sMean} {meanR : Tensor R sMean}
    {varianceS : Tensor ℝ sVar} {varianceR : Tensor R sVar}
    {gammaS : Tensor ℝ sGamma} {gammaR : Tensor R sGamma}
    {betaS : Tensor ℝ sBeta} {betaR : Tensor R sBeta}
    {epsX epsMean epsVariance epsGamma epsBeta epsEpsilon η : ℝ}
    (hη : 0 < η)
    (hx : ApproxTensor (α := R) (toSpec := NF.toReal)
      xS xR epsX)
    (hmean : ApproxTensor (α := R) (toSpec := NF.toReal)
      meanS meanR epsMean)
    (hvariance : ApproxTensor (α := R) (toSpec := NF.toReal)
      varianceS varianceR epsVariance)
    (hgamma : ApproxTensor (α := R) (toSpec := NF.toReal)
      gammaS gammaR epsGamma)
    (hbeta : ApproxTensor (α := R) (toSpec := NF.toReal)
      betaS betaR epsBeta)
    (hepsilon :
      _root_.abs (NF.toReal epsilonR - epsilonS) ≤ epsEpsilon)
    (hstabilized :
      Tensor.Forall (fun z : ℝ => η ≤ z)
        (Tensor.broadcast varianceS _ varDims hVar +
            Tensor.full s epsilonS)) :
    let trace := normalizeErrorTrace (β := β) (fexp := fexp) (rnd := rnd)
      hMean hVar hGamma hBeta epsilonR xR meanR varianceR gammaR betaR
      epsX epsMean epsVariance epsGamma epsBeta epsEpsilon η
    trace.stabilizedError < η →
    trace.stdError < Real.sqrt η →
      ApproxTensor (α := R) (toSpec := NF.toReal)
        (Normalization.normalize (xS - Tensor.broadcast meanS _ meanDims hMean) varianceS
          (Tensor.full sVar epsilonS) gammaS betaS hVar hGamma hBeta)
        (Normalization.normalize (xR - Tensor.broadcast meanR _ meanDims hMean) varianceR
          (Tensor.full sVar epsilonR) gammaR betaR hVar hGamma hBeta)
        trace.outputError := by
  dsimp only
  intro _hstabilizedMargin hstdMargin
  rw [normalize_full, normalize_full]
  let meanBroadcastS := Tensor.broadcast meanS _ meanDims hMean
  let meanBroadcastR := Tensor.broadcast meanR _ meanDims hMean
  let varianceBroadcastS := Tensor.broadcast varianceS _ varDims hVar
  let varianceBroadcastR := Tensor.broadcast varianceR _ varDims hVar
  let gammaBroadcastS := Tensor.broadcast gammaS _ gammaDims hGamma
  let gammaBroadcastR := Tensor.broadcast gammaR _ gammaDims hGamma
  let betaBroadcastS := Tensor.broadcast betaS _ betaDims hBeta
  let betaBroadcastR := Tensor.broadcast betaR _ betaDims hBeta
  let epsilonFillS := Tensor.full s epsilonS
  let epsilonFillR := Tensor.full s epsilonR
  let centeredS := xS - meanBroadcastS
  let centeredR := xR - meanBroadcastR
  let stabilizedS := varianceBroadcastS + epsilonFillS
  let stabilizedR := varianceBroadcastR + epsilonFillR
  let centeredError :=
    linfNorm (subBoundTensor (β := β) (fexp := fexp) epsX epsMean xR meanBroadcastR)
  let stabilizedError :=
    linfNorm
      (addBoundTensor (β := β) (fexp := fexp) epsVariance epsEpsilon
        varianceBroadcastR epsilonFillR)
  let stdS := Tensor.sqrt stabilizedS .clampNegative
  let stdR := Tensor.sqrt stabilizedR .clampNegative
  let stdError :=
    linfNorm
      (sqrtPosBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
        η stabilizedError stabilizedR)
  let normalizedS := centeredS / stdS
  let normalizedR := centeredR / stdR
  let normalizedError :=
    linfNorm
      (divPosBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
        (Real.sqrt η) centeredError stdError centeredR stdR)
  let scaledS := normalizedS * gammaBroadcastS
  let scaledR := normalizedR * gammaBroadcastR
  let scaledError :=
    linfNorm
      (mulBoundTensor (β := β) (fexp := fexp)
        normalizedError epsGamma normalizedR gammaBroadcastR)

  have hmeanBroadcast := broadcast
    (β := β) (fexp := fexp) (rnd := rnd) meanDims hMean hmean
  have hvarianceBroadcast := broadcast
    (β := β) (fexp := fexp) (rnd := rnd) varDims hVar hvariance
  have hgammaBroadcast := broadcast
    (β := β) (fexp := fexp) (rnd := rnd) gammaDims hGamma hgamma
  have hbetaBroadcast := broadcast
    (β := β) (fexp := fexp) (rnd := rnd) betaDims hBeta hbeta
  have hepsilonFill := full
    (β := β) (fexp := fexp) (rnd := rnd) hepsilon (s := s)
  have hcentered := sub
    (β := β) (fexp := fexp) (rnd := rnd) hx hmeanBroadcast
  have hstabilizedApprox := add
    (β := β) (fexp := fexp) (rnd := rnd) hvarianceBroadcast hepsilonFill
  have hstd := sqrt_of_pos_lb
    (β := β) (fexp := fexp) (rnd := rnd) η hη
    hstabilizedApprox hstabilized
  have hstdLower : Tensor.Forall (fun z : ℝ => Real.sqrt η ≤ z) stdS := by
    rw [show stdS = _ from Tensor.sqrt_clampNegative stabilizedS]
    apply Tensor.forall_map hstabilized
    intro z hz
    change Real.sqrt η ≤ Real.sqrt (max z 0)
    exact Real.sqrt_le_sqrt (le_trans hz (le_max_left z 0))
  have hnormalized := div_of_pos_lb
    (β := β) (fexp := fexp) (rnd := rnd) (Real.sqrt η)
    hcentered hstd hstdLower hstdMargin
  have hscaled := mul
    (β := β) (fexp := fexp) (rnd := rnd) hnormalized hgammaBroadcast
  have houtput := add
    (β := β) (fexp := fexp) (rnd := rnd) hscaled hbetaBroadcast
  simpa [normalizeErrorTrace, meanBroadcastS, meanBroadcastR,
    varianceBroadcastS, varianceBroadcastR, gammaBroadcastS, gammaBroadcastR,
    betaBroadcastS, betaBroadcastR, epsilonFillS, epsilonFillR, centeredS, centeredR,
    stabilizedS, stabilizedR, centeredError, stabilizedError, stdS, stdR, stdError,
    normalizedS, normalizedR, normalizedError, scaledS, scaledR, scaledError] using houtput

end

end
end TorchLean
