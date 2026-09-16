/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.RuntimeApprox.NF.Linalg
public import NN.Proofs.RuntimeApprox.NF.SoftmaxAxis

/-!
# Rounded scaled dot-product attention

This module connects TorchLean's stable last-axis softmax theorem to the matrix operations used by
attention. The forward theorem follows the actual unmasked computation

`softmax(c * (Q Kᵀ)) V`,

where `c` is normally `1 / sqrt(d)`. Both the matrices and `c` may be approximate. The result is a
single infinity-norm budget assembled from the existing transpose, matrix-multiplication,
coefficient-aware scaling, and row-softmax contracts.

The theorem keeps the rounded coefficient error explicit. A caller may obtain it from the concrete
construction of `1 / sqrt(d)`, from an interval certificate, or from a backend capsule. Hiding that
error would incorrectly treat a rounded normalization factor as an exact real constant.

References:

* A. Vaswani et al., "Attention Is All You Need," NeurIPS 2017.
* N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM, 2002.
* PyTorch, `torch.nn.functional.scaled_dot_product_attention`, for the runtime operator and
  last-axis normalization convention.
-/

@[expose] public section

namespace Proofs
namespace RuntimeApprox
namespace Attention

open Spec TorchLean
open TorchLean TorchLean.Tensor
open NN.MLTheory.Robustness.Spec
open FloatLib FloatLib.Numerics FloatLib.Floats.Formats
open Flocq

noncomputable section

variable {β : Radix} {fexp : ℤ -> ℤ} [ValidExp fexp]
variable {rnd : ℝ -> ℤ} [ValidRndToNearest rnd]

local notation "R" => NF β fexp rnd

/-! ## The canonical `1 / sqrt(d)` coefficient -/

/-- Rounding budget for embedding the positive feature dimension into `NF`. -/
def dimensionCastError (d : Nat) : ℝ :=
  ulp β fexp (Nat.succ d : ℝ) / 2

/-- Error budget for the rounded square root of the embedded feature dimension. -/
def dimensionSqrtError (d : Nat) : ℝ :=
  let dR : R := (Nat.succ d : Nat)
  dimensionCastError (β := β) (fexp := fexp) d / Real.sqrt 1 +
    ulp β fexp (Real.sqrt (NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd) dR)) / 2

/-- End-to-end error budget for constructing `1 / sqrt(d)` in the rounded backend. -/
def canonicalScaleErrorBound (d : Nat) : ℝ :=
  let oneR : R := 1
  let dR : R := (Nat.succ d : Nat)
  let sqrtR : R := Numerics.MathFunctions.sqrt dR
  NFBackend.divPosErrorBound (β := β) (fexp := fexp) 1
    (ulp β fexp 1 / 2)
    (dimensionSqrtError (β := β) (fexp := fexp) (rnd := rnd) d)
    (NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd) oneR)
    (NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd) sqrtR)

/-- The canonical rounded attention scale approximates the exact real `1 / sqrt(d)`.

The side condition is format-sensitive and executable: the accumulated square-root error must be
smaller than the exact lower bound one. For IEEE binary32 and ordinary transformer dimensions it
is tiny; keeping it explicit also makes the theorem valid for low-precision experimental formats.
-/
theorem approx_canonicalAttentionScale (d : Nat)
    (hbudget : dimensionSqrtError (β := β) (fexp := fexp) (rnd := rnd) d < 1) :
    abs
        (NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd)
            (1 / Spec.attentionScaleDenom (α := R) (Nat.succ d)) -
          (1 / Spec.attentionScaleDenom (α := ℝ) (Nat.succ d))) ≤
      canonicalScaleErrorBound (β := β) (fexp := fexp) (rnd := rnd) d := by
  let dS : ℝ := (Nat.succ d : ℝ)
  let dR : R := (Nat.succ d : Nat)
  let epsD := dimensionCastError (β := β) (fexp := fexp) d
  let sqrtR : R := Numerics.MathFunctions.sqrt dR
  let epsSqrt := dimensionSqrtError (β := β) (fexp := fexp) (rnd := rnd) d
  let oneR : R := 1
  let epsOne := ulp β fexp 1 / 2
  have hdLower : (1 : ℝ) ≤ dS := by
    dsimp [dS]
    exact_mod_cast Nat.succ_le_succ (Nat.zero_le d)
  have hcast :
      abs (NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd) dR - dS) ≤ epsD := by
    simpa [dR, dS, epsD, dimensionCastError, NF.instNatCast] using
      (NFBackend.approx_ofReal_nf (β := β) (fexp := fexp) (rnd := rnd) dS)
  have hdR0 : 0 ≤ NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd) dR := by
    dsimp [dR, NFBackend.toSpec, NF.toReal, NF.instNatCast,
      NF.ofReal, NF.roundR]
    exact round_nonneg rnd (by positivity)
  have hsqrt :
      abs (NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd) sqrtR - Real.sqrt dS) ≤
        epsSqrt := by
    have h := NFBackend.approx_sqrt_nf_of_pos_lb (β := β) (fexp := fexp) (rnd := rnd)
      (x := dS) (xR := dR) (eps := epsD) (η := (1 : ℝ))
      (by norm_num) hdLower hdR0 hcast
    simpa [sqrtR, epsSqrt, dimensionSqrtError, dR, epsD] using h
  have hsqrtLower : (1 : ℝ) ≤ Real.sqrt dS := by
    exact Real.one_le_sqrt.mpr hdLower
  have hone :
      abs (NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd) oneR - (1 : ℝ)) ≤ epsOne := by
    have honeEq : oneR =
        NF.ofReal (β := β) (fexp := fexp) (rnd := rnd) (1 : ℝ) := by
      rfl
    rw [honeEq]
    simpa [epsOne] using
      (NFBackend.approx_ofReal_nf (β := β) (fexp := fexp) (rnd := rnd) (1 : ℝ))
  have hdiv := NFBackend.approx_div_nf_of_pos_lb
    (β := β) (fexp := fexp) (rnd := rnd)
    (x := (1 : ℝ)) (y := Real.sqrt dS) (xR := oneR) (yR := sqrtR)
    (epsx := epsOne) (epsy := epsSqrt) (η := (1 : ℝ))
    hsqrtLower (by simpa [epsSqrt] using hbudget) hone hsqrt
  have hscaleR :
      1 / Spec.attentionScaleDenom (α := R) (Nat.succ d) = oneR / sqrtR := by
    unfold Spec.attentionScaleDenom
    simp only [ite_eq_right (Nat.succ_ne_zero d)]
    change 1 / Numerics.MathFunctions.sqrt dR = oneR / sqrtR
    rfl
  have hscaleS :
      1 / Spec.attentionScaleDenom (α := ℝ) (Nat.succ d) = (1 : ℝ) / Real.sqrt dS := by
    unfold Spec.attentionScaleDenom
    simp only [ite_eq_right (Nat.succ_ne_zero d)]
    change 1 / Numerics.MathFunctions.sqrt dS = (1 : ℝ) / Real.sqrt dS
    rfl
  rw [hscaleR, hscaleS]
  simpa [dR, sqrtR, oneR, epsOne, epsSqrt, canonicalScaleErrorBound,
    dimensionSqrtError] using hdiv

/-- Error after the rounded score product `Q Kᵀ`. -/
def scoreErrorBound {nQ nK d : Nat} (epsQ epsK : ℝ)
    (qR : Tensor R [nQ, d])
    (kR : Tensor R [nK, d]) : ℝ :=
  linfNorm (NFBackend.matMulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
    (m := nQ) (n := d) (p := nK) epsQ epsK qR
    (Tensor.swapAdjacentAxes kR 0))

/-- Error after multiplying scores by an approximate runtime scale. -/
def scaledScoreErrorBound {nQ nK d : Nat} (epsQ epsK epsScale : ℝ)
    (scaleR : R)
    (qR : Tensor R [nQ, d])
    (kR : Tensor R [nK, d]) : ℝ :=
  let scoresR := matMulSpec qR (Tensor.swapAdjacentAxes kR 0)
  linfNorm (NFBackend.scaleApproxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
    (scoreErrorBound (β := β) (fexp := fexp) (rnd := rnd) epsQ epsK qR kR)
    epsScale scaleR scoresR)

/-- Shared numerical certificate for the scaled score matrix `scale * (Q Kᵀ)`.

Both masked and unmasked attention use this prefix. Keeping it as one theorem also identifies the
natural contract boundary for fused score kernels: a provider may replace the implementation as
long as it supplies this same approximation statement.
-/
theorem approxTensor_scaledAttentionScores {nQ nK d : Nat}
    {qS : SpecTensor [nQ, d]}
    {kS : SpecTensor [nK, d]}
    {qR : Tensor R [nQ, d]}
    {kR : Tensor R [nK, d]}
    {scaleS : ℝ} {scaleR : R} {epsQ epsK epsScale : ℝ}
    (hq : approxTensor (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd)) qS qR epsQ)
    (hk : approxTensor (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd)) kS kR epsK)
    (hscale : abs (NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd) scaleR - scaleS) ≤
      epsScale) :
    approxTensor (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (scaleSpec (matMulSpec qS (Tensor.swapAdjacentAxes kS 0)) scaleS)
      (scaleSpec (matMulSpec qR (Tensor.swapAdjacentAxes kR 0)) scaleR)
      (scaledScoreErrorBound (β := β) (fexp := fexp) (rnd := rnd)
        epsQ epsK epsScale scaleR qR kR) := by
  have hkT := NFBackend.approxTensor_swapAdjacentAxes (depth := 0)
    (β := β) (fexp := fexp) (rnd := rnd) hk
  have hscores := NFBackend.approxTensor_mat_mul_spec
    (β := β) (fexp := fexp) (rnd := rnd) hq hkT
  have hscaled := NFBackend.approxTensor_scale_spec_of_approx
    (β := β) (fexp := fexp) (rnd := rnd) scaleS scaleR hscores hscale
  simpa [scaledScoreErrorBound, scoreErrorBound] using hscaled

/-- Error in the row-wise attention weights. -/
def weightErrorBound {nQ nK d : Nat} (epsQ epsK epsScale : ℝ)
    (scaleR : R)
    (qR : Tensor R [nQ, d])
    (kR : Tensor R [Nat.succ nK, d]) : ℝ :=
  let scoresR := matMulSpec qR (Tensor.swapAdjacentAxes kR 0)
  let scaledR := scaleSpec scoresR scaleR
  AxisSoftmax.softmaxRowsErrorBound (β := β) (fexp := fexp) (rnd := rnd)
    (scaledScoreErrorBound (β := β) (fexp := fexp) (rnd := rnd)
      epsQ epsK epsScale scaleR qR kR) scaledR

/-- End-to-end output error for unmasked scaled dot-product attention. -/
def outputErrorBound {nQ nK d : Nat} (epsQ epsK epsV epsScale : ℝ)
    (scaleR : R)
    (qR : Tensor R [nQ, d])
    (kR vR : Tensor R [Nat.succ nK, d]) : ℝ :=
  let scoresR := matMulSpec qR (Tensor.swapAdjacentAxes kR 0)
  let scaledR := scaleSpec scoresR scaleR
  let weightsR := Activation.softmaxSpec 1 scaledR
  linfNorm (NFBackend.matMulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
    (m := nQ) (n := Nat.succ nK) (p := d)
    (weightErrorBound (β := β) (fexp := fexp) (rnd := rnd)
      epsQ epsK epsScale scaleR qR kR)
    epsV weightsR vR)

/-- Error in hard-masked attention weights, preserving one certificate record per query row. -/
def maskedWeightErrorBound {nQ nK d : Nat}
    (η epsMax : Fin nQ → ℝ) (rowMaxR : Fin nQ → R)
    (epsQ epsK epsScale : ℝ) (scaleR : R)
    (qR : Tensor R [nQ, d])
    (kR : Tensor R [Nat.succ nK, d])
    (mask : Tensor Bool [nQ, Nat.succ nK]) : ℝ :=
  let scaledR := scaleSpec (matMulSpec qR (Tensor.swapAdjacentAxes kR 0)) scaleR
  linfNorm
    (AxisSoftmax.hardMaskedRowsBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
      η
      (scaledScoreErrorBound (β := β) (fexp := fexp) (rnd := rnd)
        epsQ epsK epsScale scaleR qR kR)
      scaledR mask rowMaxR epsMax)

/-- End-to-end output error for hard-masked scaled dot-product attention. -/
def maskedOutputErrorBound {nQ nK d : Nat}
    (η epsMax : Fin nQ → ℝ) (rowMaxR : Fin nQ → R)
    (epsQ epsK epsV epsScale : ℝ) (scaleR : R)
    (qR : Tensor R [nQ, d])
    (kR vR : Tensor R [Nat.succ nK, d])
    (mask : Tensor Bool [nQ, Nat.succ nK]) : ℝ :=
  let scaledR := scaleSpec (matMulSpec qR (Tensor.swapAdjacentAxes kR 0)) scaleR
  let weightsR := Spec.hardMaskedSoftmaxSpec scaledR mask
  linfNorm
    (NFBackend.matMulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
      (m := nQ) (n := Nat.succ nK) (p := d)
      (maskedWeightErrorBound (β := β) (fexp := fexp) (rnd := rnd)
        η epsMax rowMaxR epsQ epsK epsScale scaleR qR kR mask)
      epsV weightsR vR)

/-- Numerical forward theorem for hard-masked scaled dot-product attention.

Every query row must have an allowed key. This covers causal masks, where the diagonal is allowed;
the all-blocked vector theorem remains available for APIs that intentionally permit empty rows.
The selected allowed-row maxima and denominator margins are explicit certificate data, so no
finite value is ever substituted for negative infinity.
-/
theorem approxTensor_hardMaskedScaledDotProductAttentionCore {nQ nK d : Nat}
    {qS : SpecTensor [nQ, d]}
    {kS vS : SpecTensor [Nat.succ nK, d]}
    {qR : Tensor R [nQ, d]}
    {kR vR : Tensor R [Nat.succ nK, d]}
    (mask : Tensor Bool [nQ, Nat.succ nK])
    {scaleS : ℝ} {scaleR : R} {epsQ epsK epsV epsScale : ℝ}
    (hq : approxTensor (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd)) qS qR epsQ)
    (hk : approxTensor (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd)) kS kR epsK)
    (hv : approxTensor (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd)) vS vR epsV)
    (hscale : abs (NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd) scaleR - scaleS) ≤
      epsScale)
    (evidence : AxisSoftmax.HardMaskedRowsEvidence (β := β) (fexp := fexp) (rnd := rnd)
      (scaleSpec (matMulSpec qS (Tensor.swapAdjacentAxes kS 0)) scaleS)
      (scaleSpec (matMulSpec qR (Tensor.swapAdjacentAxes kR 0)) scaleR)
      mask
      (scaledScoreErrorBound (β := β) (fexp := fexp) (rnd := rnd)
        epsQ epsK epsScale scaleR qR kR)) :
    approxTensor (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (matMulSpec
        (Spec.hardMaskedSoftmaxSpec
          (scaleSpec (matMulSpec qS (Tensor.swapAdjacentAxes kS 0)) scaleS) mask) vS)
      (matMulSpec
        (Spec.hardMaskedSoftmaxSpec
          (scaleSpec (matMulSpec qR (Tensor.swapAdjacentAxes kR 0)) scaleR) mask) vR)
      (maskedOutputErrorBound (β := β) (fexp := fexp) (rnd := rnd)
        evidence.eta evidence.epsMax evidence.rowMaxR
        epsQ epsK epsV epsScale scaleR qR kR vR mask) := by
  let scaledS := scaleSpec (matMulSpec qS (Tensor.swapAdjacentAxes kS 0)) scaleS
  let scaledR := scaleSpec (matMulSpec qR (Tensor.swapAdjacentAxes kR 0)) scaleR
  let epsScaled := scaledScoreErrorBound (β := β) (fexp := fexp) (rnd := rnd)
    epsQ epsK epsScale scaleR qR kR
  let weightsS := Spec.hardMaskedSoftmaxSpec scaledS mask
  let weightsR := Spec.hardMaskedSoftmaxSpec scaledR mask
  let epsWeights := maskedWeightErrorBound (β := β) (fexp := fexp) (rnd := rnd)
    evidence.eta evidence.epsMax evidence.rowMaxR
    epsQ epsK epsScale scaleR qR kR mask
  have hscaled : approxTensor (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))
      scaledS scaledR epsScaled := by
    simpa [scaledS, scaledR, epsScaled] using
      (approxTensor_scaledAttentionScores (β := β) (fexp := fexp) (rnd := rnd)
        hq hk hscale)
  have hweights : approxTensor (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))
      weightsS weightsR epsWeights := by
    have h := AxisSoftmax.approxTensor_hardMaskedSoftmaxRowsSpec_of_max
      (β := β) (fexp := fexp) (rnd := rnd)
      mask hscaled (by simpa [scaledS, scaledR, epsScaled] using evidence)
    simpa [weightsS, weightsR, epsWeights, maskedWeightErrorBound,
      scaledR, epsScaled] using h
  have hout := NFBackend.approxTensor_mat_mul_spec
    (β := β) (fexp := fexp) (rnd := rnd) hweights hv
  simpa [scaledS, scaledR, weightsS, weightsR, epsScaled, epsWeights,
    maskedOutputErrorBound] using hout

/-- Numerical forward theorem for the unmasked scaled-dot-product attention core.

`hdenom` is checked on each rounded score row after matrix multiplication and scaling. This is the
only nonlocal side condition introduced by stable softmax; it certifies that the accumulated
denominator error remains below the exact lower bound one.
-/
theorem approxTensor_scaledDotProductAttentionCore {nQ nK d : Nat}
    {qS : SpecTensor [nQ, d]}
    {kS vS : SpecTensor [Nat.succ nK, d]}
    {qR : Tensor R [nQ, d]}
    {kR vR : Tensor R [Nat.succ nK, d]}
    {scaleS : ℝ} {scaleR : R} {epsQ epsK epsV epsScale : ℝ}
    (hq : approxTensor (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd)) qS qR epsQ)
    (hk : approxTensor (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd)) kS kR epsK)
    (hv : approxTensor (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd)) vS vR epsV)
    (hscale : abs (NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd) scaleR - scaleS) ≤
      epsScale)
    (hdenom : ∀ i : Fin nQ,
      AxisSoftmax.denominatorErrorBound (β := β) (fexp := fexp) (rnd := rnd)
        (scaledScoreErrorBound (β := β) (fexp := fexp) (rnd := rnd)
          epsQ epsK epsScale scaleR qR kR)
        (Spec.get
          (scaleSpec (matMulSpec qR (Tensor.swapAdjacentAxes kR 0)) scaleR) i) < 1) :
    approxTensor (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (matMulSpec
        (Activation.softmaxSpec 1
          (scaleSpec (matMulSpec qS (Tensor.swapAdjacentAxes kS 0)) scaleS)) vS)
      (matMulSpec
        (Activation.softmaxSpec 1
          (scaleSpec (matMulSpec qR (Tensor.swapAdjacentAxes kR 0)) scaleR)) vR)
      (outputErrorBound (β := β) (fexp := fexp) (rnd := rnd)
        epsQ epsK epsV epsScale scaleR qR kR vR) := by
  let kTS := Tensor.swapAdjacentAxes kS 0
  let kTR := Tensor.swapAdjacentAxes kR 0
  let scoresS := matMulSpec qS kTS
  let scoresR := matMulSpec qR kTR
  let epsScores := scoreErrorBound (β := β) (fexp := fexp) (rnd := rnd) epsQ epsK qR kR
  let scaledS := scaleSpec scoresS scaleS
  let scaledR := scaleSpec scoresR scaleR
  let epsScaled := scaledScoreErrorBound (β := β) (fexp := fexp) (rnd := rnd)
    epsQ epsK epsScale scaleR qR kR
  let weightsS := Activation.softmaxSpec 1 scaledS
  let weightsR := Activation.softmaxSpec 1 scaledR
  let epsWeights := weightErrorBound (β := β) (fexp := fexp) (rnd := rnd)
    epsQ epsK epsScale scaleR qR kR

  have hscaled : approxTensor (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))
      scaledS scaledR epsScaled := by
    simpa [scaledS, scaledR, scoresS, scoresR, kTS, kTR, epsScaled] using
      (approxTensor_scaledAttentionScores (β := β) (fexp := fexp) (rnd := rnd)
        hq hk hscale)
  have hweights : approxTensor (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))
      weightsS weightsR epsWeights := by
    have h := AxisSoftmax.approxTensor_softmaxRowsSpec (β := β) (fexp := fexp) (rnd := rnd)
      hscaled (by simpa [scaledR, epsScaled] using hdenom)
    simpa [weightsS, weightsR, epsWeights, weightErrorBound, scaledR, epsScaled] using h
  have hout := NFBackend.approxTensor_mat_mul_spec (β := β) (fexp := fexp) (rnd := rnd) hweights hv
  simpa [kTS, kTR, scoresS, scoresR, scaledS, scaledR, weightsS, weightsR,
    epsScores, epsScaled, epsWeights, outputErrorBound] using hout

/-- Canonical TorchLean attention corollary for the unmasked branch.

This theorem is stated directly over `Spec.AttentionContext`, so model proofs do not need to
reconstruct the core expression by hand. The scale approximation and row-denominator checks are
the numerical evidence attached to the runtime context.
-/
theorem approxTensor_scaledDotProductAttention_unmasked {nQ nK d : Nat}
    {hQ : nQ ≠ 0} {hK : Nat.succ nK ≠ 0}
    (ctxS : Spec.AttentionContext ℝ nQ (Nat.succ nK) d hQ hK)
    (ctxR : Spec.AttentionContext R nQ (Nat.succ nK) d hQ hK)
    {epsQ epsK epsV epsScale : ℝ}
    (hmaskS : ctxS.mask = none) (hmaskR : ctxR.mask = none)
    (hq : approxTensor (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))
      ctxS.Q ctxR.Q epsQ)
    (hk : approxTensor (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))
      ctxS.K ctxR.K epsK)
    (hv : approxTensor (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))
      ctxS.V ctxR.V epsV)
    (hscale :
      abs
        (NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd)
            (1 / Spec.attentionScaleDenom (α := R) d) -
          (1 / Spec.attentionScaleDenom (α := ℝ) d)) ≤ epsScale)
    (hdenom : ∀ i : Fin nQ,
      AxisSoftmax.denominatorErrorBound (β := β) (fexp := fexp) (rnd := rnd)
          (scaledScoreErrorBound (β := β) (fexp := fexp) (rnd := rnd)
            epsQ epsK epsScale (1 / Spec.attentionScaleDenom (α := R) d) ctxR.Q ctxR.K)
          (Spec.get
            (scaleSpec
              (matMulSpec ctxR.Q (Tensor.swapAdjacentAxes ctxR.K 0))
              (1 / Spec.attentionScaleDenom (α := R) d)) i) < 1) :
    approxTensor (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (Spec.scaledDotProductAttention ctxS) (Spec.scaledDotProductAttention ctxR)
      (outputErrorBound (β := β) (fexp := fexp) (rnd := rnd)
        epsQ epsK epsV epsScale (1 / Spec.attentionScaleDenom (α := R) d)
        ctxR.Q ctxR.K ctxR.V) := by
  have hcore := approxTensor_scaledDotProductAttentionCore
    (β := β) (fexp := fexp) (rnd := rnd)
    (qS := ctxS.Q) (kS := ctxS.K) (vS := ctxS.V)
    (qR := ctxR.Q) (kR := ctxR.K) (vR := ctxR.V)
    (scaleS := 1 / Spec.attentionScaleDenom (α := ℝ) d)
    (scaleR := 1 / Spec.attentionScaleDenom (α := R) d)
    hq hk hv hscale hdenom
  simpa only [Spec.scaledDotProductAttention, hmaskS, hmaskR] using hcore

/-- Canonical TorchLean attention corollary for a shared hard mask.

The row certificate is tied to the scaled score matrices in the two contexts. Thus a certificate
cannot be replayed against a different mask, scale, or set of parameters merely because the tensor
shapes happen to agree.
-/
theorem approxTensor_scaledDotProductAttention_masked {nQ nK d : Nat}
    {hQ : nQ ≠ 0} {hK : Nat.succ nK ≠ 0}
    (ctxS : Spec.AttentionContext ℝ nQ (Nat.succ nK) d hQ hK)
    (ctxR : Spec.AttentionContext R nQ (Nat.succ nK) d hQ hK)
    (mask : Tensor Bool [nQ, Nat.succ nK])
    {epsQ epsK epsV epsScale : ℝ}
    (hmaskS : ctxS.mask = some mask) (hmaskR : ctxR.mask = some mask)
    (hq : approxTensor (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))
      ctxS.Q ctxR.Q epsQ)
    (hk : approxTensor (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))
      ctxS.K ctxR.K epsK)
    (hv : approxTensor (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))
      ctxS.V ctxR.V epsV)
    (hscale :
      abs
        (NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd)
            (1 / Spec.attentionScaleDenom (α := R) d) -
          (1 / Spec.attentionScaleDenom (α := ℝ) d)) ≤ epsScale)
    (evidence : AxisSoftmax.HardMaskedRowsEvidence (β := β) (fexp := fexp) (rnd := rnd)
      (scaleSpec (matMulSpec ctxS.Q (Tensor.swapAdjacentAxes ctxS.K 0))
        (1 / Spec.attentionScaleDenom (α := ℝ) d))
      (scaleSpec (matMulSpec ctxR.Q (Tensor.swapAdjacentAxes ctxR.K 0))
        (1 / Spec.attentionScaleDenom (α := R) d))
      mask
      (scaledScoreErrorBound (β := β) (fexp := fexp) (rnd := rnd)
        epsQ epsK epsScale (1 / Spec.attentionScaleDenom (α := R) d) ctxR.Q ctxR.K)) :
    approxTensor (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (Spec.scaledDotProductAttention ctxS) (Spec.scaledDotProductAttention ctxR)
      (maskedOutputErrorBound (β := β) (fexp := fexp) (rnd := rnd)
        evidence.eta evidence.epsMax evidence.rowMaxR
        epsQ epsK epsV epsScale (1 / Spec.attentionScaleDenom (α := R) d)
        ctxR.Q ctxR.K ctxR.V mask) := by
  have hcore := approxTensor_hardMaskedScaledDotProductAttentionCore
    (β := β) (fexp := fexp) (rnd := rnd)
    (qS := ctxS.Q) (kS := ctxS.K) (vS := ctxS.V)
    (qR := ctxR.Q) (kR := ctxR.K) (vR := ctxR.V)
    mask
    (scaleS := 1 / Spec.attentionScaleDenom (α := ℝ) d)
    (scaleR := 1 / Spec.attentionScaleDenom (α := R) d)
    hq hk hv hscale evidence
  simpa only [Spec.scaledDotProductAttention, hmaskS, hmaskR] using hcore

/-- Fully instantiated unmasked attention theorem for a positive feature dimension.

This corollary discharges the scale-coefficient approximation with
`approx_canonicalAttentionScale`; callers provide only tensor approximation hypotheses and the two
checkable safety margins for square root and row normalization.
-/
theorem approxTensor_scaledDotProductAttention_unmasked_canonical {nQ nK d : Nat}
    {hQ : nQ ≠ 0} {hK : Nat.succ nK ≠ 0}
    (ctxS : Spec.AttentionContext ℝ nQ (Nat.succ nK) (Nat.succ d) hQ hK)
    (ctxR : Spec.AttentionContext R nQ (Nat.succ nK) (Nat.succ d) hQ hK)
    {epsQ epsK epsV : ℝ}
    (hmaskS : ctxS.mask = none) (hmaskR : ctxR.mask = none)
    (hq : approxTensor (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))
      ctxS.Q ctxR.Q epsQ)
    (hk : approxTensor (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))
      ctxS.K ctxR.K epsK)
    (hv : approxTensor (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))
      ctxS.V ctxR.V epsV)
    (hscaleMargin : dimensionSqrtError (β := β) (fexp := fexp) (rnd := rnd) d < 1)
    (hdenom : ∀ i : Fin nQ,
      AxisSoftmax.denominatorErrorBound (β := β) (fexp := fexp) (rnd := rnd)
          (scaledScoreErrorBound (β := β) (fexp := fexp) (rnd := rnd)
            epsQ epsK (canonicalScaleErrorBound (β := β) (fexp := fexp) (rnd := rnd) d)
            (1 / Spec.attentionScaleDenom (α := R) (Nat.succ d)) ctxR.Q ctxR.K)
          (Spec.get
            (scaleSpec
              (matMulSpec ctxR.Q (Tensor.swapAdjacentAxes ctxR.K 0))
              (1 / Spec.attentionScaleDenom (α := R) (Nat.succ d))) i) < 1) :
    approxTensor (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (Spec.scaledDotProductAttention ctxS) (Spec.scaledDotProductAttention ctxR)
      (outputErrorBound (β := β) (fexp := fexp) (rnd := rnd)
        epsQ epsK epsV
        (canonicalScaleErrorBound (β := β) (fexp := fexp) (rnd := rnd) d)
        (1 / Spec.attentionScaleDenom (α := R) (Nat.succ d))
        ctxR.Q ctxR.K ctxR.V) := by
  exact approxTensor_scaledDotProductAttention_unmasked
    (β := β) (fexp := fexp) (rnd := rnd)
    ctxS ctxR hmaskS hmaskR hq hk hv
    (approx_canonicalAttentionScale (β := β) (fexp := fexp) (rnd := rnd) d hscaleMargin)
    hdenom

end

end Attention
end RuntimeApprox
end Proofs
