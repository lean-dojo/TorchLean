/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.RuntimeApprox.NF.Linalg
public import TorchLean.Verify.RuntimeApprox.NF.Softmax
import FloatLib.Floats.Formats.Flocq.Theory.Rounding.Order

/-!
# Rounded scaled dot-product attention

Forward error bounds for attention in the rounded-real model `NF`, built from the bounds for
transpose, sequential matrix multiplication, coefficient-aware scaling and the stable last-axis
softmax. The unmasked theorem follows

`softmax(c * (Q Kᵀ)) V`,

where `c` is normally `1 / sqrt(d)`. The matrices and `c` may all be approximate, and the result
is a single infinity-norm budget. Queries, keys and values share one feature width. A native or
fused attention kernel is outside these statements.

`NF` keeps the ordinary rounded-real expression. Finite IEEE carriers can also replay scaled
contractions and pullbacks from their original coefficients after range loss or absorption, and
that separate finite-codec execution path is not covered here.

The rounded coefficient error is a hypothesis of the theorems. A caller can obtain it from the
concrete construction of `1 / sqrt(d)` (`approx_canonicalScale`) or from another proved
coefficient bound, and backend metadata alone does not provide it. Hiding the error would treat a
rounded scale as an exact constant.

References:

* A. Vaswani et al., "Attention Is All You Need," NeurIPS 2017.
* N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM, 2002.
* PyTorch, `torch.nn.functional.scaled_dot_product_attention`, for the runtime operator and
  last-axis normalization convention.
-/

@[expose] public section

namespace TorchLean

open TorchLean.Tensor
open Robustness
open FloatLib FloatLib.Numerics FloatLib.Floats.Formats
open Flocq

noncomputable section

variable {β : Radix} {fexp : ℤ -> ℤ} [ValidExp fexp]
variable {rnd : ℝ -> ℤ} [ValidRndToNearest rnd]

local notation "R" => NF β fexp rnd

/-! ## The canonical `1 / sqrt(d)` coefficient -/

/-- Half-ULP budget for embedding feature dimension `d + 1` into `NF`. -/
def NF.Attention.dimensionCastError (d : Nat) : ℝ :=
  ulp β fexp (Nat.succ d : ℝ) / 2

/-- Error budget for the rounded square root of the embedded feature dimension. -/
def NF.Attention.dimensionSqrtError (d : Nat) : ℝ :=
  let dR : R := (Nat.succ d : Nat)
  dimensionCastError (β := β) (fexp := fexp) d / Real.sqrt 1 +
    ulp β fexp (Real.sqrt (NF.toReal dR)) / 2

/-- Error budget for constructing `1 / sqrt(d + 1)` in the rounded-real model. -/
def NF.Attention.canonicalScaleErrorBound (d : Nat) : ℝ :=
  let oneR : R := 1
  let dR : R := (Nat.succ d : Nat)
  let sqrtR : R := Numerics.MathFunctions.sqrt dR
  divPosErrorBound (β := β) (fexp := fexp) 1
    (ulp β fexp 1 / 2)
    (dimensionSqrtError (β := β) (fexp := fexp) (rnd := rnd) d)
    (NF.toReal oneR)
    (NF.toReal sqrtR)

/-- The canonical rounded attention scale approximates the exact real `1 / sqrt(d + 1)`.

The hypothesis `hbudget` asks the accumulated square-root error to be below one, the lower bound
on `sqrt(d + 1)` that the division bound uses. It concerns a noncomputable real bound, so the
caller proves it for the format at hand.
-/
theorem NF.Attention.approx_canonicalScale (d : Nat)
    (hbudget : dimensionSqrtError (β := β) (fexp := fexp) (rnd := rnd) d < 1) :
    _root_.abs
        (NF.toReal
            (1 / Attention.scaleDenom (α := R) (Nat.succ d)) -
          (1 / Attention.scaleDenom (α := ℝ) (Nat.succ d))) ≤
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
      abs (NF.toReal dR - dS) ≤ epsD := by
    simpa [dR, dS, epsD, dimensionCastError, NF.instNatCast] using
      (approx_ofReal (β := β) (fexp := fexp) (rnd := rnd) dS)
  have hdR0 : 0 ≤ NF.toReal dR := by
    dsimp [dR, NF.toReal, NF.instNatCast,
      NF.ofReal, NF.roundR]
    exact round_nonneg rnd (by positivity)
  have hsqrt :
      abs (NF.toReal sqrtR - Real.sqrt dS) ≤
        epsSqrt := by
    have h := approx_sqrt_of_pos_lb (β := β) (fexp := fexp) (rnd := rnd)
      (x := dS) (xR := dR) (eps := epsD) (η := (1 : ℝ))
      (by norm_num) hdLower hdR0 hcast
    simpa [sqrtR, epsSqrt, dimensionSqrtError, dR, epsD] using h
  have hsqrtLower : (1 : ℝ) ≤ Real.sqrt dS := by
    exact Real.one_le_sqrt.mpr hdLower
  have hone :
      abs (NF.toReal oneR - (1 : ℝ)) ≤ epsOne := by
    have honeEq : oneR =
        NF.ofReal (β := β) (fexp := fexp) (rnd := rnd) (1 : ℝ) := by
      rfl
    rw [honeEq]
    simpa [epsOne] using
      (approx_ofReal (β := β) (fexp := fexp) (rnd := rnd) (1 : ℝ))
  have hdiv := approx_div_of_pos_lb
    (β := β) (fexp := fexp) (rnd := rnd)
    (x := (1 : ℝ)) (y := Real.sqrt dS) (xR := oneR) (yR := sqrtR)
    (epsx := epsOne) (epsy := epsSqrt) (η := (1 : ℝ))
    hsqrtLower (by simpa [epsSqrt] using hbudget) hone hsqrt
  have hscaleR :
      1 / Attention.scaleDenom (α := R) (Nat.succ d) = oneR / sqrtR := by
    unfold Attention.scaleDenom
    simp only [ite_eq_right (Nat.succ_ne_zero d)]
    change 1 / Numerics.MathFunctions.sqrt dR = oneR / sqrtR
    rfl
  have hscaleS :
      1 / Attention.scaleDenom (α := ℝ) (Nat.succ d) = (1 : ℝ) / Real.sqrt dS := by
    unfold Attention.scaleDenom
    simp only [ite_eq_right (Nat.succ_ne_zero d)]
    change 1 / Numerics.MathFunctions.sqrt dS = (1 : ℝ) / Real.sqrt dS
    rfl
  rw [hscaleR, hscaleS]
  simpa [dR, sqrtR, oneR, epsOne, epsSqrt, canonicalScaleErrorBound,
    dimensionSqrtError] using hdiv

/-- Error after the rounded score product `Q Kᵀ`. -/
def NF.Attention.scoreErrorBound {nQ nK d : Nat} (epsQ epsK : ℝ)
    (qR : Tensor R [nQ, d])
    (kR : Tensor R [nK, d]) : ℝ :=
  linfNorm (matMulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
    (m := nQ) (n := d) (p := nK) epsQ epsK qR
    (Tensor.permute kR (Shape.adjacentSwap [nK, d] 0) [d, nK]))

/-- Error after multiplying scores by an approximate runtime scale. -/
def NF.Attention.scaledScoreErrorBound {nQ nK d : Nat} (epsQ epsK epsScale : ℝ)
    (scaleR : R)
    (qR : Tensor R [nQ, d])
    (kR : Tensor R [nK, d]) : ℝ :=
  let scoresR := Tensor.matmul qR (Tensor.permute kR (Shape.adjacentSwap [nK, d] 0) [d, nK])
  linfNorm (scaleApproxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
    (scoreErrorBound (β := β) (fexp := fexp) (rnd := rnd) epsQ epsK qR kR)
    epsScale scaleR scoresR)

/-- The rounded scaled score matrix `scale * (Q Kᵀ)` is within `NF.Attention.scaledScoreErrorBound`
of the exact one, given bounds on `Q`, `K` and the scale.

Masked and unmasked attention share this prefix. It is also the contract boundary for fused score
kernels: a provider can replace the implementation as long as it proves the same approximation
statement.
-/
theorem ApproxTensor.attentionScores {nQ nK d : Nat}
    {qS : Tensor ℝ [nQ, d]}
    {kS : Tensor ℝ [nK, d]}
    {qR : Tensor R [nQ, d]}
    {kR : Tensor R [nK, d]}
    {scaleS : ℝ} {scaleR : R} {epsQ epsK epsScale : ℝ}
    (hq : ApproxTensor (α := R)
      (toSpec := NF.toReal) qS qR epsQ)
    (hk : ApproxTensor (α := R)
      (toSpec := NF.toReal) kS kR epsK)
    (hscale : _root_.abs (NF.toReal scaleR - scaleS) ≤
      epsScale) :
    ApproxTensor (α := R)
      (toSpec := NF.toReal)
      (Tensor.scale (Tensor.matmul qS (Tensor.permute kS
          (Shape.adjacentSwap [nK, d] 0) [d, nK])) scaleS)
      (Tensor.scale (Tensor.matmul qR (Tensor.permute kR
          (Shape.adjacentSwap [nK, d] 0) [d, nK])) scaleR)
      (NF.Attention.scaledScoreErrorBound (β := β) (fexp := fexp) (rnd := rnd)
        epsQ epsK epsScale scaleR qR kR) := by
  have hkT := permute (σ := Shape.adjacentSwap [nK, d] 0) (out := [d, nK])
    (hout := by tensor_shape) (β := β) (fexp := fexp) (rnd := rnd) hk
  have hscores := matmul
    (β := β) (fexp := fexp) (rnd := rnd) hq hkT
  have hscaled := scale_of_approx
    (β := β) (fexp := fexp) (rnd := rnd) scaleS scaleR hscores hscale
  simpa [NF.Attention.scaledScoreErrorBound, NF.Attention.scoreErrorBound] using hscaled

/-- Error in the row-wise attention weights. -/
def NF.Attention.weightErrorBound {nQ nK d : Nat} (epsQ epsK epsScale : ℝ)
    (scaleR : R)
    (qR : Tensor R [nQ, d])
    (kR : Tensor R [Nat.succ nK, d]) : ℝ :=
  let scoresR := Tensor.matmul qR (Tensor.permute kR
      (Shape.adjacentSwap [nK.succ, d] 0) [d, nK.succ])
  let scaledR := Tensor.scale scoresR scaleR
  Softmax.rowsErrorBound (β := β) (fexp := fexp) (rnd := rnd)
    (scaledScoreErrorBound (β := β) (fexp := fexp) (rnd := rnd)
      epsQ epsK epsScale scaleR qR kR) scaledR

/-- End-to-end output error for unmasked scaled dot-product attention. -/
def NF.Attention.outputErrorBound {nQ nK d : Nat} (epsQ epsK epsV epsScale : ℝ)
    (scaleR : R)
    (qR : Tensor R [nQ, d])
    (kR vR : Tensor R [Nat.succ nK, d]) : ℝ :=
  let scoresR := Tensor.matmul qR (Tensor.permute kR
      (Shape.adjacentSwap [nK.succ, d] 0) [d, nK.succ])
  let scaledR := Tensor.scale scoresR scaleR
  let weightsR := Tensor.softmax scaledR ⟨1, Shape.AxisInBounds.proof⟩
  linfNorm (matMulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
    (m := nQ) (n := Nat.succ nK) (p := d)
    (weightErrorBound (β := β) (fexp := fexp) (rnd := rnd)
      epsQ epsK epsScale scaleR qR kR)
    epsV weightsR vR)

/-- Error in hard-masked attention weights, preserving one certificate record per query row. -/
def NF.Attention.maskedWeightErrorBound {nQ nK d : Nat}
    (η epsMax : Fin nQ → ℝ) (rowMaxR : Fin nQ → R)
    (epsQ epsK epsScale : ℝ) (scaleR : R)
    (qR : Tensor R [nQ, d])
    (kR : Tensor R [Nat.succ nK, d])
    (mask : Tensor Bool [nQ, Nat.succ nK]) : ℝ :=
  let scaledR := Tensor.scale (Tensor.matmul qR (Tensor.permute kR
      (Shape.adjacentSwap [nK.succ, d] 0) [d, nK.succ])) scaleR
  linfNorm
    (Softmax.hardMaskedRowsBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
      η
      (scaledScoreErrorBound (β := β) (fexp := fexp) (rnd := rnd)
        epsQ epsK epsScale scaleR qR kR)
      scaledR mask rowMaxR epsMax)

/-- End-to-end output error for hard-masked scaled dot-product attention. -/
def NF.Attention.maskedOutputErrorBound {nQ nK d : Nat}
    (η epsMax : Fin nQ → ℝ) (rowMaxR : Fin nQ → R)
    (epsQ epsK epsV epsScale : ℝ) (scaleR : R)
    (qR : Tensor R [nQ, d])
    (kR vR : Tensor R [Nat.succ nK, d])
    (mask : Tensor Bool [nQ, Nat.succ nK]) : ℝ :=
  let scaledR := Tensor.scale (Tensor.matmul qR (Tensor.permute kR
      (Shape.adjacentSwap [nK.succ, d] 0) [d, nK.succ])) scaleR
  let weightsR := Softmax.hardMasked scaledR mask
  linfNorm
    (matMulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
      (m := nQ) (n := Nat.succ nK) (p := d)
      (maskedWeightErrorBound (β := β) (fexp := fexp) (rnd := rnd)
        η epsMax rowMaxR epsQ epsK epsScale scaleR qR kR mask)
      epsV weightsR vR)

/-- Numerical forward theorem for hard-masked scaled dot-product attention.

Every query row needs an allowed key, which covers causal masks because the diagonal is allowed.
For APIs that permit empty rows, the vector theorem
`ApproxTensor.hardMaskedSoftmaxVec_allBlocked` handles a fully blocked row. The
allowed-row maxima and denominator margins are certificate data in `evidence`, so no finite value
ever stands in for negative infinity.
-/
theorem ApproxTensor.hardMaskedAttention {nQ nK d : Nat}
    {qS : Tensor ℝ [nQ, d]}
    {kS vS : Tensor ℝ [Nat.succ nK, d]}
    {qR : Tensor R [nQ, d]}
    {kR vR : Tensor R [Nat.succ nK, d]}
    (mask : Tensor Bool [nQ, Nat.succ nK])
    {scaleS : ℝ} {scaleR : R} {epsQ epsK epsV epsScale : ℝ}
    (hq : ApproxTensor (α := R)
      (toSpec := NF.toReal) qS qR epsQ)
    (hk : ApproxTensor (α := R)
      (toSpec := NF.toReal) kS kR epsK)
    (hv : ApproxTensor (α := R)
      (toSpec := NF.toReal) vS vR epsV)
    (hscale : _root_.abs (NF.toReal scaleR - scaleS) ≤
      epsScale)
    (evidence : NF.Softmax.HardMaskedRowsEvidence (β := β) (fexp := fexp) (rnd := rnd)
      (Tensor.scale (Tensor.matmul qS (Tensor.permute kS
          (Shape.adjacentSwap [nK.succ, d] 0) [d, nK.succ])) scaleS)
      (Tensor.scale (Tensor.matmul qR (Tensor.permute kR
          (Shape.adjacentSwap [nK.succ, d] 0) [d, nK.succ])) scaleR)
      mask
      (NF.Attention.scaledScoreErrorBound (β := β) (fexp := fexp) (rnd := rnd)
        epsQ epsK epsScale scaleR qR kR)) :
    ApproxTensor (α := R)
      (toSpec := NF.toReal)
      (Tensor.matmul
        (Softmax.hardMasked
          (Tensor.scale (Tensor.matmul qS (Tensor.permute kS
              (Shape.adjacentSwap [nK.succ, d] 0) [d, nK.succ])) scaleS) mask) vS)
      (Tensor.matmul
        (Softmax.hardMasked
          (Tensor.scale (Tensor.matmul qR (Tensor.permute kR
              (Shape.adjacentSwap [nK.succ, d] 0) [d, nK.succ])) scaleR) mask) vR)
      (NF.Attention.maskedOutputErrorBound (β := β) (fexp := fexp) (rnd := rnd)
        evidence.eta evidence.epsMax evidence.rowMaxR
        epsQ epsK epsV epsScale scaleR qR kR vR mask) := by
  let scaledS := Tensor.scale (Tensor.matmul qS (Tensor.permute kS
      (Shape.adjacentSwap [nK.succ, d] 0) [d, nK.succ])) scaleS
  let scaledR := Tensor.scale (Tensor.matmul qR (Tensor.permute kR
      (Shape.adjacentSwap [nK.succ, d] 0) [d, nK.succ])) scaleR
  let epsScaled := NF.Attention.scaledScoreErrorBound (β := β) (fexp := fexp) (rnd := rnd)
    epsQ epsK epsScale scaleR qR kR
  let weightsS := Softmax.hardMasked scaledS mask
  let weightsR := Softmax.hardMasked scaledR mask
  let epsWeights := NF.Attention.maskedWeightErrorBound (β := β) (fexp := fexp) (rnd := rnd)
    evidence.eta evidence.epsMax evidence.rowMaxR
    epsQ epsK epsScale scaleR qR kR mask
  have hscaled : ApproxTensor (α := R)
      (toSpec := NF.toReal)
      scaledS scaledR epsScaled := by
    simpa [scaledS, scaledR, epsScaled] using
      (attentionScores (β := β) (fexp := fexp) (rnd := rnd)
        hq hk hscale)
  have hweights : ApproxTensor (α := R)
      (toSpec := NF.toReal)
      weightsS weightsR epsWeights := by
    have h := hardMaskedSoftmax_of_max
      (β := β) (fexp := fexp) (rnd := rnd)
      mask hscaled (by simpa [scaledS, scaledR, epsScaled] using evidence)
    simpa [weightsS, weightsR, epsWeights, NF.Attention.maskedWeightErrorBound,
      scaledR, epsScaled] using h
  have hout := matmul
    (β := β) (fexp := fexp) (rnd := rnd) hweights hv
  simpa [scaledS, scaledR, weightsS, weightsR, epsScaled, epsWeights,
    NF.Attention.maskedOutputErrorBound] using hout

/-- Numerical forward theorem for the unmasked scaled dot-product attention core.

`hdenom` bounds each rounded score row after matrix multiplication and scaling: its accumulated
denominator error must stay below one, the lower bound of the exact max-shifted denominator. The
caller proves this bound, and the theorem performs no executable check.
-/
theorem ApproxTensor.softmaxAttention {nQ nK d : Nat}
    {qS : Tensor ℝ [nQ, d]}
    {kS vS : Tensor ℝ [Nat.succ nK, d]}
    {qR : Tensor R [nQ, d]}
    {kR vR : Tensor R [Nat.succ nK, d]}
    {scaleS : ℝ} {scaleR : R} {epsQ epsK epsV epsScale : ℝ}
    (hq : ApproxTensor (α := R)
      (toSpec := NF.toReal) qS qR epsQ)
    (hk : ApproxTensor (α := R)
      (toSpec := NF.toReal) kS kR epsK)
    (hv : ApproxTensor (α := R)
      (toSpec := NF.toReal) vS vR epsV)
    (hscale : _root_.abs (NF.toReal scaleR - scaleS) ≤
      epsScale)
    (hdenom : ∀ i : Fin nQ,
      NF.Softmax.denominatorErrorBound (β := β) (fexp := fexp) (rnd := rnd)
        (NF.Attention.scaledScoreErrorBound (β := β) (fexp := fexp) (rnd := rnd)
          epsQ epsK epsScale scaleR qR kR)
        (Tensor.unstack
          (Tensor.scale (Tensor.matmul qR (Tensor.permute kR
              (Shape.adjacentSwap [nK.succ, d] 0) [d, nK.succ])) scaleR) i) < 1) :
    ApproxTensor (α := R)
      (toSpec := NF.toReal)
      (Tensor.matmul
        (Tensor.softmax (Tensor.scale (Tensor.matmul qS (Tensor.permute kS
            (Shape.adjacentSwap [nK.succ, d] 0) [d, nK.succ])) scaleS)
          ⟨1, Nat.one_lt_two⟩) vS)
      (Tensor.matmul
        (Tensor.softmax (Tensor.scale (Tensor.matmul qR (Tensor.permute kR
            (Shape.adjacentSwap [nK.succ, d] 0) [d, nK.succ])) scaleR)
          ⟨1, Nat.one_lt_two⟩) vR)
      (NF.Attention.outputErrorBound (β := β) (fexp := fexp) (rnd := rnd)
        epsQ epsK epsV epsScale scaleR qR kR vR) := by
  let kTS := Tensor.permute kS (Shape.adjacentSwap [nK.succ, d] 0) [d, nK.succ]
  let kTR := Tensor.permute kR (Shape.adjacentSwap [nK.succ, d] 0) [d, nK.succ]
  let scoresS := Tensor.matmul qS kTS
  let scoresR := Tensor.matmul qR kTR
  let scaledS := Tensor.scale scoresS scaleS
  let scaledR := Tensor.scale scoresR scaleR
  let epsScaled := NF.Attention.scaledScoreErrorBound (β := β) (fexp := fexp) (rnd := rnd)
    epsQ epsK epsScale scaleR qR kR
  let weightsS := Tensor.softmax scaledS ⟨1, Shape.AxisInBounds.proof⟩
  let weightsR := Tensor.softmax scaledR ⟨1, Shape.AxisInBounds.proof⟩
  let epsWeights := NF.Attention.weightErrorBound (β := β) (fexp := fexp) (rnd := rnd)
    epsQ epsK epsScale scaleR qR kR
  have hscaled : ApproxTensor (α := R)
      (toSpec := NF.toReal)
      scaledS scaledR epsScaled := by
    simpa [scaledS, scaledR, scoresS, scoresR, kTS, kTR, epsScaled] using
      (attentionScores (β := β) (fexp := fexp) (rnd := rnd)
        hq hk hscale)
  have hweights : ApproxTensor (α := R)
      (toSpec := NF.toReal)
      weightsS weightsR epsWeights := by
    have h := softmax_rows (β := β) (fexp := fexp) (rnd := rnd)
      hscaled (by simpa [scaledR, epsScaled] using hdenom)
    simpa [weightsS, weightsR, epsWeights, NF.Attention.weightErrorBound, scaledR,
      epsScaled] using h
  have hout := matmul (β := β) (fexp := fexp) (rnd := rnd) hweights hv
  simpa [kTS, kTR, scoresS, scoresR, scaledS, scaledR, weightsS, weightsR,
    epsScaled, epsWeights, NF.Attention.outputErrorBound] using hout

/-- Unmasked attention over `Attention.Inputs`: when neither context has a mask, the rounded
`Attention.scaledDotProduct` is within `NF.Attention.outputErrorBound` of the exact one.

Stating it over the context saves model proofs from rebuilding the core expression by hand. The
scale approximation `hscale` and the row-denominator bounds `hdenom` are the numerical evidence
attached to the runtime context.
-/
theorem ApproxTensor.scaledDotProduct_unmasked {nQ nK d : Nat}
    {hQ : nQ ≠ 0} {hK : Nat.succ nK ≠ 0}
    (ctxS : Attention.Inputs ℝ nQ (Nat.succ nK) d hQ hK)
    (ctxR : Attention.Inputs R nQ (Nat.succ nK) d hQ hK)
    {epsQ epsK epsV epsScale : ℝ}
    (hmaskS : ctxS.mask = none) (hmaskR : ctxR.mask = none)
    (hq : ApproxTensor (α := R)
      (toSpec := NF.toReal)
      ctxS.query ctxR.query epsQ)
    (hk : ApproxTensor (α := R)
      (toSpec := NF.toReal)
      ctxS.key ctxR.key epsK)
    (hv : ApproxTensor (α := R)
      (toSpec := NF.toReal)
      ctxS.value ctxR.value epsV)
    (hscale :
      _root_.abs
        (NF.toReal
            (1 / Attention.scaleDenom (α := R) d) -
          (1 / Attention.scaleDenom (α := ℝ) d)) ≤ epsScale)
    (hdenom : ∀ i : Fin nQ,
      NF.Softmax.denominatorErrorBound (β := β) (fexp := fexp) (rnd := rnd)
          (NF.Attention.scaledScoreErrorBound (β := β) (fexp := fexp) (rnd := rnd)
            epsQ epsK epsScale (1 / Attention.scaleDenom (α := R) d) ctxR.query ctxR.key)
          (Tensor.unstack
            (Tensor.scale
              (Tensor.matmul ctxR.query (Tensor.permute ctxR.key
                  (Shape.adjacentSwap [nK.succ, d] 0) [d, nK.succ]))
              (1 / Attention.scaleDenom (α := R) d)) i) < 1) :
    ApproxTensor (α := R)
      (toSpec := NF.toReal)
      (Attention.scaledDotProduct ctxS) (Attention.scaledDotProduct ctxR)
      (NF.Attention.outputErrorBound (β := β) (fexp := fexp) (rnd := rnd)
        epsQ epsK epsV epsScale (1 / Attention.scaleDenom (α := R) d)
        ctxR.query ctxR.key ctxR.value) := by
  have hcore := softmaxAttention
    (β := β) (fexp := fexp) (rnd := rnd)
    (qS := ctxS.query) (kS := ctxS.key) (vS := ctxS.value)
    (qR := ctxR.query) (kR := ctxR.key) (vR := ctxR.value)
    (scaleS := 1 / Attention.scaleDenom (α := ℝ) d)
    (scaleR := 1 / Attention.scaleDenom (α := R) d)
    hq hk hv hscale hdenom
  have hnf : Numeric.QuotientArithmetic.supported (α := R) = false := rfl
  simpa only [Attention.scaledDotProduct, hmaskS, hmaskR,
    Attention.scaledProductsSlice_one Numeric.QuotientArithmetic.supported_real,
    Attention.scaledProductsSlice_one hnf, Bool.false_eq_true, ↓reduceIte] using hcore

/-- Hard-masked attention over `Attention.Inputs`, for two contexts that carry the same mask.

The row certificate `evidence` is indexed by the scaled score matrices of both contexts and by the
mask, so it cannot be replayed against another mask, scale or set of parameters just because the
tensor shapes agree.
-/
theorem ApproxTensor.scaledDotProduct_masked {nQ nK d : Nat}
    {hQ : nQ ≠ 0} {hK : Nat.succ nK ≠ 0}
    (ctxS : Attention.Inputs ℝ nQ (Nat.succ nK) d hQ hK)
    (ctxR : Attention.Inputs R nQ (Nat.succ nK) d hQ hK)
    (mask : Tensor Bool [nQ, Nat.succ nK])
    {epsQ epsK epsV epsScale : ℝ}
    (hmaskS : ctxS.mask = some mask) (hmaskR : ctxR.mask = some mask)
    (hq : ApproxTensor (α := R)
      (toSpec := NF.toReal)
      ctxS.query ctxR.query epsQ)
    (hk : ApproxTensor (α := R)
      (toSpec := NF.toReal)
      ctxS.key ctxR.key epsK)
    (hv : ApproxTensor (α := R)
      (toSpec := NF.toReal)
      ctxS.value ctxR.value epsV)
    (hscale :
      _root_.abs
        (NF.toReal
            (1 / Attention.scaleDenom (α := R) d) -
          (1 / Attention.scaleDenom (α := ℝ) d)) ≤ epsScale)
    (evidence : NF.Softmax.HardMaskedRowsEvidence (β := β) (fexp := fexp) (rnd := rnd)
      (Tensor.scale (Tensor.matmul ctxS.query (Tensor.permute ctxS.key
          (Shape.adjacentSwap [nK.succ, d] 0) [d, nK.succ]))
        (1 / Attention.scaleDenom (α := ℝ) d))
      (Tensor.scale (Tensor.matmul ctxR.query (Tensor.permute ctxR.key
          (Shape.adjacentSwap [nK.succ, d] 0) [d, nK.succ]))
        (1 / Attention.scaleDenom (α := R) d))
      mask
      (NF.Attention.scaledScoreErrorBound (β := β) (fexp := fexp) (rnd := rnd)
        epsQ epsK epsScale (1 / Attention.scaleDenom (α := R) d) ctxR.query ctxR.key)) :
    ApproxTensor (α := R)
      (toSpec := NF.toReal)
      (Attention.scaledDotProduct ctxS) (Attention.scaledDotProduct ctxR)
      (NF.Attention.maskedOutputErrorBound (β := β) (fexp := fexp) (rnd := rnd)
        evidence.eta evidence.epsMax evidence.rowMaxR
        epsQ epsK epsV epsScale (1 / Attention.scaleDenom (α := R) d)
        ctxR.query ctxR.key ctxR.value mask) := by
  have hcore := hardMaskedAttention
    (β := β) (fexp := fexp) (rnd := rnd)
    (qS := ctxS.query) (kS := ctxS.key) (vS := ctxS.value)
    (qR := ctxR.query) (kR := ctxR.key) (vR := ctxR.value)
    mask
    (scaleS := 1 / Attention.scaleDenom (α := ℝ) d)
    (scaleR := 1 / Attention.scaleDenom (α := R) d)
    hq hk hv hscale evidence
  have hnf : Numeric.QuotientArithmetic.supported (α := R) = false := rfl
  simpa only [Attention.scaledDotProduct, hmaskS, hmaskR,
    Attention.scaledProductsSlice_one Numeric.QuotientArithmetic.supported_real,
    Attention.scaledProductsSlice_one hnf, Bool.false_eq_true, ↓reduceIte] using hcore

/-- Unmasked attention for a positive feature dimension `d + 1`, with the canonical scale
`1 / sqrt(d + 1)`.

`NF.Attention.approx_canonicalScale` supplies the scale approximation. Callers supply the tensor
approximations and the two error margins, `hscaleMargin` for the square root and `hdenom` for row
normalization.
-/
theorem ApproxTensor.scaledDotProduct_unmasked_canonical {nQ nK d : Nat}
    {hQ : nQ ≠ 0} {hK : Nat.succ nK ≠ 0}
    (ctxS : Attention.Inputs ℝ nQ (Nat.succ nK) (Nat.succ d) hQ hK)
    (ctxR : Attention.Inputs R nQ (Nat.succ nK) (Nat.succ d) hQ hK)
    {epsQ epsK epsV : ℝ}
    (hmaskS : ctxS.mask = none) (hmaskR : ctxR.mask = none)
    (hq : ApproxTensor (α := R)
      (toSpec := NF.toReal)
      ctxS.query ctxR.query epsQ)
    (hk : ApproxTensor (α := R)
      (toSpec := NF.toReal)
      ctxS.key ctxR.key epsK)
    (hv : ApproxTensor (α := R)
      (toSpec := NF.toReal)
      ctxS.value ctxR.value epsV)
    (hscaleMargin : NF.Attention.dimensionSqrtError (β := β) (fexp := fexp) (rnd := rnd) d < 1)
    (hdenom : ∀ i : Fin nQ,
      NF.Softmax.denominatorErrorBound (β := β) (fexp := fexp) (rnd := rnd)
          (NF.Attention.scaledScoreErrorBound (β := β) (fexp := fexp) (rnd := rnd)
            epsQ epsK (NF.Attention.canonicalScaleErrorBound (β := β) (fexp := fexp) (rnd := rnd) d)
            (1 / Attention.scaleDenom (α := R) (Nat.succ d)) ctxR.query ctxR.key)
          (Tensor.unstack
            (Tensor.scale
              (Tensor.matmul ctxR.query (Tensor.permute ctxR.key
                  (Shape.adjacentSwap [nK.succ, d.succ] 0) [d.succ, nK.succ]))
              (1 / Attention.scaleDenom (α := R) (Nat.succ d))) i) < 1) :
    ApproxTensor (α := R)
      (toSpec := NF.toReal)
      (Attention.scaledDotProduct ctxS) (Attention.scaledDotProduct ctxR)
      (NF.Attention.outputErrorBound (β := β) (fexp := fexp) (rnd := rnd)
        epsQ epsK epsV
        (NF.Attention.canonicalScaleErrorBound (β := β) (fexp := fexp) (rnd := rnd) d)
        (1 / Attention.scaleDenom (α := R) (Nat.succ d))
        ctxR.query ctxR.key ctxR.value) := by
  exact scaledDotProduct_unmasked
    (β := β) (fexp := fexp) (rnd := rnd)
    ctxS ctxR hmaskS hmaskR hq hk hv
    (NF.Attention.approx_canonicalScale (β := β) (fexp := fexp) (rnd := rnd) d hscaleMargin)
    hdenom

end

end TorchLean
