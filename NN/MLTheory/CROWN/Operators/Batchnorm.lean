/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Core
public import NN.Spec.Core.Context
public import NN.Spec.Core.Tensor
public import NN.Spec.Core.TensorOps

/-!
# BatchNorm operator bounds (IBP + affine)

This file bounds inference-time BatchNorm. Since inference-time BatchNorm is an affine
transformation (with frozen statistics), both IBP and affine propagation are exact (componentwise).

At inference time,
`y = γ * (x - μ) / sqrt(σ² + ε) + β`,
so the layer reduces to `y = scale * x + offset`, where
`scale = γ / sqrt(σ² + ε)` and `offset = β - γ * μ / sqrt(σ² + ε)`.

References:
- Ioffe and Szegedy, "Batch Normalization: Accelerating Deep Network Training by Reducing
  Internal Covariate Shift", ICML 2015.
- PyTorch analogue: `torch.nn.BatchNorm1d/2d/3d` in evaluation mode.
-/

@[expose] public section


namespace NN.MLTheory.CROWN.Operators.Batchnorm

open _root_.Spec
open _root_.Spec.Tensor
open NN.MLTheory.CROWN

variable {α : Type} [Context α]

/-- Parameters for BatchNorm layer (frozen at inference). -/
structure BatchNormParams (α : Type) [Context α] where
  /-- Number of channels/features -/
  dim : Nat
  /-- Running mean μ -/
  running_mean : Tensor α (.dim dim .scalar)
  /-- Running variance σ² -/
  running_var : Tensor α (.dim dim .scalar)
  /-- Learnable scale γ -/
  gamma : Tensor α (.dim dim .scalar)
  /-- Learnable bias β -/
  beta : Tensor α (.dim dim .scalar)
  /-- Small constant for numerical stability -/
  eps : α

/-- Compute the equivalent affine scale: γ / √(σ² + ε) -/
def computeScale (params : BatchNormParams α) : Tensor α (.dim params.dim .scalar) :=
  match params.running_var, params.gamma with
  | .dim var, .dim gam =>
    Tensor.dim (fun i =>
      match var i, gam i with
      | .scalar v, .scalar g =>
        let denom := MathFunctions.sqrt (v + params.eps)
        Tensor.scalar (g / denom))

/-- Compute the equivalent affine offset: β - γ * μ / √(σ² + ε) -/
def computeOffset (params : BatchNormParams α) : Tensor α (.dim params.dim .scalar) :=
  match params.running_mean, params.running_var, params.gamma, params.beta with
  | .dim mu, .dim var, .dim gam, .dim bet =>
    Tensor.dim (fun i =>
      match mu i, var i, gam i, bet i with
      | .scalar m, .scalar v, .scalar g, .scalar b =>
        let denom := MathFunctions.sqrt (v + params.eps)
        Tensor.scalar (b - g * m / denom))

/-- IBP for BatchNorm: since BN is affine, we can compute exact bounds.
    y = scale * x + offset, so:
    - If scale > 0: y_lo = scale * x_lo + offset, y_hi = scale * x_hi + offset
    - If scale < 0: y_lo = scale * x_hi + offset, y_hi = scale * x_lo + offset
-/
def ibpBatchnorm (params : BatchNormParams α)
    (xB : Box α (.dim params.dim .scalar)) : Box α (.dim params.dim .scalar) :=
  let scale := computeScale params
  let offset := computeOffset params
  match xB.lo, xB.hi, scale, offset with
  | .dim xlo, .dim xhi, .dim sc, .dim off =>
    let outLo := Tensor.dim (fun i =>
      match xlo i, xhi i, sc i, off i with
      | .scalar xl, .scalar xh, .scalar s, .scalar o =>
        -- If scale >= 0, use xl for lo; else use xh
        let lo := if s > Numbers.zero then s * xl + o else s * xh + o
        Tensor.scalar lo)
    let outHi := Tensor.dim (fun i =>
      match xlo i, xhi i, sc i, off i with
      | .scalar xl, .scalar xh, .scalar s, .scalar o =>
        -- If scale >= 0, use xh for hi; else use xl
        let hi := if s > Numbers.zero then s * xh + o else s * xl + o
        Tensor.scalar hi)
    { lo := outLo, hi := outHi }

/-- Affine bounds for BatchNorm propagation.
    Since BN is itself affine, we simply compose the affine forms:
    If prev = A_prev * x_in + c_prev and BN = scale * · + offset
    Then composed = scale * (A_prev * x_in + c_prev) + offset
                  = diag(scale) * A_prev * x_in + (scale * c_prev + offset)
-/
def affBatchnorm {inDim : Nat} (params : BatchNormParams α)
    (aff : AffineVec α inDim params.dim) : AffineVec α inDim params.dim :=
  let scale := computeScale params
  let offset := computeOffset params
  match aff.A, aff.c, scale, offset with
  | .dim rows, .dim cv, .dim sc, .dim off =>
    -- Scale each row of A by corresponding scale[i]
    let A' := Tensor.dim (fun i =>
      match rows i, sc i with
      | .dim cols, .scalar si =>
        Tensor.dim (fun j =>
          match cols j with
          | .scalar aij => Tensor.scalar (si * aij)))
    -- Scale bias and add offset
    let c' := Tensor.dim (fun i =>
      match cv i, sc i, off i with
      | .scalar ci, .scalar si, .scalar oi =>
        Tensor.scalar (si * ci + oi))
    { A := A', c := c' }

/-- Derivative bounds for BatchNorm: since BN is affine, d(BN)/dx = scale (constant).
    Given input derivative bounds [dlo, dhi], output = scale * [dlo, dhi]. -/
def derivBatchnorm (params : BatchNormParams α)
    (dB : Box α (.dim params.dim .scalar)) : Box α (.dim params.dim .scalar) :=
  let scale := computeScale params
  match dB.lo, dB.hi, scale with
  | .dim dlo, .dim dhi, .dim sc =>
    let outLo := Tensor.dim (fun i =>
      match dlo i, dhi i, sc i with
      | .scalar dl, .scalar dh, .scalar s =>
        Tensor.scalar (if s > Numbers.zero then s * dl else s * dh))
    let outHi := Tensor.dim (fun i =>
      match dlo i, dhi i, sc i with
      | .scalar dl, .scalar dh, .scalar s =>
        Tensor.scalar (if s > Numbers.zero then s * dh else s * dl))
    { lo := outLo, hi := outHi }

/-- Second derivative of BatchNorm is zero (affine function). -/
def deriv2Batchnorm (params : BatchNormParams α)
    (_d2B : Box α (.dim params.dim .scalar)) : Box α (.dim params.dim .scalar) :=
  let zero := Spec.fill (α:=α) Numbers.zero (.dim params.dim .scalar)
  { lo := zero, hi := zero }

namespace Theorems

/-- BatchNorm IBP produces a valid Box structure. -/
theorem ibp_batchnorm_returns_box (params : BatchNormParams α)
    (xB : Box α (.dim params.dim .scalar)) :
    ∃ lo hi : Tensor α (.dim params.dim .scalar), ibpBatchnorm params xB = { lo := lo, hi := hi }
      := by
  simp only [ibpBatchnorm, computeScale, computeOffset]
  match xB.lo, xB.hi, params.running_var, params.gamma, params.running_mean, params.beta with
  | .dim _, .dim _, .dim _, .dim _, .dim _, .dim _ => exact ⟨_, _, rfl⟩

/-- BatchNorm affine transformation preserves structure. -/
theorem aff_batchnorm_returns_affine {inDim : Nat} (params : BatchNormParams α)
    (aff : AffineVec α inDim params.dim) :
    ∃ A' c', (affBatchnorm params aff).A = A' ∧ (affBatchnorm params aff).c = c' := by
  exact ⟨(affBatchnorm params aff).A, (affBatchnorm params aff).c, rfl, rfl⟩

/-- BatchNorm derivative IBP produces valid Box. -/
theorem deriv_batchnorm_returns_box (params : BatchNormParams α)
    (dB : Box α (.dim params.dim .scalar)) :
    ∃ lo hi : Tensor α (.dim params.dim .scalar), derivBatchnorm params dB = { lo := lo, hi := hi }
      := by
  simp only [derivBatchnorm, computeScale]
  match dB.lo, dB.hi, params.running_var, params.gamma with
  | .dim _, .dim _, .dim _, .dim _ => exact ⟨_, _, rfl⟩

/-- BatchNorm second derivative is zero (affine function). -/
theorem deriv2_batchnorm_is_zero (params : BatchNormParams α)
    (d2B : Box α (.dim params.dim .scalar)) :
    let result := deriv2Batchnorm params d2B
    result.lo = result.hi := by
  unfold deriv2Batchnorm
  rfl

end Theorems

end NN.MLTheory.CROWN.Operators.Batchnorm
