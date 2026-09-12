/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Core
public import NN.Spec.Core.Tensor -- shake: keep

/-!
# BatchNorm operator bounds (IBP + affine)

This file bounds inference-time BatchNorm. Since inference-time BatchNorm is an affine
transformation (with frozen statistics), both IBP and affine propagation are exact (componentwise).

At inference time, TorchLean uses
`y = γ * (x - μ) / sqrt(max(σ², 0) + ε) + β`,
so the layer reduces to `y = scale * x + offset`, where
`scale = γ / sqrt(max(σ², 0) + ε)` and
`offset = β - γ * μ / sqrt(max(σ², 0) + ε)`.

The `max` is the same totalization used by `Spec.batchNormInference` and the IR evaluator. It has no
effect on valid nonnegative running variances, while keeping every TorchLean layer aligned on
malformed approximate-runtime inputs.

References:
- Ioffe and Szegedy, "Batch Normalization: Accelerating Deep Network Training by Reducing
  Internal Covariate Shift", ICML 2015.
- PyTorch analogue: `torch.nn.BatchNorm1d/2d/3d` in evaluation mode.
-/

@[expose] public section


namespace NN.MLTheory.CROWN.Operators.BatchNorm

open Spec TorchLean
open TorchLean.Tensor
open NN.MLTheory.CROWN

variable {α : Type} [TorchLean.Storage α] [Context α]

/-- Parameters for BatchNorm layer (frozen at inference). -/
structure BatchNormParams (α : Type) [TorchLean.Storage α] [Context α] where
  /-- Number of channels/features -/
  dim : Nat
  /-- Running mean μ -/
  running_mean : Tensor α [dim]
  /-- Running variance σ² -/
  running_var : Tensor α [dim]
  /-- Learnable scale γ -/
  gamma : Tensor α [dim]
  /-- Learnable bias β -/
  beta : Tensor α [dim]
  /-- Small constant for numerical stability -/
  eps : α

/-- Compute the equivalent affine scale: `γ / sqrt(max(σ², 0) + ε)`. -/
def computeScale (params : BatchNormParams α) : Tensor α [params.dim] :=
  Tensor.dim (fun i =>
    let v := params.running_var.getScalar i
    let g := params.gamma.getScalar i
    let denom := MathFunctions.sqrt (max v 0 + params.eps)
    Tensor.scalar (g / denom))

/-- Compute the equivalent affine offset: `β - γ * μ / sqrt(max(σ², 0) + ε)`. -/
def computeOffset (params : BatchNormParams α) : Tensor α [params.dim] :=
  Tensor.dim (fun i =>
    let m := params.running_mean.getScalar i
    let v := params.running_var.getScalar i
    let g := params.gamma.getScalar i
    let b := params.beta.getScalar i
    let denom := MathFunctions.sqrt (max v 0 + params.eps)
    Tensor.scalar (b - g * m / denom))

/-- IBP for BatchNorm. Since BatchNorm is affine, its bounds are exact.

For $y=sx+o$:

- if $s>0$, then $y_{\mathrm{lo}}=s x_{\mathrm{lo}}+o$ and
  $y_{\mathrm{hi}}=s x_{\mathrm{hi}}+o$;
- if $s<0$, then $y_{\mathrm{lo}}=s x_{\mathrm{hi}}+o$ and
  $y_{\mathrm{hi}}=s x_{\mathrm{lo}}+o$.
-/
def ibpBatchNorm (params : BatchNormParams α)
    (xB : Box α (.dim params.dim .scalar)) : Box α (.dim params.dim .scalar) :=
  let scale := computeScale params
  let offset := computeOffset params
  let outLo := Tensor.dim (fun i =>
    let xl := xB.lo.getScalar i
    let xh := xB.hi.getScalar i
    let s := scale.getScalar i
    let o := offset.getScalar i
    Tensor.scalar (if s > 0 then s * xl + o else s * xh + o))
  let outHi := Tensor.dim (fun i =>
    let xl := xB.lo.getScalar i
    let xh := xB.hi.getScalar i
    let s := scale.getScalar i
    let o := offset.getScalar i
    Tensor.scalar (if s > 0 then s * xh + o else s * xl + o))
  { lo := outLo, hi := outHi }

/-- Affine bounds for BatchNorm propagation.

Since BatchNorm is affine, compose the two affine forms:

$$
\begin{aligned}
y_{\mathrm{prev}} &= A_{\mathrm{prev}}x_{\mathrm{in}}+c_{\mathrm{prev}},\\
\operatorname{BN}(y) &= sy+o,\\
\operatorname{BN}(y_{\mathrm{prev}})
  &= \operatorname{diag}(s)A_{\mathrm{prev}}x_{\mathrm{in}}
     +(s c_{\mathrm{prev}}+o).
\end{aligned}
$$
-/
def affBatchNorm {inDim : Nat} (params : BatchNormParams α)
    (aff : AffineVec α inDim params.dim) : AffineVec α inDim params.dim :=
  let scale := computeScale params
  let offset := computeOffset params
  let A' := Tensor.dim (fun i =>
    let si := scale.getScalar i
    Tensor.dim (fun j => Tensor.scalar (si * get2 aff.A i j)))
  let c' := Tensor.dim (fun i =>
    Tensor.scalar (scale.getScalar i * aff.c.getScalar i + offset.getScalar i))
  { A := A', c := c' }

/-- Derivative bounds for BatchNorm. Since BatchNorm is affine,
$\frac{d}{dx}\operatorname{BN}(x)=s$ is constant. Input bounds
$[d_{\mathrm{lo}},d_{\mathrm{hi}}]$ therefore become
$s[d_{\mathrm{lo}},d_{\mathrm{hi}}]$. -/
def derivBatchNorm (params : BatchNormParams α)
    (dB : Box α (.dim params.dim .scalar)) : Box α (.dim params.dim .scalar) :=
  let scale := computeScale params
  let outLo := Tensor.dim (fun i =>
    let dl := dB.lo.getScalar i
    let dh := dB.hi.getScalar i
    let s := scale.getScalar i
    Tensor.scalar (if s > 0 then s * dl else s * dh))
  let outHi := Tensor.dim (fun i =>
    let dl := dB.lo.getScalar i
    let dh := dB.hi.getScalar i
    let s := scale.getScalar i
    Tensor.scalar (if s > 0 then s * dh else s * dl))
  { lo := outLo, hi := outHi }

/--
Propagate second-derivative bounds through inference-time BatchNorm.

Although the second derivative of the affine map `x ↦ scale * x + offset` with respect to `x` is
zero, composition with a curve `x(t)` gives `d²/dt² BN(x(t)) = scale * x''(t)`. Consequently this
uses the same signed scaling rule as first-derivative propagation.
-/
def secondDerivBatchNorm (params : BatchNormParams α)
    (d2B : Box α (.dim params.dim .scalar)) : Box α (.dim params.dim .scalar) :=
  derivBatchNorm params d2B

end NN.MLTheory.CROWN.Operators.BatchNorm
