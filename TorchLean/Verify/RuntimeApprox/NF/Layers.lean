/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.RuntimeApprox.NF.Linalg
public import TorchLean.Verify.RuntimeApprox.NF.Elementwise.Binary
public import TorchLean.Tensor.Layers.Linear

/-!
# Linear layers in the NF model

This module bounds the forward error of a linear layer `y = Wx + b` in the rounded-real model
`NF β fexp rnd`. The matrix-vector product rounds every multiplication and addition in index
order, and the bias addition rounds once more.

The model rounds onto the grid of `fexp` without an upper exponent bound, so it has no overflow,
NaN or infinity. `TorchLean.FP32` is the instance at `binaryRadix`,
`Model.fexpOf FloatFormat.binary32` and `nearestEven`. Applying these bounds to IEEE execution
needs a refinement that excludes those exceptional results, such as the finite refinements in
`TorchLean/Verify/Floats/IEEE754.lean`.

`linearErrorBudget` names the propagated infinity-norm bound, and `approxTensor_linear` proves it.
-/

@[expose] public section

namespace TorchLean
namespace NF

open TorchLean.Tensor
open Robustness

noncomputable section

open FloatLib FloatLib.Numerics FloatLib.Floats.Formats
open Flocq

variable {β : Radix} {fexp : ℤ → ℤ} [ValidExp fexp]
variable {rnd : ℝ → ℤ} [ValidRndToNearest rnd]

local notation "R" => NF β fexp rnd

/-- The infinity-norm error bound of the rounded linear layer `WR.weights xR + WR.bias`, given the
errors `epsW`, `epsb` and `epsx` of its weights, bias and input. -/
def linearErrorBudget {inDim outDim : Nat} (epsW epsb epsx : ℝ) (WR : Linear R inDim outDim)
    (xR : Tensor R [inDim]) : ℝ :=
  linfNorm
    (addBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := Shape.dim outDim .scalar)
      (linfNorm
        (matVecMulBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (m := outDim) (n := inDim)
          epsW epsx WR.weights xR))
      epsb
      (Tensor.contract WR.weights xR _ _ [outDim] .matVec)
      WR.bias)

/-- If the runtime weights, bias and input approximate the real ones within `epsW`, `epsb` and
`epsx` (`hW`, `hb`, `hx`), the rounded linear layer approximates the real one within
`linearErrorBudget epsW epsb epsx WR xR`. -/
theorem approxTensor_linear {inDim outDim : Nat}
    {WS : Linear ℝ inDim outDim} {xS : Tensor ℝ [inDim]}
    {WR : Linear R inDim outDim} {xR : Tensor R [inDim]}
    {epsW epsb epsx : ℝ}
    (hW : ApproxTensor (α := R) (toSpec := NF.toReal)
      WS.weights WR.weights epsW)
    (hb : ApproxTensor (α := R) (toSpec := NF.toReal)
      WS.bias WR.bias epsb)
    (hx : ApproxTensor (α := R) (toSpec := NF.toReal)
      xS xR epsx) :
    ApproxTensor (α := R) (toSpec := NF.toReal)
      (Linear.forward (α := ℝ) WS xS)
      (Linear.forward (α := R) WR xR)
      (linearErrorBudget epsW epsb epsx WR xR) := by
  have hmv :=
    ApproxTensor.contract_matVec (β := β) (fexp := fexp) (rnd := rnd) (m := outDim) (n := inDim)
      (AS := WS.weights) (vS := xS) (AR := WR.weights) (vR := xR)
      (epsA := epsW) (epsV := epsx) hW hx
  have hadd :=
    ApproxTensor.add (β := β) (fexp := fexp) (rnd := rnd) (s := Shape.dim outDim .scalar)
      (xS := Tensor.contract WS.weights xS _ _ [outDim] .matVec)
      (yS := WS.bias)
      (xR := Tensor.contract WR.weights xR _ _ [outDim] .matVec)
      (yR := WR.bias)
      (epsx := linfNorm
        (matVecMulBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (m := outDim) (n := inDim)
          epsW epsx WR.weights xR))
      (epsy := epsb)
      hmv hb
  simpa [linearErrorBudget, Linear.forward] using hadd

end
end NF
end TorchLean
