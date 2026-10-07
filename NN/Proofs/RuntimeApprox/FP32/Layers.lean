/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.RuntimeApprox.NF.Linalg
public import NN.Spec.Layers.Linear
public import NN.Floats.FP32
public import NN.Proofs.RuntimeApprox.NF.Ops.Elementwise.Binary

/-!
# FP32 Layer Approximation

Specialize the generic NF bounds to `TorchLean.Floats.FP32` and compose matrix-vector
multiplication with bias addition. `linearErrorBudget` exposes the resulting infinity-norm bound
for use in the MLP and CROWN/IBP proofs.

This rounded-real model uses nearest-even binary32 precision without an upper exponent bound
or NaN/Inf values. Applying these bounds to native or encoded IEEE execution requires a separate
refinement excluding exceptional results. See `NN/Floats/IEEEExec/Bridge/Finite.lean` and
`NN/Proofs/RuntimeApprox/IEEE32/Arithmetic.lean`.
-/

@[expose] public section

open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)

open FloatLib FloatLib.Numerics FloatLib.Floats.Formats
open Flocq

namespace NN.Proofs.RuntimeApprox.FP32

open _root_.Spec _root_.TorchLean
open _root_.TorchLean.Tensor

open _root_.Proofs
open _root_.Proofs.RuntimeApprox
open _root_.Proofs.RuntimeApprox.NFBackend
open TorchLean.Floats

noncomputable section

/-- Runtime scalar type: float32 rounding model (`TorchLean.Floats.FP32`). -/
abbrev R : Type := TorchLean.Floats.FP32

/-- Radix for the `FP32` rounding model (binary). -/
abbrev β : Radix := binaryRadix
/-- Exponent function used by the `FP32` rounding model. -/
abbrev fexp : ℤ → ℤ := (Model.fexpOf FloatFormat.binary32)
/-- Round-to-nearest-even function used by the `FP32` rounding model. -/
abbrev rnd : ℝ → ℤ := FloatLib.Floats.Formats.Flocq.nearestEven

/-- Interpretation of runtime scalars as real spec scalars, specialized to `FP32`. -/
abbrev toSpec : R → ℝ :=
  _root_.Proofs.RuntimeApprox.NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd)

/-- Explicit expression for the propagated infinity-norm error of an FP32 linear layer. -/
def linearErrorBudget {inDim outDim : Nat}
    (epsW epsb epsx : ℝ) (WR : LinearSpec R inDim outDim)
    (xR : Tensor R [inDim]) : ℝ :=
  linfNorm
    (Proofs.RuntimeApprox.NFBackend.addBoundTensor
      (β := β) (fexp := fexp) (rnd := rnd) (s := Shape.dim outDim .scalar)
      (linfNorm
        (Proofs.RuntimeApprox.NFBackend.matVecMulBoundTensor
          (β := β) (fexp := fexp) (rnd := rnd) (m := outDim) (n := inDim)
          epsW epsx WR.weights xR))
      epsb
      (Spec.matVecMulSpec (α := R) WR.weights xR)
      WR.bias)

/--
Forward error bound for a linear layer `y = Wx + b` under the `FP32` rounding semantics.

Inputs:
- `hW`, `hb`, and `hx` say the runtime weights, bias, and input approximate their real-spec
  counterparts.

The conclusion exposes `linearErrorBudget`, rather than hiding the propagated quantity behind an
existential. It combines the matrix-vector product budget with the final rounded bias addition.

This is the base layer theorem used by the MLP and CROWN/IBP FP32 wrappers.
-/
theorem approxTensor_linear {inDim outDim : Nat}
    {WS : LinearSpec ℝ inDim outDim} {xS : SpecTensor [inDim]}
    {WR : LinearSpec R inDim outDim} {xR : Tensor R [inDim]}
    {epsW epsb epsx : ℝ}
    (hW : approxTensor (α := R) (toSpec := toSpec) WS.weights WR.weights epsW)
    (hb : approxTensor (α := R) (toSpec := toSpec) WS.bias WR.bias epsb)
    (hx : approxTensor (α := R) (toSpec := toSpec) xS xR epsx) :
    approxTensor (α := R) (toSpec := toSpec)
      (Spec.linearSpec (α := ℝ) WS xS)
      (Spec.linearSpec (α := R) WR xR)
      (linearErrorBudget epsW epsb epsx WR xR) := by
  have hmv :=
    Proofs.RuntimeApprox.NFBackend.approxTensor_mat_vec_mul_spec
      (β := β) (fexp := fexp) (rnd := rnd) (m := outDim) (n := inDim)
      (AS := WS.weights) (vS := xS) (AR := WR.weights) (vR := xR)
      (epsA := epsW) (epsV := epsx) hW hx
  have hadd :=
    Proofs.RuntimeApprox.NFBackend.approxTensor_add_spec
      (β := β) (fexp := fexp) (rnd := rnd) (s := Shape.dim outDim .scalar)
      (xS := Spec.matVecMulSpec (α := ℝ) WS.weights xS)
      (yS := WS.bias)
      (xR := Spec.matVecMulSpec (α := R) WR.weights xR)
      (yR := WR.bias)
      (epsx := linfNorm
        (Proofs.RuntimeApprox.NFBackend.matVecMulBoundTensor
          (β := β) (fexp := fexp) (rnd := rnd) (m := outDim) (n := inDim)
          epsW epsx WR.weights xR))
      (epsy := epsb)
      hmv hb
  simpa [linearErrorBudget, Spec.linearSpec] using hadd

end

end NN.Proofs.RuntimeApprox.FP32
