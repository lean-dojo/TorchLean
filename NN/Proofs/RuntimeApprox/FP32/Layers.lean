/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.RuntimeApprox.NF.Linalg
public import NN.Spec.Layers.Linear
public import NN.Floats.FP32.Core
public import NN.Proofs.RuntimeApprox.NF.Ops.Elementwise.Binary

/-!
# FP32 Layer Approximation

This module specializes the backend-generic runtime-approximation framework
(`NN.Proofs.RuntimeApprox`) to the rounded-real float32 model
`TorchLean.Floats.FP32 := NF binaryRadix fexp32 rnd32` (round-to-nearest-even with the IEEE-754
binary32 exponent function). Every theorem in this directory, including those whose names end in
`_fp32`, is a statement about that rounded-real model. None of them is a statement about Lean's
`Float32` type or about the bit-level `ExecFloat.Binary 8 23` model. Finite binary32 add/mul
refinements are in
`NN/Floats/IEEEExec/Bridge/Finite.lean`; further arithmetic refinements are in
`NN/Proofs/RuntimeApprox/IEEE32/Arithmetic.lean`.

The lemmas here are *compositional*: they let you relate a real-valued spec computation
to its float32 execution under an explicit error budget, so that larger network theorems can be
proved by chaining smaller ones.

Trust boundary: `TorchLean.Floats.FP32` is a finite rounding model exposed to Lean. These statements
are about real-valued spec computations and their rounded counterparts, under the intended side
condition that execution stays finite (no NaN/Inf/overflow in an IEEE-754 hardware sense).
-/

@[expose] public section

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
abbrev fexp : ℤ → ℤ := TorchLean.Floats.fexp32
/-- Round-to-nearest-even function used by the `FP32` rounding model. -/
abbrev rnd : ℝ → ℤ := TorchLean.Floats.rnd32

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
theorem approxTensor_linear_fp32 {inDim outDim : Nat}
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
  -- The linear layer factors into matvec followed by bias addition. Each operation already has
  -- an NF-backend approximation theorem; this theorem specializes and composes them for
  -- the concrete FP32 rounding model.
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
