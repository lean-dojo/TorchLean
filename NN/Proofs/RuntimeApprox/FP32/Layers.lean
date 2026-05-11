/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.FP32
public import NN.Proofs.RuntimeApprox.NF.Linalg
public import NN.Spec.Layers.Activation
public import NN.Spec.Layers.Linear
public import NN.Runtime.Context

/-!
# FP32 Layer Approximation

This module specializes the backend-generic runtime-approximation framework
(`NN.Proofs.RuntimeApprox`) to the concrete float32 rounding model `TorchLean.Floats.FP32`
(round-to-nearest-even with an IEEE-754 binary32-style exponent function).

The lemmas here are *compositional*: they let you relate a real-valued spec computation
to its float32 execution under an explicit error budget, so that larger network theorems can be
proved by chaining smaller ones.

Trust boundary: `TorchLean.Floats.FP32` is a finite rounding model exposed to Lean. These statements
are about real-valued spec computations and their rounded counterparts, under the intended side
condition that execution stays finite (no NaN/Inf/overflow in an IEEE-754 hardware sense).
-/

@[expose] public section


namespace NN.Proofs.RuntimeApprox.FP32

open _root_.Spec
open _root_.Spec.Tensor

open _root_.Proofs
open _root_.Proofs.RuntimeApprox
open _root_.Proofs.RuntimeApprox.NFBackend
open TorchLean.Floats

noncomputable section

/-- Runtime scalar type: float32 rounding model (`TorchLean.Floats.FP32`). -/
abbrev R : Type := TorchLean.Floats.FP32

/-- Radix for the `FP32` rounding model (binary). -/
abbrev β : NeuralRadix := binaryRadix
/-- Exponent function used by the `FP32` rounding model. -/
abbrev fexp : ℤ → ℤ := TorchLean.Floats.fexp32
/-- Round-to-nearest-even function used by the `FP32` rounding model. -/
abbrev rnd : ℝ → ℤ := TorchLean.Floats.rnd32

/-- Interpretation of runtime scalars as real spec scalars, specialized to `FP32`. -/
abbrev toSpec : R → ℝ :=
  _root_.Proofs.RuntimeApprox.NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd)

/--
Forward error bound for a linear layer `y = Wx + b` under the `FP32` rounding semantics.

Inputs:
- `hW`, `hb`, and `hx` say the runtime weights, bias, and input approximate their real-spec
  counterparts.

Output:
- an explicit existential error budget `eps` such that the whole FP32 linear-layer result
  approximates the real-spec linear-layer result.

This is the base layer theorem used by the MLP and CROWN/IBP FP32 wrappers.
-/
theorem approxT_linear_fp32 {inDim outDim : Nat}
    {WS : LinearSpec ℝ inDim outDim} {xS : SpecTensor (.dim inDim .scalar)}
    {WR : LinearSpec R inDim outDim} {xR : Tensor R (.dim inDim .scalar)}
    {epsW epsb epsx : ℝ}
    (hW : approxT (α := R) (toSpec := toSpec) WS.weights WR.weights epsW)
    (hb : approxT (α := R) (toSpec := toSpec) WS.bias WR.bias epsb)
    (hx : approxT (α := R) (toSpec := toSpec) xS xR epsx) :
    ∃ eps : ℝ,
      approxT (α := R) (toSpec := toSpec)
        (Spec.linearSpec (α := ℝ) WS xS)
        (Spec.linearSpec (α := R) WR xR)
        eps := by
  -- The linear layer factors into matvec followed by bias addition. Each operation already has
  -- an NF-backend approximation theorem; this theorem simply specializes and composes them for
  -- the concrete FP32 rounding model.
  have hmv :=
    Proofs.RuntimeApprox.NFBackend.approxT_mat_vec_mul_spec
      (β := β) (fexp := fexp) (rnd := rnd) (m := outDim) (n := inDim)
      (AS := WS.weights) (vS := xS) (AR := WR.weights) (vR := xR)
      (epsA := epsW) (epsV := epsx) hW hx
  have hadd :=
    Proofs.RuntimeApprox.NFBackend.approxT_add_spec
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
  let eps : ℝ :=
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
  refine ⟨eps, ?_⟩
  simpa [eps, Spec.linearSpec] using hadd

end

end NN.Proofs.RuntimeApprox.FP32
