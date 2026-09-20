/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.RuntimeApprox.FP32.Layers
public import NN.Proofs.RuntimeApprox.NF.Ops.Elementwise.Unary

/-!
# FP32 MLP Approximation

This module builds on `NN.Proofs.RuntimeApprox.FP32.Layers` and packages end-to-end error bounds
for small MLP patterns that show up frequently in examples and verification pipelines.

The theorems here are intentionally architecture-shaped rather than fully generic. They are the
readable bridge lemmas that downstream verification examples can cite: “this whole FP32 MLP is
within some explicit real error budget of the corresponding real-spec MLP.”

The `_fp32` suffix refers to the rounded-real model `TorchLean.Floats.FP32 := NF binaryRadix fexp32
rnd32`, not to Lean's `Float32` or to the bit-level `ExecFloat.Binary 8 23` model. Finite binary32
add/mul
refinements are in `NN/Floats/IEEEExec/Bridge/Finite.lean`; further arithmetic refinements are in
`NN/Proofs/RuntimeApprox/IEEE32/Arithmetic.lean`.
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

/-- Explicit propagated error budget for `Linear → tanh → Linear → tanh → Linear`. -/
def tanhMlp3ErrorBudget {d0 d1 d2 d3 : Nat}
    (e0W e0b e1W e1b e2W e2b ex : ℝ)
    (L0R : LinearSpec R d0 d1) (L1R : LinearSpec R d1 d2)
    (L2R : LinearSpec R d2 d3) (xR : Tensor R [d0]) : ℝ :=
  let z0R := Spec.linearSpec (α := R) L0R xR
  let eZ0 := linearErrorBudget e0W e0b ex L0R xR
  let a0R := mapSpec Numerics.MathFunctions.tanh z0R
  let eA0 := linfNorm (Proofs.RuntimeApprox.NFBackend.tanhBoundTensor
    (β := β) (fexp := fexp) (rnd := rnd) (s := Shape.dim d1 .scalar) eZ0 z0R)
  let z1R := Spec.linearSpec (α := R) L1R a0R
  let eZ1 := linearErrorBudget e1W e1b eA0 L1R a0R
  let a1R := mapSpec Numerics.MathFunctions.tanh z1R
  let eA1 := linfNorm (Proofs.RuntimeApprox.NFBackend.tanhBoundTensor
    (β := β) (fexp := fexp) (rnd := rnd) (s := Shape.dim d2 .scalar) eZ1 z1R)
  linearErrorBudget e2W e2b eA1 L2R a1R

/--
Compositional FP32 approximation theorem for a 3-layer tanh MLP:

`Linear → tanh → Linear → tanh → Linear`.

Each parameter/input hypothesis is an `approxTensor` statement comparing the real-spec tensor with
the FP32 runtime tensor. The conclusion exposes the composed `tanhMlp3ErrorBudget`, built from the
NF backend's matrix-vector, activation, and addition bounds.
-/
theorem approxTensor_tanhMlp3_fp32 {d0 d1 d2 d3 : Nat}
    {L0S : LinearSpec ℝ d0 d1} {L1S : LinearSpec ℝ d1 d2} {L2S : LinearSpec ℝ d2 d3}
    {L0R : LinearSpec R d0 d1} {L1R : LinearSpec R d1 d2} {L2R : LinearSpec R d2 d3}
    {xS : SpecTensor [d0]} {xR : Tensor R [d0]}
    {e0W e0b e1W e1b e2W e2b ex : ℝ}
    (h0W : approxTensor (α := R) (toSpec := toSpec) L0S.weights L0R.weights e0W)
    (h0b : approxTensor (α := R) (toSpec := toSpec) L0S.bias L0R.bias e0b)
    (h1W : approxTensor (α := R) (toSpec := toSpec) L1S.weights L1R.weights e1W)
    (h1b : approxTensor (α := R) (toSpec := toSpec) L1S.bias L1R.bias e1b)
    (h2W : approxTensor (α := R) (toSpec := toSpec) L2S.weights L2R.weights e2W)
    (h2b : approxTensor (α := R) (toSpec := toSpec) L2S.bias L2R.bias e2b)
    (hx : approxTensor (α := R) (toSpec := toSpec) xS xR ex) :
    approxTensor (α := R) (toSpec := toSpec)
      (let z0 := Spec.linearSpec (α := ℝ) L0S xS
       let a0 := mapSpec Numerics.MathFunctions.tanh z0
       let z1 := Spec.linearSpec (α := ℝ) L1S a0
       let a1 := mapSpec Numerics.MathFunctions.tanh z1
       Spec.linearSpec (α := ℝ) L2S a1)
      (let z0 := Spec.linearSpec (α := R) L0R xR
       let a0 := mapSpec Numerics.MathFunctions.tanh z0
       let z1 := Spec.linearSpec (α := R) L1R a0
       let a1 := mapSpec Numerics.MathFunctions.tanh z1
       Spec.linearSpec (α := R) L2R a1)
      (tanhMlp3ErrorBudget e0W e0b e1W e1b e2W e2b ex L0R L1R L2R xR) := by
  -- First linear layer: propagate input/weight/bias error to the first pre-activation.
  let eZ0 := linearErrorBudget e0W e0b ex L0R xR
  have hZ0 :
      approxTensor (α := R) (toSpec := toSpec)
        (Spec.linearSpec (α := ℝ) L0S xS)
        (Spec.linearSpec (α := R) L0R xR) eZ0 := by
    simpa [eZ0] using approxTensor_linear_fp32
      (WS := L0S) (WR := L0R) (xS := xS) (xR := xR)
      (epsW := e0W) (epsb := e0b) (epsx := ex) h0W h0b hx
  have hA0 :=
    Proofs.RuntimeApprox.NFBackend.approxTensor_tanh_spec
      (β := β) (fexp := fexp) (rnd := rnd) (s := Shape.dim d1 .scalar)
      (xS := Spec.linearSpec (α := ℝ) L0S xS)
      (xR := Spec.linearSpec (α := R) L0R xR)
      (eps := eZ0) hZ0
  let eA0 : ℝ :=
    linfNorm (Proofs.RuntimeApprox.NFBackend.tanhBoundTensor
      (β := β) (fexp := fexp) (rnd := rnd) (s := Shape.dim d1 .scalar) eZ0
      (Spec.linearSpec (α := R) L0R xR))
  have hA0' :
      approxTensor (α := R) (toSpec := toSpec)
        (mapSpec Numerics.MathFunctions.tanh (Spec.linearSpec (α := ℝ) L0S xS))
        (mapSpec Numerics.MathFunctions.tanh (Spec.linearSpec (α := R) L0R xR))
        eA0 := by
      simpa [toSpec, NFBackend.toSpec, eA0] using hA0

  -- Second linear layer: use the tanh activation bound as this layer's input bound.
  let eZ1 := linearErrorBudget e1W e1b eA0 L1R
    (mapSpec Numerics.MathFunctions.tanh (Spec.linearSpec (α := R) L0R xR))
  have hZ1 :
      approxTensor (α := R) (toSpec := toSpec)
        (Spec.linearSpec (α := ℝ) L1S
          (mapSpec Numerics.MathFunctions.tanh (Spec.linearSpec (α := ℝ) L0S xS)))
        (Spec.linearSpec (α := R) L1R
          (mapSpec Numerics.MathFunctions.tanh (Spec.linearSpec (α := R) L0R xR))) eZ1 := by
    simpa [eZ1] using approxTensor_linear_fp32
      (WS := L1S) (WR := L1R)
      (xS := mapSpec Numerics.MathFunctions.tanh (Spec.linearSpec (α := ℝ) L0S xS))
      (xR := mapSpec Numerics.MathFunctions.tanh (Spec.linearSpec (α := R) L0R xR))
      (epsW := e1W) (epsb := e1b) (epsx := eA0) h1W h1b hA0'
  have hA1 :=
    Proofs.RuntimeApprox.NFBackend.approxTensor_tanh_spec
      (β := β) (fexp := fexp) (rnd := rnd) (s := Shape.dim d2 .scalar)
      (xS := Spec.linearSpec (α := ℝ) L1S
        (mapSpec Numerics.MathFunctions.tanh (Spec.linearSpec (α := ℝ) L0S xS)))
      (xR := Spec.linearSpec (α := R) L1R
        (mapSpec Numerics.MathFunctions.tanh (Spec.linearSpec (α := R) L0R xR)))
      (eps := eZ1) hZ1
  let eA1 : ℝ :=
    linfNorm (Proofs.RuntimeApprox.NFBackend.tanhBoundTensor
      (β := β) (fexp := fexp) (rnd := rnd) (s := Shape.dim d2 .scalar) eZ1
      (Spec.linearSpec (α := R) L1R
        (mapSpec Numerics.MathFunctions.tanh (Spec.linearSpec (α := R) L0R xR))))
  have hA1' :
      approxTensor (α := R) (toSpec := toSpec)
        (mapSpec Numerics.MathFunctions.tanh
          (Spec.linearSpec (α := ℝ) L1S
            (mapSpec Numerics.MathFunctions.tanh (Spec.linearSpec (α := ℝ) L0S xS))))
        (mapSpec Numerics.MathFunctions.tanh
          (Spec.linearSpec (α := R) L1R
            (mapSpec Numerics.MathFunctions.tanh (Spec.linearSpec (α := R) L0R xR))))
        eA1 := by
      simpa [toSpec, NFBackend.toSpec, eA1] using hA1

  -- Final linear layer: produces the network-level output approximation.
  have hOut := approxTensor_linear_fp32
      (WS := L2S) (WR := L2R)
      (xS := mapSpec Numerics.MathFunctions.tanh
        (Spec.linearSpec (α := ℝ) L1S
          (mapSpec Numerics.MathFunctions.tanh (Spec.linearSpec (α := ℝ) L0S xS))))
      (xR := mapSpec Numerics.MathFunctions.tanh
        (Spec.linearSpec (α := R) L1R
          (mapSpec Numerics.MathFunctions.tanh (Spec.linearSpec (α := R) L0R xR))))
      (epsW := e2W) (epsb := e2b) (epsx := eA1) h2W h2b hA1'

  simpa [tanhMlp3ErrorBudget, eZ0, eA0, eZ1, eA1] using hOut

/-!
## 2-layer ReLU MLP

This is the FP32 analogue of the 2-layer ReLU MLP used by CROWN/IBP:
`Linear → ReLU → Linear`.

Note: the runtime ReLU here is the *rounded* variant `reluR` used by the NFBackend forward
approximation framework (apply `max · 0` in $\mathbb R$, then round once).
-/

/-- Explicit propagated error budget for `Linear → ReLU → Linear`. -/
def reluTwoLayerMlpErrorBudget {d0 d1 d2 : Nat}
    (e0W e0b e1W e1b ex : ℝ)
    (L0R : LinearSpec R d0 d1) (L1R : LinearSpec R d1 d2)
    (xR : Tensor R [d0]) : ℝ :=
  let z0R := Spec.linearSpec (α := R) L0R xR
  let eZ0 := linearErrorBudget e0W e0b ex L0R xR
  let a0R := mapSpec (reluR (β := β) (fexp := fexp) (rnd := rnd)) z0R
  let eA0 := linfNorm (Proofs.RuntimeApprox.NFBackend.reluBoundTensor
    (β := β) (fexp := fexp) (rnd := rnd) (s := Shape.dim d1 .scalar) eZ0 z0R)
  linearErrorBudget e1W e1b eA0 L1R a0R

/--
Compositional FP32 approximation theorem for a 2-layer ReLU MLP:

`Linear → ReLU → Linear`.

This is the network-level bound consumed by the CROWN/IBP integration in
`NN.Proofs.RuntimeApprox.FP32.CROWN`.
-/
theorem approxTensor_reluTwoLayerMlp_fp32 {d0 d1 d2 : Nat}
    {L0S : LinearSpec ℝ d0 d1} {L1S : LinearSpec ℝ d1 d2}
    {L0R : LinearSpec R d0 d1} {L1R : LinearSpec R d1 d2}
    {xS : SpecTensor [d0]} {xR : Tensor R [d0]}
    {e0W e0b e1W e1b ex : ℝ}
    (h0W : approxTensor (α := R) (toSpec := toSpec) L0S.weights L0R.weights e0W)
    (h0b : approxTensor (α := R) (toSpec := toSpec) L0S.bias L0R.bias e0b)
    (h1W : approxTensor (α := R) (toSpec := toSpec) L1S.weights L1R.weights e1W)
    (h1b : approxTensor (α := R) (toSpec := toSpec) L1S.bias L1R.bias e1b)
    (hx : approxTensor (α := R) (toSpec := toSpec) xS xR ex) :
    approxTensor (α := R) (toSpec := toSpec)
      (let z0 := Spec.linearSpec (α := ℝ) L0S xS
       let a0 := mapSpec (fun x => max x 0) z0
       Spec.linearSpec (α := ℝ) L1S a0)
      (let z0 := Spec.linearSpec (α := R) L0R xR
       let a0 := mapSpec (reluR (β := β) (fexp := fexp) (rnd := rnd)) z0
       Spec.linearSpec (α := R) L1R a0)
      (reluTwoLayerMlpErrorBudget e0W e0b e1W e1b ex L0R L1R xR) := by
  -- First linear layer: real/FP32 pre-activations are close.
  let eZ0 := linearErrorBudget e0W e0b ex L0R xR
  have hZ0 :
      approxTensor (α := R) (toSpec := toSpec)
        (Spec.linearSpec (α := ℝ) L0S xS)
        (Spec.linearSpec (α := R) L0R xR) eZ0 := by
    simpa [eZ0] using approxTensor_linear_fp32
      (WS := L0S) (WR := L0R) (xS := xS) (xR := xR)
      (epsW := e0W) (epsb := e0b) (epsx := ex) h0W h0b hx

  -- ReLU: the NF backend supplies a rounded-ReLU bound from the pre-activation bound.
  have hA0 :=
    Proofs.RuntimeApprox.NFBackend.approxTensor_relu_spec
      (β := β) (fexp := fexp) (rnd := rnd) (s := Shape.dim d1 .scalar)
      (xS := Spec.linearSpec (α := ℝ) L0S xS)
      (xR := Spec.linearSpec (α := R) L0R xR)
      (eps := eZ0) hZ0
  let eA0 : ℝ :=
    linfNorm (Proofs.RuntimeApprox.NFBackend.reluBoundTensor
      (β := β) (fexp := fexp) (rnd := rnd) (s := Shape.dim d1 .scalar) eZ0
      (Spec.linearSpec (α := R) L0R xR))
  have hA0' :
      approxTensor (α := R) (toSpec := toSpec)
        (mapSpec (fun x => max x 0) (Spec.linearSpec (α := ℝ) L0S xS))
        (mapSpec (reluR (β := β) (fexp := fexp) (rnd := rnd)) (Spec.linearSpec (α := R) L0R xR))
        eA0 := by
    -- `relu_spec` is `map_spec (max · 0)` on ℝ.
      simpa [toSpec, NFBackend.toSpec, eA0] using hA0

  -- Final linear layer: propagate the activation error to the network output.
  have hOut := approxTensor_linear_fp32
      (WS := L1S) (WR := L1R)
      (xS := mapSpec (fun x => max x 0) (Spec.linearSpec (α := ℝ) L0S xS))
      (xR := mapSpec (reluR (β := β) (fexp := fexp) (rnd := rnd))
        (Spec.linearSpec (α := R) L0R xR))
      (epsW := e1W) (epsb := e1b) (epsx := eA0) h1W h1b hA0'

  simpa [reluTwoLayerMlpErrorBudget, eZ0, eA0] using hOut

end

end NN.Proofs.RuntimeApprox.FP32
