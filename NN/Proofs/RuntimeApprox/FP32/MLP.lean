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

Compose the generic layer and activation bounds for a three-layer tanh MLP and a two-layer
ReLU MLP. These are rounded-real `TorchLean.Floats.FP32` bounds, not native kernel guarantees;
the numerical model and separate IEEE refinements are described in
`NN.Proofs.RuntimeApprox.FP32.Layers`.
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

/-- Error budget for `Linear → tanh → Linear → tanh → Linear`.

Each tanh uses the coarse boundedness estimate `2 + ULP/2`, independent of its input-error
magnitude. Thus the first two layers' error magnitudes do not affect the final budget; they are
not propagated through a Lipschitz estimate. -/
def tanhMlp3ErrorBudget {d0 d1 d2 d3 : Nat}
    (e0W e0b e1W e1b e2W e2b ex : ℝ)
    (L0R : LinearSpec R d0 d1) (L1R : LinearSpec R d1 d2)
    (L2R : LinearSpec R d2 d3) (xR : Tensor R [d0]) : ℝ :=
  let z0R := Spec.linearSpec (α := R) L0R xR
  let _eZ0 := linearErrorBudget e0W e0b ex L0R xR
  let a0R := Tensor.map Numerics.MathFunctions.tanh z0R
  let eA0 := linfNorm (Proofs.RuntimeApprox.NFBackend.tanhBoundTensor
    (β := β) (fexp := fexp) (rnd := rnd) (s := Shape.dim d1 .scalar) z0R)
  let z1R := Spec.linearSpec (α := R) L1R a0R
  let _eZ1 := linearErrorBudget e1W e1b eA0 L1R a0R
  let a1R := Tensor.map Numerics.MathFunctions.tanh z1R
  let eA1 := linfNorm (Proofs.RuntimeApprox.NFBackend.tanhBoundTensor
    (β := β) (fexp := fexp) (rnd := rnd) (s := Shape.dim d2 .scalar) z1R)
  linearErrorBudget e2W e2b eA1 L2R a1R

/--
Compositional FP32 approximation theorem for a 3-layer tanh MLP:

`Linear → tanh → Linear → tanh → Linear`.

The parameter/input hypotheses compare real-spec tensors with rounded-real tensors. The
conclusion uses `tanhMlp3ErrorBudget`, including its coarse tanh estimates rather than tight
propagation of the earlier layers' errors.
-/
theorem approxTensor_tanhMlp3 {d0 d1 d2 d3 : Nat}
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
       let a0 := Tensor.map Numerics.MathFunctions.tanh z0
       let z1 := Spec.linearSpec (α := ℝ) L1S a0
       let a1 := Tensor.map Numerics.MathFunctions.tanh z1
       Spec.linearSpec (α := ℝ) L2S a1)
      (let z0 := Spec.linearSpec (α := R) L0R xR
       let a0 := Tensor.map Numerics.MathFunctions.tanh z0
       let z1 := Spec.linearSpec (α := R) L1R a0
       let a1 := Tensor.map Numerics.MathFunctions.tanh z1
       Spec.linearSpec (α := R) L2R a1)
      (tanhMlp3ErrorBudget e0W e0b e1W e1b e2W e2b ex L0R L1R L2R xR) := by
  -- First linear layer: propagate input/weight/bias error to the first pre-activation.
  let eZ0 := linearErrorBudget e0W e0b ex L0R xR
  have hZ0 :
      approxTensor (α := R) (toSpec := toSpec)
        (Spec.linearSpec (α := ℝ) L0S xS)
        (Spec.linearSpec (α := R) L0R xR) eZ0 := by
    simpa [eZ0] using approxTensor_linear
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
      (β := β) (fexp := fexp) (rnd := rnd) (s := Shape.dim d1 .scalar)
      (Spec.linearSpec (α := R) L0R xR))
  have hA0' :
      approxTensor (α := R) (toSpec := toSpec)
        (Tensor.map Numerics.MathFunctions.tanh (Spec.linearSpec (α := ℝ) L0S xS))
        (Tensor.map Numerics.MathFunctions.tanh (Spec.linearSpec (α := R) L0R xR))
        eA0 := by
    simpa [toSpec, NFBackend.toSpec, eA0] using hA0

  -- Second linear layer: use the tanh activation bound as this layer's input bound.
  let eZ1 := linearErrorBudget e1W e1b eA0 L1R
    (Tensor.map Numerics.MathFunctions.tanh (Spec.linearSpec (α := R) L0R xR))
  have hZ1 :
      approxTensor (α := R) (toSpec := toSpec)
        (Spec.linearSpec (α := ℝ) L1S
          (Tensor.map Numerics.MathFunctions.tanh (Spec.linearSpec (α := ℝ) L0S xS)))
        (Spec.linearSpec (α := R) L1R
          (Tensor.map Numerics.MathFunctions.tanh (Spec.linearSpec (α := R) L0R xR))) eZ1 := by
    simpa [eZ1] using approxTensor_linear
      (WS := L1S) (WR := L1R)
      (xS := Tensor.map Numerics.MathFunctions.tanh (Spec.linearSpec (α := ℝ) L0S xS))
      (xR := Tensor.map Numerics.MathFunctions.tanh (Spec.linearSpec (α := R) L0R xR))
      (epsW := e1W) (epsb := e1b) (epsx := eA0) h1W h1b hA0'
  have hA1 :=
    Proofs.RuntimeApprox.NFBackend.approxTensor_tanh_spec
      (β := β) (fexp := fexp) (rnd := rnd) (s := Shape.dim d2 .scalar)
      (xS := Spec.linearSpec (α := ℝ) L1S
        (Tensor.map Numerics.MathFunctions.tanh (Spec.linearSpec (α := ℝ) L0S xS)))
      (xR := Spec.linearSpec (α := R) L1R
        (Tensor.map Numerics.MathFunctions.tanh (Spec.linearSpec (α := R) L0R xR)))
      (eps := eZ1) hZ1
  let eA1 : ℝ :=
    linfNorm (Proofs.RuntimeApprox.NFBackend.tanhBoundTensor
      (β := β) (fexp := fexp) (rnd := rnd) (s := Shape.dim d2 .scalar)
      (Spec.linearSpec (α := R) L1R
        (Tensor.map Numerics.MathFunctions.tanh (Spec.linearSpec (α := R) L0R xR))))
  have hA1' :
      approxTensor (α := R) (toSpec := toSpec)
        (Tensor.map Numerics.MathFunctions.tanh
          (Spec.linearSpec (α := ℝ) L1S
            (Tensor.map Numerics.MathFunctions.tanh (Spec.linearSpec (α := ℝ) L0S xS))))
        (Tensor.map Numerics.MathFunctions.tanh
          (Spec.linearSpec (α := R) L1R
            (Tensor.map Numerics.MathFunctions.tanh (Spec.linearSpec (α := R) L0R xR))))
        eA1 := by
    simpa [toSpec, NFBackend.toSpec, eA1] using hA1

  -- Final linear layer: produces the network-level output approximation.
  have hOut := approxTensor_linear
      (WS := L2S) (WR := L2R)
      (xS := Tensor.map Numerics.MathFunctions.tanh
        (Spec.linearSpec (α := ℝ) L1S
          (Tensor.map Numerics.MathFunctions.tanh (Spec.linearSpec (α := ℝ) L0S xS))))
      (xR := Tensor.map Numerics.MathFunctions.tanh
        (Spec.linearSpec (α := R) L1R
          (Tensor.map Numerics.MathFunctions.tanh (Spec.linearSpec (α := R) L0R xR))))
      (epsW := e2W) (epsb := e2b) (epsx := eA1) h2W h2b hA1'

  simpa [tanhMlp3ErrorBudget, eA0, eA1] using hOut

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
  let a0R := Tensor.map (reluR (β := β) (fexp := fexp) (rnd := rnd)) z0R
  let eA0 := linfNorm (Proofs.RuntimeApprox.NFBackend.reluBoundTensor
    (β := β) (fexp := fexp) (rnd := rnd) (s := Shape.dim d1 .scalar) eZ0 z0R)
  linearErrorBudget e1W e1b eA0 L1R a0R

/--
Compositional FP32 approximation theorem for a 2-layer ReLU MLP:

`Linear → ReLU → Linear`.

This is the network-level bound consumed by the CROWN/IBP integration in
`NN.Proofs.RuntimeApprox.FP32.CROWN`.
-/
theorem approxTensor_reluTwoLayerMlp {d0 d1 d2 : Nat}
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
       let a0 := Tensor.map (fun x => max x 0) z0
       Spec.linearSpec (α := ℝ) L1S a0)
      (let z0 := Spec.linearSpec (α := R) L0R xR
       let a0 := Tensor.map (reluR (β := β) (fexp := fexp) (rnd := rnd)) z0
       Spec.linearSpec (α := R) L1R a0)
      (reluTwoLayerMlpErrorBudget e0W e0b e1W e1b ex L0R L1R xR) := by
  -- First linear layer: real/FP32 pre-activations are close.
  let eZ0 := linearErrorBudget e0W e0b ex L0R xR
  have hZ0 :
      approxTensor (α := R) (toSpec := toSpec)
        (Spec.linearSpec (α := ℝ) L0S xS)
        (Spec.linearSpec (α := R) L0R xR) eZ0 := by
    simpa [eZ0] using approxTensor_linear
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
        (Tensor.map (fun x => max x 0) (Spec.linearSpec (α := ℝ) L0S xS))
        (Tensor.map (reluR (β := β) (fexp := fexp) (rnd := rnd)) (Spec.linearSpec (α := R) L0R xR))
        eA0 := by
    -- The NF backend states the real side of the ReLU bound as `Tensor.map (fun x => max x 0)`.
    simpa [toSpec, NFBackend.toSpec, eA0] using hA0

  -- Final linear layer: propagate the activation error to the network output.
  have hOut := approxTensor_linear
      (WS := L1S) (WR := L1R)
      (xS := Tensor.map (fun x => max x 0) (Spec.linearSpec (α := ℝ) L0S xS))
      (xR := Tensor.map (reluR (β := β) (fexp := fexp) (rnd := rnd))
        (Spec.linearSpec (α := R) L0R xR))
      (epsW := e1W) (epsb := e1b) (epsx := eA0) h1W h1b hA0'

  simpa [reluTwoLayerMlpErrorBudget, eZ0, eA0] using hOut

end

end NN.Proofs.RuntimeApprox.FP32
