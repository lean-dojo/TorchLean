/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.RuntimeApprox.FP32.Layers

/-!
# FP32 MLP Approximation

This module builds on `NN.Proofs.RuntimeApprox.FP32.Layers` and packages end-to-end error bounds
for small MLP patterns that show up frequently in examples and verification pipelines.

The theorems here are intentionally architecture-shaped rather than fully generic. They are the
readable bridge lemmas that downstream verification examples can cite: “this whole FP32 MLP is
within some explicit real error budget of the corresponding real-spec MLP.”
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

/--
Compositional FP32 approximation theorem for a 3-layer tanh MLP:

`Linear → tanh → Linear → tanh → Linear`.

Each parameter/input hypothesis is an `approxT` statement comparing the real-spec tensor with the
FP32 runtime tensor. The conclusion existentially packages the propagated output budget. We keep the
budget existential because the exact expression is intentionally produced by the NF backend
combinators (`matVecMulBoundTensor`, `tanhBoundTensor`, `addBoundTensor`) rather than hand-expanded
at every call site.
-/
theorem approxT_tanhMlp3_fp32 {d0 d1 d2 d3 : Nat}
    {L0S : LinearSpec ℝ d0 d1} {L1S : LinearSpec ℝ d1 d2} {L2S : LinearSpec ℝ d2 d3}
    {L0R : LinearSpec R d0 d1} {L1R : LinearSpec R d1 d2} {L2R : LinearSpec R d2 d3}
    {xS : SpecTensor (.dim d0 .scalar)} {xR : Tensor R (.dim d0 .scalar)}
    {e0W e0b e1W e1b e2W e2b ex : ℝ}
    (h0W : approxT (α := R) (toSpec := toSpec) L0S.weights L0R.weights e0W)
    (h0b : approxT (α := R) (toSpec := toSpec) L0S.bias L0R.bias e0b)
    (h1W : approxT (α := R) (toSpec := toSpec) L1S.weights L1R.weights e1W)
    (h1b : approxT (α := R) (toSpec := toSpec) L1S.bias L1R.bias e1b)
    (h2W : approxT (α := R) (toSpec := toSpec) L2S.weights L2R.weights e2W)
    (h2b : approxT (α := R) (toSpec := toSpec) L2S.bias L2R.bias e2b)
    (hx : approxT (α := R) (toSpec := toSpec) xS xR ex) :
    ∃ eps : ℝ,
      approxT (α := R) (toSpec := toSpec)
        (let z0 := Spec.linearSpec (α := ℝ) L0S xS
         let a0 := Activation.tanhSpec (α := ℝ) z0
         let z1 := Spec.linearSpec (α := ℝ) L1S a0
         let a1 := Activation.tanhSpec (α := ℝ) z1
         Spec.linearSpec (α := ℝ) L2S a1)
        (let z0 := Spec.linearSpec (α := R) L0R xR
         let a0 := Activation.tanhSpec (α := R) z0
         let z1 := Spec.linearSpec (α := R) L1R a0
         let a1 := Activation.tanhSpec (α := R) z1
         Spec.linearSpec (α := R) L2R a1)
        eps := by
  -- First linear layer: propagate input/weight/bias error to the first pre-activation.
  rcases approxT_linear_fp32
    (WS := L0S) (WR := L0R) (xS := xS) (xR := xR)
    (epsW := e0W) (epsb := e0b) (epsx := ex) h0W h0b hx with ⟨eZ0, hZ0⟩
  have hA0 :=
    Proofs.RuntimeApprox.NFBackend.approxT_tanh_spec
      (β := β) (fexp := fexp) (rnd := rnd) (s := Shape.dim d1 .scalar)
      (xS := Spec.linearSpec (α := ℝ) L0S xS)
      (xR := Spec.linearSpec (α := R) L0R xR)
      (eps := eZ0) hZ0
  let eA0 : ℝ :=
    linfNorm (Proofs.RuntimeApprox.NFBackend.tanhBoundTensor
      (β := β) (fexp := fexp) (rnd := rnd) (s := Shape.dim d1 .scalar) eZ0
      (Spec.linearSpec (α := R) L0R xR))
  have hA0' :
      approxT (α := R) (toSpec := toSpec)
        (Activation.tanhSpec (α := ℝ) (Spec.linearSpec (α := ℝ) L0S xS))
        (Activation.tanhSpec (α := R) (Spec.linearSpec (α := R) L0R xR))
        eA0 := by
    simpa [Activation.tanhSpec, eA0] using hA0

  -- Second linear layer: use the tanh activation bound as this layer's input bound.
  rcases approxT_linear_fp32
    (WS := L1S) (WR := L1R)
    (xS := Activation.tanhSpec (α := ℝ) (Spec.linearSpec (α := ℝ) L0S xS))
    (xR := Activation.tanhSpec (α := R) (Spec.linearSpec (α := R) L0R xR))
    (epsW := e1W) (epsb := e1b) (epsx := eA0) h1W h1b hA0' with ⟨eZ1, hZ1⟩
  have hA1 :=
    Proofs.RuntimeApprox.NFBackend.approxT_tanh_spec
      (β := β) (fexp := fexp) (rnd := rnd) (s := Shape.dim d2 .scalar)
      (xS := Spec.linearSpec (α := ℝ) L1S (Activation.tanhSpec (α := ℝ) (Spec.linearSpec (α := ℝ)
        L0S xS)))
      (xR := Spec.linearSpec (α := R) L1R (Activation.tanhSpec (α := R) (Spec.linearSpec (α := R)
        L0R xR)))
      (eps := eZ1) hZ1
  let eA1 : ℝ :=
    linfNorm (Proofs.RuntimeApprox.NFBackend.tanhBoundTensor
      (β := β) (fexp := fexp) (rnd := rnd) (s := Shape.dim d2 .scalar) eZ1
      (Spec.linearSpec (α := R) L1R (Activation.tanhSpec (α := R) (Spec.linearSpec (α := R) L0R
        xR))))
  have hA1' :
      approxT (α := R) (toSpec := toSpec)
        (Activation.tanhSpec (α := ℝ)
          (Spec.linearSpec (α := ℝ) L1S
            (Activation.tanhSpec (α := ℝ) (Spec.linearSpec (α := ℝ) L0S xS))))
        (Activation.tanhSpec (α := R)
          (Spec.linearSpec (α := R) L1R
            (Activation.tanhSpec (α := R) (Spec.linearSpec (α := R) L0R xR))))
        eA1 := by
    simpa [Activation.tanhSpec, eA1] using hA1

  -- Final linear layer: produces the network-level output approximation.
  rcases approxT_linear_fp32
    (WS := L2S) (WR := L2R)
    (xS := Activation.tanhSpec (α := ℝ)
      (Spec.linearSpec (α := ℝ) L1S
        (Activation.tanhSpec (α := ℝ) (Spec.linearSpec (α := ℝ) L0S xS))))
    (xR := Activation.tanhSpec (α := R)
      (Spec.linearSpec (α := R) L1R
        (Activation.tanhSpec (α := R) (Spec.linearSpec (α := R) L0R xR))))
    (epsW := e2W) (epsb := e2b) (epsx := eA1) h2W h2b hA1' with ⟨eOut, hOut⟩

  exact ⟨eOut, by
    simpa using hOut⟩

/-!
## 2-layer ReLU MLP

This is the FP32 analogue of the 2-layer ReLU MLP used by CROWN/IBP:
`Linear → ReLU → Linear`.

Note: the runtime ReLU here is the *rounded* variant `reluR` used by the NFBackend forward
approximation framework (apply `max · 0` in ℝ, then round once).
-/

/--
Compositional FP32 approximation theorem for a 2-layer ReLU MLP:

`Linear → ReLU → Linear`.

This is the network-level bound consumed by the CROWN/IBP integration in
`NN.Proofs.RuntimeApprox.FP32.CROWN`.
-/
theorem approxT_reluMlp2_fp32 {d0 d1 d2 : Nat}
    {L0S : LinearSpec ℝ d0 d1} {L1S : LinearSpec ℝ d1 d2}
    {L0R : LinearSpec R d0 d1} {L1R : LinearSpec R d1 d2}
    {xS : SpecTensor (.dim d0 .scalar)} {xR : Tensor R (.dim d0 .scalar)}
    {e0W e0b e1W e1b ex : ℝ}
    (h0W : approxT (α := R) (toSpec := toSpec) L0S.weights L0R.weights e0W)
    (h0b : approxT (α := R) (toSpec := toSpec) L0S.bias L0R.bias e0b)
    (h1W : approxT (α := R) (toSpec := toSpec) L1S.weights L1R.weights e1W)
    (h1b : approxT (α := R) (toSpec := toSpec) L1S.bias L1R.bias e1b)
    (hx : approxT (α := R) (toSpec := toSpec) xS xR ex) :
    ∃ eps : ℝ,
      approxT (α := R) (toSpec := toSpec)
        (let z0 := Spec.linearSpec (α := ℝ) L0S xS
         let a0 := Activation.reluSpec (α := ℝ) z0
         Spec.linearSpec (α := ℝ) L1S a0)
        (let z0 := Spec.linearSpec (α := R) L0R xR
         let a0 := mapSpec (reluR (β := β) (fexp := fexp) (rnd := rnd)) z0
         Spec.linearSpec (α := R) L1R a0)
        eps := by
  -- First linear layer: real/FP32 pre-activations are close.
  rcases approxT_linear_fp32
    (WS := L0S) (WR := L0R) (xS := xS) (xR := xR)
    (epsW := e0W) (epsb := e0b) (epsx := ex) h0W h0b hx with ⟨eZ0, hZ0⟩

  -- ReLU: the NF backend supplies a rounded-ReLU bound from the pre-activation bound.
  have hA0 :=
    Proofs.RuntimeApprox.NFBackend.approxT_relu_spec
      (β := β) (fexp := fexp) (rnd := rnd) (s := Shape.dim d1 .scalar)
      (xS := Spec.linearSpec (α := ℝ) L0S xS)
      (xR := Spec.linearSpec (α := R) L0R xR)
      (eps := eZ0) hZ0
  let eA0 : ℝ :=
    linfNorm (Proofs.RuntimeApprox.NFBackend.reluBoundTensor
      (β := β) (fexp := fexp) (rnd := rnd) (s := Shape.dim d1 .scalar) eZ0
      (Spec.linearSpec (α := R) L0R xR))
  have hA0' :
      approxT (α := R) (toSpec := toSpec)
        (Activation.reluSpec (α := ℝ) (Spec.linearSpec (α := ℝ) L0S xS))
        (mapSpec (reluR (β := β) (fexp := fexp) (rnd := rnd)) (Spec.linearSpec (α := R) L0R xR))
        eA0 := by
    -- `relu_spec` is `map_spec (max · 0)` on ℝ.
    simpa [Activation.reluSpec, Activation.Math.reluSpec, eA0] using hA0

  -- Final linear layer: propagate the activation error to the network output.
  rcases approxT_linear_fp32
    (WS := L1S) (WR := L1R)
    (xS := Activation.reluSpec (α := ℝ) (Spec.linearSpec (α := ℝ) L0S xS))
    (xR := mapSpec (reluR (β := β) (fexp := fexp) (rnd := rnd)) (Spec.linearSpec (α := R) L0R xR))
    (epsW := e1W) (epsb := e1b) (epsx := eA0) h1W h1b hA0' with ⟨eOut, hOut⟩

  exact ⟨eOut, by simpa using hOut⟩

end

end NN.Proofs.RuntimeApprox.FP32
