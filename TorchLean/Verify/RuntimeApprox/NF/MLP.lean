/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.RuntimeApprox.NF.Layers
public import TorchLean.Verify.RuntimeApprox.NF.Elementwise.Unary

/-!
# MLPs in the NF model

This module composes the linear-layer bound of `TorchLean.Verify.RuntimeApprox.NF.Layers` with the
tanh and ReLU bounds into forward-error bounds for two small networks over `NF β fexp rnd`. They
are the three-layer tanh MLP `Linear → tanh → Linear → tanh → Linear` and the two-layer ReLU MLP
`Linear → ReLU → Linear`.

At a coordinate `a` of its rounded input, tanh contributes `min e 2 + ulp (tanh a) / 2` for an
incoming error `e`. The two exact values differ by at most `e` because tanh is 1-Lipschitz, and by
at most 2 because they lie in `[-1, 1]`. The result rounds once. The budget of the tanh MLP
therefore depends on the errors of all three layers and of the input. ReLU adds one rounding to its
incoming error, since `max · 0` is 1-Lipschitz.

The main declarations are `approxTensor_tanhMLP3` and `approxTensor_reluTwoLayerMLP`, with their
budgets `tanhMLP3ErrorBudget` and `reluTwoLayerMLPErrorBudget`.
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

/-! ## Three-layer tanh MLP -/

/-- The forward-error bound of the rounded `Linear → tanh → Linear → tanh → Linear`. Each tanh
turns the bound `e` of its pre-activation `z` into `linfNorm (tanhBoundTensor e z)` (entries
`min e 2 + ulp (tanh a) / 2`), the input error of the next linear layer. -/
def tanhMLP3ErrorBudget {d0 d1 d2 d3 : Nat}
    (e0W e0b e1W e1b e2W e2b ex : ℝ)
    (L0R : Linear R d0 d1) (L1R : Linear R d1 d2)
    (L2R : Linear R d2 d3) (xR : Tensor R [d0]) : ℝ :=
  let z0R := Linear.forward (α := R) L0R xR
  let eZ0 := linearErrorBudget e0W e0b ex L0R xR
  let a0R := Tensor.map Numerics.MathFunctions.tanh z0R
  let eA0 := linfNorm (tanhBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
    (s := Shape.dim d1 .scalar) eZ0 z0R)
  let z1R := Linear.forward (α := R) L1R a0R
  let eZ1 := linearErrorBudget e1W e1b eA0 L1R a0R
  let a1R := Tensor.map Numerics.MathFunctions.tanh z1R
  let eA1 := linfNorm (tanhBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
    (s := Shape.dim d2 .scalar) eZ1 z1R)
  linearErrorBudget e2W e2b eA1 L2R a1R

/-- If the weights, biases and input of `Linear → tanh → Linear → tanh → Linear` approximate the
real ones within `e0W`, …, `ex` (`h0W`, …, `hx`), the rounded network approximates the real one
within `tanhMLP3ErrorBudget`. -/
theorem approxTensor_tanhMLP3 {d0 d1 d2 d3 : Nat}
    {L0S : Linear ℝ d0 d1} {L1S : Linear ℝ d1 d2} {L2S : Linear ℝ d2 d3}
    {L0R : Linear R d0 d1} {L1R : Linear R d1 d2} {L2R : Linear R d2 d3}
    {xS : Tensor ℝ [d0]} {xR : Tensor R [d0]}
    {e0W e0b e1W e1b e2W e2b ex : ℝ}
    (h0W : ApproxTensor (α := R) (toSpec := NF.toReal)
      L0S.weights L0R.weights e0W)
    (h0b : ApproxTensor (α := R) (toSpec := NF.toReal)
      L0S.bias L0R.bias e0b)
    (h1W : ApproxTensor (α := R) (toSpec := NF.toReal)
      L1S.weights L1R.weights e1W)
    (h1b : ApproxTensor (α := R) (toSpec := NF.toReal)
      L1S.bias L1R.bias e1b)
    (h2W : ApproxTensor (α := R) (toSpec := NF.toReal)
      L2S.weights L2R.weights e2W)
    (h2b : ApproxTensor (α := R) (toSpec := NF.toReal)
      L2S.bias L2R.bias e2b)
    (hx : ApproxTensor (α := R) (toSpec := NF.toReal)
      xS xR ex) :
    ApproxTensor (α := R) (toSpec := NF.toReal)
      (let z0 := Linear.forward (α := ℝ) L0S xS
       let a0 := Tensor.map Numerics.MathFunctions.tanh z0
       let z1 := Linear.forward (α := ℝ) L1S a0
       let a1 := Tensor.map Numerics.MathFunctions.tanh z1
       Linear.forward (α := ℝ) L2S a1)
      (let z0 := Linear.forward (α := R) L0R xR
       let a0 := Tensor.map Numerics.MathFunctions.tanh z0
       let z1 := Linear.forward (α := R) L1R a0
       let a1 := Tensor.map Numerics.MathFunctions.tanh z1
       Linear.forward (α := R) L2R a1)
      (tanhMLP3ErrorBudget e0W e0b e1W e1b e2W e2b ex L0R L1R L2R xR) := by
  have hZ0 := approxTensor_linear (WS := L0S) (WR := L0R) (xS := xS) (xR := xR)
    (epsW := e0W) (epsb := e0b) (epsx := ex) h0W h0b hx
  have hA0 := ApproxTensor.tanh (β := β) (fexp := fexp) (rnd := rnd)
    (s := Shape.dim d1 .scalar) hZ0
  have hZ1 := approxTensor_linear (WS := L1S) (WR := L1R) (epsW := e1W) (epsb := e1b)
    h1W h1b hA0
  have hA1 := ApproxTensor.tanh (β := β) (fexp := fexp) (rnd := rnd)
    (s := Shape.dim d2 .scalar) hZ1
  exact approxTensor_linear (WS := L2S) (WR := L2R) (epsW := e2W) (epsb := e2b) h2W h2b hA1

/-! ## Two-layer ReLU MLP

The network `Linear → ReLU → Linear` of the CROWN/IBP development, with the rounded ReLU `reluR`:
`max · 0` over the reals, then one rounding.
-/

/-- The forward-error bound of the rounded `Linear → reluR → Linear`. -/
def reluTwoLayerMLPErrorBudget {d0 d1 d2 : Nat}
    (e0W e0b e1W e1b ex : ℝ)
    (L0R : Linear R d0 d1) (L1R : Linear R d1 d2)
    (xR : Tensor R [d0]) : ℝ :=
  let z0R := Linear.forward (α := R) L0R xR
  let eZ0 := linearErrorBudget e0W e0b ex L0R xR
  let a0R := Tensor.map (reluR (β := β) (fexp := fexp) (rnd := rnd)) z0R
  let eA0 := linfNorm (reluBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
    (s := Shape.dim d1 .scalar) eZ0 z0R)
  linearErrorBudget e1W e1b eA0 L1R a0R

/-- If the weights, biases and input of `Linear → ReLU → Linear` approximate the real ones within
`e0W`, …, `ex` (`h0W`, …, `hx`), the rounded network with `reluR` approximates the real network
with `max · 0` within `reluTwoLayerMLPErrorBudget`. -/
theorem approxTensor_reluTwoLayerMLP {d0 d1 d2 : Nat}
    {L0S : Linear ℝ d0 d1} {L1S : Linear ℝ d1 d2}
    {L0R : Linear R d0 d1} {L1R : Linear R d1 d2}
    {xS : Tensor ℝ [d0]} {xR : Tensor R [d0]}
    {e0W e0b e1W e1b ex : ℝ}
    (h0W : ApproxTensor (α := R) (toSpec := NF.toReal)
      L0S.weights L0R.weights e0W)
    (h0b : ApproxTensor (α := R) (toSpec := NF.toReal)
      L0S.bias L0R.bias e0b)
    (h1W : ApproxTensor (α := R) (toSpec := NF.toReal)
      L1S.weights L1R.weights e1W)
    (h1b : ApproxTensor (α := R) (toSpec := NF.toReal)
      L1S.bias L1R.bias e1b)
    (hx : ApproxTensor (α := R) (toSpec := NF.toReal)
      xS xR ex) :
    ApproxTensor (α := R) (toSpec := NF.toReal)
      (let z0 := Linear.forward (α := ℝ) L0S xS
       let a0 := Tensor.map (fun x => max x 0) z0
       Linear.forward (α := ℝ) L1S a0)
      (let z0 := Linear.forward (α := R) L0R xR
       let a0 := Tensor.map (reluR (β := β) (fexp := fexp) (rnd := rnd)) z0
       Linear.forward (α := R) L1R a0)
      (reluTwoLayerMLPErrorBudget e0W e0b e1W e1b ex L0R L1R xR) := by
  have hZ0 := approxTensor_linear (WS := L0S) (WR := L0R) (xS := xS) (xR := xR)
    (epsW := e0W) (epsb := e0b) (epsx := ex) h0W h0b hx
  have hA0 := ApproxTensor.relu (β := β) (fexp := fexp) (rnd := rnd)
    (s := Shape.dim d1 .scalar) hZ0
  exact approxTensor_linear (WS := L1S) (WR := L1R) (epsW := e1W) (epsb := e1b) h1W h1b hA0

end
end NF
end TorchLean
