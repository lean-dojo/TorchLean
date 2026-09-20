/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Models.Mlp
public import NN.Spec.Core.Context.Rational

/-!
# Closed MLP Model Facts

Small rational MLP examples belong in the proof layer. The forward composition law is already
proved generally by `Examples.mlp_spec_forward_eq`; the facts below state a deterministic backward
calculation as coordinate-level Lean theorems.
-/

open scoped Spec.RationalAlgebraic

@[expose] public section

namespace NN.Proofs.Models.Mlp

open Spec TorchLean
open TorchLean.Tensor
open Examples
open Spec.Module

/-- Input width of the worked example: two features. -/
abbrev exInDim  := 2

/-- Hidden width of the worked example: three ReLU units. -/
abbrev exHidDim := 3

/-- Output width of the worked example: one scalar prediction. -/
abbrev exOutDim := 1

/-- Hidden weight matrix, with entries chosen as small rationals.

Everything in this file is over `ℚ`, so the numbers below are exact and the examples at the end of
the file close by `simp` alone. The same computation in `Float` would only be true up to rounding,
which is the point of keeping the closed-form check separate from the runtime. -/
def exampleHiddenWeight : TorchLean.Tensor ℚ [exHidDim, exInDim] :=
  TorchLean.Tensor.matrix (fun i j =>
    match i.val, j.val with
    | 0, 0 => 1 / 10
    | 0, 1 => 1 / 5
    | 1, 0 => 3 / 10
    | 1, 1 => 2 / 5
    | 2, 0 => 1 / 2
    | 2, 1 => 3 / 5
    | _, _ => 0)

/-- Hidden bias of the worked example. -/
def exampleHiddenBias : TorchLean.Tensor ℚ [exHidDim] :=
  TorchLean.Tensor.ofFn (fun i =>
    match i.val with
    | 0 => 1 / 10
    | 1 => 1 / 5
    | 2 => 3 / 10
    | _ => 0)

/-- Output weight row of the worked example. All entries are positive, so the single output unit is
active and the backward pass exercises the interesting branch of `reluDerivSpec`. -/
def exampleOutputWeight : TorchLean.Tensor ℚ [exOutDim, exHidDim] :=
  TorchLean.Tensor.matrix (fun _ j =>
    match j.val with
    | 0 => 7 / 10
    | 1 => 4 / 5
    | 2 => 9 / 10
    | _ => 0)

/-- Output bias of the worked example. -/
def exampleOutputBias : TorchLean.Tensor ℚ [exOutDim] :=
  TorchLean.Tensor.ofFn (fun _ => 2 / 5)

/-- The hidden layer, packaging its weight and bias. -/
def exampleHiddenLayer : Spec.LinearSpec ℚ exInDim exHidDim :=
  { weights := exampleHiddenWeight, bias := exampleHiddenBias }

/-- The output layer, packaging its weight and bias. -/
def exampleOutputLayer : Spec.LinearSpec ℚ exHidDim exOutDim :=
  { weights := exampleOutputWeight, bias := exampleOutputBias }

/-- The single input vector used by every example below. -/
def exInput : TorchLean.Tensor ℚ [exInDim] :=
  TorchLean.Tensor.ofFn (fun i =>
    match i.val with
    | 0 => 1 / 2
    | 1 => 4 / 5
    | _ => 0)

/-- The two-layer network as a `Chain`, built by the general `mlpSpec` constructor. -/
def exNet : Spec.Module.Chain ℚ (.dim exInDim .scalar) (.dim exOutDim .scalar) :=
  Examples.mlpSpec (α := ℚ) exampleHiddenLayer exampleOutputLayer

/-- Output obtained by running the chain, that is, by the generic module forward pass. -/
def exOutput : TorchLean.Tensor ℚ [exOutDim] :=
  Spec.Module.Chain.forward (α := ℚ) exNet exInput

/-- The same output written out by hand: linear, ReLU, linear.

Stating both and proving them equal is the whole point. It checks that `Chain.forward` really does
compose the layers in the order a reader would expect, rather than only that it type-checks. -/
def exExpected : TorchLean.Tensor ℚ [exOutDim] :=
  let z1 := Spec.linearSpec (α := ℚ) exampleHiddenLayer exInput
  let a1 := Activation.reluSpec z1
  Spec.linearSpec (α := ℚ) exampleOutputLayer a1

/-- Incoming cotangent, taken to be all ones so the backward pass reads off plain derivatives. -/
def exDLdy : TorchLean.Tensor ℚ [exOutDim] :=
  TorchLean.Tensor.ofFn (fun _ => 1)

/-- Gradients from the hand-written backward pass: weight, bias and input gradients per layer. -/
def exGrad :=
  Examples.mlpBackward (α := ℚ) exampleHiddenLayer exampleOutputLayer exInput exDLdy

/--
Input gradient from the `OpSpec` backward pass, which composes per-operation adjoints instead.
-/
def exDXOpspec :=
  Examples.mlpOpspecBackward (α := ℚ) exampleHiddenLayer exampleOutputLayer exInput exDLdy

/-- Input gradient projected out of `exGrad`, so the two routes can be compared directly. -/
def dXHand : TorchLean.Tensor ℚ [exInDim] :=
  match exGrad with
  | (_, _, _, _, dX) => dX

example :
    exOutput = exExpected := by
  simpa [exOutput, exExpected, exNet] using
    (Examples.mlp_spec_forward_eq (α := ℚ) exampleHiddenLayer exampleOutputLayer exInput)

example :
    dXHand = exDXOpspec := by
  simp [dXHand, exGrad, exDXOpspec, Examples.mlpBackward, Examples.mlpOpspecBackward,
    Examples.mlpOpspec, Spec.OpSpec.compose, Spec.linearOp, Spec.reluOp,
    Spec.liftElementwiseBackward, Spec.liftElementwise, Activation.reluDerivSpec]

end NN.Proofs.Models.Mlp
