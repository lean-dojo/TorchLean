/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Context
public import NN.Spec.Core.Tensor
public import NN.Spec.Core.Utils
public import NN.Spec.Models.Mlp
public import NN.Spec.Module.Linear
public import NN.Entrypoint.Tensor
public import Batteries.Data.Rat.Float

/-!
# MlpTest

 Rational-arithmetic tests for the MLP specification. -/

open scoped NN.Spec.RationalAlgebraic

@[expose] public section

open Spec
open Tensor
open Examples
open ModSpec
namespace Tests
namespace Rationals

-- ============================================================================
-- Small 2-layer MLP with (inDim = 2, hidDim = 3, outDim = 1)
-- ============================================================================

/-- Abbreviations for the layer dimensions. -/
abbrev exInDim  := 2
abbrev exHidDim := 3
abbrev exOutDim := 1

/-- Helper tensors for layer-1 parameters. -/
def exW1 : Tensor ℚ (.dim exHidDim (.dim exInDim .scalar)) :=
  tensorND! [exHidDim, exInDim] [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

def exb1 : Tensor ℚ (.dim exHidDim .scalar) :=
  tensorND! [exHidDim] [0.1, 0.2, 0.3]

/-- Layer-2 parameters. Weight matrix is [outDim × hidDim]. -/
def exW2 : Tensor ℚ (.dim exOutDim (.dim exHidDim .scalar)) :=
  tensorND! [exOutDim, exHidDim] [0.7, 0.8, 0.9]

def exb2 : Tensor ℚ (.dim exOutDim .scalar) :=
  tensorND! [exOutDim] [0.4]

/-- Assemble `LinearSpec`s. -/
def exLin1 : Spec.LinearSpec ℚ exInDim exHidDim :=
{ weights := exW1, bias := exb1 }

def exLin2 : Spec.LinearSpec ℚ exHidDim exOutDim :=
{ weights := exW2, bias := exb2 }

/-- Input vector `[0.5, 0.8]`. -/
def exInput : Tensor ℚ (.dim exInDim .scalar) :=
  tensorND! [exInDim] [0.5, 0.8]

/-- Build the MLP SpecChain and run the forward pass. -/
def exNet : SpecChain ℚ (.dim exInDim .scalar) (.dim exOutDim .scalar) :=
  Examples.mlpSpec (α:=ℚ) exLin1 exLin2

def exOutput : Tensor ℚ (.dim exOutDim .scalar) :=
  SpecChain.forward (α:=ℚ) exNet exInput

/-- Manually compute the expected output to confirm composition correctness. -/
def exExpected : Tensor ℚ (.dim exOutDim .scalar) :=
  let z1 := Spec.linearSpec (α:=ℚ) exLin1 exInput
  let a1 := Activation.reluSpec z1
  Spec.linearSpec (α:=ℚ) exLin2 a1

-- Gradient verification -----------------------------------------------------

def exDLdy : Tensor ℚ (.dim exOutDim .scalar) :=
  tensorND! [exOutDim] [1.0]

def exGrad := Examples.mlpBackward (α:=ℚ) exLin1 exLin2 exInput exDLdy

def exDXOpspec :=
  Examples.mlpOpspecBackward (α:=ℚ) exLin1 exLin2 exInput exDLdy

/-- Extract ∂L/∂x from the 5-tuple returned by `mlp_backward`. -/
def dXHand : Tensor ℚ (.dim exInDim .scalar) :=
  match exGrad with
  | (_, _, _, _, dX) => dX

-- ============================================================================
-- Simple training loop (SGD + MSE) for the example MLP
-- Trains on the single sample xInput → yTarget = 1.0 for 10 epochs.
-- ============================================================================

def lr : ℚ := 0.1
def yTarget : Tensor ℚ (.dim exOutDim .scalar) :=
  tensorND! [exOutDim] [1.0]

/-- One SGD update step returning new layer specs. -/
def sgdStep
  (l1 : Spec.LinearSpec ℚ exInDim exHidDim)
  (l2 : Spec.LinearSpec ℚ exHidDim exOutDim) :
  Spec.LinearSpec ℚ exInDim exHidDim × Spec.LinearSpec ℚ exHidDim exOutDim :=
  let yPred := Examples.mlpForward (α:=ℚ) l1 l2 exInput
  -- PyTorch MSELoss: grad = 2 * (ŷ - y) / N, here N = 1
  let diff  := Tensor.scaleSpec (Tensor.subSpec yPred yTarget) (2.0 : ℚ)
  let (dW1, db1, dW2, db2, _) :=
    Examples.mlpBackward (α:=ℚ) l1 l2 exInput diff
  -- SGD update: param ← param - lr * grad
  let newW1 := Tensor.subSpec l1.weights (Tensor.scaleSpec dW1 lr)
  let newb1 := Tensor.subSpec l1.bias    (Tensor.scaleSpec db1 lr)
  let newW2 := Tensor.subSpec l2.weights (Tensor.scaleSpec dW2 lr)
  let newb2 := Tensor.subSpec l2.bias    (Tensor.scaleSpec db2 lr)
  ({ weights := newW1, bias := newb1 }, { weights := newW2, bias := newb2 })

/-- Train for `n` epochs (tail-recursive). -/
def trainN : Nat → Spec.LinearSpec ℚ exInDim exHidDim → Spec.LinearSpec ℚ exHidDim exOutDim →
  Spec.LinearSpec ℚ exInDim exHidDim × Spec.LinearSpec ℚ exHidDim exOutDim
| 0, l1, l2 => (l1, l2)
| Nat.succ k, l1, l2 =>
  let (l1', l2') := sgdStep l1 l2
  trainN k l1' l2'

def finalPair := trainN 4 exLin1 exLin2
def finalL1 : Spec.LinearSpec ℚ exInDim exHidDim := finalPair.fst
def finalL2 : Spec.LinearSpec ℚ exHidDim exOutDim := finalPair.snd

def yAfterTrain := Examples.mlpForward (α:=ℚ) finalL1 finalL2 exInput

def ratToFloatString (q : ℚ) : String :=
  toString (Rat.toFloat q)

def prettyRatVecApprox {n : Nat} (x : Tensor ℚ (.dim n .scalar)) : String :=
  match x with
  | Tensor.dim values =>
      "[" ++ String.intercalate ", "
        ((List.finRange n).map (fun i => ratToFloatString (Tensor.toScalar (values i)))) ++ "]"

def run : IO Unit := do
  IO.println "=== Rationals MLP Test ==="
  IO.println s!"Output: {pretty exOutput}"
  IO.println s!"Expected: {pretty exExpected}"
  let forwardOk := decide (pretty exOutput = pretty exExpected)
  if forwardOk then
    IO.println "Forward pass matches manual computation."
  else
    throw <| IO.userError "Forward pass mismatch (Rationals MLP test)."

  IO.println s!"dX (hand): {pretty dXHand}"
  IO.println s!"dX (opspec): {pretty exDXOpspec}"
  let gradOk := decide (pretty dXHand = pretty exDXOpspec)
  if gradOk then
    IO.println "Backward dX matches opspec."
  else
    throw <| IO.userError "Backward dX mismatch (Rationals MLP test)."

  IO.println s!"Output after 4 epochs (approx): {prettyRatVecApprox yAfterTrain}"
  IO.println s!"Prediction error (approx, ŷ - y): {prettyRatVecApprox (Tensor.subSpec yAfterTrain yTarget)}"
  IO.println s!"Initial output: {pretty (Examples.mlpForward (α:=ℚ) exLin1 exLin2 exInput)}"
  IO.println s!"SpecChain forward: {pretty (SpecChain.forward (α:=ℚ) exNet exInput)}"

end Rationals
end Tests
/-!
Rational-mode MLP regression tests.

This file exercises a small MLP end-to-end in the rational context to catch API/semantics mismatches.
-/
