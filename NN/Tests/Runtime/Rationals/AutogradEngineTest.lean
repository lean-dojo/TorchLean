/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Train
public import NN.Spec.Models.Mlp
public import NN.Tensor
public import NN.Tests.Utils
public import NN.Spec.Core.Context.Rational
import Mathlib.Algebra.Order.Ring.Unbundled.Rat

/-!
# AutogradEngineTest

Regression tests for `Runtime.Autograd` dynamic tape over `ℚ`.

We check that for a simple 2-layer MLP, the tape-based gradients match the existing
hand-derived `Examples.mlpBackward`.
-/

open scoped Spec.RationalAlgebraic

@[expose] public section


open Spec TorchLean
open TorchLean TorchLean.Tensor
open Examples

namespace Tests
namespace Rationals
namespace AutogradEngine

open Runtime.Autograd

abbrev inDim  := 2
abbrev hidDim := 3
abbrev outDim := 1

-- Small tag used for readable error messages.
abbrev tag : String := "autograd_engine_test (Rat)"

-- The parameter-id record is shared with the `Float` transpose of this test; see `Tests.Utils`.
open Tests.Utils (ParamIds)

/-!
## Fixed inputs and parameters

We use a small deterministic 2-layer MLP so the gradients are stable.
-/
def hiddenWeight : Tensor ℚ [hidDim, inDim] :=
  (Tensor.from #[(1 : ℚ) / 10, 2 / 10, 3 / 10, 4 / 10, 5 / 10, 6 / 10]).reshape
    [hidDim, inDim] (by dsimp; decide)

def hiddenBias : Tensor ℚ [hidDim] :=
  (Tensor.from #[(1 : ℚ) / 10, 2 / 10, 3 / 10]).reshape [hidDim] (by dsimp; decide)

def outputWeight : Tensor ℚ [outDim, hidDim] :=
  (Tensor.from #[(7 : ℚ) / 10, 8 / 10, 9 / 10]).reshape
    [outDim, hidDim] (by dsimp; decide)

def outputBias : Tensor ℚ [outDim] :=
  (Tensor.from #[(4 : ℚ) / 10]).reshape [outDim] (by dsimp; decide)

def x : Tensor ℚ [inDim] :=
  (Tensor.from #[(5 : ℚ) / 10, 8 / 10]).reshape [inDim] (by dsimp; decide)

def dLdy : Tensor ℚ [outDim] :=
  (Tensor.from #[(1 : ℚ)]).reshape [outDim] (by dsimp; decide)

def hiddenLayer : Spec.LinearSpec ℚ inDim hidDim := { weights := hiddenWeight, bias := hiddenBias }
def outputLayer : Spec.LinearSpec ℚ hidDim outDim := { weights := outputWeight, bias := outputBias }

def expected :=
  Examples.mlpBackward hiddenLayer outputLayer x dLdy

/-!
## Test: dynamic tape gradients vs. reference

We compare the autograd tape gradients against the hand-derived MLP backward pass.
-/
def checkMlpGrads :
  Runtime.Autograd.Result Bool := do
  let t0 : Tape ℚ := Tape.empty

  -- Build the graph in TapeM for readability.
  let m : TapeM ℚ _ := do
    let hiddenWeightId ← Train.TapeM.param hiddenWeight (name := some "hiddenWeight")
    let hiddenBiasId ← Train.TapeM.param hiddenBias (name := some "hiddenBias")
    let outputWeightId ← Train.TapeM.param outputWeight (name := some "outputWeight")
    let outputBiasId ← Train.TapeM.param outputBias (name := some "outputBias")
    let xId ← Train.TapeM.const x (name := some "x")

    -- Forward pass: linear -> relu -> linear
    let z1Id ← TapeM.linear (inDim:=inDim) (outDim:=hidDim) hiddenWeightId hiddenBiasId xId
    let a1Id ← TapeM.relu (s := [hidDim]) z1Id
    let yId ← TapeM.linear (inDim:=hidDim) (outDim:=outDim) outputWeightId outputBiasId a1Id

    let t ← TapeM.getTape
    let grads ← liftM (Tape.backward (t:=t) yId (Spec.SomeTensor.ofTensor dLdy))

    let ids : ParamIds :=
      { hiddenWeightId := hiddenWeightId, hiddenBiasId := hiddenBiasId,
        outputWeightId := outputWeightId, outputBiasId := outputBiasId }
    pure (ids, grads)

  let ((ids, grads), _) ← TapeM.run t0 m

  let (dW1_exp, db1_exp, dW2_exp, db2_exp, _dX_exp) := expected

  let dW1_dyn ← Train.requireGradTensor (tag := tag)
    (s := [hidDim, inDim]) grads ids.hiddenWeightId
  let db1_dyn ← Train.requireGradTensor (tag := tag)
    (s := [hidDim]) grads ids.hiddenBiasId
  let dW2_dyn ← Train.requireGradTensor (tag := tag)
    (s := [outDim, hidDim]) grads ids.outputWeightId
  let db2_dyn ← Train.requireGradTensor (tag := tag)
    (s := [outDim]) grads ids.outputBiasId

  let ok1 := decide (pretty dW1_dyn = pretty dW1_exp)
  let ok2 := decide (pretty db1_dyn = pretty db1_exp)
  let ok3 := decide (pretty dW2_dyn = pretty dW2_exp)
  let ok4 := decide (pretty db2_dyn = pretty db2_exp)
  pure (ok1 && ok2 && ok3 && ok4)

def run : IO Unit := do
  match checkMlpGrads with
  | .ok true => IO.println "autograd_engine_test (Rat): OK"
  | .ok false => throw <| IO.userError "autograd_engine_test (Rat): FAILED"
  | .error msg => throw <| IO.userError s!"autograd_engine_test (Rat): {msg}"

end AutogradEngine
end Rationals
end Tests
