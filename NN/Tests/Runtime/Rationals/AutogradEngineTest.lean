/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Utils
public import NN.Spec.Models.Mlp
public import NN.Entrypoint.Tensor

/-!
# AutogradEngineTest

Regression tests for `Runtime.Autograd` dynamic tape over `ℚ`.

We check that for a simple 2-layer MLP, the tape-based gradients match the existing
hand-derived `Examples.mlp_backward`.
-/

open scoped NN.Spec.RationalAlgebraic

@[expose] public section


open Spec
open Tensor
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

-- Parameter node ids we want to read gradients for.
structure ParamIds where
  /-- w 1 Id. -/
  w1Id : Nat
  /-- b 1 Id. -/
  b1Id : Nat
  /-- w 2 Id. -/
  w2Id : Nat
  /-- b 2 Id. -/
  b2Id : Nat

/-!
## Fixed inputs and parameters

We use a small deterministic 2-layer MLP so the gradients are stable.
-/
def W1 : Tensor ℚ (.dim hidDim (.dim inDim .scalar)) :=
  tensorND! [hidDim, inDim] [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

def b1 : Tensor ℚ (.dim hidDim .scalar) :=
  tensorND! [hidDim] [0.1, 0.2, 0.3]

def W2 : Tensor ℚ (.dim outDim (.dim hidDim .scalar)) :=
  tensorND! [outDim, hidDim] [0.7, 0.8, 0.9]

def b2 : Tensor ℚ (.dim outDim .scalar) :=
  tensorND! [outDim] [0.4]

def x : Tensor ℚ (.dim inDim .scalar) :=
  tensorND! [inDim] [0.5, 0.8]

def dLdy : Tensor ℚ (.dim outDim .scalar) :=
  tensorND! [outDim] [1.0]

def layer1 : Spec.LinearSpec ℚ inDim hidDim := { weights := W1, bias := b1 }
def layer2 : Spec.LinearSpec ℚ hidDim outDim := { weights := W2, bias := b2 }

def expected :=
  Examples.mlpBackward layer1 layer2 x dLdy

/-!
## Test: dynamic tape gradients vs. reference

We compare the autograd tape gradients against the hand-derived MLP backward pass.
-/
def checkMlpGrads :
  Runtime.Autograd.Result Bool := do
  let t0 : Tape ℚ := Tape.empty

  -- Build the graph in TapeM for readability.
  let m : TapeM ℚ _ := do
    let w1Id ← Train.TapeM.param W1 (name := some "W1")
    let b1Id ← Train.TapeM.param b1 (name := some "b1")
    let w2Id ← Train.TapeM.param W2 (name := some "W2")
    let b2Id ← Train.TapeM.param b2 (name := some "b2")
    let xId ← Train.TapeM.const x (name := some "x")

    -- Forward pass: linear -> relu -> linear
    let z1Id ← TapeM.linear (inDim:=inDim) (outDim:=hidDim) w1Id b1Id xId
    let a1Id ← TapeM.relu (s:=.dim hidDim .scalar) z1Id
    let yId ← TapeM.linear (inDim:=hidDim) (outDim:=outDim) w2Id b2Id a1Id

    let t ← TapeM.getTape
    let grads ← liftM (Tape.backward (t:=t) yId (Runtime.Autograd.AnyTensor.mk dLdy))

    let ids : ParamIds := { w1Id := w1Id, b1Id := b1Id, w2Id := w2Id, b2Id := b2Id }
    pure (ids, grads)

  let ((ids, grads), _) ← TapeM.run t0 m

  let (dW1_exp, db1_exp, dW2_exp, db2_exp, _dX_exp) := expected

  let dW1_dyn ← Train.requireGradTensor (tag := tag)
    (s:=.dim hidDim (.dim inDim .scalar)) grads ids.w1Id
  let db1_dyn ← Train.requireGradTensor (tag := tag)
    (s:=.dim hidDim .scalar) grads ids.b1Id
  let dW2_dyn ← Train.requireGradTensor (tag := tag)
    (s:=.dim outDim (.dim hidDim .scalar)) grads ids.w2Id
  let db2_dyn ← Train.requireGradTensor (tag := tag)
    (s:=.dim outDim .scalar) grads ids.b2Id

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
