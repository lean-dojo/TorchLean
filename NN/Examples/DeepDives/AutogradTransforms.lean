/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API

/-!
# Autograd Transforms

Advanced differentiation operations are kept out of the first autograd tutorial:

- `jacrev` and `hessian` for tensor functions;
- `jvp` and `hvp` for model state;
- `Loss.detach` for an explicit gradient stop.

Build this module after `NN.Examples.Quickstart.AutogradBasics`.

Run:

```bash
lake exe torchlean autograd_transforms
```
-/

@[expose] public section

namespace NN.Examples.DeepDives.AutogradTransforms

open TorchLean

/-- Subcommand name, used in the usage text and in argument-error messages. -/
def exeName : String := "autograd_transforms"

/--
Componentwise squaring, whose Jacobian is the diagonal matrix `2x`. Small enough that the printed
rows can be checked by hand.
-/
def square : autograd.Function [2] [2] :=
  fun x => nn.functional.square x

/-- Mean of the squares: a scalar-valued function, so it has a Hessian to compute. -/
def meanSquare : autograd.Function [2] [] :=
  fun x => do
    let squared ← nn.functional.square x
    nn.functional.mean squared

/-- One linear layer `2 -> 3`, seeded deterministically so the printed numbers are reproducible. -/
def model : nn.Sequential [2] [3] :=
  nn.build 0 (nn.linear 2 3)

/-- Mean squared error, the loss the directional-derivative examples differentiate. -/
def mse : autograd.model.Loss [3] [3] :=
  autograd.model.Loss.meanSquaredError

/-- Run higher-order and directional differentiation examples. -/
def runDemo : IO Unit := do
  let x : Tensor Float [2] := [0.5, -1.2]

  let jacobian ← autograd.jacrev square x
  IO.println s!"Jacobian of x^2: {reprStr jacobian}"

  let hessian ← autograd.hessian meanSquare x
  IO.println s!"Hessian of mean(x^2): {reprStr hessian}"

  let state : autograd.model.State model Float :=
    autograd.model.initialState model
  let target : Tensor Float [3] := [0.7, 0.1, -0.5]
  let direction : autograd.model.State model Float :=
    autograd.model.fullState model 0.1

  let directionalDerivative ←
    autograd.model.jvp model mse state x target direction
  IO.println s!"loss directional derivative = {directionalDerivative}"

  let curvature ←
    autograd.model.hvp model mse state x target direction
  IO.println s!"loss Hessian-vector product = {reprStr curvature}"

  let detached : autograd.model.Loss [3] [3] :=
    autograd.model.Loss.detach mse
  let stoppedGradient ← autograd.model.grad model detached state x target
  IO.println s!"state gradient after detaching the model output = {reprStr stoppedGradient}"

/-- Help text; the demo takes no flags. -/
def usage : String :=
  String.intercalate "\n"
    [ "TorchLean autograd transforms"
    , ""
    , "Usage:"
    , "  lake exe torchlean autograd_transforms"
    ]

/-- Entry point. -/
def main (args : List String) : IO Unit := do
  let args := CLI.dropDashDash args
  if CLI.hasHelp args then
    IO.println usage
    return
  CLI.requireNoArgs exeName args
  runDemo

end NN.Examples.DeepDives.AutogradTransforms
