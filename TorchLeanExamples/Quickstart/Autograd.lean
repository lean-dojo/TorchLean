/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Autograd.Function
public import TorchLean.NN.Autograd.Model
public import TorchLean.NN.Data.Source
public import TorchLean.NN.Builder
public import TorchLean.NN.Training.Reporting

/-!
# Quickstart: Autograd

Use `autograd.grad` for scalar tensor functions and `autograd.model.grad` when model state is also
being differentiated. Add `value := true` when the scalar value is needed with the gradient.

Jacobians, Hessians, JVPs, VJPs, and gradient stopping live in
`TorchLeanExamples.DeepDives.AutogradTransforms`.

Run:

```bash
scripts/lake.sh exe torchlean quickstart_autograd
```
-/

@[expose] public section

namespace TorchLean.Examples.Quickstart.Autograd

/-- Subcommand name, kept in one place so the usage text and the error messages agree. -/
def exeName : String := "quickstart_autograd"

/-- Mean squared magnitude of one tensor. -/
def meanSquare {shape : Shape} : autograd.Function shape [] :=
  fun x => do
    let squared ← nn.functional.square x
    nn.functional.mean squared

/-- A single affine model, `y = W x + b`. -/
def model : nn.Sequential [2] [1] :=
  nn.build 0 (nn.linear 2 1)

/-- Run the two common differentiation paths. -/
def run : IO Unit := do
  IO.println "== Differentiate a tensor function =="
  let x : Tensor Float [3] := [1.0, 2.0, 3.0]
  let (gradient, value) ← autograd.grad meanSquare x (value := true)
  IO.println s!"mean(x^2) = {value}"
  IO.println s!"d/dx       = {reprStr gradient}"

  IO.println ""
  IO.println "== Differentiate a model loss =="
  let state : autograd.model.State model Float :=
    nn.initialState model
  let input : Tensor Float [2] := [0.5, -1.0]
  let target : Tensor Float [1] := [0.25]
  let (gradient, loss) ←
    autograd.model.grad
      model autograd.model.Loss.mse state input target (value := true)
  IO.println s!"loss     = {loss}"
  IO.println s!"gradient = {reprStr gradient}"

/-- Help text for a demo that takes no flags of its own. -/
def usage : String :=
  String.intercalate "\n"
    [ "TorchLean autograd quickstart"
    , ""
    , "Usage:"
    , "  scripts/lake.sh exe torchlean quickstart_autograd"
    ]

/--
Entry point. `CLI.dropDashDash` drops a leading `--` separator, `--help` or `-h` prints the usage
text, and `requireNoArgs` fails on any other argument.
-/
def main (args : List String) : IO Unit := do
  let args := CLI.dropDashDash args
  if CLI.hasHelp args then
    IO.println usage
    return
  CLI.requireNoArgs exeName args
  run

end TorchLean.Examples.Quickstart.Autograd
