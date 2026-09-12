/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API

/-!
# Quickstart: Tensor Basics

This first TorchLean example introduces typed tensor construction, reshaping,
and element-type conversion.

What it covers:
- rank-zero and arbitrary-rank literals,
- reshaping without changing entry order,
- the fact that the element type `α` selects the tensor's arithmetic,
- conversion to native `Float32`.

Run:
  `scripts/lake.sh exe torchlean quickstart_tensors`
-/

@[expose] public section

namespace NN.Examples.Quickstart.TensorBasics

open TorchLean

/-- Command-line help for the tensor-basics quickstart. -/
def usage : String :=
  String.intercalate "\n"
    [ "TorchLean tensor basics quickstart"
    , ""
    , "Usage:"
    , "  scripts/lake.sh exe torchlean quickstart_tensors"
    , ""
    , "This demo has no tutorial-specific flags."
    ]

/-- Entry point. Nothing to configure, so the only accepted argument is `--help`. -/
def main (args : List String) : IO Unit := do
  let args := CLI.dropDashDash args
  if CLI.hasHelp args then
    IO.println usage
    return
  CLI.requireNoArgs "quickstart_tensors" args
  IO.println "== Quickstart: tensor basics =="

  -- Rank-zero tensors use the empty shape and an ordinary numeric literal.
  let threshold : Tensor Float [] := 0.5
  IO.println s!"Rank-zero tensor: {reprStr threshold}"

  -- Each tensor has one element type `α`.
  let floatTensor : Tensor Float [4] := [0.1, 0.2, 0.3, 0.4]
  let rationalTensor : Tensor ℚ [4] := [0.1, 0.2, 0.3, 0.4]
  let integerTensor : Tensor Int [4] := [1, 2, 3, 4]

  IO.println s!"Float tensor:    {reprStr floatTensor}"
  IO.println s!"Rational tensor: {reprStr rationalTensor}"
  IO.println s!"Integer tensor:  {reprStr integerTensor}"

  -- A cast changes the element type while preserving the shape.
  let nativeFloat32 : Tensor Float32 [4] := Tensor.cast floatTensor Float32
  IO.println s!"Float32 cast:    {reprStr nativeFloat32}"

  -- Nested brackets show where every value lies along each axis.
  let cube : Tensor Float [2, 2, 2] :=
    [
      [ [1, 2], [3, 4] ],
      [ [5, 6], [7, 8] ]
    ]
  IO.println s!"Rank-3 tensor:   {reprStr cube}"

  -- Reshape preserves the values and their row-major order.
  let matrix : Tensor Float [2, 2] := floatTensor.reshape [2, 2]
  IO.println s!"Reshaped matrix: {reprStr matrix}"

end NN.Examples.Quickstart.TensorBasics
