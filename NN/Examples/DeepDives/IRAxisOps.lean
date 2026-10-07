/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API
public import NN.Runtime.Autograd.IRExec

/-!
# IR axis operations

Evaluate axis operations through the checked IR lowering.

This tutorial constructs and evaluates three IR operations with an explicit `axis`:

- `softmax axis` (PyTorch: `torch.softmax(x, dim=axis)`)
- `concat axis` (PyTorch: `torch.cat(xs, dim=axis)`)
- `layernorm axis`
  PyTorch: `F.layer_norm(x, normalized_shape=x.shape[axis:])`

The axis is part of each operation's meaning. The evaluator checks graph and input shapes, lowers
the graph and computes its selected output with Lean tensor operations. This example does not
dispatch to LibTorch or select a GPU device; `--arithmetic` chooses the scalar implementation.

Run:

`scripts/lake.sh exe torchlean ir_axis_ops --arithmetic ieee`
-/

@[expose] public section

namespace NN.Examples.DeepDives.IRAxisOps

open TorchLean

/-- Command-line help for the IR axis-ops tutorial. -/
def usage : String :=
  String.intercalate "\n"
    [ "TorchLean IR axis-ops tutorial"
    , ""
    , "Usage:"
    , "  scripts/lake.sh exe torchlean ir_axis_ops [options]"
    , ""
    , "Options:"
    , "  --arithmetic native|ieee|complex"
    , ""
    , "Evaluates the checked IR on CPU; no device or execution-mode selection."
    ]

/-!
## Small IR Graphs
-/

/-- Softmax along a chosen axis of the example's `[2, 3, 4]` input. -/
def softmax (axis : Nat := 1) : NN.IR.Graph :=
  { nodes := #[
      { id := 0, parents := #[], kind := .input, outShape := [2, 3, 4] }
    , { id := 1, parents := #[0], kind := .softmax axis, outShape := [2, 3, 4] }
    ] }

/--
LayerNorm over the suffix starting at `axis`. The default normalizes `[3, 4]` together, rather
than just the final dimension.
-/
def layerNorm (axis : Nat := 1) : NN.IR.Graph :=
  { nodes := #[
      { id := 0, parents := #[], kind := .input, outShape := [2, 3, 4] }
    , { id := 1, parents := #[0], kind := .layernorm axis, outShape := [2, 3, 4] }
    ] }

/--
Join `[2, 3, 4]` and `[2, 5, 4]` along axis one. Other axis choices demonstrate shape rejection:
the remaining dimensions must agree and the declared output must match the inferred shape.
-/
def concat (axis : Nat := 1) : NN.IR.Graph :=
  -- The concat example uses `rand_uniform` sources, so the input node only fixes the graph's
  -- single-input interface for the shared runner.
  { nodes := #[
      { id := 0, parents := #[], kind := .input, outShape := [] }
    , { id := 1, parents := #[], kind := .randUniform (seed := 0), outShape := [2, 3, 4] }
    , { id := 2, parents := #[], kind := .randUniform (seed := 1),
        outShape := [2, 5, 4] }
    , { id := 3, parents := #[1, 2], kind := .concat axis,
        outShape := [2, 8, 4] }
    ] }

/-!
## Runner Helpers
-/

/-- Print a compact preview of a tensor. -/
def preview {α : Type} [Storage α] [ToString α] {σ : Shape}
    (tensor : Tensor α σ) : String :=
  let entries : Tensor α [min 8 σ.size] :=
    Tensor.flattenThenTake [] (min 8 σ.size) (Nat.min_le_right 8 σ.size) tensor
  let text := Spec.pretty entries
  if σ.size > 8 then (text.dropEnd 1).toString ++ ", ...]" else text

/-- Evaluate an example graph with Lean tensor operations and print its selected output. -/
def print
    {α : Type} {σ : Shape} [Storage α] [Context α] [ToString α]
    (tag : String) (g : NN.IR.Graph) (payload : NN.IR.Payload α)
    (x : Tensor α σ) (outputId : Fin g.nodes.size) : IO Unit := do
  let ⟨shape, output⟩ ← CLI.orThrow tag <|
    Runtime.Autograd.IRExec.evaluate g payload x outputId
  IO.println s!"[{tag}] output shape: {repr shape}"
  IO.println s!"[{tag}] first scalars: {preview output}"

/--
Run every axis-op example at scalar type `α`.

Generic in `α` so the tutorial can be run under native `Float` or under the bit-level IEEE model
with no change to the graphs.
-/
def run
    {α : Type} [Storage α] [Context α] [ToString α]
    [Runtime.FromFloat α]
    : IO Unit := do
  let payload : NN.IR.Payload α := {}

  -- A small but nontrivial 2×3×4 input tensor.
  let inputTensor : Tensor α [2, 3, 4] :=
    Tensor.generateFlat [2, 3, 4] (fun i =>
      Runtime.ofFloat ((Float.ofNat i) / 10.0 - 1.0))

  IO.println ""
  IO.println "== IR axis ops tutorial =="
  IO.println "The IR evaluator validates each graph and computes its selected output on CPU."

  IO.println ""
  IO.println "-- softmax axis=1 on shape [2,3,4]"
  print (α := α) (tag := "softmax_middle_axis") (g := softmax (axis := 1))
    (payload := payload) (x := inputTensor) (outputId := ⟨1, by decide⟩)

  IO.println ""
  IO.println "-- layernorm axis=1 on shape [2,3,4]"
  IO.println "PyTorch meaning: normalized_shape = x.shape[axis:] = [3,4]"
  print (α := α) (tag := "layernorm_middle_axis") (g := layerNorm (axis := 1))
    (payload := payload) (x := inputTensor) (outputId := ⟨1, by decide⟩)

  IO.println ""
  IO.println "-- concat axis=1: [2,3,4] ++ [2,5,4] -> [2,8,4]"
  let scalarInput : Tensor α [] :=
    Tensor.full [] (Runtime.ofFloat (α := α) 0.0)
  print (α := α) (tag := "concat_middle_axis") (g := concat (axis := 1))
    (payload := payload) (x := scalarInput) (outputId := ⟨3, by decide⟩)

/-- Entry point; `--arithmetic` picks the scalar type the whole tutorial runs at. -/
def main (args : List String) : IO Unit := do
  let args := CLI.dropDashDash args
  if CLI.hasHelp args then
    IO.println usage
    return
  let (arithmetic, rest) ← CLI.orThrow "ir_axis_ops" <|
    Runtime.Arithmetic.parseAndStrip args
  CLI.requireNoArgs "ir_axis_ops" rest
  arithmetic.log
  Runtime.Arithmetic.withRuntime arithmetic @run

end NN.Examples.DeepDives.IRAxisOps
