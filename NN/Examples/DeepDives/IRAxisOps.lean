/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API
public import NN.API.Module.Command
public import NN.Runtime.Autograd.IRExec

/-!
# IR axis operations

IR axis-ops runtime tutorial.

This tutorial constructs and evaluates three IR operations with an explicit `axis`:

- `softmax axis` (PyTorch: `torch.softmax(x, dim=axis)`)
- `concat axis` (PyTorch: `torch.cat(xs, dim=axis)`)
- `layernorm axis`
  PyTorch: `F.layer_norm(x, normalized_shape=x.shape[axis:])`

Each operation accepts any in-bounds tensor dimension. The implementation may move that dimension
to an innermost position while evaluating an optimized kernel, but its public semantics preserves
the original shape and dimension numbering. Forward graph execution reports unsupported backend
cases explicitly.

Run:

`lake exe torchlean ir_axis_ops --execution eager`
-/

@[expose] public section

namespace NN.Examples.DeepDives.IRAxisOps

open TorchLean
open NN.IR

open Runtime.Autograd.TypedGraph

/-- Command-line help for the IR axis-ops tutorial. -/
def usage : String :=
  String.intercalate "\n"
    [ "TorchLean IR axis-ops tutorial"
    , ""
    , "Usage:"
    , "  lake exe torchlean ir_axis_ops [options]"
    , ""
    , "Options:"
    , "  --arithmetic native|ieee|complex"
    , "  --execution eager|typed-graph"
    , "  --device auto|cpu|cuda|rocm|metal|wasm|tpu|trainium|custom|external"
    , "  --show-backend                    print backend capsules as they execute"
    ]

/-!
## Tensor Shapes

These shapes illustrate an axis that is neither first nor last.
-/

abbrev baseTensorShape : Shape := [2, 3, 4]
abbrev widerMiddleAxisShape : Shape := [2, 5, 4]
abbrev concatenatedMiddleAxisShape : Shape := [2, 8, 4]

/-!
## Small IR Graphs
-/

def softmaxMiddleAxisGraph : NN.IR.Graph :=
  { nodes := #[
      { id := 0, parents := #[], kind := .input, outShape := baseTensorShape }
    , { id := 1, parents := #[0], kind := .softmax (axis := 1), outShape := baseTensorShape }
    ] }

/--
LayerNorm over axis 1 of a rank-three tensor, the interesting case because the normalized axis is
neither the first nor the last.
-/
def layerNormMiddleAxisGraph : NN.IR.Graph :=
  { nodes := #[
      { id := 0, parents := #[], kind := .input, outShape := baseTensorShape }
    , { id := 1, parents := #[0], kind := .layernorm (axis := 1), outShape := baseTensorShape }
    ] }

/-- Concatenation along axis 1, joining a `[2, 3, 4]` and a `[2, 5, 4]` tensor into `[2, 8, 4]`. -/
def concatMiddleAxisGraph : NN.IR.Graph :=
  -- The concat example uses `rand_uniform` sources, so the input node only fixes the graph's
  -- single-input interface for the shared runner.
  { nodes := #[
      { id := 0, parents := #[], kind := .input, outShape := [] }
    , { id := 1, parents := #[], kind := .randUniform (seed := 0), outShape := baseTensorShape }
    , { id := 2, parents := #[], kind := .randUniform (seed := 1),
        outShape := widerMiddleAxisShape }
    , { id := 3, parents := #[1, 2], kind := .concat (axis := 1),
        outShape := concatenatedMiddleAxisShape }
    ] }

/-!
## Runner Helpers
-/

/-- Print a compact preview of a tensor. -/
def firstScalars {α : Type} [Storage α] [ToString α] {σ : Shape}
    (tensor : Tensor α σ) : String :=
  let xs := Tensor.to tensor (Array α)
  let ys := xs.extract 0 8
  let body := String.intercalate ", " (ys.toList.map toString)
  "[" ++ body ++ (if xs.size > ys.size then ", ..." else "") ++ "]"

/-- Evaluate an example graph through the runtime API and print its selected output. -/
def printEvaluation
    {α : Type} {σ : Shape} [Storage α] [Context α] [ToString α]
    (tag : String) (g : NN.IR.Graph) (payload : NN.IR.Payload α)
    (x : Tensor α σ) (outputId : Fin g.nodes.size) : IO Unit := do
  let ⟨shape, output⟩ ← CLI.orThrow tag <|
    Runtime.Autograd.IRExec.evaluate g payload x outputId
  IO.println s!"[{tag}] output shape: {repr shape}"
  IO.println s!"[{tag}] first scalars: {firstScalars output}"

/--
Run every axis-op example at scalar type `α`.

Generic in `α` so the tutorial can be run under native `Float` or under the bit-level IEEE model
with no change to the graphs.
-/
def runOnce
    {α : Type} [Storage α] [Context α] [ToString α]
    [Runtime.FromFloat α]
    : IO Unit := do
  let payload : NN.IR.Payload α := {}

  -- A small but nontrivial 2×3×4 input tensor.
  let inputTensor : Tensor α baseTensorShape :=
    Tensor.generateFlat [2, 3, 4] (fun i =>
      Runtime.ofFloat ((Float.ofNat i) / 10.0 - 1.0))

  IO.println ""
  IO.println "== IR axis ops tutorial =="
  IO.println "The runtime validates each graph and evaluates its selected output."

  IO.println ""
  IO.println "-- softmax axis=1 on shape [2,3,4]"
  printEvaluation (α := α) (tag := "softmax_middle_axis") (g := softmaxMiddleAxisGraph)
    (payload := payload) (x := inputTensor) (outputId := ⟨1, by decide⟩)

  IO.println ""
  IO.println "-- layernorm axis=1 on shape [2,3,4]"
  IO.println "PyTorch meaning: normalized_shape = x.shape[axis:] = [3,4]"
  printEvaluation (α := α) (tag := "layernorm_middle_axis") (g := layerNormMiddleAxisGraph)
    (payload := payload) (x := inputTensor) (outputId := ⟨1, by decide⟩)

  IO.println ""
  IO.println "-- concat axis=1: [2,3,4] ++ [2,5,4] -> [2,8,4]"
  let scalarInput : Tensor α [] :=
    Tensor.full [] (Runtime.ofFloat (α := α) 0.0)
  printEvaluation (α := α) (tag := "concat_middle_axis") (g := concatMiddleAxisGraph)
    (payload := payload) (x := scalarInput) (outputId := ⟨3, by decide⟩)

/-- Runtime-selected entrypoint body for the axis-ops tutorial. -/
def runSelected
    {α : Type} [Storage α] [Context α] [ToString α]
    [Runtime.FromFloat α]
    (rest : List String) : IO Unit := do
  CLI.requireNoArgs "ir_axis_ops" rest
  runOnce (α := α)

/-- Entry point; `--arithmetic` picks the scalar type the whole tutorial runs at. -/
def main (args : List String) : IO Unit := do
  let args := CLI.dropDashDash args
  if CLI.hasHelp args then
    IO.println usage
    return
  Module.withSelectedRuntime args
    (fun {α} _ _ _ _ _cast _opts rest => runSelected (α := α) rest)

end NN.Examples.DeepDives.IRAxisOps
