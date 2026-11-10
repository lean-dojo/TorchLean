/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.CLI.Command
public import TorchLean.NN.CLI.Parser
public import TorchLean.NN.IR.Builder
public import TorchLean.NN.IR.Eval
public import TorchLean.Tensor.FromFloat
public import TorchLean.Tensor.Literal
public import TorchLean.Verify.CROWN.BoundOps.FP32
public import TorchLean.Verify.CROWN.Op.Exec

/-!
# One graph, several scalar interpretations

Evaluate one graph, generic in its scalar type, over executable binary32 values, then propagate
intervals through its operations.

The tutorial builds one graph of registered operators with `IR.Graph.Builder`, ten nodes: the
input, the four parameter tensors as constant nodes, and five operators:

$$
x\mapsto\tanh\!\left(\operatorname{sum}
  \left(\operatorname{Linear}_2(\operatorname{ReLU}(\operatorname{Linear}_1(x)))\right)\right)
$$

It then takes three steps:
1) Evaluate the graph in executable binary32 (`Binary32`).
2) Run IBP on the operator graph (`TorchLean.CROWN.Op.runIBP`) with `Binary32` endpoints and
   directed rounding.
3) Point to the Lean theorem that the Boolean checker is sound (`Box.containsDecBool_sound`).

`Binary32` is executable inside Lean, so the runnable example uses it.

Run:
  `scripts/lake.sh exe torchlean graph_semantics`
-/

@[expose] public section

open FloatLib.Floats (Binary32)

open TorchLean.Tensor

open TorchLean.CROWN

namespace TorchLean.Examples.GraphSemantics

/-- Command-line help for the graph-semantics tutorial, which takes no flags. -/
def usage : String :=
  String.intercalate "\n"
    [ "TorchLean graph semantics tutorial"
    , ""
    , "Usage:"
    , "  scripts/lake.sh exe torchlean graph_semantics"
    ]

/--
The network's parameters, generic in the scalar type.

`Parameters.map` transports one parameter record from `Float` to the bit-level IEEE model, and
`run` evaluates the graph only with the transported `Binary32` parameters.
-/
structure Parameters (α : Type) [Storage α] where
  /-- Weight matrix for layer 1. -/
  hiddenWeight : Tensor α [5, 4]
  /-- Bias for layer 1. -/
  hiddenBias : Tensor α [5]
  /-- Weight matrix for layer 2. -/
  outputWeight : Tensor α [3, 5]
  /-- Bias for layer 2. -/
  outputBias : Tensor α [3]

/-- Transport every parameter tensor along a scalar conversion. -/
def Parameters.map {α β : Type}
    [Storage α] [Storage β]
    (f : α → β) (parameters : Parameters α) : Parameters β :=
  { hiddenWeight := Tensor.map f parameters.hiddenWeight
    hiddenBias := Tensor.map f parameters.hiddenBias
    outputWeight := Tensor.map f parameters.outputWeight
    outputBias := Tensor.map f parameters.outputBias }

/--
Concrete parameters, written as small decimal literals.

`run` converts these host `Float` values to executable binary32 with `Parameters.map`, which may
round the coefficients but leaves the graph structure and the parameter layout unchanged.
-/
def floatParameters : Parameters Float :=
  { hiddenWeight :=
      [ [0.15, -0.12, 0.08, 0.05]
      , [0.02, 0.11, -0.09, 0.07]
      , [-0.04, 0.06, 0.10, -0.03]
      , [0.09, 0.01, 0.04, 0.13]
      , [-0.07, 0.03, 0.12, -0.02] ]
    hiddenBias := [0.01, -0.02, 0.03, 0.0, 0.02]
    outputWeight :=
      [ [0.05, 0.08, -0.06, 0.03, 0.07]
      , [-0.04, 0.02, 0.09, -0.01, 0.06]
      , [0.10, -0.03, 0.04, 0.05, -0.08] ]
    outputBias := [0.02, -0.01, 0.00] }

/--
The network as a graph of registered operators: the input, the weight and bias constants of the
first layer, `Op.linear`, `Op.relu`, the constants of the second layer, `Op.linear`, the sum over
the only axis (`Op.reduce .sum`) and `Op.tanh`. `build` runs the structural check `Graph.check`.

The graph is written out node by node, so the reader sees each node that the evaluator and the
interval propagator receive.
-/
def graph : Except String IR.Graph :=
  let (b, input) := ({} : IR.Graph.Builder).input 0 (.value [4])
  let (b, hiddenWeight) := b.const (.value [5, 4])
  let (b, hiddenBias) := b.const (.value [5])
  let (b, layer1) := b.op (Op.linear (lead := []) (inDim := 4) (outDim := 5))
    [hiddenWeight, hiddenBias, input]
  let (b, activation) := b.op (Op.relu (s := [5])) layer1
  let (b, outputWeight) := b.const (.value [3, 5])
  let (b, outputBias) := b.const (.value [3])
  let (b, output) := b.op (Op.linear (lead := []) (inDim := 5) (outDim := 3))
    ([outputWeight, outputBias] ++ activation)
  let (b, total) := b.op (Op.reduce (s := [3]) .sum [⟨0, by decide⟩] false
    (by decide)) output
  let (b, result) := b.op (Op.tanh (s := [])) total
  b.build result.toArray

/-- The weight and bias tensors of the two linear layers, at the constant nodes `1`, `2`, `5` and
`6`. -/
def payload {α : Type} [Storage α] [Context α]
    (parameters : Parameters α) : IR.Payload α :=
  { const? := fun
      | 1 => some ⟨.value [5, 4], parameters.hiddenWeight⟩
      | 2 => some ⟨.value [5], parameters.hiddenBias⟩
      | 5 => some ⟨.value [3, 5], parameters.outputWeight⟩
      | 6 => some ⟨.value [3], parameters.outputBias⟩
      | _ => none }

/-- Evaluate the graph at scalar type `α` and check that the output is one scalar. The definition
is neither inlined nor specialized, because compiling a copy of the evaluator at the binary32 scalar
of `run` exceeds the compiler's heartbeat limit. -/
@[noinline, nospecialize]
def evaluate
    {α : Type} [Storage α] [Context α]
    (parameters : Parameters α) (inputTensor : Tensor α [4]) :
    Except String (Tensor α []) := do
  let g ← graph
  match ← g.eval (payload parameters) [⟨.value [4], inputTensor⟩] with
  | [⟨⟨.value, []⟩, output⟩] => pure output
  | _ => throw "graph semantics: expected one scalar value output"

/-- The centre of the input box. -/
def center : Tensor Float [4] :=
  [0.3, -0.2, 0.1, 0.4]

/-- An `eps`-ball around `center`, in the scalar type `α`. -/
def inputBox {α : Type} [Storage α] [Context α] [FromFloat α] (eps : Float) :
    Box α [4] :=
  let center : Tensor α [4] :=
    Tensor.map ofFloat center
  let r : α := ofFloat eps
  let radius : Tensor α [4] := Tensor.full (α := α) [4] r
  { lo := (center - radius)
    hi := center + radius }

/--
Read a one-dimensional flat box back as a scalar box, failing loudly if the dimension is not one.
-/
def scalarBox {α : Type} [Storage α] [Context α] (box : FlatBox α) :
    Except String (Box α []) := do
  if h : box.dim = 1 then
    let lowerTensor : Tensor α [1] :=
      Tensor.castShape box.lo
        (congrArg (fun extent => ([extent] : Shape)) h)
    let upperTensor : Tensor α [1] :=
      Tensor.castShape box.hi
        (congrArg (fun extent => ([extent] : Shape)) h)
    let lower : α := lowerTensor[0]
    let upper : α := upperTensor[0]
    pure { lo := Tensor.full [] lower, hi := Tensor.full [] upper }
  else
    throw s!"expected a scalar FlatBox (dim=1), got dim={box.dim}"

/-- Run the tutorial: evaluate at the center of the input box and propagate bounds over it. -/
def run : IO Unit := do
  IO.println "== Graph semantics tutorial =="
  let g ← IO.ofExcept graph
  IO.println s!"graph nodes = {g.nodes.size}"

  let ieeeParameters : Parameters Binary32 := floatParameters.map ofFloat
  let box : Box Binary32 [4] := inputBox (eps := 0.05)
  let flatInputBox : FlatBox Binary32 := { dim := 4, lo := box.lo, hi := box.hi }

  -- Evaluate at the center point.
  let referenceInputIEEE : Tensor Binary32 [4] :=
    Tensor.map ofFloat center
  let outputAtCenter ← IO.ofExcept (evaluate ieeeParameters referenceInputIEEE)
  IO.println s!"[eval configured binary32] y(x0) = {Tensor.pretty outputAtCenter}"

  -- Compute the IBP box with directed rounding via
  -- `BoundOps Binary32`.
  let ibp := Op.runIBP g (payload ieeeParameters) (Op.inputSeeds g [flatInputBox])
  let outFlat ← IO.ofExcept (Op.outputBox? g ibp 0)
  let outBox ← IO.ofExcept (scalarBox outFlat)
  IO.println s!"[IBP IEEE endpoints] lo = {Tensor.pretty outBox.lo}"
  IO.println s!"[IBP IEEE endpoints] hi = {Tensor.pretty outBox.hi}"
  IO.println "checker theorem: `TorchLean.CROWN.Box.containsDecBool_sound`"

/-- Entry point. -/
def main (args : List String) : IO Unit := do
  let args := CLI.dropDashDash args
  if CLI.hasHelp args then
    IO.println usage
    return
  CLI.requireNoArgs "graph_semantics" args
  run

end TorchLean.Examples.GraphSemantics
