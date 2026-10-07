/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API
public import NN.MLTheory.CROWN.Extras.FP32

/-!
# One semantic universe

End-to-end “one semantic universe” tutorial (single graph, many semantics, one checker).

This tutorial lives under `NN/Examples/DeepDives` because it connects execution, interval semantics,
and checker soundness in one graph.

We build one six-node IR graph:

$$
x\mapsto\tanh\!\left(\operatorname{sum}
  \left(\operatorname{Linear}_2(\operatorname{ReLU}(\operatorname{Linear}_1(x)))\right)\right)
$$

and then:
1) evaluate it under multiple scalar semantics (`ℝ`, `FP32`, `ExecFloat.Binary 8 23`);
2) run IBP under multiple interval semantics (endpoints in `ℝ`, `FP32`, `ExecFloat.Binary 8 23` with
   directed rounding);
3) empirically check that `evalIEEE(G,x)` lies in the IBP output box for random $x\in B$;
4) point to the Lean theorem that the Boolean checker is sound (`Box.containsDecBool_sound`).

Notes:
- `ℝ` and `FP32` instantiations are proof-oriented and noncomputable (they typecheck, but do not run
  as an executable).
- `ExecFloat.Binary 8 23` is executable inside Lean, so we use it for the runnable
  consistency check.

Run:
  `scripts/lake.sh exe torchlean one_semantic_universe --samples 50`
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.ExecFloat (Binary)

open Spec TorchLean
open TorchLean.Tensor

open NN.IR
open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Graph

open TorchLean.Floats

namespace NN.Examples.DeepDives.OneSemanticUniverse

/-- Command-line help for the one-semantics tutorial. -/
def usage : String :=
  String.intercalate "\n"
    [ "TorchLean one semantic universe tutorial"
    , ""
    , "Usage:"
    , "  scripts/lake.sh exe torchlean one_semantic_universe [options]"
    , ""
    , "Options:"
    , "  --samples N"
    ]

/--
The network's parameters, generic in the scalar type.

Being generic in `α` is the whole point of this tutorial: one parameter record is transported to
`Float`, to the bit-level IEEE model, to `ℝ` and to the rounded-real `FP32`, and the same graph is
evaluated in each.
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

The executable binary32 parameters are obtained from these host `Float` values by `Parameters.map`.
Conversion may round coefficients; the graph structure and parameter layout remain fixed.
-/
def parameters : Parameters Float :=
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
The network as six IR nodes: input, linear, ReLU, linear, sum, tanh.

Written out node by node rather than built by a combinator, so the reader can see exactly what the
evaluator and the interval propagator are given.
-/
def graph : NN.IR.Graph :=
  let inputNode : NN.IR.Node :=
    { id := 0, parents := #[], kind := .input, outShape := [4] }
  let hiddenLinearNode : NN.IR.Node :=
    { id := 1, parents := #[0], kind := .linear, outShape := [5] }
  let hiddenActivationNode : NN.IR.Node :=
    { id := 2, parents := #[1], kind := .relu, outShape := [5] }
  let outputLinearNode : NN.IR.Node :=
    { id := 3, parents := #[2], kind := .linear, outShape := [3] }
  let reductionNode : NN.IR.Node :=
    { id := 4, parents := #[3], kind := .sum, outShape := [] }
  let outputNode : NN.IR.Node :=
    { id := 5, parents := #[4], kind := .tanh, outShape := [] }
  { nodes :=
      #[inputNode, hiddenLinearNode, hiddenActivationNode, outputLinearNode, reductionNode,
        outputNode] }

/-- Attach the weight and bias tensors to the two `linear` nodes. -/
def payload {α : Type} [Storage α] [Context α]
    (parameters : Parameters α) : NN.IR.Payload α :=
  { linear? := fun id =>
      if id = 1 then
        some {
          outDim := 5
          inDim := 4
          W := parameters.hiddenWeight
          b := parameters.hiddenBias
        }
      else if id = 3 then
        some {
          outDim := 3
          inDim := 5
          W := parameters.outputWeight
          b := parameters.outputBias
        }
      else
        none }

/-- The same parameters in the form interval propagation wants, together with the input box. -/
def parameterStore {α : Type} [Storage α] [Context α]
    (parameters : Parameters α) (inputBox : FlatBox α) : ParamStore α :=
  { inputBoxes := (Std.HashMap.emptyWithCapacity).insert 0 inputBox
    linearWB :=
      (Std.HashMap.emptyWithCapacity)
        |>.insert 1 {
          m := 5
          n := 4
          w := parameters.hiddenWeight
          b := parameters.hiddenBias
        }
        |>.insert 3 {
          m := 3
          n := 5
          w := parameters.outputWeight
          b := parameters.outputBias
        } }

/-- Evaluate the graph at scalar type `α` and check that the result really is a scalar. -/
def evaluate
    {α : Type} [Storage α] [Context α]
    (parameters : Parameters α) (inputTensor : Tensor α [4]) :
    Except String (Tensor α []) := do
  let graphPayload := payload (α := α) parameters
  let input : Spec.SomeTensor α := Spec.SomeTensor.ofTensor inputTensor
  let output ←
    NN.IR.Graph.denote (α := α) (g := graph) (payload := graphPayload) (input := input)
      (outputId := 5)
  NN.IR.Graph.expectShape (α := α) (expected := []) output

/-!
### Proof-oriented instantiations

The same graph evaluator and interval propagation algorithm specialize directly to `ℝ` and
proof-oriented `FP32`. These definitions are noncomputable because their scalar semantics are
intended for reasoning rather than native execution.
-/
section ProofOnly

/-- Evaluation over the reals: the mathematical meaning of the network, with no rounding at all. -/
noncomputable example
    (parameters : Parameters ℝ) (input : Tensor ℝ [4]) :
    Except String (Tensor ℝ []) :=
  evaluate (α := ℝ) parameters input

/--
Evaluation over rounded reals: each operation rounds to binary32 precision. Values carry a real
interpretation for proofs rather than an executable bit pattern; this model has no upper exponent
cutoff.
-/
noncomputable example
    (parameters : Parameters TorchLean.Floats.FP32)
    (input : Tensor TorchLean.Floats.FP32 [4]) :
    Except String (Tensor TorchLean.Floats.FP32 []) :=
  evaluate (α := TorchLean.Floats.FP32) parameters input

/-- Interval bound propagation over the reals. -/
noncomputable example
    (parameters : ParamStore ℝ) : Array (Option (FlatBox ℝ)) :=
  runIBP graph parameters

/-- The same propagation over rounded reals. -/
noncomputable example
    (parameters : ParamStore TorchLean.Floats.FP32) :
    Array (Option (FlatBox TorchLean.Floats.FP32)) :=
  runIBP graph parameters

end ProofOnly

/-- The centre of the input box. -/
def center : Tensor Float [4] :=
  [0.3, -0.2, 0.1, 0.4]

/-- An `eps`-ball around `center`, in the scalar type `α`. -/
def inputBox {α : Type} [Context α] [Runtime.FromFloat α] (eps : Float) :
    Box α [4] :=
  let center : Tensor α [4] :=
    Tensor.map Runtime.ofFloat center
  let r : α := Runtime.ofFloat eps
  let radius : Tensor α [4] := Tensor.full (α := α) [4] r
  { lo := Tensor.subSpec (α := α) center radius
    hi := Tensor.addSpec (α := α) center radius }

/--
Read a one-dimensional flat box back as a scalar box, failing loudly if the dimension is not one.
-/
def scalarBox {α : Type} [Storage α] [Context α] (box : FlatBox α) :
    Except String (Box α []) := do
  if h : box.dim = 1 then
    let lowerTensor : Tensor α [1] :=
      Tensor.castShape box.lo
        (congrArg (fun extent => ([extent] : Spec.Shape)) h)
    let upperTensor : Tensor α [1] :=
      Tensor.castShape box.hi
        (congrArg (fun extent => ([extent] : Spec.Shape)) h)
    let lower : α := lowerTensor[0]
    let upper : α := upperTensor[0]
    pure { lo := Tensor.full [] lower, hi := Tensor.full [] upper }
  else
    throw s!"expected a scalar FlatBox (dim=1), got dim={box.dim}"

/--
Draw a deterministic sample using the box's scalar arithmetic and shape.

Rounding in `lo + u * (hi - lo)` can move a sample outside the endpoints. Clamp after the
interpolation; the runnable example still checks membership before evaluating the sample.
-/
def sample {α : Type} [Storage α] [Context α] {shape : Shape}
    (seed idx : Nat) (box : Box α shape) : Tensor α shape :=
  let key := rand.keyOf seed idx
  let unitSample : Tensor α shape := rand.uniform key
  let width := Tensor.subSpec box.hi box.lo
  let rawSample :=
    Tensor.addSpec box.lo (Tensor.mulSpec unitSample width)
  -- Clamp before the caller's membership check.
  let lowerClamped := Tensor.maxSpec rawSample box.lo
  Tensor.minSpec lowerClamped box.hi

/--
Run the tutorial: evaluate at the centre, propagate bounds, then check that randomly drawn samples
from the input box land inside the propagated output interval.
-/
def run (samples : Nat) : IO Unit := do
  IO.println "== One semantic universe tutorial =="
  IO.println s!"graph nodes = {graph.nodes.size}"

  let ieeeParameters : Parameters (Binary 8 23) := parameters.map Runtime.ofFloat
  let box : Box (Binary 8 23) [4] := inputBox (eps := 0.05)
  let flatInputBox : FlatBox (Binary 8 23) := { dim := 4, lo := box.lo, hi := box.hi }

  -- Evaluate at the center point.
  let referenceInputIEEE : Tensor (Binary 8 23) [4] :=
    Tensor.map Runtime.ofFloat center
  let outputAtCenter ← IO.ofExcept (evaluate ieeeParameters referenceInputIEEE)
  IO.println s!"[eval configured binary32] y(x0) = {Spec.pretty outputAtCenter}"

  -- Compute the IBP box with directed rounding via
  -- `BoundOps (FloatLib.Floats.ExecFloat.Binary 8 23)`.
  let parameters := parameterStore (α := (Binary 8 23)) ieeeParameters flatInputBox
  let ibp := runIBP graph parameters
  let outEntry ←
    match ibp[5]? with
    | some outEntry => pure outEntry
    | none => throw <| IO.userError "IBP did not produce an entry for node 5"
  let some outFlat := outEntry | throw <| IO.userError "IBP produced no output box at node 5"
  let outBox ← IO.ofExcept (scalarBox outFlat)
  IO.println s!"[IBP IEEE endpoints] lo = {Spec.pretty outBox.lo}"
  IO.println s!"[IBP IEEE endpoints] hi = {Spec.pretty outBox.hi}"

  -- Empirical consistency: random x ∈ B, check eval(x) ∈ IBP(B).
  let mut okCount : Nat := 0
  for k in [0:samples] do
    let x := sample (seed := 12345) (idx := k) box
    let inOk := Box.containsDecBool (α := (Binary 8 23)) (s := [4]) box x
    if inOk != true then
      throw <| IO.userError s!"internal error: sampled x not in box (k={k})"
    let y ← IO.ofExcept (evaluate ieeeParameters x)
    let outOk := Box.containsDecBool outBox y
    if outOk then
      okCount := okCount + 1
    else
      IO.println s!"[counterexample?] k={k}"
      IO.println s!"x = {Spec.pretty x}"
      IO.println s!"y = {Spec.pretty y}"
  IO.println s!"consistency: {okCount}/{samples} samples satisfied evalIEEE(x) ∈ IBP(B)"
  unless okCount == samples do
    throw <| IO.userError "an evaluated sample escaped the propagated interval"
  IO.println "checker theorem: `NN.MLTheory.CROWN.Box.containsDecBool_sound`"

/-- Entry point; `--samples` controls how many random points are checked against the bounds. -/
def main (args : List String) : IO Unit := do
  let args := CLI.dropDashDash args
  if CLI.hasHelp args then
    IO.println usage
    return
  let (samples?, rest) ← CLI.orThrow "OneSemanticUniverse" <| CLI.takeNatFlag? args
    "samples"
  CLI.requireNoArgs "OneSemanticUniverse" rest
  run (samples := samples?.getD 50)

end NN.Examples.DeepDives.OneSemanticUniverse
