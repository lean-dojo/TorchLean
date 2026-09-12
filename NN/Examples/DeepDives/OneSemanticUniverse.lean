/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API
public import NN.IR.Semantics
public import NN.MLTheory.CROWN.Extras.FP32
public import NN.Floats.Float32

/-!
# One semantic universe

End-to-end “one semantic universe” tutorial (single graph, many semantics, one checker).

This tutorial lives under `NN/Examples/DeepDives` because it connects execution, interval semantics,
and checker soundness in one graph.

We build one medium IR graph:

$$
x\mapsto\tanh\!\left(\operatorname{sum}
  \left(\operatorname{Linear}_2(\operatorname{ReLU}(\operatorname{Linear}_1(x)))\right)\right)
$$

and then:
1) evaluate it under multiple scalar semantics (`ℝ`, `FP32`, `IEEE32Exec`);
2) run IBP under multiple interval semantics (endpoints in `ℝ`, `FP32`, `IEEE32Exec` with directed
  rounding);
3) empirically check that `evalIEEE(G,x)` lies in the IBP output box for random $x\in B$;
4) point to the Lean theorem that the Boolean checker is sound (`Box.containsDecBool_sound`).

Notes:
- `ℝ` and `FP32` instantiations are proof-oriented and noncomputable (they typecheck, but do not run
  as an executable).
- `IEEE32Exec` is fully executable inside Lean, so we use it for the runnable consistency check.

Run:
  `lake exe torchlean one_semantic_universe --samples 50`
-/

@[expose] public section

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
    , "  lake exe torchlean one_semantic_universe [options]"
    , ""
    , "Options:"
    , "  --samples N"
    ]

/-- Four input features. -/
def inputWidth : Nat := 4
/-- Five hidden units. -/
def hiddenWidth : Nat := 5
/-- Three outputs, then summed and squashed to a scalar. -/
def outputWidth : Nat := 3

/-- Shape of one input vector. -/
abbrev input : Spec.Shape := [inputWidth]
/-- Shape of the hidden activation. -/
abbrev hiddenShape : Spec.Shape := [hiddenWidth]
/-- Shape of the second layer's output, before the reduction. -/
abbrev output : Spec.Shape := [outputWidth]

/-- First weight matrix, `[out, in]` as in PyTorch. -/
abbrev hiddenWeightShape : Spec.Shape := [hiddenWidth, inputWidth]
/-- First bias. -/
abbrev hiddenBiasShape : Spec.Shape := [hiddenWidth]
/-- Second weight matrix. -/
abbrev outputWeightShape : Spec.Shape := [outputWidth, hiddenWidth]
/-- Second bias. -/
abbrev outputBiasShape : Spec.Shape := [outputWidth]

/--
The network's parameters, generic in the scalar type.

Being generic in `α` is the whole point of this tutorial: one parameter record is transported to
`Float`, to the bit-level IEEE model, to `ℝ` and to the rounded-real `FP32`, and the same graph is
evaluated in each.
-/
structure Parameters (α : Type) [Storage α] where
  /-- Weight matrix for layer 1. -/
  hiddenWeight : Tensor α hiddenWeightShape
  /-- Bias for layer 1. -/
  hiddenBias : Tensor α hiddenBiasShape
  /-- Weight matrix for layer 2. -/
  outputWeight : Tensor α outputWeightShape
  /-- Bias for layer 2. -/
  outputBias : Tensor α outputBiasShape

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

These are the only numbers in the file; every other instantiation is obtained from them by `map`, so
the four semantics are guaranteed to be looking at the same network.
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
The network as six IR nodes: input, linear, ReLU, linear, sum, tanh.

Written out node by node rather than built by a combinator, so the reader can see exactly what the
evaluator and the interval propagator are given.
-/
def graph : NN.IR.Graph :=
  let inputNode : NN.IR.Node :=
    { id := 0, parents := #[], kind := .input, outShape := input }
  let hiddenLinearNode : NN.IR.Node :=
    { id := 1, parents := #[0], kind := .linear, outShape := hiddenShape }
  let hiddenActivationNode : NN.IR.Node :=
    { id := 2, parents := #[1], kind := .relu, outShape := hiddenShape }
  let outputLinearNode : NN.IR.Node :=
    { id := 3, parents := #[2], kind := .linear, outShape := output }
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
          outDim := hiddenWidth
          inDim := inputWidth
          W := parameters.hiddenWeight
          b := parameters.hiddenBias
        }
      else if id = 3 then
        some {
          outDim := outputWidth
          inDim := hiddenWidth
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
          m := hiddenWidth
          n := inputWidth
          w := parameters.hiddenWeight
          b := parameters.hiddenBias
        }
        |>.insert 3 {
          m := outputWidth
          n := hiddenWidth
          w := parameters.outputWeight
          b := parameters.outputBias
        } }

/-- Evaluate the graph at scalar type `α` and check that the result really is a scalar. -/
def evaluateOutput
    {α : Type} [Storage α] [Context α]
    (parameters : Parameters α) (inputTensor : Tensor α input) :
    Except String (Tensor α []) :=
      do
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
noncomputable def evaluateReal
    (parameters : Parameters ℝ) (input : Tensor ℝ input) :
    Except String (Tensor ℝ []) :=
  evaluateOutput (α := ℝ) parameters input

/--
Evaluation over rounded reals: each operation rounds to nearest binary32, but the carrier is still
`ℝ`, which is what makes the error proofs possible.
-/
noncomputable def evaluateFP32
    (parameters : Parameters TorchLean.Floats.FP32)
    (input : Tensor TorchLean.Floats.FP32 input) :
    Except String (Tensor TorchLean.Floats.FP32 []) :=
  evaluateOutput (α := TorchLean.Floats.FP32) parameters input

/-- Interval bound propagation over the reals. -/
noncomputable def propagateRealBounds
    (parameters : ParamStore ℝ) : Array (Option (FlatBox ℝ)) :=
  runIBP (α := ℝ) graph parameters

/-- The same propagation over rounded reals. -/
noncomputable def propagateFP32Bounds
    (parameters : ParamStore TorchLean.Floats.FP32) :
    Array (Option (FlatBox TorchLean.Floats.FP32)) :=
  runIBP (α := TorchLean.Floats.FP32) graph parameters

end ProofOnly

/-- The centre of the input box. -/
def referenceInputFloat : Tensor Float input :=
  [0.3, -0.2, 0.1, 0.4]

/-- An `eps`-ball around `referenceInputFloat`, in the scalar type `α`. -/
def inputBoxOf (α : Type) [Context α] [Runtime.FromFloat α] (eps : Float) :
    Box α input :=
  let center : Tensor α input :=
    Tensor.map Runtime.ofFloat referenceInputFloat
  let r : α := Runtime.ofFloat eps
  let radius : Tensor α input := Tensor.full (α := α) input r
  { lo := Tensor.subSpec (α := α) center radius
    hi := Tensor.addSpec (α := α) center radius }

/-- Present an input box in the flat form the propagator consumes. -/
def flattenInputBox {α : Type} [Storage α] [Context α]
    (box : Box α input) : FlatBox α :=
  { dim := inputWidth, lo := box.lo, hi := box.hi }

/--
Read a one-dimensional flat box back as a scalar box, failing loudly if the dimension is not one.
-/
def scalarBoxOfFlat (box : FlatBox IEEE32Exec) :
    Except String (Box IEEE32Exec []) :=
  do
  if h : box.dim = 1 then
    let lowerTensor : Tensor IEEE32Exec [1] :=
      Tensor.castShape box.lo
        (congrArg (fun extent => ([extent] : Spec.Shape)) h)
    let upperTensor : Tensor IEEE32Exec [1] :=
      Tensor.castShape box.hi
        (congrArg (fun extent => ([extent] : Spec.Shape)) h)
    let lower : IEEE32Exec := lowerTensor[0]
    let upper : IEEE32Exec := upperTensor[0]
    pure { lo := Tensor.full [] lower, hi := Tensor.full [] upper }
  else
    throw s!"expected a scalar FlatBox (dim=1), got dim={box.dim}"

/--
Draw one sample from an input box under the bit-level IEEE model.

The final clamp is not cosmetic: `lo + u * (hi - lo)` is computed in binary32, so rounding can push
the result a fraction of an ulp outside the box. Clamping makes the sample genuinely a member of the
box, which is what the enclosure check below assumes.
-/
def sampleInBoxIEEE (seed idx : Nat) (box : Box IEEE32Exec input) :
    Tensor IEEE32Exec input :=
  let key := rand.keyOf seed idx
  let unitSample : Tensor IEEE32Exec input :=
    rand.uniform (α := IEEE32Exec) key (s := input)
  let width := Tensor.subSpec (α := IEEE32Exec) box.hi box.lo
  let rawSample :=
    Tensor.addSpec (α := IEEE32Exec) box.lo
      (Tensor.mulSpec (α := IEEE32Exec) unitSample width)
  -- Clamp to be sure we land inside `[lo,hi]` despite rounding.
  let lowerClamped := Tensor.maxSpec (α := IEEE32Exec) rawSample box.lo
  Tensor.minSpec (α := IEEE32Exec) lowerClamped box.hi

/--
Run the tutorial: evaluate at the centre, propagate bounds, then check that randomly drawn samples
from the input box land inside the propagated output interval.
-/
def showIEEECheck (samples : Nat) : IO Unit := do
  IO.println "== One semantic universe tutorial =="
  IO.println s!"graph nodes = {graph.nodes.size}"

  let ieeeParameters : Parameters IEEE32Exec := floatParameters.map Runtime.ofFloat
  let inputBox : Box IEEE32Exec input := inputBoxOf (α := IEEE32Exec) (eps := 0.05)
  let flatInputBox : FlatBox IEEE32Exec := flattenInputBox (α := IEEE32Exec) inputBox

  -- Evaluate at the center point.
  let referenceInputIEEE : Tensor IEEE32Exec input :=
    Tensor.map Runtime.ofFloat referenceInputFloat
  match evaluateOutput (α := IEEE32Exec) ieeeParameters referenceInputIEEE with
  | .error msg => throw <| IO.userError msg
  | .ok outputAtCenter =>
      IO.println s!"[eval IEEE32Exec] y(x0) = {Spec.pretty outputAtCenter}"

  -- Compute IBP box (IEEE endpoints with directed rounding via `BoundOps IEEE32Exec`).
  let parameters := parameterStore (α := IEEE32Exec) ieeeParameters flatInputBox
  let ibp := runIBP (α := IEEE32Exec) graph parameters
  let outEntry ←
    match ibp[5]? with
    | some outEntry => pure outEntry
    | none => throw <| IO.userError "IBP did not produce an entry for node 5"
  let some outFlat := outEntry | throw <| IO.userError "IBP produced no output box at node 5"
  let outBox ←
    match scalarBoxOfFlat outFlat with
    | .error msg => throw <| IO.userError msg
    | .ok b => pure b
  IO.println s!"[IBP IEEE endpoints] lo = {Spec.pretty outBox.lo}"
  IO.println s!"[IBP IEEE endpoints] hi = {Spec.pretty outBox.hi}"

  -- Empirical consistency: random x ∈ B, check eval(x) ∈ IBP(B).
  let mut okCount : Nat := 0
  for k in [0:samples] do
    let x := sampleInBoxIEEE (seed := 12345) (idx := k) inputBox
    let inOk := Box.containsDecBool (α := IEEE32Exec) (s := input) inputBox x
    if inOk != true then
      throw <| IO.userError s!"internal error: sampled x not in box (k={k})"
    match evaluateOutput (α := IEEE32Exec) ieeeParameters x with
    | .error msg => throw <| IO.userError msg
    | .ok y =>
        let outOk :=
          Box.containsDecBool (α := IEEE32Exec) (s := []) outBox y
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
  showIEEECheck (samples := samples?.getD 50)

end NN.Examples.DeepDives.OneSemanticUniverse
