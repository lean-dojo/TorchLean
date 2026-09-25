/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.IRExec.API

/-!
# IRExec Scaling Semantic Regressions

Check the compiled lowering path against the independent IR evaluator, including every intermediate
value, heterogeneous shapes, shared parents, deterministic random nodes, and a nonempty prefix.
The Float runtime suite runs these checks; the dedicated scaling driver also exposes `--check`.
-/

public section

open Spec TorchLean
open Runtime.Autograd.IRExec

namespace Tests.IRExecScalingRegression

private def unwrap {α : Type} (result : Except String α) : IO α :=
  match result with
  | .ok value => pure value
  | .error error => throw <| IO.userError error

private def checkValues (label : String) (actual expected : Array (Spec.SomeTensor Float)) :
    IO Unit := do
  unless actual.size == expected.size do
    throw <| IO.userError s!"{label}: value-table lengths differ"
  for i in [:expected.size] do
    let some a := actual[i]? | throw <| IO.userError s!"{label}: missing actual node {i}"
    let some b := expected[i]? | throw <| IO.userError s!"{label}: missing expected node {i}"
    unless a.shape == b.shape && a.tensor.data.size == b.tensor.data.size do
      throw <| IO.userError s!"{label}: shape or length mismatch at node {i}"
    for j in [:b.tensor.data.size] do
      unless a.tensor.data[j]!.toBits == b.tensor.data[j]!.toBits do
        throw <| IO.userError s!"{label}: bit mismatch at node {i}, entry {j}"

private def checkForward {shape : Shape} (label : String) (graph : NN.IR.Graph)
    (exec : ForwardGraph Float) (x : Tensor Float shape)
    (expected : Array (Spec.SomeTensor Float)) : IO Unit := do
  unless exec.ss == (graph.nodes.toList.drop 1).map (·.outShape) do
    throw <| IO.userError s!"{label}: chronological shape list differs"
  if h : shape = exec.inShape then
    let input := Tensor.castShape x h
    checkValues label (exec.denoteAll input) expected
    checkValues s!"{label} typed pack" (exec.eval input).toShapeErasedArray expected
  else
    throw <| IO.userError s!"{label}: input shape differs"

private def checkGraph {shape : Shape} (label : String) (graph : NN.IR.Graph)
    (payload : NN.IR.Payload Float) (x : Tensor Float shape) : IO Unit := do
  let expected ← unwrap <| graph.denoteAll payload (Spec.SomeTensor.ofTensor x)
  let exec ← unwrap <| lowerToForwardGraph graph payload
  checkForward label graph exec x expected

private def mixedGraph : NN.IR.Graph :=
  { nodes := #[
      { id := 0, parents := #[], kind := .input, outShape := [4] },
      { id := 1, parents := #[0], kind := .relu, outShape := [4] },
      { id := 2, parents := #[1], kind := .sum, outShape := .scalar },
      { id := 3, parents := #[2], kind := .broadcastTo .scalar [4], outShape := [4] },
      { id := 4, parents := #[0, 3], kind := .sub, outShape := [4] },
      { id := 5, parents := #[4], kind := .sum, outShape := .scalar },
      { id := 6, parents := #[2, 5], kind := .add, outShape := .scalar },
      { id := 7, parents := #[0], kind := .detach, outShape := [4] },
      { id := 8, parents := #[7, 1], kind := .concat 0, outShape := [8] },
      { id := 9, parents := #[8], kind := .reshape [8] [2, 4], outShape := [2, 4] },
      { id := 10, parents := #[9], kind := .transpose 0 1, outShape := [4, 2] },
      { id := 11, parents := #[10], kind := .permute #[1, 0], outShape := [2, 4] },
      { id := 12, parents := #[11], kind := .reshape [2, 4] [8], outShape := [8] },
      { id := 13, parents := #[12], kind := .sum, outShape := .scalar }
    ] }

private def checkPrefix (x : Tensor Float [4]) : IO Unit := do
  let prefixGraph : NN.IR.Graph := { nodes := mixedGraph.nodes.extract 0 4 }
  let exec ← unwrap <| lowerToForwardGraph (α := Float) prefixGraph {}
  let state ← unwrap <| Internal.buildFrom mixedGraph {} exec.inShape 4 ⟨exec.ss, exec.body⟩
  let extended : ForwardGraph Float :=
    { inShape := exec.inShape, ss := state.1, body := state.2 }
  let expected ← unwrap <| mixedGraph.denoteAll {} (Spec.SomeTensor.ofTensor x)
  checkForward "nonempty prefix" mixedGraph extended x expected
  -- Starting beyond the graph must preserve even a heterogeneous existing prefix.
  let unchanged ← unwrap <|
    Internal.buildFrom mixedGraph {} extended.inShape (mixedGraph.nodes.size + 3)
      ⟨extended.ss, extended.body⟩
  checkForward "finished prefix" mixedGraph
    { inShape := extended.inShape, ss := unchanged.1, body := unchanged.2 } x expected

private def checkRandom : IO Unit := do
  let graph : NN.IR.Graph :=
    { nodes := #[
        { id := 0, parents := #[], kind := .input, outShape := .scalar },
        { id := 1, parents := #[], kind := .randUniform 17, outShape := [4] },
        { id := 2, parents := #[0], kind := .bernoulliMask 17, outShape := [4] },
        { id := 3, parents := #[], kind := .const [4], outShape := [4] },
        { id := 4, parents := #[1, 3], kind := .add, outShape := [4] },
        { id := 5, parents := #[4, 2], kind := .mulElem, outShape := [4] }
      ] }
  let payload : NN.IR.Payload Float :=
    { const? := fun id =>
        if id == 3 then some ⟨4, Tensor.full [4] 0.25⟩ else none }
  checkGraph "random and payload" graph payload (Tensor.scalar 0.5)

private def checkIndices : IO Unit := do
  let ss : List Shape := [Shape.scalar, [2, 2], [0]]
  let ctx := Internal.ShapeArray.ofList ([([4] : Shape)] ++ ss)
  for id in [0, 1, 2, 3, 4, 100] do
    for shape in [([4] : Shape), Shape.scalar, [2, 2], [0], [5]] do
      match ctx.mkIdx id shape, Internal.mkIdx [4] ss id shape with
      | .ok a, .ok b =>
          unless a.i.val == b.i.val do
            throw <| IO.userError "array lookup returned a different parent id"
      | .error a, .error b =>
          unless a == b do
            throw <| IO.userError "array lookup changed the diagnostic"
      | _, _ => throw <| IO.userError "array lookup changed success or failure"

private def checkError {α : Type} (expected : String) (result : Except String α) : IO Unit :=
  match result with
  | .ok _ => throw <| IO.userError s!"expected lowering failure: {expected}"
  | .error actual =>
      unless actual == expected do
        throw <| IO.userError s!"expected {expected}, got {actual}"

private def checkFailures : IO Unit := do
  let wrongShape : NN.IR.Graph :=
    { nodes := #[
        { id := 0, parents := #[], kind := .input, outShape := [4] },
        { id := 1, parents := #[0], kind := .relu, outShape := [5] }
      ] }
  checkError "IRExec: shape mismatch at id=0: expected [5], got [4]"
    (lowerToForwardGraph (α := Float) wrongShape {})
  let missingPayload : NN.IR.Graph :=
    { nodes := #[
        { id := 0, parents := #[], kind := .input, outShape := [4] },
        { id := 1, parents := #[], kind := .const [4], outShape := [4] }
      ] }
  checkError "IR eval: missing const payload for node 1"
    (lowerToForwardGraph (α := Float) missingPayload {})
  -- The internal entry point may start before its context contains a valid graph parent.
  checkError "IRExec: invalid id=1 for ctxLen=1"
    (Internal.buildFrom (α := Float) mixedGraph {} [4] 2 ⟨[], .nil⟩)

/-- Check compiled lowering against independent value tables and exact failure diagnostics. -/
def check : IO Unit := do
  let x : Tensor Float [4] :=
    Tensor.Internal.Rep.ofFlatFn fun i => #[-3.0, 2.0, -0.0, 7.0][i.val]!
  checkGraph "input only" { nodes := mixedGraph.nodes.extract 0 1 } {} x
  checkGraph "mixed shapes and shared parents" mixedGraph {} x
  checkPrefix x
  checkRandom
  checkIndices
  checkFailures
  IO.println "IRExec scaling semantic regressions passed"

end Tests.IRExecScalingRegression
