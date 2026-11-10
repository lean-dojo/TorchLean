/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.TypedGraph.Data
public import TorchLean.NN.Tape.CPU
import TorchLean.Tensor.Tactic.Except

/-!
# Typed Graph Core

Pure and checked lowering of a shape-indexed graph onto the runtime tape, and checked forward-mode
evaluation. Applications run typed graphs through `compileChecked` in `TypedGraph.Compiled`, while
this module holds the raw-tape bridge for tape consumers and its correspondence proofs. Both the
pure `GraphData.toTape` and the checked lowering return the complete context of inputs
and intermediate values.

`GraphData` is executable data and carries no derivative-correctness certificate.
`GraphData.backwardDenseFrom_toTape` shows that
`Tape.backwardDenseFrom` on the lowered tape computes `GraphData.backpropAllCtx`. To prove that
this is the adjoint of the graph JVP, build the proof-carrying `TorchLean.Tape.Graph`,
whose nodes include their local adjointness laws.

Notes / trust boundaries:
- With `α := Float` or `α := FloatLib.Floats.ExecFloat.Binary 8 23` the engine executes, and the
  link from those runs to hardware arithmetic is a trusted interface.
- `TorchLean.Tape.Graph` needs a `CommSemiring` scalar on which the node laws can be
  proved, such as `ℝ` or `ℚ`.

## Main declarations

- `GraphData.toTape`: add one leaf per input with `addLeaves`, then record each node with
  `NodeData.toNode`, whose `backward` closure checks the cotangent's shape and runs the stored VJP.
- `lowerToTapeChecked`: validate, evaluate and record each node on the tape in one pass.
- `lowerToArrayChecked`: the same pass over an array context, which compiled code runs through
  the `@[csimp]` lemma `lowerToTapeChecked_eq_array`.
- `jvpChecked`: validate each node while evaluating the primal and tangent contexts.

See also:
- `TorchLean.TypedGraph.GraphM`, the authoring DSL for typed graphs, defined in
  `TorchLean.NN.TypedGraph.GraphM` and
  `TorchLean.NN.TypedGraph.GraphM.Apply`.
- `TorchLean.NN.Tape.Lowering` and the modules under `TorchLean/NN/Tape/Lowering/`: lowered-tape
  backpropagation equals executable graph backpropagation.
-/

@[expose] public section

namespace TorchLean
namespace Tape

open TorchLean.Tensor

/--
Extend a tape with leaf nodes for every tensor in the input context `Γ`.

Each leaf has `requiresGrad = true` and an empty backward contribution array, so the runtime loop
treats them as gradient accumulation slots but never produces parent contributions from them.
-/
def addLeaves {α : Type} [Storage α]
    (t : Tape (SomeTensor α)) : {Γ : List Shape} → TensorPack α Γ → Tape
        (SomeTensor α)
  | [], .nil => t
  | _ :: Γ, .cons x xs =>
      let (t', _id) := Tape.leaf (t := t) (.ofTensor x)
      addLeaves (t := t') (Γ := Γ) xs

/--
Runtime node produced by lowering one graph node.

The node stores the forward value and wraps the stored `vjp` in a runtime `backward` closure
that checks the shape of the upstream cotangent and emits one contribution per entry of the
node's input context, indexed from `0`. The `name` only serves debugging output.
-/
def NodeData.toNode {α : Type} {Δ : Type} [Storage α] {Γ : List Shape} {τ : Shape}
    (name : Option String) (node : NodeData α Δ Γ τ) (ctx : TensorPack α Γ)
    (d : Δ) : Node (SomeTensor α) :=
  { name := name
    value := SomeTensor.ofTensor (node.forward ctx d)
    requiresGrad := true
    parents := #[]
    backward := fun dLdyValue =>
      if h : dLdyValue.shape = τ then
        .ok (TensorPack.eraseIndexed (α := α) (ss := Γ)
          (node.vjp ctx d (dLdyValue.cast h)) 0)
      else
        .error "autograd: upstream gradient shape mismatch" }

/--
Lower an executable graph (`GraphData`) to a runtime tape by evaluating forward nodes and storing
each node's `vjp` program in its runtime `backward` closure.

PyTorch analogy: this corresponds to building a tape of autograd nodes during the forward pass,
where each node stores enough information to compute parent contributions when given an upstream
cotangent.
-/
def GraphData.toTape {α : Type} {Δ : Type} [Storage α]
  {Γ : List Shape} {ss : List Shape} (g : GraphData α Δ Γ ss) (x : TensorPack α Γ)
  (d : Δ) : Tape (SomeTensor α) × TensorPack α (Γ ++ ss) :=
  match g with
  | .nil =>
      let t := addLeaves (α := α) (t := Tape.empty) (Γ := Γ) x
      (t, Pack.cast (h := (List.append_nil Γ).symm) x)
  | .snoc (ss := ssPrev) (τ := τ) g node =>
      let (tPrev, ctxPrev) := toTape (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d
      let y := node.forward ctxPrev d
      let (tNext, _id) := Tape.addNode (t := tPrev)
        (NodeData.toNode (some "typed-graph") node ctxPrev d)
      let ctxNext :=
        Pack.cast (h := List.append_assoc Γ ssPrev [τ])
          (Pack.snoc (ss := Γ ++ ssPrev) (τ := τ) ctxPrev y)
      (tNext, ctxNext)

end Tape
end TorchLean

namespace TorchLean
namespace TypedGraph


open Tape
-- Typed context indices come from `TorchLean.NN.TypedGraph.Idx`, the one place
-- `Idx` and `getIdx` are defined.
open TorchLean (Idx)
open TensorPack (getIdx)

/-- Validate, evaluate and record each node on the runtime tape in one pass, stopping at the first
validation failure. -/
def lowerToTapeChecked {α Δ : Type} [Storage α]
    {Γ ss : List Shape} (graph : Tape.GraphData α Δ Γ ss)
    (inputs : TensorPack α Γ) (data : Δ) :
    Result
      (Tape (SomeTensor α) × TensorPack α (Γ ++ ss)) :=
  match graph with
  | .nil =>
      pure (Tape.addLeaves Tape.empty inputs,
        Pack.cast (h := (List.append_nil Γ).symm) inputs)
  | .snoc (ss := previousShapes) (τ := outputShape) previous node => do
      let (tape, context) ← lowerToTapeChecked previous inputs data
      node.validate context data
      let value := node.forward context data
      let (nextTape, _) := tape.addNode (NodeData.toNode (some "typed-graph") node context data)
      let nextContext := Pack.cast
        (h := List.append_assoc Γ previousShapes [outputShape])
        (Pack.snoc context value)
      pure (nextTape, nextContext)

/-- Successful checked lowering produces the tape and context of `GraphData.toTape`. -/
theorem lowerToTapeChecked_eq {α Δ : Type} [Storage α]
    {Γ ss : List Shape} (graph : Tape.GraphData α Δ Γ ss)
    (inputs : TensorPack α Γ) (data : Δ)
    (result : Tape (SomeTensor α) × TensorPack α (Γ ++ ss))
    (checked : lowerToTapeChecked graph inputs data = .ok result) :
    result = GraphData.toTape graph inputs data := by
  induction graph with
  | nil =>
      change Except.ok _ = Except.ok result at checked
      simpa only [GraphData.toTape] using (Except.ok.inj checked).symm
  | snoc previous node ih =>
      simp only [lowerToTapeChecked] at checked
      except_cases previousResult : lowerToTapeChecked previous inputs data
          using checked with pair =>
        obtain ⟨tape, context⟩ := pair
        have same := ih (tape, context) previousResult
        simp only [previousResult, Bind.bind, Except.bind] at checked
        except_cases validation : node.validate context data using checked with done =>
          cases done
          simp only [validation, Pure.pure, Except.pure, Except.ok.injEq] at checked
          subst result
          simp only [GraphData.toTape, ← same]

/-- Tape node of a prepared node. It stores the primal `value` computed after validation, and its
backward pulls the upstream gradient back through `prepared.vjp`. -/
def lowerPreparedNode {α : Type} [Storage α] {Γ : List Shape} {τ : Shape}
    (prepared : PreparedNode α Γ τ) (value : Tensor α τ) : Tape.Node
        (SomeTensor α) :=
  { name := some "typed-graph"
    value := SomeTensor.ofTensor value
    requiresGrad := true
    parents := #[]
    backward := fun upstream =>
      if h : upstream.shape = τ then
        .ok (TensorPack.eraseIndexed
          (prepared.vjp (upstream.cast h)) 0)
      else
        .error "autograd: upstream gradient shape mismatch" }

/-- Lowering a node prepared on the context of a pack, with its forward value, gives
`NodeData.toNode` of the node. -/
@[simp] theorem lowerPreparedNode_ofPack {α Δ : Type} [Storage α]
    {Γ : List Shape} {τ : Shape} (node : NodeData α Δ Γ τ)
    (context : TensorPack α Γ) (data : Δ) :
    lowerPreparedNode (node.prepare (TensorContext.ofPack context) data)
      (node.forward context data) = NodeData.toNode (some "typed-graph") node context data := by
  simp only [lowerPreparedNode, NodeData.toNode, NodeData.prepare_vjp_ofPack]

/-- Checked lowering over one array context, where each tape node closes over the inputs its
prepared node saved. -/
def lowerToArrayChecked {α Δ : Type} [Storage α]
    {Γ ss : List Shape} (graph : Tape.GraphData α Δ Γ ss)
    (inputs : TensorPack α Γ) (data : Δ) :
    Result (Tape (SomeTensor α) × TensorContext α
        (Γ ++ ss)) :=
  match graph with
  | .nil =>
      pure (Tape.addLeaves Tape.empty inputs,
        TensorContext.cast (List.append_nil Γ).symm (TensorContext.ofPack inputs))
  | .snoc (ss := previousShapes) (τ := outputShape) previous node => do
      let (tape, context) ← lowerToArrayChecked previous inputs data
      let prepared := node.prepare context data
      prepared.validate ()
      let value := prepared.value ()
      let (nextTape, _) := tape.addNode (lowerPreparedNode prepared value)
      pure (nextTape,
        TensorContext.cast (List.append_assoc Γ previousShapes [outputShape])
          (context.push value))

/-- Array lowering agrees with `lowerToTapeChecked` on errors, the tape and the context. -/
theorem lowerToArrayChecked_eq {α Δ : Type} [Storage α]
    {Γ ss : List Shape} (graph : Tape.GraphData α Δ Γ ss)
    (inputs : TensorPack α Γ) (data : Δ) :
    lowerToArrayChecked graph inputs data =
      (lowerToTapeChecked graph inputs data).map
        (fun result => (result.1, TensorContext.ofPack result.2)) := by
  induction graph with
  | nil =>
      simp only [lowerToArrayChecked, lowerToTapeChecked, TensorContext.cast_ofPack]
      rfl
  | snoc previous node ih =>
      simp only [lowerToArrayChecked, lowerToTapeChecked, ih]
      cases h : lowerToTapeChecked previous inputs data with
      | error message => rfl
      | ok result =>
          obtain ⟨tape, context⟩ := result
          simp only [Except.map, Bind.bind, Except.bind, NodeData.prepare_validate_ofPack,
            NodeData.prepare_value_ofPack]
          cases validation : node.validate context data with
          | error message => rfl
          | ok done =>
              cases done
              simp only [Pure.pure, Except.pure,
                TensorContext.push_ofPack, TensorContext.cast_ofPack,
                lowerPreparedNode_ofPack]

/-- `lowerToArrayChecked` with its context converted to a pack once, at the end. -/
def lowerToTapeCheckedArray {α Δ : Type} [Storage α]
    {Γ ss : List Shape} (graph : Tape.GraphData α Δ Γ ss)
    (inputs : TensorPack α Γ) (data : Δ) :
    Result
      (Tape (SomeTensor α) × TensorPack α (Γ ++ ss)) :=
  (lowerToArrayChecked graph inputs data).map (fun result => (result.1, result.2.toPack))

/-- `@[csimp]` lemma: compiled code runs `lowerToTapeChecked` through the array implementation. -/
@[csimp] theorem lowerToTapeChecked_eq_array :
    @lowerToTapeChecked = @lowerToTapeCheckedArray := by
  funext α Δ storage Γ ss graph inputs data
  simp only [lowerToTapeCheckedArray, lowerToArrayChecked_eq]
  cases lowerToTapeChecked graph inputs data with
  | error message => rfl
  | ok result =>
      obtain ⟨tape, context⟩ := result
      simp only [Except.map, TensorContext.toPack_ofPack]

/-- Validate runtime domains while evaluating the primal and tangent contexts together. -/
def jvpChecked {α Δ : Type} [Storage α]
    {Γ ss : List Shape} (graph : Tape.GraphData α Δ Γ ss)
    (inputs tangents : TensorPack α Γ) (data : Δ) :
    Result
      (TensorPack α (Γ ++ ss) × TensorPack α (Γ ++ ss)) :=
  match graph with
  | .nil =>
      pure (Pack.cast (h := (List.append_nil Γ).symm) inputs,
        Pack.cast (h := (List.append_nil Γ).symm) tangents)
  | .snoc (ss := previousShapes) (τ := outputShape) previous node => do
      let (context, tangentContext) ← jvpChecked previous inputs tangents data
      node.validate context data
      let value := node.forward context data
      let tangent := node.jvp context tangentContext data
      let extend := fun (context : TensorPack α (Γ ++ previousShapes))
          (value : Tensor α outputShape) =>
        Pack.cast (h := List.append_assoc Γ previousShapes [outputShape])
          (Pack.snoc context value)
      pure (extend context value, extend tangentContext tangent)

/-- A successful `jvpChecked` returns `graph.eval inputs data` and
`graph.jvpCtx inputs tangents data`. -/
theorem jvpChecked_eq {α Δ : Type} [Storage α]
    {Γ ss : List Shape} (graph : Tape.GraphData α Δ Γ ss)
    (inputs tangents : TensorPack α Γ) (data : Δ)
    (result : TensorPack α (Γ ++ ss) × TensorPack α (Γ ++ ss))
    (checked : jvpChecked graph inputs tangents data = .ok result) :
    result = (graph.eval inputs data, graph.jvpCtx inputs tangents data) := by
  induction graph with
  | nil =>
      change Except.ok _ = Except.ok result at checked
      simpa only [GraphData.eval,
        GraphData.jvpCtx] using (Except.ok.inj checked).symm
  | snoc previous node ih =>
      simp only [jvpChecked] at checked
      except_cases previousResult : jvpChecked previous inputs tangents data
          using checked with pair =>
        obtain ⟨context, tangentContext⟩ := pair
        have same := ih (context, tangentContext) previousResult
        simp only [previousResult, Bind.bind, Except.bind] at checked
        except_cases validation : node.validate context data using checked with done =>
          cases done
          simp only [validation, Pure.pure, Except.pure, Except.ok.injEq] at checked
          subst result
          obtain ⟨rfl, rfl⟩ := Prod.mk.inj same
          rfl

end TypedGraph
end TorchLean
