/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.IR.Semantics -- shake: keep
public import NN.Proofs.Autograd.Runtime.Link -- shake: keep
public import NN.Runtime.Autograd.IRExec.Context

/-!
# Forward IR Execution

This module validates an op-tagged `NN.IR.Graph` and translates its nodes into the shape-indexed
`ForwardData` representation used for evaluation. `ForwardGraph` packages that result with its
input and intermediate shapes.

The translation is forward-only. It neither supplies derivative rules nor performs optimization,
fusion, scheduling, or native code generation. The semantic-equivalence theorem for the supported
IR fragment lives in `NN.Runtime.Autograd.IRExec.Correctness.SemanticEquivalence` so ordinary
runtime imports do not pull in that proof.

## Main declarations

- `ForwardGraph` packages a forward-only graph lowered from `NN.IR.Graph`.
- `Internal.packedTensorsOfContext` converts typed runtime contexts back into IR-style value arrays.
- `Internal.buildFrom` is the lowering pass from `NN.IR.Graph` to executable graph data.
- `lowerToForwardGraph` is the public lowering entry point.

Numeric IR node identifiers are converted through checked typed indices (`Idx`). The resulting
types contain no derivative operations: lowering to `ForwardGraph` cannot be mistaken for an
autograd lowering.

Node closures use `TensorReader` to select parents. `ForwardNode.eval` and `ForwardData.eval`
retain their typed-pack semantics, while proved compiler simplification rules execute the graph
with an array context. Each node appends one value; conversion to a typed pack happens only when
the caller requests that representation. `ForwardGraph.denoteAll` returns the array directly.

Internally, shape prefixes are stored in reverse order, so each node shares its predecessor's
shape list. The public shape indices and value tables remain in node-id order. The checked lowering
loop uses an array for parent-shape lookup and materializes the public shape list once at the end.
The convenience `IRExec.evaluate` function lowers again on every call; repeated execution can reuse
the resulting `ForwardGraph`.
-/

@[expose] public section


namespace Runtime
namespace Autograd
namespace IRExec

open Spec TorchLean
open TorchLean TorchLean.Tensor
open Proofs.Autograd.Algebra
open NN.IR
-- Typed context indices come from `NN.Proofs.Autograd.Tape.Util.Idx`, the one place
-- `Idx` and `getIdx` are defined.
open Proofs (Idx getIdx)

/--
`simp` rule for `Except`-`do` chains: binding an `.ok` value is just function application.
-/
@[simp] theorem Except.ok_bind {ε α β : Type} (a : α) (f : α → Except ε β) :
    (Except.ok a >>= f) = f a := by
  simp [Bind.bind, Except.bind]

/--
`simp` rule for `Except`-`do` chains: binding an `.error` short-circuits.

Used heavily when discharging impossible branches in lowering correctness proofs.
-/
@[simp] theorem Except.error_bind {ε α β : Type} (e : ε) (f : α → Except ε β) :
    (Except.error e >>= f) = Except.error e := by
  simp [Bind.bind, Except.bind]


/--
One forward-only SSA node over the typed context `Γ`.

The `run` closure reads its inputs through a `TensorReader` using `readTensor`.
The `eval` method evaluates the node on a `TensorPack` with the same typed result.
-/
structure ForwardNode (α : Type) [TorchLean.Storage α] (Γ : List Shape) (τ : Shape) where
  /-- Evaluate the node using typed reads of the input and preceding node values. -/
  run : TensorReader α Γ → Tensor α τ

/-- Evaluate a node on a typed pack, as in the logical forward semantics. -/
@[simp] def ForwardNode.eval {α : Type} [Storage α] {Γ : List Shape} {τ : Shape}
    (node : ForwardNode α Γ τ) (ctx : TensorPack α Γ) : Tensor α τ :=
  node.run (TensorReader.ofPack ctx)

namespace Internal

/--
Forward nodes indexed by their reversed output shapes.

The implicit list index is a runtime constructor field. Consing the new shape shares all previous
prefixes; storing chronological lists here would retain a separate copy of every prefix.
-/
inductive ReverseData (α : Type) [Storage α] (Γ : List Shape) : List Shape → Type where
  /-- A graph with no computed nodes. -/
  | nil : ReverseData α Γ []
  /-- Append a node that may read the graph input and every preceding result. -/
  | snoc {rev : List Shape} {τ : Shape} :
      ReverseData α Γ rev → ForwardNode α (Γ ++ rev.reverse) τ →
        ReverseData α Γ (τ :: rev)

namespace ReverseData

/-- Typed chronological evaluation of a graph with shared reversed shape prefixes. -/
def eval {α : Type} [Storage α] {Γ rev : List Shape}
    (g : ReverseData α Γ rev) (x : TensorPack α Γ) : TensorPack α (Γ ++ rev.reverse) :=
  match g with
  | .nil => TensorPack.cast (List.append_nil Γ).symm x
  | .snoc g node =>
      let ctx := eval g x
      TensorPack.cast (by simp [List.reverse_cons, List.append_assoc])
        (ctx.snoc (node.eval ctx))

/-- Evaluate with array reads and one append per node, retaining a proof of the context shapes. -/
def evalArray {α : Type} [Storage α] {Γ rev : List Shape}
    (g : ReverseData α Γ rev) (x : TensorPack α Γ) : ContextArray α (Γ ++ rev.reverse) :=
  match g with
  | .nil => (ContextArray.ofPack x).cast (List.append_nil Γ).symm
  | .snoc g node =>
      let ctx := evalArray g x
      let y := node.run ctx.reader
      (ctx.push y).cast (by simp [List.reverse_cons, List.append_assoc])

/-- The array evaluator represents exactly the original typed result, for every node closure. -/
theorem evalArray_eq {α : Type} [Storage α] {Γ rev : List Shape}
    (g : ReverseData α Γ rev) (x : TensorPack α Γ) :
    evalArray g x = ContextArray.ofPack (eval g x) := by
  induction g with
  | nil => simp [evalArray, eval]
  | snoc g node ih => simp [evalArray, eval, ih]

end ReverseData
end Internal

/--
A shape-indexed forward SSA graph with output shapes `ss` in node-id order.

The executable nodes share reversed shape prefixes internally. `nil`, `snoc`, and `eval` retain
the chronological typed interface. Unlike autograd `GraphData`, this representation has no JVP
or VJP fields.
-/
structure ForwardData (α : Type) [Storage α] (Γ ss : List Shape) where
  /-- Output shapes in reverse node-id order, shared with the final node. -/
  revShapes : List Shape
  /-- Forward closures indexed by the shared reversed prefixes. -/
  body : Internal.ReverseData α Γ revShapes
  /-- The internal order represents the public chronological shape index. -/
  shape_eq : revShapes.reverse = ss

namespace ForwardData

variable {α : Type} [Storage α] {Γ ss : List Shape}

/-- A graph with no computed nodes. -/
def nil : ForwardData α Γ [] := ⟨[], .nil, rfl⟩

/-- Append a node while sharing the existing graph and its reversed shape prefix. -/
def snoc {τ : Shape} (g : ForwardData α Γ ss) (node : ForwardNode α (Γ ++ ss) τ) :
    ForwardData α Γ (ss ++ [τ]) :=
  ⟨τ :: g.revShapes, .snoc g.body (g.shape_eq.symm ▸ node), by simp [g.shape_eq]⟩

/-- Evaluate every node and return the input followed by all intermediate values. -/
def eval (g : ForwardData α Γ ss) (x : TensorPack α Γ) : TensorPack α (Γ ++ ss) :=
  TensorPack.cast (congrArg (Γ ++ ·) g.shape_eq) (g.body.eval x)

/-- The empty graph returns the input context. -/
@[simp] theorem eval_nil (x : TensorPack α Γ) :
    eval (nil : ForwardData α Γ []) x = TensorPack.cast (List.append_nil Γ).symm x := rfl

/-- Appending a node preserves the original chronological typed evaluation equation. -/
@[simp] theorem eval_snoc {τ : Shape} (g : ForwardData α Γ ss)
    (node : ForwardNode α (Γ ++ ss) τ) (x : TensorPack α Γ) :
    eval (snoc g node) x = TensorPack.cast (List.append_assoc Γ ss [τ])
      ((eval g x).snoc (node.eval (eval g x))) := by
  rcases g with ⟨rev, body, rfl⟩
  simp only [snoc, eval, Internal.ReverseData.eval, TensorPack.cast_rfl]
  exact TensorPack.cast_cast _ _ _

end ForwardData

namespace Internal

/-- Evaluate a public forward graph using the array implementation of its shared internal data. -/
def evalArray {α : Type} [Storage α] {Γ ss : List Shape}
    (g : ForwardData α Γ ss) (x : TensorPack α Γ) : ContextArray α (Γ ++ ss) :=
  (g.body.evalArray x).cast (congrArg (Γ ++ ·) g.shape_eq)

/-- Public array evaluation agrees with the typed semantics for every forward graph. -/
theorem evalArray_eq {α : Type} [Storage α] {Γ ss : List Shape}
    (g : ForwardData α Γ ss) (x : TensorPack α Γ) :
    evalArray g x = ContextArray.ofPack (ForwardData.eval g x) := by
  simp [evalArray, ReverseData.evalArray_eq, ForwardData.eval]

/-- Recover a typed pack only after all node executions have finished. -/
def evalWithArray {α : Type} [Storage α] {Γ ss : List Shape}
    (g : ForwardData α Γ ss) (x : TensorPack α Γ) : TensorPack α (Γ ++ ss) :=
  (evalArray g x).toPack

end Internal

/-- Compile the typed forward evaluator using its proved array implementation. -/
@[csimp] theorem ForwardData.eval_eq_evalWithArray :
    @ForwardData.eval = @Internal.evalWithArray := by
  funext α inst Γ ss g x
  simp [Internal.evalWithArray, Internal.evalArray_eq]

/--
A forward-executable SSA graph derived from an `NN.IR.Graph`.

The lowered graph stores:
- one distinguished input shape (`inShape`),
- one shape per lowered node (`ss`, corresponding to IR node ids `1..n-1`),
- and forward-only node closures (`body`) consumed by `ForwardData.eval`.
-/
structure ForwardGraph (α : Type) [TorchLean.Storage α] where
  /-- The distinguished IR input node’s shape (node id 0). -/
  inShape : Shape
  /-- Shapes of the IR nodes 1..(n-1) (one per executable SSA node). -/
  ss : List Shape
  /-- Forward SSA/DAG for nodes 1..(n-1); inputs live in `Γ := [inShape]`. -/
  body : ForwardData α [inShape] ss

namespace ForwardGraph

variable {α : Type} [TorchLean.Storage α]

/--
Evaluate the lowered forward graph on a concrete input tensor.

The result is the full typed runtime context `[inShape] ++ ss`, i.e. input followed by every
lowered node value in topological order.
-/
def eval (e : ForwardGraph α) (x : Tensor α e.inShape) :
    TorchLean.TensorPack α ([e.inShape] ++ e.ss) :=
  ForwardData.eval (α := α) (Γ := [e.inShape]) (ss := e.ss) e.body (.cons x .nil)

end ForwardGraph

/-!
## Denotation Table Helper

`ForwardGraph.eval` produces a typed runtime context `TorchLean.TensorPack α ([inShape] ++ ss)`.

For debugging and for the forward-correctness development in
`NN.Runtime.Autograd.IRExec.Correctness`,
we provide a helper that erases this context
into an IR-style value table `Array (Spec.SomeTensor α)` in node-id order.
-/

namespace Internal

/--
Convert a typed runtime context `TorchLean.TensorPack α ss` into an IR-style value table.

This is phrased in terms of `Array (Spec.SomeTensor α)` because the IR denotation functions
(`denoteAll*`) are array-based, while forward-graph execution evaluates into a typed context
(`TorchLean.TensorPack`).
-/
def packedTensorsOfContext {α : Type} [TorchLean.Storage α] {ss : List Shape}
    (ctx : TorchLean.TensorPack α ss) : Array (Spec.SomeTensor α) :=
  TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := ss) ctx

end Internal

namespace ForwardGraph

variable {α : Type} [TorchLean.Storage α] [Context α]

/--
Convert the full evaluated context into an IR-style value table (one `Spec.SomeTensor` per node id).

This is the bridge used to compare forward-graph evaluation with `NN.IR.Graph.denoteAll*`.
-/
def denoteAll (e : Runtime.Autograd.IRExec.ForwardGraph α)
    (x : Tensor α e.inShape) : Array (Spec.SomeTensor α) :=
  Internal.packedTensorsOfContext (α := α) (ss := [e.inShape] ++ e.ss)
    (Runtime.Autograd.IRExec.ForwardGraph.eval e x)

/-- Return the evaluated array directly, avoiding a pack conversion at the untyped boundary. -/
def denoteAllWithArray (e : ForwardGraph α) (x : Tensor α e.inShape) :
    Array (Spec.SomeTensor α) :=
  (Internal.evalArray e.body (.cons x .nil)).values

/-- Array denotation preserves the typed logical evaluator and every intermediate value. -/
@[csimp] theorem denoteAll_eq_denoteAllWithArray : @denoteAll = @denoteAllWithArray := by
  funext α inst e x
  simp [denoteAllWithArray, Internal.evalArray_eq, denoteAll, ForwardGraph.eval,
    Internal.packedTensorsOfContext]

end ForwardGraph

end IRExec
end Autograd
end Runtime
