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

This concerns execution of an already-lowered graph. The convenience `IRExec.evaluate` function
lowers again on every call. Lowering itself still appends to shape lists and can take quadratic
time; repeated-execution measurements should construct the `ForwardGraph` before timing.
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

/--
A shape-indexed forward SSA graph.

Unlike autograd `GraphData`, this representation has no JVP or VJP fields. It is therefore
impossible to request derivatives from an artifact produced by the forward IR lowering pass.
-/
inductive ForwardData (α : Type) [TorchLean.Storage α] (Γ : List Shape) : List Shape → Type where
  /-- A graph with no computed nodes. -/
  | nil : ForwardData α Γ []
  /-- Append a node that may read the graph input and every preceding result. -/
  | snoc {ss : List Shape} {τ : Shape} :
      ForwardData α Γ ss → ForwardNode α (Γ ++ ss) τ → ForwardData α Γ (ss ++ [τ])

namespace ForwardData

/-- Evaluate every node and return the input followed by all intermediate values. -/
def eval {α : Type} [TorchLean.Storage α] {Γ ss : List Shape}
    (g : ForwardData α Γ ss) (x : TorchLean.TensorPack α Γ) : TorchLean.TensorPack α (Γ ++ ss) :=
  match g with
  | .nil => TorchLean.TensorPack.cast (α := α) (h := (List.append_nil Γ).symm) x
  | .snoc (ss := ss) (τ := τ) g node =>
      let ctx := eval g x
      let y := node.eval ctx
      TorchLean.TensorPack.cast (α := α) (h := List.append_assoc Γ ss [τ])
        (TorchLean.TensorPack.snoc (α := α) (ss := Γ ++ ss) (τ := τ) ctx y)

end ForwardData

namespace Internal

/-- Evaluate with array reads and one append per node, retaining a proof of the context shapes. -/
def evalArray {α : Type} [Storage α] {Γ ss : List Shape}
    (g : ForwardData α Γ ss) (x : TensorPack α Γ) : ContextArray α (Γ ++ ss) :=
  match g with
  | .nil => (ContextArray.ofPack x).cast (List.append_nil Γ).symm
  | .snoc (ss := ss) g node =>
      let ctx := evalArray g x
      let y := node.run ctx.reader
      (ctx.push y).cast (List.append_assoc Γ ss _)

/-- The array evaluator represents exactly the original typed result, for every node closure. -/
theorem evalArray_eq {α : Type} [Storage α] {Γ ss : List Shape}
    (g : ForwardData α Γ ss) (x : TensorPack α Γ) :
    evalArray g x = ContextArray.ofPack (ForwardData.eval g x) := by
  induction g with
  | nil => simp [evalArray, ForwardData.eval]
  | snoc g node ih => simp [evalArray, ForwardData.eval, ih]

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
