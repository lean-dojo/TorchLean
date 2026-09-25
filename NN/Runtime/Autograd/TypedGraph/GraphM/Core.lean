/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Tape.Algebra.Soundness
public import NN.Runtime.Autograd.Engine.Core.Base
public import NN.Spec.Core.Random

/-!
# GraphM Core

Typed variables, builder state, input binding, constants, random nodes, and detach for the
executable typed graph authoring API.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace TypedGraph
namespace GraphM

open Spec TorchLean
open TorchLean TorchLean.Tensor
open TorchLean
open Proofs.Autograd.Algebra
-- Typed context indices come from `NN.Proofs.Autograd.Tape.Util.Idx`, the one place
-- `Idx` and `getIdx` are defined.
open Proofs (Idx getIdx)

/--
A typed handle to a value in the growing graph context.

`Var s` carries its expected `Shape` at the type level, while `id` is the runtime index into the
concatenated context `Γ ++ ss`.
-/
structure Var (s : Shape) where
  /--
  Runtime id of the value inside the concatenated context `Γ ++ ss`.

  The shape index on `Var s` is the static guarantee; this numeric id is the executable handle used
  when constructing `Idx` proofs for `GraphData` nodes.
  -/
  id : Nat
deriving Repr

/-!
`GraphM.arg` is correct but a little noisy for examples (you must repeat the index and shape).

`VarList` + `args` give a typed variable layer: `args` returns one `Var` per entry in `Γ`,
in order, without spelling indices.
-/

/--
Dependent list of typed variables, aligned with a list of shapes.

`VarList Γ` contains exactly one `Var s` for each `s ∈ Γ`, in order.
-/
inductive VarList : List Shape → Type where
  | nil : VarList []
  | cons {s : Shape} {ss : List Shape} : Var s → VarList ss → VarList (s :: ss)

namespace VarList

/-- First variable in a nonempty `VarList`. -/
def head {s : Shape} {ss : List Shape} : VarList (s :: ss) → Var s
  | cons v _ => v

/-- Tail variables in a nonempty `VarList`. -/
def tail {s : Shape} {ss : List Shape} : VarList (s :: ss) → VarList ss
  | cons _ vs => vs

end VarList

/-- Buffer writes computed from one runtime forward's recorded values. -/
abbrev BufferUpdate (α : Type) [TorchLean.Storage α] :=
  (∀ {s : Shape}, Nat → IO (Tensor α s)) →
  (∀ {s : Shape}, Nat → IO (Tensor α s)) → IO (Array (Nat × Spec.SomeTensor α))

/-- Graph builder state, including runtime-only buffer observers outside the pure graph. -/
structure StateWith (α : Type) [TorchLean.Storage α] (Δ : Type) (Γ : List Shape) where
  /-- Shapes of the intermediate values. -/
  nodeShapes : List Shape
  /-- Pure forward and derivative semantics. -/
  data : GraphData α Δ Γ nodeShapes
  /-- Buffer observers to execute against the completed runtime tape. -/
  bufferUpdates : Array (BufferUpdate α) := #[]

/-- Default `GraphM` state with no extra environment (`Δ := Unit`). -/
abbrev State (α : Type) [TorchLean.Storage α] (Γ : List Shape) : Type :=
  StateWith α Unit Γ

/-- `StateT` builder monad for authoring a `GraphData` program, with explicit environment `Δ`. -/
abbrev MWith (α : Type) [TorchLean.Storage α] (Δ : Type) (Γ : List Shape) : Type → Type :=
  StateT (StateWith α Δ Γ) (Runtime.Autograd.Result)

/-- Default `GraphM` builder monad with `Δ := Unit`. -/
abbrev M (α : Type) [TorchLean.Storage α] (Γ : List Shape) : Type → Type :=
  MWith α Unit Γ

/-- Empty builder state (no intermediate nodes yet). -/
def empty {α : Type} [TorchLean.Storage α] {Γ : List Shape} : State α Γ :=
  ⟨[], .nil, #[]⟩

/-- Empty builder state for an explicit environment type `Δ`. -/
def emptyWith {α : Type} [TorchLean.Storage α]
    {Δ : Type} {Γ : List Shape} : StateWith α Δ Γ :=
  ⟨[], .nil, #[]⟩

/-- Run a `GraphM` program from an empty state. -/
def run {α : Type} [TorchLean.Storage α] {Γ : List Shape} {β : Type} (m : M α Γ β) :
    Runtime.Autograd.Result (β × State α Γ) :=
  StateT.run m empty

/-- Length of the current context `Γ ++ ss` (inputs + intermediates). -/
def ctxLen {Γ : List Shape} (ss : List Shape) : Nat :=
  (Γ ++ ss).length

/--
Convert a `Var s` into a dependent `Idx (Γ ++ ss) s`.

This performs bounds checking and a runtime shape check, returning a structured error if the
variable points outside the current context or has the wrong shape.
-/
def mkIdx {_α : Type} {Γ : List Shape} (ss : List Shape) {s : Shape}
    (v : Var s) : Runtime.Autograd.Result (Idx (Γ ++ ss) s) := by
  let n := v.id
  if h : n < ctxLen (Γ := Γ) ss then
    let i : Fin (ctxLen (Γ := Γ) ss) := ⟨n, h⟩
    let got : Shape := (Γ ++ ss).get i
    if hg : got = s then
      exact .ok ⟨i, hg⟩
    else
      exact .error <|
        s!"typed GraphM: shape mismatch at id={n}: expected {Shape.pretty s}, " ++
          s!"got {Shape.pretty got}"
  else
      exact .error s!"typed GraphM: invalid id={n} for ctxLen={ctxLen (Γ := Γ) ss}"

/--
Append a node to the graph state and return a fresh `Var` pointing to its output.

The returned variable id is `Γ.length + ss.length`, i.e. it points at the newly appended entry.
-/
def push {α : Type} [TorchLean.Storage α]
    {Δ : Type} {Γ : List Shape} {ss : List Shape} {s : Shape}
    (g : GraphData α Δ Γ ss) (node : NodeData α Δ (Γ ++ ss) s) : MWith α Δ Γ (Var s) := do
  modify fun state =>
    { nodeShapes := ss ++ [s]
      data := .snoc g node
      bufferUpdates := state.bufferUpdates }
  pure { id := Γ.length + ss.length }

/--
Reference an input variable from the initial context `Γ`.

This checks that the provided index is within bounds and that the requested shape matches the
shape at that position in `Γ`.

PyTorch comparison: this is like naming a graph input tensor in a traced graph.
-/
def arg {α : Type} [TorchLean.Storage α] {Δ : Type}
    {Γ : List Shape} (i : Nat) (s : Shape) :
    MWith α Δ Γ (Var s) := do
  if h : i < Γ.length then
    let fin : Fin Γ.length := ⟨i, h⟩
    let got : Shape := Γ.get fin
    if _hg : got = s then
      pure { id := i }
    else
      throw <|
        s!"typed GraphM: input shape mismatch at i={i}: expected " ++
          s!"{Shape.pretty s}, got {Shape.pretty got}"
  else
    throw s!"typed GraphM: input index out of bounds i={i} (Γ.length={Γ.length})"

namespace Internal

/-- Worker for `args`: hand out consecutive ids starting at `i`.

The ids match the positions the graph builder assigns to inputs, which is why this can produce a
`VarList Γ` without consulting the graph state at all. -/
def args : (Γ : List Shape) → Nat → VarList Γ
  | [], _i => .nil
  | _s :: ss, i => .cons { id := i } (args ss (i + 1))

end Internal

/--
Return one `Var` per entry of `Γ`, in order.

This is the canonical argument environment for a graph with input context `Γ`.
-/
def args {α : Type} [TorchLean.Storage α]
    {Δ : Type} {Γ : List Shape} : MWith α Δ Γ (VarList Γ) := do
  pure (Internal.args Γ 0)

/--
Embed a constant tensor as a node in the typed graph.

This node has no input dependencies (`vjp = 0`, `jvp = 0`), i.e. it is treated as a constant
with respect to the graph inputs.

PyTorch comparison: a constant literal captured into a traced/typed graph.
-/
def const {α : Type} [TorchLean.Storage α]
    {Δ : Type} [Zero α] {Γ : List Shape} {s : Shape} (t : Tensor α s) :
    MWith α Δ Γ (Var s) := do
  let ⟨ss, g, _⟩ ← get
  let node : NodeData α Δ (Γ ++ ss) s :=
    NodeData.ofLocalCompact (fun _ => ())
      (forward := fun _ctx _d => t)
      (jvp := fun _ctx _dctx _d => Tensor.full s (0 : α))
      (vjp := fun _ctx _d _δ => Contributions.zero (α := α) (shapes := Γ ++ ss))
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := s) g node

/-- Deterministic `U[0,1)` tensor generator (seeded, pure). -/
def randUniform {α : Type} [TorchLean.Storage α] [Context α] {Δ : Type} {Γ : List Shape} {s : Shape}
    (seed : Nat) : MWith α Δ Γ (Var s) := do
  let ⟨ss, g, _⟩ ← get
  let counter := ss.length
  let key := Spec.Random.keyOf seed counter
  let t : Tensor α s := Spec.Random.uniform (α := α) key (s := s)
  let node : NodeData α Δ (Γ ++ ss) s :=
    NodeData.ofLocalCompact (fun _ => ())
      (forward := fun _ctx _d => t)
      (jvp := fun _ctx _dctx _d => Tensor.full s (0 : α))
      (vjp := fun _ctx _d _δ => Contributions.zero (α := α) (shapes := Γ ++ ss))
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := s) g node

/--
Deterministic `{0,1}` mask generator (seeded, pure).

Note: for differentiation purposes, this node is treated as a **stop-gradient** op:
`jvp = 0` and `vjp = 0` for all inputs (including `keepProb`). This matches the intended use in
dropout where the probability is a hyperparameter (not differentiated), while keeping execution
deterministic in `.typedGraph` execution.
-/
def bernoulliMask {α : Type} [TorchLean.Storage α] [Context α]
    {Δ : Type} {Γ : List Shape} {s : Shape}
    (keepProb : Var Shape.scalar) (seed : Nat) :
    MWith α Δ Γ (Var s) := do
  let ⟨ss, g, _⟩ ← get
  let counter := ss.length
  let key := Spec.Random.keyOf seed counter
  let ikp ← liftM (mkIdx (_α := α) (Γ := Γ) ss keepProb)
  let node : NodeData α Δ (Γ ++ ss) s :=
    NodeData.ofLocalCompact (fun lookup => lookup.read ikp)
      (forward := fun ctx _d =>
        let kpT := ctx
        let kp : α := kpT.item
        Spec.Random.mask (α := α) key kp (s := s))
      (jvp := fun _ctx _dctx _d => Tensor.full s (0 : α))
      (vjp := fun _ctx _d _δ => Contributions.zero (α := α) (shapes := Γ ++ ss))
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := s) g node

/--
Keep the primal value and stop differentiation through this reference.

Both graph JVP and VJP return zero. The forward value also clears scalar tangents when the graph
runs over dual numbers, so a later reverse rule cannot recover a dependency from a detached value.
-/
def detach {α : Type} [TorchLean.Storage α] [Context α]
    {Δ : Type} {Γ : List Shape} {s : Shape}
    (x : Var s) : MWith α Δ Γ (Var s) := do
  let ⟨ss, g, _⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let node : NodeData α Δ (Γ ++ ss) s :=
    NodeData.ofLocalCompact (fun lookup => lookup.read ix)
      (forward := fun ctx _d => Tensor.detachSpec (ctx))
      (jvp := fun _ctx _dctx _d => Tensor.full s (0 : α))
      (vjp := fun _ctx _d _δ => Contributions.zero (α := α) (shapes := Γ ++ ss))
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := s) g node

end GraphM
end TypedGraph
end Autograd
end Runtime
