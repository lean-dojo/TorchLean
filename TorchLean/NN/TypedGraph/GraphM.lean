/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.TypedGraph.Data
public import TorchLean.NN.Tape.CPU
public import TorchLean.Tensor.Random

/-!
# GraphM Core

Typed variables, builder state, input binding, constants and the random key stream of the
executable typed graph authoring API.
-/

@[expose] public section

namespace TorchLean
namespace TypedGraph
namespace GraphM

open TorchLean.Tensor
open Tape
-- Typed context indices come from `TorchLean.NN.TypedGraph.Idx`, the one place
-- `Idx` and `getIdx` are defined.
open TorchLean (Idx)
open TensorPack (getIdx)

/--
A typed handle to a value in the growing graph context.

`Var s` carries its expected `Shape` at the type level, while `id` is the runtime index into the
concatenated context `Γ ++ ss`.
-/
structure Var (s : Shape) where
  /--
  Position of the value in the concatenated context `Γ ++ ss`. `mkIdx` checks its bounds and shape
  when it turns the variable into an `Idx` for a `GraphData` node.
  -/
  id : Nat
deriving Repr

/-!
`inputVars` returns a `Pack Var Γ` with one `Var` per entry of `Γ`, in order, so code that reads
every input need not repeat the index and shape of each `GraphM.arg`.
-/

/-- Buffer writes computed from one runtime forward's recorded values. -/
abbrev BufferUpdate (α : Type) [Storage α] :=
  (∀ {s : Shape}, Nat → IO (Tensor α s)) →
  (∀ {s : Shape}, Nat → IO (Tensor α s)) → IO (Array (Nat × SomeTensor α))

/-- Graph builder state, including runtime-only buffer observers outside the pure graph. -/
structure StateWith (α : Type) [Storage α] (Δ : Type) (Γ : List Shape) where
  /-- Shapes of the intermediate values. -/
  nodeShapes : List Shape
  /-- The graph recorded so far, with its forward and derivative programs. -/
  data : GraphData α Δ Γ nodeShapes
  /-- Buffer observers to execute against the completed runtime tape. -/
  bufferUpdates : Array (BufferUpdate α) := #[]
  /-- Position in the random key stream, which only random operators advance (one step each). -/
  randomCounter : Nat := 0

/-- Default `GraphM` state with no extra environment (`Δ := Unit`). -/
abbrev State (α : Type) [Storage α] (Γ : List Shape) : Type :=
  StateWith α Unit Γ

/-- `StateT` builder monad for authoring a `GraphData` program over the environment `Δ`. -/
abbrev MWith (α : Type) [Storage α] (Δ : Type) (Γ : List Shape) : Type → Type :=
  StateT (StateWith α Δ Γ) (Result)

/-- Default `GraphM` builder monad with `Δ := Unit`. -/
abbrev M (α : Type) [Storage α] (Γ : List Shape) : Type → Type :=
  MWith α Unit Γ

/-- Empty builder state, with the environment `Δ` inferred from the expected type. -/
def empty {α : Type} [Storage α]
    {Δ : Type} {Γ : List Shape} : StateWith α Δ Γ :=
  ⟨[], .nil, #[], 0⟩

/-- Run a `GraphM` program from an empty state. -/
def run {α : Type} [Storage α] {Δ : Type} {Γ : List Shape} {β : Type}
    (m : MWith α Δ Γ β) : Result (β × StateWith α Δ Γ) :=
  StateT.run m empty

/-- Length of the context `Γ ++ ss` (inputs and intermediates). -/
def ctxLen {Γ : List Shape} (ss : List Shape) : Nat :=
  (Γ ++ ss).length

/-- Count inputs and intermediates without allocating their concatenation. -/
def ctxLenFast {Γ : List Shape} (ss : List Shape) : Nat :=
  Γ.length + ss.length

/-- `@[csimp]` lemma: compiled code computes `ctxLen` as `Γ.length + ss.length` and builds no
concatenated list. -/
@[csimp] theorem ctxLen_eq_fast : @ctxLen = @ctxLenFast := by
  funext Γ ss
  simp only [ctxLen, ctxLenFast, List.length_append]

/--
Convert a `Var s` into a dependent `Idx (Γ ++ ss) s`, failing with an error message when the id
lies outside the context `Γ ++ ss` or the shape at that position is not `s`.
-/
def mkIdx {Γ : List Shape} (ss : List Shape) {s : Shape}
    (v : Var s) : Result (Idx (Γ ++ ss) s) := by
  let n := v.id
  if h : n < ctxLen (Γ := Γ) ss then
    let i : Fin (ctxLen (Γ := Γ) ss) := ⟨n, h⟩
    let got : Shape := (Γ ++ ss).get i
    if hg : got = s then
      exact .ok ⟨i, hg⟩
    else
      exact .error <|
        s!"typed GraphM: shape mismatch at id={n}: expected {s}, " ++
          s!"got {got}"
  else
      exact .error s!"typed GraphM: invalid id={n} for ctxLen={ctxLen (Γ := Γ) ss}"

/--
Append `node` to the graph state and return the `Var` of its output, whose id
`Γ.length + ss.length` is the position of the appended entry.
-/
def push {α : Type} [Storage α]
    {Δ : Type} {Γ : List Shape} {ss : List Shape} {s : Shape}
    (g : GraphData α Δ Γ ss) (node : NodeData α Δ (Γ ++ ss) s) : MWith α Δ Γ (Var s) := do
  modify fun state =>
    { nodeShapes := ss ++ [s]
      data := .snoc g node
      bufferUpdates := state.bufferUpdates
      randomCounter := state.randomCounter }
  pure { id := Γ.length + ss.length }

/--
The `Var` of input `i` of the context `Γ`, after checking that `i < Γ.length` and that the shape at
position `i` is `s`.

PyTorch analogue: naming an input tensor of a traced graph.
-/
def arg {α : Type} [Storage α] {Δ : Type}
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
          s!"{s}, got {got}"
  else
    throw s!"typed GraphM: input index out of bounds i={i} (Γ.length={Γ.length})"

/-- Worker for `inputVars`: consecutive ids starting at `i`.

The ids match the positions the graph builder assigns to inputs, so this can produce a
`Pack Var Γ` without consulting the graph state. -/
def inputVarsFrom : (Γ : List Shape) → Nat → Pack Var Γ
  | [], _i => .nil
  | _s :: ss, i => .cons { id := i } (inputVarsFrom ss (i + 1))

/--
The argument environment of a graph with input context `Γ`: one `Var` per entry, in order, with
ids counting up from `0`.
-/
def inputVars {α : Type} [Storage α]
    {Δ : Type} {Γ : List Shape} : MWith α Δ Γ (Pack Var Γ) := do
  pure (inputVarsFrom Γ 0)

/--
Record the constant tensor `t` as a node whose JVP and VJP are zero, so it depends on no graph
input.

PyTorch analogue: a constant literal captured in a traced graph.
-/
def const {α : Type} [Storage α]
    {Δ : Type} [Zero α] {Γ : List Shape} {s : Shape} (t : Tensor α s) :
    MWith α Δ Γ (Var s) := do
  let ⟨ss, g, _, _⟩ ← get
  let node : NodeData α Δ (Γ ++ ss) s :=
    NodeData.ofLocalCompact (fun _ => ())
      (forward := fun _ctx _d => t)
      (jvp := fun _ctx _dctx _d => Tensor.full s (0 : α))
      (vjp := fun _ctx _d _δ => Contributions.zero (α := α) (shapes := Γ ++ ss))
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := s) g node

/-- Draw `Random.keyOf seed` at the builder's random counter and advance the counter by one. -/
def nextRandomKey {α : Type} [Storage α] {Δ : Type} {Γ : List Shape}
    (seed : Nat) : MWith α Δ Γ UInt64 := do
  let counter := (← get).randomCounter
  modify fun state => { state with randomCounter := counter + 1 }
  pure (Random.keyOf seed counter)

end GraphM
end TypedGraph
end TorchLean
