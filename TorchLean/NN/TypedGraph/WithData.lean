/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.TypedGraph.GraphM
public import TorchLean.NN.TypedGraph.Compiled

/-!
# Typed Executable Graphs

Typed SSA graphs with forward, JVP, and VJP entry points. Construction records the operations once,
in call order, and every later execution reuses the same shape-indexed operation data, running the
implementation attached to each node.
-/


@[expose] public section

namespace TorchLean

open TorchLean.Tensor
open Tape
-- Typed context indices come from `TorchLean.NN.TypedGraph.Idx`, the one place
-- `Idx` and `getIdx` are defined.
open TorchLean (Idx)
open TensorPack (getIdx)

/-!
`TypedGraph` is a persistent value whose inputs are supplied at evaluation time. It stores
`GraphData`: shape-indexed forward, JVP, and VJP functions that callers can reuse with new inputs.
`TorchLean.Tape.NodeData.Correct` is the local adjoint law that pairs a node's JVP with
its VJP, and `Tape.Graph` stores graph data with a proof of that law at every node.

Checked compilation evaluates the forward operations and saves their local reverse programs. Those
programs belong to the forward evaluation that created them, so each new set of inputs needs a fresh
compiled execution and the recorded graph is the object that gets reused.

The pure entry points below assume each node's runtime preconditions. The IO autodiff calls use
`TypedGraph.compileChecked`, `TypedGraph.jvpChecked` or, in `autograd.model.derivative`,
`TypedGraph.lowerToTapeChecked`, which report `NodeData.validate` failures before evaluating a
node. The semantic and proof layers use the pure graph functions.
-/

/--
Typed graph with differentiable tensor inputs `Γ`, non-differentiable runtime data `Δ`, and output
of shape `τ`.

`Δ` is reserved for non-differentiable data such as token ids, labels, gather indices, and masks.
It affects evaluation but does not appear in the returned input gradients.
-/
structure TypedGraph.WithData (α : Type) (Δ : Type) [Storage α]
    (Γ : List Shape) (τ : Shape) where
  /-- Shapes of the recorded SSA nodes. -/
  nodeShapes : List Shape
  /-- Executable data for all recorded nodes. -/
  data : Tape.GraphData α Δ Γ nodeShapes
  /-- Typed reference to the graph output, which may be an input or any recorded node. -/
  output : Idx (Γ ++ nodeShapes) τ
  /-- Runtime buffer observers, which pure evaluation and differentiation leave unapplied. -/
  bufferUpdates : Array (TypedGraph.GraphM.BufferUpdate α) := #[]

namespace TypedGraph.WithData

/-- Evaluate the output tensor for leaf values `x` and auxiliary input `d`. -/
def forward {α Δ : Type} [Storage α] {Γ : List Shape} {τ : Shape}
    (c : WithData α Δ Γ τ) (x : TensorPack α Γ) (d : Δ) : Tensor α τ :=
  getIdx (GraphData.eval (g := c.data) x d) c.output

/-- Forward-mode Jacobian-vector product at `x`, with `d` held fixed. -/
def jvp {α Δ : Type} [Storage α] {Γ : List Shape} {τ : Shape}
    (c : WithData α Δ Γ τ) (x dx : TensorPack α Γ) (d : Δ) : Tensor α τ :=
  getIdx (GraphData.jvpCtx (g := c.data) x dx d) c.output

/-- Pull `seedOut` back from the selected output to the inputs. With storage equality and certified
compact support, stored VJPs unrelated to the output stay unevaluated. Opaque VJPs conservatively
visit the full prefix, and carriers without storage equality use the dense reference. -/
def vjp {α Δ : Type} [Storage α] [Add α] [Zero α]
    {Γ : List Shape} {τ : Shape} (c : WithData α Δ Γ τ)
    (x : TensorPack α Γ) (d : Δ) (seedOut : Tensor α τ) : TensorPack α Γ :=
  let saved := (SavedGraph.ofGraph c.data x d).1
  ((saved.backpropFrom c.output seedOut).toPack.split (ss₁ := Γ) (ss₂ := c.nodeShapes)).1

/--
Evaluate the output and its seeded input pullback, checking every node's runtime domain first.

The result is the pair `(inputGradients, output)`, with the auxiliary data held fixed. One checked
forward execution supplies both the output and the saved reverse programs. With storage equality
and certified compact support, the backward pass visits only the selected output's dependencies and
so avoids unrelated singular VJPs. Opaque VJPs and carriers without storage equality keep the
conservative traversal of `vjp`.
-/
def vjpChecked {α Δ : Type} [Storage α] [Add α] [Zero α]
    {Γ : List Shape} {τ : Shape} (graph : WithData α Δ Γ τ)
    (inputs : TensorPack α Γ) (data : Δ) (seed : Tensor α τ) :
    Result (TensorPack α Γ × Tensor α τ) := do
  let compiled ← TypedGraph.compileChecked graph.data inputs data
  let gradients := ((compiled.saved.backpropFrom graph.output seed).toPack.split
    (ss₁ := Γ) (ss₂ := graph.nodeShapes)).1
  pure (gradients, compiled.context.lookup.read graph.output)

end TypedGraph.WithData

/-- Typed graph with no auxiliary, non-differentiable runtime inputs. The `TypedGraph.WithData`
operations evaluate it with the data `()`. -/
abbrev TypedGraph (α : Type) [Storage α] (Γ : List Shape) (τ : Shape) : Type :=
  TypedGraph.WithData α Unit Γ τ

/-- Lower a graph builder with non-differentiable runtime data (`Unit` for a `GraphM.M` builder)
into a reusable typed graph. The output may be an input or any recorded node, and lowering checks
its index and shape. -/
def TypedGraph.WithData.lower {α Δ : Type} [Storage α]
    {Γ : List Shape} {τ : Shape}
    (build : TypedGraph.GraphM.MWith α Δ Γ
      (TypedGraph.GraphM.Var τ)) :
    Result (WithData α Δ Γ τ) := do
  let (outVar, st) ← TypedGraph.GraphM.run build
  let output ← TypedGraph.GraphM.mkIdx
    (Γ := Γ) st.nodeShapes outVar
  pure
    { nodeShapes := st.nodeShapes
      data := st.data
      output := output
      bufferUpdates := st.bufferUpdates }
end TorchLean
