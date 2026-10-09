/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.TypedGraph.GraphM.Core
public import NN.Runtime.Autograd.TypedGraph.Compiled

/-!
# Typed Executable Graphs

Typed SSA graphs with forward, JVP, and VJP entry points. The graph records the computation once
and reuses the same shape-indexed operation data for repeated execution. Construction records
operations in call order. Execution uses the implementation attached to each node.
-/


@[expose] public section

namespace Runtime
namespace Autograd
namespace Torch

open Spec
open TorchLean TorchLean.Tensor
open Proofs.Autograd.Algebra
-- Typed context indices come from `NN.Proofs.Autograd.Tape.Util.Idx`, the one place
-- `Idx` and `getIdx` are defined.
open Proofs (Idx getIdx)

/-!
`TypedGraph` is a persistent value whose inputs are supplied at evaluation time. It stores
`GraphData`: shape-indexed forward, JVP, and VJP functions that callers can reuse with new inputs.
The local laws identifying these functions with derivatives live in
`Proofs.Autograd.Algebra.Node`; `Proofs.Autograd.Algebra.Graph` stores nodes together with those
laws.

Checked compilation evaluates forward operations and saves their local reverse programs.
Each new set of inputs needs a fresh compiled execution. The reusable object is the recorded
graph; saved programs belong to the forward evaluation that created them.

The pure entry points below assume each node's runtime preconditions. Checked IO autodiff and
session execution use `TypedGraph.compileChecked` or `TypedGraph.jvpChecked` to report
`NodeData.validate` failures before evaluating a node. The pure graph functions remain available
for the semantic and proof layers.

The imperative API records operations through `Runtime.Autograd.Model.Session`, which selects
the eager runtime or `Internal.TypedGraphSession`. That session owns mutable recording state.
Imported `NN.IR.Graph` programs use `Runtime.Autograd.IRExec.ForwardGraph`, whose nodes provide
forward execution.
-/

/--
Typed graph with differentiable tensor inputs `Γ`, non-differentiable runtime data `Δ`, and output
of shape `τ`.

`Δ` is reserved for non-differentiable data such as token ids, labels, gather indices, and masks.
It affects evaluation but does not appear in the returned input gradients.
-/
structure TypedGraphWithData (α : Type) (Δ : Type) [TorchLean.Storage α]
    (Γ : List Shape) (τ : Shape) where
  /-- Shapes of the recorded SSA nodes. -/
  nodeShapes : List Shape
  /-- Executable data for all recorded nodes. -/
  data : Proofs.Autograd.Algebra.GraphData α Δ Γ nodeShapes
  /-- Typed reference to the graph output, which may be an input or any recorded node. -/
  output : Proofs.Idx (Γ ++ nodeShapes) τ
  /-- Runtime buffer observers; pure evaluation and differentiation leave these unapplied. -/
  bufferUpdates : Array (Runtime.Autograd.TypedGraph.GraphM.BufferUpdate α) := #[]

namespace TypedGraphWithData

/-- Evaluate the output tensor for leaf values `x` and auxiliary input `d`. -/
def forward {α Δ : Type} [TorchLean.Storage α] {Γ : List Shape} {τ : Shape}
    (c : TypedGraphWithData α Δ Γ τ) (x : TorchLean.TensorPack α Γ) (d : Δ) : Tensor α τ :=
  getIdx (Proofs.Autograd.Algebra.GraphData.eval (g := c.data) x d) c.output

/-- Forward-mode Jacobian-vector product at `x`, with `d` held fixed. -/
def jvp {α Δ : Type} [TorchLean.Storage α] {Γ : List Shape} {τ : Shape}
    (c : TypedGraphWithData α Δ Γ τ) (x dx : TorchLean.TensorPack α Γ) (d : Δ) : Tensor α τ :=
  getIdx (Proofs.Autograd.Algebra.GraphData.jvpCtx (g := c.data) x dx d) c.output

/-- Reverse-mode vector-Jacobian product with an explicit output cotangent seed. -/
def vjp {α Δ : Type} [TorchLean.Storage α] [Add α] [Zero α]
    {Γ : List Shape} {τ : Shape} (c : TypedGraphWithData α Δ Γ τ)
    (x : TorchLean.TensorPack α Γ) (d : Δ) (seedOut : Tensor α τ) : TorchLean.TensorPack α Γ :=
  Proofs.Autograd.Algebra.GraphData.backpropCtx
    (α := α) (Δ := Δ) (Γ := Γ) (g := c.data) x d (TensorPack.single c.output seedOut)

/-- Recover the input pullback from a previously checked forward tape. -/
def vjpFromTape {α Δ : Type} [Storage α] [Add α] [Zero α]
    {Γ : List Shape} {τ : Shape} (graph : TypedGraphWithData α Δ Γ τ)
    (tape : Runtime.Autograd.Tape α) (seed : Tensor α τ) :
    Runtime.Autograd.Result (TorchLean.TensorPack α Γ) := do
  let gradients ← Runtime.Autograd.TypedGraph.backwardDenseAllFrom tape graph.output seed
  TorchLean.TensorPack.ofShapeErasedArray gradients (shapes := Γ)

/--
Evaluate the output and its seeded input pullback, checking every node's runtime domain first.

The returned pair is `(inputGradients, output)`. Auxiliary data is held fixed. The compiled
implementation uses one checked execution for both values, with a proved replacement of the
raw Tape specification below.
-/
def vjpChecked {α Δ : Type} [Storage α] [Add α] [Zero α]
    {Γ : List Shape} {τ : Shape} (graph : TypedGraphWithData α Δ Γ τ)
    (inputs : TorchLean.TensorPack α Γ) (data : Δ) (seed : Tensor α τ) :
    Runtime.Autograd.Result (TorchLean.TensorPack α Γ × Tensor α τ) := do
  let (tape, context) ← Runtime.Autograd.TypedGraph.lowerToTapeChecked graph.data inputs data
  let gradients ← graph.vjpFromTape tape seed
  pure (gradients, getIdx context graph.output)

/-- Execute the checked VJP through indexed primals and saved reverse programs. -/
def vjpCompiledChecked {α Δ : Type} [Storage α] [Add α] [Zero α]
    {Γ : List Shape} {τ : Shape} (graph : TypedGraphWithData α Δ Γ τ)
    (inputs : TorchLean.TensorPack α Γ) (data : Δ) (seed : Tensor α τ) :
    Runtime.Autograd.Result (TorchLean.TensorPack α Γ × Tensor α τ) := do
  let compiled ← Runtime.Autograd.TypedGraph.compileChecked graph.data inputs data
  let gradients ← TorchLean.TensorPack.ofShapeErasedArray
    (compiled.backwardDenseAllFrom graph.output seed) (shapes := Γ)
  pure (gradients, compiled.context.lookup.read graph.output)

/-- The maintained checked API compiles to the exact saved-program implementation. -/
@[csimp] theorem vjpChecked_eq_compiled : @vjpChecked = @vjpCompiledChecked := by
  funext α Δ storage add zero Γ τ graph inputs data seed
  have lowered := Runtime.Autograd.TypedGraph.compileChecked_eq_lowerToTapeChecked
    graph.data inputs data
  cases checked : Runtime.Autograd.TypedGraph.compileChecked graph.data inputs data with
  | error message =>
      simp only [checked, Except.map] at lowered
      simp only [vjpChecked, vjpCompiledChecked, ← lowered, checked,
        Bind.bind, Except.bind]
  | ok compiled =>
      simp only [checked, Except.map] at lowered
      have same := Runtime.Autograd.TypedGraph.lowerToTapeChecked_eq
        graph.data inputs data (compiled.toTape, compiled.context.toPack) lowered.symm
      have tape := congrArg Prod.fst same
      have backward := Runtime.Autograd.TypedGraph.compileChecked_backwardDenseFrom_eq_tape
        graph.data inputs data compiled checked (TensorPack.single graph.output seed)
      have backward' :
          Runtime.Autograd.TypedGraph.backwardDenseAllFrom
            compiled.toTape graph.output seed =
          .ok (compiled.backwardDenseAllFrom graph.output seed) := by
        simp only [Runtime.Autograd.TypedGraph.backwardDenseAllFrom,
          Runtime.Autograd.TypedGraph.Compiled.backwardDenseAllFrom]
        rw [show compiled.toTape = _ from tape]
        exact backward.symm
      simp only [vjpChecked, vjpCompiledChecked, ← lowered, checked,
        Bind.bind, Except.bind,
        vjpFromTape, backward']
      have output := congrArg
        (fun (reader : TensorLookup α (Γ ++ graph.nodeShapes)) => reader.read graph.output)
        (TensorContext.lookup_toPack compiled.context)
      rw [show getIdx compiled.context.toPack graph.output =
        compiled.context.lookup.read graph.output from output]

end TypedGraphWithData

/-- Typed graph with no auxiliary, non-differentiable runtime inputs. -/
abbrev TypedGraph (α : Type) [TorchLean.Storage α] (Γ : List Shape) (τ : Shape) : Type :=
  TypedGraphWithData α Unit Γ τ

namespace TypedGraph

/-- Evaluate the output tensor for leaf values `x`. -/
def forward {α : Type} [TorchLean.Storage α] {Γ : List Shape} {τ : Shape}
  (c : TypedGraph α Γ τ) (x : TorchLean.TensorPack α Γ) : Tensor α τ :=
  TypedGraphWithData.forward c x ()

/-- Forward-mode Jacobian-vector product (JVP) at `x` with tangent `dx`. -/
def jvp {α : Type} [TorchLean.Storage α] {Γ : List Shape} {τ : Shape}
  (c : TypedGraph α Γ τ) (x dx : TorchLean.TensorPack α Γ) : Tensor α τ :=
  TypedGraphWithData.jvp c x dx ()

/--
Reverse-mode vector-Jacobian product (VJP) with an explicit output cotangent seed.

Scalar outputs use the same operation with `Tensor.scalar seed`; a seed of `1` computes the
ordinary scalar-loss gradient. No `One` instance is required when the seed is supplied explicitly.
-/
def vjp {α : Type} [TorchLean.Storage α] [Add α] [Zero α]
    {Γ : List Shape} {τ : Shape}
    (c : TypedGraph α Γ τ) (x : TorchLean.TensorPack α Γ) (seedOut : Tensor α τ) :
    TorchLean.TensorPack α Γ :=
  TypedGraphWithData.vjp c x () seedOut

end TypedGraph

/-- Lower a graph builder with non-differentiable runtime data into a reusable typed graph. -/
def lowerToTypedGraphWithData {α Δ : Type} [TorchLean.Storage α]
    {Γ : List Shape} {τ : Shape}
    (build : Runtime.Autograd.TypedGraph.GraphM.MWith α Δ Γ
      (Runtime.Autograd.TypedGraph.GraphM.Var τ)) :
    Runtime.Autograd.Result (TypedGraphWithData α Δ Γ τ) := do
  let (outVar, st) ← Runtime.Autograd.TypedGraph.GraphM.run build
  let output ← Runtime.Autograd.TypedGraph.GraphM.mkIdx
    (Γ := Γ) st.nodeShapes outVar
  pure
    { nodeShapes := st.nodeShapes
      data := st.data
      output := output
      bufferUpdates := st.bufferUpdates }

/--
Lower a tensor-output graph builder into a `TypedGraph`.

The returned variable may reference an input or any recorded node. Lowering validates its runtime
index and preserves its statically known output shape.
-/
def lowerToTypedGraph {α : Type} [TorchLean.Storage α]
    {Γ : List Shape} {τ : Shape}
  (build : Runtime.Autograd.TypedGraph.GraphM.M α Γ (Runtime.Autograd.TypedGraph.GraphM.Var τ)) :
  Runtime.Autograd.Result (TypedGraph α Γ τ) :=
  lowerToTypedGraphWithData (α := α) (Δ := Unit) (Γ := Γ) (τ := τ) build
end Torch
end Autograd
end Runtime
