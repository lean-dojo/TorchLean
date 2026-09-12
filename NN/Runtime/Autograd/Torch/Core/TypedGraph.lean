/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.TypedGraph.GraphM.Core

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

open Spec TorchLean
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

Tape lowering evaluates the forward operations and captures their values in backward closures.
Each new set of inputs therefore needs a fresh lowered tape. The reusable object is the recorded
graph, while a lowered tape belongs to the particular forward evaluation that created it.

The pure entry points below assume each node's runtime preconditions. Checked IO autodiff and
session execution use `TypedGraph.lowerToTapeChecked` or `TypedGraph.jvpChecked` to report
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
def vjpWithSeed {α Δ : Type} [TorchLean.Storage α] [Add α] [Zero α]
    {Γ : List Shape} {τ : Shape} (c : TypedGraphWithData α Δ Γ τ)
    (x : TorchLean.TensorPack α Γ) (d : Δ) (seedOut : Tensor α τ) : TorchLean.TensorPack α Γ :=
  Proofs.Autograd.Algebra.GraphData.backpropCtx
    (α := α) (Δ := Δ) (Γ := Γ) (g := c.data) x d (TensorPack.single c.output seedOut)

end TypedGraphWithData

/-- Typed graph with no auxiliary, non-differentiable runtime inputs. -/
abbrev TypedGraph (α : Type) [TorchLean.Storage α] (Γ : List Shape) (τ : Shape) : Type :=
  TypedGraphWithData α Unit Γ τ

/-- Scalar-output typed graph with no auxiliary, non-differentiable runtime inputs. -/
abbrev TypedScalarGraph (α : Type) [TorchLean.Storage α] (Γ : List Shape) : Type :=
  TypedGraph α Γ Shape.scalar

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

This is the tensor-valued analogue of `TypedScalarGraph.backwardWithSeed`.
PyTorch comparison: `out.backward(gradient=seedOut)` (for a tensor output).
-/
def vjpWithSeed {α : Type} [TorchLean.Storage α] [Add α] [Zero α]
    {Γ : List Shape} {τ : Shape}
    (c : TypedGraph α Γ τ) (x : TorchLean.TensorPack α Γ) (seedOut : Tensor α τ) :
    TorchLean.TensorPack α Γ :=
  TypedGraphWithData.vjpWithSeed c x () seedOut

end TypedGraph

namespace TypedScalarGraph

/-- Evaluate the scalar output for leaf values `x`. -/
def forward {α : Type} [TorchLean.Storage α] {Γ : List Shape}
    (c : TypedScalarGraph α Γ) (x : TorchLean.TensorPack α Γ) : Tensor α .scalar :=
  TypedGraph.forward c x

/-- Forward-mode Jacobian-vector product at `x` with tangent `dx`. -/
def jvp {α : Type} [TorchLean.Storage α] {Γ : List Shape}
    (c : TypedScalarGraph α Γ) (x dx : TorchLean.TensorPack α Γ) : Tensor α .scalar :=
  TypedGraph.jvp c x dx

/-- Reverse-mode backpropagation for a scalar output with implicit cotangent seed `1`. -/
def backward {α : Type} [TorchLean.Storage α] [Add α] [Zero α] [One α]
    {Γ : List Shape} (c : TypedScalarGraph α Γ) (x : TorchLean.TensorPack α Γ) :
    TorchLean.TensorPack α Γ :=
  TypedGraph.vjpWithSeed c x (Tensor.scalar (1 : α))

/-- Reverse-mode backpropagation for a scalar output with an explicit scalar seed. -/
def backwardWithSeed {α : Type} [TorchLean.Storage α] [Add α] [Zero α]
    {Γ : List Shape} (c : TypedScalarGraph α Γ) (x : TorchLean.TensorPack α Γ) (seedOut : α) :
    TorchLean.TensorPack α Γ :=
  TypedGraph.vjpWithSeed c x (Tensor.scalar seedOut)

end TypedScalarGraph

/-- Lower a graph builder with non-differentiable runtime data into a reusable typed graph. -/
def lowerToTypedGraphWithData {α Δ : Type} [TorchLean.Storage α]
    {Γ : List Shape} {τ : Shape}
    (build : Runtime.Autograd.TypedGraph.GraphM.MWith α Δ Γ
      (Runtime.Autograd.TypedGraph.GraphM.Var τ)) :
    Runtime.Autograd.Result (TypedGraphWithData α Δ Γ τ) := do
  let (outVar, st) ← StateT.run build Runtime.Autograd.TypedGraph.GraphM.emptyWith
  let output ← Runtime.Autograd.TypedGraph.GraphM.mkIdx
    (_α := α) (Γ := Γ) st.1 outVar
  pure { nodeShapes := st.1, data := st.2, output := output }

/--
Lower a scalar-output graph builder into a `TypedScalarGraph`.

The builder is expressed in the `TypedGraph.GraphM` monad. Its returned scalar variable may be an
input or any recorded node; lowering preserves that reference as the graph output.
-/
def lowerScalarToTypedGraph {α : Type} [TorchLean.Storage α]
    {Γ : List Shape}
    (build : Runtime.Autograd.TypedGraph.GraphM.M α Γ
      (Runtime.Autograd.TypedGraph.GraphM.Var Shape.scalar)) :
    Runtime.Autograd.Result (TypedScalarGraph α Γ) :=
  lowerToTypedGraphWithData (α := α) (Δ := Unit) (Γ := Γ) (τ := Shape.scalar) build

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
