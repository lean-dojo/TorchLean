/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Program
public import TorchLean.NN.Dual
public import TorchLean.NN.ScalarTrainer.Construction
public import TorchLean.NN.ScalarTrainer.GraphOps
public import TorchLean.NN.TypedGraph.WithData

/-!
# Autodiff

Autodiff utilities beyond basic `.backward()`:

- `jacrev`: the reverse-mode Jacobian of a program's output with respect to the parameters or
  inputs a caller picks.
- `hvp`: the Hessian-vector product of a scalar loss in its parameters and inputs together, using
  forward-over-reverse via `Dual` scalars.

They are executable runtime conveniences, separate from the `fderiv` proof developments.
-/

@[expose] public section


open TorchLean.Shape (size_append)

namespace TorchLean
namespace Autodiff

open TorchLean.Tensor

/-- Return the seeded input gradients and forward result, reporting domain errors through IO. -/
def pullback {α : Type} [Storage α] [Add α] [Zero α]
    {shapes : List Shape} {output : Shape}
    (graph : TypedGraph α shapes output)
    (inputs : TensorPack α shapes) (seed : Tensor α output) :
    IO (TensorPack α shapes × Tensor α output) :=
  IO.ofExcept <| graph.vjpChecked inputs () seed

open Autodiff (pullback)

/--
Lower a TorchLean program to a reusable `TypedGraph`.

The graph retains a typed reference to the value returned by the program, which may be an input or
any recorded node. Scalar outputs use the same path with shape `[]`.
-/
def lowerToTypedGraph {α : Type} [Storage α] [Context α]
    {paramShapes inputShapes : List Shape} {τ : Shape}
    (f :
      ∀ {β : Type}, [Storage β] → [Context β] →
        Program β (paramShapes ++ inputShapes) τ) :
    IO (TypedGraph α (paramShapes ++ inputShapes) τ) := do
  let Γ : List Shape := paramShapes ++ inputShapes
  let build :
      TypedGraph.GraphM.M α Γ (TypedGraph.GraphM.Var τ) := do
    let vs ← TypedGraph.GraphM.inputVars (α := α) (Γ := Γ)
    CurriedRef.uncurry (ss := Γ)
      (β := TypedGraph.GraphM.M α Γ (TypedGraph.GraphM.Var τ))
      (f (β := α) (m := TypedGraph.GraphM.M α Γ)) vs
  IO.ofExcept (TypedGraph.WithData.lower (α := α) (Δ := Unit) (Γ := Γ)
    (τ := τ) build)

/-- Reverse-mode Jacobian of the output with respect to the tensors that `pick` keeps from the
parameter and input gradient packs, with the output axes prepended to each kept tensor. -/
def jacrev {α : Type} [Storage α] [Context α]
    {paramShapes inputShapes shapes : List Shape} {τ : Shape}
    (f :
      ∀ {β : Type}, [Storage β] → [Context β] →
        Program β (paramShapes ++ inputShapes) τ)
    (params : TensorPack α paramShapes)
    (xs : TensorPack α inputShapes)
    (pick : TensorPack α paramShapes × TensorPack α inputShapes →
      TensorPack α shapes) :
    IO (TensorPack α (shapes.map (τ ++ ·))) := do
  let c ← lowerToTypedGraph (α := α) (paramShapes := paramShapes) (inputShapes := inputShapes)
    (τ := τ) f
  let Γ : List Shape := paramShapes ++ inputShapes
  let args : TensorPack α Γ :=
    Pack.append (ss₁ := paramShapes) (ss₂ := inputShapes) params xs
  let compiled ← IO.ofExcept <|
    TypedGraph.compileChecked c.data args ()
  TensorPack.stackLeadingM τ fun index => do
    let seedOut := (Tensor.oneHot (α := α) τ.size index).reshape τ (by simp)
    let allGradients := ((compiled.saved.backpropFrom c.output seedOut).toPack.split
      (ss₁ := Γ) (ss₂ := c.nodeShapes)).1
    pure (pick (Pack.split
      (ss₁ := paramShapes) (ss₂ := inputShapes) allGradients))

/--
Compute the forward Jacobian for a single tensor input.

Each input basis vector runs through the checked graph JVP, which applies the primitive rules of
directional differentiation, including stopped gradients and the runtime's choices at nonsmooth
points. The returned tensor has output axes followed by input axes.
-/
def jacfwdInput {α : Type} [Storage α] [Context α]
    {σ τ : Shape}
    (f :
      ∀ {β : Type}, [Storage β] → [Context β] →
        Program β [σ] τ)
    (x : Tensor α σ) :
    IO (Tensor α (τ ++ σ)) := do
  let c ← lowerToTypedGraph (α := α)
    (paramShapes := ([] : List Shape)) (inputShapes := [σ]) (τ := τ)
    (fun {β} _ _ _ => f (β := β))
  if σ.size = 0 then
    let _ ← IO.ofExcept <|
      TypedGraph.jvpChecked c.data (.cons x .nil)
        (.cons (Tensor.zeros σ) .nil) ()
  let columns ← Tensor.stackLeadingM fun (index : Fin σ.size) => do
    let dx := (Tensor.oneHot (α := α) σ.size index).reshape σ (by simp)
    let (_, tangentContext) ← IO.ofExcept <|
      TypedGraph.jvpChecked c.data (.cons x .nil) (.cons dx .nil) ()
    pure (TensorPack.getIdx tangentContext c.output)
  let matrix := columns.reshape [σ.size, τ.size] (by simp)
  pure <| (Tensor.permute matrix
      (Shape.adjacentSwap [σ.size, τ.size] 0) [τ.size, σ.size]).reshape (τ ++ σ)
    (by simp [Shape.size_append])

/--
Compute a tensor-output VJP with respect to parameters and inputs in one reverse pass.
-/
def vjp {α : Type} [Storage α] [Context α]
    {paramShapes inputShapes : List Shape} {τ : Shape}
    (f :
      ∀ {β : Type}, [Storage β] → [Context β] →
        Program β (paramShapes ++ inputShapes) τ)
    (params : TensorPack α paramShapes)
    (xs : TensorPack α inputShapes)
    (seedOut : Tensor α τ) :
    IO (TensorPack α paramShapes ×
      TensorPack α inputShapes) := do
  let c ← lowerToTypedGraph (α := α) (paramShapes := paramShapes) (inputShapes := inputShapes)
    (τ := τ) f
  let Γ : List Shape := paramShapes ++ inputShapes
  let args : TensorPack α Γ :=
    Pack.append (ss₁ := paramShapes) (ss₂ := inputShapes) params xs
  let (gAll, _) ← pullback c args seedOut
  pure (Pack.split
    (ss₁ := paramShapes) (ss₂ := inputShapes) gAll)

/-- Differentiate a scalar loss with respect to parameters and inputs in one reverse pass, as
`vjp` with a unit scalar seed. -/
def gradients {α : Type} [Storage α] [Context α]
    {paramShapes inputShapes : List Shape}
    (loss :
      ∀ {β : Type}, [Storage β] → [Context β] →
        Program β (paramShapes ++ inputShapes) Shape.scalar)
    (params : TensorPack α paramShapes)
    (xs : TensorPack α inputShapes) :
    IO (TensorPack α paramShapes ×
      TensorPack α inputShapes) :=
  vjp loss params xs (Tensor.scalar (1 : α))

/-- Directional derivative of scalar loss along `vparams` (forward-mode JVP). -/
def jvpLossParams {α : Type} [Storage α] [Context α]
    {paramShapes inputShapes : List Shape}
    (loss :
      ∀ {β : Type}, [Storage β] → [Context β] →
        Program β (paramShapes ++ inputShapes) Shape.scalar)
    (params : TensorPack α paramShapes)
    (xs : TensorPack α inputShapes)
    (vparams : TensorPack α paramShapes) :
    IO (Tensor α []) := do
  let c ← lowerToTypedGraph (α := α)
    (paramShapes := paramShapes) (inputShapes := inputShapes) loss
  let Γ : List Shape := paramShapes ++ inputShapes
  let args : TensorPack α Γ :=
    Pack.append (ss₁ := paramShapes) (ss₂ := inputShapes) params xs
  let zerosX : TensorPack α inputShapes :=
    TensorPack.zero (α := α) (ss := inputShapes)
  let dargs : TensorPack α Γ :=
    Pack.append (ss₁ := paramShapes) (ss₂ := inputShapes) vparams zerosX
  let (_, tangentContext) ← IO.ofExcept <|
    TypedGraph.jvpChecked c.data args dargs ()
  pure (TensorPack.getIdx tangentContext c.output)

/--
Hessian-vector product (HVP) of a scalar loss in its parameters and inputs together.

The result is `d/dε (∇ loss(params + ε*vparams, xs + ε*vxs)) |_{ε=0}`, split into its parameter
and input halves, aligned with `paramShapes` and `inputShapes`.

Implementation: reverse-mode AD over dual scalars (`Dual`) whose tangents are `vparams` and `vxs`.
The tangent part of the resulting gradients is the HVP.
-/
def hvp {α : Type} [Storage α] [Context α]
    {paramShapes inputShapes : List Shape}
    (loss :
      ∀ {β : Type}, [Storage β] → [Context β] →
        Program β (paramShapes ++ inputShapes) Shape.scalar)
    (params : TensorPack α paramShapes)
    (xs : TensorPack α inputShapes)
    (vparams : TensorPack α paramShapes)
    (vxs : TensorPack α inputShapes) :
    IO (TensorPack α paramShapes × TensorPack α inputShapes) := do
  let αD := Dual α
  let paramsD : TensorPack αD paramShapes :=
    Pack.zipWith DualTensor.withTangents params vparams
  let xsD : TensorPack αD inputShapes :=
    Pack.zipWith DualTensor.withTangents xs vxs
  let argsD : TensorPack αD (paramShapes ++ inputShapes) :=
    Pack.append (ss₁ := paramShapes) (ss₂ := inputShapes) paramsD xsD
  let graph ← lowerToTypedGraph (α := Dual α)
    (paramShapes := paramShapes) (inputShapes := inputShapes) loss
  let (gradsD, _) ← pullback graph argsD (Tensor.scalar (1 : Dual α))
  let (paramGradsD, inputGradsD) :=
    Pack.split (ss₁ := paramShapes) (ss₂ := inputShapes) gradsD
  pure (Pack.map DualTensor.tangent paramGradsD,
    Pack.map DualTensor.tangent inputGradsD)

/--
Full Hessian tensor for a scalar function of a single tensor input.

Columns are evaluated as `H * e_i` in the flattened input basis, then arranged with one copy of
the input axes for each derivative.
-/
def hessianInput {α : Type} [Storage α] [Context α]
    {σ : Shape}
    (f :
      ∀ {β : Type}, [Storage β] → [Context β] →
        Program β [σ] Shape.scalar)
    (x : Tensor α σ) :
    IO (Tensor α (σ ++ σ)) := do
  if σ.size = 0 then
    let _ ← hvp (α := α) (paramShapes := []) (inputShapes := [σ])
      (fun {β} _ _ _ => f (β := β)) .nil (.cons x .nil) .nil (.cons (Tensor.zeros σ) .nil)
  let columns ← Tensor.stackLeadingM fun (index : Fin σ.size) => do
    let dx := (Tensor.oneHot (α := α) σ.size index).reshape σ (by simp)
    let (_, .cons col .nil) ←
      hvp (α := α)
        (paramShapes := ([] : List Shape)) (inputShapes := [σ])
        (fun {β} _ _ _ => f (β := β)) .nil (.cons x .nil) .nil (.cons dx .nil)
    pure col
  let matrix := columns.reshape [σ.size, σ.size] (by simp)
  pure <| (Tensor.permute matrix
      (Shape.adjacentSwap [σ.size, σ.size] 0) [σ.size, σ.size]).reshape (σ ++ σ)
    (by simp [Shape.size_append])

end Autodiff
end TorchLean
