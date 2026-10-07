/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Model.Program
public import NN.Runtime.Autograd.Model.Dual
public import NN.Runtime.Autograd.Torch.Core.Trainer

/-!
# Autodiff

Autodiff utilities beyond basic `.backward()`:

- `hvpParams`: Hessian-vector product for scalar losses w.r.t. parameters, using
  forward-over-reverse via `Dual` scalars.

This is runtime/executable functionality intended for TorchLean ergonomics; it is separate from
the `fderiv` proof developments.
-/

@[expose] public section


namespace Runtime
namespace Autograd
namespace Model

open Spec TorchLean
open TorchLean TorchLean.Tensor
open Proofs.Autograd.Algebra

namespace Autodiff

namespace Impl

/-- Return the seeded input gradients and forward result, reporting domain errors through IO. -/
def pullback {α : Type} [Storage α] [Add α] [Zero α]
    {shapes : List Shape} {output : Shape}
    (graph : Runtime.Autograd.Torch.TypedGraph α shapes output)
    (inputs : TorchLean.TensorPack α shapes) (seed : Tensor α output) :
    IO (TorchLean.TensorPack α shapes × Tensor α output) :=
  IO.ofExcept <| graph.vjpChecked inputs () seed

end Impl

open Impl

/--
Lower a TorchLean program to a reusable `TypedGraph`.

The graph retains a typed reference to the value returned by the program, which may be an input or
any recorded node. Scalar outputs use the same path with shape `[]`.
-/
def lowerToTypedGraph {α : Type} [TorchLean.Storage α] [Context α]
    {paramShapes inputShapes : List Shape} {τ : Shape}
    (f :
      ∀ {β : Type}, [TorchLean.Storage β] → [Context β] →
        Runtime.Autograd.Model.Program β (paramShapes ++ inputShapes) τ) :
    IO (Runtime.Autograd.Torch.TypedGraph α (paramShapes ++ inputShapes) τ) := do
  let Γ : List Shape := paramShapes ++ inputShapes
  let build :
      Runtime.Autograd.TypedGraph.GraphM.M α Γ (Runtime.Autograd.TypedGraph.GraphM.Var τ) := do
    let vs ← Runtime.Autograd.TypedGraph.GraphM.args (α := α) (Γ := Γ)
    CurriedRef.applyVarList (Γ := Γ)
      (β := Runtime.Autograd.TypedGraph.GraphM.M α Γ (Runtime.Autograd.TypedGraph.GraphM.Var τ))
      (f (β := α) (m := Runtime.Autograd.TypedGraph.GraphM.M α Γ)) vs
  IO.ofExcept (Runtime.Autograd.Torch.lowerToTypedGraph (α := α) (Γ := Γ) (τ := τ) build)

/-- Reverse Jacobian, with output axes prepended to each parameter tensor. -/
def jacrevParams {α : Type} [TorchLean.Storage α] [Context α]
    {paramShapes inputShapes : List Shape} {τ : Shape}
    (f :
      ∀ {β : Type}, [TorchLean.Storage β] → [Context β] →
        Runtime.Autograd.Model.Program β (paramShapes ++ inputShapes) τ)
    (params : TorchLean.TensorPack α paramShapes)
    (xs : TorchLean.TensorPack α inputShapes) :
    IO (TorchLean.TensorPack α (paramShapes.map τ.concat)) := do
  let c ← lowerToTypedGraph (α := α) (paramShapes := paramShapes) (inputShapes := inputShapes)
    (τ := τ) f
  let Γ : List Shape := paramShapes ++ inputShapes
  let args : TorchLean.TensorPack α Γ :=
    TorchLean.TensorPack.append (α := α) (ss₁ := paramShapes) (ss₂ := inputShapes) params xs
  let (tape, _) ← IO.ofExcept <|
    Runtime.Autograd.TypedGraph.lowerToTapeChecked c.data args ()
  TensorPack.stackLeadingM τ fun index => do
    let seedOut := (Tensor.oneHot (α := α) τ.size index).reshape τ (by simp [Spec.Shape.size])
    let allGradients ← IO.ofExcept <| c.vjpFromTape tape seedOut
    pure (TorchLean.TensorPack.split
      (ss₁ := paramShapes) (ss₂ := inputShapes) allGradients).1

/-- Reverse Jacobian, with output axes prepended to each input tensor. -/
def jacrevInputs {α : Type} [TorchLean.Storage α] [Context α]
    {paramShapes inputShapes : List Shape} {τ : Shape}
    (f :
      ∀ {β : Type}, [TorchLean.Storage β] → [Context β] →
        Runtime.Autograd.Model.Program β (paramShapes ++ inputShapes) τ)
    (params : TorchLean.TensorPack α paramShapes)
    (xs : TorchLean.TensorPack α inputShapes) :
    IO (TorchLean.TensorPack α (inputShapes.map τ.concat)) := do
  let c ← lowerToTypedGraph (α := α) (paramShapes := paramShapes) (inputShapes := inputShapes)
    (τ := τ) f
  let Γ : List Shape := paramShapes ++ inputShapes
  let args : TorchLean.TensorPack α Γ :=
    TorchLean.TensorPack.append (α := α) (ss₁ := paramShapes) (ss₂ := inputShapes) params xs
  let (tape, _) ← IO.ofExcept <|
    Runtime.Autograd.TypedGraph.lowerToTapeChecked c.data args ()
  TensorPack.stackLeadingM τ fun index => do
    let seedOut := (Tensor.oneHot (α := α) τ.size index).reshape τ (by simp [Spec.Shape.size])
    let allGradients ← IO.ofExcept <| c.vjpFromTape tape seedOut
    pure (TorchLean.TensorPack.split
      (ss₁ := paramShapes) (ss₂ := inputShapes) allGradients).2

/--
Compute the forward Jacobian for a single tensor input.

Each input basis vector runs through the checked graph JVP. This uses the same primitive rules as
directional differentiation, including stopped gradients and the runtime's choices at nonsmooth
points. The returned tensor has output axes followed by input axes.
-/
def jacfwdInput {α : Type} [TorchLean.Storage α] [Context α]
    {σ τ : Shape}
    (f :
      ∀ {β : Type}, [TorchLean.Storage β] → [Context β] →
        Runtime.Autograd.Model.Program β [σ] τ)
    (x : Tensor α σ) :
    IO (Tensor α (τ.concat σ)) := do
  let c ← lowerToTypedGraph (α := α)
    (paramShapes := ([] : List Shape)) (inputShapes := [σ]) (τ := τ)
    (fun {β} _ _ _ => f (β := β))
  if σ.size = 0 then
    let _ ← IO.ofExcept <|
      Runtime.Autograd.TypedGraph.jvpChecked c.data (.cons x .nil)
        (.cons (Tensor.zeros σ) .nil) ()
  let columns ← Tensor.stackLeadingM fun (index : Fin σ.size) => do
    let dx := (Tensor.oneHot (α := α) σ.size index).reshape σ (by simp [Spec.Shape.size])
    let (_, tangentContext) ← IO.ofExcept <|
      Runtime.Autograd.TypedGraph.jvpChecked c.data (.cons x .nil) (.cons dx .nil) ()
    pure (Proofs.getIdx tangentContext c.output)
  let matrix := columns.reshape [σ.size, τ.size] (by simp [Spec.Shape.size])
  pure <| (Tensor.swapAdjacentAxes matrix 0).reshape (τ.concat σ)
    (by simp [Spec.Shape.size_concat, Spec.Shape.size])

/--
Compute a tensor-output VJP with respect to parameters and inputs in one reverse pass.
-/
def vjp {α : Type} [TorchLean.Storage α] [Context α]
    {paramShapes inputShapes : List Shape} {τ : Shape}
    (f :
      ∀ {β : Type}, [TorchLean.Storage β] → [Context β] →
        Runtime.Autograd.Model.Program β (paramShapes ++ inputShapes) τ)
    (params : TorchLean.TensorPack α paramShapes)
    (xs : TorchLean.TensorPack α inputShapes)
    (seedOut : Tensor α τ) :
    IO (TorchLean.TensorPack α paramShapes ×
      TorchLean.TensorPack α inputShapes) := do
  let c ← lowerToTypedGraph (α := α) (paramShapes := paramShapes) (inputShapes := inputShapes)
    (τ := τ) f
  let Γ : List Shape := paramShapes ++ inputShapes
  let args : TorchLean.TensorPack α Γ :=
    TorchLean.TensorPack.append (α := α) (ss₁ := paramShapes) (ss₂ := inputShapes) params xs
  let (gAll, _) ← pullback c args seedOut
  pure (TorchLean.TensorPack.split (α := α)
    (ss₁ := paramShapes) (ss₂ := inputShapes) gAll)

/-- Differentiate a scalar loss with respect to parameters and inputs in one reverse pass.

This is the general VJP with a unit scalar seed. -/
def gradients {α : Type} [TorchLean.Storage α] [Context α]
    {paramShapes inputShapes : List Shape}
    (loss :
      ∀ {β : Type}, [TorchLean.Storage β] → [Context β] →
        Runtime.Autograd.Model.Program β (paramShapes ++ inputShapes) Shape.scalar)
    (params : TorchLean.TensorPack α paramShapes)
    (xs : TorchLean.TensorPack α inputShapes) :
    IO (TorchLean.TensorPack α paramShapes ×
      TorchLean.TensorPack α inputShapes) :=
  vjp loss params xs (Tensor.scalar (1 : α))

/-- Directional derivative of scalar loss along `vparams` (forward-mode JVP). -/
def jvpLossParams {α : Type} [TorchLean.Storage α] [Context α]
    {paramShapes inputShapes : List Shape}
    (loss :
      ∀ {β : Type}, [TorchLean.Storage β] → [Context β] →
        Runtime.Autograd.Model.Program β (paramShapes ++ inputShapes) Shape.scalar)
    (params : TorchLean.TensorPack α paramShapes)
    (xs : TorchLean.TensorPack α inputShapes)
    (vparams : TorchLean.TensorPack α paramShapes) :
    IO (Tensor α []) := do
  let c ← lowerToTypedGraph (α := α)
    (paramShapes := paramShapes) (inputShapes := inputShapes) loss
  let Γ : List Shape := paramShapes ++ inputShapes
  let args : TorchLean.TensorPack α Γ :=
    TorchLean.TensorPack.append (α := α) (ss₁ := paramShapes) (ss₂ := inputShapes) params xs
  let zerosX : TorchLean.TensorPack α inputShapes :=
    TorchLean.TensorPack.zero (α := α) (ss := inputShapes)
  let dargs : TorchLean.TensorPack α Γ :=
    TorchLean.TensorPack.append (α := α) (ss₁ := paramShapes) (ss₂ := inputShapes) vparams zerosX
  let (_, tangentContext) ← IO.ofExcept <|
    Runtime.Autograd.TypedGraph.jvpChecked c.data args dargs ()
  pure (Proofs.getIdx tangentContext c.output)

/-- Reverse-mode gradients over dual-valued arguments, shared by both HVP projections. -/
def Impl.dualGradients {α : Type} [TorchLean.Storage α] [Context α]
    {paramShapes inputShapes : List Shape}
    (loss : ∀ {β : Type}, [TorchLean.Storage β] → [Context β] →
      Runtime.Autograd.Model.Program β (paramShapes ++ inputShapes) Shape.scalar)
    (argsD : TorchLean.TensorPack (Dual α) (paramShapes ++ inputShapes)) :
    IO (TorchLean.TensorPack (Dual α) (paramShapes ++ inputShapes)) := do
  let αD := Dual α
  let Γ : List Shape := paramShapes ++ inputShapes
  let graph ← lowerToTypedGraph (α := αD)
    (paramShapes := paramShapes) (inputShapes := inputShapes) loss
  let ssFull : List Shape := graph.nodeShapes
  let fullGraph : Proofs.Autograd.Algebra.GraphData αD Unit Γ ssFull :=
    graph.data

  let (tape, _ctx) ← IO.ofExcept <|
    Runtime.Autograd.TypedGraph.lowerToTapeChecked fullGraph argsD ()
  IO.ofExcept <| graph.vjpFromTape tape (Tensor.scalar (1 : αD))

/--
Hessian-vector product (HVP) for a scalar loss w.r.t. *parameters*.

This computes `d/dε (∇_params loss(params + ε*vparams)) |_{ε=0}` and returns a
`TorchLean.TensorPack` aligned with `paramShapes`.

Implementation: run reverse-mode AD over dual scalars (`Dual`), with parameter tangents set to
`vparams` and input tangents set to `0`. The tangent part of the resulting gradients is the HVP.
-/
def hvpParams {α : Type} [TorchLean.Storage α] [Context α]
    {paramShapes inputShapes : List Shape}
    (loss :
      ∀ {β : Type}, [TorchLean.Storage β] → [Context β] →
        Runtime.Autograd.Model.Program β (paramShapes ++ inputShapes) Shape.scalar)
    (params : TorchLean.TensorPack α paramShapes)
    (xs : TorchLean.TensorPack α inputShapes)
    (vparams : TorchLean.TensorPack α paramShapes) :
    IO (TorchLean.TensorPack α paramShapes) := do
  let αD := Dual α

  let paramsD : TorchLean.TensorPack αD paramShapes :=
    TorchLean.TensorPack.zipWith DualTensor.withTangents params vparams
  let xsD : TorchLean.TensorPack αD inputShapes :=
    TorchLean.TensorPack.map DualTensor.ofPrimal xs

  let Γ : List Shape := paramShapes ++ inputShapes
  let argsD : TorchLean.TensorPack αD Γ :=
    TorchLean.TensorPack.append (α := αD) (ss₁ := paramShapes) (ss₂ := inputShapes) paramsD xsD

  let gradsD ← dualGradients loss argsD
  let gradsParamsD : TorchLean.TensorPack αD paramShapes :=
    (TorchLean.TensorPack.split (α := αD) (ss₁ := paramShapes) (ss₂ := inputShapes) gradsD).1

  pure (TorchLean.TensorPack.map DualTensor.tangent gradsParamsD)

/--
Hessian-vector product (HVP) for a scalar loss w.r.t. *inputs*.

This computes `d/dε (∇_xs loss(xs + ε*vxs)) |_{ε=0}` and returns a `TorchLean.TensorPack`
aligned with `inputShapes`.

Implementation: the same forward-over-reverse trick as `hvpParams`, but we attach tangents to
inputs instead of parameters.
-/
def hvpInputs {α : Type} [TorchLean.Storage α] [Context α]
    {paramShapes inputShapes : List Shape}
    (loss :
      ∀ {β : Type}, [TorchLean.Storage β] → [Context β] →
        Runtime.Autograd.Model.Program β (paramShapes ++ inputShapes) Shape.scalar)
    (params : TorchLean.TensorPack α paramShapes)
    (xs : TorchLean.TensorPack α inputShapes)
    (vxs : TorchLean.TensorPack α inputShapes) :
    IO (TorchLean.TensorPack α inputShapes) := do
  let αD := Dual α

  let paramsD : TorchLean.TensorPack αD paramShapes :=
    TorchLean.TensorPack.map DualTensor.ofPrimal params
  let xsD : TorchLean.TensorPack αD inputShapes :=
    TorchLean.TensorPack.zipWith DualTensor.withTangents xs vxs

  let Γ : List Shape := paramShapes ++ inputShapes
  let argsD : TorchLean.TensorPack αD Γ :=
    TorchLean.TensorPack.append (α := αD) (ss₁ := paramShapes) (ss₂ := inputShapes) paramsD xsD

  let gradsD ← dualGradients loss argsD
  let gradsInputsD : TorchLean.TensorPack αD inputShapes :=
    (TorchLean.TensorPack.split (α := αD) (ss₁ := paramShapes) (ss₂ := inputShapes) gradsD).2

  pure (TorchLean.TensorPack.map DualTensor.tangent gradsInputsD)

/--
Full Hessian tensor for a scalar function of a single tensor input.

Columns are evaluated as `H * e_i` in the flattened input basis, then arranged with one copy of
the input axes for each derivative.
-/
def hessianInput {α : Type} [TorchLean.Storage α] [Context α]
    {σ : Shape}
    (f :
      ∀ {β : Type}, [TorchLean.Storage β] → [Context β] →
        Runtime.Autograd.Model.Program β [σ] Shape.scalar)
    (x : Tensor α σ) :
    IO (Tensor α (σ.concat σ)) := do
  if σ.size = 0 then
    let _ ← hvpInputs (α := α) (paramShapes := []) (inputShapes := [σ])
      (fun {β} _ _ _ => f (β := β)) .nil (.cons x .nil) (.cons (Tensor.zeros σ) .nil)
  let columns ← Tensor.stackLeadingM fun (index : Fin σ.size) => do
    let dx := (Tensor.oneHot (α := α) σ.size index).reshape σ (by simp [Spec.Shape.size])
    let hvp : TorchLean.TensorPack α [σ] ←
      hvpInputs (α := α)
        (paramShapes := ([] : List Shape)) (inputShapes := [σ])
        (fun {β} _ _ _ => f (β := β)) .nil (.cons x .nil) (.cons dx .nil)
    let .cons col .nil := hvp
    pure col
  let matrix := columns.reshape [σ.size, σ.size] (by simp [Spec.Shape.size])
  pure <| (Tensor.swapAdjacentAxes matrix 0).reshape (σ.concat σ)
    (by simp [Spec.Shape.size_concat, Spec.Shape.size])

end Autodiff

end Model
end Autograd
end Runtime
