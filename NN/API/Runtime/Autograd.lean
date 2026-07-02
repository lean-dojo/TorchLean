/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Core
public import NN.API.Rand
public import NN.API.TorchLean.ParamIO
public import NN.API.TorchLean.Schedulers
public import NN.GraphSpec.Models.TorchLean
public import NN.Runtime.Autograd.TorchLean
public import NN.Runtime.Autograd.Train.Dataset
public import NN.Runtime.RL
public import NN.Spec.RL.MDP
public import NN.Spec.RL.MarkovMDP
public import NN.Spec.RL.FiniteStochasticMDP
public import NN.API.Runtime.Core
public import NN.API.Runtime.Layers

import Mathlib.Algebra.Order.Algebra
import NN.Spec.Autograd.AutogradSpec

@[expose] public section

namespace NN
namespace API
namespace TorchLean

/-!
# Runtime Autograd Helpers

Model-shaped and function-shaped differentiation helpers over TorchLean runtime programs.
-/

namespace Autodiff

namespace Model

/-
The declarations below provide "model-shaped" autodiff helpers:
- a `Seq σ τ` model,
- an `OutputLoss τ υ` (loss built from model output + target),
- and model-shaped entrypoints for VJP/Jacobian/HVP/JVP and gradient extraction.

These helpers delegate to `Runtime.Autograd.TorchLean.Autodiff` while handling the program/argument
packing that model call sites would otherwise repeat.
-/

/-- Parameter pack type for a given model (a `TensorPack` over `Seq.paramShapes`). -/
abbrev Params {σ τ : Spec.Shape} (model : API.TorchLean.NN.Seq σ τ) (α : Type) :=
  API.TorchLean.TensorPack α (API.TorchLean.NN.Seq.paramShapes model)

/--
Loss function over a model output and a target.

This is expressed in terms of `RefTy` so it works uniformly for eager execution and compiled
execution.
-/
abbrev OutputLoss (τ υ : Spec.Shape) :=
  ∀ {α : Type}, [Context α] → [DecidableEq Spec.Shape] →
    {m : Type → Type} → [Monad m] → [API.TorchLean.Ops (m := m) (α := α)] →
      API.TorchLean.RefTy (m := m) (α := α) τ →
      API.TorchLean.RefTy (m := m) (α := α) υ →
      m (API.TorchLean.RefTy (m := m) (α := α) Spec.Shape.scalar)

/--
Initialize model parameters by casting the model's `Float` initializers elementwise using `cast`.
-/
def initParamsWith {σ τ : Spec.Shape} (model : API.TorchLean.NN.Seq σ τ)
    {α : Type} (cast : Float → α) :
    Params model α :=
  _root_.Runtime.Autograd.TorchLean.Module.castTList cast (API.TorchLean.NN.Seq.initParams model)

/-- Initialize model parameters using the runtime literal injection `API.Runtime.ofFloat`. -/
def initParams {σ τ : Spec.Shape} (model : API.TorchLean.NN.Seq σ τ)
    {α : Type} [API.Runtime.Scalar α] :
    Params model α :=
  Model.initParamsWith (model := model) API.Runtime.ofFloat

/-- Pack explicit weight and bias tensors for a single `Layers.linear` model. -/
def linearParams {α : Type} {inDim outDim : Nat} {seedW seedB : Nat}
    (w : _root_.Spec.Tensor α (NN.Tensor.Shape.Mat outDim inDim))
    (b : _root_.Spec.Tensor α (NN.Tensor.Shape.Vec outDim)) :
    Params (API.TorchLean.Layers.linear inDim outDim seedW seedB) α :=
  API.TorchLean.tensorpack2 w b

namespace OutputLoss

/-- Mean-squared error loss (`mse`) between `yhat` and `y`. -/
def mse {τ : Spec.Shape} (reduction : API.TorchLean.Loss.Reduction := .mean) :
    OutputLoss τ τ :=
  fun {α} _ _ =>
    fun {m} _ _ yhat y =>
      API.TorchLean.Loss.mse (m := m) (α := α) (s := τ) yhat y (reduction := reduction)

/-- Cross-entropy loss between logits and one-hot targets. PyTorch analogue: `nn.CrossEntropyLoss`.
  -/
def crossEntropyOneHot {τ : Spec.Shape} (reduction : API.TorchLean.Loss.Reduction := .mean) :
    OutputLoss τ τ :=
  fun {α} _ _ =>
    fun {m} _ _ logits targetOneHot =>
      API.TorchLean.Loss.crossEntropyOneHot (m := m) (α := α) (s := τ) logits targetOneHot
        (reduction := reduction)

/--
Detach the model output before feeding it into a loss.

This is useful when you want to compute a metric loss without backpropagating through it.
-/
def detach {τ υ : Spec.Shape} (loss : OutputLoss τ υ) : OutputLoss τ υ :=
  fun {α} _ _ =>
    fun {m} _ _ yhat y => do
      let yhat' ← API.TorchLean.F.detach (m := m) (α := α) (s := τ) yhat
      loss (α := α) (m := m) yhat' y

end OutputLoss

/--
Build a TorchLean `Program` that computes a scalar loss from `(params, x, target)`.

This is the bridge between `Seq.program` (which produces model outputs) and the autograd entry
points (which expect a scalar-valued program).
-/
def lossProgram {σ τ υ : Spec.Shape} (model : API.TorchLean.NN.Seq σ τ) (loss : OutputLoss τ υ) :
    ∀ {α : Type}, [Context α] → [DecidableEq Spec.Shape] →
      API.TorchLean.Program α (API.TorchLean.NN.Seq.paramShapes model ++ [σ, υ]) Spec.Shape.scalar
        :=
  fun {α} _ _ =>
    fun {m} _ _ =>
      _root_.Runtime.Autograd.Torch.CurriedRef.curry
        (Ref := fun s => API.TorchLean.RefTy (m := m) (α := α) s)
        (ss := API.TorchLean.NN.Seq.paramShapes model ++ [σ, υ])
        (β := m (API.TorchLean.RefTy (m := m) (α := α) Spec.Shape.scalar))
        (fun args => do
          let (ps, xy) :=
            _root_.Runtime.Autograd.Torch.RefList.split
              (Ref := fun s => API.TorchLean.RefTy (m := m) (α := α) s)
              (ss₁ := API.TorchLean.NN.Seq.paramShapes model) (ss₂ := [σ, υ]) args
          let (x, y) := RefList.unpack2 xy
          let yhat ←
            _root_.Runtime.Autograd.Torch.CurriedRef.uncurry
              (Ref := fun s => API.TorchLean.RefTy (m := m) (α := α) s)
              (ss := API.TorchLean.NN.Seq.paramShapes model ++ [σ])
              (β := m (API.TorchLean.RefTy (m := m) (α := α) τ))
              (API.TorchLean.NN.Seq.program (model := model) (α := α))
              (_root_.Runtime.Autograd.Torch.RefList.append ps (.cons x .nil))
          loss (α := α) (m := m) yhat y)

/-- VJP of the model output w.r.t. parameters. -/
def vjpParams {σ τ : Spec.Shape} (model : API.TorchLean.NN.Seq σ τ)
    {α : Type} [Context α] [DecidableEq Spec.Shape]
    (params : Params model α) (x : Spec.Tensor α σ) (seedOut : Spec.Tensor α τ) :
    IO (Params model α) :=
  _root_.Runtime.Autograd.TorchLean.Autodiff.vjpOutParams
    (α := α)
    (paramShapes := API.TorchLean.NN.Seq.paramShapes model) (inputShapes := [σ]) (τ := τ)
    (fun {β} _ _ => API.TorchLean.NN.Seq.program (model := model) (α := β))
    params (API.TorchLean.tensorpack1 x) seedOut

/-- VJP of the model output w.r.t. inputs. -/
def vjpInputs {σ τ : Spec.Shape} (model : API.TorchLean.NN.Seq σ τ)
    {α : Type} [Context α] [DecidableEq Spec.Shape]
    (params : Params model α) (x : Spec.Tensor α σ) (seedOut : Spec.Tensor α τ) :
    IO (API.TorchLean.TensorPack α [σ]) :=
  _root_.Runtime.Autograd.TorchLean.Autodiff.vjpOutInputs
    (α := α)
    (paramShapes := API.TorchLean.NN.Seq.paramShapes model) (inputShapes := [σ]) (τ := τ)
    (fun {β} _ _ => API.TorchLean.NN.Seq.program (model := model) (α := β))
    params (API.TorchLean.tensorpack1 x) seedOut

/-- Jacobian (reverse-mode) of the model output w.r.t. parameters, returned as rows. -/
def jacrevParams {σ τ : Spec.Shape} (model : API.TorchLean.NN.Seq σ τ)
    {α : Type} [Context α] [DecidableEq Spec.Shape]
    (params : Params model α) (x : Spec.Tensor α σ) :
    IO (Array (Params model α)) :=
  _root_.Runtime.Autograd.TorchLean.Autodiff.jacrevOutParams
    (α := α)
    (paramShapes := API.TorchLean.NN.Seq.paramShapes model) (inputShapes := [σ]) (τ := τ)
    (fun {β} _ _ => API.TorchLean.NN.Seq.program (model := model) (α := β))
    params (API.TorchLean.tensorpack1 x)

/-- Gradient of `loss(model(params, x), target)` w.r.t. parameters. -/
def gradParams {σ τ υ : Spec.Shape} (model : API.TorchLean.NN.Seq σ τ) (loss : OutputLoss τ υ)
    {α : Type} [Context α] [DecidableEq Spec.Shape]
    (params : Params model α) (x : Spec.Tensor α σ) (target : Spec.Tensor α υ) :
    IO (Params model α) :=
  _root_.Runtime.Autograd.TorchLean.Autodiff.gradParams
    (α := α)
    (paramShapes := API.TorchLean.NN.Seq.paramShapes model) (inputShapes := [σ, υ])
    (lossProgram (model := model) loss)
    params (API.TorchLean.tensorpack2 x target)

/-- Gradient of `loss(model(params, x), target)` w.r.t. inputs (`x` and `target`). -/
def gradInputs {σ τ υ : Spec.Shape} (model : API.TorchLean.NN.Seq σ τ) (loss : OutputLoss τ υ)
    {α : Type} [Context α] [DecidableEq Spec.Shape]
    (params : Params model α) (x : Spec.Tensor α σ) (target : Spec.Tensor α υ) :
    IO (API.TorchLean.TensorPack α [σ, υ]) :=
  _root_.Runtime.Autograd.TorchLean.Autodiff.gradInputs
    (α := α)
    (paramShapes := API.TorchLean.NN.Seq.paramShapes model) (inputShapes := [σ, υ])
    (lossProgram (model := model) loss)
    params (API.TorchLean.tensorpack2 x target)

/-- JVP of a scalar loss w.r.t. parameters in direction `vparams`. -/
def jvpParams {σ τ υ : Spec.Shape} (model : API.TorchLean.NN.Seq σ τ) (loss : OutputLoss τ υ)
    {α : Type} [Context α] [DecidableEq Spec.Shape]
    (params : Params model α) (x : Spec.Tensor α σ) (target : Spec.Tensor α υ)
    (vparams : Params model α) :
    IO α :=
  _root_.Runtime.Autograd.TorchLean.Autodiff.jvpLossParams
    (α := α)
    (paramShapes := API.TorchLean.NN.Seq.paramShapes model) (inputShapes := [σ, υ])
    (lossProgram (model := model) loss)
    params (API.TorchLean.tensorpack2 x target) vparams

/-- HVP (Hessian-vector product) of a scalar loss w.r.t. parameters in direction `vparams`. -/
def hvpParams {σ τ υ : Spec.Shape} (model : API.TorchLean.NN.Seq σ τ) (loss : OutputLoss τ υ)
    {α : Type} [Context α] [DecidableEq Spec.Shape]
    (params : Params model α) (x : Spec.Tensor α σ) (target : Spec.Tensor α υ)
    (vparams : Params model α) :
    IO (Params model α) :=
  _root_.Runtime.Autograd.TorchLean.Autodiff.hvpParams
    (α := α)
    (paramShapes := API.TorchLean.NN.Seq.paramShapes model) (inputShapes := [σ, υ])
    (lossProgram (model := model) loss)
    params (API.TorchLean.tensorpack2 x target) vparams

end Model

namespace Function1

/-
Function-1 autodiff helpers.

This is the "no parameters" case: treat a pure tensor function `f : Tensor σ -> Tensor τ` as the
thing we differentiate, rather than a model with an explicit parameter list.
-/

/--
Type of a pure tensor function expressed in `RefTy` form.

This matches the calling convention expected by `TorchLean.Program`/autodiff compilation.
-/
abbrev Fn (σ τ : Spec.Shape) :=
  ∀ {α : Type}, [Context α] → [DecidableEq Spec.Shape] →
    {m : Type → Type} → [Monad m] → [API.TorchLean.Ops (m := m) (α := α)] →
      API.TorchLean.RefTy (m := m) (α := α) σ →
      m (API.TorchLean.RefTy (m := m) (α := α) τ)

/-- Turn an `Fn` into a single-input TorchLean `Program`. -/
def program {σ τ : Spec.Shape} (f : Fn σ τ) :
    ∀ {α : Type}, [Context α] → [DecidableEq Spec.Shape] → API.TorchLean.Program α [σ] τ :=
  fun {α} _ _ =>
    fun {m} _ _ =>
      _root_.Runtime.Autograd.Torch.CurriedRef.curry
        (Ref := fun s => API.TorchLean.RefTy (m := m) (α := α) s)
        (ss := [σ]) (β := m (API.TorchLean.RefTy (m := m) (α := α) τ))
        (fun args =>
          match args with
          | .cons x .nil => f (α := α) (m := m) x)

/-- Forward-mode Jacobian (rows) of a pure function. -/
def jacfwd {σ τ : Spec.Shape} (f : Fn σ τ)
    {α : Type} [Context α] [DecidableEq Spec.Shape]
    (x : Spec.Tensor α σ) :
    IO (Array (Spec.Tensor α τ)) :=
  _root_.Runtime.Autograd.TorchLean.Autodiff.jacfwd1
    (α := α) (σ := σ) (τ := τ) (program f) x

/-- Hessian for a scalar-valued pure function. -/
def hessian {σ : Spec.Shape} (f : Fn σ Spec.Shape.scalar)
    {α : Type} [Context α] [DecidableEq Spec.Shape]
    (x : Spec.Tensor α σ) :
    IO (Array (Spec.Tensor α σ)) :=
  _root_.Runtime.Autograd.TorchLean.Autodiff.hessian1
    (α := α) (σ := σ) (program f) x

end Function1

end Autodiff

end TorchLean
end API
end NN
