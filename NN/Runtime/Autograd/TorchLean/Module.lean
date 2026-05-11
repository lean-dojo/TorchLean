/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Utils
public import NN.Runtime.Autograd.TorchLean.Backend
public import NN.Runtime.Autograd.TorchLean.Training

import Mathlib.Algebra.Order.Algebra

/-!
# Module

TorchLean module wrappers with PyTorch-style ergonomics.

TorchLean already provides the core ingredients:
- a small `Ops` interface, so you write a model once and run it on different backends;
- `scalarTrainer`, which builds an eager or compiled training loop for scalar losses.

This file adds a thin “`nn.Module`-style” wrapper so users can:
- package **initial parameters** + a **loss definition** as a single object,
- instantiate it under a chosen backend (`.eager` / `.compiled`),
- call `forward / backward / step / params` with a small, consistent API.

Important: dtype selection is handled in `NN.API.DType` (because it picks the Lean type `α`).
The module definitions here are **polymorphic in `α`**, so the same module can be:
- used in executables with `Float` / `IEEE32Exec`, or
- instantiated at `ℝ` in proofs (noncomputable; not for `IO` execution).
-/

@[expose] public section


namespace Runtime
namespace Autograd
namespace TorchLean

open Spec
open Tensor
open Proofs.Autograd.Algebra

/-! ## Small helpers -/

namespace Module

/--
Cast a Float tensor to a backend scalar type `α` by mapping a scalar cast function.

This is mainly used to turn `tensorND!`-authored Float initializers into `Float`/`IEEE32Exec`/etc.
-/
def castTensor {α : Type} (cast : Float → α) {s : Shape} (t : Tensor Float s) : Tensor α s :=
  Spec.mapTensor cast t

/-- List-shaped version of `castTensor` for TorchLean's `TList` parameter bundles. -/
def castTList {α : Type} (cast : Float → α) : {ss : List Shape} → Torch.TList Float ss → Torch.TList
  α ss
  | [], .nil => .nil
  | _s :: ss, .cons x xs => .cons (castTensor cast x) (castTList (cast := cast) (ss := ss) xs)

/-! ## Scalar-loss module (training) -/

/--
A scalar-loss module definition:
- `initParams` is stored as Float constants (easy to write with `tensorND!`),
- `loss` is *polymorphic in the scalar backend* (same code works for Float/IEEE32Exec/…).

You can instantiate this definition as a `ScalarModule` under a chosen backend and dtype.
-/
structure ScalarModuleDef (paramShapes inputShapes : List Shape) where
  /-- Initial parameter values, stored as `Float` tensors and cast at instantiation time. -/
  initParams : Torch.TList Float paramShapes
  /-- Per-parameter `requiresGrad` flags aligned with `paramShapes`. -/
  initRequiresGrad : List Bool := List.replicate paramShapes.length true
  /-- Scalar loss program over `(params ++ inputs)`, polymorphic in the scalar backend. -/
  loss :
    ∀ {α : Type}, [Context α] → [DecidableEq Shape] →
      TorchLean.Program α (paramShapes ++ inputShapes) Shape.scalar

/--
Runtime module instance (the thing you "run").

This wraps `Torch.ScalarTrainer`, but exposes a more `Module`-like set of methods.
-/
structure ScalarModule (α : Type) [Context α] [DecidableEq Shape]
    (paramShapes inputShapes : List Shape) where
  trainer : Torch.ScalarTrainer α paramShapes inputShapes

namespace ScalarModule

/--
Create a runtime scalar-loss module from an explicit loss program and initial parameter values.

This is the low-level constructor; most users start from a `ScalarModuleDef` and call
`ScalarModuleDef.instantiate`.
-/
def create {α : Type} [Context α] [DecidableEq Shape]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    {paramShapes inputShapes : List Shape}
    (opts : Torch.Options := {})
    (initRequiresGrad : List Bool := List.replicate paramShapes.length true)
    (loss :
      ∀ {m : Type → Type}, [Monad m] → [Torch.Ops (m := m) (α := α)] →
        Torch.CurriedRef (fun s => Torch.Ops.Ref (m := m) (α := α) s) (paramShapes ++ inputShapes)
          (m (Torch.Ops.Ref (m := m) (α := α) Shape.scalar)))
    (initParams : Torch.TList α paramShapes) :
    IO (ScalarModule α paramShapes inputShapes) := do
  let mkTr :=
    Torch.scalarTrainer (α := α) (paramShapes := paramShapes) (inputShapes := inputShapes)
      (opts := opts) (initRequiresGrad := initRequiresGrad) (loss := loss)
  let tr ← Torch.Curried.uncurry (α := α) (ss := paramShapes)
    (β := IO (Torch.ScalarTrainer α paramShapes inputShapes)) mkTr initParams
  pure { trainer := tr }

/-- Run the forward pass and return the scalar loss value. -/
def forward {α : Type} [Context α] [DecidableEq Shape]
    {paramShapes inputShapes : List Shape}
    (m : ScalarModule α paramShapes inputShapes) (xs : Torch.TList α inputShapes) :
    IO (Tensor α Shape.scalar) :=
  Torch.ScalarTrainer.forwardT (α := α) (paramShapes := paramShapes) (inputShapes := inputShapes)
    m.trainer xs

/-- Run one forward/backward pass and return gradients for all parameters. -/
def backward {α : Type} [Context α] [DecidableEq Shape]
    {paramShapes inputShapes : List Shape}
    (m : ScalarModule α paramShapes inputShapes) (xs : Torch.TList α inputShapes) :
    IO (Torch.TList α paramShapes) :=
  Torch.ScalarTrainer.backwardT (α := α) (paramShapes := paramShapes) (inputShapes := inputShapes)
    m.trainer xs

/-- Convenience "one-step SGD": compute gradients and apply an SGD update with learning rate `lr`.
  -/
def step {α : Type} [Context α] [DecidableEq Shape]
    {paramShapes inputShapes : List Shape}
    (m : ScalarModule α paramShapes inputShapes) (lr : α) (xs : Torch.TList α inputShapes) :
    IO Unit :=
  Torch.ScalarTrainer.stepT (α := α) (paramShapes := paramShapes) (inputShapes := inputShapes)
    m.trainer lr xs

/-- Initialize an optimizer state for this module's parameters. -/
def initOptim {α : Type} [Context α] [DecidableEq Shape]
    {paramShapes inputShapes : List Shape}
    (m : ScalarModule α paramShapes inputShapes)
    (opt : TorchLean.Optim.Optimizer α paramShapes) :
    IO opt.State :=
  opt.init m.trainer.params

/--
Run one optimizer step using an explicit optimizer + state.

This mirrors a PyTorch training step:
1. compute gradients (`backwardT`)
2. update parameters via `opt.step` and return the new optimizer state
-/
def stepWith {α : Type} [Context α] [DecidableEq Shape]
    {paramShapes inputShapes : List Shape}
    (m : ScalarModule α paramShapes inputShapes)
    (opt : TorchLean.Optim.Optimizer α paramShapes) (st : opt.State)
    (xs : Torch.TList α inputShapes) :
    IO opt.State := do
  match ← opt.trainerStep? m.trainer st xs with
  | some st' =>
      pure st'
  | none =>
      let grads ← Torch.ScalarTrainer.backwardT (α := α)
        (paramShapes := paramShapes) (inputShapes := inputShapes) m.trainer xs
      opt.step st m.trainer.params grads

/-- Fetch the current parameter values as a shape-indexed list. -/
def params {α : Type} [Context α] [DecidableEq Shape]
    {paramShapes inputShapes : List Shape}
    (m : ScalarModule α paramShapes inputShapes) : IO (Torch.TList α paramShapes) :=
  m.trainer.getParams

/-- Overwrite all parameter values. -/
def setParams {α : Type} [Context α] [DecidableEq Shape]
    {paramShapes inputShapes : List Shape}
    (m : ScalarModule α paramShapes inputShapes) (ps : Torch.TList α paramShapes) : IO Unit :=
  Torch.ParamList.setValues (α := α) (ss := paramShapes) m.trainer.params ps

/-- Train with vanilla SGD for a fixed number of steps on a fixed list of samples. -/
def trainSGD {α : Type} [Context α] [DecidableEq Shape] [ToString α]
    {paramShapes inputShapes : List Shape}
    (m : ScalarModule α paramShapes inputShapes)
    (lr : α) (steps : Nat) (samples : List (Torch.TList α inputShapes))
    (logEvery : Nat := 1) : IO Unit :=
  Torch.trainCycleSGD (α := α) (paramShapes := paramShapes) (inputShapes := inputShapes)
    m.trainer lr steps samples (logEvery := logEvery)

/-- Like `trainSGD`, but with an explicit optimizer + mutable optimizer state. -/
def trainWith {α : Type} [Context α] [DecidableEq Shape] [ToString α]
    {paramShapes inputShapes : List Shape}
    (m : ScalarModule α paramShapes inputShapes)
    (opt : TorchLean.Optim.Optimizer α paramShapes) (st0 : opt.State)
    (steps : Nat) (samples : List (Torch.TList α inputShapes))
    (logEvery : Nat := 1) : IO opt.State :=
  TorchLean.trainCycleOptim (α := α) (paramShapes := paramShapes) (inputShapes := inputShapes)
    m.trainer opt st0 steps samples (logEvery := logEvery)

/-- Compute the mean loss over a list of samples (no parameter updates). -/
def meanLoss {α : Type} [Context α] [DecidableEq Shape] [ToString α] [Add α] [Div α] [Zero α] [Coe
  Nat α]
    {paramShapes inputShapes : List Shape}
    (m : ScalarModule α paramShapes inputShapes)
    (samples : List (Torch.TList α inputShapes)) : IO α :=
  Torch.meanLoss (α := α) (paramShapes := paramShapes) (inputShapes := inputShapes) m.trainer
    samples

end ScalarModule

namespace ScalarModuleDef

/--
Instantiate a `ScalarModuleDef` by casting Float initializers to `α` and choosing Torch options.

This is the most general constructor; `instantiate` is a convenience wrapper that just selects a
backend.
-/
def instantiateWith {α : Type} [Context α] [DecidableEq Shape]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    {paramShapes inputShapes : List Shape}
    (d : ScalarModuleDef paramShapes inputShapes)
    (cast : Float → α) (opts : Torch.Options) :
    IO (ScalarModule α paramShapes inputShapes) := do
  let initParams : Torch.TList α paramShapes := castTList (α := α) cast d.initParams
  ScalarModule.create (α := α) (paramShapes := paramShapes) (inputShapes := inputShapes)
    (opts := opts) (initRequiresGrad := d.initRequiresGrad) (loss := d.loss (α := α)) initParams

/-- Convenience instantiator that chooses only the backend (`.eager` or `.compiled`). -/
def instantiate {α : Type} [Context α] [DecidableEq Shape]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    {paramShapes inputShapes : List Shape}
    (d : ScalarModuleDef paramShapes inputShapes)
    (cast : Float → α) (backend : Torch.Backend := .eager) :
    IO (ScalarModule α paramShapes inputShapes) := do
  instantiateWith (α := α) (paramShapes := paramShapes) (inputShapes := inputShapes)
    d cast { backend := backend }

end ScalarModuleDef

end Module

end TorchLean
end Autograd
end Runtime
