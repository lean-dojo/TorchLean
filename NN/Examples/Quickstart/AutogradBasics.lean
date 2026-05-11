/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN

/-!
# Quickstart: Autograd Basics

Tour of the public autograd APIs beyond `.backward()`:

- `API.autograd.model.*` for model-level VJP / jacobian / loss gradients.
- `API.autograd.model.OutputLoss.*` for reusable scalar losses on model outputs.
- `API.autograd.fn1.*` for Jacobian / Hessian helpers on single-input tensor functions.
- `API.nn.functional.detach` for stop-gradient behavior.

Run:
  `lake exe torchlean quickstart_autograd --dtype float --backend eager`
  `lake exe torchlean quickstart_autograd --dtype float32 --backend compiled`
-/

@[expose] public section


namespace NN.Examples.Quickstart.AutogradBasics

open Spec
open Tensor
open NN.Tensor
open NN.API

/-!
This file is a curated "autodiff API tour". It intentionally avoids:

- low-level runtime tape/session code,
- hand-written parameter-shape bookkeeping,
- noisy `castTensor` helpers.

Instead, it uses the public `API.autograd.*` facade and `tensorF! cast ...` to build deterministic
constants for any runtime scalar `α`.
-/

def model : nn.Sequential (Shape.Vec 2) (Shape.Vec 3) :=
  -- One Linear layer: y = x ↦ W*x + b
  nn.build 0 <| nn.linear 2 3 (pfx := Spec.Shape.scalar)

def mseLoss : autograd.model.OutputLoss (Shape.Vec 3) (Shape.Vec 3) :=
  -- Reusable scalar loss on model outputs: MSE(pred, target).
  autograd.model.OutputLoss.mse (τ := Shape.Vec 3)

def detachedMSELoss : autograd.model.OutputLoss (Shape.Vec 3) (Shape.Vec 3) :=
  -- Same forward value as `mseLoss`, but all gradients are zero (stop-gradient / detach).
  autograd.model.OutputLoss.detach mseLoss

def squareFn : autograd.fn1.Fn (Shape.Vec 2) (Shape.Vec 2) :=
  fun x => nn.functional.square x

def sumsqFn : autograd.fn1.Fn (Shape.Vec 2) Spec.Shape.scalar :=
  fun x => do
    let y ← nn.functional.square x
    -- `mean` is a convenient scalar reduction, like `torch.mean`.
    nn.functional.mean y

def runOnce {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    (cast : Float → α) : IO Unit := do

  -- Deterministic parameters + a single (x, y_target) sample.
  --
  -- `tensorF! cast ...` lets us write Float literals once and then map them into any runtime scalar
  -- `α`.
  let W : Spec.Tensor α (Shape.Mat 3 2) :=
    tensorF! cast [3, 2] [
      0.2, -0.1,
      0.0,  0.3,
     -0.4,  0.1
    ]
  let b : Spec.Tensor α (Shape.Vec 3) := tensorF! cast [3] [0.01, -0.02, 0.03]
  let x : Spec.Tensor α (Shape.Vec 2) := tensorF! cast [2] [0.5, -1.2]
  let y : Spec.Tensor α (Shape.Vec 3) := tensorF! cast [3] [0.7, 0.1, -0.5]

  -- `autograd.model.linearParams` builds parameters for the *bare* `TorchLean.Layers.linear` layer.
  -- Here `model` is the public `API.nn.linear` sequential wrapper, so we construct its parameter
  -- record directly as the expected `TList` shape.
  let params : autograd.model.Params model α := by
    simpa [autograd.model.Params, model] using (tlist! W, b)

  -- ------------------------------------------------------------
  -- 1) Tensor-output compilation: VJP with an explicit output seed
  -- ------------------------------------------------------------
  --
  -- `vjpParams` computes a vector-Jacobian product (reverse-mode) for a tensor-output model.
  -- You provide an explicit output cotangent `seedOut` and get cotangents for the parameters.
  --
  -- Here the model output is `Vec 3`, and we choose `seedOut = ones`.
  -- Intuition: we backprop `sum(y)` w.r.t. parameters.
  let seedOut : Spec.Tensor α (Shape.Vec 3) := Spec.fill (α := α) (cast 1.0) (Shape.Vec 3)
  let vjpParams ←
    autograd.model.vjpParams (α := α) model params x seedOut

  let (dW, db) := tlist.unpack2 vjpParams
  IO.println s!"vjpOutParams (seed=ones) dW = {Spec.pretty dW}"
  IO.println s!"vjpOutParams (seed=ones) db = {Spec.pretty db}"

  -- `jacrevParams` returns the full Jacobian of the model output w.r.t. parameters:
  -- one row per output coordinate. Each row is itself a typed list matching the parameter
  -- structure.
  let jacRows ←
    autograd.model.jacrevParams (α := α) model params x
  IO.println s!"jacrevOutParams rows = {jacRows.size} (should be size(out)=3)"
  for i in List.finRange jacRows.size do
    let row := jacRows[i.1]'i.2
    let (dWi, dbi) := tlist.unpack2 row
    IO.println s!"  row[{i.1}] dW = {Spec.pretty dWi}; db = {Spec.pretty dbi}"

  -- ------------------------------------------------------------
  -- 2) Reverse-mode grad for scalar loss
  -- ------------------------------------------------------------
  --
  -- This is the "PyTorch-style training" case: differentiate a scalar loss.
  -- `valueAndGradParamsScalar` is the one-liner: it runs forward+backward and returns:
  -- - the scalar loss value
  -- - parameter gradients (same typed-list structure/order as `params`)
  let (lossMSE, gParams) ←
    autograd.model.valueAndGradParamsScalar (α := α) model mseLoss params x y
  let (gW, gb) := tlist.unpack2 gParams
  IO.println s!"loss(mse) = {lossMSE}"
  IO.println s!"gradParams (mse) gW = {Spec.pretty gW}"
  IO.println s!"gradParams (mse) gb = {Spec.pretty gb}"

  -- ------------------------------------------------------------
  -- 3) Detach semantics: same forward value, zero gradient
  -- ------------------------------------------------------------
  --
  -- This matches PyTorch `detach()`: stop-gradient on the loss computation.
  let (lossDetached, gParamsDetached) ←
    autograd.model.valueAndGradParamsScalar (α := α) model detachedMSELoss params x y
  let (gW0, gb0) := tlist.unpack2 gParamsDetached
  IO.println s!"loss(mse ∘ detach) = {lossDetached}"
  IO.println s!"gradParams (mse ∘ detach) gW = {Spec.pretty gW0}"
  IO.println s!"gradParams (mse ∘ detach) gb = {Spec.pretty gb0}"

  -- ------------------------------------------------------------
  -- 4) Forward-mode JVP of the scalar loss along a parameter direction
  -- ------------------------------------------------------------
  --
  -- JVP = Jacobian-vector product (forward-mode). Here we compute the directional derivative of the
  -- scalar loss along a parameter perturbation direction `vparams`.
  let vW : Spec.Tensor α (Shape.Mat 3 2) :=
    Spec.fill (α := α) (cast 0.1) (Shape.Mat 3 2)
  let vb : Spec.Tensor α (Shape.Vec 3) :=
    Spec.fill (α := α) (cast (-0.2)) (Shape.Vec 3)
  let vparams : autograd.model.Params model α := tlist! vW, vb
  let dl ←
    autograd.model.jvpParams (α := α) model mseLoss params x y vparams
  IO.println s!"jvpLossParams dl = {dl}"

  -- ------------------------------------------------------------
  -- 5) Hessian-vector product (HVP) w.r.t. parameters
  -- ------------------------------------------------------------
  --
  -- HVP = (Hessian of loss) applied to a direction vector, without materializing the full Hessian.
  let hvp ←
    autograd.model.hvpParams (α := α) model mseLoss params x y vparams
  let (hW, hb) := tlist.unpack2 hvp
  IO.println s!"hvpParams hW = {Spec.pretty hW}"
  IO.println s!"hvpParams hb = {Spec.pretty hb}"

  -- ------------------------------------------------------------
  -- 6) jacfwd/jacrev/hessian for a function of a single tensor input
  -- ------------------------------------------------------------
  let jacCols ← autograd.fn1.jacfwd (α := α) squareFn x
  IO.println s!"jacfwd1(square) cols = {jacCols.size} (should be size(in)=2)"
  for i in List.finRange jacCols.size do
    let col := jacCols[i.1]'i.2
    IO.println s!"  col[{i.1}] = {Spec.pretty col}"

  let hessCols ← autograd.fn1.hessian (α := α) sumsqFn x
  IO.println s!"hessian1(mean(x^2)) cols = {hessCols.size} (should be size(in)=2)"
  for i in List.finRange hessCols.size do
    let col := hessCols[i.1]'i.2
    IO.println s!"  H*e[{i.1}] = {Spec.pretty col}"

  -- ------------------------------------------------------------
  -- 7) One-liners: vjp / jacrev / grad / valueAndGrad
  -- ------------------------------------------------------------
  --
  -- These wrappers are "no `TList` noise" helpers for the common case of a single tensor input.
  let seedSq : Spec.Tensor α (Shape.Vec 2) :=
    Spec.fill (α := α) (cast 1.0) (Shape.Vec 2)
  let vjpSq ← autograd.fn1.vjp (α := α) squareFn x seedSq
  IO.println s!"vjp(square, seed=ones) = {Spec.pretty vjpSq}"

  let jacRowsSq ← autograd.fn1.jacrev (α := α) squareFn x
  IO.println s!"jacrev1(square) rows = {jacRowsSq.size} (should be size(out)=2)"
  for i in List.finRange jacRowsSq.size do
    let row := jacRowsSq[i.1]'i.2
    IO.println s!"  row[{i.1}] = {Spec.pretty row}"

  let gSumsq ← autograd.fn1.grad (α := α) sumsqFn x
  IO.println s!"grad1(mean(x^2)) = {Spec.pretty gSumsq}"

  let (valSumsq, gSumsq2) ← autograd.fn1.valueAndGradScalar (α := α) sumsqFn x
  IO.println s!"valueAndGradScalar(mean(x^2)) value = {valSumsq}, grad = {Spec.pretty gSumsq2}"

def main (args : List String) : IO Unit := do
  let args := API.CLI.dropDashDash args
  _root_.NN.API.TorchLean.Module.withRuntime args (fun {α} _ _ _ _ cast _opts rest => do
    API.Common.orThrow "quickstart_autograd" <| API.CLI.requireNoArgs rest
    runOnce (α := α) cast)

end NN.Examples.Quickstart.AutogradBasics
