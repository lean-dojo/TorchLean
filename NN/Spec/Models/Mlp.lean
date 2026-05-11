/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Autograd.Ops
public import NN.Spec.Module.Activation
public import NN.Spec.Module.Linear

/-!
# MLP (spec wiring example)

This file defines a small 2-layer MLP by composing `SpecChain`s from module specs:

`Linear → ReLU → Linear` (optionally followed by a softmax head).

This is intentionally written in a "wiring-first" style: instead of re-implementing matrix
multiplications directly, we reuse the spec-layer definitions for `Linear` and `ReLU` and compose
them through `NNModuleSpec` / `SpecChain`. This matches how most PyTorch users think about MLPs:
define a few modules, then run a forward pass.
-/

@[expose] public section


namespace Examples

open Spec
open Tensor
open ModSpec
open Activation

/-- A 2-layer MLP as a `SpecChain`:

`Linear(inDim → hidDim)` then `ReLU` then `Linear(hidDim → outDim)`.

PyTorch analogy: `nn.Sequential(nn.Linear(inDim, hidDim), nn.ReLU(), nn.Linear(hidDim, outDim))`.
-/
def mlpSpec
  {α : Type} [Context α]
  {inDim hidDim outDim : Nat}
  (l1 : Spec.LinearSpec α inDim hidDim)
  (l2 : Spec.LinearSpec α hidDim outDim) :
  SpecChain α (.dim inDim .scalar) (.dim outDim .scalar) :=
  let linear1 := Spec.LinearModuleSpec (α:=α) l1
  let relu    := Spec.ReLUModuleSpec (α:=α) (.dim hidDim .scalar)
  let linear2 := Spec.LinearModuleSpec (α:=α) l2
  SpecChain.single linear1
    |>.composeRight relu
    |>.composeRight linear2

/-- MLP with a softmax head (`Linear → ReLU → Linear → Softmax`).

PyTorch analogy: `nn.Sequential(..., nn.Softmax(dim=-1))`.

Note: this is a *shape-safe* softmax spec (applied along the last dimension). In PyTorch you
choose `dim` at runtime; here the shape index already tells us what "the last dim" is.
-/
def mlpWithSoftmaxSpec
  {α : Type} [Context α]
  {inDim hidDim outDim : Nat}
  (l1 : Spec.LinearSpec α inDim hidDim)
  (l2 : Spec.LinearSpec α hidDim outDim) :
  SpecChain α (.dim inDim .scalar) (.dim outDim .scalar) :=
  let linear1 := Spec.LinearModuleSpec (α:=α) l1
  let relu    := Spec.ReLUModuleSpec (α:=α) (.dim hidDim .scalar)
  let linear2 := Spec.LinearModuleSpec (α:=α) l2
  let softmax := Spec.SoftmaxModuleSpec (α:=α) (.dim outDim .scalar)
  SpecChain.single linear1
    |>.composeRight relu
    |>.composeRight linear2
    |>.composeRight softmax

/-- Run the MLP forward on a single input vector. -/
def mlpForward
  {α : Type} [Context α]
  {inDim hidDim outDim : Nat}
  (l1 : Spec.LinearSpec α inDim hidDim)
  (l2 : Spec.LinearSpec α hidDim outDim)
  (x : Tensor α (.dim inDim .scalar)) :
  Tensor α (.dim outDim .scalar) :=
  let net := mlpSpec (α:=α) l1 l2
  SpecChain.forward (α:=α) net x

/-- Backward pass for the 2-layer MLP.
Returns (∂L/∂W1, ∂L/∂b1, ∂L/∂W2, ∂L/∂b2, ∂L/∂x).
-/
def mlpBackward
  {α : Type} [Context α]
  {inDim hidDim outDim : Nat}
  (l1 : Spec.LinearSpec α inDim hidDim)
  (l2 : Spec.LinearSpec α hidDim outDim)
  (x : Tensor α (.dim inDim .scalar))
  (dLdy : Tensor α (.dim outDim .scalar)) :
  ( Tensor α (.dim hidDim (.dim inDim .scalar))
  × Tensor α (.dim hidDim .scalar)
  × Tensor α (.dim outDim (.dim hidDim .scalar))
  × Tensor α (.dim outDim .scalar)
  × Tensor α (.dim inDim .scalar) ) :=

  -- Forward intermediates
  let z1 := Spec.linearSpec (α:=α) l1 x
  let a1 := Activation.reluSpec z1
  let _y := Spec.linearSpec (α:=α) l2 a1

  -- Layer 2 grads
  let dW2 := Spec.linearWeightsDerivSpec (α:=α) a1 dLdy
  let db2 := Spec.linearBiasDerivSpec (α:=α) dW2 dLdy a1
  let da1 := Spec.linearInputDerivSpec (α:=α) l2.weights dLdy

  -- ReLU backprop
  let grad_relu := Activation.reluDerivSpec z1
  let dz1 := mulSpec grad_relu da1

  -- Layer 1 grads
  let dW1 := Spec.linearWeightsDerivSpec (α:=α) x dz1
  let db1 := Spec.linearBiasDerivSpec (α:=α) dW1 dz1 x
  let dX  := Spec.linearInputDerivSpec (α:=α) l1.weights dz1

  (dW1, db1, dW2, db2, dX)

/-
Phase 1: Composition correctness (shape + functional) for the MLP SpecChain.
This lemma states that evaluating the composed chain equals the sequential computation.
-/
/-- Sanity lemma: the composed `SpecChain` forward equals the hand-written `Linear → ReLU → Linear`
computation. -/
theorem mlp_spec_forward_eq
  {α : Type} [Context α]
  {inDim hidDim outDim : Nat}
  (l1 : Spec.LinearSpec α inDim hidDim)
  (l2 : Spec.LinearSpec α hidDim outDim)
  (x : Tensor α (.dim inDim .scalar)) :
  SpecChain.forward (α:=α)
    (mlpSpec (α:=α) l1 l2) x
  =
  let z1 := Spec.linearSpec (α:=α) l1 x
  let a1 := Activation.reluSpec z1
  Spec.linearSpec (α:=α) l2 a1 := by
  dsimp [SpecChain.forward, mlpSpec, SpecChain.composeRight, NNModuleSpec.forward]
  dsimp [Spec.LinearModuleSpec, Spec.ReLUModuleSpec]


/-Phase 2: Symbolic gradient verification via OpSpec composition.
We build an OpSpec for the 2-layer MLP and expose its composed backward.-/

/-- `OpSpec` for the same 2-layer MLP.

This packaging is convenient for symbolic gradient checks: `OpSpec` pairs a forward definition with
an explicit reverse-mode definition, and it composes cleanly.
-/
def mlpOpspec
  {α : Type} [Context α]
  {inDim hidDim outDim : Nat}
  (l1 : Spec.LinearSpec α inDim hidDim)
  (l2 : Spec.LinearSpec α hidDim outDim) :
  Spec.OpSpec α (.dim inDim .scalar) (.dim outDim .scalar) :=
  let lin1 := Spec.linearOp (α:=α) l1
  let relu := Spec.reluOp (α:=α) (s:=.dim hidDim .scalar)
  let lin2 := Spec.linearOp (α:=α) l2
  Spec.OpSpec.compose (α:=α)
    (Spec.OpSpec.compose (α:=α) lin1 relu)
    lin2

/-- Composed backward of the MLP using the OpSpec chain. -/
def mlpOpspecBackward
  {α : Type} [Context α]
  {inDim hidDim outDim : Nat}
  (l1 : Spec.LinearSpec α inDim hidDim)
  (l2 : Spec.LinearSpec α hidDim outDim)
  (x : Tensor α (.dim inDim .scalar))
  (dLdy : Tensor α (.dim outDim .scalar)) :
  Tensor α (.dim inDim .scalar) :=
  let op := mlpOpspec (α:=α) l1 l2
  op.backward x dLdy

end Examples
