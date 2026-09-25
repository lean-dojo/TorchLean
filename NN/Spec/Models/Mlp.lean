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

This file defines a 2-layer MLP by composing `Spec.Module.Chain`s from module specs:

`Linear → ReLU → Linear` (optionally followed by a softmax head).

The file is organized around module wiring rather than re-implementing matrix multiplications
directly. `Linear` and `ReLU` come from the spec layer and are composed through `Spec.Module` /
`Spec.Module.Chain`, matching the usual PyTorch workflow: define a few modules, then run a forward
pass.

## Implementation status

There is no dedicated `nn.models` builder; the API composes `nn.linear` and `nn.relu` directly.
The relating theorem is `mlp_interp_eq_spec_mlp_forward` in
`NN/GraphSpec/Models/MlpSpecEquivalence.lean`, which shows the typed-graph interpretation of the
MLP equals this spec's forward pass; `NN/Proofs/Models/Mlp.lean` and
`NN/Tests/Runtime/Floats/TorchLeanSpecMlpEquivCheck.lean` exercise it further.
-/

@[expose] public section


namespace Examples

open Spec TorchLean
open TorchLean TorchLean.Tensor
open Spec.Module
open Activation

/-- A 2-layer MLP as a `Spec.Module.Chain`:

`Linear(inDim → hidDim)` then `ReLU` then `Linear(hidDim → outDim)`.

PyTorch analogy: `nn.Sequential(nn.Linear(inDim, hidDim), nn.ReLU(), nn.Linear(hidDim, outDim))`.
-/
def mlpSpec
  {α : Type} [TorchLean.Storage α] [Context α]
  {inDim hidDim outDim : Nat}
  (l1 : Spec.LinearSpec α inDim hidDim)
  (l2 : Spec.LinearSpec α hidDim outDim) :
  Spec.Module.Chain α ([inDim]) ([outDim]) :=
  let linear1 := Spec.Module.linear (α:=α) l1
  let relu    := Spec.Module.relu (α:=α) ([hidDim])
  let linear2 := Spec.Module.linear (α:=α) l2
  Spec.Module.Chain.single linear1
    |>.append relu
    |>.append linear2

/-- Run the MLP forward on a single input vector. -/
def mlpForward
  {α : Type} [TorchLean.Storage α] [Context α]
  {inDim hidDim outDim : Nat}
  (l1 : Spec.LinearSpec α inDim hidDim)
  (l2 : Spec.LinearSpec α hidDim outDim)
  (x : Tensor α [inDim]) :
  Tensor α [outDim] :=
  let net := mlpSpec (α:=α) l1 l2
  Spec.Module.Chain.forward (α:=α) net x

/-- Backward pass for the 2-layer MLP.
Returns (∂L/∂W1, ∂L/∂b1, ∂L/∂W2, ∂L/∂b2, ∂L/∂x).
-/
def mlpBackward
  {α : Type} [TorchLean.Storage α] [Context α]
  {inDim hidDim outDim : Nat}
  (l1 : Spec.LinearSpec α inDim hidDim)
  (l2 : Spec.LinearSpec α hidDim outDim)
  (x : Tensor α [inDim])
  (dLdy : Tensor α [outDim]) :
  ( Tensor α [hidDim, inDim]
  × Tensor α [hidDim]
  × Tensor α [outDim, hidDim]
  × Tensor α [outDim]
  × Tensor α [inDim] ) :=

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
Phase 1: Composition correctness (shape + functional) for the MLP Spec.Module.Chain.
This lemma states that evaluating the composed chain equals the sequential computation.
-/
/-- The composed `Spec.Module.Chain` forward equals the hand-written `Linear → ReLU → Linear`
computation. -/
theorem mlp_spec_forward_eq
  {α : Type} [TorchLean.Storage α] [Context α]
  {inDim hidDim outDim : Nat}
  (l1 : Spec.LinearSpec α inDim hidDim)
  (l2 : Spec.LinearSpec α hidDim outDim)
  (x : Tensor α [inDim]) :
  Spec.Module.Chain.forward (α:=α)
    (mlpSpec (α:=α) l1 l2) x
  =
  let z1 := Spec.linearSpec (α:=α) l1 x
  let a1 := Activation.reluSpec z1
  Spec.linearSpec (α:=α) l2 a1 := by
  dsimp [Spec.Module.Chain.forward, mlpSpec, Spec.Module.Chain.append, Spec.Module.forward]
  dsimp [Spec.Module.linear, Spec.Module.relu]


/-Phase 2: Symbolic gradient verification via OpSpec composition.
We build an OpSpec for the 2-layer MLP and expose its composed backward.-/

/-- `OpSpec` for the same 2-layer MLP.

This packaging is convenient for symbolic gradient checks: `OpSpec` pairs a forward definition with
an explicit reverse-mode definition, and it composes cleanly.
-/
def mlpOpspec
  {α : Type} [TorchLean.Storage α] [Context α]
  {inDim hidDim outDim : Nat}
  (l1 : Spec.LinearSpec α inDim hidDim)
  (l2 : Spec.LinearSpec α hidDim outDim) :
  Spec.OpSpec α ([inDim]) ([outDim]) :=
  let lin1 := Spec.linearOp (α:=α) l1
  let relu := Spec.reluOp (α:=α) (s:=[hidDim])
  let lin2 := Spec.linearOp (α:=α) l2
  Spec.OpSpec.compose (α:=α)
    (Spec.OpSpec.compose (α:=α) lin1 relu)
    lin2

/-- Composed backward of the MLP using the OpSpec chain. -/
def mlpOpspecBackward
  {α : Type} [TorchLean.Storage α] [Context α]
  {inDim hidDim outDim : Nat}
  (l1 : Spec.LinearSpec α inDim hidDim)
  (l2 : Spec.LinearSpec α hidDim outDim)
  (x : Tensor α [inDim])
  (dLdy : Tensor α [outDim]) :
  Tensor α [inDim] :=
  let op := mlpOpspec (α:=α) l1 l2
  op.backward x dLdy

end Examples
