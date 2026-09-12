/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Neural.Builders
public import NN.API.Runtime -- shake: keep

/-!
# Models over Leading Dimensions

This module lifts sequential models over any number of leading tensor dimensions. A batch is the
common case `leading = [batch]`; shapes such as `[batch, time]` use the same machinery.

`mapLeading` applies a model separately at every leading index. The implementation module behind
the layer constructors also flattens leading dimensions for layers that already accept one outer
dimension; keeping that distinction explicit matters for stateful layers, whose buffer updates may
depend on whether the leading positions are processed together or one at a time.
-/

@[expose] public section

namespace TorchLean
namespace nn

/-- Apply `layer` separately at every position of one new leading dimension. -/
private def mapLayerOverAxis (n : Nat) {σ τ : Spec.Shape} (layer : Layer σ τ) :
    Layer (σ.prependDim n) (τ.prependDim n) :=
  { kind := layer.kind
    stateShapes := layer.stateShapes
    initState := layer.initState
    runtimeInit := layer.runtimeInit
    requiresGrad := layer.requiresGrad
    validateConfig := layer.validateConfig
    updateBuffers := layer.updateBuffers.map fun update mode {_α} _ _ state input =>
      (List.finRange n).foldlM (init := state) fun nextState index =>
        update mode nextState (TorchLean.Tensor.unstack input index)
    forward := fun mode {α} _ _ =>
      fun {m} _ _ =>
        Runtime.Autograd.Torch.CurriedRef.curry
          (Ref := fun shape => TorchLean.Runtime.ValueRef (m := m) (α := α) shape)
          (ss := layer.stateShapes ++ [σ.prependDim n])
          (β := m (TorchLean.Runtime.ValueRef (m := m) (α := α)
            (τ.prependDim n)))
          (fun arguments => do
            let (state, inputBatch) :=
              Runtime.Autograd.Torch.RefList.splitLast
                (Ref := fun shape =>
                  TorchLean.Runtime.ValueRef (m := m) (α := α) shape)
                (ss := layer.stateShapes) (τ := σ.prependDim n) arguments
            Runtime.Autograd.Torch.mapOuterAxis (m := m) (α := α)
              (fun input => layer.forwardRef (α := α) (m := m) mode state input)
              inputBatch) }

/-- Apply every layer of `model` over one new leading dimension. -/
private def mapModelOverAxis (n : Nat) {σ τ : Spec.Shape} :
    Sequential σ τ → Sequential (σ.prependDim n) (τ.prependDim n)
  | .id shape => .id (shape.prependDim n)
  | .cons layer rest => .cons (mapLayerOverAxis n layer) (mapModelOverAxis n rest)

/--
Apply a sequential model separately at every index of `leading`.

All positions use the same model parameters. Buffer updates are evaluated in lexicographic order
over the leading indices.

Example:
```lean
def perSample : nn.Builder (nn.Sequential [2] [1]) :=
  nn.Sequential![nn.linear 2 8, nn.relu, nn.linear 8 1]

-- One model, applied at each of five positions of a new leading axis, sharing its parameters.
def model : nn.Builder (nn.Sequential [5, 2] [5, 1]) := do
  pure (nn.mapLeading [5] (← perSample))
```
-/
opaque mapLeading (leading : Spec.Shape) {σ τ : Spec.Shape} :
    Sequential σ τ → Sequential (leading.concat σ) (leading.concat τ) :=
  fun model =>
    match leading with
    | .scalar => model
    | .dim n rest => mapModelOverAxis n (mapLeading rest model)

end nn
end TorchLean
