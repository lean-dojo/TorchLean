/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Builder.Layer

/-!
# Models over Leading Dimensions

`mapLeading` lifts a sequential model over any number of leading tensor dimensions and applies it
separately at every leading index. A batch is the common case `leading = [batch]`, and shapes such
as `[batch, time]` use the same machinery.

The layer constructors take a second route, `Layer.adaptLeadingShape`, which flattens the
leading dimensions into the one outer dimension a layer already accepts. The two routes differ for
stateful layers, whose buffer updates may depend on whether the leading positions are processed
together or one at a time.
-/

@[expose] public section

namespace TorchLean
namespace nn

/-- Apply `layer` separately at every position of one extra leading dimension. -/
private def mapLayerOverAxis (n : Nat) {σ τ : Shape} (layer : Layer σ τ) :
    Layer (Shape.dim n σ) (Shape.dim n τ) :=
  { kind := layer.kind
    stateShapes := layer.stateShapes
    initState := layer.initState
    runtimeInit := layer.runtimeInit
    requiresGrad := layer.requiresGrad
    validateConfig := layer.validateConfig
    updatesBuffersInForward := layer.updatesBuffersInForward
    updateBuffers := layer.updateBuffers.map fun update mode {_α} _ _ state input =>
      (List.finRange n).foldlM (init := state) fun nextState index =>
        update mode nextState (TorchLean.Tensor.unstack input index)
    forward := fun mode {α} _ _ =>
      fun {m} _ _ =>
        CurriedRef.curry
          (Ref := fun shape => Ops.Ref (m := m) (α := α) shape)
          (ss := layer.stateShapes ++ [Shape.dim n σ])
          (β := m (Ops.Ref (m := m) (α := α)
            (Shape.dim n τ)))
          (fun arguments => do
            let (state, inputBatch) :=
              TorchLean.Pack.unsnoc
                (F := fun shape =>
                  Ops.Ref (m := m) (α := α) shape)
                (ss := layer.stateShapes) (τ := Shape.dim n σ) arguments
            Ops.mapBatch (m := m) (α := α)
              (fun input => layer.forwardRef (α := α) (m := m) mode state input)
              inputBatch) }

/-- Lift every layer of `model` over one extra leading dimension. -/
private def mapModelOverAxis (n : Nat) {σ τ : Shape} :
    Sequential σ τ → Sequential (Shape.dim n σ) (Shape.dim n τ)
  | .id shape => .id (Shape.dim n shape)
  | .cons layer rest => .cons (mapLayerOverAxis n layer) (mapModelOverAxis n rest)

/--
Apply a sequential model separately at every index of `leading`.

All positions use the same model parameters. Buffer updates are evaluated in lexicographic order
over the leading indices.

Example:
```lean
def perSample : nn.Builder (nn.Sequential [2] [1]) :=
  nn.Sequential![nn.linear 2 8, nn.op .relu, nn.linear 8 1]

-- One model, applied at each of five positions of a new leading axis, sharing its parameters.
def model : nn.Builder (nn.Sequential [5, 2] [5, 1]) := do
  pure (nn.mapLeading [5] (← perSample))
```
-/
@[no_expose] def mapLeading (leading : Shape) {σ τ : Shape} :
    Sequential σ τ → Sequential (leading ++ σ) (leading ++ τ) :=
  fun model =>
    match leading with
    | .scalar => model
    | .dim n rest => mapModelOverAxis n (mapLeading rest model)

/-- Mapping over no leading axes leaves the model unchanged. -/
@[simp] theorem mapLeading_scalar {σ τ : Shape} (model : Sequential σ τ) :
    mapLeading .scalar model = model := by
  rfl

end nn
end TorchLean
