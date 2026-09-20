/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Seeded -- shake: keep

/-!
# Model Summaries

Structured model-summary rendering for checked sequential models.
-/

@[expose] public section

namespace TorchLean

namespace nn

/-- Number of scalar elements represented by a list of tensor shapes. -/
private def elementCount (shapes : List Shape) : Nat :=
  shapes.foldl (fun acc s => acc + Shape.size s) 0

/-- Number of trainable scalar elements in a model-state shape list. -/
private def trainableCount (shapes : List Shape) (flags : Array Bool) : Nat :=
  let rec go : List Shape → Nat → Nat
    | .nil, _ => 0
    | .cons shape rest, index =>
        (if flags[index]?.getD false then Shape.size shape else 0) + go rest (index + 1)
  go shapes 0

/-- User-facing tensor shape display for one model-summary shape. -/
private def shapeDisplay (shape : Shape) : String :=
  match Shape.toList shape with
  | .nil => "scalar"
  | .cons dim dims =>
      "[" ++ String.intercalate ", " ((dim :: dims).map toString) ++ "]"

/-- User-facing display for an array of tensor shapes. -/
private def shapeArrayString (shapes : Array Shape) : String :=
  if shapes.isEmpty then
    "[]"
  else
    "[" ++ String.intercalate ", " (shapes.map shapeDisplay).toList ++ "]"

/-- Structured per-layer summary derived from a checked sequential model. -/
structure LayerSummary where
  /-- Zero-based position in the sequential layer list. -/
  index : Nat
  /-- User-facing layer kind string. -/
  kind : String
  /-- Checked input shape for this layer. -/
  input : Shape
  /-- Checked output shape for this layer. -/
  output : Shape
  /-- Trainable parameter and persistent-buffer shapes owned by this layer. -/
  stateShapes : Array Shape
  /-- Number of trainable scalar parameters in this layer. -/
  parameterCount : Nat
  /-- Number of scalar elements in this layer's complete state. -/
  stateElementCount : Nat

namespace LayerSummary

/-- One-line rendering of one layer summary. -/
@[no_expose] def render (summary : LayerSummary) : String :=
  s!"  [{summary.index}] {summary.kind}: {shapeDisplay summary.input} -> " ++
    s!"{shapeDisplay summary.output} params={summary.parameterCount}, " ++
    s!"state={summary.stateElementCount} " ++
    shapeArrayString summary.stateShapes

instance : ToString LayerSummary where
  toString := render

end LayerSummary

/-- Structured whole-model summary derived from a checked sequential model. -/
structure ModelSummary where
  /-- Checked input shape for the full model. -/
  input : Shape
  /-- Checked output shape for the full model. -/
  output : Shape
  /-- Per-layer summaries in order. -/
  layers : Array LayerSummary
  /-- Total number of layers in the sequential model. -/
  layerCount : Nat
  /-- Total scalar parameter count across all layers. -/
  totalParameterCount : Nat
  /-- Total scalar element count across trainable parameters and persistent buffers. -/
  totalStateElementCount : Nat

namespace ModelSummary

/-- Header line for the model summary. -/
@[no_expose] def header (summary : ModelSummary) : String :=
  s!"Sequential: {shapeDisplay summary.input} -> " ++
    s!"{shapeDisplay summary.output}, " ++
    s!"layers={summary.layerCount}, params={summary.totalParameterCount}, " ++
    s!"state={summary.totalStateElementCount}"

/-- Multi-line rendering of the structured model summary. -/
def render (summary : ModelSummary) : String :=
  String.intercalate "\n"
    (#[header summary] ++ summary.layers.map LayerSummary.render).toList

instance : ToString ModelSummary where
  toString := render

end ModelSummary

/-- Recursive worker that records one summary row for each layer in a sequential model. -/
private def layerSummaries :
    {σ τ : Shape} → Nat → Sequential σ τ → Array LayerSummary
  | _, _, _, .id _ => #[]
  | σ, _, i, .cons (τ := τ') layer rest =>
      let row : LayerSummary :=
        { index := i
          kind := layer.kind
          input := σ
          output := τ'
          stateShapes := layer.stateShapes.toArray
          parameterCount := trainableCount layer.stateShapes layer.requiresGrad
          stateElementCount := elementCount layer.stateShapes }
      #[row] ++
      layerSummaries (i + 1) rest

/-- Validate a sequential model and construct its structured summary. -/
@[no_expose] def summary {σ τ : Shape} (model : Sequential σ τ) : Except String ModelSummary := do
  validate model
  let layers := layerSummaries 0 model
  pure
    { input := σ
      output := τ
      layers := layers
      layerCount := layers.size
      totalParameterCount := trainableCount (stateShapes model) (requiresGrad model)
      totalStateElementCount := elementCount (stateShapes model) }

/--
Print the structured summary of a checked sequential model.

Example:
```lean
def model : nn.Sequential [2] [1] :=
  nn.build 0 nn.Sequential![nn.linear 2 8, nn.relu, nn.linear 8 1]

-- Prints one row per layer with its kind, shapes, and parameter count, then the totals. The
-- counterpart of `print(model)` plus `torchinfo.summary`.
def main : IO Unit := nn.printSummary model
```
-/
def printSummary {σ τ : Shape} (model : Sequential σ τ) : IO Unit := do
  match summary model with
  | .ok details => IO.println details
  | .error message => throw <| IO.userError message

end nn

end TorchLean
