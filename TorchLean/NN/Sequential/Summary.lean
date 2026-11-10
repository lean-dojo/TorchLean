/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Builder -- shake: keep

/-!
# Model Summaries

`nn.summary` validates a sequential model and returns a `ModelSummary`: the model's input and
output shapes, its totals, and one `LayerSummary` per layer with the layer's kind, shapes and
parameter count. `nn.printSummary` prints that summary.
-/

@[expose] public section

open TorchLean.Shape (size)

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

/-- Display a shape as a bracketed list of its dimensions, or as `scalar` when it has none. -/
private def shapeDisplay (shape : Shape) : String :=
  match shape with
  | .nil => "scalar"
  | .cons dim dims =>
      "[" ++ String.intercalate ", " ((dim :: dims).map toString) ++ "]"

/-- Display an array of shapes as a bracketed list of `shapeDisplay` strings. -/
private def shapeArrayString (shapes : Array Shape) : String :=
  "[" ++ String.intercalate ", " (shapes.map shapeDisplay).toList ++ "]"

/-- Summary of one layer of a sequential model: its position, kind, shapes and state sizes. -/
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

/-- Summary of a whole sequential model: its shapes, its layers in order and their totals. -/
structure ModelSummary where
  /-- Checked input shape for the full model. -/
  input : Shape
  /-- Checked output shape for the full model. -/
  output : Shape
  /-- Per-layer summaries in order. -/
  layers : Array LayerSummary
  /-- Total number of layers in the sequential model. -/
  layerCount : Nat
  /-- Number of trainable scalars across all layers. -/
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

/-- Render the summary as its header line followed by one line per layer. -/
def render (summary : ModelSummary) : String :=
  String.intercalate "\n"
    (#[header summary] ++ summary.layers.map LayerSummary.render).toList

instance : ToString ModelSummary where
  toString := render

end ModelSummary

/-- Append one `LayerSummary` per layer of a sequential model, numbering from the given index. -/
private def layerSummaries :
    {σ τ : Shape} → Nat → Sequential σ τ → Array LayerSummary → Array LayerSummary
  | _, _, _, .id _, rows => rows
  | σ, _, i, .cons (τ := τ') layer rest, rows =>
      let row : LayerSummary :=
        { index := i
          kind := layer.kind
          input := σ
          output := τ'
          stateShapes := layer.stateShapes.toArray
          parameterCount := trainableCount layer.stateShapes layer.requiresGrad
          stateElementCount := elementCount layer.stateShapes }
      layerSummaries (i + 1) rest (rows.push row)

/-- Validate a sequential model and construct its structured summary. -/
@[no_expose] def summary {σ τ : Shape} (model : Sequential σ τ) : Except String ModelSummary := do
  Sequential.validate model
  let layers := layerSummaries 0 model #[]
  pure
    { input := σ
      output := τ
      layers := layers
      layerCount := layers.size
      totalParameterCount := layers.foldl (fun total layer => total + layer.parameterCount) 0
      totalStateElementCount :=
        layers.foldl (fun total layer => total + layer.stateElementCount) 0 }

/--
Validate a sequential model and print its summary, failing in `IO` when the model is invalid.

Example:
```lean
def model : nn.Sequential [2] [1] :=
  nn.build 0 nn.Sequential![nn.linear 2 8, nn.op .relu, nn.linear 8 1]

-- Prints the totals, then one row per layer with its kind, shapes, and parameter count. The
-- counterpart of `print(model)` plus `torchinfo.summary`.
def main : IO Unit := nn.printSummary model
```
-/
def printSummary {σ τ : Shape} (model : Sequential σ τ) : IO Unit := do
  IO.println (← IO.ofExcept (summary model))

end nn

end TorchLean
