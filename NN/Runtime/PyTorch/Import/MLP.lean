/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.PyTorch.Import.Core
public import NN.Tensor

/-!
# MLP PyTorch Reference Import

MLP reference weight import from a PyTorch-style `state_dict`.

On the Python side we usually write JSON (nested lists of floats) under keys that mirror the names
you would see in `model.state_dict()`:

- `fc1.weight`, `fc1.bias`, `fc2.weight`, `fc2.bias` for a hand-written `nn.Module` with `fc1/fc2`,
- or `layers.0.weight`, `layers.0.bias`, ... if the model was built from an `nn.Sequential`.

This file keeps the parsing logic in one place so the rest of the codebase can talk in terms of
typed Lean tensors.
-/

@[expose] public section

open Import.PyTorch
open Spec
open TorchLean
open Lean

namespace Import.PyTorch.MLP

-- We support two key conventions:
-- - PyTorch `nn.Linear` style: `fc1.weight`, `fc1.bias`, `fc2.weight`, `fc2.bias`
-- - sequential parameter style: `layers.0.weight`, `layers.0.bias`, `layers.2.weight`,
--   `layers.2.bias`
/-- Parameters for a two-layer MLP imported from a PyTorch `state_dict`.

We keep the tensors as `Float` because these importers are meant for runtime examples: train in
Python, export to JSON, then run/verify in TorchLean.
-/
structure Parameters (inputWidth hiddenWidth outputWidth : Nat) where
  /-- First linear layer weight, PyTorch shape `(hidden, input)`. -/
  inputWeight : Tensor Float [hiddenWidth, inputWidth]
  /-- First linear layer bias. -/
  inputBias : Tensor Float [hiddenWidth]
  /-- Second linear layer weight, PyTorch shape `(output, hidden)`. -/
  outputWeight : Tensor Float [outputWidth, hiddenWidth]
  /-- Second linear layer bias. -/
  outputBias : Tensor Float [outputWidth]

/-- Load MLP parameters from JSON using either supported PyTorch key convention. -/
def load (inputWidth hiddenWidth outputWidth : Nat) (json : Json) :
    Option (Parameters inputWidth hiddenWidth outputWidth) :=
  let inputWeightShape : Shape := [hiddenWidth, inputWidth]
  let inputBiasShape : Shape := [hiddenWidth]
  let outputWeightShape : Shape := [outputWidth, hiddenWidth]
  let outputBiasShape : Shape := [outputWidth]
  do
    -- `loadWeights?` accepts either:
    -- - `{ ...state_dict... }`, or
    -- - `{ "params": { ...state_dict... } }` (a common wrapper in our Python scripts).
    let weights ← loadWeights? json
    let tryKeys (inputWeightKey inputBiasKey outputWeightKey outputBiasKey : String) :
        Option (Parameters inputWidth hiddenWidth outputWidth) := do
      let inputWeight ← getTensor? weights inputWeightKey inputWeightShape
      let inputBias ← getTensor? weights inputBiasKey inputBiasShape
      let outputWeight ← getTensor? weights outputWeightKey outputWeightShape
      let outputBias ← getTensor? weights outputBiasKey outputBiasShape
      pure { inputWeight, inputBias, outputWeight, outputBias }
    tryKeys "fc1.weight" "fc1.bias" "fc2.weight" "fc2.bias" <|>
      tryKeys "layers.0.weight" "layers.0.bias" "layers.2.weight" "layers.2.bias"

/-- Run the imported two-layer MLP. -/
def forward {inputWidth hiddenWidth outputWidth : Nat}
    (parameters : Parameters inputWidth hiddenWidth outputWidth)
    (input : Tensor Float [inputWidth]) : Tensor Float [outputWidth] :=
  (input.linear parameters.inputWeight parameters.inputBias).relu.linear
    parameters.outputWeight parameters.outputBias

end Import.PyTorch.MLP
