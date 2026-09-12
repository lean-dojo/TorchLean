/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.PyTorch.Import.Core
public import NN.Tensor

/-!
# Transformer PyTorch Import

Transformer weight import from JSON.

TorchLean's executable Transformer keeps query/key/value/output projections, feed-forward weights,
and LayerNorm affine parameters in one shape-indexed state pack. In PyTorch these values are usually
spread across several `nn.Linear` and `nn.LayerNorm` submodules.

For round-trip examples we accept a *stable, explicit key format* in JSON:
`Wq`, `Wk`, `Wv`, `Wo`, `W1`, `W2`, `b1`, `b2`, `norm1_gamma`, `norm1_beta`, `norm2_gamma`,
  `norm2_beta`.

We also accept the nested PyTorch module keys emitted by
`Export.PyTorch.Transformer.withParameters`, such as
`layers.0.mha.q_proj.weight`. Explicit attention keys use `(input, output)` matrices. Nested
`nn.Linear` projection keys use `(output, input)` and are transposed during import; feed-forward
weights retain PyTorch's orientation under either naming scheme.
-/

@[expose] public section

open Import.PyTorch
open Spec
open TorchLean TorchLean.Tensor
open Lean

namespace Import.PyTorch.Transformer

/-- Parameters for a single-layer Transformer encoder imported from a PyTorch `state_dict`.

This is the normalized typed view returned by the JSON loader. The loader accepts both TorchLean's
explicit keys and the nested PyTorch module keys emitted by the exporter.
-/
structure Parameters (modelWidth feedForwardWidth : Nat) where
  /-- Query projection matrix in `(input, output)` orientation. -/
  queryWeight : Tensor Float [modelWidth, modelWidth]
  /-- Key projection matrix. -/
  keyWeight : Tensor Float [modelWidth, modelWidth]
  /-- Value projection matrix. -/
  valueWeight : Tensor Float [modelWidth, modelWidth]
  /-- Output projection matrix. -/
  outputWeight : Tensor Float [modelWidth, modelWidth]
  /-- Input feed-forward weight in PyTorch `(output, input)` layout. -/
  feedForwardInputWeight : Tensor Float [feedForwardWidth, modelWidth]
  /-- Output feed-forward weight in PyTorch `(output, input)` layout. -/
  feedForwardOutputWeight : Tensor Float [modelWidth, feedForwardWidth]
  /-- Input feed-forward projection bias. -/
  feedForwardInputBias : Tensor Float [feedForwardWidth]
  /-- Output feed-forward projection bias. -/
  feedForwardOutputBias : Tensor Float [modelWidth]
  /-- First LayerNorm scale. -/
  norm1Scale : Tensor Float [modelWidth]
  /-- First LayerNorm bias. -/
  norm1Bias : Tensor Float [modelWidth]
  /-- Second LayerNorm scale. -/
  norm2Scale : Tensor Float [modelWidth]
  /-- Second LayerNorm bias. -/
  norm2Bias : Tensor Float [modelWidth]

/-- Load Transformer parameters from JSON matching either supported export key format. -/
def load (modelWidth feedForwardWidth : Nat) (json : Json) :
    Option (Parameters modelWidth feedForwardWidth) :=
  let projectionShape : Shape := [modelWidth, modelWidth]
  let feedForwardInputWeightShape : Shape := [feedForwardWidth, modelWidth]
  let feedForwardOutputWeightShape : Shape := [modelWidth, feedForwardWidth]
  let feedForwardInputBiasShape : Shape := [feedForwardWidth]
  let feedForwardOutputBiasShape : Shape := [modelWidth]
  let normShape : Shape := [modelWidth]
  do
    -- Accepts both `{...}` and `{ "params": {...} }`.
    let weights ← loadWeights? json
    -- Explicit keys store input×output matrices; nn.Linear state_dict keys store output×input.
    let projection (explicitKey moduleKey : String) :
        Option (Tensor Float [modelWidth, modelWidth]) :=
      getTensor? weights explicitKey projectionShape <|> do
        let matrix ← getTensor? weights moduleKey projectionShape
        pure <| rearrange matrix "output input -> input output"
    let queryWeight ← projection "Wq" "layers.0.mha.q_proj.weight"
    let keyWeight ← projection "Wk" "layers.0.mha.k_proj.weight"
    let valueWeight ← projection "Wv" "layers.0.mha.v_proj.weight"
    let outputWeight ← projection "Wo" "layers.0.mha.out_proj.weight"
    let feedForwardInputWeight ← getTensorFirst? weights
      ["W1", "layers.0.ffn.fc1.weight"] feedForwardInputWeightShape
    let feedForwardOutputWeight ← getTensorFirst? weights
      ["W2", "layers.0.ffn.fc2.weight"] feedForwardOutputWeightShape
    let feedForwardInputBias ← getTensorFirst? weights
      ["b1", "layers.0.ffn.fc1.bias"] feedForwardInputBiasShape
    let feedForwardOutputBias ← getTensorFirst? weights
      ["b2", "layers.0.ffn.fc2.bias"] feedForwardOutputBiasShape
    let norm1Scale ←
      getTensorFirst? weights ["norm1_gamma", "layers.0.norm1.weight"] normShape
    let norm1Bias ←
      getTensorFirst? weights ["norm1_beta", "layers.0.norm1.bias"] normShape
    let norm2Scale ←
      getTensorFirst? weights ["norm2_gamma", "layers.0.norm2.weight"] normShape
    let norm2Bias ←
      getTensorFirst? weights ["norm2_beta", "layers.0.norm2.bias"] normShape
    pure {
      queryWeight, keyWeight, valueWeight, outputWeight
      feedForwardInputWeight, feedForwardOutputWeight
      feedForwardInputBias, feedForwardOutputBias
      norm1Scale, norm1Bias, norm2Scale, norm2Bias
    }

end Import.PyTorch.Transformer
