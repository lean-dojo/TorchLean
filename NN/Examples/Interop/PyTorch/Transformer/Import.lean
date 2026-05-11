/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.PyTorch.Import.Core
public import NN.Spec.Models.Transformer

/-!
# Transformer PyTorch Fixture Import

Transformer fixture weight import from JSON.

In the spec layer, our transformer encoder parameters are explicit tensors (query/key/value/output
projections, feed-forward weights, and LayerNorm affine parameters). In PyTorch these are usually
spread across multiple `nn.Linear` and `nn.LayerNorm` submodules.

For round-trip demos we accept a *stable, explicit key format* in JSON:
`Wq`, `Wk`, `Wv`, `Wo`, `W1`, `W2`, `b1`, `b2`, `norm1_gamma`, `norm1_beta`, `norm2_gamma`,
  `norm2_beta`.

We also accept the nested PyTorch module keys emitted by
`Export.TransformerPyTorch.generateTransformerEncoderWithWeights`, such as
`layers.0.mha.q_proj.weight`. That keeps generated export state dicts loadable by both PyTorch and
this Lean importer.
-/

@[expose] public section


namespace Import
namespace TransformerPyTorch
open PyTorch

open Spec
open Tensor
open Shape
open Lean
open Data
open Json

-- Transformer Encoder state dict structure (simplified, for one layer)
/-- Typed view of a single-layer Transformer encoder `state_dict` (Float tensors).

This is the normalized typed view returned by the JSON loader. The loader accepts both TorchLean's
explicit keys and the nested PyTorch module keys emitted by the exporter.
-/
structure TransformerEncoderStateDict (embedDim headCount hiddenDim : Nat) where
  /-- Query projection matrix. -/
  Wq : Tensor Float (.dim embedDim (.dim embedDim .scalar))
  /-- Key projection matrix. -/
  Wk : Tensor Float (.dim embedDim (.dim embedDim .scalar))
  /-- Value projection matrix. -/
  Wv : Tensor Float (.dim embedDim (.dim embedDim .scalar))
  /-- Output projection matrix. -/
  Wo : Tensor Float (.dim embedDim (.dim embedDim .scalar))
  /-- Weight matrix for layer 1. -/
  W1 : Tensor Float (.dim embedDim (.dim hiddenDim .scalar))
  /-- Weight matrix for layer 2. -/
  W2 : Tensor Float (.dim hiddenDim (.dim embedDim .scalar))
  /-- Bias for layer 1. -/
  b1 : Tensor Float (.dim hiddenDim .scalar)
  /-- Bias for layer 2. -/
  b2 : Tensor Float (.dim embedDim .scalar)
  /-- First LayerNorm scale. -/
  norm1_gamma : Tensor Float (.dim embedDim .scalar)
  /-- First LayerNorm bias. -/
  norm1_beta  : Tensor Float (.dim embedDim .scalar)
  /-- Second LayerNorm scale. -/
  norm2_gamma : Tensor Float (.dim embedDim .scalar)
  /-- Second LayerNorm bias. -/
  norm2_beta  : Tensor Float (.dim embedDim .scalar)

def getTensorAny? (o : StateDict) (s : Shape) (keys : List String) :
    Option (Tensor Float s) :=
  match keys with
  | [] => none
  | k :: ks =>
      match getTensor? o k s with
      | some t => some t
      | none => getTensorAny? o s ks

/-- Load Transformer Encoder state dict from JSON matching either supported export key format. -/
def loadTransformerEncoderStateDict (embedDim headCount hiddenDim : Nat) (j : Json) : Option
  (TransformerEncoderStateDict embedDim headCount hiddenDim) :=
  let _ := headCount
  let WqShape : Shape := .dim embedDim (.dim embedDim .scalar)
  let WkShape : Shape := .dim embedDim (.dim embedDim .scalar)
  let WvShape : Shape := .dim embedDim (.dim embedDim .scalar)
  let WoShape : Shape := .dim embedDim (.dim embedDim .scalar)
  let W1Shape : Shape := .dim embedDim (.dim hiddenDim .scalar)
  let W2Shape : Shape := .dim hiddenDim (.dim embedDim .scalar)
  let b1Shape : Shape := .dim hiddenDim .scalar
  let b2Shape : Shape := .dim embedDim .scalar
  let normShape : Shape := .dim embedDim .scalar
  do
    -- Accepts both `{...}` and `{ "params": {...} }`.
    let o ← loadWeights? j
    let Wq ← getTensorAny? o WqShape ["Wq", "layers.0.mha.q_proj.weight"]
    let Wk ← getTensorAny? o WkShape ["Wk", "layers.0.mha.k_proj.weight"]
    let Wv ← getTensorAny? o WvShape ["Wv", "layers.0.mha.v_proj.weight"]
    let Wo ← getTensorAny? o WoShape ["Wo", "layers.0.mha.out_proj.weight"]
    let W1 ← getTensorAny? o W1Shape ["W1", "layers.0.ffn.fc1.weight"]
    let W2 ← getTensorAny? o W2Shape ["W2", "layers.0.ffn.fc2.weight"]
    let b1 ← getTensorAny? o b1Shape ["b1", "layers.0.ffn.fc1.bias"]
    let b2 ← getTensorAny? o b2Shape ["b2", "layers.0.ffn.fc2.bias"]
    let norm1_gamma ← getTensorAny? o normShape ["norm1_gamma", "layers.0.norm1.weight"]
    let norm1_beta ← getTensorAny? o normShape ["norm1_beta", "layers.0.norm1.bias"]
    let norm2_gamma ← getTensorAny? o normShape ["norm2_gamma", "layers.0.norm2.weight"]
    let norm2_beta ← getTensorAny? o normShape ["norm2_beta", "layers.0.norm2.bias"]
    pure {
      Wq := Wq, Wk := Wk, Wv := Wv, Wo := Wo
      W1 := W1, W2 := W2, b1 := b1, b2 := b2
      norm1_gamma := norm1_gamma, norm1_beta  := norm1_beta
      norm2_gamma := norm2_gamma, norm2_beta  := norm2_beta
    }

end TransformerPyTorch
end Import
