/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.PyTorch.Import
public import TorchLeanExamples.Verification.PINN.Architecture

/-!
# PINN PyTorch Checkpoint Loading

PINN weight import (JSON → typed tensors).

The loading half of the PINN PyTorch bridge works in three steps:

- read a PyTorch-style state dictionary encoded as JSON,
- parse each weight matrix and bias vector into a `Tensor Float ...` whose shape is inferred and
  checked,
- infer a TorchLean `Architecture` from the layer shapes plus optional activation metadata.

The generic JSON helpers (`loadWeights?`, `linearLayers`, …) live in
`TorchLean/NN/PyTorch/Import.lean`.
-/

@[expose] public section


namespace TorchLean
namespace Examples.PINN

open TorchLean.Tensor
open Lean Data Json

open PyTorch

/-- A parsed PINN state dict together with the inferred TorchLean sequential PINN architecture. -/
structure Checkpoint where
  /-- Inferred sequential fully-connected PINN architecture. -/
  arch   : Architecture
  /-- Layer stack, with weights shaped as PyTorch stores `Linear.weight`. -/
  layers : Array (SomeLinear Float)

/-!
## Activation metadata

Python training scripts often record which nonlinearity they used, and the loader reads it as
optional metadata under `meta.activation`. A missing value selects `tanh`, and an unrecognized
one rejects the checkpoint so that a network with another nonlinearity is never bounded as a
tanh network.
-/

namespace Checkpoint

/--
Parse the optional activation metadata `meta.activation`, selecting `tanh` when `meta` or its
`activation` field is absent.

A present value must be a string naming `tanh`, `relu` or `sin` (also spelled `sine` or
`siren`), in any letter case. Anything else, including a `meta` value that is not an object,
gives `none`, which makes `Examples.PINN.Checkpoint.ofJson?` fail.
-/
def parseActivation (metaOpt : Option Json) : Option HiddenActivation :=
  match metaOpt with
  | none => some HiddenActivation.tanh
  | some (Json.obj metaObj) =>
    match metaObj.get? "activation" with
    | none => some HiddenActivation.tanh
    | some (Json.str s) =>
      match s.toLower with
      | "tanh" => some HiddenActivation.tanh
      | "relu" => some HiddenActivation.relu
      | "sin" | "sine" | "siren" => some HiddenActivation.sin
      | _ => none
    | some _ => none
  | some _ => none

end Checkpoint

/--
Load a PINN state dict, inferring each layer's `(outDim, inDim)` from the shape of its JSON
weight matrix so that the hidden widths are arbitrary.

Expected keys:

- `layers.<i>.weight` and `layers.<i>.bias` for each layer index `i`
- optional `meta.activation`, read by `parseActivation`, which rejects unsupported
  values

The result is `none` when `loadWeights?` rejects the JSON, `linearLayers` rejects the state dict
or the activation is unsupported. `linearLayers` rejects a state dict with no `layers.<i>.weight`
key, a missing bias, a tensor that fails to parse, an entry that is not finite or a layer whose
input width is not the previous layer's output width.
-/
def Checkpoint.ofJson? (j : Json) : Option Checkpoint := do
  let o ← loadWeights? j
  let layers ← (linearLayers o).toOption
  let activation ← parseActivation (o.get? "meta")
  let first ← layers[0]?
  let outDims := layers.map (·.outDim)
  let arch : Architecture :=
    { inputDim := first.inDim
      hiddenDims := outDims.extract 0 (outDims.size - 1)
      outputDim := outDims.back?.getD 0
      activation := activation }
  pure { arch := arch, layers := layers }

end Examples.PINN
end TorchLean
