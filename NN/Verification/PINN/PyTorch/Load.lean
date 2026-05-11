/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.PyTorch.Import.Core
public import NN.Verification.PINN.Architecture

/-!
# PINN PyTorch Checkpoint Loading

PINN weight import (JSON → typed tensors).

This is the “loading” half of the PINN PyTorch bridge:

- read a PyTorch-style state dictionary encoded as JSON,
- parse each weight/bias matrix into a shape-checked `Tensor Float ...` (shape inferred/checked),
- infer a TorchLean `SequentialPINNArch` from the layer shapes plus optional activation metadata.

The generic JSON helpers (`loadWeights?`, `parseTensor`, `inferMatrixDims`, …) live in
`NN/Runtime/PyTorch/Import/Core.lean`.
-/

@[expose] public section


namespace Import
namespace PINNPyTorch

open Spec
open Tensor
open Shape
open Lean
open Data
open Json

open NN.Verification.PINN
open Import.PyTorch

/-
We keep the state representation explicit:

- `PinnLayer` stores the raw weight/bias tensors as trained in Python,
- `PinnState` pairs those tensors with the inferred sequential PINN architecture.
-/

/-- One fully-connected layer of a PINN (weights + bias) as trained in PyTorch. -/
structure PinnLayer where
  /-- Input dimension. -/
  inDim  : Nat
  /-- Output dimension. -/
  outDim : Nat
  /-- weights. -/
  weights : Tensor Float (.dim outDim (.dim inDim .scalar))
  /-- bias. -/
  bias    : Tensor Float (.dim outDim .scalar)

/-- A parsed PINN state dict together with the inferred TorchLean sequential PINN architecture. -/
structure PinnState where
  /-- Inferred sequential fully-connected PINN architecture. -/
  arch   : SequentialPINNArch
  /-- Layer stack. -/
  layers : List PinnLayer

/-!
## Activation metadata

Python training scripts often record which nonlinearity they used. We treat that as optional
metadata under `meta.activation`. If it is missing (or unknown), we default to `tanh`.
-/

/- Parse optional activation metadata from JSON (`meta.activation`). -/
namespace Internal

/-- Internal: parse optional activation metadata from JSON (`meta.activation`). -/
def parseActivation (metaOpt : Option Json) : HiddenActivation :=
  match metaOpt with
  | some (Json.obj metaObj) =>
    match metaObj.get? "activation" with
    | some (Json.str s) =>
      let s := s.toLower
      if s = "relu" then HiddenActivation.relu
      else if s = "sin" ∨ s = "sine" ∨ s = "siren" then HiddenActivation.sin
      else HiddenActivation.tanh
    | _ => HiddenActivation.tanh
  | _ => HiddenActivation.tanh

end Internal

/--
Load a PINN state dict with arbitrary hidden widths.

Expected keys:

- `layers.<i>.weight` and `layers.<i>.bias` for each layer index `i`
- optional `meta.activation`

Unlike fixed-shape demos, we infer `(outDim, inDim)` for each layer from the JSON matrix shape.
-/
def loadPinnState (j : Json) : Option PinnState := do
  -- Accepts both `{...}` and `{ "params": {...} }`.
  let o ← loadWeights? j

  -- Collect `i` from `layers.<i>.weight` keys, then parse in ascending order.
  let weightIdxs :=
    (o.toList.foldl
      (fun acc kv =>
        match parseIndexedKey "layers." ".weight" kv.fst with
        | some idx => idx :: acc
        | none => acc)
      ([] : List Nat)).toArray.qsort (· < ·) |>.toList

  match weightIdxs with
  | [] => none
  | _ =>
    let activation := Internal.parseActivation (o.get? "meta")
    let layersRev ←
      weightIdxs.foldlM (fun acc idx => do
        let base := s!"layers.{idx}"
        let wJson ← getJson? o (base ++ ".weight")
        let bJson ← getJson? o (base ++ ".bias")
        let (outDim, inDim) ← inferMatrixDims wJson
        let weights ← parseTensor (.dim outDim (.dim inDim .scalar)) wJson
        let bias ← parseTensor (.dim outDim .scalar) bJson
        pure ({ inDim := inDim, outDim := outDim, weights := weights, bias := bias } :: acc)) []

    let layers := layersRev.reverse
    match layers with
    | [] => none
    | first :: _ =>
      let outDims := layers.map (·.outDim)
      let hidden := dropLastNat outDims
      let outputDim :=
        match outDims.reverse with
        | [] => 0
        | x :: _ => x
      let arch : SequentialPINNArch :=
        { inputDim := first.inDim
          hiddenDims := hidden
          outputDim := outputDim
          activation := activation }
      pure { arch := arch, layers := layers }

end PINNPyTorch
end Import
