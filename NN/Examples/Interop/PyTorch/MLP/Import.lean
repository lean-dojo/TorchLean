/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.PyTorch.Import.Core
public import NN.Spec.Layers.Activation
public import NN.Spec.Module.Linear

/-!
# MLP PyTorch Fixture Import

MLP fixture weight import from a PyTorch-style `state_dict`.

On the Python side we usually write JSON (nested lists of floats) under keys that mirror the names
you would see in `model.state_dict()`:

- `fc1.weight`, `fc1.bias`, `fc2.weight`, `fc2.bias` for a hand-written `nn.Module` with `fc1/fc2`,
- or `layers.0.weight`, `layers.0.bias`, ... if the model was built from an `nn.Sequential`.

This file keeps the parsing logic in one place so the rest of the codebase can talk in terms of
typed Lean tensors.
-/

@[expose] public section


namespace Import
namespace MLPPyTorch
open PyTorch

open Spec
open Tensor
open Shape

open Lean
open Data
open Json

-- MLP state dict structure that matches the export format.
--
-- We support two key conventions:
-- - PyTorch `nn.Linear` style: `fc1.weight`, `fc1.bias`, `fc2.weight`, `fc2.bias`
-- - sequential parameter style: `layers.0.weight`, `layers.0.bias`, `layers.2.weight`, `layers.2.bias`
/-- Typed view of an MLP PyTorch `state_dict` (two linear layers).

We keep the tensors as `Float` because these importers are meant for runtime demos: train in Python,
export to JSON, then run/verify in TorchLean.
-/
structure MlpStateDict (inDim hidDim outDim : Nat) where
  /-- First linear layer weight, PyTorch shape `(hidden, input)`. -/
  w1 : Tensor Float (.dim hidDim (.dim inDim .scalar))  -- fc1.weight
  /-- Bias for layer 1. -/
  b1 : Tensor Float (.dim hidDim .scalar)               -- fc1.bias
  /-- Second linear layer weight, PyTorch shape `(output, hidden)`. -/
  w2 : Tensor Float (.dim outDim (.dim hidDim .scalar)) -- fc2.weight
  /-- Bias for layer 2. -/
  b2 : Tensor Float (.dim outDim .scalar)               -- fc2.bias

/-- Load an MLP state dict from JSON (accepts both key conventions described above). -/
def loadMlpStateDict (inDim hidDim outDim : Nat) (j : Json) : Option (MlpStateDict inDim hidDim
  outDim) :=
  let w1Shape : Shape := .dim hidDim (.dim inDim .scalar)
  let b1Shape : Shape := .dim hidDim .scalar
  let w2Shape : Shape := .dim outDim (.dim hidDim .scalar)
  let b2Shape : Shape := .dim outDim .scalar
  do
    -- `loadWeights?` accepts either:
    -- - `{ ...state_dict... }`, or
    -- - `{ "params": { ...state_dict... } }` (a common wrapper in our Python scripts).
    let o ← loadWeights? j
    let tryKeys (w1K b1K w2K b2K : String) : Option (MlpStateDict inDim hidDim outDim) := do
      let w1 ← getTensor? o w1K w1Shape
      let b1 ← getTensor? o b1K b1Shape
      let w2 ← getTensor? o w2K w2Shape
      let b2 ← getTensor? o b2K b2Shape
      pure { w1 := w1, b1 := b1, w2 := w2, b2 := b2 }
    tryKeys "fc1.weight" "fc1.bias" "fc2.weight" "fc2.bias" <|>
      tryKeys "layers.0.weight" "layers.0.bias" "layers.2.weight" "layers.2.bias"

/-- Construct LinearSpec for Float from an MLP state dict. -/
def toLinearSpecs {inDim hidDim outDim : Nat}
  (sd : MlpStateDict inDim hidDim outDim) :
  (LinearSpec Float inDim hidDim × LinearSpec Float hidDim outDim) :=
  ( { weights := sd.w1, bias := sd.b1 }
  , { weights := sd.w2, bias := sd.b2 } )

/-- Convenience: run the Float MLP forward given a state dict and input. -/
def forward {inDim hidDim outDim : Nat}
  (sd : MlpStateDict inDim hidDim outDim)
  (x : Tensor Float (.dim inDim .scalar)) : Tensor Float (.dim outDim .scalar) :=
  let (l1, l2) := toLinearSpecs sd
  let z1 := Spec.linearSpec (α:=Float) l1 x
  let a1 := Activation.reluSpec (α:=Float) z1
  Spec.linearSpec (α:=Float) l2 a1

end MLPPyTorch
end Import
