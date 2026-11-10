/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Layers.Linear

/-!
# Float tensors from nested JSON arrays

`parseTensor` reads a nested JSON array of numbers into `Tensor Float s`. `loadWeights?` returns
the object of such arrays that a JSON document is. `parseIndexedKey` and `inferMatrixDims` read
layer indices from keys and matrix sizes from arrays, for loaders that learn a model's widths from
its weights. With them, `linearLayers` reads the linear layers `layers.<i>.weight` and
`layers.<i>.bias` of a state dict and learns their widths from the weights.

Each number goes through `JsonNumber.toFloat`, so `parseTensor` is not word-exact.
`TorchLean.PyTorch.state` in `TorchLean.NN.PyTorch.Import.State` reads a PyTorch `state_dict`
word for word into `TensorPack α shapes`.
-/

@[expose] public section


namespace TorchLean
namespace PyTorch

open TorchLean.Tensor
open Lean
open Data
open Json

/-- A PyTorch-style `state_dict` encoded as a JSON object. -/
abbrev StateDict := Std.TreeMap.Raw String Json

/--
Parse a JSON value into a `Tensor Float s`.

The JSON encoding follows the tensor shape:

- scalars are JSON numbers,
- `Shape.dim n s` is a JSON array of length `n` whose entries recursively encode `s`.

A payload that does not match the expected shape gives `none`.

A JSON negative zero (`-0.0` or `-0`) is read as `+0.0`: `Lean.JsonNumber` stores an `Int`
mantissa, which has no negative zero.
-/
def parseTensor : (s : Shape) → Json → Option (Tensor Float s)
  | .scalar, j =>
    match j with
    | .num n => some (Tensor.scalar n.toFloat)
    | _ => none
  | .dim n s, j =>
    match j with
    | .arr xs =>
      if h : xs.size = n then
        match Array.mapM' (fun x => parseTensor s x) xs with
        | some ts =>
            -- The library traversal preserves left-to-right parsing and supplies the size proof.
            some <|
              Tensor.dim (fun i : Fin n =>
                ts.val[i.val]'(by simpa only [ts.property, h] using i.isLt))
        | none => none
      else
        none
    | _ => none

/-!
## Weight objects

A weight object is keyed by strings, as a PyTorch `state_dict` is.
-/

/-- The weight object of a JSON document: the document itself when it is a JSON object. -/
def loadWeights? (j : Json) : Option StateDict :=
  match j with
  | .obj o => some o
  | _ => none

/-!
## Small parsing helpers used by shape-inferring importers

Some importers accept variable-width stacks, for example a PINN that reads its hidden widths from
the checkpoint, and infer layer indices and matrix dimensions from the JSON with these helpers.
`linearLayers` reads such a stack of linear layers with them.
-/

/--
Parse keys of the form `prefix ++ <nat> ++ suffix`.

Example: `parseIndexedKey "layers." ".weight" "layers.3.weight" = some 3`.
-/
def parseIndexedKey (pref suff key : String) : Option Nat :=
  if key.startsWith pref && key.endsWith suff then
    -- Convert slices to owned strings before measuring them, so the importer does not depend on
    -- slice-specific API details.
    let core : String := (key.drop pref.length).toString
    let slice : String := (core.take (core.length - suff.length)).toString
    slice.toNat?
  else
    none

/--
Infer `(rows, cols)` for a JSON matrix encoded as an array of arrays.

The dimensions come from the outer length and the length of the first row, and an empty array gives
`(0, 0)`. Rows are not checked for equal length, so call sites that need it check it themselves.
-/
def inferMatrixDims : Json → Option (Nat × Nat)
  | Json.arr rows =>
      if rows.size = 0 then
        some (0, 0)
      else
        match rows[0]? with
        | some (Json.arr cols) => some (rows.size, cols.size)
        | _ => none
  | _ => none

/-- The linear layers of a state dict: for each `i` with a key `layers.<i>.weight`, in increasing
order, that matrix and the vector `layers.<i>.bias`, the matrix's rows and columns being the layer's
output and input widths. An error unless there is such a key, every entry is a finite number and
each layer's input width is the previous layer's output width (`linearLayers_chain`). -/
def linearLayers (sd : StateDict) : Except String (Array (SomeLinear Float)) := do
  let indices := (sd.foldl (init := #[]) fun acc key _ =>
    match parseIndexedKey "layers." ".weight" key with
    | some i => acc.push i
    | none => acc).qsort (· < ·) |>.eraseReps
  if indices.isEmpty then
    throw "state dict: expected keys of the form layers.<i>.weight"
  let layers ← indices.mapM fun i => do
    let some w := sd.get? s!"layers.{i}.weight"
      | throw s!"state dict: missing key layers.{i}.weight"
    let some b := sd.get? s!"layers.{i}.bias"
      | throw s!"state dict: missing key layers.{i}.bias"
    let some (rows, cols) := inferMatrixDims w
      | throw s!"layers.{i}.weight: expected a matrix of numbers"
    let some weights := parseTensor (.dim rows (.dim cols .scalar)) w
      | throw s!"layers.{i}.weight: expected a matrix of numbers"
    let some bias := parseTensor (.dim rows .scalar) b
      | throw s!"layers.{i}.bias: expected {rows} numbers"
    unless weights.all Float.isFinite && bias.all Float.isFinite do
      throw s!"layers.{i}: expected finite entries"
    pure ({ inDim := cols, outDim := rows, spec := { weights, bias } } : SomeLinear Float)
  if ∀ i (h : i < layers.size - 1),
      (layers[i]'(by omega)).outDim = (layers[i + 1]'(by omega)).inDim then
    pure layers
  else
    throw s!"state dict: layer widths {layers.map fun l => (l.inDim, l.outDim)} do not chain"

/-- Each layer `linearLayers` returns takes the previous layer's outputs as its inputs. -/
theorem linearLayers_chain {sd : StateDict} {layers : Array (SomeLinear Float)}
    (h : linearLayers sd = .ok layers) (i : Nat) (hi : i < layers.size - 1) :
    (layers[i]'(by omega)).outDim = (layers[i + 1]'(by omega)).inDim := by
  unfold linearLayers at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  split at h
  · simp [throw, throwThe, MonadExceptOf.throw] at h
  · split at h
    · cases h
    · split at h
      next hc =>
        cases h
        exact hc i hi
      next => simp [throw, throwThe, MonadExceptOf.throw] at h

end PyTorch
end TorchLean
