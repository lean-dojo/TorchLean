/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

import Mathlib.Tactic.Bound.Init
public import NN.Spec.Core.Tensor.Core
public import NN.Spec.Core.Tensor -- shake: keep

/-!
# Import Core

PyTorch import core (JSON parsing).

The Python side of TorchLean round-trips usually writes a JSON object containing nested arrays of
floats (a Lean-readable projection of a PyTorch `state_dict`). The model-agnostic
adapter emitted by `NN.Runtime.PyTorch.Export.StateDict` is the intended path from `.pt` / `.pth`
checkpoints into this JSON format.

Design note:

In typical PyTorch workflows, weights are often serialized via `torch.save(model.state_dict(), ...)`
or related checkpoint wrappers. TorchLean avoids parsing those PyTorch binary formats
directly in Lean. Instead, PyTorch loads the checkpoint and emits a small JSON representation that
is easy to validate against a Lean `Shape` and easy to diff in tests.

This importer is about **weights only**. Importing a captured *graph* (e.g. ONNX or `torch.export`)
is a separate problem and lives at a different abstraction layer than this JSON `state_dict`
adapter.

This module is where we keep the shared logic that most PyTorch → TorchLean importers need:

- parse nested JSON arrays into shape-checked `Tensor Float s`,
- handle a small amount of “state_dict ergonomics” (key lookup, optional wrappers, index parsing),
- keep everything model-agnostic, so the *model-specific* code can stay small and readable.

The public helpers are organized as follows:

- `parseTensor` is the core JSON-to-tensor conversion.
- `loadWeights?` and `unwrapParams` handle the two JSON layouts we accept.
- `getTensor?` and `getTensorFirst?` are the main lookup helpers used by the model-specific
  importers.
-/

@[expose] public section


namespace Import
namespace PyTorch

open Spec TorchLean
open TorchLean TorchLean.Tensor
open Shape
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

If the JSON payload does not match the expected shape, we return `none`.
-/
def parseTensor : (s : Shape) → Json → Option (Tensor Float s)
  | .scalar, j =>
    match j with
    | .num n => some (Tensor.scalar n.toFloat)
    | _ => none
  | .dim n s, j =>
    match j with
    | .arr xs =>
      if _h : xs.size = n then
        match xs.mapM (fun x => parseTensor s x) with
        | some ts =>
            -- `Array.mapM` should preserve size, but we keep a runtime check here so we can build a
            -- tensor with O(1) indexing (instead of the O(n) if-chain used by the previous parser).
            if hts : ts.size = n then
              some <|
                Tensor.dim (fun i : Fin n =>
                  ts[i.val]'(by
                    cases hts
                    exact i.2))
            else
              none
        | none => none
      else
        none
    | _ => none

/-!
## state_dict helpers

We use JSON objects keyed by strings because that mirrors PyTorch’s `state_dict` convention.
Some TorchLean Python scripts wrap the object as `{ "params": { ... } }`; `loadWeights?` accepts
both formats.
-/

/-- Read a JSON value as a `StateDict`. -/
def loadStateDict? : Json → Option StateDict
  | .obj o => some o
  | _ => none

/--
If the object contains a `"params"` field that is itself an object, unwrap it.

We also merge any *other* top-level fields (e.g. `"meta"`) into the returned dictionary so model
importers can still read them. Parameter entries take precedence: wrapper metadata must never
replace a tensor with the same key.
-/
def unwrapParams (o : StateDict) : StateDict :=
  match o.get? "params" with
  | some (.obj p) =>
      let withWrapperFields := o.foldl (init := p) (fun acc k v =>
        if k = "params" then acc else acc.insert k v)
      p.foldl (init := withWrapperFields) (fun acc k v => acc.insert k v)
  | _ => o

/-- Whether a JSON string names the float32 format accepted by this importer. -/
def supportedDType : Json → Bool
  | .str "float32" => true
  | .str "torch.float32" => true
  | _ => false

/--
Whether optional state-dict metadata describes the numeric format supported by this importer.

The general adapter records one metadata object per parameter. Older checked-in examples use one
model-level `"dtype": "float32"` entry instead. Both formats are accepted, but an explicit bf16,
float64, or integer dtype is rejected rather than silently reinterpreted as `Tensor Float`.
-/
def supportedWrapperDTypes (o : StateDict) : Bool :=
  match o.get? "params", o.get? "meta" with
  | some (.obj _), some (.obj metadata) =>
      match metadata.get? "dtype" with
      | some dtype => supportedDType dtype
      | none =>
          metadata.foldl (init := true) (fun ok _ entry =>
            ok &&
              match entry with
              | .obj fields =>
                  match fields.get? "dtype" with
                  | some dtype => supportedDType dtype
                  | none => false
              | _ => false)
  | some (.obj _), none => true
  | some (.obj _), some _ => false
  | _, _ => true

/--
Load weights from JSON, accepting either:

- `{ ...state_dict... }`, or
- `{ "params": { ...state_dict... } }`.
-/
def loadWeights? (j : Json) : Option StateDict := do
  let o ← loadStateDict? j
  guard (supportedWrapperDTypes o)
  pure (unwrapParams o)

/-- Look up a JSON field by key in a `StateDict`. -/
def getJson? (o : StateDict) (k : String) : Option Json :=
  o.get? k

/-- Look up a JSON field and require it to be a JSON object. -/
def getObj? (o : StateDict) (k : String) : Option StateDict := do
  match o.get? k with
  | some (.obj o) => some o
  | _ => none

/-- Look up a JSON field and require it to be a JSON string. -/
def getStr? (o : StateDict) (k : String) : Option String := do
  match o.get? k with
  | some (.str s) => some s
  | _ => none

/--
Look up a key and parse it as a tensor of a given expected shape.

This is the helper most model-specific importers use to keep the “key wiring” readable.
-/
def getTensor? (o : StateDict) (k : String) (s : Shape) : Option (Tensor Float s) := do
  let j ← getJson? o k
  parseTensor s j

/--
Try a list of state-dict keys in order and return the first tensor matching the expected shape.

This supports importers that accept both a compact interchange name and a framework-native module
path for the same parameter.
-/
def getTensorFirst? (o : StateDict) (keys : List String) (s : Shape) :
    Option (Tensor Float s) :=
  match keys with
  | [] => none
  | key :: remaining =>
      getTensor? o key s <|> getTensorFirst? o remaining s

/-!
## Error-reporting variants (ergonomics)

Most importers in this folder use `Option` for direct structural parsing. Round-trip checks need
more precise failures, especially when distinguishing a missing key from a wrong JSON type or shape.

The helpers below provide small `Except String` wrappers around the `Option`-based core.
-/

/-!
## Small parsing helpers used by shape-inferring importers

Some importers allow variable-width stacks (e.g. a PINN that learns its hidden widths from the
checkpoint). Those cases need a little help to infer indices and matrix dimensions from JSON.
-/

/--
Parse keys of the form `prefix ++ <nat> ++ suffix`.

Example: `parseIndexedKey "layers." ".weight" "layers.3.weight" = some 3`.
-/
def parseIndexedKey (pref suff key : String) : Option Nat :=
  if key.startsWith pref && key.endsWith suff then
    -- Convert slices to owned strings before measuring them; the importer should not depend on
    -- slice-specific API details.
    let core : String := (key.drop pref.length).toString
    let slice : String := (core.take (core.length - suff.length)).toString
    slice.toNat?
  else
    none

/--
Infer `(rows, cols)` for a JSON matrix encoded as an array of arrays.

This helper infers dimensions from the outer length and first row length.
Call sites that need stronger validation (all rows same length) should add an explicit check.
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

/-!
## Convenience parsers for function-based constructors

Some call sites build tensors via `Spec.vector_tensor` / `Spec.matrix_tensor`, whose inputs are
functions (`Fin n → Float` and `Fin m → Fin n → Float`).

`parseFloatVec` and `parseFloatMatrix` keep those call sites readable without duplicating JSON
parsing logic outside this core module.
-/

/-- Parse a JSON array into a `Fin n → Float` function. -/
def parseFloatVec (n : Nat) (j : Json) : Option (Fin n → Float) := do
  let t ← parseTensor (.dim n .scalar) j
  pure (fun i => Tensor.getScalar t i)

/-- Parse a JSON matrix into a `Fin rows → Fin cols → Float` function. -/
def parseFloatMatrix (rows cols : Nat) (j : Json) : Option (Fin rows → Fin cols → Float) := do
  let t ← parseTensor (.dim rows (.dim cols .scalar)) j
  pure (fun i j => Spec.get2 t i j)

end PyTorch
end Import
