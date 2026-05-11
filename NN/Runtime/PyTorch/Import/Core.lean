/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Tensor

/-!
# Import Core

PyTorch import core (JSON parsing).

The Python side of TorchLean round-trips usually writes a JSON object containing nested arrays of
floats (a lightweight, Lean-readable projection of a PyTorch `state_dict`). The model-agnostic
adapter emitted by `NN.Runtime.PyTorch.Export.StateDict` is the intended path from `.pt` / `.pth`
checkpoints into this JSON format.

Design note:

In typical PyTorch workflows, weights are often serialized via `torch.save(model.state_dict(), ...)`
or related checkpoint wrappers. TorchLean deliberately avoids parsing those PyTorch binary formats
directly in Lean. Instead, PyTorch loads the checkpoint and emits a small JSON representation that
is easy to validate against a Lean `Shape` and easy to diff in tests.

This importer is about **weights only**. Importing a captured *graph* (e.g. ONNX or `torch.export`)
is a separate problem and lives at a different abstraction layer than this JSON `state_dict` shim.

This module is where we keep the shared logic that most PyTorch → TorchLean importers need:

- parse nested JSON arrays into shape-checked `Tensor Float s`,
- handle a small amount of “state_dict ergonomics” (key lookup, optional wrappers, index parsing),
- keep everything model-agnostic, so the *model-specific* code can stay small and readable.

Reading map:

- `parseTensor` is the core JSON-to-tensor conversion.
- `loadWeights?` and `unwrapParams` handle the two JSON layouts we accept.
- `getTensor?` / `getTensorE` are the main lookup helpers used by the model-specific importers.
-/

@[expose] public section


namespace Import
namespace PyTorch

open Spec
open Tensor
open Shape
open Lean
open Data
open Json

/-- A PyTorch-style `state_dict` encoded as a JSON object. -/
abbrev StateDict := Std.TreeMap.Raw String Json

/--
`defaultTensor s` is a zero-filled sentinel used in internal parsing helpers.

It is only used in code paths that are unreachable once we have validated the JSON shape
(e.g. after checking an array has exactly the expected length).
-/
def defaultTensor : (s : Shape) → Tensor Float s
  | .scalar => Tensor.scalar 0.0
  | .dim _n s => Tensor.dim (fun _ => defaultTensor s)

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
importers can still read them.
-/
def unwrapParams (o : StateDict) : StateDict :=
  match o.get? "params" with
  | some (.obj p) =>
      o.foldl (init := p) (fun acc k v =>
        if k = "params" then acc else acc.insert k v)
  | _ => o

/--
Load weights from JSON, accepting either:

- `{ ...state_dict... }`, or
- `{ "params": { ...state_dict... } }`.
-/
def loadWeights? (j : Json) : Option StateDict := do
  let o ← loadStateDict? j
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

/-!
## Error-reporting variants (ergonomics)

Most of the import code in this folder is written in the `Option` monad to keep examples short.
When you are debugging a round-trip, it is often more helpful to get a concrete *reason* why an
import failed (missing key vs wrong JSON type vs wrong shape).

The helpers below provide small `Except String` wrappers around the `Option`-based core.
-/

/--
Load weights from JSON with an error message on failure.

This is the `Except` analogue of `loadWeights?`.
-/
def loadWeightsE (j : Json) : Except String StateDict :=
  match loadWeights? j with
  | some o => .ok o
  | none =>
      .error
        "PyTorch import: expected a JSON object `{...}` or a wrapper `{ \"params\": {...} }`."

/--
Look up a tensor by key, returning a human-friendly error on failure.

This is the `Except` analogue of `getTensor?`.
-/
def getTensorE (o : StateDict) (k : String) (s : Shape) : Except String (Tensor Float s) :=
  match getTensor? o k s with
  | some t => .ok t
  | none =>
      match o.get? k with
      | none => .error s!"PyTorch import: missing key `{k}`."
      | some _ =>
          .error s!"PyTorch import: key `{k}` is present, but did not match the expected shape."

/-- Convenience: parse a length-`n` vector tensor. -/
abbrev parseVecTensor (n : Nat) (j : Json) :
    Option (Tensor Float (.dim n .scalar)) :=
  parseTensor (.dim n .scalar) j

/-- Convenience: parse a `rows × cols` matrix tensor. -/
abbrev parseMatTensor (rows cols : Nat) (j : Json) :
    Option (Tensor Float (.dim rows (.dim cols .scalar))) :=
  parseTensor (.dim rows (.dim cols .scalar)) j

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
    -- In Lean 4.28+, `String.drop/take/trim...` produce `String.Slice`, and some slice APIs are
    -- being phased out. Convert to `String` early so we can use the usual `String.length`.
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

/-- Drop the last element of a list of `Nat` (used when inferring hidden layer widths). -/
def dropLastNat : List Nat → List Nat
  | [] => []
  | [_] => []
  | x :: xs => x :: dropLastNat xs

/-!
## Convenience parsers for function-based constructors

Some call sites build tensors via `Spec.vector_tensor` / `Spec.matrix_tensor`, whose inputs are
functions (`Fin n → Float` and `Fin m → Fin n → Float`).

`parseFloatVec` and `parseFloatMatrix` keep those call sites readable without duplicating JSON
parsing logic outside this core module.
-/

/-- Parse a JSON array into a `Fin n → Float` function. -/
def parseFloatVec (n : Nat) (j : Json) : Option (Fin n → Float) := do
  let t ← parseVecTensor n j
  pure (fun i => Tensor.vecGet t i)

/-- Parse a JSON matrix into a `Fin rows → Fin cols → Float` function. -/
def parseFloatMatrix (rows cols : Nat) (j : Json) : Option (Fin rows → Fin cols → Float) := do
  let t ← parseMatTensor rows cols j
  pure (fun i j => Spec.get2 t i j)

end PyTorch
end Import
