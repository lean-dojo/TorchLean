/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.RL.Boundary.Core
public import NN.Runtime.PyTorch.Import.Core
import Lean.Data.Json

/-!
# RL Trust Boundary JSON Loader

This module contains the JSON-lines/interchange side of the RL trust boundary. The contract itself
lives in `NN.Runtime.RL.Boundary.Core`; this file only explains how to parse a small Python-friendly
rollout schema and immediately validate it against that contract.

Keeping JSON separate matters because proof files usually need the contract/checker semantics, not
the parser or PyTorch import layer. Runtime bridges can import this file when they need external
rollout loading.
-/

@[expose] public section

namespace Runtime
namespace RL
namespace Boundary

open Spec
open Tensor
open Import
open Lean
open Json

/-!
## JSON loading (one simple interchange format)

We provide a small JSON schema that is easy to emit from Python:

```json
{
  "transitions": [
    {
      "obs": <nested arrays matching obsShape>,
      "action": <nat>,
      "reward": <float>,
      "terminated": <bool>,
      "truncated": <bool>,
      "next_obs": <nested arrays matching obsShape>
    },
    ...
  ]
}
```

This is one supported integration path, using a practical “lowest common
denominator” that is easy to diff and validate.
-/

/-!
### JSON Primitive Parsers

These are small helpers used by the trust-boundary JSON loader and by runtime bridges (e.g. a
Gymnasium subprocess client).

They return `Except String ...` so they can be used from both pure loaders and `IO` code that wants
to convert errors into `IO.userError`.
-/

/--
Parse a JSON number as a nonnegative integer.

This is *strict*: it rejects non-integers (e.g. `1.5`) and rejects numbers not exactly
representable as an integer when converted through `Float`.
-/
def parseNatStrict (j : Json) : Except String Nat :=
  match j with
  | .num n =>
      let f := n.toFloat
      if f.isNaN || f.isInf then
        .error s!"RL boundary: expected integer JSON number, got {f}."
      else if f < 0 then
        .error s!"RL boundary: expected nonnegative integer, got {f}."
      else
        let u := f.toUInt64
        if (Float.ofNat u.toNat) == f then
          .ok (u.toNat)
        else
          .error s!"RL boundary: expected integer JSON number, got {f}."
  | _ => .error "RL boundary: expected a JSON number."

/-- Parse a boolean field from JSON. -/
def parseBool (field : String) (j : Json) : Except String Bool :=
  match j with
  | .bool b => .ok b
  | _ => .error s!"RL boundary: expected `{field}` to be a JSON boolean."

/-- Parse a numeric field from JSON as a host `Float`. -/
def parseFloat (field : String) (j : Json) : Except String Float :=
  match j with
  | .num n => .ok n.toFloat
  | _ => .error s!"RL boundary: expected `{field}` to be a JSON number."

/--
Parse a JSON value as a typed tensor of shape `s`.

This relies on TorchLean's JSON tensor encoding used by Python bridges.
-/
def parseTensorE (field : String) (s : Shape) (j : Json) : Except String (Tensor Float s) :=
  match Import.PyTorch.parseTensor s j with
  | some t => .ok t
  | none => .error s!"RL boundary: `{field}` did not match the expected shape {Spec.Shape.pretty s}."

/-- Parse a single transition object using the schema described above. -/
def parseTransitionJson {obsShape : Shape} {nActions : Nat}
    (c : Contract obsShape nActions) (j : Json) :
    Except String (Transition obsShape nActions) := do
  let o ←
    match j.getObj? with
    | .ok o => pure o
    | .error msg => throw s!"RL boundary: transition must be an object: {msg}"

  let obsJ ←
    match o.get? "obs" with
    | some v => pure v
    | none => throw "RL boundary: transition missing field `obs`."
  let nextObsJ ←
    match o.get? "next_obs" with
    | some v => pure v
    | none => throw "RL boundary: transition missing field `next_obs`."
  let actionJ ←
    match o.get? "action" with
    | some v => pure v
    | none => throw "RL boundary: transition missing field `action`."
  let rewardJ ←
    match o.get? "reward" with
    | some v => pure v
    | none => throw "RL boundary: transition missing field `reward`."
  let terminatedJ ←
    match o.get? "terminated" with
    | some v => pure v
    | none => throw "RL boundary: transition missing field `terminated`."
  let truncatedJ ←
    match o.get? "truncated" with
    | some v => pure v
    | none => throw "RL boundary: transition missing field `truncated`."

  let obs ← parseTensorE (field := "obs") obsShape obsJ
  let nextObs ← parseTensorE (field := "next_obs") obsShape nextObsJ
  let action ← parseNatStrict actionJ
  let reward ← parseFloat (field := "reward") rewardJ
  let terminated ← parseBool (field := "terminated") terminatedJ
  let truncated ← parseBool (field := "truncated") truncatedJ

  checkTransition (obsShape := obsShape) (nActions := nActions) c obs nextObs action reward terminated truncated

/-- Load and validate a rollout file, returning an array of typed transitions. -/
def loadRollout {obsShape : Shape} {nActions : Nat}
    (path : String)
    (c : Contract obsShape nActions) :
    IO (Array (Transition obsShape nActions)) := do
  let jsonStr ← IO.FS.readFile path
  let j ←
    match Json.parse jsonStr with
    | .ok j => pure j
    | .error msg => throw <| IO.userError s!"RL boundary: bad JSON: {msg}"
  let o ←
    match j.getObj? with
    | .ok o => pure o
    | .error msg => throw <| IO.userError s!"RL boundary: top-level JSON must be an object: {msg}"

  let transitionsJ ←
    match o.get? "transitions" with
    | some v => pure v
    | none => throw <| IO.userError "RL boundary: missing required field `transitions`"
  let transitionsArr ←
    match transitionsJ with
    | .arr xs => pure xs
    | _ => throw <| IO.userError "RL boundary: field `transitions` must be an array"

  let mut out : Array (Transition obsShape nActions) := #[]
  for tj in transitionsArr do
    match parseTransitionJson (obsShape := obsShape) (nActions := nActions) c tj with
    | .ok t => out := out.push t
    | .error e => throw <| IO.userError e
  pure out

/--
Load a rollout file and validate every transition, returning per-transition results.

Unlike `loadRollout`, this function does **not** stop at the first bad transition. Instead it
returns an array aligned with the input JSON `transitions` array:

- `ok t` means the transition decoded and passed the contract check.
- `error msg` means decoding or contract checking failed for that transition.

This is useful for widget-style debugging where users want to see a *summary* of violations rather
than only the first failure.
-/
def loadRolloutAll {obsShape : Shape} {nActions : Nat}
    (path : String)
    (c : Contract obsShape nActions) :
    IO (Array (Except String (Transition obsShape nActions))) := do
  let jsonStr ← IO.FS.readFile path
  let j ←
    match Json.parse jsonStr with
    | .ok j => pure j
    | .error msg => throw <| IO.userError s!"RL boundary: bad JSON: {msg}"
  let o ←
    match j.getObj? with
    | .ok o => pure o
    | .error msg => throw <| IO.userError s!"RL boundary: top-level JSON must be an object: {msg}"

  let transitionsJ ←
    match o.get? "transitions" with
    | some v => pure v
    | none => throw <| IO.userError "RL boundary: missing required field `transitions`"
  let transitionsArr ←
    match transitionsJ with
    | .arr xs => pure xs
    | _ => throw <| IO.userError "RL boundary: field `transitions` must be an array"

  let mut out : Array (Except String (Transition obsShape nActions)) := #[]
  for tj in transitionsArr do
    out := out.push (parseTransitionJson (obsShape := obsShape) (nActions := nActions) c tj)
  pure out

end Boundary
end RL
end Runtime
