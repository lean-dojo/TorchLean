/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.Certificate.CROWNQuery
public import TorchLean.Verify.Certificate.RationalJSON
public import TorchLean.Verify.JSON

/-!
# JSON acceptance implies output-query safety

A `crown_query_v1` document describes an exact rational dense/ReLU network, an input box, proposed
ReLU α values and affine output inequalities. The checker recomputes affine bounds with the CROWN
transfers. Parameters are integer or fraction strings, and a numeric JSON literal is rejected.

The soundness theorems concern the real TorchLean model and query decoded from these bytes, and
they do not identify this model with an ONNX file or a floating-point deployment. The format is
separate from the binary32 node-replay transcript and has no β dual variables, cuts or branch
trees.
-/

@[expose] public section

namespace TorchLean.Certificate.CROWNQuery

open Lean RationalJson

/-- Decode one `linear` or `relu` layer whose input dimension is `n`. -/
def decodeLayer (n : Nat) (j : Json) : Except String (Σ m : Nat, Network n m) := do
  match ← (← j.getObjVal? "kind").getStr? with
  | "linear" =>
      let ⟨m, layer⟩ ← decodeLinear n j
      unless 0 < m do throw "empty linear layer"
      return ⟨m, .linear layer⟩
  | "relu" => return ⟨n, .relu (← decodeVector n (← j.getObjVal? "alpha"))⟩
  | kind => throw s!"unsupported layer kind: {kind}"

/-- Decode the model, input domain, and output property from the same document. -/
def decode (j : Json) : Except String (Σ n m : Nat, Query n m) := do
  unless (← (← j.getObjVal? "format").getStr?) == "crown_query_v1" do
    throw "unsupported query format"
  let n ← (← j.getObjVal? "input_dim").getNat?
  unless 0 < n do throw "empty input dimension"
  let input ← j.getObjVal? "input"
  let lo ← decodeVector n (← input.getObjVal? "lo")
  let hi ← decodeVector n (← input.getObjVal? "hi")
  let ⟨m, network⟩ ← decodeChain decodeLayer .comp n (← (← j.getObjVal? "layers").getArr?).toList
  let query ← j.getObjVal? "query"
  let ⟨k, inequalities⟩ ← decodeLinear m query
  let strict ← (← query.getObjVal? "strict").getBool?
  return ⟨n, m, ⟨network, ⟨lo, hi⟩, k, inequalities, strict⟩⟩

/-- Decode and check a query document, rejecting a malformed one. -/
def acceptsJson (j : Json) : Bool :=
  Json.accepts decode (fun ⟨_, _, q⟩ => q.check) j

/-- Every accepted document denotes a model whose outputs satisfy its query throughout its box. -/
theorem acceptsJson_sound (j : Json) (h : acceptsJson j = true) :
    ∃ (n m : Nat) (q : Query n m), decode j = .ok ⟨n, m, q⟩ ∧ q.Safe := by
  obtain ⟨⟨n, m, q⟩, hd, hq⟩ := Json.accepts_sound
    (P := fun ⟨_, _, q⟩ => q.Safe) (fun ⟨_, _, q⟩ => q.check_sound) h
  exact ⟨n, m, q, hd, hq⟩

/-- Parse and check a query document given as text. The `crown-query` command of `lake exe verify`
prints this verdict. -/
def acceptsText (source : String) : Bool :=
  Json.acceptsText decode (fun ⟨_, _, q⟩ => q.check) source

/-- Accepted text parses and decodes to a query that is safe for every real input of its box, with
no axiom about the producer. -/
theorem acceptsText_sound (source : String) (h : acceptsText source = true) :
    ∃ (j : Json) (n m : Nat) (q : Query n m),
      Json.parse source = .ok j ∧ decode j = .ok ⟨n, m, q⟩ ∧ q.Safe := by
  obtain ⟨j, ⟨n, m, q⟩, hp, hd, hq⟩ := Json.acceptsText_sound
    (P := fun ⟨_, _, q⟩ => q.Safe) (fun ⟨_, _, q⟩ => q.check_sound) h
  exact ⟨j, n, m, q, hp, hd, hq⟩

/-- Why `acceptsText` rejected a document, for error messages. The verdict is `acceptsText`. -/
def rejectionReason (source : String) : String :=
  match DecimalJson.parse source, Json.parse source with
  | .error e, _ | _, .error e => s!"invalid JSON: {e}"
  | .ok _, .ok j =>
      match decode j with
      | .error e => e
      | .ok ⟨_, _, q⟩ =>
          if q.wellFormed then "the recomputed bounds do not prove the query"
          else "the input box is reversed or the query has no inequality"

end TorchLean.Certificate.CROWNQuery
