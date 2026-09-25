/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.Cert.CROWNQuery
public import NN.Verification.Cert.RationalJson
public import NN.API.CLI.Parser

/-!
# JSON acceptance implies output-query safety

`crown_query_v1` describes an exact rational dense/ReLU network, an input box, proposed ReLU
α values, and affine output inequalities. The checker recomputes affine bounds using existing
CROWN transfers. Parameters use integer or fraction strings; numeric JSON literals are rejected.

The theorem concerns the real TorchLean model and query decoded from these bytes. It does not
identify this model with an ONNX file or a floating-point deployment. This is a separate format
from the binary32 node-replay transcript; β dual variables, cuts, and branch trees are unsupported.
-/

@[expose] public section

namespace NN.Verification.CROWNQuery

open Lean _root_.Spec NN.Verification.Cert.RationalJson

def decodeLayer (n : Nat) (j : Json) : Except String (Σ m : Nat, Network n m) := do
  match ← (← j.getObjVal? "kind").getStr? with
  | "linear" =>
      let ⟨m, layer⟩ ← decodeLinear n j
      unless 0 < m do throw "empty linear layer"
      return ⟨m, .linear layer⟩
  | "relu" => return ⟨n, .relu (← decodeVector n (← j.getObjVal? "alpha"))⟩
  | kind => throw s!"unsupported layer kind: {kind}"

def decodeLayers (n : Nat) : List Json → Except String (Σ m : Nat, Network n m)
  | [] => .error "expected at least one layer"
  | j :: rest => do
      let ⟨m, first⟩ ← decodeLayer n j
      match rest with
      | [] => return ⟨m, first⟩
      | _ :: _ =>
          let ⟨k, tail⟩ ← decodeLayers m rest
          return ⟨k, .comp first tail⟩

/-- Decode the model, input domain, and output property from the same document. -/
def decode (j : Json) : Except String (Σ n m : Nat, Query n m) := do
  unless (← (← j.getObjVal? "format").getStr?) == "crown_query_v1" do
    throw "unsupported query format"
  let n ← (← j.getObjVal? "input_dim").getNat?
  unless 0 < n do throw "empty input dimension"
  let input ← j.getObjVal? "input"
  let lo ← decodeVector n (← input.getObjVal? "lo")
  let hi ← decodeVector n (← input.getObjVal? "hi")
  let ⟨m, network⟩ ← decodeLayers n (← (← j.getObjVal? "layers").getArr?).toList
  let query ← j.getObjVal? "query"
  let ⟨k, inequalities⟩ ← decodeLinear m query
  let strict ← (← query.getObjVal? "strict").getBool?
  return ⟨n, m, ⟨network, ⟨lo, hi⟩, k, inequalities, strict⟩⟩

def acceptsJson (j : Json) : Bool :=
  match decode j with
  | .error _ => false
  | .ok ⟨_, _, q⟩ => q.check

/-- Every accepted document denotes a model whose outputs satisfy its query throughout its box. -/
theorem acceptsJson_sound (j : Json) (h : acceptsJson j = true) :
    ∃ (n m : Nat) (q : Query n m), decode j = .ok ⟨n, m, q⟩ ∧ q.Safe := by
  unfold acceptsJson at h
  cases hd : decode j with
  | error error => simp [hd] at h
  | ok decoded =>
      rcases decoded with ⟨n, m, q⟩
      exact ⟨n, m, q, rfl, q.check_sound (by simpa [hd] using h)⟩

def acceptsText (source : String) : Bool :=
  match Json.parse source with
  | .error _ => false
  | .ok j => acceptsJson j

/-- Text acceptance implies the universal real output property, without a producer axiom. -/
theorem acceptsText_sound (source : String) (h : acceptsText source = true) :
    ∃ (j : Json) (n m : Nat) (q : Query n m),
      Json.parse source = .ok j ∧ decode j = .ok ⟨n, m, q⟩ ∧ q.Safe := by
  unfold acceptsText at h
  cases hp : Json.parse source with
  | error error => simp [hp] at h
  | ok j =>
      obtain ⟨n, m, q, hd, hs⟩ := acceptsJson_sound j (by simpa [hp] using h)
      exact ⟨j, n, m, q, rfl, hd, hs⟩

/-- Why `acceptsText` rejected a document, for error messages. The verdict is `acceptsText`. -/
def rejectionReason (source : String) : String :=
  match Json.parse source with
  | .error e => s!"invalid JSON: {e}"
  | .ok j =>
      match decode j with
      | .error e => e
      | .ok _ => "the recomputed bounds do not prove the query"

/-- CLI entry point: `lake exe verify -- crown-query <query.json>`. -/
def run (args : List String) : IO Unit := do
  let usage := "Usage:\n  lake exe verify -- crown-query <path/to/query.json>"
  if TorchLean.CLI.hasHelp args then
    IO.println usage
    return
  let (path, rest) ← IO.ofExcept <| TorchLean.CLI.takePositional? (TorchLean.CLI.dropDashDash args)
  IO.ofExcept <| TorchLean.CLI.checkNoArgs rest
  let some path := path
    | throw <| IO.userError s!"missing query path\n\n{usage}"
  let source ← IO.FS.readFile path
  if acceptsText source then
    IO.println s!"[crown-query] {path}: proved for every real input in the box (acceptsText_sound)"
  else
    throw <| IO.userError s!"[crown-query] {path}: not proved: {rejectionReason source}"

end NN.Verification.CROWNQuery
