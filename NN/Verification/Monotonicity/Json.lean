/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.Monotonicity
public import NN.Verification.Cert.RationalJson

/-!
# JSON acceptance implies real monotonicity

The `monotonicity_v1` format records an `input_dim` and a nonempty `layers` array. Linear layers
have `kind = "linear"`, a row-major `weights` array and a `bias` array; ReLU layers have
`kind = "relu"`. Parameters are exact integer or fraction strings, never rounded JSON numbers.

The decoder constructs shape-checked certificates using TorchLean tensors. Acceptance implies
global real monotonicity of the decoded model, with no external transfer-soundness premise.
This format does not import ONNX or certify the CROWN JSON formats. It says nothing about a
different model that an external producer may claim the document represents.
-/

@[expose] public section

namespace NN.Verification.Monotonicity

open Lean _root_.Spec
open NN.Verification.Cert.RationalJson

/-- Read one vector layer, checking its input dimension. -/
def decodeLayer (n : Nat) (j : Json) :
    Except String (Σ m : Nat, Certificate [n] [m]) := do
  match ← (← j.getObjVal? "kind").getStr? with
  | "relu" => return ⟨n, .relu [n]⟩
  | "linear" =>
      let ⟨m, layer⟩ ← decodeLinear n j
      return ⟨m, .linear layer⟩
  | kind => throw s!"unsupported layer kind: {kind}"

/-- Read a nonempty chain; each output dimension determines the next input dimension. -/
def decodeLayers (n : Nat) : List Json →
    Except String (Σ m : Nat, Certificate [n] [m])
  | [] => .error "expected at least one layer"
  | j :: rest => do
      let ⟨m, first⟩ ← decodeLayer n j
      match rest with
      | [] => return ⟨m, first⟩
      | _ :: _ =>
          let ⟨k, tail⟩ ← decodeLayers m rest
          return ⟨k, .comp first tail⟩

/-- Decode exact parameters, rejecting malformed dimensions and unsupported operations. -/
def decode (j : Json) : Except String (Σ n m : Nat, Certificate [n] [m]) := do
  unless (← (← j.getObjVal? "format").getStr?) == "monotonicity_v1" do
    throw "unsupported certificate format"
  let n ← (← j.getObjVal? "input_dim").getNat?
  let layers ← (← j.getObjVal? "layers").getArr?
  let ⟨m, cert⟩ ← decodeLayers n layers.toList
  return ⟨n, m, cert⟩

/-- The executable JSON checker: malformed documents and failed weight checks reject. -/
def acceptsJson (j : Json) : Bool :=
  match decode j with
  | .error _ => false
  | .ok ⟨_, _, cert⟩ => check cert

/-- Every accepted JSON document denotes a globally monotone real TorchLean model. -/
theorem acceptsJson_sound (j : Json) (h : acceptsJson j = true) :
    ∃ (n m : Nat) (cert : Certificate [n] [m]),
      decode j = .ok ⟨n, m, cert⟩ ∧ PreservesOrder cert.model.forward := by
  unfold acceptsJson at h
  cases hd : decode j with
  | error error => simp [hd] at h
  | ok decoded =>
      rcases decoded with ⟨n, m, cert⟩
      exact ⟨n, m, cert, rfl, check_sound cert (by simpa [hd] using h)⟩

/-- Parse and check a complete JSON document, rejecting syntax errors. -/
def acceptsText (source : String) : Bool :=
  match Json.parse source with
  | .error _ => false
  | .ok j => acceptsJson j

/-- Text acceptance establishes monotonicity of the model decoded from that text. -/
theorem acceptsText_sound (source : String) (h : acceptsText source = true) :
    ∃ (j : Json) (n m : Nat) (cert : Certificate [n] [m]),
      Json.parse source = .ok j ∧ decode j = .ok ⟨n, m, cert⟩ ∧
        PreservesOrder cert.model.forward := by
  unfold acceptsText at h
  cases hp : Json.parse source with
  | error error => simp [hp] at h
  | ok j =>
      obtain ⟨n, m, cert, hd, hmono⟩ := acceptsJson_sound j (by simpa [hp] using h)
      exact ⟨j, n, m, cert, rfl, hd, hmono⟩

end NN.Verification.Monotonicity
