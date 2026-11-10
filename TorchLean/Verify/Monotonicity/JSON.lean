/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.Monotonicity
public import TorchLean.Verify.Certificate.RationalJSON
public import TorchLean.Verify.JSON

/-!
# JSON acceptance implies real monotonicity

A `monotonicity_v1` document records an `input_dim` and a nonempty `layers` array. Linear layers
have `kind = "linear"`, a row-major `weights` array and a `bias` array, and ReLU layers have
`kind = "relu"`. Parameters are exact integer or fraction strings, never rounded JSON numbers.

The decoder builds shape-checked certificates from TorchLean tensors. Acceptance implies that the
decoded model is globally monotone over the reals, with no external transfer-soundness premise.
The guarantee is about the decoded model only: a producer's claim that the document represents
some other model is not checked, and the format neither imports ONNX nor certifies the CROWN JSON
formats.
-/

@[expose] public section

namespace TorchLean.Monotonicity

open Lean
open Certificate.RationalJson
open Monotonicity.Certificate (check check_sound)

/-- Decode one `relu` or `linear` layer with input dimension `n`, rejecting any other `kind`. -/
def decodeLayer (n : Nat) (j : Json) :
    Except String (Σ m : Nat, Certificate [n] [m]) := do
  match ← (← j.getObjVal? "kind").getStr? with
  | "relu" => return ⟨n, .relu [n]⟩
  | "linear" =>
      let ⟨m, layer⟩ ← decodeLinear n j
      return ⟨m, .linear layer⟩
  | kind => throw s!"unsupported layer kind: {kind}"

/--
Decode a `monotonicity_v1` document into a layer chain, rejecting another `format`, malformed
dimensions and unsupported layer kinds.
-/
def decode (j : Json) : Except String (Σ n m : Nat, Certificate [n] [m]) := do
  unless (← (← j.getObjVal? "format").getStr?) == "monotonicity_v1" do
    throw "unsupported certificate format"
  let n ← (← j.getObjVal? "input_dim").getNat?
  let layers ← (← j.getObjVal? "layers").getArr?
  let ⟨m, cert⟩ ← decodeChain (Chain := fun n m => Certificate [n] [m]) decodeLayer .comp
    n layers.toList
  return ⟨n, m, cert⟩

/-- The executable JSON checker, which rejects malformed documents and failed weight checks. -/
def acceptsJson (j : Json) : Bool :=
  Json.accepts decode (fun ⟨_, _, cert⟩ => check cert) j

/-- Every accepted JSON document denotes a globally monotone real TorchLean model. -/
theorem acceptsJson_sound (j : Json) (h : acceptsJson j = true) :
    ∃ (n m : Nat) (cert : Certificate [n] [m]),
      decode j = .ok ⟨n, m, cert⟩ ∧ PreservesOrder (cert.model.eval (cert.params (α := ℝ))) := by
  obtain ⟨⟨n, m, cert⟩, hd, hmono⟩ := Json.accepts_sound
    (P := fun ⟨_, _, cert⟩ => PreservesOrder (cert.model.eval (cert.params (α := ℝ))))
    (fun ⟨_, _, cert⟩ => check_sound (α := ℝ) cert) h
  exact ⟨n, m, cert, hd, hmono⟩

/-- Parse and check a complete JSON document, rejecting syntax errors. -/
def acceptsText (source : String) : Bool :=
  Json.acceptsText decode (fun ⟨_, _, cert⟩ => check cert) source

/-- Text acceptance establishes monotonicity of the model decoded from that text. -/
theorem acceptsText_sound (source : String) (h : acceptsText source = true) :
    ∃ (j : Json) (n m : Nat) (cert : Certificate [n] [m]),
      Json.parse source = .ok j ∧ decode j = .ok ⟨n, m, cert⟩ ∧
        PreservesOrder (cert.model.eval (cert.params (α := ℝ))) := by
  obtain ⟨j, ⟨n, m, cert⟩, hp, hd, hmono⟩ := Json.acceptsText_sound
    (P := fun ⟨_, _, cert⟩ => PreservesOrder (cert.model.eval (cert.params (α := ℝ))))
    (fun ⟨_, _, cert⟩ => check_sound (α := ℝ) cert) h
  exact ⟨j, n, m, cert, hp, hd, hmono⟩

end TorchLean.Monotonicity
