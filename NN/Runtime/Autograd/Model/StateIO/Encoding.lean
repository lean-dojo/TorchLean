/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Complex
public import FloatLib.Floats.Formats.BinaryInterchange.Configured

/-!
# Scalar encodings for immutable model checkpoints

The tensor container checks shapes and lengths; `Checkpoint.Encoding` checks scalar payloads.
Native formats preserve the bits their values export; Lean canonicalizes native NaNs. Their
decoders reject noncanonical NaN words instead of silently changing a payload. Configured binary
formats preserve all encodings, including NaN payloads. Complex values encode both coordinates
using the component encoding. No value passes through a narrower native format.
-/

@[expose] public section

namespace TorchLean.Checkpoint

/-- A versioned scalar checkpoint representation, with explicit rejection of malformed values. -/
class Encoding (α : Type) where
  /-- Identifies the scalar format and schema. Different interpretations need different tags. -/
  format : String
  /-- Encode one scalar without changing its precision. -/
  encode : α → Lean.Json
  /-- Decode one scalar, rejecting out-of-range or incorrectly shaped payloads. -/
  decode : Lean.Json → Except String α

namespace Encoding

/-- Parse a storage word without silently reducing an out-of-range input modulo its width. -/
def checkedWord (width : Nat) (json : Lean.Json) : Except String Nat := do
  let bits ← json.getNat?
  if bits < 2 ^ width then pure bits
  else throw s!"Checkpoint: scalar bits out of range for {width}-bit storage"

/-- The existing binary64 JSON format, preserving compatibility with saved Float checkpoints. -/
instance : Encoding Float where
  format := "torchlean_state_bits_v1"
  encode value := Lean.toJson value.toBits.toNat
  decode json := do
    let bits ← checkedWord 64 json
    let value := Float.ofBits (UInt64.ofNat bits)
    if value.toBits.toNat == bits then pure value
    else throw "Checkpoint: noncanonical native Float NaN encoding"

/-- Exact binary32 words for immutable CPU state. Streamed runtime checkpoints remain separate. -/
instance : Encoding Float32 where
  format := "torchlean_state_float32_bits_v1"
  encode value := Lean.toJson value.toBits.toNat
  decode json := do
    let bits ← checkedWord 32 json
    let value := Float32.ofBits (UInt32.ofNat bits)
    if value.toBits.toNat == bits then pure value
    else throw "Checkpoint: noncanonical native Float32 NaN encoding"

open FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange

/--
Configured binary checkpoints include the complete numerical format, independently of storage plan.

Exponent width, fraction width, bias, and exceptional-value encoding all participate in the tag.
Two equal-width formats with different interpretations therefore cannot be confused on loading.
-/
instance {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
    [ExecFloat.ModelCodec plan (Model format) code] :
    Encoding (ExecFloat (Configured.Family format code plan)) where
  format :=
    let encoding := match format.encoding with
      | .ieee => "ieee"
      | .finiteMaxNaN => "finite-max-nan"
      | .finiteUnsignedZero => "finite-unsigned-zero"
      | .finite => "finite"
    s!"torchlean_state_binary_{format.expWidth}_{format.fracWidth}_" ++
      s!"{format.exponentBias}_{encoding}_v1"
  encode value := Lean.toJson (ExecFloat.Binary.toNatBits value)
  decode json := do
    pure (ExecFloat.Binary.ofNatBits (← checkedWord format.bitWidth json))

/-- A complex scalar is an ordered pair of independently encoded real and imaginary coordinates. -/
instance {α : Type} [Encoding α] : Encoding (Complex α) where
  format := s!"torchlean_state_complex_v1({Encoding.format (α := α)})"
  encode value := Lean.Json.arr #[Encoding.encode value.re, Encoding.encode value.im]
  decode json := do
    let components ← json.getArr?
    unless components.size == 2 do
      throw "Checkpoint: a complex scalar requires exactly two components"
    pure ⟨← Encoding.decode (components[0]!), ← Encoding.decode (components[1]!)⟩

end Encoding
end TorchLean.Checkpoint
